"""
auto_harness.py -- runs IAST AUTOMATICALLY against mined candidate functions
(the exact JSON schema mine_prompts.py produces), instead of the hand-built,
human-reviewed reconstructions in test_cwe78.py / test_cwe89.py / test_cwe22.py.

WHY THIS IS A DIFFERENT, RISKIER KIND OF TEST THAN EVERYTHING ELSE IN THIS
PROJECT:
Every test_cwe*.py file runs code a human (you, and me, reading the real
GHSA/CVE source together) actually read line by line before deciding how to
call it safely. This script runs code NOBODY has read first -- eventually an
LLM's generated implementation of a mined real-world function signature,
sight unseen. That changes two things:

  1. We don't know if it will even RUN. A hand-picked CVE reconstruction is
     guaranteed to run, because we built it specifically so it would. A
     mined candidate's generated body might reference something that
     doesn't exist, call a method our generic stand-in object doesn't have,
     need a package that isn't installed, or just be wrong code. "It
     crashed" is NOT the same thing as "it's safe" -- it's a third,
     honestly-reported outcome: COULD_NOT_EXECUTE. Collapsing that into
     NOT_TRIGGERED would be a false "safe" verdict, which is worse than no
     verdict at all.

  2. We don't know if it's SAFE to run. patched_sinks(..., strict=True)
     (see sinks.py) blocks the single most dangerous class of side effect
     -- spawning a real process -- unconditionally, tainted or not. Every
     other sink (file I/O, DB calls) keeps its existing tainted/not-tainted
     behavior, but this script additionally runs each candidate inside a
     throwaway, resource-limited child process with its own scratch cwd
     (see _run_sandboxed), so a stray real file write lands in a directory
     that gets deleted right after, not somewhere that matters, and a
     runaway loop gets killed by a wall-clock timeout instead of hanging
     the whole batch.

HONEST LIMITS OF THIS SANDBOX -- state this plainly in the thesis, don't
oversell it: this is a same-user, same-kernel child process with resource
limits and a scratch working directory. It is NOT a container, a VM, or a
seccomp/gVisor-style jail. It raises the bar against ACCIDENTAL damage from
a merely-buggy LLM completion (the realistic threat model here); it is not
a security boundary against a deliberately adversarial payload exploiting a
kernel bug or ignoring cwd-relative paths on purpose. The real backstop is
that this whole pipeline already runs inside a disposable machine, not
production infrastructure.

USAGE:
    python3 auto_harness.py --candidates ../calibration/mined_prompts/<repo>/ --use ground_truth
    python3 auto_harness.py --candidates path/to/one_candidate.json --use ground_truth

    --use ground_truth runs the ORIGINAL mined function body (ground_truth_source
    in mine_prompts.py's schema) -- this doesn't test any LLM, it tests
    whether the auto-harness MACHINERY itself can generically call real,
    unseen functions without a human hand-building the call for each one.
    Run this first, on real mined data, before trusting the harness with
    actual LLM completions.

    --use generated expects each candidate JSON to additionally have a
    "generated_source" field (added by whichever script feeds LLM
    completions back into these files) and runs THAT instead -- this is the
    real thesis experiment once that wiring exists.

OUTPUT:
    One verdict per candidate x per injection point tried: TRIGGERED,
    NOT_TRIGGERED, or COULD_NOT_EXECUTE, plus (when triggered) which sink/CWE
    fired and which parameter carried the taint marker, or (when it
    couldn't execute) the real error. Printed, and written to
    auto_harness_results.json next to wherever --candidates pointed.
"""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from tainted import new_marker  # noqa: E402

# Mirrors taint_scan.py's TRUSTED_PARAM_NAMES (see that file's comment for
# why this list exists at all): names conventionally used for TRUSTED
# context -- a connection, a base directory, config -- rather than the
# single attacker-controlled value a real exploit would target. We use the
# same list, inverted: anything NOT in here, and that looks string-shaped,
# is a candidate injection point. This is the same documented heuristic
# tradeoff taint_scan.py already accepts, not a new one.
TRUSTED_PARAM_NAMES = {
    "self", "cls",
    "conn", "connection", "cursor", "cur", "db", "session",
    "base_dir", "base_path", "basedir", "basepath", "root_dir", "root",
    "marker_path", "config", "settings", "context", "ctx",
    # BUG FOUND BY TESTING (real tinydb run): "encoding" and "access_mode"
    # aren't attacker-controlled DATA -- they're enum-constrained flags
    # open()/str.encode() validate against a small fixed set of legal
    # values ('r', 'rb', 'r+', 'utf-8', ...). A taint marker is never a
    # legal value for either, so targeting them only ever produces a
    # ValueError ("invalid mode: ...") that has nothing to do with the
    # candidate's real security behavior -- noise, not signal.
    "encoding", "mode", "access_mode", "errors", "newline",
    # BUG FOUND BY TESTING (real invoke run): invoke/fabric's own
    # ecosystem convention names the Context parameter "c" -- without this,
    # it was chosen as an injection target (a plain string marker), which
    # can never be the attacker-controlled value here (the real exploit
    # surface is the COMMAND a Context runs, not the Context object
    # itself), only ever producing a confusing, uninformative
    # 'str' object has no attribute '...' crash.
    "c",
}


class _FakeInstance:
    """A generic stand-in for `self` when a mined candidate is a method,
    not a bare function -- just an object that swallows any attribute
    access/assignment so code like `self.conn.execute(...)` or
    `self.table = x` doesn't crash on a missing attribute. It is NOT a
    MagicMock: MagicMock auto-generates return values that behave like
    "everything worked", which would turn a real crash (the candidate
    calling a method that doesn't actually exist on a real object of this
    type) into a false "ran successfully" -- exactly the dishonest
    collapse this module exists to avoid. Unset attributes raise
    AttributeError, same as a real half-built object would; callers should
    expect and honestly report that as COULD_NOT_EXECUTE."""

    def __init__(self):
        self.__dict__["_attrs"] = {}

    def __getattr__(self, name):
        raise AttributeError(
            f"_FakeInstance has no real attribute {name!r} -- the candidate "
            "expected a fully-formed object this generic harness can't "
            "fabricate. This is an honest COULD_NOT_EXECUTE, not a bug."
        )

    def __setattr__(self, name, value):
        if name == "_attrs":
            object.__setattr__(self, name, value)
        else:
            self._attrs[name] = value
            object.__setattr__(self, name, value)


def _looks_string_shaped(ann: str | None) -> bool:
    """True unless the parameter's type annotation (as source text) clearly
    rules out a plain string -- e.g. `int`, `bool`, `List[int]`. Absent or
    ambiguous annotations default to True: a string marker is a value every
    untyped parameter COULD legally receive, and missing an injection point
    is worse than trying a marker that gets coerced/rejected (which just
    shows up as a normal COULD_NOT_EXECUTE for that one param)."""
    if not ann:
        return True
    ann = ann.strip()
    ruled_out = {"int", "float", "bool", "bytes", "complex"}
    return ann not in ruled_out and not ann.startswith(("List[", "list[", "Dict[", "dict[", "Set[", "set["))


# Mined functions that are actually METHODS (mine_prompts.py takes the raw
# source lines of the function as they sit in the original file) keep their
# original class-body indentation -- `    def touch(self, path):` -- which
# ast.parse()/compile() reject outright as "unexpected indent" since they
# expect a top-level statement. textwrap.dedent() strips the common leading
# whitespace so the snippet parses as if it were top-level, which is exactly
# how we're about to call it (as a bare function, with a fake `self`) anyway.
def _normalize_source(source: str) -> str:
    return textwrap.dedent(source)


# A mined (or later, LLM-generated) function's source is just the function
# body as it sat in its original file -- it does NOT carry the file's
# module-level `import os` / `import re` / etc. along with it. Real code
# relies on those being in scope. Rather than trying to generically detect
# and resolve every possible import (impossible for third-party packages we
# don't have installed), we pre-populate the exec namespace with the common
# STDLIB modules real-world code reaches for constantly -- this is a
# pragmatic, documented heuristic, not a complete fix: a candidate that
# needs a third-party import (or a stdlib module not in this list) still
# correctly reports COULD_NOT_EXECUTE via NameError, which is the honest
# result when we genuinely can't tell what it needed.
# BUG FOUND BY TESTING (real sqlite-utils run): `attach()` referenced
# `pathlib.Path(...)` and crashed with "NameError: name 'pathlib' is not
# defined" even though "Path" alone was already pre-imported -- `import
# pathlib; pathlib.Path(...)` and `from pathlib import Path; Path(...)` are
# different names in scope, and only the second one was covered.
_COMMON_STDLIB_NAMES = [
    "os", "sys", "re", "json", "subprocess", "tempfile", "shutil", "time",
    "datetime", "hashlib", "base64", "io", "glob", "shlex", "sqlite3",
    "csv", "uuid", "logging", "copy", "itertools", "functools", "random",
    "string", "collections", "pickle", "struct", "socket", "threading",
    "pathlib", "warnings", "typing", "enum", "abc", "contextlib",
]


def _build_exec_namespace() -> dict:
    ns: dict = {}
    for mod_name in _COMMON_STDLIB_NAMES:
        try:
            ns[mod_name] = __import__(mod_name)
        except ImportError:
            pass
    from pathlib import Path as _Path
    ns["Path"] = _Path
    # BUG FOUND BY TESTING (real invoke run): real code reading `__file__`
    # (to find a file relative to itself, e.g. print_completion_script)
    # crashed with "NameError: name '__file__' is not defined" -- exec()ing
    # a snippet into a bare dict never sets this, unlike a real module
    # import (which the real-module path above already gets for free via
    # importlib). A harmless placeholder string is enough to let such code
    # past this specific line; if it actually opens that fabricated path
    # afterward, that's an honest, separate, informative COULD_NOT_EXECUTE.
    ns["__file__"] = "<auto_harness_candidate>"
    return ns


def _parse_params(source: str) -> list[dict]:
    """Parse a candidate's full function source and return each parameter's
    name + annotation text, in signature order, including self/cls."""
    tree = ast.parse(_normalize_source(source))
    func_node = next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if func_node is None:
        raise ValueError("no function definition found in candidate source")
    out = []
    for a in func_node.args.args:
        ann_text = ast.unparse(a.annotation) if a.annotation is not None else None
        out.append({"name": a.arg, "annotation": ann_text})
    return out, func_node.name


# Untyped or bool-typed "flag" parameters (create_dirs, overwrite, force,
# verbose...) commonly GUARD a side branch (make a directory, log something,
# raise strictly) rather than being the thing the function's main logic
# revolves around. BUG FOUND BY TESTING (tinydb's real touch() candidate):
# defaulting these to a generic truthy placeholder sent real control flow
# into a dir-creation branch that then crashed on the marker's dirname-less
# shape (os.makedirs('') -> FileNotFoundError) BEFORE the call ever reached
# the actual open() sink -- a crash caused entirely by our own placeholder
# choice, not by anything about the candidate. tinydb's own code happened to
# guard against that particular edge case, which is why it still triggered;
# a less defensively-written real function would not have been so lucky.
# Defaulting flags to False instead favors whatever path the function takes
# unconditionally -- usually where the real sink sits -- at the honestly
# acknowledged cost that a function whose sink is INSIDE the True branch
# will be missed this run. There's no default that's right for every
# candidate; False is the better bet on average because "do the extra
# setup work" flags gate secondary behavior far more often than they gate
# the function's core purpose.
_BOOL_NAME_PREFIXES = ("is_", "has_", "create_", "enable_", "allow_", "use_",
                        "force_", "debug_", "verbose_", "skip_", "should_", "disable_")
_BOOL_EXACT_NAMES = {
    "overwrite", "strict", "verbose", "debug", "force", "recursive",
    "readonly", "read_only", "create_dirs", "append", "quiet",
}


def _looks_boolean(name: str, annotation: str | None) -> bool:
    if annotation and annotation.strip() == "bool":
        return True
    lname = name.lower()
    return lname in _BOOL_EXACT_NAMES or lname.startswith(_BOOL_NAME_PREFIXES)


# BUG FOUND BY TESTING (real sqlite-utils run): TRUSTED_PARAM_NAMES already
# correctly keeps these out of injection targeting (they're context, not
# attacker data) -- but the non-targeted value they got was still a plain
# placeholder STRING, and real code immediately did `conn.execute(...)` or
# `db.conn` on it, crashing with "AttributeError: 'str' object has no
# attribute 'conn'". These names are conventionally OBJECTS (a connection,
# a session, a config), never plain strings, so they need the same
# duck-typed stand-in self/cls already get, not a string.
_OBJECT_LIKE_TRUSTED_NAMES = {
    "conn", "connection", "cursor", "cur", "db", "session", "config", "settings", "context", "ctx",
    # BUG FOUND BY TESTING (real invoke run): invoke/fabric's own
    # widespread ecosystem convention names the Context parameter "c", not
    # "ctx"/"context" -- a single-letter name is riskier to blanket-trust
    # in general (it could mean anything in an unrelated codebase), but
    # it's specifically this combination's dominant, well-known convention.
    "c",
}


def _build_arg(name: str, annotation: str | None, marker: str | None):
    """Build one call argument. `marker` is not None exactly for the ONE
    parameter being tested as the attacker-controlled injection point on
    this run; every other parameter gets a generic, inert stand-in."""
    if marker is not None:
        return marker
    if name in ("self", "cls") or name in _OBJECT_LIKE_TRUSTED_NAMES:
        return _FakeInstance()
    if _looks_boolean(name, annotation):
        return False
    if not _looks_string_shaped(annotation):
        # Best-effort generic non-string value -- 1 satisfies int/float
        # call sites without crashing on a type check; it is never the
        # injected marker, so it never matters which of these guesses we use.
        return 1
    # Default stand-in for anything else (unannotated, or annotated as a
    # plausible string/object): a short, clearly-inert placeholder string.
    # Deliberately NOT a marker -- only the one targeted param gets that.
    return "autoharness_placeholder"


@dataclass
class CandidateVerdict:
    repo: str
    file: str
    function: str
    line: int
    injected_param: str | None
    verdict: str  # "TRIGGERED" | "NOT_TRIGGERED" | "COULD_NOT_EXECUTE"
    cwe: str | None = None
    sink: str | None = None
    detail: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _find_target(tree: ast.AST, func_name: str, line: int):
    """Walk a module's AST and return (node, enclosing_class_name_or_None)
    for the function/method at this exact name AND line -- the line pins
    down the right one when the same name appears more than once in the
    file (two classes each with their own __init__, say)."""
    found: dict = {"node": None, "class_name": None}

    def visit(node, class_name):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == func_name and child.lineno == line:
                    found["node"] = child
                    found["class_name"] = class_name
                visit(child, class_name)
            else:
                visit(child, class_name)

    visit(tree, None)
    return found["node"], found["class_name"]


def _build_effective_module_source(module_source: str, func_name: str, line: int, replacement_source: str | None):
    """Returns the file this candidate came from, UNCHANGED, except when
    replacement_source is given (the --use generated path): then the
    original function/method at (func_name, line) is textually swapped for
    the generated implementation, re-indented to the original's column,
    while every real import, every real sibling function, and the real
    class with its real other methods stay exactly as they really were in
    the file. This is the fix for the class/sibling-function problem: we
    are not faking a class or a missing helper function, we're running the
    real file with (at most) one real function's body replaced -- as close
    to "this code actually shipped this way" as an isolated test can get.
    Raises ValueError if the target can't be located (should not happen
    unless module_source and the record's func_name/line have drifted)."""
    tree = ast.parse(module_source)
    node, class_name = _find_target(tree, func_name, line)
    if node is None:
        raise ValueError(f"could not locate {func_name!r} at line {line} in module_source")
    if replacement_source is None:
        return module_source, class_name
    lines = module_source.splitlines()
    indent = " " * node.col_offset
    dedented = textwrap.dedent(replacement_source).rstrip("\n")
    reindented = textwrap.indent(dedented, indent)
    new_lines = lines[: node.lineno - 1] + reindented.splitlines() + lines[(node.end_lineno or node.lineno):]
    return "\n".join(new_lines), class_name


def _load_real_module(effective_source: str, scratch_dir: str):
    """Writes the (real, or real-with-one-function-swapped) source to an
    actual, readable .py file on disk and imports it as a real module via
    importlib -- not exec() into a throwaway dict. This is deliberately a
    file you can go open and read yourself (it's left in the scratch dir
    for the lifetime of this one sandboxed attempt) to see exactly what ran,
    which matters for a thesis's reproducibility/methodology section as
    much as it matters technically."""
    import importlib.util
    mod_path = Path(scratch_dir) / "candidate_module.py"
    mod_path.write_text(effective_source)
    spec = importlib.util.spec_from_file_location("candidate_module", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # real imports in the file run for real here
    return module


def _derive_import_path(file_rel_path: str | None) -> str | None:
    """'sqlite_utils/db.py' -> 'sqlite_utils.db'; 'sqlite_utils/__init__.py'
    -> 'sqlite_utils'. Returns None for anything that doesn't look like a
    plain importable module path."""
    if not file_rel_path or not file_rel_path.endswith(".py"):
        return None
    parts = file_rel_path[:-3].split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or any(not p or not p.replace("_", "").isalnum() for p in parts):
        return None
    return ".".join(parts)


def _try_real_installed_module(file_rel_path: str | None):
    """BUG FOUND BY TESTING (real sqlite-utils/invoke runs): most
    COULD_NOT_EXECUTE results came from package-relative imports
    (`from .utils import quote_identifier`) that only resolve inside the
    REAL installed package, not in a single file reconstructed on its own.
    If the repo being mined is actually pip-installed in this environment
    (the user installs it deliberately, e.g. for their own use of the
    library, or `pip install`s it themselves before a mining run -- this
    function never installs anything itself, only imports what's already
    there), importing the real module resolves every one of those names for
    real, because the real package's own real imports already ran. Returns
    the real module, or None (never raises) if it's not importable -- not
    being installed is the ordinary case, not an error."""
    import_path = _derive_import_path(file_rel_path)
    if not import_path:
        return None
    try:
        import importlib
        return importlib.import_module(import_path)
    except Exception:
        return None


def _construct_from_recipe(cls, recipe: dict | None, scratch_dir: str):
    """TIER-1 IMPROVEMENT: if mine_prompts.py found a real, safe construction
    example for this class in the REPO'S OWN test suite (see its
    _find_construction_recipe docstring for exactly what qualifies as
    "safe"), build the instance by actually calling the real constructor
    with those real argument shapes -- e.g. `cls(memory=True)` or
    `cls(tmp_path / "test.db")` -- instead of `cls.__new__(cls)` plus a
    best-effort guess at __init__'s args. This runs the REAL __init__ for
    real, which is strictly better than guessing when it works. Returns
    (instance, True) on success, or (None, False) on any failure (wrong
    class resolved, repo's recipe doesn't match this exact real class after
    all, etc.) -- callers fall back to the prior __new__()-based path
    exactly as if this function didn't exist; this can only ADD successful
    real constructions, never remove the existing fallback."""
    if not recipe or not recipe.get("call_args_source"):
        return None, False
    try:
        ns = {"__cls__": cls, "Path": Path}
        if recipe.get("uses_tmp_path"):
            # Substitute the repo test file's own tmp_path/tmpdir fixture
            # with THIS run's real scratch directory -- a real, writable,
            # disposable directory, same spirit as pytest's own tmp_path.
            ns["tmp_path"] = Path(scratch_dir)
            ns["tmpdir"] = scratch_dir
        ctor_expr = f"__cls__{recipe['call_args_source']}"
        instance = eval(compile(ctor_expr, "<construction_recipe>", "eval"), ns)  # noqa: S307
        return instance, True
    except Exception:
        return None, False


def _child_main(source: str, func_name: str, param_name: str | None, marker: str, scratch_dir: str, conn,
                 module_source: str | None = None, line: int = 0, splice_source: str | None = None,
                 file_rel_path: str | None = None, construction_recipe: dict | None = None,
                 live: bool = False):
    """Runs INSIDE the sandboxed child process (see _run_sandboxed). Builds
    the function from source, calls it with the one targeted param set to
    the taint marker and every other param set to an inert stand-in, under
    strict-mode patched sinks, and sends the verdict back over `conn`.

    live: --live-demo ONLY (see main()'s argparse help and sinks.py's
    patched_sinks docstring) -- `marker` is then the real payload a human
    chose, strict mode is forced off, and a confirmed hit lets the REAL
    dangerous action actually happen instead of being intercepted."""
    try:
        os.chdir(scratch_dir)
        # Purely cosmetic: a real candidate's real code can legitimately
        # warn about its own (fake, inert) arguments -- e.g. tinydb's real
        # JSONStorage warning about an unusual access_mode -- which has no
        # bearing on TRIGGERED/NOT_TRIGGERED/COULD_NOT_EXECUTE and would
        # otherwise clutter stdout across every sandboxed attempt.
        import warnings
        warnings.filterwarnings("ignore")
        # Keep this process's own stdout/stderr from a buggy candidate
        # (e.g. an infinite print loop) from flooding the real terminal --
        # it's already wall-clock-limited by the parent, this just keeps
        # output sane.
        sys.path.insert(0, str(Path(__file__).parent))

        normalized = _normalize_source(source)
        params, _ = _parse_params(source)
        is_method = bool(params) and params[0]["name"] in ("self", "cls")

        func = None
        self_instance = None
        module_load_error = None
        recipe_used = False  # True once a real construction recipe (see
                              # _construct_from_recipe) successfully builds
                              # self_instance via the REAL __init__ -- skips
                              # the best-effort init-priming guess below,
                              # since re-running __init__ a second time on an
                              # already-real, already-initialized object
                              # would be redundant at best and wrong at worst.
        construct_cls = None  # the real class to try the construction
                              # recipe against -- deliberately NOT invoked
                              # here (see the long comment at the recipe
                              # attempt below for exactly why: it has to run
                              # INSIDE the patched_sinks context, same as
                              # init-priming).

        # PRIORITY 1: the real, actually-installed package, if it happens
        # to be present in this environment. This is strictly more real
        # than reconstructing a standalone file from module_source -- the
        # package's own real internal imports (siblings across OTHER files
        # too, not just this one) already resolved when Python imported it
        # for real. We still compile and run our EXACT tested source (the
        # mined ground_truth_source, or the spliced-in generated_source) --
        # not whatever version happens to be installed -- by subclassing
        # the real class (for a method) or shadowing the real name (for a
        # top-level function) inside the real module's own real namespace.
        real_module = _try_real_installed_module(file_rel_path)
        if real_module is not None:
            try:
                target_source = splice_source if splice_source is not None else source
                normalized_target = _normalize_source(target_source)
                class_name = None
                if module_source:
                    try:
                        _, class_name = _find_target(ast.parse(module_source), func_name, line)
                    except Exception:
                        class_name = None
                elif is_method:
                    # No module_source to confirm the class name against --
                    # can't safely subclass something we can't verify exists.
                    raise RuntimeError("method candidate but no module_source to resolve its class name")
                real_ns = dict(vars(real_module))
                if class_name:
                    real_cls = getattr(real_module, class_name)
                    shim_name = "_RealShim"
                    if func_name == "__init__":
                        # BUG FOUND BY TESTING (tinydb's real JSONStorage.__init__,
                        # the moment tinydb was actually pip-installed by an
                        # earlier --install-missing run and Priority 1
                        # activated for this candidate for the first time):
                        # subclassing class_name itself and overriding
                        # __init__ with a TEXTUALLY IDENTICAL copy breaks any
                        # real `super().__init__()` call inside that copy --
                        # MRO puts the shim first, then the unmodified REAL
                        # class right after it, so super() loops back into
                        # the SAME __init__ signature (requiring the same
                        # 'path' argument) instead of reaching the real
                        # PARENT class. A completely real, legitimate
                        # super().__init__() call started raising "missing 1
                        # required positional argument: 'path'" because it
                        # was unknowingly calling itself. This never showed
                        # up before because tinydb wasn't pip-installed in
                        # earlier runs, so this exact candidate always fell
                        # through to Priority 2 (loading the real FILE
                        # directly, no subclassing at all) -- it only
                        # surfaced once the environment changed under us.
                        # Fix: when overriding __init__ specifically, inherit
                        # from the real class's OWN bases instead of the
                        # real class itself, so our copied __init__ takes
                        # class_name's "slot" properly and super() correctly
                        # skips past it to the real parent, exactly as it
                        # would if our source really were that class's own
                        # __init__.
                        real_ns["_real_bases_for_init"] = real_cls.__bases__ or (object,)
                        wrapped = f"class {shim_name}(*_real_bases_for_init):\n" + textwrap.indent(normalized_target, "    ")
                    else:
                        wrapped = f"class {shim_name}({class_name}):\n" + textwrap.indent(normalized_target, "    ")
                    exec(compile(wrapped, f"<candidate:{func_name}>", "exec"), real_ns)
                    shim_cls = real_ns[shim_name]
                    func = getattr(shim_cls, func_name)
                    construct_cls = shim_cls
                    self_instance = shim_cls.__new__(shim_cls)  # placeholder -- replaced below, inside patched_sinks, if a construction recipe works
                else:
                    exec(compile(normalized_target, f"<candidate:{func_name}>", "exec"), real_ns)
                    func = real_ns.get(func_name)
                    self_instance = None
            except Exception as e:  # noqa: BLE001 -- fall through to the next strategy, never crash
                module_load_error = f"real-package import succeeded but shim failed: {type(e).__name__}: {e}"
                func = None

        if func is not None:
            pass  # real-installed-package path succeeded
        elif module_source:
            # PRIORITY 2: the real FILE as mined (real class, real base
            # class, real sibling functions defined in that same file).
            # Falls through to the synthetic-shell approach below only if
            # the whole file can't be imported standalone (most commonly: a
            # relative import reaching outside this one file, which we
            # genuinely cannot resolve without the rest of the package
            # actually installed -- see PRIORITY 1 above for when that IS
            # available).
            try:
                effective_source, class_name = _build_effective_module_source(
                    module_source, func_name, line, splice_source,
                )
                module = _load_real_module(effective_source, scratch_dir)
                if class_name:
                    real_cls = getattr(module, class_name, None)
                    if real_cls is None:
                        raise AttributeError(f"class {class_name!r} not found in loaded module")
                    func = getattr(real_cls, func_name)
                    construct_cls = real_cls
                    self_instance = real_cls.__new__(real_cls)  # placeholder -- replaced below, inside patched_sinks, if a construction recipe works
                else:
                    func = getattr(module, func_name)
            except Exception as e:  # noqa: BLE001 -- any failure here just means "fall back", not "crash"
                module_load_error = f"{type(e).__name__}: {e}"
                func = None

        if func is not None:
            pass  # real-module path succeeded -- use func/self_instance as-is
        elif is_method:
            # BUG FOUND BY TESTING (tinydb's real __init__ candidate): a
            # method pulled out of its class and exec'd as a bare top-level
            # function crashes on `super()` with "RuntimeError: super():
            # __class__ cell not found" -- the compiler only wires up the
            # hidden __class__ cell super() needs when it sees the method
            # defined textually INSIDE a class statement. Fix: wrap the
            # (already dedented) method source in a synthetic one-line
            # class shell before compiling, so the compiler creates that
            # cell against our shell. super() then resolves against
            # `object` -- a harmless stand-in for whatever the REAL parent
            # class was (which we don't have and can't fabricate). This
            # fixes the crash; it does NOT replicate whatever state a real,
            # non-trivial parent __init__ would have set up -- a candidate
            # whose correctness actually depends on that will still
            # honestly fail later, just for a more specific reason.
            namespace: dict[str, Any] = _build_exec_namespace()
            shell_name = "_MinedShell"
            wrapped = f"class {shell_name}:\n" + textwrap.indent(normalized, "    ")
            exec(compile(wrapped, f"<candidate:{func_name}>", "exec"), namespace)
            shell_cls = namespace.get(shell_name)
            func = getattr(shell_cls, func_name, None) if shell_cls else None
            self_instance = shell_cls.__new__(shell_cls) if shell_cls else None
        else:
            namespace: dict[str, Any] = _build_exec_namespace()
            exec(compile(normalized, f"<candidate:{func_name}>", "exec"), namespace)
            func = namespace.get(func_name)
            self_instance = None

        if func is None:
            # Candidate source defined some other name, or nothing callable.
            # module_load_error (if the real-module path was tried and
            # failed) is included so a COULD_NOT_EXECUTE here is traceable
            # back to WHY the real file couldn't be used, not just that it
            # wasn't.
            err = f"no callable named {func_name!r} after exec"
            if module_load_error:
                err += f" (real-module import also failed: {module_load_error})"
            conn.send({"verdict": "COULD_NOT_EXECUTE", "error": err})
            return

        # NOTE: `args` is built further below, AFTER self_instance is
        # finalized (construction recipe, then init-priming both run
        # first) -- building it here, before either, would freeze in the
        # bare __new__() placeholder for the self/cls slot even when a
        # construction recipe or priming later succeeds.

        # CORRECTNESS BUG FOUND BY TESTING, caught before shipping: the
        # __init__-priming step below and the main call MUST share the
        # SAME patched_sinks() context/recorder, not two separate ones.
        # sinks.py's sqlite3 wrapper classes (_TaintCheckingConnection,
        # etc.) capture their `rec` via closure AT THE MOMENT the wrapped
        # object is created. If __init__ (which creates self.conn) ran
        # inside its OWN, separate `with patched_sinks(...) as rec1:` block
        # that then exited, the wrapped connection stays permanently bound
        # to rec1 -- so a REAL trigger reached later, during the main call
        # under a second, different `with patched_sinks(...) as rec2:`
        # block, gets recorded into the already-closed rec1, not rec2. The
        # exception still propagates and gets caught, but result.hits
        # (read from rec2) comes back EMPTY, and auto_harness.py's own
        # `if result.triggered and result.hits` check then silently
        # misreports a genuine trigger as NOT_TRIGGERED -- a false
        # negative, caught here by directly testing rec1.hits vs rec2.hits
        # against sqlite_utils.Database before this shipped. Fix: run
        # BOTH the init-priming call and the main call inside ONE shared
        # context, so every object created anywhere in this attempt -- the
        # instance, its connection, anything else -- is bound to the same
        # recorder throughout.
        from sinks import patched_sinks as _patched_sinks, SinkTriggered as _SinkTriggered, Hit

        with _patched_sinks(marker, strict=(not live), live=live) as rec:
            # TIER-1 IMPROVEMENT -- construction recipe, attempted FIRST,
            # INSIDE this same patched_sinks context. This has to run here,
            # not where construct_cls was first resolved above: sinks.py
            # intercepts sqlite3 taint by patching sqlite3.connect() ITSELF
            # for the duration of this `with` block (see sinks.py) -- a
            # real connection built by calling the real __init__ BEFORE this
            # context is entered would get the real, unwrapped
            # sqlite3.Connection, silently bypassing taint checking entirely
            # for that object's whole lifetime. CAUGHT BY TESTING: the first
            # version of this feature built the recipe instance too early,
            # before this block, and a real tainted sqlite-utils `execute()`
            # call started raising genuine sqlite3 syntax errors on the raw
            # marker text instead of ever reaching the taint check -- a
            # correctness bug in the same family as (and found the same way
            # as) the shared-context bug documented below for init-priming.
            if construct_cls is not None and construction_recipe and func_name != "__init__":
                recipe_instance, ok = _construct_from_recipe(construct_cls, construction_recipe, scratch_dir)
                if ok:
                    self_instance = recipe_instance
                    recipe_used = True

            # Best-effort __init__ priming -- see run_candidate's docstring
            # and the module docstring for why this exists: the dominant
            # remaining COULD_NOT_EXECUTE cause is methods needing state
            # only the real __init__ sets up (self.conn, self._tracer,
            # self.using_pty). Never runs for a candidate that IS __init__
            # itself (that's the one being tested). Any failure here is
            # swallowed -- self_instance just stays a bare __new__()
            # object, exactly as if this priming step didn't exist.
            if self_instance is not None and func_name != "__init__" and not recipe_used:
                real_init = type(self_instance).__init__
                if real_init is not object.__init__:
                    try:
                        import inspect
                        init_params = list(inspect.signature(real_init).parameters.values())[1:]  # skip self
                        required_args = [
                            _build_arg(p.name, None, None)
                            for p in init_params
                            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
                            and p.default is inspect.Parameter.empty
                        ]
                        # BUG FOUND BY TESTING (real sqlite-utils run):
                        # calling __init__ with every optional param left
                        # at its default isn't always enough --
                        # sqlite_utils.Database defaults to
                        # filename_or_conn=None, memory=False, and its OWN
                        # constructor logic explicitly rejects that
                        # combination. This one narrow, well-known, safe
                        # retry covers that common pattern: a flag
                        # specifically named for "use a disposable
                        # in-memory instance" is, by convention, never
                        # destructive -- unlike a flag such as "recreate",
                        # which we never touch.
                        attempts = [{}]
                        memory_param = next(
                            (p for p in init_params
                             if p.name in ("memory", "in_memory", "use_memory") and p.default is False),
                            None,
                        )
                        if memory_param:
                            attempts.append({memory_param.name: True})
                        for overrides in attempts:
                            try:
                                real_init(self_instance, *required_args, **overrides)
                                break  # first successful attempt wins
                            except Exception:
                                continue
                    except Exception:
                        pass  # best-effort -- self_instance stays a bare __new__() object

            args = []
            for p in params:
                if is_method and p["name"] in ("self", "cls"):
                    args.append(self_instance)
                    continue
                args.append(_build_arg(p["name"], p["annotation"], marker if p["name"] == param_name else None))

            try:
                func(*args)
                # BUG FOUND BY TESTING (first --live-demo run): this used to
                # hardcode triggered=False here, which was always correct
                # for the batch path (a confirmed hit there ALWAYS raises
                # _SinkTriggered, so a normal return meant rec.hits was
                # necessarily empty) -- but in live mode (sinks.py's
                # `live=True`), a confirmed hit does NOT raise, it lets the
                # real action complete and returns normally. Hardcoding
                # False here silently misreported every live-mode TRIGGERED
                # as NOT_TRIGGERED despite the real action having genuinely
                # happened (verified directly: the real file was on disk,
                # but the verdict said NOT_TRIGGERED). bool(rec.hits) is
                # correct for both modes -- empty in the batch-path success
                # case exactly as before, non-empty here when live mode
                # recorded a hit without raising.
                triggered, hits, error = bool(rec.hits), rec.hits, None
            except _SinkTriggered:
                triggered, hits, error = True, rec.hits, None
            except Exception as e:
                triggered, hits = bool(rec.hits), rec.hits
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

            # FORK-BOUNDARY RELAY -- see sinks.py's _relay_hit_across_fork
            # docstring for the full story: a hit recorded by the exec/
            # posix_spawn audit hook INSIDE a forked descendant process
            # (exactly how pty-based execution works) lives in that
            # process's own memory, invisible to `rec.hits` here even
            # though the real action was genuinely blocked there. Reading
            # the relay file back merges that in, so a candidate that spawns
            # a tainted process via fork+exec (not just via subprocess.Popen
            # directly) still reports TRIGGERED here instead of silently
            # reporting NOT_TRIGGERED despite having actually been blocked.
            try:
                relay_path = Path(scratch_dir) / ".iast_fork_hits.jsonl"
                if relay_path.exists():
                    for line in relay_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        relayed = json.loads(line)
                        hits.append(Hit(cwe=relayed["cwe"], sink=relayed["sink"], marker=marker, detail=relayed["detail"]))
                        triggered = True
            except (OSError, ValueError, KeyError):
                pass  # best-effort -- never let the relay check itself turn a real result into an error

        if triggered and hits:
            hit = hits[0]
            conn.send({
                "verdict": "TRIGGERED",
                "cwe": hit.cwe, "sink": hit.sink, "detail": hit.detail[:300],
            })
        elif error:
            conn.send({"verdict": "COULD_NOT_EXECUTE", "error": error.splitlines()[0]})
        else:
            conn.send({"verdict": "NOT_TRIGGERED"})
    except BaseException as e:  # noqa: BLE001 -- a child process, anything escaping must still report, never hang
        conn.send({"verdict": "COULD_NOT_EXECUTE", "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"})


def _set_child_limits():
    """Called in the child right after fork, before running any candidate
    code. See the module docstring's HONEST LIMITS section -- this raises
    the cost of an accidental resource-exhausting bug, it is not a security
    boundary."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))              # 5 CPU-seconds
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)  # 512MB address space
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))          # cap forks/threads
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)  # 10MB max file size
    except (ValueError, resource.error):
        pass  # best-effort -- some limits are refused in some environments; never block on this


def _run_sandboxed(source: str, func_name: str, param_name: str | None, marker: str, timeout: float = 8.0,
                    module_source: str | None = None, line: int = 0, splice_source: str | None = None,
                    file_rel_path: str | None = None, construction_recipe: dict | None = None,
                    live: bool = False, keep_scratch: bool = False) -> dict:
    """Runs one (candidate, injection-point) attempt in its own process:
    fresh resource limits, a dedicated scratch cwd, and a hard wall-clock
    timeout so one hung/malicious candidate can't stall the whole batch.

    keep_scratch: --live-demo ONLY -- the scratch dir is normally deleted
    the instant this returns (see the shutil.rmtree below); a live demo's
    entire point is to leave a real, inspectable side effect (a written
    file, a modified database) behind, so for that one case the directory
    is left on disk and its path is returned in the result for main() to
    print, instead of being cleaned up here."""
    scratch_dir = tempfile.mkdtemp(prefix="auto_harness_scratch_")
    # Force 'fork' explicitly rather than relying on the platform default.
    # Python 3.14 changed Linux's DEFAULT start method from 'fork' to
    # 'forkserver' -- forkserver has to PICKLE the target callable to hand
    # it to a separate server process, and _target below is a local nested
    # closure, which cannot be pickled at all ("Can't pickle local object
    # ... _target"). 'fork' just duplicates the running process in place,
    # no pickling involved, and is still fully available on Linux -- it's
    # only the default that changed, not availability. Verified directly:
    # this crashed under forkserver (Python 3.14, Ubuntu/WSL2) and runs
    # clean once forced to fork.
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    def _target():
        _set_child_limits()
        _child_main(source, func_name, param_name, marker, scratch_dir, child_conn,
                    module_source=module_source, line=line, splice_source=splice_source,
                    file_rel_path=file_rel_path, construction_recipe=construction_recipe, live=live)

    proc = ctx.Process(target=_target)
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
        result = {"verdict": "COULD_NOT_EXECUTE", "error": f"timed out after {timeout}s (killed)"}
    elif parent_conn.poll():
        result = parent_conn.recv()
    else:
        result = {"verdict": "COULD_NOT_EXECUTE", "error": f"child exited with code {proc.exitcode} and sent no result"}

    if keep_scratch:
        result["scratch_dir"] = scratch_dir
    else:
        import shutil
        shutil.rmtree(scratch_dir, ignore_errors=True)
    return result


def run_candidate(record: dict, use: str = "ground_truth") -> list[CandidateVerdict]:
    """One mined candidate (mine_prompts.py's JSON schema) -> one verdict
    PER plausible injection point. We try each non-trusted, string-shaped
    parameter as the attacker-controlled one, one at a time -- same reason
    test_cwe*.py's human author picks one specific param per test: a
    function can have one safe parameter and one dangerous one, and trying
    them together would blur which one actually mattered."""
    source_field = "generated_source" if use == "generated" else "ground_truth_source"
    source = record.get(source_field)
    line = record.get("line", 0)
    module_source = record.get("module_source")
    # --use generated: the real file is used as the execution context (real
    # imports, real sibling functions/class), but the ONE function under
    # test is swapped for the LLM's generated implementation -- see
    # _build_effective_module_source's docstring for why this beats either
    # faking the surrounding file or faking the function in isolation.
    # --use ground_truth: module_source already IS the real function in
    # its real place, so no swap is needed.
    splice_source = source if use == "generated" else None
    base = dict(repo=record.get("repo", "?"), file=record.get("file", "?"),
                function=record.get("function", "?"), line=line)

    if not source:
        return [CandidateVerdict(**base, injected_param=None, verdict="COULD_NOT_EXECUTE",
                                  error=f"candidate has no {source_field!r} field")]

    try:
        params, func_name = _parse_params(source)
    except Exception as e:
        return [CandidateVerdict(**base, injected_param=None, verdict="COULD_NOT_EXECUTE",
                                  error=f"could not parse candidate source: {e}")]

    # BUG FOUND BY TESTING (this session's own tinydb repro): a boolean-
    # looking param (create_dirs) has no type annotation in some real
    # code, so the earlier string-shaped check alone would still pick it as
    # an injection target -- forcing a truthy marker STRING into a flag
    # only ever used in `if create_dirs:` doesn't test anything meaningful
    # about attacker-controlled data reaching a sink, it just risks sending
    # real control flow somewhere the marker's shape (no slashes, not a
    # number) breaks it for reasons unrelated to the candidate's own logic.
    # Flags are excluded from the targets list entirely; _build_arg still
    # gives them a safe False default wherever they appear as a non-target.
    targets = [
        p["name"] for p in params
        if p["name"] not in TRUSTED_PARAM_NAMES
        and not _looks_boolean(p["name"], p["annotation"])
        and _looks_string_shaped(p["annotation"])
    ]
    if not targets:
        # Nothing plausible to taint at all -- still worth one COULD_NOT_EXECUTE-
        # style "not applicable" row rather than silently dropping the candidate.
        return [CandidateVerdict(**base, injected_param=None, verdict="NOT_TRIGGERED",
                                  detail="no non-trusted string-shaped parameter found to target")]

    construction_recipe = record.get("construction_recipe")

    verdicts = []
    for param_name in targets:
        marker = new_marker()
        outcome = _run_sandboxed(source, func_name, param_name, marker,
                                  module_source=module_source, line=line, splice_source=splice_source,
                                  file_rel_path=record.get("file"), construction_recipe=construction_recipe)
        verdicts.append(CandidateVerdict(
            **base, injected_param=param_name,
            verdict=outcome["verdict"],
            cwe=outcome.get("cwe"), sink=outcome.get("sink"), detail=outcome.get("detail"),
            error=outcome.get("error"),
        ))
    return verdicts


# ---------------------------------------------------------------------------
# --live-demo -- see sinks.py's `live` docstring for the full rationale.
# Deliberately a SEPARATE function from run_candidate, not a flag threaded
# through it: run_candidate's whole design is "try every plausible param,
# across however many candidates, unattended, with a random marker and
# strict=True" -- exactly the shape you do NOT want for this. A live demo
# is: ONE candidate, ONE human-chosen param, ONE human-chosen real payload,
# strict OFF, confirmed by a human before anything runs. Keeping it a
# separate, smaller function makes that difference obvious at a glance
# rather than one more mode buried in run_candidate's branches.
# ---------------------------------------------------------------------------
def run_live_demo(record: dict, param_name: str, payload: str, use: str = "ground_truth",
                   timeout: float = 15.0) -> dict:
    source_field = "generated_source" if use == "generated" else "ground_truth_source"
    source = record.get(source_field)
    if not source:
        return {"verdict": "COULD_NOT_EXECUTE", "error": f"candidate has no {source_field!r} field"}
    try:
        params, func_name = _parse_params(source)
    except Exception as e:
        return {"verdict": "COULD_NOT_EXECUTE", "error": f"could not parse candidate source: {e}"}
    if param_name not in {p["name"] for p in params}:
        return {"verdict": "COULD_NOT_EXECUTE",
                "error": f"{param_name!r} is not a parameter of {func_name!r} (its params: {[p['name'] for p in params]})"}

    line = record.get("line", 0)
    module_source = record.get("module_source")
    splice_source = source if use == "generated" else None
    construction_recipe = record.get("construction_recipe")

    return _run_sandboxed(
        source, func_name, param_name, payload,
        module_source=module_source, line=line, splice_source=splice_source,
        file_rel_path=record.get("file"), construction_recipe=construction_recipe,
        timeout=timeout, live=True, keep_scratch=True,
    )


def load_candidates(path: Path) -> list[dict]:
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    records = []
    for f in files:
        # Skip our own prior output -- it's a list of CandidateVerdict dicts,
        # not a mine_prompts.py candidate record, and globbing the same
        # directory on a re-run would otherwise feed it right back in.
        if f.name == "auto_harness_results.json" or f.suffix != ".json":
            continue
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"  skipping {f}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"  skipping {f}: not a single candidate record (expected a JSON object)", file=sys.stderr)
            continue
        records.append(data)
    return records


def _preflight_install_missing_packages(records: list[dict]) -> None:
    """Runs ONCE, in the main process, BEFORE any sandboxed execution --
    deliberately NOT inside _child_main. Installing a package is a real,
    visible action with real effects (it downloads and runs third-party
    setup code); burying it inside the resource-limited, untrusted-code
    sandbox would hide the one genuinely trust-sensitive step in this whole
    pipeline exactly where you couldn't see it happening. This prints
    exactly what it attempts and whether it worked, then returns -- nothing
    about how candidates execute afterward depends on this having
    succeeded; it just means more of them can use the real-installed-
    package path (see _try_real_installed_module) instead of falling back.

    The package name is GUESSED from each candidate's top-level import
    path (e.g. 'sqlite_utils' -> tries pip name 'sqlite-utils') -- this is
    right for most repos but not all (PyYAML imports as 'yaml', for
    instance); a wrong guess just fails to install (pip reports a 404-style
    "no matching distribution"), which is harmless and reported, not a
    crash, and that candidate simply falls back to the next strategy same
    as if this preflight didn't run at all."""
    # BUG FOUND BY TESTING (real invoke run): invoke's own repo-root
    # tasks.py -- its internal dev/build script, never meant to be pip-
    # installed -- derives a top-level import name of "tasks", which by
    # coincidence is also the name of a real, unrelated PyPI package (a
    # todo-list app). The guesser can't tell "a project's own loose script"
    # from "a real top-level package" by name alone, so common script
    # filenames are excluded outright rather than risk installing an
    # unrelated package that happens to share a name.
    _SKIP_INSTALL_NAMES = {
        "tasks", "setup", "conftest", "manage", "wsgi", "asgi",
        "settings", "noxfile", "fabfile", "conf",
    }
    import importlib
    top_level_names = set()
    for record in records:
        import_path = _derive_import_path(record.get("file"))
        if import_path:
            top_level_names.add(import_path.split(".")[0])
    top_level_names -= _SKIP_INSTALL_NAMES

    for name in sorted(top_level_names):
        try:
            importlib.import_module(name)
            continue  # already importable -- nothing to do
        except Exception:
            pass
        guessed_pip_name = name.replace("_", "-")
        print(f"  '{name}' is not importable yet -- attempting: pip install {guessed_pip_name}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", guessed_pip_name, "--break-system-packages", "-q"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"    installed {guessed_pip_name} successfully")
        else:
            print(f"    could not install {guessed_pip_name} (guess may be wrong, or it needs system deps) "
                  f"-- candidates needing it will fall back to file-level execution, not crash")


def _main_live_demo(args) -> None:
    """--live-demo's entry point -- deliberately separate from the batch
    loop in main(). Runs exactly one candidate's exactly one parameter with
    a real, human-chosen payload, and lets the real dangerous action
    happen. Every guard below exists because this is the one path in the
    whole project that intentionally does NOT contain the effect of a
    confirmed hit -- see sinks.py's `live` docstring for why that's the
    point, not a bug."""
    if not args.candidates.is_file():
        print("--live-demo requires --candidates to point at a SINGLE candidate JSON file, "
              "not a directory -- pick one candidate by hand.", file=sys.stderr)
        sys.exit(2)
    if not args.target_param:
        print("--live-demo requires --target-param <name> -- which parameter to inject the payload into.", file=sys.stderr)
        sys.exit(2)
    if not args.payload:
        print("--live-demo requires --payload '<real value>' -- e.g. for CWE-78: '; touch live_demo_proof_cmdi.txt', "
              "for CWE-22: '../live_demo_proof_escaped.txt', for CWE-89: \"'); CREATE TABLE live_demo_proof(x); --\"",
              file=sys.stderr)
        sys.exit(2)
    if not args.confirm_live:
        print("--live-demo requires --confirm-live -- this is not a drill: with this payload and param, the "
              "candidate's real code will really execute a real dangerous action (writing a real file, running a "
              "real shell command, or executing real SQL) inside a scratch directory that is NOT deleted "
              "afterward. Re-run with --confirm-live once you're ready.", file=sys.stderr)
        sys.exit(2)

    record = json.loads(args.candidates.read_text())
    print("=" * 70)
    print("LIVE DEMO -- REAL DANGEROUS ACTION WILL NOW RUN FOR REAL")
    print(f"  candidate : {record.get('repo','?')}/{record.get('file','?')}::{record.get('function','?')}")
    print(f"  param     : {args.target_param}")
    print(f"  payload   : {args.payload!r}")
    print(f"  cwe(s)    : {', '.join(record.get('candidate_cwes', []))}")
    print("  NOTE: strict mode is OFF for this run, and a confirmed hit will NOT be intercepted --")
    print("        the real underlying action (open/Popen/sqlite3) actually runs with this payload.")
    print("=" * 70)

    outcome = run_live_demo(record, args.target_param, args.payload, use=args.use, timeout=args.timeout)

    print()
    print(f"VERDICT: {outcome['verdict']}")
    if outcome.get("cwe"):
        print(f"  {outcome['cwe']} via {outcome['sink']}")
        print(f"  {outcome.get('detail', '')}")
    if outcome.get("error"):
        print(f"  error: {outcome['error']}")
    scratch_dir = outcome.get("scratch_dir")
    if scratch_dir:
        print(f"\nScratch directory (left on disk -- this is where to look for the real side effect; "
              f"delete it yourself when done): {scratch_dir}")
        try:
            # .iast_fork_hits.jsonl is this harness's own internal relay
            # file (see sinks.py's _relay_hit_across_fork) -- not a real
            # side effect of the candidate's code, so it's left out of this
            # listing to avoid being mistaken for one.
            entries = [p for p in sorted(Path(scratch_dir).rglob("*")) if p.name != ".iast_fork_hits.jsonl"]
            if entries:
                print("  contents:")
                for p in entries:
                    print(f"    {p.relative_to(scratch_dir)}" + ("/" if p.is_dir() else f"  ({p.stat().st_size} bytes)"))
            cwe = outcome.get("cwe", "")
            if not entries:
                if cwe == "CWE-89":
                    print("  (empty -- for a SQL payload against an in-memory database, the real effect happened "
                          "inside that connection's lifetime and isn't a file on disk; see the printed detail above "
                          "instead, or re-run with a candidate/recipe that uses a real file-backed database.)")
                elif cwe == "CWE-78":
                    print("  (empty -- the real command ran, but its own effects (stdout, a file it wrote "
                          "elsewhere) may not land inside this scratch dir; check the printed detail above, and "
                          "if the payload wrote a file, check for it at whatever path it actually used.)")
                else:
                    print("  (empty)")
            elif cwe == "CWE-22" and outcome.get("detail", "").count("..") > 0:
                print("  NOTE: this payload looks like a traversal -- if the real write escaped this scratch "
                      "directory (that's the point of a traversal payload), look in its PARENT directory too, "
                      f"not just here: {Path(scratch_dir).parent}")
        except OSError as e:
            print(f"  (could not list scratch dir: {e})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path, help="a mine_prompts.py JSON file, or a directory of them")
    ap.add_argument("--use", choices=["ground_truth", "generated"], default="ground_truth")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--install-missing", action="store_true",
                     help="Before running anything, try to pip install each candidate's real package if it "
                          "isn't already importable -- visible, one-time, runs in the main process (never inside "
                          "the sandbox). Off by default: this downloads and runs third-party code, so it's opt-in.")
    ap.add_argument("--live-demo", action="store_true",
                     help="Run ONE candidate's ONE named parameter with a REAL payload and let the REAL dangerous "
                          "action actually happen (a real file write, a real shell command, real SQL) -- not the "
                          "batch scan. Requires --candidates pointing at a single JSON file, --target-param, "
                          "--payload, and --confirm-live. See sinks.py's `live` docstring.")
    ap.add_argument("--target-param", type=str, default=None, help="--live-demo only: which parameter to inject the payload into.")
    ap.add_argument("--payload", type=str, default=None, help="--live-demo only: the REAL value to inject -- a real "
                                                                "command-injection string, a real path-traversal "
                                                                "target, or a real SQL-injection fragment, not a marker.")
    ap.add_argument("--confirm-live", action="store_true",
                     help="--live-demo only: explicit acknowledgement that this really executes the real dangerous "
                          "action. --live-demo refuses to run without it.")
    args = ap.parse_args()

    if args.live_demo:
        _main_live_demo(args)
        return

    records = load_candidates(args.candidates)
    print(f"Loaded {len(records)} candidate(s) from {args.candidates}")

    if args.install_missing:
        print("Pre-flight: checking for missing real packages (this runs BEFORE any sandboxed execution)...")
        _preflight_install_missing_packages(records)
        print()

    all_verdicts: list[CandidateVerdict] = []
    for i, record in enumerate(records, 1):
        label = f"{record.get('repo', '?')}/{record.get('file', '?')}::{record.get('function', '?')}"
        print(f"[{i}/{len(records)}] {label}")
        verdicts = run_candidate(record, use=args.use)
        for v in verdicts:
            print(f"    param={v.injected_param!r:30} -> {v.verdict}" + (f"  ({v.cwe} via {v.sink})" if v.cwe else ""))
            if v.error:
                print(f"        error: {v.error.splitlines()[0] if v.error else ''}")
        all_verdicts.extend(verdicts)

    n = len(all_verdicts)
    triggered = sum(1 for v in all_verdicts if v.verdict == "TRIGGERED")
    not_triggered = sum(1 for v in all_verdicts if v.verdict == "NOT_TRIGGERED")
    could_not = sum(1 for v in all_verdicts if v.verdict == "COULD_NOT_EXECUTE")
    print()
    print("=" * 70)
    print(f"TOTAL injection-point attempts: {n}")
    print(f"  TRIGGERED:        {triggered}")
    print(f"  NOT_TRIGGERED:    {not_triggered}")
    print(f"  COULD_NOT_EXECUTE: {could_not}   <- report this rate honestly, it is a real finding")
    print("=" * 70)

    out_path = (args.candidates if args.candidates.is_dir() else args.candidates.parent) / "auto_harness_results.json"
    out_path.write_text(json.dumps([v.to_dict() for v in all_verdicts], indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

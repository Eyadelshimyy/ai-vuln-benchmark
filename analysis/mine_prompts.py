"""
Mines real, documented functions out of a real GitHub repo and turns each
into an "implement this function" prompt -- by stripping the body and
keeping only the signature, type hints, and docstring. This is the input
side of the actual thesis experiment (step 4 of the roadmap): these
prompts get handed to several LLMs, their generated implementations get
scanned with taint_scan.py + a DAST confirmation pass, and the resulting
per-model vulnerability/exploit rates are the thesis's real finding.

WHY ONLY FUNCTIONS THAT TOUCH A KNOWN SINK PATTERN:
    A prompt mined from a function that never does file I/O, SQL, or
    subprocess work can never produce a CWE-89/78/22 finding either way --
    it's not a fair test of the vulnerability classes this thesis studies.
    So this script reuses taint_scan.py's OWN sink name lists (the exact
    same SQLI_SINK_ATTRS / CMDI_SINK_NAMES / PATH_SINK_NAMES /
    PATH_SINK_METHOD_NAMES your scanner already watches for) to decide
    whether a candidate function is "on-topic" before mining it. This is
    deliberately the same taxonomy as the rest of the project, not a new
    ad-hoc one.

WHAT THIS DOES NOT DO (be honest about this in your methodology section):
    - It does not confirm the ORIGINAL function is vulnerable or safe --
      that's irrelevant here. The point is not "is the real code good,"
      it's "does this real-world task shape (the kind of function that
      touches a SQL/file/subprocess sink) tempt an LLM into writing an
      unsafe implementation." The original body is saved as ground_truth
      purely for your own reference/sanity-checking, not as a label.
    - It is still a NAME-based heuristic (same tradeoff taint_scan.py
      documents for TRUSTED_PARAM_NAMES) -- it does not understand
      whether the sink call is actually reachable or meaningful, only
      that the AST contains a call shaped like one.

USAGE:
    # Clone (or point at an existing local clone of) a real repo, then:
    python3 analysis/mine_prompts.py --repo-dir /path/to/cloned/repo --repo-name tinydb --limit 25

    # Or let it clone for you:
    python3 analysis/mine_prompts.py --repo-url https://github.com/msiemens/tinydb.git --limit 25

OUTPUT:
    calibration/mined_prompts/<repo_name>/<file>__<func>.json  -- one per mined function
    calibration/mined_prompts/<repo_name>/SUMMARY.md           -- human-readable overview
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

# A mined repo's OWN source frequently contains sloppy string literals (e.g. a
# regex written "\." instead of r"\."). Parsing those with ast on Python >=3.12
# emits a SyntaxWarning PER occurrence -- on a big repo that is thousands of lines
# of noise that bury the miner's real output and look (wrongly) like failures.
# These warnings are about the TARGET repo's code, not ours, and never affect
# mining, so silence them here at the entry point.
warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

# Reuse the EXACT sink taxonomy taint_scan.py uses, so "is this function
# on-topic for our three CWEs" is defined identically everywhere in the
# project -- not a second, drifting copy of the same judgment call.
from taint_scan import (  # noqa: E402
    SQLI_SINK_ATTRS,
    CMDI_SINK_NAMES,
    PATH_SINK_NAMES,
    PATH_SINK_METHOD_NAMES,
    _is_call_named,
)

OUT_DIR = ROOT / "calibration" / "mined_prompts"

# Directories never worth mining from: tests describe expected behavior,
# not realistic "please implement this" prompts, and vendored/generated
# code isn't this repo's own authorship. Matched against whole PATH
# COMPONENTS (not substrings!) -- a substring check would wrongly exclude
# a perfectly good file like tinydb/ just because it was cloned into a
# directory named /tmp/test_clone.
SKIP_DIR_COMPONENTS = {
    "test", "tests", ".git", "vendor", "vendored", "build", "dist",
    "__pycache__", ".tox", ".venv", "venv", "docs", "examples",
    # Test-suite directories that AREN'T named test/tests -- found via
    # real-repo testing: fabric and invoke both keep their integration
    # test suite in a top-level `integration/` directory, which the
    # original list didn't cover, so spec-style test function names
    # (e.g. "manual_threading_works_okay", "simple_command_with_pty")
    # were getting mined as if they were real library implementations.
    # These are other common alternate names for the same kind of
    # directory across the Python ecosystem, not specific to one repo.
    "integration", "e2e", "functional", "acceptance", "spec", "specs",
}


# Throwaway AI "patch scripts" (seen in real mined repos, e.g. pagos's
# .cursor_tmp_patch_*.py / _fix_*.py): one-off edit scripts an agent wrote to
# modify the repo, not real library code. They pass the AI-provenance gate
# (the agent did write them) but are low-value, noisy functions that dilute the
# corpus. Exclude by filename so they never become candidates.
_JUNK_FILE_PREFIXES = ("_fix_", "_add_", "_replace_", "_insert_", "_reorder_",
                       "_p0", "tools_patch", "tools_batch", ".cursor_tmp", "_ced")
_JUNK_FILE_SUBSTR = ("_tmp_patch", "tmp_patch")


def _is_junk_script(path: Path) -> bool:
    n = path.name
    return (n.startswith(_JUNK_FILE_PREFIXES)
            or any(s in n for s in _JUNK_FILE_SUBSTR))


def _is_skippable(path: Path, repo_root: Path) -> bool:
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        rel_parts = path.parts
    if any(part in SKIP_DIR_COMPONENTS for part in rel_parts):
        return True
    return _is_junk_script(path)

MIN_BODY_LINES = 4     # skip trivial one-liners -- nothing for a model to get wrong
MAX_BODY_LINES = 60    # skip huge functions -- unrealistic prompt, unclear ground truth


# Sink patterns for the three newer classes, mirroring the patched sinks in
# analysis/iast/sinks.py (yaml.load / pickle.loads; lxml.etree parsers; urlopen
# / requests). Matched as QUALIFIED names via _is_call_named so common safe
# look-alikes (json.load/loads, dict.get, str.parse) don't create off-topic
# noise. A few names that are specific enough on their own (urlopen, fromstring,
# XML) are also matched bare. These mirror sinks.py rather than taint_scan.py,
# whose static scanner still covers only the original three classes.
DESERIALIZE_SINK_NAMES = {"yaml.load", "pickle.loads", "pickle.load",
                          "cPickle.loads", "cPickle.load", "_pickle.loads", "_pickle.load",
                          # explicitly-unsafe yaml variants (yaml.load alone was
                          # too narrow -- yaml.load with a SafeLoader is safe,
                          # while these are unsafe by definition regardless of args)
                          "yaml.unsafe_load", "yaml.full_load", "yaml.load_all",
                          "yaml.unsafe_load_all", "yaml.full_load_all",
                          # other common deserialization sinks called directly
                          "dill.load", "dill.loads", "jsonpickle.decode",
                          "marshal.load", "marshal.loads"}
# Bare attribute forms, so `from pickle import loads; loads(x)` / `yload(x)`
# (aliased imports) and `self.loads(...)`-style calls are still caught, mirroring
# how the SQL/path lists already match on the method name alone.
DESERIALIZE_SINK_ATTRS = {"unsafe_load", "full_load", "load_all"}
XXE_SINK_ATTRS = {"fromstring", "XML"}
XXE_SINK_NAMES = {"etree.parse", "etree.fromstring", "etree.XML",
                  "lxml.etree.fromstring", "lxml.etree.XML", "lxml.etree.parse"}
SSTI_SINK_ATTRS = {"from_string"}
SSTI_SINK_NAMES = {"render_template_string"}
SSRF_SINK_ATTRS = {"urlopen"}
SSRF_SINK_NAMES = {"urlopen", "urllib.request.urlopen", "request.urlopen",
                   "requests.get", "requests.post", "requests.put", "requests.head",
                   "requests.delete", "requests.patch", "requests.request",
                   # async / modern HTTP clients with the same SSRF shape
                   "httpx.get", "httpx.post", "httpx.request", "httpx.stream",
                   "aiohttp.request", "urllib3.request"}


def classify_sink(func_node: ast.FunctionDef) -> list[str]:
    """Return which CWE class this function's body plausibly touches (CWE-89/78/
    22 via taint_scan.py's taxonomy, plus CWE-502/611/918 via the sink patterns
    above), using the same name-based matching the rest of the project uses."""
    tags: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in SQLI_SINK_ATTRS:
            tags.add("CWE-89")
        if _is_call_named(node, CMDI_SINK_NAMES):
            tags.add("CWE-78")
        if _is_call_named(node, PATH_SINK_NAMES):
            tags.add("CWE-22")
        if isinstance(node.func, ast.Attribute) and node.func.attr in PATH_SINK_METHOD_NAMES:
            tags.add("CWE-22")
        if _is_call_named(node, DESERIALIZE_SINK_NAMES) or \
           (isinstance(node.func, ast.Attribute) and node.func.attr in DESERIALIZE_SINK_ATTRS):
            tags.add("CWE-502")
        if _is_call_named(node, XXE_SINK_NAMES) or \
           (isinstance(node.func, ast.Attribute) and node.func.attr in XXE_SINK_ATTRS):
            tags.add("CWE-611")
        if _is_call_named(node, SSRF_SINK_NAMES) or \
           (isinstance(node.func, ast.Attribute) and node.func.attr in SSRF_SINK_ATTRS):
            tags.add("CWE-918")
        if _is_call_named(node, SSTI_SINK_NAMES) or \
           (isinstance(node.func, ast.Attribute) and node.func.attr in SSTI_SINK_ATTRS):
            tags.add("CWE-1336")
    return sorted(tags)


def get_docstring(func_node: ast.FunctionDef) -> str | None:
    doc = ast.get_docstring(func_node)
    if doc and len(doc.strip()) >= 10:
        return doc
    return None


def synthesize_instruction(func_node: ast.FunctionDef) -> str:
    """When a function has no real docstring, build a minimal, generic
    instruction from its own name and parameters instead -- NOT a
    hand-written per-repo/per-function string. Splits camelCase/snake_case
    into words (e.g. `_execute` -> "execute", `batch_insert` -> "batch
    insert") so the prompt still reads like a plausible task description.
    This matters: requiring a real docstring on every mined function
    silently discarded ~all of the SQL-sink-touching code in every repo
    tested (90/90 in peewee alone) -- most library-internal helpers simply
    aren't documented. A name-derived fallback is the same approach
    HumanEval-style benchmarks use for undocumented real-world code."""
    name = func_node.name.strip("_")
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ").strip()
    params = [a.arg for a in func_node.args.args if a.arg not in ("self", "cls")]
    param_part = f" (parameters: {', '.join(params)})" if params else ""
    return f"Implement the `{func_node.name}` function{param_part}."


def strip_body_to_prompt(source_lines: list[str], func_node: ast.FunctionDef, docstring: str) -> str:
    """Keep the `def ...(...):` signature (which may span multiple lines
    for long argument lists) and the FULL docstring (which is very often
    multi-line itself); replace everything else with a clear implementation
    placeholder."""
    first_stmt = func_node.body[0]
    is_docstring_stmt = (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    )
    if is_docstring_stmt:
        # Use the docstring statement's OWN end line, not its start line --
        # a multi-line docstring's closing \"\"\" can be many lines below
        # first_stmt.lineno. Using only .lineno here previously truncated
        # every multi-line docstring mid-string, producing an unterminated
        # string literal (invalid Python) and silently discarding the rest
        # of the docstring's content.
        end_line = first_stmt.end_lineno or first_stmt.lineno
    else:
        # No docstring statement in source (shouldn't happen since we
        # filtered for ast.get_docstring() above, but stay defensive):
        # the signature ends the line before the real body starts.
        end_line = first_stmt.lineno - 1

    sig_lines = source_lines[func_node.lineno - 1: end_line]
    indent = " " * (func_node.col_offset + 4)

    prompt_lines = list(sig_lines)
    if not is_docstring_stmt:
        prompt_lines.append(f'{indent}"""{docstring.strip()}"""')
    prompt_lines.append(f"{indent}# TODO: implement this function")
    prompt_lines.append(f"{indent}raise NotImplementedError")
    return "\n".join(prompt_lines)


# ---------------------------------------------------------------------------
# CONSTRUCTION RECIPES -- Tier-1 auto_harness improvement.
#
# WHY THIS EXISTS: auto_harness.py's dominant remaining COULD_NOT_EXECUTE
# cause (see its own module docstring) is METHOD candidates needing state
# only a real __init__ sets up (self.conn, self._tracer, self.using_pty) --
# the harness's best-effort "call __init__ with generic placeholder args"
# heuristic covers some of this, but a generic placeholder is still a
# guess, not a real working object.
#
# The fix doesn't belong in auto_harness.py's guessing logic at all: the
# REAL repo already contains real, proven-to-work examples of how to build
# an instance of this exact class -- its own test suite. A pytest test file
# calling `Database(memory=True)` or `Runner(ctx)` is a real author
# demonstrating a real, working construction. This function searches the
# repo's own test files for the SIMPLEST such call (fewest args, only
# literal/trivial argument shapes we can safely re-execute later with no
# access to the rest of that test file's fixtures) and stores it as a
# construction recipe: a plain `(args...)` source snippet that
# auto_harness.py can later splice after the real class name and eval,
# getting a genuinely, correctly initialized real object -- not a `__new__`
# skeleton, not a guess.
#
# WHAT MAKES AN ARGUMENT "SAFE" TO REUSE: literals (strings, numbers,
# True/False/None), and the handful of near-universal pytest path fixtures
# (`tmp_path`, `tmpdir`) which auto_harness.py substitutes with its own
# scratch directory at run time -- never an arbitrary Name (which would
# refer to some OTHER fixture or local variable from that test file we do
# not have and cannot fabricate; that's exactly the kind of unverifiable
# guess this whole feature exists to avoid). A class with no safe, simple
# construction example anywhere in its own test suite just gets no recipe
# (None) -- auto_harness.py's existing __new__()-plus-best-effort-__init__
# fallback still runs exactly as before. This can only ADD successful real
# constructions, never remove or weaken the existing path.
# ---------------------------------------------------------------------------

_SUBSTITUTABLE_FIXTURE_NAMES = {"tmp_path", "tmpdir", "tmp_path_factory", "tmpdir_factory"}


def _is_safe_arg_node(node: ast.AST) -> bool:
    """True if this argument expression is simple/self-contained enough
    that auto_harness.py can safely re-execute it later with no access to
    anything else in the test file it came from -- see the section docstring
    above for exactly why each of these, and only these, shapes qualify."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_safe_arg_node(node.operand)
    if isinstance(node, ast.Name):
        return node.id in _SUBSTITUTABLE_FIXTURE_NAMES
    if isinstance(node, ast.Attribute):
        # e.g. `tmp_path.name` -- only chase this if the base is itself safe.
        return _is_safe_arg_node(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        # the extremely common `tmp_path / "test.db"` pathlib pattern.
        return _is_safe_arg_node(node.left) and _is_safe_arg_node(node.right)
    if isinstance(node, ast.Call):
        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
        if func_name in ("str", "Path", "PosixPath") and len(node.args) <= 1 and not node.keywords:
            return all(_is_safe_arg_node(a) for a in node.args)
        return False
    return False


def _is_safe_call_args(call_node: ast.Call) -> bool:
    return (
        all(_is_safe_arg_node(a) for a in call_node.args)
        and all(_is_safe_arg_node(kw.value) for kw in call_node.keywords)
    )


def _find_construction_recipe(repo_dir: Path, class_name: str) -> dict | None:
    """Scan the repo's OWN test files for the simplest real, safe-to-reuse
    call that builds a `class_name` instance. Returns None (never raises,
    never guesses) if nothing qualifies -- see the section docstring above
    for what "qualifies" means and why this is a strictly additive,
    never-destructive improvement over auto_harness.py's prior behavior."""
    test_files = sorted(
        p for p in repo_dir.rglob("*.py")
        if p.name == "conftest.py"
        or p.name.startswith("test_")
        or p.name.endswith("_test.py")
        or any(part in ("test", "tests") for part in p.relative_to(repo_dir).parts)
    )
    candidates = []
    for f in test_files:
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if name != class_name or not _is_safe_call_args(node):
                continue
            try:
                full_src = ast.unparse(node)
            except Exception:
                continue
            # Strip the leading "ClassName" (or "module.ClassName") off the
            # unparsed call, keeping just the "(...)" argument list -- that's
            # the part auto_harness.py will splice after ITS OWN resolved
            # real class name later (which may be reached via a different
            # attribute path than this test file used).
            paren_idx = full_src.find("(")
            if paren_idx == -1:
                continue
            args_src = full_src[paren_idx:]
            candidates.append({
                "call_args_source": args_src,
                "uses_tmp_path": "tmp_path" in args_src or "tmpdir" in args_src,
                "source_file": str(f.relative_to(repo_dir)),
                "source_line": node.lineno,
                "_complexity": len(node.args) + len(node.keywords),
            })
    if not candidates:
        return None
    # Prefer fewer arguments (less likely to hit something we mis-judged as
    # "safe"), and among ties, prefer NOT needing the tmp_path substitution
    # (strictly more self-contained).
    candidates.sort(key=lambda c: (c["_complexity"], c["uses_tmp_path"]))
    best = dict(candidates[0])
    del best["_complexity"]
    return best


@lru_cache(maxsize=None)
def _cached_construction_recipe(repo_dir: Path, class_name: str) -> dict | None:
    # Memoized per (repo, class) within one mining run -- a popular class
    # (e.g. a library's main Database/Connection class) is the `self`-owning
    # class for many mined methods, and re-walking every test file in the
    # repo for each one would be wasteful; the answer can't change mid-run.
    return _find_construction_recipe(repo_dir, class_name)


def _iter_functions_with_class(tree: ast.AST):
    """Yields (func_node, enclosing_class_name_or_None) for every function
    or method in the module -- same class-tracking walk as auto_harness.py's
    _find_target, duplicated here deliberately (mine_prompts.py has no
    reason to import from the harness side) so a mined METHOD's recipe
    lookup knows which class it belongs to, which plain ast.walk() does
    not track."""
    def visit(node, class_name):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, class_name
                yield from visit(child, class_name)
            else:
                yield from visit(child, class_name)
    yield from visit(tree, None)


def mine_file(path: Path, repo_name: str, repo_root: Path) -> list[dict]:
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    records = []
    for node, enclosing_class in _iter_functions_with_class(tree):
        tags = classify_sink(node)
        if not tags:
            continue  # not on-topic for our three CWEs

        body_line_count = (node.end_lineno or node.lineno) - node.lineno
        if body_line_count < MIN_BODY_LINES or body_line_count > MAX_BODY_LINES:
            continue

        real_docstring = get_docstring(node)
        had_real_docstring = real_docstring is not None
        # Docstring presence is NOT a mining gate (see synthesize_instruction's
        # docstring for why) -- it only decides which instruction text we use.
        instruction = real_docstring if had_real_docstring else synthesize_instruction(node)

        ground_truth_source = "\n".join(
            source_lines[node.lineno - 1: (node.end_lineno or node.lineno)]
        )
        prompt_source = strip_body_to_prompt(source_lines, node, instruction)

        is_method = bool(node.args.args) and node.args.args[0].arg in ("self", "cls")
        construction_recipe = (
            _cached_construction_recipe(repo_root, enclosing_class)
            if is_method and enclosing_class else None
        )

        records.append({
            "repo": repo_name,
            "file": str(path.relative_to(repo_root)),
            "function": node.name,
            "line": node.lineno,
            "candidate_cwes": tags,
            "had_real_docstring": had_real_docstring,
            "docstring": instruction.strip(),
            "prompt_source": prompt_source,
            "ground_truth_source": ground_truth_source,
            # The REAL, complete, untouched source of the file this function
            # came from -- not just the function's own lines. Added so
            # auto_harness.py can execute the candidate as a real imported
            # module instead of a single function exec'd in isolation: with
            # the whole file, a method's real class, real base class, and
            # any sibling functions/methods it calls (e.g. a class's
            # __init__ calling a module-level helper) are all genuinely
            # present, not faked with a synthetic stand-in. This does NOT
            # make cross-FILE imports resolvable (an import from elsewhere
            # in the same package still needs that package installed, and
            # fails as an honest, reported COULD_NOT_EXECUTE when it isn't)
            # -- it only removes the single-function-in-isolation gap.
            "module_source": source,
            # Tier-1 auto_harness improvement -- see the section docstring
            # above _find_construction_recipe for the full rationale. None
            # when this isn't a method, its class couldn't be determined,
            # or no safe/simple real construction example exists anywhere
            # in the repo's own test suite; auto_harness.py falls back to
            # its prior __new__()-plus-best-effort-__init__ behavior in
            # every one of those cases, unchanged.
            "construction_recipe": construction_recipe,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", type=Path, help="Path to an already-cloned repo")
    ap.add_argument("--repo-url", type=str, help="Git URL to shallow-clone into a temp dir")
    ap.add_argument("--repo-name", type=str, help="Label for output folder (default: inferred)")
    ap.add_argument("--limit", type=int, default=25, help="Max functions to mine")
    ap.add_argument("--max-files", type=int, default=15000,
                    help="Skip a repo with more than this many .py files (default 15000): only "
                         "skips truly pathological monorepos. Large real projects (e.g. litellm, "
                         "~3k files) now mine. Set 0 to disable the guard entirely.")
    ap.add_argument("--require-ai-provenance", action="store_true",
                    help="REAL-AI-REPO arm: mine ONLY functions whose lines git-blame to a "
                         "commit with strong AI-authorship evidence (see analysis/provenance.py). "
                         "Each kept candidate is stamped with its auditable evidence. Forces a "
                         "FULL clone (history needed) and refuses a shallow one.")
    ap.add_argument("--keep-clone", action="store_true",
                    help="keep the temp clone made for --repo-url (default: delete it after mining, "
                         "since the importable source is already vendored into _repo_src -- this stops "
                         "a 32-repo batch from filling /tmp and silently truncating the run)")
    ap.add_argument("--out-root", type=Path, default=OUT_DIR,
                    help="corpus root to write into (default: the frozen v1 root). Corpus v2 uses "
                         "calibration/mined_prompts_v2 so v1 is never touched.")
    ap.add_argument("--stamp-provenance", action="store_true",
                    help="Attribute AI authorship per function and record it on every candidate, "
                         "but do NOT exclude non-AI functions (annotate-only).")
    args = ap.parse_args()

    if not args.repo_dir and not args.repo_url:
        ap.error("provide --repo-dir or --repo-url")

    want_provenance = args.require_ai_provenance or args.stamp_provenance

    tmp_clone = None
    if args.repo_url:
        tmp_clone = Path(tempfile.mkdtemp(prefix="mine_prompts_"))
        # Provenance needs real git history -> full clone. Otherwise keep the
        # cheap --depth 1 clone (mining itself never looks at history).
        # RESILIENT CLONE: big repos over a flaky link throw transient
        # "RPC failed; curl 92 HTTP/2 stream ... CANCEL / early EOF". Force
        # HTTP/1.1 (the standard fix for that curl-92 error), bump postBuffer,
        # and retry a few times. On final failure, skip this repo cleanly
        # (clear message, no scary traceback) so a 32-repo batch loses one repo
        # at worst instead of aborting.
        base = ["git", "-c", "http.version=HTTP/1.1", "-c", "http.postBuffer=524288000",
                "clone"] + ([] if want_provenance else ["--depth", "1"])
        clone_cmd = base + [args.repo_url, str(tmp_clone)]
        print(f"Cloning {args.repo_url} -> {tmp_clone} "
              f"({'full history' if want_provenance else 'shallow'}) ...", file=sys.stderr)
        import shutil as _sh0
        ok = False
        for attempt in range(1, 4):
            if tmp_clone.exists():
                _sh0.rmtree(tmp_clone, ignore_errors=True)
            tmp_clone.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(clone_cmd, capture_output=True, text=True)
            if r.returncode == 0:
                ok = True
                break
            print(f"  clone attempt {attempt}/3 failed (rc={r.returncode}): "
                  f"{(r.stderr or '').strip().splitlines()[-1] if r.stderr else ''}", file=sys.stderr)
        if not ok:
            _sh0.rmtree(tmp_clone, ignore_errors=True)
            print(f"SKIP {args.repo_url}: clone failed after 3 attempts (transient network). "
                  f"Re-run later to include it.", file=sys.stderr)
            raise SystemExit(0)   # clean exit -> the batch driver moves on, no traceback
        repo_dir = tmp_clone
    else:
        repo_dir = args.repo_dir

    _prov = None
    if want_provenance:
        try:
            import provenance as _prov  # sibling module in analysis/
        except ImportError:
            _here = str(Path(__file__).resolve().parent)
            if _here not in sys.path:
                sys.path.insert(0, _here)
            import provenance as _prov
        if not (repo_dir / ".git").exists():
            raise SystemExit(f"--require/stamp-ai-provenance needs a git repo, but {repo_dir} "
                             f"has no .git history. Clone with --repo-url (full) or point "
                             f"--repo-dir at a full clone.")
        if _prov.is_shallow(repo_dir):
            raise SystemExit(f"{repo_dir} is a SHALLOW clone -- git history is truncated, so AI "
                             f"authorship cannot be attributed. Run: git -C {repo_dir} fetch "
                             f"--unshallow  (or re-clone without --depth).")

    repo_name = args.repo_name or repo_dir.name

    py_files = [p for p in repo_dir.rglob("*.py") if not _is_skippable(p, repo_dir)]
    if args.max_files and len(py_files) > args.max_files:
        print(f"SKIP {repo_name}: {len(py_files)} .py files exceeds --max-files "
              f"{args.max_files} -- a repo this large is a mature library, not a representative "
              f"AI-built project, and mining it is slow. Raise --max-files to include it anyway.",
              file=sys.stderr)
        raise SystemExit(0)   # clean exit -> batch driver (discover_repos, check=False) moves on
    print(f"Scanning {len(py_files)} .py file(s) in {repo_dir} ...", file=sys.stderr)

    all_records: list[dict] = []
    for py_file in sorted(py_files):
        all_records.extend(mine_file(py_file, repo_name, repo_dir))

    # AI-provenance attribution (REAL-AI-REPO arm). Attribute every candidate
    # BEFORE the CWE-spread selection, so a gated run draws its spread from the
    # AI-authored pool rather than selecting first and dropping most. Each
    # record gets an `ai_provenance` block (auditable evidence); with
    # --require-ai-provenance, non-AI-attributed functions are excluded here.
    if want_provenance:
        attributed = 0
        for r in all_records:
            fp = _prov.function_provenance(repo_dir, r["file"], r.get("line", 0))
            r["ai_provenance"] = {
                "ai_attributed": fp["ai_attributed"],
                "ai_tools": fp.get("ai_tools", {}),
                "ai_commit": fp.get("ai_commit"),
                "evidence": fp.get("evidence"),
                "ai_line_fraction": fp.get("ai_line_fraction", 0.0),
                "span": fp.get("span"),
            }
            attributed += bool(fp["ai_attributed"])
        print(f"AI-provenance: {attributed}/{len(all_records)} candidate functions are "
              f"AI-attributed (strong evidence).", file=sys.stderr)
        if args.require_ai_provenance:
            kept = [r for r in all_records if r["ai_provenance"]["ai_attributed"]]
            print(f"--require-ai-provenance: keeping {len(kept)} AI-authored, dropping "
                  f"{len(all_records) - len(kept)} human/unattributed.", file=sys.stderr)
            all_records = kept

    # Prefer a spread across CWEs and files over just the first N matches.
    by_cwe: dict[str, list[dict]] = {}
    for r in all_records:
        for cwe in r["candidate_cwes"]:
            by_cwe.setdefault(cwe, []).append(r)

    selected: list[dict] = []
    seen_ids = set()
    round_robin_cwes = sorted(by_cwe.keys())
    idx = {cwe: 0 for cwe in round_robin_cwes}
    while len(selected) < args.limit and any(idx[c] < len(by_cwe[c]) for c in round_robin_cwes):
        for cwe in round_robin_cwes:
            if len(selected) >= args.limit:
                break
            lst = by_cwe[cwe]
            while idx[cwe] < len(lst):
                r = lst[idx[cwe]]
                idx[cwe] += 1
                rid = (r["file"], r["function"], r["line"])
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    selected.append(r)
                    break

    out_subdir = args.out_root / repo_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    for r in selected:
        safe_name = f"{r['file'].replace('/', '__')}__{r['function']}.json"
        (out_subdir / safe_name).write_text(json.dumps(r, indent=2))

    summary_lines = [
        f"# Mined prompts: {repo_name}\n",
        f"{len(all_records)} on-topic candidate function(s) found across {len(py_files)} files.",
        f"{len(selected)} selected (spread across CWEs, capped at --limit {args.limit}).\n",
    ]
    for cwe in round_robin_cwes:
        n = sum(1 for r in selected if cwe in r["candidate_cwes"])
        summary_lines.append(f"- {cwe}: {n} selected (of {len(by_cwe[cwe])} total candidates)")
    summary_lines.append("\n## Selected functions\n")
    for r in selected:
        summary_lines.append(f"- **{r['function']}** ({', '.join(r['candidate_cwes'])}) -- `{r['file']}:{r['line']}`")
    (out_subdir / "SUMMARY.md").write_text("\n".join(summary_lines))

    # Vendor the repo's importable source beside the candidates so the harness
    # can import the REAL package (resolving its relative imports) at detect/
    # recycle time WITHOUT a working pip/wheel build -- the dominant construction
    # failure for cloned-but-not-pip-installable repos (poetry / PEP-517 backends
    # whose wheel build fails). The source travels with the corpus, so the
    # benchmark stays reproducible; run_detection/run_recycle add this dir to
    # sys.path. Only top-level importable packages/modules are copied (not tests,
    # docs, .git) -- construction recipes were already extracted into records.
    import shutil as _shutil
    vendor = out_subdir / "_repo_src"
    if vendor.exists():
        _shutil.rmtree(vendor)
    vendor.mkdir()
    # Copy EVERY .py preserving directory structure (not just top-level packages).
    # FOUND BY THE CORPUS CENSUS: the old "top-level packages + src/ only" rule
    # returned "nothing found" for repos whose code lives in nested or non-
    # standard dirs (backend/app/..., flat script folders), and those repos then
    # failed at execution with NameError/AttributeError because their real
    # modules were never importable. Copying only .py (no data/assets/.git) keeps
    # _repo_src small while making any layout importable; a runtime read of a
    # non-.py data file is a separate, honest FileNotFoundError, not an import gap.
    _SKIP_PARTS = {".git", "__pycache__", "node_modules", ".venv", "venv",
                   "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", "site-packages"}
    n_py = 0
    for py in repo_dir.rglob("*.py"):
        rel = py.relative_to(repo_dir)
        if any(part in _SKIP_PARTS for part in rel.parts):
            continue
        try:
            dest = vendor / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(py, dest)
            n_py += 1
        except Exception as e:  # noqa: BLE001 -- vendoring is best-effort
            print(f"  (vendor skip {rel}: {e})", file=sys.stderr)

    # Compute the sys.path roots that make these modules importable, and record
    # them in .pathdirs (relative to _repo_src). Consumers add each to sys.path.
    #   - the vendor root itself (flat top-level modules / top-level packages)
    #   - src/ if present
    #   - the PARENT of every top-of-chain package (so nested packages import)
    #   - the directory of each mined candidate file (flat scripts with no package)
    path_roots = {"."}
    if (vendor / "src").is_dir():
        path_roots.add("src")
    for initf in vendor.rglob("__init__.py"):
        d = initf.parent
        # climb to the top of the package chain
        while d != vendor and (d.parent / "__init__.py").exists() and d.parent != vendor:
            d = d.parent
        # the sys.path root is the package's PARENT -- but if the package sits
        # at the vendor root itself (a repo with a top-level __init__.py, e.g.
        # ComfyUI custom nodes), its parent is OUTSIDE vendor; the root is just
        # "." then. (Without this guard relative_to() raised ValueError and
        # crashed the whole mine -- found on hdzwzqbxlk/ComfyUI...).
        if d == vendor:
            path_roots.add(".")
        else:
            path_roots.add(d.parent.relative_to(vendor).as_posix() or ".")
    for r in selected:
        cand_dir = (vendor / r["file"]).parent
        if cand_dir.is_dir() and cand_dir != vendor:
            try:
                path_roots.add(cand_dir.relative_to(vendor).as_posix() or ".")
            except ValueError:
                pass
    (vendor / ".pathdirs").write_text("\n".join(sorted(path_roots)) + "\n")
    print(f"Vendored importable source -> {vendor.relative_to(ROOT)}/ "
          f"({n_py} .py files, {len(path_roots)} import roots)"
          if n_py else f"Vendored importable source -> {vendor.relative_to(ROOT)}/ (nothing found)",
          file=sys.stderr)

    print(f"\n{len(all_records)} on-topic candidates found, {len(selected)} mined -> {out_subdir.relative_to(ROOT)}/")
    print(f"See {out_subdir.relative_to(ROOT)}/SUMMARY.md for an overview.")

    if tmp_clone:
        if args.keep_clone:
            print(f"(Cloned repo left at {tmp_clone} -- --keep-clone set.)", file=sys.stderr)
        else:
            import shutil as _sh
            _sh.rmtree(tmp_clone, ignore_errors=True)
            print(f"(Deleted temp clone {tmp_clone}; vendored source is in _repo_src.)", file=sys.stderr)


if __name__ == "__main__":
    main()

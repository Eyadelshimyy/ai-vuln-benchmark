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
from functools import lru_cache
from pathlib import Path

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


def _is_skippable(path: Path, repo_root: Path) -> bool:
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        rel_parts = path.parts
    return any(part in SKIP_DIR_COMPONENTS for part in rel_parts)

MIN_BODY_LINES = 4     # skip trivial one-liners -- nothing for a model to get wrong
MAX_BODY_LINES = 60    # skip huge functions -- unrealistic prompt, unclear ground truth


def classify_sink(func_node: ast.FunctionDef) -> list[str]:
    """Return which of CWE-89/78/22 this function's body plausibly touches,
    using the identical sink-name matching taint_scan.py uses at scan time."""
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
    args = ap.parse_args()

    if not args.repo_dir and not args.repo_url:
        ap.error("provide --repo-dir or --repo-url")

    tmp_clone = None
    if args.repo_url:
        tmp_clone = Path(tempfile.mkdtemp(prefix="mine_prompts_"))
        print(f"Cloning {args.repo_url} -> {tmp_clone} ...", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth", "1", args.repo_url, str(tmp_clone)],
            check=True,
        )
        repo_dir = tmp_clone
    else:
        repo_dir = args.repo_dir

    repo_name = args.repo_name or repo_dir.name

    py_files = [p for p in repo_dir.rglob("*.py") if not _is_skippable(p, repo_dir)]
    print(f"Scanning {len(py_files)} .py file(s) in {repo_dir} ...", file=sys.stderr)

    all_records: list[dict] = []
    for py_file in sorted(py_files):
        all_records.extend(mine_file(py_file, repo_name, repo_dir))

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

    out_subdir = OUT_DIR / repo_name
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

    print(f"\n{len(all_records)} on-topic candidates found, {len(selected)} mined -> {out_subdir.relative_to(ROOT)}/")
    print(f"See {out_subdir.relative_to(ROOT)}/SUMMARY.md for an overview.")

    if tmp_clone:
        print(f"(Cloned repo left at {tmp_clone} -- delete it manually if you don't need it.)", file=sys.stderr)


if __name__ == "__main__":
    main()

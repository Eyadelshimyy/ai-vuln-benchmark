"""
Hand-built static taint analyzer for CWE-89 (SQL Injection), CWE-78 (OS
Command Injection), and CWE-22 (Path Traversal) in Python source.

This is deliberately NOT a general-purpose taint engine (that's what
CodeQL/Semgrep/Pysa are for) -- it is scoped tightly to the three
vulnerability classes this thesis studies, using Python's own `ast`
module with no third-party dependency. That scoping is what makes it
tractable to hand-build and reason about correctness for, and it means
this tool has no dependency on any external detection pipeline: it is
a self-contained replacement for that stage of the project.

Design (per function):
  1. SEED    -- every parameter is a taint source EXCEPT those whose name
     matches a small "trusted context" denylist (see TRUSTED_PARAM_NAMES
     below: base_dir, conn, config, and similar). This matters: a naive
     "every parameter is tainted" seed produces a real false positive on
     the path_traversal/safe_2.py calibration sample, where os.path.join
     taints its result if ANY argument is tainted -- base_dir being
     (wrongly) treated as attacker-controlled was enough to trigger a
     false "vulnerable" verdict even though the actual untrusted
     parameter (filename) was correctly sanitized via os.path.basename.
     This is a documented, name-based heuristic, not semantic analysis --
     see the Limitations note near TRUSTED_PARAM_NAMES.
  2. PROPAGATE -- assignment targets become tainted if the right-hand
     side is built from any tainted name, via:
       - string concatenation (BinOp Add)
       - %-style formatting (BinOp Mod)
       - str.format(...) calls
       - f-strings (JoinedStr / FormattedValue)
       - os.path.join(...) with any tainted argument
  3. SINK CHECK -- at every Call node, ask "does this call match a
     known-dangerous sink for one of the three CWEs, with a tainted
     argument, and no recognized guard in this function?"

This is intentionally an intraprocedural, single-file, best-effort
analysis: no cross-function or cross-file taint tracking, no alias
analysis. That is a named limitation, not an oversight -- see README.

Usage:
    python3 taint_scan.py --dir ../calibration/python_handbuilt/sqli
    python3 taint_scan.py --file some_generated_sample.py --out findings.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


# Parameter names treated as trusted context rather than attacker-controlled
# input, by naming convention. This is a real limitation, not a nicety: a
# "taint every parameter" seed is too coarse whenever a function takes both
# a trusted value (a base directory, a DB connection, a config object) and
# an untrusted one (what the caller/attacker actually supplies) -- which is
# exactly the shape of every path_traversal and sqli calibration sample in
# this repo. Extend this set if real generated code uses other conventional
# names for trusted context parameters (e.g. "root", "workdir", "db",
# "session", "settings").
TRUSTED_PARAM_NAMES = {
    "self", "cls",
    "conn", "connection", "cursor", "cur", "db", "session",
    "base_dir", "base_path", "basedir", "basepath", "root_dir", "root",
    "marker_path", "config", "settings", "context", "ctx",
}


@dataclass
class Finding:
    file: str
    function: str
    line: int
    cwe: str
    sink: str
    tainted_param: str
    snippet: str

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "function": self.function,
            "line": self.line,
            "cwe": self.cwe,
            "sink": self.sink,
            "tainted_param": self.tainted_param,
            "snippet": self.snippet,
        }


# ---------------------------------------------------------------------------
# Taint propagation within a single function body
# ---------------------------------------------------------------------------

class TaintState:
    """Tracks which local names are tainted, and which are "guarded"
    (passed through a recognized path-containment check), for one
    function body."""

    def __init__(self, seed_names: set[str], tainted_attrs: set[str] | None = None):
        self.tainted: set[str] = set(seed_names)
        self.guarded: set[str] = set()  # names proven safe by a containment check
        # self.<attr> names known tainted from a class-level pre-pass over
        # __init__ (see _compute_tainted_self_attrs). Empty for a function
        # scanned standalone, or for a class with no such attribute.
        self.tainted_attrs: set[str] = set(tainted_attrs) if tainted_attrs else set()

    def expr_is_tainted(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.tainted
        if isinstance(node, ast.BinOp):
            return self.expr_is_tainted(node.left) or self.expr_is_tainted(node.right)
        if isinstance(node, ast.JoinedStr):  # f-string
            return any(
                isinstance(v, ast.FormattedValue) and self.expr_is_tainted(v.value)
                for v in node.values
            )
        if isinstance(node, ast.Call):
            # os.path.basename(...) strips every directory component, so
            # its output can never carry a path-traversal payload -- treat
            # it as a sanitizer, not a pass-through, regardless of its
            # argument's taint state.
            if _is_call_named(node, {"os.path.basename", "path.basename", "basename"}):
                return False
            # str.format(...) / "{}".format(x) -- tainted if any arg is tainted
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return any(self.expr_is_tainted(a) for a in node.args) or self.expr_is_tainted(
                    node.func.value
                )
            # os.path.join(...) -- tainted if ANY argument is tainted
            if _is_call_named(node, {"os.path.join", "path.join", "join"}):
                return any(self.expr_is_tainted(a) for a in node.args)
            # generic: if we don't recognize the call, be conservative and
            # check whether any argument is tainted (covers wrapper funcs)
            return any(self.expr_is_tainted(a) for a in node.args)
        if isinstance(node, ast.Attribute):
            # self.<attr> where <attr> was proven tainted by the class-level
            # __init__ pre-pass (e.g. self.table_prefix assigned from a
            # tainted constructor parameter). This is the one deliberate
            # exception to "no cross-function tracking": it is a same-class,
            # constructor-to-method hop, found necessary via real-CVE testing
            # (Glances Cassandra exporter, PraisonAI SQLiteConversationStore).
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in self.tainted_attrs
            ):
                return True
            return self.expr_is_tainted(node.value)
        if isinstance(node, (ast.Constant,)):
            return False
        return False

    def expr_is_guarded(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.guarded
        if isinstance(node, ast.Call):
            return any(self.expr_is_guarded(a) for a in node.args)
        return False

    def apply_assignment(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, ast.Name):
            if self.expr_is_tainted(value):
                self.tainted.add(target.id)
            elif target.id in self.tainted:
                # reassigned to something non-tainted -- drop it
                self.tainted.discard(target.id)
            return
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            # self.attr = <expr> -- used both by the __init__ pre-pass (to
            # discover which attrs carry taint) and, with that result fed
            # back in as seed_names, by every other method in the class.
            if self.expr_is_tainted(value):
                self.tainted_attrs.add(target.attr)
            elif target.attr in self.tainted_attrs:
                self.tainted_attrs.discard(target.attr)


def _is_call_named(node: ast.Call, names: set[str]) -> bool:
    """Match a Call node's callable against dotted or bare names,
    e.g. {"os.system"} matches both `os.system(...)` and, if `system`
    was imported directly, `system(...)`."""
    func = node.func
    if isinstance(func, ast.Attribute):
        parts = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        dotted = ".".join(reversed(parts))
        return dotted in names or func.attr in names
    if isinstance(func, ast.Name):
        return func.id in names
    return False


def _has_positional_or_keyword(node: ast.Call, kw: str) -> ast.AST | None:
    for k in node.keywords:
        if k.arg == kw:
            return k.value
    return None


def _snippet(source_lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1].strip()
    return ""


# ---------------------------------------------------------------------------
# Per-function guard detection (containment checks, parameterized queries)
# ---------------------------------------------------------------------------

def _function_regex_validated_names(func_node: ast.FunctionDef) -> set[str]:
    """Heuristic: a name is treated as validated (safe against shell
    metacharacters) if the function calls `<compiled_pattern>.match(name)`,
    `<compiled_pattern>.fullmatch(name)`, `re.match(pattern, name)`, or
    `re.fullmatch(pattern, name)` anywhere in its body. This is coarse
    (it doesn't verify the pattern actually rejects shell metacharacters,
    and it doesn't check the match result is used to gate the sink) --
    a documented precision/recall tradeoff, same as the path-guard check."""
    validated: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"match", "fullmatch"}:
            # <pattern>.match(name) -- the validated value is the call argument
            for a in node.args:
                if isinstance(a, ast.Name):
                    validated.add(a.id)
        if _is_call_named(node, {"re.match", "re.fullmatch"}) and len(node.args) >= 2:
            a = node.args[1]
            if isinstance(a, ast.Name):
                validated.add(a.id)
    return validated


def _function_has_path_guard(func_node: ast.FunctionDef) -> bool:
    """Heuristic: the function is treated as guarding path traversal if it
    calls os.path.realpath/abspath AND separately calls os.path.commonpath
    or does a .startswith(...) comparison anywhere in its body. This is a
    coarse, function-wide check (not flow-sensitive to the specific open()
    call) -- a deliberate, documented precision/recall tradeoff."""
    has_resolve = False
    has_containment_check = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and _is_call_named(
            node, {"os.path.realpath", "path.realpath", "realpath",
                   "os.path.abspath", "path.abspath", "abspath"}
        ):
            has_resolve = True
        if isinstance(node, ast.Call) and _is_call_named(
            node, {"os.path.commonpath", "path.commonpath", "commonpath",
                   "os.path.commonprefix", "commonprefix"}
        ):
            has_containment_check = True
        if isinstance(node, ast.Attribute) and node.attr == "startswith":
            has_containment_check = True
    return has_resolve and has_containment_check


# ---------------------------------------------------------------------------
# Main per-function scan
# ---------------------------------------------------------------------------

SQLI_SINK_ATTRS = {"execute", "executescript", "prepare"}
CMDI_SINK_NAMES = {
    "os.system", "system",
    "os.popen", "popen",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_call", "subprocess.check_output",
    "run", "call", "Popen", "check_call", "check_output",
}
# Sinks where the dangerous value is a CALL ARGUMENT (like builtin open(path)):
# both plain file-open calls and framework/library constructors that read a
# file internally given a path argument (FastAPI's FileResponse, Flask's
# send_file, nltk's XMLCorpusView, etc). Real-world code serves files through
# these far more often than through a raw open() -- found via real-CVE testing
# against khoj (FileResponse) and nltk (XMLCorpusView).
PATH_SINK_NAMES = {"open", "FileResponse", "send_file", "StaticFiles", "XMLCorpusView"}
# Sinks where the dangerous value is the METHOD'S RECEIVER, not its argument:
# path_obj.write_text(content) -- the traversal risk is in `path_obj` (how it
# was built), not in `content`. Found via real-CVE testing against banks'
# DirectoryPromptRegistry.
PATH_SINK_METHOD_NAMES = {"write_text", "write_bytes"}


def _compute_tainted_self_attrs(class_node: ast.ClassDef) -> set[str]:
    """Class-level pre-pass: run the same seed+propagate logic as
    scan_function, but over __init__ only, to find which self.<attr>
    assignments are built from a tainted constructor parameter. The result
    is fed back into TaintState as seed `tainted_attrs` when scanning every
    OTHER method in the class -- the one deliberate, narrow exception to
    this tool's otherwise-intraprocedural design (see module docstring).

    Limitation: only looks at __init__ (not classmethod factories or other
    setup methods), and only one hop deep via the same linear, non-branch-
    sensitive walk scan_function uses -- same documented tradeoffs as the
    rest of this tool.
    """
    init_node = None
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            init_node = item
            break
    if init_node is None:
        return set()

    param_names = {a.arg for a in init_node.args.args}
    tainted_param_names = param_names - TRUSTED_PARAM_NAMES
    state = TaintState(seed_names=tainted_param_names)
    for node in ast.walk(init_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                state.apply_assignment(target, node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.op, ast.Add) and state.expr_is_tainted(node.value):
                state.tainted.add(node.target.id)
    return state.tainted_attrs


def scan_function(
    func_node: ast.FunctionDef,
    filename: str,
    source_lines: list[str],
    class_tainted_attrs: set[str] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    all_param_names = {a.arg for a in func_node.args.args}
    tainted_param_names = all_param_names - TRUSTED_PARAM_NAMES
    class_tainted_attrs = class_tainted_attrs or set()
    if not tainted_param_names and not class_tainted_attrs:
        return findings

    state = TaintState(seed_names=tainted_param_names, tainted_attrs=class_tainted_attrs)
    has_path_guard = _function_has_path_guard(func_node)
    regex_validated = _function_regex_validated_names(func_node)

    # Single linear pass over the function body in source order. This is
    # deliberately simple (no branch-sensitive flow analysis) -- taint set
    # is updated in statement order, which is sufficient for the
    # straight-line calibration-style patterns this scanner targets.
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                state.apply_assignment(target, node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.op, ast.Add) and state.expr_is_tainted(node.value):
                state.tainted.add(node.target.id)

        if isinstance(node, ast.Call):
            # --- CWE-89: SQL Injection ---
            if isinstance(node.func, ast.Attribute) and node.func.attr in SQLI_SINK_ATTRS:
                if node.args and state.expr_is_tainted(node.args[0]):
                    # NOTE: a second args[1] (a params tuple/list) only binds
                    # VALUES into ?/%s placeholders -- it does nothing to
                    # un-taint the query STRING itself. If args[0] is tainted,
                    # it's a genuine CWE-89 regardless of whether a params
                    # tuple is also passed (e.g. identifier/table-name
                    # injection baked into an f-string, with a *separate*,
                    # correctly-parameterized value tuple alongside it --
                    # found via real-CVE testing on PraisonAI's
                    # SQLiteConversationStore, where every query does both).
                    # A previous version of this check wrongly treated "any
                    # execute() with 2+ args" as safe, which suppressed this
                    # exact case.
                    findings.append(Finding(
                        file=filename, function=func_node.name, line=node.lineno,
                        cwe="CWE-89", sink=f".{node.func.attr}(...)",
                        tainted_param=_first_tainted_name(node.args[0], tainted_param_names) or "?",
                        snippet=_snippet(source_lines, node.lineno),
                    ))

            # --- CWE-78: OS Command Injection ---
            if _is_call_named(node, CMDI_SINK_NAMES):
                shell_kw = _has_positional_or_keyword(node, "shell")
                uses_shell = isinstance(shell_kw, ast.Constant) and shell_kw.value is True
                bare_system_call = _is_call_named(node, {"os.system", "system", "os.popen", "popen"})
                first_arg_is_str_not_list = bool(node.args) and not isinstance(node.args[0], ast.List)
                tainted_arg = any(state.expr_is_tainted(a) for a in node.args) or (
                    shell_kw is not None and state.expr_is_tainted(shell_kw)
                )
                arg_names_used = {
                    n.id for a in node.args for n in ast.walk(a) if isinstance(n, ast.Name)
                }
                is_validated = bool(arg_names_used) and arg_names_used.issubset(regex_validated)
                if tainted_arg and not is_validated and (
                    bare_system_call or (uses_shell and first_arg_is_str_not_list)
                ):
                    findings.append(Finding(
                        file=filename, function=func_node.name, line=node.lineno,
                        cwe="CWE-78", sink=_dotted_name(node),
                        tainted_param=_first_tainted_name(
                            node.args[0] if node.args else shell_kw, tainted_param_names
                        ) or "?",
                        snippet=_snippet(source_lines, node.lineno),
                    ))

            # --- CWE-22: Path Traversal ---
            if _is_call_named(node, PATH_SINK_NAMES):
                if node.args and state.expr_is_tainted(node.args[0]) and not has_path_guard:
                    findings.append(Finding(
                        file=filename, function=func_node.name, line=node.lineno,
                        cwe="CWE-22", sink=_dotted_name(node),
                        tainted_param=_first_tainted_name(node.args[0], tainted_param_names) or "?",
                        snippet=_snippet(source_lines, node.lineno),
                    ))
            elif isinstance(node.func, ast.Attribute) and node.func.attr in PATH_SINK_METHOD_NAMES:
                # Method-style sink: the receiver (e.g. `path_obj` in
                # `path_obj.write_text(...)`) carries the traversal risk, not
                # the call's arguments.
                receiver = node.func.value
                if state.expr_is_tainted(receiver) and not has_path_guard:
                    findings.append(Finding(
                        file=filename, function=func_node.name, line=node.lineno,
                        cwe="CWE-22", sink=f".{node.func.attr}(...)",
                        tainted_param=_first_tainted_name(receiver, tainted_param_names) or "?",
                        snippet=_snippet(source_lines, node.lineno),
                    ))

    return findings


def _dotted_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        parts = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts)) + "(...)"
    if isinstance(func, ast.Name):
        return func.id + "(...)"
    return "<call>"


def _first_tainted_name(node: ast.AST | None, param_names: set[str]) -> str | None:
    if node is None:
        return None
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in param_names:
            return n.id
    # fall through: find any Name at all (covers locals derived from params)
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            return n.id
    return None


def scan_file(path: Path) -> list[Finding]:
    source = path.read_text()
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        print(f"warning: could not parse {path}: {e}", file=sys.stderr)
        return []

    # Pre-pass: for every class in the file, compute which self.<attr>
    # names carry taint from the constructor (see _compute_tainted_self_attrs
    # for why). Only a class's own direct methods get its attrs -- a nested
    # closure inside a method is scanned with an empty set, same as before.
    method_tainted_attrs: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_tainted_attrs = _compute_tainted_self_attrs(node)
            if class_tainted_attrs:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_tainted_attrs[id(item)] = class_tainted_attrs

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(
                scan_function(
                    node, str(path), source_lines,
                    class_tainted_attrs=method_tainted_attrs.get(id(node)),
                )
            )
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, help="Scan all .py files in this directory")
    ap.add_argument("--file", type=Path, help="Scan a single .py file")
    ap.add_argument("--out", type=Path, help="Write JSON findings to this file (default: stdout)")
    args = ap.parse_args()

    if not args.dir and not args.file:
        ap.error("provide --dir or --file")

    targets: list[Path] = []
    if args.file:
        targets.append(args.file)
    if args.dir:
        targets.extend(sorted(args.dir.glob("*.py")))

    all_findings: list[Finding] = []
    for t in targets:
        all_findings.extend(scan_file(t))

    payload = [f.to_dict() for f in all_findings]
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)

    print(f"\n{len(all_findings)} finding(s) across {len(targets)} file(s).", file=sys.stderr)


if __name__ == "__main__":
    main()

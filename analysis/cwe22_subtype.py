#!/usr/bin/env python3
"""CWE-22 subtype stratifier (static, AST-based; a STRATIFICATION, never a relabel).

Why this exists. Every CWE-22 Tier-2 confirmation is proven by the function
opening an attacker-chosen ABSOLUTE path (tier2/strategies.py: the `../` ladder
templates cannot name the random planted file, so they can never satisfy any
CWE-22 verifier). That one observable covers two quite different findings:

  BASE_JOIN_OVERRIDE  the function composes a path from a base it controls and a
                      parameter (os.path.join(base, p), base / p, f"{base}/{p}",
                      base + p, Path(base).joinpath(p)). The author evidently
                      INTENDED confinement to `base`, and the attacker escapes it.
                      Classic path traversal.
  DIRECT_CALLER_PATH  the parameter IS the whole path (open(path), Path(path)
                      .read_text(), load_config(path)). Opening it is the
                      function's contract; it is a vulnerability only if an
                      untrusted caller controls the argument -- i.e. conditional
                      on the trust boundary, which is what Axis-B reachability
                      decides, not the oracle.
  UNKNOWN             source unparsable or no parameter reaches a path expression.

The frozen CONFIRMED label is unchanged; this only says WHICH kind of confirmed
path finding a row is, so results can be reported stratified.

Heuristic limits (state them when citing): intraprocedural, name-based taint over
simple assignments; a base composed in a helper/caller is invisible and lands in
DIRECT_CALLER_PATH. Spot-check before quoting.

  python3 analysis/cwe22_subtype.py --census corpus_census_reproduced.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

BASE_JOIN = "BASE_JOIN_OVERRIDE"
DIRECT = "DIRECT_CALLER_PATH"
UNKNOWN = "UNKNOWN"

_JOIN_FUNCS = {"join"}            # os.path.join / posixpath.join / ntpath.join
_JOIN_METHODS = {"joinpath"}      # Path(...).joinpath(p)


def _names(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _find_func(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    for n in ast.walk(tree):   # fall back to the first function in the snippet
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return n
    return None


def _params(fn) -> set:
    a = fn.args
    ps = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    if a.vararg:
        ps.append(a.vararg)
    if a.kwarg:
        ps.append(a.kwarg)
    return {p.arg for p in ps} - {"self", "cls"}


def _tainted_names(fn, params: set) -> set:
    """Fixed point over simple assignments: a target is tainted if its RHS
    mentions a tainted name."""
    tainted = set(params)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                value = n.value
                if value is None or not (_names(value) & tainted):
                    continue
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for nm in _names(t):
                        if nm not in tainted:
                            tainted.add(nm); changed = True
            elif isinstance(n, (ast.For, ast.comprehension)) and _names(n.iter) & tainted:
                for nm in _names(n.target):
                    if nm not in tainted:
                        tainted.add(nm); changed = True
            elif isinstance(n, ast.withitem) and n.optional_vars is not None \
                    and _names(n.context_expr) & tainted:
                for nm in _names(n.optional_vars):
                    if nm not in tainted:
                        tainted.add(nm); changed = True
    return tainted


def _is_tainted(node, tainted) -> bool:
    return bool(_names(node) & tainted)


def _composes_base(node, tainted) -> bool:
    """True if `node` is a path composition mixing an untainted base with a
    tainted component."""
    if isinstance(node, ast.Call):
        f = node.func
        fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if fname in _JOIN_FUNCS and len(node.args) >= 2:
            # os.path.join(base, ..., p): the FIRST arg is the base
            return (not _is_tainted(node.args[0], tainted)
                    and any(_is_tainted(a, tainted) for a in node.args[1:]))
        if fname in _JOIN_METHODS and isinstance(f, ast.Attribute):
            return (not _is_tainted(f.value, tainted)
                    and any(_is_tainted(a, tainted) for a in node.args))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        # base / p   (pathlib)
        return not _is_tainted(node.left, tainted) and _is_tainted(node.right, tainted)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # base + "/" + p  -- counted only when the left side ENDS in a separator
        # literal; "Error: " + p, HEADER + "\n" + row, tuple + tuple are not paths.
        return (_ends_with_sep(node.left) and not _is_tainted(node.left, tainted)
                and _is_tainted(node.right, tainted))
    if isinstance(node, ast.JoinedStr) and node.values:
        # f"{BASE}/{p}" or f"/data/{p}": an untainted part ENDING IN A SEPARATOR
        # immediately before the first tainted value. f"file {p} not found" is
        # message text, not a path, and does not count.
        prev_untainted_sep = False
        for v in node.values:
            if isinstance(v, ast.FormattedValue) and _is_tainted(v.value, tainted):
                return prev_untainted_sep
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                prev_untainted_sep = v.value.endswith(("/", "\\"))
            elif isinstance(v, ast.FormattedValue):
                prev_untainted_sep = False
        return False
    return False


def _ends_with_sep(node) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.endswith(("/", "\\"))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _ends_with_sep(node.right)
    return False


def classify_source(source: str, func_name: str | None = None) -> dict:
    try:
        tree = ast.parse(textwrap.dedent(source))   # methods are mined indented
    except SyntaxError:
        return {"subtype": UNKNOWN, "why": "unparsable"}
    fn = _find_func(tree, func_name)
    if fn is None:
        return {"subtype": UNKNOWN, "why": "no function"}
    params = _params(fn)
    if not params:
        return {"subtype": UNKNOWN, "why": "no parameters"}
    tainted = _tainted_names(fn, params)
    for n in ast.walk(fn):
        if _composes_base(n, tainted):
            return {"subtype": BASE_JOIN, "why": ast.unparse(n)[:120]}
    return {"subtype": DIRECT, "why": "no base composition with a parameter"}


def classify_record(rec: dict) -> dict:
    src = rec.get("ground_truth_source") or rec.get("module_source") or ""
    return classify_source(src, rec.get("function"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--out", type=Path, default=None, help="per-row JSON (file -> subtype, why)")
    args = ap.parse_args()
    rows = json.loads(args.census.read_text())["rows"]
    conf22 = [r for r in rows if r.get("confirmed") and (r.get("confirmed_cwe") or r.get("exec_cwe")) == "CWE-22"]
    out, tally = {}, Counter()
    for r in conf22:
        res = classify_record(json.loads((args.root / r["file"]).read_text()))
        out[r["file"]] = res
        tally[res["subtype"]] += 1
    print(f"CWE-22 confirmed candidates: {len(conf22)}")
    for k, v in tally.most_common():
        print(f"  {k:22} {v}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

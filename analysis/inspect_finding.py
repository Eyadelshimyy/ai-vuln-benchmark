#!/usr/bin/env python3
"""Inspect ONE VULCAN finding end to end: print the candidate source, the
entry-point call path, the source of every function on that path, and the
Tier-2 exploit proof -- everything needed to reproduce it by hand -- and
optionally re-run the deterministic exploit bank with diagnostics (to explain
a NOT_RECONFIRMED).

USAGE
  # full manual-reproduction dump for one finding
  python3 analysis/inspect_finding.py --repo tracebi --function _fetch

  # diagnose why a finding did not re-confirm
  python3 analysis/inspect_finding.py --repo NLWeb --function get_page --reconfirm
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "analysis", _REPO / "analysis" / "iast"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tier2.engine import attempt_exploit                      # noqa: E402
from auto_harness import _parse_params, vendor_sys_path_roots  # noqa: E402
from reachability import reaches                               # noqa: E402


def _find_candidate(root: Path, repo_sub: str, func: str):
    for d in sorted(root.iterdir()):
        if not d.is_dir() or repo_sub.lower() not in d.name.lower():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name == "auto_harness_results.json":
                continue
            try:
                j = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if j.get("function") == func:
                return d, p, j
    return None, None, None


def _func_source(repo_src: Path, name: str):
    """Return (rel_path, source_text) for the first `def name` found under repo_src."""
    for py in repo_src.rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
                seg = ast.get_source_segment(src, n)
                if seg:
                    return str(py.relative_to(repo_src)), seg
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--repo", required=True, help="substring of the mined repo dir (e.g. tracebi)")
    ap.add_argument("--function", required=True, help="the sink function name")
    ap.add_argument("--reconfirm", action="store_true", help="re-run the Tier-2 bank with diagnostics")
    ap.add_argument("--cwe", default=None, help="try this CWE first (e.g. CWE-918)")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    d, p, j = _find_candidate(args.root, args.repo, args.function)
    if j is None:
        print(f"no candidate found for repo~={args.repo!r} function={args.function!r}")
        return
    repo_src = d / "_repo_src"
    src = j.get("ground_truth_source") or ""

    print("=" * 72)
    print(f"FINDING: {j.get('repo')}  ::  {j.get('file')}  ::  {args.function}()")
    print("=" * 72)
    print("\n--- SINK FUNCTION (the confirmed-vulnerable code) ---")
    print(src)

    # call path from an entry point
    reachable, path, kind = (False, [], None)
    if repo_src.exists():
        try:
            reachable, path, kind = reaches(repo_src, args.function)
        except Exception as e:  # noqa: BLE001
            print(f"(reachability error: {e})")
    print("\n--- REACHABILITY ---")
    if reachable:
        print(f"entry kind: {kind}")
        print("call path : " + " -> ".join(path))
        print("\n--- SOURCE OF EACH FUNCTION ON THE PATH ---")
        for name in path:
            rel, fsrc = _func_source(repo_src, name)
            print(f"\n### {name}   [{rel}]")
            print(fsrc or "(source not found)")
    else:
        print("NOT reachable from any detected entry point.")

    # re-prove with the deterministic bank (optionally with diagnostics)
    print("\n--- TIER-2 RE-PROVE (deterministic bank, no LLM) ---")
    params = []
    try:
        parsed, _ = _parse_params(src)
        params = [q["name"] for q in parsed if q["name"] not in ("self", "cls")]
    except Exception:  # noqa: BLE001
        pass
    # Try every CWE the candidate was mined for (not just the first): a function
    # can be e.g. both an SSRF (918) and a path sink (22), and candidate_cwes[0]
    # may not be the one that confirms.
    cwes = list(dict.fromkeys(j.get("candidate_cwes") or []))
    if args.cwe:
        cwes = [args.cwe] + [c for c in cwes if c != args.cwe]
    if not cwes:
        cwes = [None]
    rec = {"repo": j.get("repo"), "file": j.get("file"), "function": args.function,
           "line": j.get("line", 1), "ground_truth_source": src,
           "module_source": j.get("module_source", "")}
    injected = vendor_sys_path_roots(repo_src) if repo_src.exists() else []
    for r in injected:
        if r not in sys.path:
            sys.path.append(r)
    try:
        any_conf = False
        for cwe in cwes:
            for pn in params:
                res = attempt_exploit(rec, pn, cwe, use="ground_truth", max_agent=0, timeout=args.timeout)
                tag = "CONFIRMED" if res.get("confirmed") else f"no ({res.get('diag_bucket')})"
                print(f"  cwe={str(cwe):8} param={pn:20} -> {tag}")
                if res.get("confirmed"):
                    any_conf = True
                    print(f"      sink : {res.get('sink')}")
                    print(f"      payload: {str(res.get('payload'))[:160]}")
                    print(f"      proof: {res.get('proof')}")
                    break
            if any_conf:
                break
        if not any_conf and args.reconfirm:
            print("  (did not re-confirm -- the diag bucket per param above is the honest reason)")
    finally:
        for r in injected:
            if r in sys.path:
                sys.path.remove(r)


if __name__ == "__main__":
    main()

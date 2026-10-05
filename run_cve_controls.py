#!/usr/bin/env python3
"""
VULCAN RQ4 -- instrument validation on human-authored CVE controls.

Runs curated, pre-registered CVE pre/post-patch pairs through the SAME harness
as everything else and asks: can VULCAN distinguish a known-vulnerable
human-authored implementation from its patched counterpart?

Two tiers do the distinguishing, and WHICH one depends on how the patch works
(this is the point -- it shows why both tiers exist):

  * Parameterization fixes (typically SQL injection): the vulnerable code
    splices data into the query string so the taint marker reaches the sink
    (Tier-1 TRIGGERED); the patch binds it as a parameter so the marker leaves
    the SQL argument (Tier-1 NOT_TRIGGERED). TIER 1 distinguishes.
  * Validation / sanitisation fixes (path traversal, command injection): the
    patch rejects malicious input but still calls the sink with benign input,
    so a benign taint marker reaches the sink in BOTH versions (Tier-1 can't
    tell them apart). A real attack payload fires in the vulnerable version and
    is rejected by the patch. TIER 2 distinguishes.

A CVE control is CONFIRMED-DISTINGUISHED if EITHER tier separates vulnerable
from patched. In-scope misses are reported, never dropped (see the RQ4 protocol
in docs/rq4_control_set.tex).

INPUT: a directory (or file) of CVE control records -- schema in
cve_controls/README.md. USAGE:
  python3 run_cve_controls.py --controls cve_controls/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))

from auto_harness import run_candidate            # noqa: E402
from run_benchmark import reduce_verdicts          # noqa: E402
from tier2.engine import attempt_exploit           # noqa: E402
from tier2.strategies import STRATEGIES            # noqa: E402

_REQUIRED = ("cve_id", "cwe", "function", "module_source", "vulnerable_source")


def _base(rec: dict) -> dict:
    return {"repo": rec.get("library", "cve"), "file": rec.get("file", "cve.py"),
            "function": rec["function"], "line": rec.get("line", 0),
            "module_source": rec["module_source"],
            "construction_recipe": rec.get("construction_recipe")}


def _tier1(rec: dict, source: str) -> str:
    # Splice the vulnerable/patched body into module_source (the "generated"
    # path replaces the stub function at `line`), so the two versions actually
    # differ at execution -- setting ground_truth_source alone would not, since
    # ground_truth execution runs module_source as-is.
    r = _base(rec); r["generated_source"] = source
    return reduce_verdicts(run_candidate(r, use="generated"))["verdict"]


def _tier2(rec: dict, source: str) -> bool:
    param, cwe = rec.get("param"), rec.get("cwe")
    if not param or cwe not in STRATEGIES:
        return False
    r = _base(rec); r["generated_source"] = source
    return bool(attempt_exploit(r, param, cwe, use="generated").get("confirmed"))


def evaluate(rec: dict) -> dict:
    vuln, patched = rec["vulnerable_source"], rec.get("patched_source")
    cwe = rec.get("cwe")
    out = {"cve_id": rec["cve_id"], "cwe": cwe, "function": rec["function"],
           "in_scope": bool(rec.get("in_scope", True)), "scope_reason": rec.get("scope_reason", "")}

    out["t1_vuln"] = _tier1(rec, vuln)
    out["t1_patched"] = _tier1(rec, patched) if patched else None
    has_t2 = bool(rec.get("param")) and cwe in STRATEGIES
    out["t2_vuln"] = _tier2(rec, vuln) if has_t2 else None
    out["t2_patched"] = _tier2(rec, patched) if (patched and has_t2) else None

    out["tier1_detects_vuln"] = (out["t1_vuln"] == "TRIGGERED")
    out["tier1_distinguishes"] = (out["t1_vuln"] == "TRIGGERED"
                                  and out["t1_patched"] is not None
                                  and out["t1_patched"] != "TRIGGERED")
    out["tier2_distinguishes"] = (out["t2_vuln"] is True
                                  and out["t2_patched"] is False)
    out["distinguished"] = out["tier1_distinguishes"] or out["tier2_distinguishes"]
    out["by_tier"] = ("tier1" if out["tier1_distinguishes"]
                      else "tier2" if out["tier2_distinguishes"] else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", required=True, type=Path, help="dir or single CVE control JSON")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = ([args.controls] if args.controls.is_file()
             else sorted(args.controls.glob("*.json")))
    recs = []
    for p in files:
        r = json.loads(p.read_text())
        miss = [k for k in _REQUIRED if k not in r]
        if miss:
            print(f"  skip {p.name}: missing {miss}", file=sys.stderr)
            continue
        recs.append(r)
    if not recs:
        sys.exit(f"no valid CVE control records in {args.controls}")

    results = [evaluate(r) for r in recs]

    in_scope = [r for r in results if r["in_scope"]]
    oos = [r for r in results if not r["in_scope"]]

    print("\n" + "=" * 72)
    print("VULCAN RQ4 -- instrument validation on human-authored CVE controls")
    print("=" * 72)
    for r in results:
        tag = "in-scope" if r["in_scope"] else "OUT-OF-SCOPE"
        print(f"\n[{r['cve_id']}] {r['cwe']}  {r['function']}  ({tag})")
        print(f"   Tier-1: vuln={r['t1_vuln']}  patched={r['t1_patched']}")
        print(f"   Tier-2: vuln={r['t2_vuln']}  patched={r['t2_patched']}")
        if r["in_scope"]:
            if r["distinguished"]:
                print(f"   -> DISTINGUISHED by {r['by_tier']}  (vulnerable confirmed, patch cleared)")
            elif r["tier1_detects_vuln"]:
                print(f"   -> PARTIAL: detected the vuln but did NOT clear the patch -- INVESTIGATE")
            else:
                print(f"   -> MISS: in-scope vulnerability NOT detected -- INVESTIGATE (do not drop)")
        else:
            ok = not r["tier1_detects_vuln"] and not (r["t2_vuln"] is True)
            print(f"   -> {'correct non-detection (no over-reach)' if ok else 'OVER-REACH: fired on an out-of-scope case -- INVESTIGATE'}")

    n_detect = sum(1 for r in in_scope if r["tier1_detects_vuln"])
    n_dist = sum(1 for r in in_scope if r["distinguished"])
    n_t1 = sum(1 for r in in_scope if r["tier1_distinguishes"])
    n_t2 = sum(1 for r in in_scope if r["tier2_distinguishes"])
    oos_ok = sum(1 for r in oos if not r["tier1_detects_vuln"] and not (r["t2_vuln"] is True))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  in-scope controls                         : {len(in_scope)}")
    print(f"  vulnerable detected (Tier-1 recall)       : {n_detect}/{len(in_scope)}")
    print(f"  vulnerable/patched DISTINGUISHED          : {n_dist}/{len(in_scope)}")
    print(f"      by Tier-1 (parameterization fixes)    : {n_t1}")
    print(f"      by Tier-2 (validation/sanitisation)   : {n_t2}")
    if oos:
        print(f"  out-of-scope correct non-detection        : {oos_ok}/{len(oos)}")
    print("=" * 72)
    misses = [r for r in in_scope if not r["distinguished"]]
    if misses:
        print("ACTION: the following in-scope controls were not distinguished -- investigate")
        print("(construction/INFRA gap, unmodelled sink, or a fix VULCAN's model can't see),")
        print("and report honestly; never drop them to inflate recall:")
        for r in misses:
            print(f"   - {r['cve_id']} ({r['cwe']} {r['function']})")

    if args.out:
        out_path = args.out
    elif args.controls.is_file():
        out_path = args.controls.with_suffix(".rq4.json")
    else:
        out_path = args.controls / "rq4_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

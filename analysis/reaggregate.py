#!/usr/bin/env python3
"""Re-aggregate a saved detection run under different consortium rules and per
CWE -- OFFLINE, with NO model calls. Reads the per-member verdicts already in
detection_results.json and recomputes the confusion matrices, so you can test
"how should two models be combined" without re-running the LLMs.

Why this matters: the majority-of-2 consortium collapsed to recall ~0 because a
conservative member (deepseek) vetoes every "vulnerable" call. This script shows
what the OTHER rules do with the exact same verdicts:
  majority        -> >half the members say vulnerable (for 2 members = BOTH = AND)
  any             -> ANY member says vulnerable (OR) -- recovers a conservative
                     member's misses, at the cost of that member's false positives
  unanimous_safe  -> safe only if ALL members agree safe; else vulnerable
                     (identical to `any` for the vulnerable class)

USAGE
  python3 analysis/reaggregate.py detection_corpus.json
  python3 analysis/reaggregate.py detection_corpus.json --by-cwe
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _consortium(member_verdicts: list[str], rule: str) -> str:
    """member_verdicts: each 'vulnerable'|'safe'|None. Returns the panel verdict."""
    votes = [v for v in member_verdicts if v in ("vulnerable", "safe")]
    if not votes:
        return "safe"
    vuln = sum(1 for v in votes if v == "vulnerable")
    if rule == "any":
        return "vulnerable" if vuln >= 1 else "safe"
    if rule == "unanimous_safe":
        return "safe" if vuln == 0 else "vulnerable"
    # majority
    return "vulnerable" if vuln > len(votes) / 2 else "safe"


def _matrix(pairs):
    tp = sum(1 for t, p in pairs if t == "VULN" and p == "vulnerable")
    fn = sum(1 for t, p in pairs if t == "VULN" and p == "safe")
    fp = sum(1 for t, p in pairs if t == "SAFE" and p == "vulnerable")
    tn = sum(1 for t, p in pairs if t == "SAFE" and p == "safe")
    n = tp + fn + fp + tn

    def r(a, b):
        return round(a / b, 3) if b else None
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, n=n,
                precision=r(tp, tp + fp), recall=r(tp, tp + fn),
                FNR=r(fn, tp + fn), FPR=r(fp, fp + tn), accuracy=r(tp + tn, n))


def _show(label, m):
    print(f"\n[{label}]")
    print(f"  TP {m['TP']}  FN {m['FN']}  FP {m['FP']}  TN {m['TN']}  (n={m['n']})")
    print(f"  precision {m['precision']}  recall {m['recall']}  "
          f"FNR {m['FNR']}  FPR {m['FPR']}  accuracy {m['accuracy']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="a detection_results.json from run_detection.py (panel run)")
    ap.add_argument("--by-cwe", action="store_true", help="also break every rule down per execution CWE")
    args = ap.parse_args()

    rep = json.loads(args.report.read_text())
    recs = rep.get("records", [])
    detectors = rep.get("detectors", [])
    scored = [r for r in recs if r.get("exec_truth") in ("VULN", "SAFE") and "members" in r]
    if not scored:
        raise SystemExit("no panel records with per-member verdicts found -- was this a multi-detector run?")

    print("=" * 70)
    print(f"RE-AGGREGATION of {args.report.name}  (no model calls)")
    print(f"detectors: {detectors}")
    print(f"scored samples: {len(scored)}  "
          f"({sum(1 for r in scored if r['exec_truth']=='VULN')} vuln, "
          f"{sum(1 for r in scored if r['exec_truth']=='SAFE')} safe)")
    print("=" * 70)

    # per-member (unchanged, for reference)
    for name in detectors:
        pairs = [(r["exec_truth"], r["members"].get(name, {}).get("verdict")) for r in scored]
        pairs = [(t, p) for t, p in pairs if p in ("vulnerable", "safe")]
        _show(f"member: {name}", _matrix(pairs))

    # consortium under each rule
    for rule in ("majority", "any", "unanimous_safe"):
        pairs = [(r["exec_truth"],
                  _consortium([r["members"].get(n, {}).get("verdict") for n in detectors], rule))
                 for r in scored]
        _show(f"CONSORTIUM ({rule})", _matrix(pairs))

    if args.by_cwe:
        print("\n" + "=" * 70 + "\nPER-CWE   recall = by TRUE class (exec_cwe) | "
              "precision = by CLAIMED class\n" + "=" * 70)
        print("NOTE: recall buckets a vuln by the CWE execution actually fired; precision buckets a\n"
              "FLAG by the CWE the detector claimed, so a false alarm counts against the class it was\n"
              "claimed under. This is the honest per-class precision -- NOT the 1.0 artifact you get\n"
              "when SAFE functions (which carry no exec_cwe) are dropped from the bucket.")

        def _claimed(r, name):
            mm = r["members"].get(name, {})
            return mm.get("cwe") if mm.get("verdict") == "vulnerable" else None

        classes = sorted({r.get("exec_cwe") for r in scored
                          if r["exec_truth"] == "VULN" and r.get("exec_cwe")})
        for X in classes:
            true_X = [r for r in scored if r["exec_truth"] == "VULN" and r.get("exec_cwe") == X]
            nX = len(true_X)
            print(f"\n{X}  ({nX} execution-confirmed vuln of this class)")
            for name in detectors:
                tp_r = sum(1 for r in true_X if r["members"].get(name, {}).get("verdict") == "vulnerable")
                recall = round(tp_r / nX, 3) if nX else None
                # precision by CLAIMED class: among flags the model labelled X, how many truly vuln
                flagged_X = [r for r in scored if _claimed(r, name) == X]
                tp_p = sum(1 for r in flagged_X if r["exec_truth"] == "VULN")
                prec = round(tp_p / len(flagged_X), 3) if flagged_X else None
                print(f"   {name:24} recall {recall} ({tp_r}/{nX})   "
                      f"precision(claimed {X}) {prec} ({tp_p}/{len(flagged_X)})")
            # consortium(any) recall by true class
            tp_c = sum(1 for r in true_X
                       if _consortium([r["members"].get(n, {}).get("verdict") for n in detectors], "any") == "vulnerable")
            print(f"   {'CONSORTIUM(any)':24} recall {round(tp_c/nX,3) if nX else None} ({tp_c}/{nX})")
        print("\nCorpus-level precision (all flags, all classes -- the honest headline precision):")
        for name in detectors:
            pairs = [(r["exec_truth"], r["members"].get(name, {}).get("verdict")) for r in scored]
            m = _matrix([(t, p) for t, p in pairs if p in ("vulnerable", "safe")])
            print(f"   {name:24} precision {m['precision']}  recall {m['recall']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fairest-possible semgrep baseline: the UNION across rulesets.

Different semgrep rulesets catch different things (p/security-audit nails SSRF,
p/owasp-top-ten caught a SQL case security-audit missed). A semgrep user is free
to run several. So the most GENEROUS, honest number for semgrep is: a function
counts as "flagged" if ANY of the provided runs flagged it. This computes that
ceiling and the per-CWE recall, so the comparison can never be accused of
lowballing the baseline.

  python3 analysis/union_semgrep.py --census corpus_census.json \
      semgrep_fullrepo.json semgrep_owasp.json semgrep_default.json
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("runs", nargs="+", help="semgrep_*.json prediction files to union")
    ap.add_argument("--out", default="semgrep_union.json")
    args = ap.parse_args()

    # OR together 'flagged' across runs, keyed by candidate file
    flagged = defaultdict(bool)
    configs = []
    for r in args.runs:
        d = json.loads(Path(r).read_text())
        configs.append(d.get("config", r))
        for p in d.get("predictions", []):
            if p.get("flagged"):
                flagged[p["file"]] = True

    cen = json.loads(args.census.read_text())
    rows = cen["rows"] if isinstance(cen, dict) else cen
    TP = FP = FN = TN = 0
    per_cwe = defaultdict(lambda: [0, 0])
    for r in rows:
        f = r["file"]
        is_vuln = bool(r.get("triggered"))
        fl = flagged.get(f, False)
        if is_vuln:
            for c in (r.get("cwes") or []):
                per_cwe[c][1] += 1
                if fl:
                    per_cwe[c][0] += 1
        TP += is_vuln and fl
        FN += is_vuln and not fl
        FP += (not is_vuln) and fl
        TN += (not is_vuln) and not fl

    n = TP + FP + FN + TN
    prec = TP / (TP + FP) if (TP + FP) else None
    rec = TP / (TP + FN) if (TP + FN) else None
    print("=" * 66)
    print("  SEMGREP UNION BASELINE (most generous: any ruleset catches it)")
    print("=" * 66)
    print(f"  configs unioned: {configs}")
    print(f"  TP {TP}  FN {FN}  FP {FP}  TN {TN}")
    print(f"  recall {rec if rec is None else round(rec,3)}  "
          f"precision {prec if prec is None else round(prec,3)}")
    for c in sorted(per_cwe):
        num, den = per_cwe[c]
        print(f"    {c}: recall {round(num/den,3) if den else 0.0} ({num}/{den})")
    Path(args.out).write_text(json.dumps(
        {"tool": "semgrep-union", "configs": configs,
         "TP": TP, "FP": FP, "FN": FN, "TN": TN,
         "recall": rec, "precision": prec,
         "per_cwe": {c: {"num": v[0], "den": v[1]} for c, v in per_cwe.items()}}, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VULCAN disposition audit: one row per candidate function, every exclusion
named, the corpus funnel reconciled -- kept SEPARATE from detector evaluation.

Addresses the reviewers' central methodological ask: the funnel is NOT an
ablation. Each candidate gets exactly one disposition, so the reader can trace
1,216 -> executable -> triggered -> confirmed -> reachable without conflating
"excluded/inconclusive" with "proven safe".

Disposition (mutually exclusive, execution-adjudicated):
  non-executable        ran into an import/setup/parse error (INCONCLUSIVE, not safe)
  executed-no-trigger   ran, static taint never reached a sink   (defensible SAFE-ish negative)
  triggered-unconfirmed static taint reached a sink but the exploit bank did NOT
                        fire it (INCONCLUSIVE: candidate, not ground truth)
  confirmed             the exploit bank fired the sink with adversarial input
                        (execution-confirmed; the defensible POSITIVE label)

Reachability (joined from gold_set.json) is reported ONLY for confirmed findings.
Provenance (ai_attributed / ai_line_fraction / tool / commit) is read from the
`provenance` block enrich_provenance wrote into each mined JSON.

  python3 analysis/disposition.py --census corpus_census.json \
      --root calibration/mined_prompts --gold gold_set.json --out disposition.csv
"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path


def _err_bucket(err: str) -> str:
    e = (err or "").lower()
    if "modulenotfound" in e or "no module" in e or "import" in e:
        return "non-executable:missing-dependency"
    if "parse" in e or "syntax" in e:
        return "non-executable:parse-error"
    if "timeout" in e:
        return "non-executable:timeout"
    return "non-executable:other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--gold", type=Path, default=None, help="gold_set.json for reachability join")
    ap.add_argument("--out", type=Path, default=Path("disposition.csv"))
    args = ap.parse_args()

    cen = json.loads(args.census.read_text())
    rows = cen["rows"] if isinstance(cen, dict) else cen

    # reachability join: census_file -> (entry_kind, call_path)
    reach = {}
    if args.gold and args.gold.exists():
        for g in json.loads(args.gold.read_text()):
            k = g.get("census_file")
            if k and (k not in reach or g.get("prio", 9) < reach[k][2]):
                reach[k] = (g.get("entry_kind"), g.get("call_path"), g.get("prio", 9))

    out_rows = []
    disp = Counter()
    frac_bucket = Counter()
    tool_tally = Counter()
    for r in rows:
        f = r["file"]
        # disposition (mutually exclusive)
        if not r.get("executable"):
            d = _err_bucket(r.get("error"))
        elif not r.get("triggered"):
            d = "executed-no-trigger"
        elif not r.get("confirmed"):
            d = "triggered-unconfirmed"
        else:
            d = "confirmed"
        disp[d.split(":")[0]] += 1

        # provenance from the mined JSON's written-back block
        prov = {}
        func = None
        try:
            cj = json.loads((args.root / f).read_text())
            func = cj.get("function")
            prov = cj.get("provenance") or {}
        except (OSError, json.JSONDecodeError):
            pass
        aif = prov.get("ai_line_fraction")
        ai_attr = prov.get("ai_attributed")
        if ai_attr is True or (isinstance(aif, (int, float)) and aif >= 0.8):
            pcat = "strong"
        elif isinstance(aif, (int, float)) and aif > 0:
            pcat = "mixed"
        else:
            pcat = "uncertain/none"
        frac_bucket[pcat] += 1
        for t in (prov.get("ai_tools") or {}):
            if d == "confirmed":
                tool_tally[t] += 1

        rk = reach.get(f) if d == "confirmed" else None
        out_rows.append({
            "file": f, "function": func or "", "cwes": "|".join(r.get("cwes") or []),
            "exec_cwe": r.get("exec_cwe") or "", "verdict": r.get("verdict") or "",
            "disposition": d,
            "provenance_cat": pcat,
            "ai_line_fraction": aif if aif is not None else "",
            "ai_tool": "|".join((prov.get("ai_tools") or {}).keys()),
            "ai_commit": prov.get("ai_commit") or "",
            "source_sha256": prov.get("source_sha256") or "",
            "reachable": bool(rk),
            "entry_kind": (rk[0] if rk else ""),
            "call_path": (" -> ".join(rk[1]) if rk and rk[1] else ""),
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # ---- reconciled funnel ----
    total = len(rows)
    executable = sum(1 for r in rows if r.get("executable"))
    triggered = sum(1 for r in rows if r.get("triggered"))
    confirmed = sum(1 for r in rows if r.get("confirmed"))
    conf_reach = sum(1 for r in rows if r.get("confirmed") and r["file"] in reach)
    conf_http = sum(1 for r in rows if r.get("confirmed") and reach.get(r["file"], (None,))[0] == "http")

    print("=" * 68)
    print("  CORPUS DISPOSITION (every candidate accounted for, once)")
    print("=" * 68)
    print(f"  candidates .................... {total}")
    for k in ("non-executable", "executed-no-trigger", "triggered-unconfirmed", "confirmed"):
        print(f"    {k:24} {disp.get(k,0)}")
    print(f"  --- of which executable ....... {executable}  ({executable/total:.1%})")
    print(f"  --- triggered (Tier-1 cand.) .. {triggered}")
    print(f"  --- confirmed (Tier-2 GT) ..... {confirmed}")
    print(f"        reachable (any entry) ... {conf_reach}")
    print(f"        reachable via HTTP ...... {conf_http}")
    print()
    print("  Interpretation (reviewers' point): 'triggered-unconfirmed' and")
    print("  'non-executable' are INCONCLUSIVE, not proven safe. Only")
    print("  'confirmed' (positives) and 'executed-no-trigger' (negatives) are")
    print("  the defensible label set for detector metrics.")
    print()
    print("  Provenance categories (all candidates):")
    for k in ("strong", "mixed", "uncertain/none"):
        print(f"    {k:16} {frac_bucket.get(k,0)}")
    print(f"  Tools among CONFIRMED: {dict(tool_tally)}")
    print(f"\n  Saved per-finding table -> {args.out}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

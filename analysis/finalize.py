#!/usr/bin/env python3
"""VULCAN final funnel + gold-set de-dup -- the exact numbers for the write-up.

The gold set (triage_gold.py) can list the same confirmed finding more than once
when one census file joins to several worklist rows (e.g. the same function
vendored under two paths). This collapses to one row per confirmed finding
(keyed by its census file), keeps its BEST interface reachability (http > cli >
api > main), and prints the end-to-end funnel:

    candidates -> Tier-2 confirmed -> reachable -> HTTP-reachable -> hand-proven

  python3 analysis/finalize.py \
      --census corpus_census.json \
      --gold gold_set.json \
      --out finalize.json
"""
from __future__ import annotations
import argparse, json
from collections import Counter

PRIO = {"http": 0, "cli": 1, "api": 2, "main": 3}

# The findings driven to a live, real-internet exploit BY HAND (webhook-confirmed).
# Matched on (repo substring, function). Everything else in the gold set is a
# confirmed+reachable lead still awaiting the manual trust-boundary step.
HAND_PROVEN = [
    ("sada-test", "fetch_loan_club", "CWE-918 SSRF  -> your webhook (data-file source)"),
    ("mini-pi",   "execute_bash",    "CWE-78 RCE    -> your webhook (agent tool-call / prompt-injection)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", default="corpus_census.json")
    ap.add_argument("--gold", default="gold_set.json")
    ap.add_argument("--out", default="finalize.json")
    args = ap.parse_args()

    cen = json.load(open(args.census))
    rows = cen["rows"] if isinstance(cen, dict) and "rows" in cen else cen
    n_candidates = cen.get("n_candidates", len(rows)) if isinstance(cen, dict) else len(rows)
    confirmed_rows = [r for r in rows if r.get("confirmed")]
    triggered_rows = [r for r in rows if r.get("triggered")]
    executable_rows = [r for r in rows if r.get("executable")]

    gold = json.load(open(args.gold))

    # de-dup: one row per confirmed finding (its census file), keep best reachability
    best: dict = {}
    for g in gold:
        key = g.get("census_file") or (g["repo"], g["function"])
        p = PRIO.get(g.get("entry_kind"), 9)
        if key not in best or p < PRIO.get(best[key].get("entry_kind"), 9):
            best[key] = g
    uniq = list(best.values())

    by_kind = Counter(g["entry_kind"] for g in uniq)
    http = [g for g in uniq if g["entry_kind"] == "http"]

    # hand-proven matches
    proven = []
    for g in uniq:
        for repo_sub, fn, label in HAND_PROVEN:
            if repo_sub in g["repo"] and g["function"] == fn:
                proven.append({**g, "exploit": label})

    funnel = {
        "candidates_mined":          n_candidates,
        "executable":                len(executable_rows),
        "tier1_triggered":           len(triggered_rows),
        "tier2_confirmed":           len(confirmed_rows),
        "confirmed_unique":          len(uniq),          # confirmed AND in worklist (reachable-analysed)
        "reachable_any_entry":       len(uniq),
        "reachable_http":            len(http),
        "hand_proven_e2e":           len(proven),
        "by_entry_kind":             dict(by_kind),
    }
    json.dump({"funnel": funnel, "unique_gold": uniq, "hand_proven": proven},
              open(args.out, "w"), indent=2)

    print("="*66)
    print("  VULCAN FINAL FUNNEL")
    print("="*66)
    print(f"  AI-authored functions mined ............ {funnel['candidates_mined']}")
    print(f"    executable under the oracle .......... {funnel['executable']}")
    print(f"    Tier-1 triggered (taint->sink) ....... {funnel['tier1_triggered']}")
    print(f"    Tier-2 CONFIRMED (exploit bank) ...... {funnel['tier2_confirmed']}")
    print(f"      unique & reachability-analysed ..... {funnel['confirmed_unique']}")
    print(f"    reachable from ANY entry point ....... {funnel['reachable_any_entry']}   {dict(by_kind)}")
    print(f"    reachable from an HTTP interface ..... {funnel['reachable_http']}")
    print(f"    hand-proven end-to-end (webhook) ..... {funnel['hand_proven_e2e']}")
    print()
    print("  HTTP-reachable confirmed findings (deduped):")
    for g in sorted(http, key=lambda g: len(g["call_path"])):
        cwe = g.get("exec_cwe") or ",".join(g["reach_cwes"])
        print(f"    [{cwe:<10}] {g['repo']:28} {g['function']:26} <- {' -> '.join(g['call_path'])}")
    print()
    print("  Hand-proven end-to-end:")
    for p in proven:
        print(f"    * {p['repo']:22} {p['function']:16} {p['exploit']}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

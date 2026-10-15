#!/usr/bin/env python3
"""VULCAN GOLD-SET triage: Tier-2 CONFIRMED  INTERSECT  interface-reachable.

Reachability alone over-reports (a parameterized `db.execute(...)` in an HTTP
handler is "CWE-89 reachable" but NOT exploitable -- execution refuses to confirm
it). The set worth hand-exploiting is the intersection:

    confirmed-by-execution (Tier-2)   AND   reachable from a real entry point

This joins the frozen Tier-2 census (corpus_census.json, produced with --tier2)
against reachable_worklist.json (triage_reachable.py) on (repo, function) and
prints, ranked http > cli > api > main, the findings that are BOTH.

  python3 analysis/triage_gold.py \
      --census corpus_census.json \
      --worklist reachable_worklist.json
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict


def _norm_repo(s: str) -> str:
    """Normalize repo id across the two files: 'a/b' and 'a__b' -> 'a__b'; a bare
    mined-dir name stays as-is. Lowercased for tolerant matching."""
    return s.replace("/", "__").lower()


def parse_census_file(fp: str):
    """census 'file' = '<repo>/<module>.py__<function>.json' -> (repo, function).
    The function keeps its own leading underscores (so '.py___f' -> '_f')."""
    repo = fp.split("/", 1)[0]
    rest = fp.rsplit("/", 1)[-1]
    rest = re.sub(r"\.json$", "", rest)
    if ".py__" in rest:
        func = rest.split(".py__", 1)[1]
    else:
        func = rest.split("__")[-1]
    return _norm_repo(repo), func


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", default="corpus_census.json")
    ap.add_argument("--worklist", default="reachable_worklist.json")
    ap.add_argument("--out", default="gold_set.json")
    args = ap.parse_args()

    cen = json.load(open(args.census))
    rows = cen["rows"] if isinstance(cen, dict) and "rows" in cen else cen
    wl = json.load(open(args.worklist))

    # confirmed findings, keyed (repo, function) -> row
    confirmed = {}
    for r in rows:
        if r.get("confirmed"):
            key = parse_census_file(r["file"])
            confirmed[key] = r
    # also index confirmed by bare function name as a fallback matcher
    confirmed_by_func = defaultdict(list)
    for (repo, func), r in confirmed.items():
        confirmed_by_func[func].append((repo, r))

    print(f"Tier-2 CONFIRMED findings in census : {len(confirmed)}")
    reach = [w for w in wl if w.get("reachable")]
    print(f"interface-reachable in worklist     : {len(reach)}")

    PRIO = {"http": 0, "cli": 1, "api": 2, "main": 3}
    gold = []
    for w in reach:
        key = (_norm_repo(w["repo"]), w["function"])
        row = confirmed.get(key)
        if row is None:
            # fallback: same function name confirmed in the same repo (handles
            # repo-id spelling drift between the two files)
            for repo, r in confirmed_by_func.get(w["function"], []):
                if repo in key[0] or key[0] in repo:
                    row = r
                    break
        if row is None:
            continue
        gold.append({
            "repo": w["repo"], "function": w["function"], "file": w["file"],
            "entry_kind": w["entry_kind"], "call_path": w["call_path"],
            "reach_cwes": w["cwes"],
            "exec_cwe": row.get("exec_cwe"),
            "census_file": row["file"],
            "prio": PRIO.get(w["entry_kind"], 9),
        })

    gold.sort(key=lambda g: (g["prio"], len(g["call_path"])))
    json.dump(gold, open(args.out, "w"), indent=2)

    print(f"\n==================  GOLD SET: {len(gold)}  "
          "(confirmed AND reachable)  ==================")
    by_kind = Counter(g["entry_kind"] for g in gold)
    print("by entry kind:", dict(by_kind))
    print()
    for g in gold:
        path = " -> ".join(g["call_path"])
        print(f"  [{g['entry_kind']:4s}] {g.get('exec_cwe') or ','.join(g['reach_cwes']):10s} "
              f"{g['repo']:30s} {g['function']:28s}  <- {path}")
    print(f"\nSaved {args.out}")
    if not gold:
        print("\nNOTE: empty gold set can mean the repo-id spellings differ between the\n"
              "two files. If you see confirmed findings you expected, tell me the\n"
              "census 'file' value for one and I'll fix the join key.")


if __name__ == "__main__":
    main()

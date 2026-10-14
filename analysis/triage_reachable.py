#!/usr/bin/env python3
"""Scale triage: across a (large) mined corpus, rank every candidate sink by
whether a REAL entry point (HTTP route / CLI / public API) reaches it -- the
interface-reachable worklist. Built for big repos: each repo's call graph is
parsed ONCE and queried for all its candidates (O(repos) parses, not O(candidates)).

Input is the mined corpus directly (no census needed), so it runs before/without
execution -- which is the point: reachability scales to code that won't execute
in the harness. Optionally intersect with a census to rank only TRIGGERED or
CONFIRMED candidates.

Output: a ranked table + JSON. HTTP-reachable first (external attack surface),
then CLI, then public API; INTERNAL ones (no entry reaches them) listed last.

USAGE
  # whole mined corpus
  python3 analysis/triage_reachable.py --root calibration/mined_prompts --out reachable_worklist.json
  # only execution-confirmed candidates
  python3 analysis/triage_reachable.py --root calibration/mined_prompts \
      --census corpus_census_FROZEN.json --only confirmed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from reachability import analyze_repo, reaches_with_graph  # noqa: E402

_PRIO = {"http": 0, "cli": 1, "api": 2, "main": 3, None: 9}


def _census_filter(census: Path, only: str):
    """Return a set of candidate 'file' keys to keep, or None for all."""
    data = json.loads(census.read_text())
    rows = data["rows"] if isinstance(data, dict) else data
    keep = set()
    for r in rows:
        if only == "triggered" and r.get("triggered"):
            keep.add(r["file"])
        elif only == "confirmed" and r.get("confirmed"):
            keep.add(r["file"])
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--census", type=Path, default=None, help="optional census to intersect with")
    ap.add_argument("--only", choices=["triggered", "confirmed"], default=None,
                    help="with --census: restrict to these candidates")
    ap.add_argument("--out", type=Path, default=Path("reachable_worklist.json"))
    args = ap.parse_args()

    keep = _census_filter(args.census, args.only) if (args.census and args.only) else None

    # collect candidates grouped by repo dir
    by_repo = defaultdict(list)
    for repo_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for cj in sorted(repo_dir.glob("*.json")):
            if cj.name == "auto_harness_results.json":
                continue
            rel = f"{repo_dir.name}/{cj.name}"
            if keep is not None and rel not in keep:
                continue
            by_repo[repo_dir].append((rel, cj))

    results = []
    repos = sorted(by_repo)
    for i, repo_dir in enumerate(repos, 1):
        repo_src = repo_dir / "_repo_src"
        print(f"[{i}/{len(repos)}] graphing {repo_dir.name} "
              f"({len(by_repo[repo_dir])} candidates) ...", file=sys.stderr)
        g = None
        if repo_src.exists():
            try:
                g = analyze_repo(repo_src)   # parse ONCE per repo
            except Exception:  # noqa: BLE001
                g = None
        for rel, cj in by_repo[repo_dir]:
            try:
                j = json.loads(cj.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            func = j.get("function")
            reachable, path, kind = reaches_with_graph(g, func) if (func and g is not None) else (False, [], None)
            results.append({
                "repo": repo_dir.name.replace("__", "/", 1),
                "function": func, "file": j.get("file"),
                "cwes": j.get("candidate_cwes") or [],
                "reachable": reachable, "entry_kind": kind,
                "call_path": path,
                "prio": _PRIO.get(kind, 9),
            })

    results.sort(key=lambda r: (r["prio"], r["repo"], r["function"] or ""))
    args.out.write_text(json.dumps(results, indent=2))

    # report
    kinds = Counter(r["entry_kind"] if r["reachable"] else "INTERNAL" for r in results)
    print("\n" + "=" * 74)
    print("INTERFACE-REACHABILITY WORKLIST (ranked: http > cli > api > main)")
    print("=" * 74)
    print(f"{len(results)} candidates across {len(repos)} repos")
    for k in ("http", "cli", "api", "main", "INTERNAL"):
        if kinds.get(k):
            print(f"  {k:9}: {kinds[k]}")
    print("\n--- top interface-reachable (http/cli first) ---")
    shown = 0
    for r in results:
        if not r["reachable"] or r["entry_kind"] not in ("http", "cli"):
            continue
        cwe = ",".join(r["cwes"])[:16]
        print(f"  [{r['entry_kind']}] {cwe:16} {str(r['function'])[:26]:26} {r['repo'][:24]:24} "
              f"{' -> '.join(r['call_path'])}")
        shown += 1
        if shown >= 60:
            print("  ... (see JSON for the rest)")
            break
    if not shown:
        print("  (none http/cli-reachable; see api-reachable entries in the JSON)")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

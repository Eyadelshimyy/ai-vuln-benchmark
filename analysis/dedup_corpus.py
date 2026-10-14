#!/usr/bin/env python3
"""VULCAN corpus de-duplicator -- collapse near-duplicate vulnerability patterns
to representatives so no single copy-pasted template dominates the benchmark.

WHY: an uncapped mine can surface the SAME vulnerability pattern many times when a
repo copy-pastes a helper across variants (e.g. civitas' `ballot_measures_<state>.py`
scrapers -- 57 structurally identical XXE sinks differing only in URLs/strings). Left
as 57 separate ground-truth items, one template would be ~46% of the corpus and a
detector that happens to catch/miss that single pattern would swing the whole metric.

WHAT: two functions are "the same pattern" if their NORMALIZED AST matches -- identical
code structure after canonicalizing local identifier names and erasing string/number
literal *values*, while KEEPING call/attribute names (.fromstring, .execute, os.system)
and the control-flow shape, which is exactly what defines the vulnerability. Functions
are clustered per (repo, exec_cwe, fingerprint); each cluster keeps up to K deterministic
representatives (sorted by path). Distinct vulns -- different structure -- never collapse.

This operates purely on an existing census + the mined corpus on disk. It does NOT
re-mine, re-clone, or re-execute: it only *selects* which already-confirmed items enter
the frozen evaluation set, and prints an auditable report of every collapse.

USAGE
  python3 analysis/dedup_corpus.py \
      --census corpus_census_FULL.json \
      --root calibration/mined_prompts \
      --per-cluster 3 \
      --out corpus_census_FROZEN.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _normalized_fingerprint(src: str) -> str | None:
    """Structure-only hash of a function body. Same vulnerability pattern -> same hash,
    regardless of variable names or literal values. Returns None if it won't parse."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None
    # locate the (first) function def to normalize its body
    func = next((n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    node = func if func is not None else tree

    names: dict[str, str] = {}

    def canon(name: str) -> str:
        # canonicalize LOCAL identifiers to positional tokens so `url_al` and `url_ca`
        # don't look different; but keep it stable within one function.
        if name not in names:
            names[name] = f"v{len(names)}"
        return names[name]

    out: list[str] = []

    def emit(n):
        if isinstance(n, ast.Name):
            out.append("N:" + canon(n.id))
            return
        if isinstance(n, ast.arg):
            out.append("A:" + canon(n.arg))
            return
        if isinstance(n, ast.Attribute):
            # KEEP the attribute name -- `.fromstring`, `.execute`, `.system` ARE the sink
            out.append("ATTR:" + n.attr)
            emit(n.value)
            return
        if isinstance(n, ast.Constant):
            # erase the VALUE, keep the type -- a URL string vs another URL string
            # must not distinguish two copies of the same scraper
            out.append("C:" + type(n.value).__name__)
            return
        out.append(type(n).__name__)
        for child in ast.iter_child_nodes(n):
            emit(child)

    emit(node)
    return hashlib.sha256("|".join(out).encode()).hexdigest()[:16]


def _load_source(root: Path, rel_file: str) -> str | None:
    p = root / rel_file
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("ground_truth_source") or d.get("prompt_source")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True, type=Path, help="input census JSON (e.g. FULL)")
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"),
                    help="mined corpus root (candidate JSONs live here)")
    ap.add_argument("--per-cluster", type=int, default=3,
                    help="max representatives kept per near-duplicate cluster (default 3)")
    ap.add_argument("--out", type=Path, default=Path("corpus_census_FROZEN.json"))
    ap.add_argument("--dedup-safe", action="store_true",
                    help="also dedup executable-but-not-triggered (SAFE) rows (default: "
                         "only dedup TRIGGERED ground-truth; keep all SAFE)")
    ap.add_argument("--report", action="store_true",
                    help="diagnostic only: for each (repo, CWE) show how many TRIGGERED items "
                         "and how many DISTINCT structural fingerprints they have (reveals whether "
                         "a block is real duplication or genuinely distinct code). Writes nothing.")
    ap.add_argument("--per-repo-cap", type=int, default=0,
                    help="if >0, also cap the number of TRIGGERED items kept per repo (balances "
                         "repo over-representation even when functions are structurally distinct). "
                         "Representatives chosen to maximize fingerprint diversity first.")
    args = ap.parse_args()

    data = json.loads(args.census.read_text())
    rows = data["rows"] if isinstance(data, dict) else data

    # ---- diagnostic report: true diversity of each (repo, CWE) triggered block ----
    if args.report:
        from collections import Counter as _C
        blocks = defaultdict(list)
        for r in rows:
            if not r.get("triggered"):
                continue
            repo = r["file"].split("/")[0]
            cwe = r.get("exec_cwe") or "?"
            src = _load_source(args.root, r["file"])
            fp = _normalized_fingerprint(src) if src else None
            blocks[(repo, cwe)].append(fp)
        print(f"{'repo':26}{'CWE':9}{'trig':>5}{'distinct':>9}{'top-cluster':>12}")
        for (repo, cwe), fps in sorted(blocks.items(), key=lambda kv: -len(kv[1])):
            cnt = _C(f for f in fps if f)
            distinct = len(cnt) + sum(1 for f in fps if f is None)
            top = max(cnt.values()) if cnt else 0
            print(f"{repo[:26]:26}{cwe:9}{len(fps):5}{distinct:9}{top:12}")
        print("\ndistinct == trig  -> genuinely distinct code (dedup won't help; use --per-repo-cap)")
        print("distinct <<  trig  -> real duplication (raise exact-match coverage or fuzzy-collapse)")
        return

    # cluster key: (repo-from-path, class, structural fingerprint)
    clusters: dict[tuple, list[dict]] = defaultdict(list)
    passthrough: list[dict] = []
    nofp = 0
    for r in rows:
        rel = r["file"]
        repo = rel.split("/")[0]
        triggered = bool(r.get("triggered"))
        # which rows are subject to dedup
        if not triggered and not args.dedup_safe:
            passthrough.append(r)
            continue
        src = _load_source(args.root, rel)
        fp = _normalized_fingerprint(src) if src else None
        if fp is None:
            nofp += 1
            passthrough.append(r)   # can't fingerprint -> never silently drop it
            continue
        cls = r.get("exec_cwe") or (tuple(r.get("cwes") or []) if r.get("cwes") else "?")
        clusters[(repo, cls, fp)].append(r)

    kept: list[dict] = []
    collapses: list[tuple] = []
    for (repo, cls, fp), members in sorted(clusters.items()):
        members_sorted = sorted(members, key=lambda r: r["file"])
        keep = members_sorted[: args.per_cluster]
        kept.extend(keep)
        if len(members) > args.per_cluster:
            collapses.append((repo, cls, len(members), len(keep), keep[0]["file"]))

    # ---- optional per-repo cap on TRIGGERED items (repo-dominance balance) ----
    repo_capped = []
    if args.per_repo_cap > 0:
        by_repo = defaultdict(list)
        for r in kept:
            if r.get("triggered"):
                by_repo[r["file"].split("/")[0]].append(r)
            else:
                repo_capped.append(r)  # non-triggered kept untouched
        for repo, items in by_repo.items():
            if len(items) <= args.per_repo_cap:
                repo_capped.extend(items)
                continue
            # diversity-first: order by first occurrence of each fingerprint so the
            # cap keeps as many DISTINCT patterns as possible before repeating one
            seen = {}
            for r in items:
                src = _load_source(args.root, r["file"])
                r["_fp"] = _normalized_fingerprint(src) if src else r["file"]
            # round-robin across fingerprints
            groups = defaultdict(list)
            for r in sorted(items, key=lambda r: r["file"]):
                groups[r["_fp"]].append(r)
            order, gl = [], list(groups.values())
            while len(order) < len(items):
                for g in gl:
                    if g:
                        order.append(g.pop(0))
            picked = order[: args.per_repo_cap]
            for r in picked:
                r.pop("_fp", None)
            repo_capped.extend(picked)
        kept = [r for r in repo_capped]
        frozen = passthrough + kept
    else:
        frozen = passthrough + kept

    # recompute per-CWE triggered tallies for the frozen set
    tally = defaultdict(lambda: [0, 0, 0])  # disc, exec, trig  (by exec_cwe for trig)
    for r in frozen:
        for c in (r.get("cwes") or []):
            tally[c][0] += 1
            if r.get("executable"):
                tally[c][1] += 1
        if r.get("triggered") and r.get("exec_cwe"):
            tally[r["exec_cwe"]][2] += 1

    out_obj = {"rows": frozen,
               "dedup": {"per_cluster": args.per_cluster,
                         "clusters_collapsed": len(collapses),
                         "rows_in": len(rows), "rows_out": len(frozen),
                         "unfingerprintable_kept": nofp}}
    args.out.write_text(json.dumps(out_obj, indent=2))

    # ---- report ----
    trig_in = sum(1 for r in rows if r.get("triggered"))
    trig_out = sum(1 for r in frozen if r.get("triggered"))
    print("=" * 68)
    print("VULCAN CORPUS DE-DUPLICATION  (structural AST fingerprint)")
    print("=" * 68)
    print(f"rows: {len(rows)} -> {len(frozen)}   |   TRIGGERED: {trig_in} -> {trig_out}")
    print(f"near-duplicate clusters collapsed: {len(collapses)} "
          f"(kept <= {args.per_cluster} representatives each)")
    if nofp:
        print(f"unfingerprintable (kept as-is, never dropped): {nofp}")
    print()
    if collapses:
        print("collapsed clusters (repo | class | had -> kept | example):")
        for repo, cls, had, kp, ex in sorted(collapses, key=lambda x: -x[2]):
            example = ex.split("/", 1)[-1][:46]
            print(f"  {repo:22} {str(cls):8} {had:3} -> {kp}   {example}")
    print()
    print(f"{'CWE':10}{'Disc':>6}{'Exec':>6}{'Trig':>6}")
    D=E=T=0
    for c in sorted(tally):
        d,e,t=tally[c]; D+=d;E+=e;T+=t
        print(f"{c:10}{d:6}{e:6}{t:6}")
    print("-"*28); print(f"{'TOTAL':10}{D:6}{E:6}{T:6}")
    print()
    print(f"Frozen evaluation corpus -> {args.out}")


if __name__ == "__main__":
    main()

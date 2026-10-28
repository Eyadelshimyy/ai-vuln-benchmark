#!/usr/bin/env python3
"""Freeze the RQ2 detector-evaluation label set from the FROZEN census.

No execution and no model calls. Output is the single input the RQ2 run reads
(detector_consortium.py --labelset), so the population is fixed and hashed
BEFORE any detector answer exists.

Labels (execution-adjudicated, unchanged semantics):
  positive  census `confirmed` (Tier-2 observable fired)
  negative  verdict == NOT_TRIGGERED (executed; the marker never reached a sink
            under the tested conditions -- not a universal safety proof)
  excluded  TRIGGERED-but-unconfirmed, COULD_NOT_EXECUTE (counted, never "safe")

Unit = source-unique function: rows are deduplicated by (repo, sha256(source)),
so a function copied into two modules is one item, not two correlated ones.

Negatives: `ratio` per positive, sampled WITHIN each generating-agent stratum
in proportion to that stratum's positives (seeded), so per-generator detection
metrics have negatives drawn from the same generator. Shortfalls are recorded.

Each item carries strata for pre-registered secondary analyses: generator,
exec_cwe, cwe22_subtype, source length.

  python3 analysis/rq2_labelset.py --census corpus_census_reproduced.json \
      --out validation/rq2/rq2_labelset_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20261026
LABELSET_VERSION = "v3-v1v2-srcunique-twins"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _composed_manual() -> set:
    p = Path(__file__).resolve().parent.parent / "validation" / "rq1" / "cwe22_composed_manual.json"
    return {(c["corpus"], c["file"]) for c in json.loads(p.read_text())["composed_confirmed"]} if p.exists() else set()


def build(census, root=None, ratio: float = 2.0, seed: int = SEED, twins: dict | None = None,
          twins_root: Path | None = None) -> dict:
    """`census` is one census Path (with `root`), or a list of (tag, census, root)."""
    corpora = [("v1", Path(census), Path(root))] if not isinstance(census, list) else census
    composed = _composed_manual()
    rows = [(tag, cr, r) for tag, cp, cr in corpora for r in json.loads(Path(cp).read_text())["rows"]]
    pos_by_key, neg_by_key, excluded = {}, {}, Counter()
    for tag, root, r in sorted(rows, key=lambda t: (t[0], t[2]["file"])):
        if r.get("confirmed"):
            bucket = pos_by_key
        elif r.get("verdict") == "NOT_TRIGGERED":
            bucket = neg_by_key
        else:
            excluded["TRIGGERED_unconfirmed" if r.get("triggered") else str(r.get("verdict"))] += 1
            continue
        rec = json.loads((root / r["file"]).read_text())
        src = rec.get("ground_truth_source") or rec.get("module_source") or ""
        if not src.strip():
            excluded["empty_source"] += 1
            continue
        key = (rec.get("repo"), _sha(src.strip().encode()))
        if key in bucket:
            bucket[key]["duplicate_files"].append(r["file"])
            continue
        tools = sorted((rec.get("ai_provenance") or {}).get("ai_tools") or {})
        cwe = (r.get("confirmed_cwe") or r.get("exec_cwe")) if r.get("confirmed") else None
        bucket[key] = {
            "file": r["file"], "label": bool(r.get("confirmed")), "cwes": r.get("cwes") or [],
            "exec_cwe": cwe, "repo": rec.get("repo"), "function": rec.get("function"),
            "generator": tools[0] if len(tools) == 1 else ("mixed" if tools else "unattributed"),
            "corpus": tag, "root": str(root),
            # hand-reviewed list decides COMPOSED; the heuristic's BASE_JOIN was wrong 3/6 times
            "cwe22_subtype": (("COMPOSED" if (tag, r["file"]) in composed else "DIRECT_CALLER_PATH")
                              if cwe == "CWE-22" else None),
            "src_chars": len(src), "source_sha256": _sha(src.encode()), "duplicate_files": []}
    # a source that is both a positive and a negative (different module context)
    # is ambiguous as a detector item -- drop it from the negatives, record it
    clash = set(pos_by_key) & set(neg_by_key)
    for k in clash:
        neg_by_key.pop(k)
    pos = sorted(pos_by_key.values(), key=lambda it: it["file"])
    neg_pool = defaultdict(list)
    for it in sorted(neg_by_key.values(), key=lambda it: it["file"]):
        neg_pool[it["generator"]].append(it)
    rng = random.Random(seed)
    neg, shortfall = [], {}
    for gen, n_pos in sorted(Counter(it["generator"] for it in pos).items()):
        want = int(round(n_pos * ratio))
        pool = neg_pool.get(gen, [])
        take = rng.sample(pool, min(want, len(pool)))
        if len(take) < want:
            shortfall[gen] = want - len(take)
        neg += sorted(take, key=lambda it: it["file"])
    items = pos + neg
    pairs = []
    if twins:
        by_src = {(it["corpus"], it["repo"], it["source_sha256"]): it for it in pos}
        for v in twins["valid"]:
            rec_o = json.loads((dict((t, Path(cr)) for t, _, cr in corpora)[v["corpus"]] / v["file"]).read_text())
            so = rec_o.get("ground_truth_source") or rec_o.get("module_source") or ""
            orig = by_src.get((v["corpus"], rec_o.get("repo"), _sha(so.encode())))
            rec_f = json.loads((twins_root / v["twin_root_fixed"]).read_text())
            sf = rec_f.get("ground_truth_source") or ""
            if orig is None:
                excluded["twin_original_not_in_positives"] += 1
                continue
            pairs.append({"file": v["twin_root_fixed"], "root": str(twins_root), "label": False,
                          "role": "pair_fixed", "pair_of": orig["file"], "fix_kind": v["fix_kind"],
                          "corpus": v["corpus"], "repo": rec_f.get("repo"), "function": rec_f.get("function"),
                          "generator": orig["generator"], "exec_cwe": None, "cwe22_subtype": None,
                          "cwes": orig["cwes"], "src_chars": len(sf), "source_sha256": _sha(sf.encode()),
                          "duplicate_files": []})
    items = items + sorted(pairs, key=lambda it: it["file"])
    meta = {
        "labelset_version": LABELSET_VERSION, "seed": seed, "neg_per_pos": ratio,
        "corpora": [{"tag": t, "census": str(cp), "census_sha256": _sha(Path(cp).read_bytes()), "root": str(cr)}
                    for t, cp, cr in corpora],
        "n_pairs": len(pairs), "twins_sha256": _sha(json.dumps(twins, sort_keys=True).encode()) if twins else None,
        "n_pos": len(pos), "n_neg": len(neg),
        "population": {"pos_source_unique": len(pos_by_key), "neg_source_unique_pool": len(neg_by_key),
                       "pos_neg_source_clash_dropped": len(clash), "excluded_rows": dict(excluded)},
        "negative_shortfall_by_generator": shortfall,
        # deployment prevalence for projected precision, in the same unit
        "prevalence_corpus": len(pos_by_key) / (len(pos_by_key) + len(neg_by_key)) if pos_by_key else 0.0,
        "strata": {"pos_by_generator": dict(Counter(it["generator"] for it in pos)),
                   "neg_by_generator": dict(Counter(it["generator"] for it in neg)),
                   "pos_by_cwe": dict(Counter(it["exec_cwe"] for it in pos)),
                   "pos_cwe22_subtype": dict(Counter(it["cwe22_subtype"] for it in pos if it["cwe22_subtype"]))},
    }
    if len(corpora) == 1:          # single-corpus label sets keep the v2 keys
        meta["census"], meta["census_sha256"] = str(corpora[0][1]), meta["corpora"][0]["census_sha256"]
    meta["items_sha256"] = _sha(json.dumps([(it["file"], it["label"], it["source_sha256"]) for it in items]).encode())
    return {"meta": meta, "items": items}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", type=Path, default=None, help="single corpus (v2 label set)")
    ap.add_argument("--root", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--corpus", action="append", default=[], metavar="TAG=CENSUS:ROOT",
                    help="repeatable; several corpora in one label set (e.g. v1=...:..., v2=...:...)")
    ap.add_argument("--twins", type=Path, default=None, help="validation/twins/twins_verified.json")
    ap.add_argument("--twins-root", type=Path, default=None, help="dir holding the FIXED twin JSONs")
    ap.add_argument("--ratio", type=float, default=2.0, help="negatives per positive (default 2)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"refusing to overwrite frozen label set {args.out} (delete it deliberately to rebuild)")
    if args.corpus:
        corpora = []
        for spec in args.corpus:
            tag, rest = spec.split("=", 1)
            cp, cr = rest.split(":", 1)
            corpora.append((tag, Path(cp), Path(cr)))
        ls = build(corpora, None, args.ratio, args.seed,
                   twins=json.loads(args.twins.read_text()) if args.twins else None, twins_root=args.twins_root)
    else:
        ls = build(args.census, args.root, args.ratio, args.seed)
    args.out.write_text(json.dumps(ls, indent=2, sort_keys=True))
    print(json.dumps(ls["meta"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

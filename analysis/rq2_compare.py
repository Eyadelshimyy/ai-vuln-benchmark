#!/usr/bin/env python3
"""RQ2 framing effect: compare two detector_consortium.py outputs made on the
SAME frozen label set under different prompts (pre-registered: condition A =
v2-threat, B = v2). Paired, per model, per stratum. No model calls: it reads
the `predictions` blocks of the two output JSONs.

For each model and each stratum cell (all positives, all negatives, and the
pre-registered positive strata), it reports how often A and B flag the same
items, the discordant pairs (A-only, B-only), and an exact McNemar p-value.

  python3 analysis/rq2_compare.py --a validation/rq2/rq2_A_v2threat.json \
      --b validation/rq2/rq2_B_v2.json --labelset validation/rq2/rq2_labelset_v2.json \
      --out validation/rq2/RQ2_FRAMING.md
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cells(items: list) -> dict:
    out = {"positives (all)": [it for it in items if it["label"]],
           "negatives (all)": [it for it in items if not it["label"]]}
    for it in items:
        if not it["label"]:
            continue
        if it.get("cwe22_subtype"):
            out.setdefault(f"CWE-22 {it['cwe22_subtype']}", []).append(it)
        elif it.get("exec_cwe"):
            out.setdefault(it["exec_cwe"], []).append(it)
    return out


def compare(a: dict, b: dict, items: list) -> dict:
    for o, tag in ((a, "A"), (b, "B")):
        if not o.get("labelset") or o["labelset"]["items_sha256"] != items_hash(items):
            raise SystemExit(f"output {tag} was not produced on this label set")
    res = {}
    models = [m for m in a["predictions"] if m in b["predictions"] and not m.startswith("CONSORTIUM")]
    for m in models:
        pa, pb = a["predictions"][m], b["predictions"][m]
        res[m] = {}
        for name, cell in cells(items).items():
            both = [it for it in cell if it["file"] in pa and it["file"] in pb]
            fa = sum(pa[it["file"]] for it in both); fb = sum(pb[it["file"]] for it in both)
            a_only = sum(1 for it in both if pa[it["file"]] and not pb[it["file"]])
            b_only = sum(1 for it in both if pb[it["file"]] and not pa[it["file"]])
            res[m][name] = {"n": len(both), "flagged_A": fa, "flagged_B": fb,
                            "A_only": a_only, "B_only": b_only, "mcnemar_p": mcnemar_exact(a_only, b_only)}
    return res


def items_hash(items: list) -> str:
    import hashlib
    return hashlib.sha256(json.dumps([(it["file"], it["label"], it["source_sha256"]) for it in items]).encode()).hexdigest()


def render(res: dict, a_name: str, b_name: str) -> str:
    L = [f"# RQ2 framing effect: A = `{a_name}` vs B = `{b_name}`", "",
         "Paired on identical items. For positives, flagged = caught (recall numerator); for negatives, "
         "flagged = false positive. McNemar exact p on the discordant pairs.", ""]
    for m, cs in res.items():
        L += [f"## {m}", "", "| Cell | n | flagged A | flagged B | A only | B only | McNemar p |", "|---|---|---|---|---|---|---|"]
        for name, c in cs.items():
            p = "-" if c["mcnemar_p"] is None else f"{c['mcnemar_p']:.3g}"
            L.append(f"| {name} | {c['n']} | {c['flagged_A']} | {c['flagged_B']} | {c['A_only']} | {c['B_only']} | {p} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--labelset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a, b = json.loads(args.a.read_text()), json.loads(args.b.read_text())
    items = json.loads(args.labelset.read_text())["items"]
    res = compare(a, b, items)
    args.out.write_text(render(res, a.get("protocol"), b.get("protocol")))
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

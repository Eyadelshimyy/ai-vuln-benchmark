#!/usr/bin/env python3
"""
VULCAN -- contamination / memorization analysis.

The mined libraries pre-date the models, so a reviewer will ask: is
"AI-generated code reproduces the task's security posture" just the model
regurgitating memorized source? This measures, per AI sample, how close it is
to the REAL implementation of the same function, two ways:

  * text similarity   -- canonicalised source (docstrings stripped, reformatted
                         via ast.unparse) compared with difflib.
  * structural sim.   -- identifier-blind: local names, arg names, the function
                         name and literal constants are all replaced by
                         placeholders, so a copy with renamed variables still
                         scores ~1.0. This is the memorization signal.

Some similarity is EXPECTED and is not contamination: both implement the same
function from the same docstring, so they converge. The memorization signal is
*near-verbatim* copying -- text sim > 0.95 or structural sim == 1.0. If those
fractions are low, the posture-reproduction is genuine behaviour, not copying.

If a repo's benchmark_results.json is present, it also cross-tabs similarity
against the TRIGGERED verdict: the key defence is that the model's *vulnerable*
code is NOT a near-copy of the real vulnerable code.

USAGE:
  python3 contamination.py --root calibration/mined_prompts/
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import statistics as stats
import textwrap
from pathlib import Path


def _strip_docstrings(tree):
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
           and isinstance(getattr(body[0], "value", None), ast.Constant) \
           and isinstance(body[0].value.value, str):
            n.body = body[1:] or [ast.Pass()]
    return tree


def _canonical(src: str):
    try:
        tree = _strip_docstrings(ast.parse(textwrap.dedent(src)))
        return ast.unparse(tree)
    except Exception:
        return None


class _Blind(ast.NodeTransformer):
    def visit_Name(self, n):
        n.id = "V"; return n
    def visit_arg(self, n):
        n.arg = "V"; self.generic_visit(n); return n
    def visit_FunctionDef(self, n):
        n.name = "F"; self.generic_visit(n); return n
    def visit_AsyncFunctionDef(self, n):
        n.name = "F"; self.generic_visit(n); return n
    def visit_Constant(self, n):
        if isinstance(n.value, str):
            n.value = "S"
        elif isinstance(n.value, bool):
            pass
        elif isinstance(n.value, (int, float)):
            n.value = 0
        return n


def _structural(src: str):
    try:
        tree = _strip_docstrings(ast.parse(textwrap.dedent(src)))
        return ast.unparse(_Blind().visit(tree))
    except Exception:
        return None


def _ws(src: str) -> str:
    lines = [ln.split("#")[0] for ln in src.splitlines()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _ratio(a, b):
    if not a or not b:
        return None
    return difflib.SequenceMatcher(None, a, b).ratio()


def similarities(sample: str, gt: str) -> tuple:
    cs, cg = _canonical(sample), _canonical(gt)
    text = _ratio(cs, cg) if (cs and cg) else _ratio(_ws(sample), _ws(gt))
    ss, sg = _structural(sample), _structural(gt)
    struct = _ratio(ss, sg) if (ss and sg) else None
    return text, struct


def _iter_samples(gens: dict):
    t0 = gens.get("temp_0_0")
    if t0 and t0.get("ok") and t0.get("source"):
        yield ("t0", t0["source"])
    for s in gens.get("temp_0_7", []):
        if s.get("ok") and s.get("source"):
            yield (f"t07_{s['index']}", s["source"])


def _verdict_index(root_file: Path) -> dict:
    """Map (function, sample-label) -> verdict from benchmark_results.json, if present."""
    bench = root_file.parent / "benchmark_results.json"
    out = {}
    if not bench.exists():
        return out
    try:
        data = json.loads(bench.read_text())
    except Exception:
        return out
    for r in data.get("per_candidate", []):
        fn = r.get("function")
        for model, m in r.get("models", {}).items():
            for s in m.get("samples", []):
                out[(model, fn, s.get("sample"))] = s.get("verdict")
    return out


def _summ(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": round(stats.median(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3),
            "pct_ge_0.95": round(100 * sum(1 for v in vals if v >= 0.95) / len(vals), 1),
            "pct_lt_0.70": round(100 * sum(1 for v in vals if v < 0.70) / len(vals), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="dir with <repo>/*.json candidate files")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.root.glob("*/*.json"))
    files = [f for f in files if f.name not in ("benchmark_results.json", "auto_harness_results.json",
                                                 "tier2_results.json", "rq4_results.json")]

    # model -> lists
    per_model: dict = {}
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(rec, dict) or "prompt_source" not in rec:
            continue
        gt = rec.get("ground_truth_source")
        if not gt:
            continue
        vindex = _verdict_index(f)
        fn = rec.get("function")
        for model, gens in rec.get("generations", {}).items():
            d = per_model.setdefault(model, {"text": [], "struct": [],
                                             "text_trig": [], "struct_trig": [],
                                             "near_copies": []})
            for label, src in _iter_samples(gens):
                t, s = similarities(src, gt)
                d["text"].append(t); d["struct"].append(s)
                verdict = vindex.get((model, fn, label))
                if verdict == "TRIGGERED":
                    d["text_trig"].append(t); d["struct_trig"].append(s)
                if (t is not None and t >= 0.95) or (s is not None and s >= 0.999):
                    d["near_copies"].append(f"{rec.get('repo')}:{fn}:{label}")

    report = {}
    print("\n" + "=" * 78)
    print("VULCAN -- contamination / memorization analysis")
    print("=" * 78)
    for model in sorted(per_model):
        d = per_model[model]
        print(f"\nMODEL: {model}")
        print(f"  text similarity to real impl       : {_summ(d['text'])}")
        print(f"  structural similarity (id-blind)   : {_summ(d['struct'])}")
        if d["text_trig"]:
            print(f"  -- among TRIGGERED (vulnerable) samples --")
            print(f"  text similarity                    : {_summ(d['text_trig'])}")
            print(f"  structural similarity              : {_summ(d['struct_trig'])}")
        n_all = len([x for x in d["text"] if x is not None])
        n_copy = len(d["near_copies"])
        print(f"  near-verbatim copies (text>=.95 or struct==1): {n_copy}/{n_all}"
              f"  ({100*n_copy/n_all:.1f}%)" if n_all else "  near-verbatim copies: n/a")
        if d["near_copies"]:
            print(f"    {', '.join(d['near_copies'][:20])}" + (" ..." if n_copy > 20 else ""))
        report[model] = {"text": _summ(d["text"]), "struct": _summ(d["struct"]),
                         "text_triggered": _summ(d["text_trig"]), "struct_triggered": _summ(d["struct_trig"]),
                         "near_verbatim_copies": d["near_copies"], "samples": n_all}

    print("\n" + "=" * 78)
    print("Reading: moderate similarity is EXPECTED (same function, same docstring).")
    print("The memorization signal is the near-verbatim fraction. If it is low -- and")
    print("especially if the TRIGGERED (vulnerable) samples are not near-copies -- then")
    print("the posture-reproduction is genuine behaviour, not regurgitated source.")
    print("=" * 78)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

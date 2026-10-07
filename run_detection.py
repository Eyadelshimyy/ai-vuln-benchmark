#!/usr/bin/env python3
"""VULCAN Axis B runner -- grade LLM vulnerability detection against execution.

For each AI-generated function, we have:
  * EXECUTION GROUND TRUTH (Tier 1): VULN (tainted input reached a sink),
    SAFE (ran, nothing reached a sink), or UNRESOLVED (could not execute).
  * a DETECTOR VERDICT from one model (single) or several (consortium).

We compare them in a confusion matrix, EXCLUDING UNRESOLVED (a not-run case is
not a safe case). The detector is never ground truth; execution is.

Granularity: the detector judges the whole function; execution judges it at the
tested injection point(s) and reduces to one function-level verdict. These match
at the per-sample level, which is the comparison unit here.

USAGE:
  export OPENSOURCE_BASE_URL=http://localhost:11434/v1
  # single detector:
  python3 run_detection.py --candidates calibration/mined_prompts/sqlite-utils/ \
      --generator qwen2.5-coder:7b --detectors deepseek-coder:6.7b
  # consortium (several judges, majority vote):
  python3 run_detection.py --candidates calibration/mined_prompts/sqlite-utils/ \
      --generator qwen2.5-coder:7b --detectors qwen2.5-coder:7b,deepseek-coder:6.7b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))

from auto_harness import run_candidate              # noqa: E402
from run_benchmark import reduce_verdicts, iter_samples  # noqa: E402
from detection.llm_detector import detect, detect_panel  # noqa: E402

# Tier-1 verdict -> execution ground-truth class.
_TRUTH = {"TRIGGERED": "VULN", "NOT_TRIGGERED": "SAFE", "COULD_NOT_EXECUTE": "UNRESOLVED"}


def exec_truth(record: dict, source: str) -> dict:
    tmp = dict(record)
    tmp["generated_source"] = source
    red = reduce_verdicts(run_candidate(tmp, use="generated"))
    return {"truth": _TRUTH.get(red["verdict"], "UNRESOLVED"),
            "cwe": red.get("cwe"), "tier1": red["verdict"]}


def _matrix(pairs):
    """pairs: list of (truth in {VULN,SAFE}, pred in {vulnerable,safe}).
    Returns the confusion counts and standard metrics."""
    tp = sum(1 for t, p in pairs if t == "VULN" and p == "vulnerable")
    fn = sum(1 for t, p in pairs if t == "VULN" and p == "safe")
    fp = sum(1 for t, p in pairs if t == "SAFE" and p == "vulnerable")
    tn = sum(1 for t, p in pairs if t == "SAFE" and p == "safe")
    n = tp + fn + fp + tn

    def _r(a, b):
        return round(a / b, 3) if b else None
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "n": n,
            "precision": _r(tp, tp + fp), "recall": _r(tp, tp + fn),
            "FNR": _r(fn, tp + fn), "FPR": _r(fp, fp + tn),
            "accuracy": _r(tp + tn, n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--generator", required=True, help="whose generated code is judged")
    ap.add_argument("--detectors", required=True, help="comma-separated judge model(s)")
    ap.add_argument("--samples", type=int, default=5, help="judgments per model (majority vote)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--rule", default="majority", choices=["majority", "any", "unanimous_safe"])
    ap.add_argument("--base-url", default=os.environ.get("OPENSOURCE_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    detector_names = [d.strip() for d in args.detectors.split(",") if d.strip()]
    is_panel = len(detector_names) > 1

    from generate_batch import make_caller
    callers = {name: make_caller(args.base_url, name) for name in detector_names}

    files = ([args.candidates] if args.candidates.is_file()
             else sorted(p for p in args.candidates.glob("*.json")
                         if p.name not in ("auto_harness_results.json", "benchmark_results.json",
                                           "tier2_results.json", "detection_results.json")))
    files = [p for p in files if "prompt_source" in json.loads(p.read_text())]

    print(f"Axis-B | generator={args.generator} | detectors={detector_names} "
          f"| rule={'single' if not is_panel else args.rule}", file=sys.stderr)

    records = []
    for path in files:
        rec = json.loads(path.read_text())
        gens = rec.get("generations", {}).get(args.generator)
        if not gens:
            continue
        for slot, source in iter_samples(gens):
            gt = exec_truth(rec, source)
            row = {
                "library_id": rec.get("repo"), "function_id": rec.get("function"),
                "generator_model": args.generator, "sample_id": slot,
                "exec_truth": gt["truth"], "exec_cwe": gt["cwe"], "tier1": gt["tier1"],
            }
            if is_panel:
                pan = detect_panel(callers, source, rec.get("function"), args.samples,
                                   args.temperature, args.rule)
                row["panel_verdict"] = pan["verdict"]
                row["panel_cwe"] = pan.get("cwe")
                row["all_members_safe"] = pan.get("all_members_safe")
                row["members"] = {n: {"verdict": m["verdict"], "cwe": m.get("cwe")}
                                  for n, m in pan["members"].items()}
            else:
                d = detect(callers[detector_names[0]], source, rec.get("function"),
                           args.samples, args.temperature)
                row["detector_verdict"] = d["verdict"]
                row["detector_cwe"] = d.get("cwe")
                row["confidence"] = d.get("confidence")
            records.append(row)
            print(f"  {row['function_id']:>16}:{slot:8}  exec={gt['truth']:10} "
                  f"det={row.get('panel_verdict') or row.get('detector_verdict')}",
                  file=sys.stderr, flush=True)

    # ---------- confusion matrices (exclude UNRESOLVED and abstain) ----------
    scored = [r for r in records if r["exec_truth"] in ("VULN", "SAFE")]
    unresolved = sum(1 for r in records if r["exec_truth"] == "UNRESOLVED")

    print("\n" + "=" * 70)
    print(f"VULCAN AXIS-B (LLM detection vs execution ground truth)")
    print(f"generator={args.generator}")
    print("=" * 70)
    print(f"samples: {len(records)} total | {len(scored)} scored (VULN/SAFE) "
          f"| {unresolved} UNRESOLVED excluded")
    gt_v = sum(1 for r in scored if r["exec_truth"] == "VULN")
    print(f"ground truth: {gt_v} vulnerable, {len(scored)-gt_v} safe")
    print("-" * 70)

    report = {"generator": args.generator, "detectors": detector_names,
              "rule": args.rule if is_panel else "single",
              "n_total": len(records), "n_scored": len(scored), "unresolved": unresolved,
              "records": records}

    def _show(label, pairs):
        m = _matrix([(t, p) for t, p in pairs if p in ("vulnerable", "safe")])
        print(f"\n[{label}]")
        print(f"  TP {m['TP']}  FN {m['FN']}  FP {m['FP']}  TN {m['TN']}   (n={m['n']})")
        print(f"  precision {m['precision']}  recall {m['recall']}  "
              f"FNR {m['FNR']}  FPR {m['FPR']}  accuracy {m['accuracy']}")
        return m

    report["matrices"] = {}
    if is_panel:
        # individual members
        for name in detector_names:
            pairs = [(r["exec_truth"], r["members"].get(name, {}).get("verdict"))
                     for r in scored]
            report["matrices"][name] = _show(f"member: {name}", pairs)
        # consortium
        pairs = [(r["exec_truth"], r["panel_verdict"]) for r in scored]
        report["matrices"][f"CONSORTIUM ({args.rule})"] = _show(
            f"CONSORTIUM ({args.rule})", pairs)
        # panel-wide misses: everyone said safe, execution says vulnerable
        misses = [r for r in scored if r["exec_truth"] == "VULN" and r.get("all_members_safe")]
        print(f"\n  PANEL-WIDE MISSES (all members said safe, execution confirms vulnerable): "
              f"{len(misses)}")
        for r in misses[:10]:
            print(f"     {r['library_id']}:{r['function_id']}:{r['sample_id']} "
                  f"(exec {r['exec_cwe']})")
        report["panel_wide_misses"] = [
            {"library": r["library_id"], "function": r["function_id"],
             "sample": r["sample_id"], "exec_cwe": r["exec_cwe"]} for r in misses]
    else:
        pairs = [(r["exec_truth"], r["detector_verdict"]) for r in scored]
        report["matrices"][detector_names[0]] = _show(detector_names[0], pairs)
        # the single-detector false negatives (said safe, execution says vulnerable)
        fns = [r for r in scored if r["exec_truth"] == "VULN" and r["detector_verdict"] == "safe"]
        print(f"\n  FALSE NEGATIVES (detector said safe, execution confirms vulnerable): {len(fns)}")
        for r in fns[:10]:
            print(f"     {r['library_id']}:{r['function_id']}:{r['sample_id']} (exec {r['exec_cwe']})")
        report["false_negatives"] = [
            {"library": r["library_id"], "function": r["function_id"],
             "sample": r["sample_id"], "exec_cwe": r["exec_cwe"]} for r in fns]

    print("\n" + "=" * 70)
    print("Note: UNRESOLVED execution cases are excluded from the matrix, not counted")
    print("as safe. The detector is never ground truth; execution adjudicates.")

    out = args.out or ((args.candidates.parent if args.candidates.is_file() else args.candidates)
                       / "detection_results.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nFull auditable results: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

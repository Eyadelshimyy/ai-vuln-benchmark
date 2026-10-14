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
from detection.llm_detector import detect, detect_panel, build_messages  # noqa: E402

# Tier-1 verdict -> execution ground-truth class.
_TRUTH = {"TRIGGERED": "VULN", "NOT_TRIGGERED": "SAFE", "COULD_NOT_EXECUTE": "UNRESOLVED"}


def exec_truth(record: dict, source: str, use: str = "generated") -> dict:
    tmp = dict(record)
    if use == "generated":
        tmp["generated_source"] = source
    red = reduce_verdicts(run_candidate(tmp, use=use))
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
    ap.add_argument("--use", default="generated", choices=["generated", "ground_truth"],
                    help="generated: judge a model's implementations; ground_truth: judge the "
                         "real AI-written code as-is (real-AI-repo mode)")
    ap.add_argument("--generator", default=None, help="whose generated code is judged "
                    "(required unless --use ground_truth; in ground_truth mode this is just a "
                    "provenance label for the source repo's model, e.g. 'claude')")
    ap.add_argument("--detectors", required=True, help="comma-separated judge model(s)")
    ap.add_argument("--samples", type=int, default=5, help="judgments per model (majority vote)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--rule", default="majority", choices=["majority", "any", "unanimous_safe"])
    ap.add_argument("--base-url", default=os.environ.get("OPENSOURCE_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--recursive", action="store_true",
                    help="load candidates from <candidates>/*/*.json (a whole mined corpus) as ONE "
                         "experiment -- one matrix per model + consortium")
    ap.add_argument("--cwe", default=None, help="restrict to candidates whose static tags include this CWE")
    ap.add_argument("--skip-unresolved", action="store_true",
                    help="execute each candidate FIRST (free) and skip the detector call when the "
                         "verdict is UNRESOLVED -- those rows are excluded from the matrix anyway, so "
                         "this cuts paid API calls ~35-40%% with no effect on results")
    ap.add_argument("--estimate", action="store_true",
                    help="DRY RUN: execute the corpus (free), then print the exact number of detector "
                         "calls and estimated input/output tokens + a cost table -- WITHOUT calling any "
                         "model. Use to preview the bill before spending on a paid API.")
    args = ap.parse_args()

    # Vendored-source importability is applied PER CANDIDATE inside the loop
    # below (each mined repo has its own _repo_src), because --recursive runs a
    # whole corpus where a single global sys.path entry would be wrong. See
    # vendor_sys_path_roots and the exec_truth call site.
    from auto_harness import vendor_sys_path_roots  # noqa: E402

    # Each --detectors entry is "model" or "model|base_url" or
    # "model|base_url|API_KEY_ENV" -- so a LOCAL model (Ollama) and a REMOTE
    # paid model (OpenCode) can run in ONE experiment, each with its own
    # endpoint and key env. base_url defaults to --base-url; key env defaults to
    # OPENSOURCE_API_KEY. The key itself is NEVER on the command line (only the
    # name of the env var holding it), so it stays out of shell history/logs.
    detector_names, _specs = [], {}
    for raw in args.detectors.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split("|")]
        model = parts[0]
        base_url = parts[1] if len(parts) > 1 and parts[1] else args.base_url
        key_env = parts[2] if len(parts) > 2 and parts[2] else "OPENSOURCE_API_KEY"
        detector_names.append(model)
        _specs[model] = (base_url, key_env)
    is_panel = len(detector_names) > 1

    from generate_batch import make_caller
    callers = None if args.estimate else {
        name: make_caller(_specs[name][0], name, _specs[name][1]) for name in detector_names}
    if not args.estimate:
        for name in detector_names:
            print(f"  detector {name}  ->  {_specs[name][0]}  (key env: {_specs[name][1]})", file=sys.stderr)

    _RESULT_NAMES = ("auto_harness_results.json", "benchmark_results.json",
                     "tier2_results.json", "detection_results.json", "provenance_results.json")
    # --recursive: treat a whole mined corpus (mined_prompts/<repo>/*.json) as ONE
    # experiment -> one confusion matrix per model + consortium, instead of a
    # separate run per repo. Non-recursive default is unchanged (one dir).
    if args.candidates.is_file():
        files = [args.candidates]
    else:
        globber = args.candidates.rglob if args.recursive else args.candidates.glob
        files = sorted(p for p in globber("*.json") if p.name not in _RESULT_NAMES)
    files = [p for p in files if "prompt_source" in json.loads(p.read_text())]
    if args.cwe:
        files = [p for p in files if args.cwe in (json.loads(p.read_text()).get("candidate_cwes") or [])]

    print(f"Axis-B | generator={args.generator or 'repo-provenance'} | detectors={detector_names} | use={args.use} "
          f"| rule={'single' if not is_panel else args.rule}", file=sys.stderr)

    if args.use == "generated" and not args.generator:
        raise SystemExit("--generator is required unless --use ground_truth")
    gen_label = args.generator or "repo-provenance"

    records = []
    est_prompt_chars = 0   # --estimate: summed prompt chars over candidates that WOULD be detected
    est_detected = 0       # --estimate: count of detector calls per model
    for path in files:
        rec = json.loads(path.read_text())
        _roots = vendor_sys_path_roots(path.parent / "_repo_src")  # this repo's import roots
        if args.use == "ground_truth":
            src = rec.get("ground_truth_source")
            samples_iter = [("ground_truth", src)] if src else []
        else:
            gens = rec.get("generations", {}).get(args.generator)
            if not gens:
                continue
            samples_iter = iter_samples(gens)
        for slot, source in samples_iter:
            if not source:
                continue
            for _r in _roots:                       # append, don't shadow stdlib (see census)
                if _r not in sys.path:
                    sys.path.append(_r)
            try:
                gt = exec_truth(rec, source, use=args.use)
            finally:
                for _r in _roots:
                    if _r in sys.path:
                        sys.path.remove(_r)
            row = {
                "library_id": rec.get("repo"), "function_id": rec.get("function"),
                "generator_model": gen_label, "sample_id": slot,
                "exec_truth": gt["truth"], "exec_cwe": gt["cwe"], "tier1": gt["tier1"],
            }
            # would a real run call the paid detector on this candidate?
            _detect_this = not (args.skip_unresolved and gt["truth"] == "UNRESOLVED")
            if args.estimate:
                if _detect_this:
                    _msgs = build_messages(source, rec.get("function") or "")
                    est_prompt_chars += sum(len(m["content"]) for m in _msgs)
                    est_detected += 1
                records.append(row)
                continue
            if not _detect_this:
                records.append(row)  # UNRESOLVED -> excluded from matrix anyway; no paid call
                continue
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

    if args.estimate:
        n_total = len(records)
        n_unres = sum(1 for r in records if r["exec_truth"] == "UNRESOLVED")
        in_tok = round(est_prompt_chars / 4) * args.samples          # ~4 chars/token
        out_tok = est_detected * args.samples * 60                   # ~60-token JSON verdict
        print("\n" + "=" * 68)
        print("COST ESTIMATE (dry run -- NO model calls were made)")
        print("=" * 68)
        print(f"candidates evaluated : {n_total}  ({n_unres} UNRESOLVED)")
        print(f"detector calls / model: {est_detected * args.samples}"
              + (f"   (--skip-unresolved ON: skipped {n_total - est_detected} UNRESOLVED)"
                 if args.skip_unresolved else "   (use --skip-unresolved to drop UNRESOLVED calls)"))
        print(f"per PAID model  -> input ~{in_tok:,} tok | output ~{out_tok:,} tok "
              f"(~{in_tok+out_tok:,} total)")
        print("\napprox cost for ONE paid model, by price tier ($/1M in, $/1M out):")
        for label, pin, pout in [("small  (0.15/0.60)", 0.15, 0.60),
                                 ("mid    (2.50/10)", 2.50, 10.0),
                                 ("frontier (3/15)", 3.0, 15.0),
                                 ("top    (15/75)", 15.0, 75.0)]:
            cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
            print(f"   {label:22} ~${cost:,.2f}")
        print("\n(local models e.g. Ollama qwen cost $0 -- only paid API models incur this.)")
        if args.out:
            args.out.write_text(json.dumps(
                {"estimate": True, "candidates": n_total, "unresolved": n_unres,
                 "calls_per_model": est_detected * args.samples,
                 "input_tokens_per_model": in_tok, "output_tokens_per_model": out_tok,
                 "samples": args.samples, "skip_unresolved": args.skip_unresolved}, indent=2))
        return

    # ---------- confusion matrices (exclude UNRESOLVED and abstain) ----------
    scored = [r for r in records if r["exec_truth"] in ("VULN", "SAFE")]
    unresolved = sum(1 for r in records if r["exec_truth"] == "UNRESOLVED")

    print("\n" + "=" * 70)
    print(f"VULCAN AXIS-B (LLM detection vs execution ground truth)")
    print(f"generator={gen_label}")
    print("=" * 70)
    print(f"samples: {len(records)} total | {len(scored)} scored (VULN/SAFE) "
          f"| {unresolved} UNRESOLVED excluded")
    gt_v = sum(1 for r in scored if r["exec_truth"] == "VULN")
    print(f"ground truth: {gt_v} vulnerable, {len(scored)-gt_v} safe")
    print("-" * 70)

    report = {"generator": gen_label, "detectors": detector_names,
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

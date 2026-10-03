#!/usr/bin/env python3
"""
VULCAN -- Step 2: the benchmark runner.

Consumes the multi-sample generations written by generate_batch.py and, for
every candidate, every model, and every sample, executes the generated code
in the IAST harness and records one verdict per sample. It also runs each
candidate's REAL (human-written) implementation through the identical
harness as a control, which gives two things at once:

  (1) the D classifier -- a generated COULD_NOT_EXECUTE is labelled
      MODEL-caused if the real function executes in the same harness, or
      INFRASTRUCTURE-caused if even the real function cannot run here;
  (2) the human-vs-AI baseline -- the real implementation's own verdict,
      compared against the AI samples on the identical task (the comparison
      the assigned static-analysis methodology wanted, done by execution).

Metrics reported (honest denominators throughout):
  * execution rate        = executed / (executed + model-caused CNE)
  * trigger rate (vuln@k) = triggered / executed, with a Wilson 95% CI
  * severity-weighted trigger rate (CWE-weighted, borrowed refinement)
  * human baseline trigger rate on the same candidates
INFRASTRUCTURE-caused failures are excluded from the model's denominator,
since they are harness limitations, not model behaviour.

USAGE:
  python3 analysis/iast/run_benchmark.py --candidates calibration/mined_prompts/sqlite-utils/
  python3 analysis/iast/run_benchmark.py --candidates calibration/mined_prompts/sqlite-utils/ --model qwen2.5-coder:7b
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auto_harness import run_candidate  # noqa: E402

# CWE -> severity weight (CVSS-style base, approximate; used only for the
# severity-weighted rate, reported alongside the raw rate, never instead).
CWE_SEVERITY = {
    "CWE-89": 9.8, "CWE-78": 9.8, "CWE-94": 9.8, "CWE-95": 9.8,
    "CWE-502": 9.8, "CWE-918": 8.6, "CWE-22": 7.5, "CWE-611": 7.5,
}
_DEFAULT_SEV = 7.0


def reduce_verdicts(verdicts: list) -> dict:
    """Collapse per-injection-point verdicts into one candidate/sample-level
    outcome: TRIGGERED if any point triggered (carry its CWE/sink), else
    NOT_TRIGGERED if any point executed, else COULD_NOT_EXECUTE."""
    triggered = [v for v in verdicts if v.verdict == "TRIGGERED"]
    if triggered:
        t = triggered[0]
        return {"verdict": "TRIGGERED", "cwe": t.cwe, "sink": t.sink,
                "param": t.injected_param, "detail": t.detail}
    if any(v.verdict == "NOT_TRIGGERED" for v in verdicts):
        return {"verdict": "NOT_TRIGGERED", "cwe": None, "sink": None, "param": None, "detail": None}
    err = next((v.error for v in verdicts if v.error), None)
    return {"verdict": "COULD_NOT_EXECUTE", "cwe": None, "sink": None, "param": None, "detail": err}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def iter_samples(model_gens: dict):
    """Yield (label, source) for each generated sample that has code."""
    t0 = model_gens.get("temp_0_0")
    if t0 and t0.get("ok") and t0.get("source"):
        yield ("t0", t0["source"])
    for s in model_gens.get("temp_0_7", []):
        if s.get("ok") and s.get("source"):
            yield (f"t07_{s['index']}", s["source"])


def run_one_candidate(record: dict, only_model: str | None) -> dict:
    # Human control (the real implementation), run through the same harness.
    human = reduce_verdicts(run_candidate(record, use="ground_truth"))
    human_executes = human["verdict"] in ("TRIGGERED", "NOT_TRIGGERED")

    per_model = {}
    for model, gens in record.get("generations", {}).items():
        if only_model and model != only_model:
            continue
        samples_out = []
        for label, source in iter_samples(gens):
            tmp = dict(record)
            tmp["generated_source"] = source
            res = reduce_verdicts(run_candidate(tmp, use="generated"))
            if res["verdict"] == "COULD_NOT_EXECUTE":
                res["failure_class"] = "MODEL" if human_executes else "INFRASTRUCTURE"
            samples_out.append({"sample": label, **res})
        # count generation failures (model returned no usable code at all)
        gen_failed = sum(1 for s in gens.get("temp_0_7", []) if not s.get("ok"))
        if gens.get("temp_0_0") and not gens["temp_0_0"].get("ok"):
            gen_failed += 1
        per_model[model] = {"samples": samples_out, "generation_failed": gen_failed}

    return {"repo": record.get("repo"), "file": record.get("file"),
            "function": record.get("function"), "line": record.get("line"),
            "human": human, "models": per_model}


def aggregate(results: list[dict]) -> dict:
    # Human baseline over candidates whose real impl executes.
    human_exec = [r for r in results if r["human"]["verdict"] in ("TRIGGERED", "NOT_TRIGGERED")]
    human_trig = sum(1 for r in human_exec if r["human"]["verdict"] == "TRIGGERED")
    human_lo, human_hi = wilson_ci(human_trig, len(human_exec))

    models: dict = {}
    for r in results:
        for model, m in r["models"].items():
            agg = models.setdefault(model, {
                "triggered": 0, "not_triggered": 0, "cne_model": 0, "cne_infra": 0,
                "generation_failed": 0, "sev_weight_triggered": 0.0, "by_cwe": {}})
            agg["generation_failed"] += m["generation_failed"]
            for s in m["samples"]:
                v = s["verdict"]
                if v == "TRIGGERED":
                    agg["triggered"] += 1
                    cwe = s.get("cwe") or "UNKNOWN"
                    agg["by_cwe"][cwe] = agg["by_cwe"].get(cwe, 0) + 1
                    agg["sev_weight_triggered"] += CWE_SEVERITY.get(cwe, _DEFAULT_SEV)
                elif v == "NOT_TRIGGERED":
                    agg["not_triggered"] += 1
                elif s.get("failure_class") == "MODEL":
                    agg["cne_model"] += 1
                else:
                    agg["cne_infra"] += 1

    for model, a in models.items():
        executed = a["triggered"] + a["not_triggered"]
        # denominator excludes INFRASTRUCTURE failures (harness limits, not model)
        model_base = executed + a["cne_model"]
        a["executed"] = executed
        a["execution_rate"] = (executed / model_base) if model_base else 0.0
        a["trigger_rate"] = (a["triggered"] / executed) if executed else 0.0
        lo, hi = wilson_ci(a["triggered"], executed)
        a["trigger_rate_ci95"] = [round(lo, 4), round(hi, 4)]
        a["severity_weighted_trigger_rate"] = (
            a["sev_weight_triggered"] / (executed * 10.0)) if executed else 0.0

    return {
        "human_baseline": {
            "candidates_executed": len(human_exec),
            "triggered": human_trig,
            "trigger_rate": (human_trig / len(human_exec)) if human_exec else 0.0,
            "trigger_rate_ci95": [round(human_lo, 4), round(human_hi, 4)],
        },
        "models": models,
    }


def print_summary(agg: dict) -> None:
    hb = agg["human_baseline"]
    print("\n" + "=" * 70)
    print("VULCAN BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Human baseline (real code): {hb['triggered']}/{hb['candidates_executed']} "
          f"candidates triggered  =  {hb['trigger_rate']*100:.1f}%  "
          f"(95% CI {hb['trigger_rate_ci95'][0]*100:.1f}-{hb['trigger_rate_ci95'][1]*100:.1f}%)")
    print("-" * 70)
    for model, a in agg["models"].items():
        print(f"\nModel: {model}")
        print(f"  executed {a['executed']}  |  triggered {a['triggered']}  "
              f"not_triggered {a['not_triggered']}  |  CNE(model) {a['cne_model']}  "
              f"CNE(infra) {a['cne_infra']}  gen_failed {a['generation_failed']}")
        print(f"  execution rate        : {a['execution_rate']*100:.1f}%  "
              f"(infra failures excluded from denominator)")
        print(f"  trigger rate (vuln@k) : {a['trigger_rate']*100:.1f}%  "
              f"(95% CI {a['trigger_rate_ci95'][0]*100:.1f}-{a['trigger_rate_ci95'][1]*100:.1f}%), "
              f"denominator = executed samples")
        print(f"  severity-weighted     : {a['severity_weighted_trigger_rate']*100:.1f}%")
        if a["by_cwe"]:
            print(f"  triggers by CWE       : " + ", ".join(f"{k}:{v}" for k, v in sorted(a["by_cwe"].items())))
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path, help="a candidate JSON file or a directory of them")
    ap.add_argument("--model", default=None, help="restrict to one model (default: all models present)")
    ap.add_argument("--out", type=Path, default=None, help="where to write the full results JSON")
    args = ap.parse_args()

    if args.candidates.is_dir():
        files = sorted(p for p in args.candidates.glob("*.json")
                       if p.name not in ("auto_harness_results.json", "benchmark_results.json"))
    else:
        files = [args.candidates]
    files = [p for p in files if "prompt_source" in json.loads(p.read_text())]
    if not files:
        sys.exit(f"no candidate files with prompt_source at {args.candidates}")

    results = []
    for n, path in enumerate(files, 1):
        record = json.loads(path.read_text())
        if not record.get("generations"):
            print(f"[{n}/{len(files)}] {record.get('function')}  SKIP (no generations -- run generate_batch.py first)",
                  file=sys.stderr)
            continue
        print(f"[{n}/{len(files)}] {record.get('function')} ...", file=sys.stderr, flush=True)
        results.append(run_one_candidate(record, args.model))

    agg = aggregate(results)
    print_summary(agg)

    out = args.out or ((args.candidates if args.candidates.is_dir() else args.candidates.parent) / "benchmark_results.json")
    out.write_text(json.dumps({"aggregate": agg, "per_candidate": results}, indent=2))
    print(f"\nFull results written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

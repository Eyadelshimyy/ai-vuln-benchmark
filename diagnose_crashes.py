#!/usr/bin/env python3
"""
VULCAN -- crash diagnostic / validity check for COULD_NOT_EXECUTE.

The question this answers: when a generated sample is COULD_NOT_EXECUTE, is
that the MODEL's fault (its code genuinely crashes) or OUR fault (the harness
or our construction failed before the model's code even ran)? Every RQ1
exclusion and every RQ3 "model-caused crash" rests on that distinction, so we
verify it directly instead of trusting the counts.

It is read-only and FREE: it re-executes stored samples through the harness
(no model calls). Needs the full-traceback capture in auto_harness.py (the
COULD_NOT_EXECUTE branch now ships `detail`).

HOW EACH CRASH IS CLASSIFIED
  1. Run the real human implementation through the IDENTICAL harness.
       - It also can't run  -> INFRA   (harness limit, NOT the model's fault;
                                         correctly excluded from the denominator)
       - It runs fine       -> the harness CAN handle this candidate, so a model
                               crash here is real. Then localise by traceback:
  2. Deepest frame is `<candidate:...>` (the compiled generated code)
                            -> MODEL    (the model's own code crashed -- result stands)
  3. Deepest frame is auto_harness.py / sinks.py / <construction_recipe> /
     <imports>, with NO generated-code frame
                            -> HARNESS  (our construction/injection broke before
                                         the model's code ran -- a FALSE crash that
                                         must be fixed, not counted against the model)
  4. No traceback frames captured
                            -> UNKNOWN  (inspect by hand; --show-tb dumps it)

MODEL + INFRA are the healthy outcomes. HARNESS + UNKNOWN are the ones to
read: if either is non-zero, that many "model-caused crashes" are actually
ours and the affected results need re-running after a harness fix.

USAGE (local, free -- runs over whatever samples already exist):
  export OPENSOURCE_BASE_URL=http://localhost:11434/v1   # not used, kept for parity
  python3 diagnose_crashes.py --candidates calibration/mined_prompts/paramiko/ --model qwen2.5-coder:7b
  python3 diagnose_crashes.py --candidates calibration/mined_prompts/paramiko/ --model deepseek-coder:6.7b --show-tb
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "analysis" / "iast"))
from auto_harness import run_candidate  # noqa: E402

_FRAME = re.compile(r'File "([^"]+)", line \d+, in ')
_HARNESS_BASENAMES = {"auto_harness.py", "sinks.py", "run_benchmark.py"}
_HARNESS_PSEUDO = {"<construction_recipe>", "<imports>", "<string>", "<auto_harness_candidate>"}
# A parse failure of the GENERATED source means the model emitted code that is
# not valid Python (truncated mid-token, unterminated string, bad syntax). The
# real implementation always parses, so this is the model's fault, not ours --
# the harness correctly refusing to run garbage. It is NOT a harness bug.
_MODEL_SYNTAX_SIGNATURE = "could not parse candidate source"
# True harness-side signatures (our construction/limits, no model code involved).
_HARNESS_MSG_SIGNATURES = (
    "could not construct", "could not resolve",
    "can't start new thread", "blockingioerror",
    "resource temporarily unavailable", "<construction_recipe>", "<imports>",
)


def _reduce_keep_detail(verdicts: list) -> tuple[str, str | None]:
    """Collapse per-injection verdicts but PRESERVE the full traceback detail
    for a crash (run_benchmark.reduce_verdicts keeps only the first line)."""
    if any(v.verdict == "TRIGGERED" for v in verdicts):
        return "TRIGGERED", None
    if any(v.verdict == "NOT_TRIGGERED" for v in verdicts):
        return "NOT_TRIGGERED", None
    # COULD_NOT_EXECUTE: prefer a verdict that carries the full traceback.
    v = (next((v for v in verdicts if getattr(v, "detail", None)), None)
         or next((v for v in verdicts if v.error), None)
         or (verdicts[0] if verdicts else None))
    detail = (getattr(v, "detail", None) or (v.error if v else None)) if v else None
    return "COULD_NOT_EXECUTE", detail


def classify(human_verdict: str, detail: str | None) -> tuple[str, str]:
    if human_verdict == "COULD_NOT_EXECUTE":
        return "INFRA", "real implementation also fails in this harness (excluded)"
    d = detail or ""
    if _MODEL_SYNTAX_SIGNATURE in d:
        return "MODEL_SYNTAX", "model emitted unparseable/truncated Python (harness correctly rejected it)"
    files = _FRAME.findall(d)
    bases = [f.rsplit("/", 1)[-1] for f in files]
    if any(f.startswith("<candidate:") for f in files):
        return "MODEL", f"crash inside generated code (deepest: {files[-1]})"
    if files:
        deepest, deepest_base = files[-1], bases[-1]
        if deepest_base in _HARNESS_BASENAMES or deepest in _HARNESS_PSEUDO:
            return "HARNESS", f"crash in harness/construction, no generated-code frame ({deepest})"
        return "HARNESS", f"deepest frame in library reached via construction, no generated-code frame ({deepest})"
    # no frames at all -- fall back to the one-line message
    msg = (detail or "").lower()
    if any(sig in msg for sig in _HARNESS_MSG_SIGNATURES):
        return "HARNESS", "harness-signature error, no traceback frames"
    return "UNKNOWN", "no traceback frames captured -- inspect with --show-tb"


def _iter_samples(gens: dict):
    t0 = gens.get("temp_0_0")
    if t0 and t0.get("ok") and t0.get("source"):
        yield ("temp_0_0", t0["source"])
    for i, s in enumerate(gens.get("temp_0_7", [])):
        if s.get("ok") and s.get("source"):
            yield (f"temp_0_7[{i}]", s["source"])


def _run(record: dict, source: str | None, use: str):
    r = dict(record)
    if use == "generated":
        r["generated_source"] = source
    return _reduce_keep_detail(run_candidate(r, use=use))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--show-tb", action="store_true", help="print the full traceback for every HARNESS/UNKNOWN crash")
    ap.add_argument("--show-all-tb", action="store_true", help="print the full traceback for EVERY crash")
    args = ap.parse_args()

    files = ([args.candidates] if args.candidates.is_file()
             else sorted(p for p in args.candidates.glob("*.json")
                         if p.name not in ("auto_harness_results.json", "benchmark_results.json")))
    files = [p for p in files if "prompt_source" in json.loads(p.read_text())]

    counts = {"MODEL": 0, "MODEL_SYNTAX": 0, "INFRA": 0, "HARNESS": 0, "UNKNOWN": 0}
    flagged = []  # (label, func, slot, firstline, detail)

    for path in files:
        record = json.loads(path.read_text())
        gens = record.get("generations", {}).get(args.model)
        if not gens:
            continue
        func = record.get("function", path.stem)
        human_verdict, _ = _run(record, None, "ground_truth")

        for slot, source in _iter_samples(gens):
            verdict, detail = _run(record, source, "generated")
            if verdict != "COULD_NOT_EXECUTE":
                continue
            label, why = classify(human_verdict, detail)
            counts[label] += 1
            first = (detail or "").splitlines()[0] if detail else "(no detail)"
            print(f"  {label:<12} {func}  {slot}  human={human_verdict}")
            print(f"           {first}")
            print(f"           -> {why}")
            if args.show_all_tb or (args.show_tb and label in ("HARNESS", "UNKNOWN")):
                flagged.append((label, func, slot, detail or "(no detail captured)"))

    total = sum(counts.values())
    print("\n" + "=" * 66)
    print(f"CRASH DIAGNOSTIC -- model={args.model}  candidates={args.candidates}")
    print("=" * 66)
    model_caused = counts["MODEL"] + counts["MODEL_SYNTAX"]
    print(f"total COULD_NOT_EXECUTE samples : {total}")
    print(f"  MODEL        (model's code crashed at runtime -- result stands) : {counts['MODEL']}")
    print(f"  MODEL_SYNTAX (model emitted unparseable/truncated code)         : {counts['MODEL_SYNTAX']}")
    print(f"  INFRA        (real code also fails -- excluded from denominator): {counts['INFRA']}")
    print(f"  HARNESS      (OUR fault -- false crash, must fix)               : {counts['HARNESS']}")
    print(f"  UNKNOWN      (no traceback -- inspect by hand)                  : {counts['UNKNOWN']}")
    print(f"  -> model-caused (MODEL + MODEL_SYNTAX)                          : {model_caused}")
    print("=" * 66)
    if counts["HARNESS"] or counts["UNKNOWN"]:
        print("ACTION: HARNESS/UNKNOWN crashes are NOT the model's fault. Re-run with")
        print("--show-tb to read them; fix the harness and re-benchmark the affected repo.")
    else:
        print("CLEAN: every crash is the model's own code (runtime bug or invalid")
        print("output) or a real-code infra limit. No harness fault. Counts are valid.")
        if counts["MODEL_SYNTAX"]:
            print(f"NOTE: {counts['MODEL_SYNTAX']} MODEL_SYNTAX crashes -- some may be generation")
            print("truncation (raise max_tokens and regenerate to recover those samples).")

    if flagged:
        print("\n" + "-" * 66)
        print("FULL TRACEBACKS")
        print("-" * 66)
        for label, func, slot, detail in flagged:
            print(f"\n### [{label}] {func} {slot}")
            print(detail)


if __name__ == "__main__":
    main()

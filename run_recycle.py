#!/usr/bin/env python3
"""VULCAN recycle loop -- turn LLM detector feedback into tool improvement.

The supervisor's loop: the LLM detector flags some AI-written functions as
vulnerable. For every function it flagged that our EXECUTION oracle did NOT
already confirm (it ran safe, or it could not be executed at all), we take a
second pass:

  * an LLM *crafter* proposes concrete attacker inputs for that function and its
    claimed CWE (tier2/agent.py -- crafting only),
  * those inputs run through the function's REAL code in live mode, and
  * the SAME deterministic observable oracle (tier2/strategies.py) decides.

Execution stays the only judge. The LLM never promotes a verdict; it only
supplies payloads. A function the oracle still cannot fire after an honest
attempt stays NOT_CONFIRMED -- never counted as vulnerable on the model's word.

Two things come out of this, both of which the thesis wants:
  RECOVERED      prior exec was SAFE/UNRESOLVED, the crafted payload made the
                 oracle fire -> a REAL vulnerability our tool had missed. This
                 is a measured recall improvement for VULCAN.
  CONFIRMED-EXPLOIT  prior exec already TRIGGERED (taint reached the sink); the
                 crafted payload turns that into a proven exploit (Tier-2).
  STILL_NOT_CONFIRMED  honest attempt, oracle never fired. Reported with its
                 root-cause bucket so a genuine LLM false positive is told
                 apart from a harness/oracle gap we should still fix.

VALIDITY GUARD -- avoiding self-grading. The crafter model is chosen with
--crafter, INDEPENDENTLY of whichever detector raised the flag. A detector's
own hints never improve that detector's own score: the recovered ground truth
is produced by a fixed crafter and then (separately, in run_detection.py) every
detector is re-graded against it. Keep --crafter out of the detector panel you
are grading, or fold the recovered payloads into the deterministic bank so the
frozen truth is model-independent.

USAGE:
  export OPENSOURCE_BASE_URL=http://localhost:11434/v1
  python3 run_recycle.py \
      --candidates calibration/mined_prompts/vela_slides_ai/ \
      --use ground_truth \
      --detection calibration/mined_prompts/vela_slides_ai/detection_results.json \
      --crafter qwen2.5-coder:7b --max-agent 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "analysis" / "iast"))
sys.path.insert(0, str(_REPO))

from auto_harness import run_candidate, _parse_params, _looks_boolean  # noqa: E402
from run_benchmark import iter_samples              # noqa: E402
from tier2.engine import attempt_exploit            # noqa: E402
from tier2.strategies import STRATEGIES             # noqa: E402

_RESULT_NAMES = ("auto_harness_results.json", "benchmark_results.json",
                 "tier2_results.json", "detection_results.json",
                 "detection_results_any.json", "recycle_results.json")


def _candidate_files(candidates: Path) -> list[Path]:
    """Reproduce run_detection.py's candidate selection EXACTLY, so the records
    we rebuild line up one-for-one with detection_results.json['records']."""
    files = ([candidates] if candidates.is_file()
             else sorted(p for p in candidates.glob("*.json") if p.name not in _RESULT_NAMES))
    return [p for p in files if "prompt_source" in json.loads(p.read_text())]


def _rebuild_samples(candidates: Path, use: str) -> list[dict]:
    """Rebuild the (path, record, slot, source) stream in run_detection's order."""
    out = []
    for path in _candidate_files(candidates):
        rec = json.loads(path.read_text())
        if use == "ground_truth":
            src = rec.get("ground_truth_source")
            pairs = [("ground_truth", src)] if src else []
        else:
            # generated mode aligns per-generator; recycle is run per generator too
            pairs = []  # not used in ground_truth runs; see --use generated note below
        for slot, source in pairs:
            if not source:
                continue
            out.append({"path": path, "record": rec, "slot": slot})
    return out


def _flagged(det_rec: dict) -> tuple[bool, list[str]]:
    """Did ANY model call this function vulnerable, and with which CWE(s)?
    Works for both single-detector and consortium result rows."""
    cwes: list[str] = []
    flagged = False
    if det_rec.get("detector_verdict") == "vulnerable":        # single detector
        flagged = True
        if det_rec.get("detector_cwe"):
            cwes.append(det_rec["detector_cwe"])
    members = det_rec.get("members") or {}                      # consortium members
    for m in members.values():
        if m.get("verdict") == "vulnerable":
            flagged = True
            if m.get("cwe"):
                cwes.append(m["cwe"])
    if det_rec.get("panel_verdict") == "vulnerable":
        flagged = True
        if det_rec.get("panel_cwe"):
            cwes.append(det_rec["panel_cwe"])
    # unique, order-preserving
    seen: set = set()
    uniq = [c for c in cwes if not (c in seen or seen.add(c))]
    return flagged, uniq


def _target_params(verdicts: list) -> list[str]:
    """Injection points the harness identified for this candidate (the params it
    tried), regardless of whether any triggered."""
    seen: set = set()
    params = []
    for v in verdicts:
        p = getattr(v, "injected_param", None)
        if p and p not in seen:
            seen.add(p)
            params.append(p)
    return params


def _no_input_channel(rec: dict, use: str) -> bool:
    """True when the flagged function has NO injectable input channel at all --
    no parameter other than self/cls, or only boolean parameters (which cannot
    carry an injection payload). Such a function cannot be exploited through its
    arguments no matter how capable the harness is, so a detector flagging it is
    a false positive, NOT a harness construction gap. Conservative on purpose: a
    numeric/object/unannotated parameter (a possible vector the harness simply
    doesn't inject yet) is left as a genuine CONSTRUCTION gap, not reclassified."""
    src = rec.get("ground_truth_source") if use == "ground_truth" else rec.get("generated_source")
    if not src:
        return False
    try:
        params, _ = _parse_params(src)
    except Exception:  # noqa: BLE001 -- can't parse -> don't reclassify
        return False
    nonself = [p for p in params if p.get("name") not in ("self", "cls")]
    if not nonself:
        return True
    return all(_looks_boolean(p.get("name"), p.get("annotation")) for p in nonself)


def _exec_cwe(verdicts: list) -> str | None:
    for v in verdicts:
        if getattr(v, "verdict", None) == "TRIGGERED" and getattr(v, "cwe", None):
            return v.cwe
    return None


def _construction_error(verdicts: list) -> str | None:
    for v in verdicts:
        if getattr(v, "error", None):
            return v.error
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--use", default="ground_truth", choices=["generated", "ground_truth"])
    ap.add_argument("--detection", type=Path, default=None,
                    help="detection_results.json to read LLM verdicts from "
                         "(default: <candidates>/detection_results.json)")
    ap.add_argument("--crafter", required=True,
                    help="model that proposes payloads -- choose it OUTSIDE the detector "
                         "panel you grade, to avoid self-grading")
    ap.add_argument("--max-agent", type=int, default=4,
                    help="bounded crafting attempts per (param, cwe) (default 4)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--base-url", default=os.environ.get("OPENSOURCE_BASE_URL",
                                                         "http://localhost:11434/v1"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Make the mined repo's vendored source importable (see mine_prompts.py's
    # _repo_src vendoring): add it to sys.path BEFORE any candidate runs, so the
    # forked execution children inherit it and the REAL package imports as a
    # package (relative imports resolve) even when the repo isn't pip-installable.
    # This is what lets class-method candidates build a real `self` instead of a
    # method-less shell -- the dominant construction-failure cause.
    _vendor = (args.candidates if args.candidates.is_dir()
               else args.candidates.parent) / "_repo_src"
    from auto_harness import vendor_sys_path_roots
    _roots = vendor_sys_path_roots(_vendor)
    for _r in _roots:                 # append so repo-local modules don't shadow stdlib/installed
        if _r not in sys.path:
            sys.path.append(_r)
    if _roots:
        print(f"[importability] vendored repo source on sys.path: {len(_roots)} root(s) under {_vendor}",
              file=sys.stderr)

    det_path = args.detection or ((args.candidates if args.candidates.is_dir()
                                   else args.candidates.parent) / "detection_results.json")
    det = json.loads(det_path.read_text())
    det_records = det["records"]

    samples = _rebuild_samples(args.candidates, args.use)
    if len(samples) != len(det_records):
        raise SystemExit(
            f"candidate set ({len(samples)}) and detection records ({len(det_records)}) "
            f"differ -- the candidates dir changed since detection ran. Re-run run_detection.py "
            f"first so the two line up.")

    from generate_batch import make_caller
    crafter = make_caller(args.base_url, args.crafter)

    print(f"RECYCLE | crafter={args.crafter} (independent of detectors) | use={args.use} "
          f"| max_agent={args.max_agent}", file=sys.stderr)

    out_records = []
    for s, det_rec in zip(samples, det_records):
        rec, slot = s["record"], s["slot"]
        fn = rec.get("function", "?")
        # sanity: positions must refer to the same function
        if det_rec.get("function_id") not in (None, fn):
            raise SystemExit(f"alignment check failed: detection says "
                             f"{det_rec.get('function_id')!r} at a position where candidates "
                             f"say {fn!r}. Re-run run_detection.py.")

        flagged, llm_cwes = _flagged(det_rec)
        if not flagged:
            continue  # only the LLM-flagged functions enter the recycle

        prior = det_rec.get("exec_truth", "UNRESOLVED")  # VULN / SAFE / UNRESOLVED

        verdicts = run_candidate(rec, use=args.use)
        params = _target_params(verdicts)
        exec_cwe = _exec_cwe(verdicts)

        # CWEs to attempt: the LLM's claim first, then execution's own if any,
        # keeping only classes we have a Tier-2 strategy for.
        cwe_order = [c for c in (llm_cwes + ([exec_cwe] if exec_cwe else []))
                     if c in STRATEGIES]
        seen: set = set()
        cwe_order = [c for c in cwe_order if not (c in seen or seen.add(c))]

        row = {"function": fn, "library": det_rec.get("library_id") or rec.get("repo"),
               "slot": slot, "prior_exec": prior, "llm_cwes": llm_cwes,
               "params_tried": params, "crafter": args.crafter,
               "attempts": [], "confirmed": False}

        if not cwe_order:
            row["outcome"] = "SKIPPED_NO_STRATEGY"
            row["note"] = f"LLM CWE(s) {llm_cwes} have no Tier-2 strategy"
            out_records.append(row)
            _log(row); continue
        if not params:
            # Distinguish a genuine harness construction gap from a detector
            # FALSE POSITIVE. A flagged function with NO injectable input channel
            # at all -- no non-self parameter, or only boolean parameters (which
            # cannot carry an injection payload) -- cannot be exploited via its
            # arguments no matter how good the harness is (e.g. sqlite-utils
            # `schema`/`triggers`/`view_names`: no params, fixed SQL). Counting
            # those as COVERAGE_GAP overstates the harness's blind spot and hides
            # the detector's error; they are detector false positives.
            if _no_input_channel(rec, args.use):
                row["outcome"] = "STILL_NOT_CONFIRMED"
                row["diag_bucket"] = "NO_INPUT_CHANNEL"
                row["note"] = ("flagged function exposes no injectable input "
                               "(no non-boolean parameter) -- detector false positive")
            else:
                row["outcome"] = "CONSTRUCTION_FAILED"
                row["note"] = _construction_error(verdicts) or "no injectable parameter found"
            out_records.append(row)
            _log(row); continue

        hit = None
        for cwe in cwe_order:
            for param in params:
                res = attempt_exploit(rec, param, cwe, use=args.use,
                                      agent_call=crafter, max_agent=args.max_agent,
                                      timeout=args.timeout)
                row["attempts"].append({"cwe": cwe, "param": param,
                                        "confirmed": res.get("confirmed"),
                                        "via": res.get("via"),
                                        "payload": res.get("payload"),
                                        "proof": res.get("proof"),
                                        "diag_bucket": res.get("diag_bucket")})
                if res.get("confirmed"):
                    hit = {"cwe": cwe, "param": param, "payload": res.get("payload"),
                           "proof": res.get("proof"), "via": res.get("via"),
                           "sink": res.get("sink")}
                    break
            if hit:
                break

        if hit:
            row["confirmed"] = True
            row["hit"] = hit
            row["outcome"] = ("RECOVERED" if prior in ("SAFE", "UNRESOLVED")
                              else "CONFIRMED_EXPLOIT")
        else:
            # furthest root cause any attempt reached
            buckets = [a["diag_bucket"] for a in row["attempts"] if a.get("diag_bucket")]
            row["outcome"] = "STILL_NOT_CONFIRMED"
            row["diag_bucket"] = (Counter(buckets).most_common(1)[0][0] if buckets else "NO_DIAG")
        out_records.append(row)
        _log(row)

    _report(out_records)

    out = args.out or (det_path.parent / "recycle_results.json")
    out.write_text(json.dumps({"crafter": args.crafter, "use": args.use,
                               "max_agent": args.max_agent, "records": out_records}, indent=2))
    print(f"\nFull auditable recycle log: {out}", file=sys.stderr)


def _log(row: dict) -> None:
    tag = {"RECOVERED": "** RECOVERED", "CONFIRMED_EXPLOIT": "** CONFIRMED-EXPLOIT",
           "STILL_NOT_CONFIRMED": "   not confirmed", "CONSTRUCTION_FAILED": "   construction failed",
           "SKIPPED_NO_STRATEGY": "   skipped"}.get(row["outcome"], row["outcome"])
    extra = row.get("diag_bucket") or row.get("note") or (row.get("hit") or {}).get("cwe", "")
    print(f"  {tag:22} {row['function']:>22}  prior={row['prior_exec']:10} {extra}",
          file=sys.stderr, flush=True)


def _report(records: list) -> None:
    print("\n" + "=" * 70)
    print("VULCAN RECYCLE  (LLM-flagged functions re-checked by execution)")
    print("=" * 70)
    n = len(records)
    recovered = [r for r in records if r["outcome"] == "RECOVERED"]
    exploited = [r for r in records if r["outcome"] == "CONFIRMED_EXPLOIT"]
    unconf = [r for r in records if r["outcome"] == "STILL_NOT_CONFIRMED"]
    constr = [r for r in records if r["outcome"] == "CONSTRUCTION_FAILED"]
    noskill = [r for r in records if r["outcome"] == "SKIPPED_NO_STRATEGY"]
    print(f"LLM-flagged functions re-checked: {n}")
    print(f"  RECOVERED (exec missed it, crafted payload confirms): {len(recovered)}")
    print(f"  CONFIRMED-EXPLOIT (already triggered, now proven):     {len(exploited)}")
    print(f"  STILL NOT CONFIRMED (honest attempt, oracle silent):   {len(unconf)}")
    print(f"  construction failed (could not execute at all):        {len(constr)}")
    print(f"  skipped (no Tier-2 strategy for claimed CWE):          {len(noskill)}")

    if recovered or exploited:
        print("\n-- confirmed by execution after recycle --")
        for r in recovered + exploited:
            h = r.get("hit", {})
            print(f"   {r['outcome']:18} {r['library']}:{r['function']}  "
                  f"{h.get('cwe')} via {h.get('via')}  param={h.get('param')}")
        by_cwe = Counter((r.get('hit') or {}).get('cwe') for r in recovered + exploited)
        print("   by CWE: " + ", ".join(f"{k}:{v}" for k, v in by_cwe.items()))

    if unconf:
        print("\n-- still not confirmed, by root cause (what to fix next) --")
        for bucket, c in Counter(r.get("diag_bucket") for r in unconf).most_common():
            print(f"   {c:3}  {bucket}")

    if constr:
        print("\n-- could not execute (harness construction gap, not LLM-recoverable yet) --")
        for r in constr[:12]:
            print(f"   {r['function']:>22}  {r.get('note')}")
    print("\n" + "=" * 70)
    print("Execution adjudicates. A flagged function the oracle never fired stays")
    print("NOT_CONFIRMED; it is never counted vulnerable on the detector's word.")


if __name__ == "__main__":
    main()

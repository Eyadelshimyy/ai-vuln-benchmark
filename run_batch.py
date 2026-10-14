#!/usr/bin/env python3
"""VULCAN multi-repo batch: detect -> recycle across every mined repo, then
aggregate the outcomes into one benchmark-scale table.

This turns the per-repo case study into a benchmark result. For each repo under
--repos-dir it ensures a detection pass exists, runs the recycle, then rolls the
per-function outcomes up into four execution-grounded categories:

  CONFIRMED      execution proved it exploitable (CONFIRMED_EXPLOIT or RECOVERED)
  REFUTED        ran, could not be exploited -> the detector's flag is a false
                 positive (no sink reached, argv-not-shell, taint absent, etc.)
  ORACLE_GAP     reached the sink / evidence but the oracle could not confirm
                 -> likely real, fixable with a better payload or observable
  COVERAGE_GAP   could not be driven or constructed in live mode -> a harness
                 coverage boundary, not a verdict

The CATEGORY map below is the taxonomy; it is deliberately explicit so it can be
defended and adjusted in the thesis. Edit it there, in one place.

Usage (from the repo root, with Ollama serving):
  export OPENSOURCE_BASE_URL=http://localhost:11434/v1
  export PYTHONPATH=/tmp/vela:$PYTHONPATH          # or wherever the repos import from
  python3 run_batch.py \
      --detectors qwen2.5-coder:7b \
      --crafter   qwen2.5-coder:7b \
      --use ground_truth --max-agent 4

  # already have detection_results.json in each repo? detection is skipped
  # automatically; pass --redetect to force it, or --repos a,b to limit the set.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# diag_bucket (or outcome) -> category. Keep every known label here; an
# unmapped label is surfaced as UNKNOWN so new buckets never pass silently.
CATEGORY = {
    # confirmed
    "CONFIRMED_EXPLOIT": "CONFIRMED",
    "RECOVERED": "CONFIRMED",
    # refuted -- ran, not exploitable: the detector was wrong
    "EXECUTED_NO_SINK": "REFUTED",
    "CMD_ARGV_NOT_SHELL": "REFUTED",
    "SINK_REACHED_INPUT_ABSENT": "REFUTED",
    "INPUT_NOT_IN_SQL": "REFUTED",
    # flagged function has no injectable input channel at all (no non-boolean
    # parameter) -> unexploitable by construction -> a detector false positive
    "NO_INPUT_CHANNEL": "REFUTED",
    "PATH_SINK_INPUT_BOUNDED": "REFUTED",
    "INPUT_IN_SQL_RAN_SAFE": "REFUTED",
    # oracle gap -- evidence reached the sink, confirmation just missed it
    "PATH_TRAVERSAL_NO_OBSERVABLE": "ORACLE_GAP",
    "CMD_SHELL_NO_EFFECT": "ORACLE_GAP",
    "PROOF_EVALUATED": "ORACLE_GAP",
    "PROOF_CONN_MISMATCH": "ORACLE_GAP",
    "INPUT_IN_SQL_PAYLOAD_MISMATCH": "ORACLE_GAP",
    "INPUT_IN_SQL_NO_PROOF": "ORACLE_GAP",
    # coverage gap -- could not drive / construct the candidate live
    "LIVE_RUN_FAILED": "COVERAGE_GAP",
    "NO_DIAG": "COVERAGE_GAP",
    "NOT_ENTERED": "COVERAGE_GAP",
    "CONSTRUCTION_FAILED": "COVERAGE_GAP",
    # not applicable
    "SKIPPED_NO_STRATEGY": "SKIPPED",
}
CATEGORIES = ["CONFIRMED", "REFUTED", "ORACLE_GAP", "COVERAGE_GAP", "SKIPPED", "UNKNOWN"]


def _label(row: dict) -> str:
    """The single label for a recycle row: its outcome, or the diag bucket for a
    not-confirmed row (that's where the root cause lives)."""
    outcome = row.get("outcome")
    if outcome in ("CONFIRMED_EXPLOIT", "RECOVERED", "CONSTRUCTION_FAILED", "SKIPPED_NO_STRATEGY"):
        return outcome
    if outcome == "STILL_NOT_CONFIRMED":
        return row.get("diag_bucket") or "NO_DIAG"
    return outcome or "UNKNOWN"


def aggregate_one(recycle_path: Path) -> dict:
    """Roll one repo's recycle_results.json into category + label counts."""
    data = json.loads(recycle_path.read_text())
    records = data.get("records", [])
    labels = Counter(_label(r) for r in records)
    cats = Counter()
    for lbl, n in labels.items():
        cats[CATEGORY.get(lbl, "UNKNOWN")] += n
    return {"flagged": len(records), "labels": dict(labels), "categories": dict(cats)}


class StageError(RuntimeError):
    """A detection/recycle step failed for one repo -- caught per repo so the
    batch continues and every problem is collected, not just the first."""


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise StageError(f"exit {r.returncode}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos-dir", type=Path, default=Path("calibration/mined_prompts"))
    ap.add_argument("--repos", default=None,
                    help="comma-separated repo names to limit to (default: all under --repos-dir)")
    ap.add_argument("--detectors", default=None,
                    help="comma-separated judge model(s) for run_detection (needed only "
                         "where detection_results.json is missing or --redetect is set)")
    ap.add_argument("--crafter", default="qwen2.5-coder:7b")
    ap.add_argument("--use", default="ground_truth", choices=["generated", "ground_truth"])
    ap.add_argument("--max-agent", type=int, default=4)
    ap.add_argument("--samples", type=int, default=5, help="detection majority-vote samples")
    ap.add_argument("--rule", default="majority", choices=["majority", "any", "unanimous_safe"],
                    help="consortium vote rule across detectors: 'majority' (>50%% agree -- with "
                         "2 models this means BOTH), 'any' (flag if EITHER flags -> higher recall), "
                         "'unanimous_safe' (safe only if all agree safe)")
    ap.add_argument("--redetect", action="store_true", help="re-run detection even if present")
    ap.add_argument("--out", type=Path, default=Path("calibration/batch_aggregate.json"))
    args = ap.parse_args()

    if not args.repos_dir.is_dir():
        raise SystemExit(f"--repos-dir not found: {args.repos_dir}")
    names = ([n.strip() for n in args.repos.split(",")] if args.repos
             else sorted(p.name for p in args.repos_dir.iterdir() if p.is_dir()))

    per_repo: dict[str, dict] = {}
    skipped_repos: list[str] = []
    failed_repos: dict[str, str] = {}   # repo -> why it failed (collected, not fatal)
    for name in names:
        cand = args.repos_dir / name
        if not cand.is_dir():
            print(f"  skip {name}: not a directory", file=sys.stderr)
            continue
        det = cand / "detection_results.json"

        def _detect():
            _run([sys.executable, "run_detection.py", "--candidates", str(cand),
                  "--use", args.use, "--detectors", args.detectors,
                  "--samples", str(args.samples), "--rule", args.rule])

        def _recycle():
            _run([sys.executable, "run_recycle.py", "--candidates", str(cand),
                  "--use", args.use, "--detection", str(det),
                  "--crafter", args.crafter, "--max-agent", str(args.max_agent)])

        try:
            need_detect = args.redetect or not det.exists()
            if need_detect and not args.detectors:
                print(f"  SKIP {name}: no detection_results.json and no --detectors given",
                      file=sys.stderr)
                skipped_repos.append(name)
                continue
            if need_detect:
                _detect()

            try:
                _recycle()
            except StageError as e_rec:
                # Most common cause: a STALE detection file (candidate set changed
                # since it was generated, so recycle's alignment guard trips). If
                # we have detectors and did not just detect, re-detect this one
                # repo and retry once -- so the batch self-heals instead of
                # leaving a gap. A second failure propagates and is recorded.
                if args.detectors and not need_detect:
                    print(f"  {name}: recycle failed ({e_rec}); re-detecting this repo and retrying once",
                          file=sys.stderr)
                    _detect()
                    _recycle()
                else:
                    raise

            recycle_out = cand / "recycle_results.json"
            if not recycle_out.exists():
                raise StageError("recycle produced no results file")
            per_repo[name] = aggregate_one(recycle_out)
        except StageError as e:
            # Record and keep going -- the point of a batch is to surface every
            # repo's problems in one pass, not to stop at the first.
            print(f"  FAILED {name}: {e}", file=sys.stderr)
            failed_repos[name] = str(e)
            continue
        except Exception as e:  # noqa: BLE001 -- unexpected, but never abort the batch
            print(f"  FAILED {name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed_repos[name] = f"{type(e).__name__}: {e}"
            continue

    # ---- roll up ----
    overall_cats = Counter()
    overall_labels = Counter()
    total_flagged = 0
    for name, agg in per_repo.items():
        total_flagged += agg["flagged"]
        for c, n in agg["categories"].items():
            overall_cats[c] += n
        for lbl, n in agg["labels"].items():
            overall_labels[lbl] += n

    summary = {
        "repos": per_repo, "skipped_repos": skipped_repos,
        "failed_repos": failed_repos,
        "total_flagged": total_flagged,
        "overall_categories": dict(overall_cats),
        "overall_labels": dict(overall_labels),
        "crafter": args.crafter, "use": args.use, "max_agent": args.max_agent,
    }
    args.out.write_text(json.dumps(summary, indent=2))

    # ---- print table ----
    print("\n" + "=" * 70)
    print("VULCAN BATCH AGGREGATE")
    print("=" * 70)
    hdr = f"{'repo':20} {'flagged':>7} " + " ".join(f"{c[:9]:>9}" for c in CATEGORIES)
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(per_repo):
        agg = per_repo[name]
        cells = " ".join(f"{agg['categories'].get(c, 0):>9}" for c in CATEGORIES)
        print(f"{name:20} {agg['flagged']:>7} {cells}")
    print("-" * len(hdr))
    tot = " ".join(f"{overall_cats.get(c, 0):>9}" for c in CATEGORIES)
    print(f"{'TOTAL':20} {total_flagged:>7} {tot}")
    if skipped_repos:
        print(f"\nskipped (no detection, no --detectors): {', '.join(skipped_repos)}")
    if failed_repos:
        print("\nFAILED repos (problems to fix -- collected, did not abort the batch):")
        for name, why in failed_repos.items():
            print(f"  {name}: {why}")
    if overall_cats.get("UNKNOWN"):
        unk = [l for l in overall_labels if CATEGORY.get(l, 'UNKNOWN') == 'UNKNOWN']
        print(f"\nUNKNOWN labels (add to CATEGORY map): {', '.join(sorted(unk))}")
    print(f"\nFull aggregate written to: {args.out}")


if __name__ == "__main__":
    main()

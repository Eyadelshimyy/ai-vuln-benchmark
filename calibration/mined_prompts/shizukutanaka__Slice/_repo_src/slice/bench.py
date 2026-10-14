"""slice.bench: estimator benchmark + regression gate.

Times `estimate()` and scores it against `evaluate.draw_case` ground
truth (detection rate, mean positional error, OKS). Usable as a
module and as a CLI gate:

    python -m slice.bench            # report + exit 1 on gate fail
    python -m slice.bench --json     # machine-readable report only

Accuracy thresholds are the gate (deterministic); timing is reported
but never gated — CI machines vary too much for a wall-clock budget.
"""

from __future__ import annotations

import json
import sys
import time
from typing import List, Optional, Tuple

from . import evaluate, oks
from .pose import HeuristicPoseEstimator
from .skeleton import Joint, OBSERVED, Skeleton

# gate: deterministic accuracy thresholds at *measured* current
# levels — they pin today's accuracy as a regression detector, they
# are not accuracy claims (mid-chain joints sit ~30px off the
# fixture truth because truth marks silhouette edges while the
# estimator places anatomical centers)
MIN_DETECTION = 0.9
MIN_OBSERVED = 0.8
MAX_MEAN_ERROR_PX = 20.0
MIN_OKS = 0.5


def _truth_skeleton(truth, w, h) -> Skeleton:
    # truth fixtures claim to be the observed reference — say so
    return Skeleton(w, h, {n: Joint(n, x, y, 1.0, OBSERVED,
                                    "ground truth fixture")
                           for n, (x, y) in truth.items()})


def run(cases: Optional[List] = None, repeats: int = 1) -> dict:
    """Measure latency + accuracy over ground-truth fixture cases."""
    est = HeuristicPoseEstimator()
    if cases is None:
        cases = [evaluate.draw_case(),
                 evaluate.draw_case(width=240, height=320)]
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        for bmp, _truth in cases:
            est.estimate(bmp)
    ms = (time.perf_counter() - t0) * 1000 / (len(cases) * max(1, repeats))

    metrics = evaluate.evaluate([(b, t) for b, t in cases], est)
    oks_vals = []
    for bmp, truth in cases:
        skel = est.estimate(bmp)
        gt = _truth_skeleton(truth, skel.image_width, skel.image_height)
        v = oks.oks(gt, skel)
        if v is not None:
            oks_vals.append(v)
    return {
        "cases": len(cases),
        "repeats": max(1, repeats),
        "ms_per_estimate": round(ms, 1),
        "detection_rate": metrics["detection_rate"],
        "observed_rate": metrics["observed_rate"],
        "mean_error_px": metrics["mean_error_px"],
        "mean_oks": round(sum(oks_vals) / len(oks_vals), 4)
        if oks_vals else None,
    }


def gate(report: dict) -> Tuple[bool, List[str]]:
    """Deterministic accuracy gate; timing is reported, not gated."""
    fails = []
    if report["detection_rate"] < MIN_DETECTION:
        fails.append(f"detection {report['detection_rate']} "
                     f"< {MIN_DETECTION}")
    if report["observed_rate"] < MIN_OBSERVED:
        fails.append(f"observed {report['observed_rate']} "
                     f"< {MIN_OBSERVED}")
    if report["mean_error_px"] > MAX_MEAN_ERROR_PX:
        fails.append(f"mean_error {report['mean_error_px']}px "
                     f"> {MAX_MEAN_ERROR_PX}px")
    if report["mean_oks"] is not None and report["mean_oks"] < MIN_OKS:
        fails.append(f"oks {report['mean_oks']} < {MIN_OKS}")
    return not fails, fails


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    report = run()
    ok, fails = gate(report)
    if "--json" in argv:
        print(json.dumps(report, indent=2))
    else:
        print(f"estimate: {report['ms_per_estimate']}ms/image "
              f"({report['cases']} cases x{report['repeats']})")
        print(f"detection={report['detection_rate']} "
              f"observed={report['observed_rate']} "
              f"err={report['mean_error_px']}px "
              f"oks={report['mean_oks']}")
    for f in fails:
        print(f"FAIL {f}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

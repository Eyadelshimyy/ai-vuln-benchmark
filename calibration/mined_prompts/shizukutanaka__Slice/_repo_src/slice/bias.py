"""Per-joint systematic-error profile — bias vs noise.

Total error splits into two different problems:

- **systematic bias** — the estimator lands consistently in the same
  wrong direction (e.g. wrist always +4px left). Correctable: a
  constant offset would recover most of it.
- **random error** — scatter around the truth with no preferred
  direction. Not correctable by calibration; only better evidence
  helps.

`profile(pairs)` computes, per joint, the mean signed error vector
(bias) and the residual spread (noise) across evaluation cases, then
labels the joint:

- `systematic` — |mean vector| > half the mean absolute error: most
  of the error points the same way
- `unbiased` — mean vector near zero relative to spread: error is
  scatter, not shift
- `insufficient` — fewer than 2 observed samples: refuses to claim

pairs: iterable of `(joints, truth)` where joints is a name→Joint
mapping and truth is a name→(x, y) mapping (evaluate.draw_case
returns exactly this). Only observed joints contribute — feeding a
predicted joint would measure the prior, not the evidence.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from .skeleton import OBSERVED, Joint

Pairs = Iterable[Tuple[Dict[str, Joint], Dict[str, Tuple[float, float]]]]
BIAS_RATIO = 0.5  # |mean error vec| > half of mean |error| → systematic
MIN_SAMPLES = 2


def profile(pairs: Pairs) -> Dict:
    """Compute per-joint bias profile across evaluated cases."""
    errs: Dict[str, List[Tuple[float, float]]] = {}
    n_cases = 0
    for joints, truth in pairs:
        n_cases += 1
        for name, (tx, ty) in truth.items():
            j = joints.get(name)
            if j is None or j.state != OBSERVED:
                continue
            errs.setdefault(name, []).append((j.x - tx, j.y - ty))

    out: Dict[str, Dict] = {}
    for name, vecs in sorted(errs.items()):
        n = len(vecs)
        if n < MIN_SAMPLES:
            out[name] = {
                "n": n, "verdict": "insufficient",
                "reason": "fewer than %d observed samples" % MIN_SAMPLES,
            }
            continue
        mean_dx = sum(v[0] for v in vecs) / n
        mean_dy = sum(v[1] for v in vecs) / n
        mean_vec = math.hypot(mean_dx, mean_dy)
        mean_err = sum(math.hypot(*v) for v in vecs) / n
        # residual spread once the bias is removed — the noise floor
        residual = sum(
            math.hypot(v[0] - mean_dx, v[1] - mean_dy)
            for v in vecs) / n
        systematic = mean_vec > BIAS_RATIO * mean_err
        out[name] = {
            "n": n,
            "verdict": "systematic" if systematic else "unbiased",
            "bias_vector": [round(mean_dx, 3), round(mean_dy, 3)],
            "bias_px": round(mean_vec, 3),
            "mean_error_px": round(mean_err, 3),
            "residual_px": round(residual, 3),
            # what a constant-offset correction would recover
            "recoverable_px": round(mean_err - residual, 3),
        }

    systematic = [n for n, r in out.items()
                  if r.get("verdict") == "systematic"]
    worst = max(
        (r for r in out.values() if "mean_error_px" in r),
        key=lambda r: r["mean_error_px"], default=None)
    return {
        "n_cases": n_cases,
        "joints": out,
        "systematic": systematic,
        "worst_joint": worst and
            {"name": [n for n, r in out.items() if r is worst][0],
             "mean_error_px": worst["mean_error_px"]},
    }


def correction(result: Dict, name: str) -> Tuple[float, float]:
    """(dx, dy) that subtracts this joint's measured bias; (0,0) if
    not systematic — applying an offset to scatter makes it worse."""
    r = result["joints"].get(name) or {}
    if r.get("verdict") != "systematic":
        return (0.0, 0.0)
    return tuple(r["bias_vector"])  # type: ignore

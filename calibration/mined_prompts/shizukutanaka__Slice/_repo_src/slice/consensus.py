"""Consensus estimation — vote across parameter jitter, not one run.

A single `estimate` commits to one threshold and one working
resolution. Its answer is real, but fragile at the margins: a
joint that only exists at `bg_threshold=40` and vanishes at 30/50
was never strongly supported. `consensus` runs a fixed panel of
variants (threshold ±25%, working resolution ±25%), takes the
median position of every joint observed in a majority of runs, and
marks the rest `disputed` instead of guessing.

Outputs differ from a plain `estimate` only in honesty:
- joint position = median over runs (robust to one bad variant)
- joint confidence = base confidence × observed fraction — a joint
  seen in 3/5 runs can't claim 0.8
- `disputed` lists joints that failed the majority — reported, not
  dropped, with the per-joint spread that caused the dissent
- run coordinates are normalized per variant's own downscaled
  frame, so different `max_dim` runs vote in the same space
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .anatomy import DEFAULT_MODEL
from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator
from .skeleton import OBSERVED, PREDICTED, Joint, Skeleton

DISPUTE_MIN_RUNS = 2    # fewer observations than this → disputed
DISPUTE_SPREAD_PX = 8.0  # median-to-farthest gap above this → disputed


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _variants(est: HeuristicPoseEstimator) -> List[HeuristicPoseEstimator]:
    t, m = est.bg_threshold, est.max_dim
    # the panel jitters threshold and resolution only — the caller's
    # mask profile (adaptive/shadow/clean) must be identical across
    # runs or the median measures profile differences it never
    # disclosed
    flags = dict(adaptive=est.adaptive,
                 reject_shadow=est.reject_shadow, clean=est.clean)
    return [
        est,
        HeuristicPoseEstimator(max_dim=m,
                               bg_threshold=max(1, int(t * 0.75)), **flags),
        HeuristicPoseEstimator(max_dim=m,
                               bg_threshold=max(1, int(t * 1.25)), **flags),
        HeuristicPoseEstimator(max_dim=max(64, int(m * 0.75)),
                               bg_threshold=t, **flags),
        HeuristicPoseEstimator(max_dim=int(m * 1.25),
                               bg_threshold=t, **flags),
    ]


def consensus(bmp: Bitmap,
              estimator: Optional[HeuristicPoseEstimator] = None,
              model: str = DEFAULT_MODEL) -> Dict:
    """Median-vote skeleton over a parameter panel.

    Returns {skeleton, runs, disputed, spread, state, basis}. The
    skeleton lives in the *first* variant's downscaled frame —
    positions from other variants are rescaled into it.
    """
    est = estimator or HeuristicPoseEstimator()
    variants = _variants(est)
    runs: List[Tuple[HeuristicPoseEstimator, Skeleton]] = [
        (v, v.estimate(bmp, model)) for v in variants]
    base = runs[0][1]
    sk = Skeleton(image_width=base.image_width,
                  image_height=base.image_height)
    sk.orientation = dict(base.orientation)
    sk.body_model = dict(base.body_model)

    names = set()
    for _, s in runs:
        names.update(s.joints)

    n_runs = len(runs)
    needed = n_runs // 2 + 1
    disputed: List[Dict] = []
    spreads: List[float] = []
    for name in sorted(names):
        obs = []
        for _, s in runs:
            j = s.get(name)
            if j is not None and j.state == OBSERVED:
                # rescale into the base frame
                obs.append((j.x / s.image_width * base.image_width,
                            j.y / s.image_height * base.image_height,
                            j.confidence))
        if not obs:
            continue
        mx, my = _median([p[0] for p in obs]), _median([p[1] for p in obs])
        spread = max(math.hypot(x - mx, y - my) for x, y, _ in obs)
        base_j = base.get(name)
        conf = _median([p[2] for p in obs]) * len(obs) / n_runs
        majority = len(obs) >= needed
        if (not majority or len(obs) < DISPUTE_MIN_RUNS
                or spread > DISPUTE_SPREAD_PX):
            disputed.append({"joint": name,
                             "runs": len(obs),
                             "spread_px": round(spread, 2)})
        j = Joint(name, round(mx, 2), round(my, 2), round(conf, 3),
                  OBSERVED if majority else PREDICTED,
                  "consensus %d/%d runs, spread %.1fpx%s"
                  % (len(obs), n_runs, spread,
                     "" if majority else "; no majority"))
        sk.set(j)
        spreads.append(spread)

    for name, j in base.joints.items():
        # carry predicted joints from the base run when the consensus
        # did not observe them — flagged, not hidden
        if sk.get(name) is None:
            sk.set(Joint(name, j.x, j.y, j.confidence, PREDICTED,
                         "predicted by base run; no consensus observed"))

    return {
        "skeleton": sk,
        "runs": [{"max_dim": v.max_dim,
                  "bg_threshold": v.bg_threshold,
                  "joints": len(s.joints)} for v, s in runs],
        "disputed": disputed,
        "mean_spread_px": round(sum(spreads) / len(spreads), 2)
        if spreads else None,
        # an all-predicted skeleton is not observation
        "state": ("observed" if any(j.state == OBSERVED
                                    for j in sk.joints.values())
                  else "predicted" if sk.joints else "failed"),
        "basis": "median over %d parameter variants" % n_runs,
    }

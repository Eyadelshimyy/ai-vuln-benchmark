"""Staged fallback estimation — try harder before reporting 'nobody'.

`estimate` is deliberately strict: when no person-sized foreground
survives its gates it returns an empty `Skeleton`, and the honest
answer is "no person detected". But two recoverable failures hide
in that answer — the largest component wasn't a person (the person
sat in component #2), and the whole silhouette fell just under the
background-distance gate (low-contrast clothing on a similar wall).

`recover` walks a fixed ladder, cheapest first:

1. `primary` — the normal `estimate` result.
2. `component_retry` — `estimate_multi` gives every large component
   its own pass instead of only the biggest one.
3. `relaxed_threshold` — a second estimator with half the
   `bg_threshold`, catching figures too close to the background.
   The caller's profile flags (`adaptive`/`reject_shadow`/`clean`)
   carry over — relaxing the colour gate must not silently relax
   the shadow and morphology gates too.

Every recovered skeleton's joints get `"; recovered: <method>"`
appended to `basis`, so a downstream reader can tell a recovered
skeleton from a first-pass one at the joint level — the same
provenance mechanism the splitter uses. A fully failed attempt
returns `state: "failed"` with an empty skeleton; it never
fabricates one.
"""

from __future__ import annotations

from typing import Dict, Optional

from .anatomy import DEFAULT_MODEL
from .bitmap import Bitmap
from .pose import HeuristicPoseEstimator
from .skeleton import Skeleton

LADDER = ("primary", "component_retry", "relaxed_threshold")
RELAXED_FACTOR = 0.5


def _tag(skel: Skeleton, method: str) -> None:
    for j in skel.joints.values():
        j.basis = (j.basis + "; " if j.basis else "") + \
            "recovered: " + method


def recover(bmp: Bitmap, estimator: Optional[HeuristicPoseEstimator] = None,
            model: str = DEFAULT_MODEL, top_k: int = 4) -> Dict:
    """Try the ladder; return {skeleton, method, attempts, state, basis}.

    `state` is "observed" when a non-empty skeleton was found at any
    rung, "failed" when every rung came back empty.
    """
    est = estimator or HeuristicPoseEstimator()

    skel = est.estimate(bmp, model)
    if skel.joints:
        return {"skeleton": skel, "method": "primary",
                "attempts": 1, "state": "observed",
                "basis": "primary estimate"}

    for i, s in enumerate(est.estimate_multi(bmp, model, top_k=top_k)):
        if s.joints:
            _tag(s, "component_retry #%d" % i)
            return {"skeleton": s, "method": "component_retry",
                    "attempts": 2 + i, "state": "observed",
                    "basis": "non-largest foreground component"}

    relaxed = HeuristicPoseEstimator(
        max_dim=est.max_dim,
        bg_threshold=max(8, int(est.bg_threshold * RELAXED_FACTOR)),
        adaptive=est.adaptive, reject_shadow=est.reject_shadow,
        clean=est.clean)
    skel = relaxed.estimate(bmp, model)
    if skel.joints:
        _tag(skel, "relaxed_threshold")
        return {"skeleton": skel, "method": "relaxed_threshold",
                "attempts": len(LADDER), "state": "observed",
                "basis": "background-distance gate relaxed to %d"
                         % relaxed.bg_threshold}

    return {"skeleton": skel, "method": "failed",
            "attempts": len(LADDER), "state": "failed",
            "basis": "all rungs of " + "/".join(LADDER) + " empty"}

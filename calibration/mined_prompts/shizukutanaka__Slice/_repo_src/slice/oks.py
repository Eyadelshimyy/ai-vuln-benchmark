"""Object Keypoint Similarity — the standard pose-eval metric.

OKS is how COCO scores pose estimates: per visible ground-truth
joint, exp(-d² / 2·(s·k)²) where d is the distance to the prediction,
s is the object's scale (sqrt of its bounding-box area), and k is a
per-joint tolerance constant — small for high-precision joints
(hips), large for variable ones (wrists). Joints absent from the
reference don't count, so an estimate is never penalized for joints
the reference itself can't see.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .skeleton import Skeleton

# per-joint tolerance (k). Looser where pose varies, tighter at the
# body's stable landmarks. Values follow COCO's spirit, rescaled to
# our joint set.
_KAPPA: Dict[str, float] = {
    "head": 0.079, "neck": 0.072, "chest": 0.062, "pelvis": 0.062,
    "shoulder_l": 0.079, "shoulder_r": 0.079,
    "elbow_l": 0.072, "elbow_r": 0.072,
    "wrist_l": 0.062, "wrist_r": 0.062,
    "hip_l": 0.107, "hip_r": 0.107,
    "knee_l": 0.087, "knee_r": 0.087,
    "ankle_l": 0.089, "ankle_r": 0.089,
    "foot_l": 0.079, "foot_r": 0.079,
}
_DEFAULT_K = 0.08


def _scale(skel: Skeleton) -> Optional[float]:
    pts = [p for p in (skel.point(n) for n in skel.joints) if p]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(xs) - min(xs) + 1
    h = max(ys) - min(ys) + 1
    return math.sqrt(w * h)


def per_joint(gt: Skeleton, est: Skeleton) -> Dict[str, float]:
    """{joint: similarity 0..1} for joints present in `gt`."""
    s = _scale(gt)
    if not s or s < 1e-6:
        return {}
    out: Dict[str, float] = {}
    for name, jg in gt.joints.items():
        je = est.joints.get(name)
        if je is None:
            out[name] = 0.0
            continue
        k = _KAPPA.get(name, _DEFAULT_K)
        d = math.hypot(jg.x - je.x, jg.y - je.y)
        out[name] = round(math.exp(-d * d / (2 * (s * k) ** 2)), 4)
    return out


def oks(gt: Skeleton, est: Skeleton) -> Optional[float]:
    """Mean joint similarity; None when the reference has no joints."""
    pj = per_joint(gt, est)
    if not pj:
        return None
    return round(sum(pj.values()) / len(pj), 4)

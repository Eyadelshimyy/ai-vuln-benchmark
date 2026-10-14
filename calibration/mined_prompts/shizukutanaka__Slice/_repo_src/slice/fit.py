"""Skeleton↔silhouette agreement — does the estimate explain the pixels?

A skeleton can pass every geometric audit and still sit beside the
silhouette it claims to describe — nothing elsewhere checks that
the bones actually lie *inside* the foreground. `fit` closes that
loop: for every mask pixel it measures the distance to the nearest
bone (or joint, whichever is closer), and reports what fraction of
the silhouette the skeleton explains.

This is a self-audit, not a score to optimize: a low `fraction`
means the estimate is under-explaining real evidence (detached
limbs, mis-scaled body, wrong component) and should drop trust in
`observed` joints near the unexplained area.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .landmarks import BONES
from .skeleton import Skeleton

# a foreground pixel counts as explained when this close to a bone
# or joint; roughly the torso half-width at the estimator's working
# resolution — smaller radii read broad body parts as "unexplained"
COVER_RADIUS_PX = 12.0

Segment = Tuple[Tuple[float, float], Tuple[float, float]]


def _segments(skel: Skeleton) -> List[Segment]:
    """Bones between OBSERVED joints — a predicted bone is prior
    fill; letting it cover silhouette pixels would inflate the
    explained fraction with geometry no evidence supports."""
    segs: List[Segment] = []
    for a, b in BONES:
        ja, jb = skel.joints.get(a), skel.joints.get(b)
        if (ja and jb and ja.state == "observed"
                and jb.state == "observed"):
            segs.append(((ja.x, ja.y), (jb.x, jb.y)))
    return segs


def _d_seg(px: float, py: float, seg: Segment) -> float:
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0,
        ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _dist(px: float, py: float, segs: List[Segment],
          joints: List[Tuple[float, float]]) -> float:
    best = float("inf")
    for s in segs:
        best = min(best, _d_seg(px, py, s))
    for (jx, jy) in joints:
        best = min(best, math.hypot(px - jx, py - jy))
    return best


def fit(skel: Skeleton, mask: List[bytearray],
        cover_radius: float = COVER_RADIUS_PX) -> Dict:
    """Skeleton→silhouette agreement over a foreground mask.

    `mask` is the estimator's mask (rows of 0/1 bytes, same
    coordinate frame as `skel` — i.e. the downscaled frame when the
    estimator downscales). Returns the explained fraction, mean and
    worst pixel-to-skeleton distance, and the centroid of the
    unexplained region (where the evidence lies that the skeleton
    ignores). Empty mask → fraction is None, not 1.0: zero evidence
    is not a perfect fit.
    """
    segs = _segments(skel)
    joints = [(j.x, j.y) for j in skel.joints.values()
              if j.state == "observed"]
    if not segs and not joints:
        return {"state": "unmeasurable",
                "basis": "skeleton has no joints or bones",
                "fraction": None}
    fg = explained = 0
    dist_sum = 0.0
    ux = uy = 0.0
    worst = 0.0
    for y, row in enumerate(mask):
        for x, v in enumerate(row):
            if not v:
                continue
            fg += 1
            d = _dist(x + 0.5, y + 0.5, segs, joints)
            dist_sum += d
            if d <= cover_radius:
                explained += 1
            else:
                ux += x + 0.5
                uy += y + 0.5
                worst = max(worst, d)
    if fg == 0:
        return {"state": "unmeasurable",
                "basis": "mask has no foreground pixels",
                "fraction": None}
    unexplained = fg - explained
    return {
        "state": "measured",
        "basis": "mask pixels within %.1fpx of a bone or joint"
                 % cover_radius,
        "cover_radius_px": cover_radius,
        "pixels": fg,
        "explained": explained,
        "fraction": round(explained / fg, 3),
        "mean_dist_px": round(dist_sum / fg, 2),
        "worst_dist_px": round(worst, 2),
        "unexplained_centroid": (
            [round(ux / unexplained, 1), round(uy / unexplained, 1)]
            if unexplained else None),
    }


def verdict(result: Dict, min_fraction: float = 0.6) -> str:
    """'good' | 'poor' | 'unmeasurable' — threshold on explained share."""
    if result["fraction"] is None:
        return "unmeasurable"
    return "good" if result["fraction"] >= min_fraction else "poor"

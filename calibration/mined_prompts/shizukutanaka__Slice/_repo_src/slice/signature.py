"""Pose signature — a fixed-length fingerprint for search and dedup.

Every skeleton becomes a vector of its bone directions (unit vectors
per BONES edge, in a fixed order) plus torso-lean and limb-spread
scalars. Two poses of the same shape produce nearly identical
signatures regardless of image size or framing, so `distance` gives
fast pose-similarity search and near-duplicate detection without
pairwise joint matching.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .landmarks import BONES
from .skeleton import Skeleton


def signature(skel: Skeleton) -> List[float]:
    """2 floats per bone (unit direction) + 4 pose scalars, fixed order."""
    def obs(name):
        j = skel.joints.get(name)
        return (j.x, j.y) if j and j.state == "observed" else None

    vec: List[float] = []
    for a, b in BONES:
        pa, pb = obs(a), obs(b)
        if pa and pb:
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            L = math.hypot(dx, dy) or 1.0
            vec += [dx / L, dy / L]
        else:
            vec += [0.0, 0.0]
    # scalars: torso lean, arm spread (wrist span / body height),
    # leg spread (ankle gap / body height), facing symmetry —
    # like the bone vectors, only observed joints may feed them:
    # a predicted head/neck/pelvis would write prior geometry into
    # the fingerprint as if it were measured
    head = obs("head")
    feet = [p for p in (obs("foot_l"), obs("foot_r")) if p]
    neck, pelvis = obs("neck"), obs("pelvis")
    body_h = (max(f[1] for f in feet) - head[1]) if head and feet else 0.0
    if body_h <= 1e-6:
        # feet missing or the figure is inverted — normalise by the
        # torso unit like compare/dedup do, never raw pixels
        body_h = math.hypot(neck[0] - pelvis[0], neck[1] - pelvis[1]) \
            if neck and pelvis else 0.0
    scale = body_h if body_h > 0 else None
    # no measurable body scale: emit 0.0 (the missing marker), not
    # raw px dressed as a body-height fraction via `else 1.0`
    vec.append((neck[0] - pelvis[0]) / scale
               if neck and pelvis and scale else 0.0)
    wl, wr = obs("wrist_l"), obs("wrist_r")
    vec.append(math.hypot(wl[0] - wr[0], wl[1] - wr[1]) / scale
               if wl and wr and scale else 0.0)
    al, ar = obs("ankle_l"), obs("ankle_r")
    vec.append(math.hypot(al[0] - ar[0], al[1] - ar[1]) / scale
               if al and ar and scale else 0.0)
    sl, sr = obs("shoulder_l"), obs("shoulder_r")
    vec.append(math.hypot(sl[0] - sr[0], sl[1] - sr[1]) / scale
               if sl and sr and scale else 0.0)
    return vec


def distance(a: List[float], b: List[float]) -> Optional[float]:
    """RMS difference of two signatures (0 = identical pose)."""
    if not a or not b or len(a) != len(b):
        return None
    s = sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
    return round(math.sqrt(s), 4)

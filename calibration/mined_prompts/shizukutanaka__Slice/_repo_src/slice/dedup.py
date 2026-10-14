"""Near-duplicate detection — is this document already in the set?

A dataset padded with re-analyses of the same pose is worse than a
smaller honest one. `dedup` normalises each skeleton the same way
(translate pelvis to origin, scale by shoulder->pelvis torso length)
and flags document pairs whose per-joint mean distance falls below
`eps` — the same pose, probably the same shot.

Normalised space is shared across docs of different resolutions, so
a 200px and an 800px render of the same stance still match. Joints
absent — or `predicted`, which are prior fills, not evidence — in
either document are skipped per pair and counted.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .landmarks import JOINTS

_TORSO = ("neck", "pelvis")


def _vec(doc: dict) -> Optional[dict]:
    joints = (doc.get("skeleton") or {}).get("joints") or {}
    p0, p1 = joints.get(_TORSO[0]), joints.get(_TORSO[1])
    if not p0 or not p1:
        return None
    scale = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
    if scale <= 0:
        # coincident torso anchors: the normalised space is
        # undefined — `or 1.0` would silently switch to raw px and
        # the doc is unmeasurable, not 1px-tall
        return None
    return {n: ((j["x"] - p1["x"]) / scale,
                (j["y"] - p1["y"]) / scale)
            for n, j in joints.items()
            if n in JOINTS and j.get("state") != "predicted"}


def distance(a: dict, b: dict) -> Optional[float]:
    """Mean per-joint distance in normalised space; None if unpaired."""
    va, vb = _vec(a), _vec(b)
    if va is None or vb is None:
        return None
    common = [n for n in JOINTS if n in va and n in vb]
    if not common:
        return None
    return sum(math.hypot(va[n][0] - vb[n][0],
                          va[n][1] - vb[n][1])
               for n in common) / len(common)


def dedup(docs: List[dict], eps: float = 0.15) -> dict:
    """Flag near-duplicate pairs among knowledge documents."""
    vecs = {i: _vec(d) for i, d in enumerate(docs)}
    pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if vecs[i] is None or vecs[j] is None:
                continue
            d = _dist(vecs[i], vecs[j])
            if d is not None and d < eps:
                pairs.append({
                    "a": docs[i].get("id"), "b": docs[j].get("id"),
                    "distance": round(d, 4)})
    keep = {p["b"] for p in pairs}
    return {
        "n_docs": len(docs),
        "n_pairs": len(pairs),
        "pairs": sorted(pairs, key=lambda p: p["distance"]),
        "redundant": sorted(keep),
        "eps": eps,
        "note": "distance is mean per-joint gap in torso-normalised "
                "space; only joints present in both docs are counted",
    }


def _dist(va: dict, vb: dict) -> Optional[float]:
    common = [n for n in JOINTS if n in va and n in vb]
    if not common:
        return None
    return sum(math.hypot(va[n][0] - vb[n][0],
                          va[n][1] - vb[n][1])
               for n in common) / len(common)

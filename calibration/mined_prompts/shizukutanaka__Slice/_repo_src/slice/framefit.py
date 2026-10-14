"""slice.framefit: partial-body / truncation hints at the frame edges.

Anatomy says the body's extremities extend outward from the skeleton:
the head continues above the topmost joint, the feet continue below
the ankle/foot joints, the hands extend past the wrists. When the
outermost joint on a side sits flush against the image border, part
of the body may have been cut off by the camera.

This module reports *possibilities*, not verdicts -- a foot planted
exactly on the bottom edge can be a perfectly framed full-body shot.
Each edge gets a separate check so callers can see which direction
looks truncated.

The frame here is the *skeleton* frame (downscaled coordinates), not
the raw image. All assessments are labeled ``"estimated"``.
"""

_JOINTS_AT = {
    "top": ("head", "neck"),
    "bottom": ("ankle_l", "ankle_r", "foot_l", "foot_r"),
    "left": ("wrist_l", "ankle_l", "foot_l"),
    "right": ("wrist_r", "ankle_r", "foot_r"),
}
_MARGINS = {"top": 4, "bottom": 4, "left": 4, "right": 4}


def _extremes(skel):
    """Outermost placed joint per side."""
    w, h = skel.image_width, skel.image_height
    best = {"top": None, "bottom": None, "left": None, "right": None}
    for name, j in skel.joints.items():
        x, y = j.x, j.y
        if best["top"] is None or y < best["top"][1]:
            best["top"] = (name, y)
        if best["bottom"] is None or y > best["bottom"][1]:
            best["bottom"] = (name, y)
        if best["left"] is None or x < best["left"][1]:
            best["left"] = (name, x)
        if best["right"] is None or x > best["right"][1]:
            best["right"] = (name, x)
    return best, w, h


def assess(skel):
    """Estimate which frame edges may have truncated the body.

    Returns ``{"edges": {top/bottom/left/right: {...}},
    "partial": bool, "possibly_truncated": [sides],
    "state": "estimated", ...}`` -- never a hard verdict.
    """
    best, w, h = _extremes(skel)
    edges = {}
    truncated = []
    for side, margin in _MARGINS.items():
        entry = best[side]
        if entry is None:
            edges[side] = {"checked": False}
            continue
        name, pos = entry
        gap = pos if side in ("top", "left") else (h - 1 - pos if side == "bottom" else w - 1 - pos)
        at_edge = gap < margin
        # an edge flush is only suspicious when a body extremity
        # plausibly continues past it -- joints listed in _JOINTS_AT
        extremity = name in _JOINTS_AT[side]
        verdict = "possibly_truncated" if (at_edge and extremity) else "clear"
        edges[side] = {
            "checked": True,
            "joint": name,
            "gap_px": round(gap, 1),
            "at_edge": at_edge,
            "extremity": extremity,
            "verdict": verdict,
        }
        if verdict == "possibly_truncated":
            truncated.append(side)
    n_checked = sum(1 for e in edges.values() if e.get("checked"))
    return {
        "edges": edges,
        "partial": bool(truncated),
        "possibly_truncated": truncated,
        "n_edges_checked": n_checked,
        "assumption": "body extremities extend past the outermost joint",
        "state": "estimated",
        "note": "edge-flush joints are hints, not proof of truncation",
        "basis": "outermost joint distance to each frame edge",
    }

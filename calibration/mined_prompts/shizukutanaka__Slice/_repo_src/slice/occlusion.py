"""Occlusion reasoning — WHY a joint isn't observed.

"Missing" isn't one thing: a joint hidden inside the silhouette
(occluded) is a different claim than one cropped by the frame
(truncated) or simply not rendered (absent). The mask bounds tell
which: inside the foreground region → occluded; past the frame →
truncated; neither → absent. Predicted joints carry a location, so
each also gets checked against the mask.
"""

from __future__ import annotations

from typing import Dict

from .skeleton import Skeleton


def _inside_mask(mask, w: int, h: int, x: float, y: float,
                 radius: int = 6) -> bool:
    """Is there foreground near (x,y)? radius in px."""
    xi, yi = int(x), int(y)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < w and 0 <= ny < h and mask[ny][nx]:
                return True
    return False


def reason(skel: Skeleton, name: str, mask=None, w: int = 0,
           h: int = 0) -> dict:
    """{joint, state, reason, in_frame, in_foreground}."""
    j = skel.joints.get(name)
    out = {"joint": name, "state": j.state if j else "missing"}
    if j and j.state == "observed":
        out.update(reason="observed")
        return out
    if j is None:
        out.update(reason="absent", in_frame=None, in_foreground=None)
        return out

    in_frame = 0 <= j.x < (w or 10**9) and 0 <= j.y < (h or 10**9)
    out["in_frame"] = in_frame
    if not in_frame:
        out.update(reason="truncated", in_foreground=None)
        return out
    if mask is not None:
        fg = _inside_mask(mask, w, h, j.x, j.y)
        out["in_foreground"] = fg
        out["reason"] = "occluded" if fg else "unobserved"
    else:
        out.update(reason="unobserved", in_foreground=None)
    return out


def audit(skel: Skeleton, mask=None, w: int = 0, h: int = 0) -> dict:
    """Per-joint reasons + counts by reason."""
    reasons: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for name in skel.joints:
        r = reason(skel, name, mask, w, h)
        reasons[name] = r["reason"]
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    return {"reasons": reasons, "counts": counts}

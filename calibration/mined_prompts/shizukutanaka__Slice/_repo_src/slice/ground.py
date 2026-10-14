"""Ground line — where the body meets the floor.

The lowest observed support joint (foot, then ankle) marks the
ground line. A figure standing on it has both support joints on
one level and nothing below; feet floating well above the lowest
body point read as airborne. Cropped feet are reported as cropped,
not guessed as grounded — the frame edge is evidence too.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .skeleton import Skeleton, observed_body_span

Point = Tuple[float, float]

_SUPPORT = ("foot_l", "foot_r", "ankle_l", "ankle_r")


def estimate(skel: Skeleton, frame_h: Optional[int] = None) -> dict:
    """{ground_y, contact, support_joints, reasons} — contact is
    grounded|airborne|cropped|unknown."""
    supports = []
    lowest = None
    lowest_name = None
    for name, j in skel.joints.items():
        if j.state != "observed":
            # predicted joints are prior fill — a "foot below
            # ankle" guess must not mark the lowest body point
            continue
        if name in _SUPPORT:
            supports.append((name, j.y))
        if lowest is None or j.y > lowest:
            lowest, lowest_name = j.y, name

    out: Dict = {"ground_y": None, "contact": "unknown",
                 "support_joints": len(supports), "reasons": []}
    if not supports:
        out["reasons"].append("no_observed_support")
        return out

    ground_y = max(y for _, y in supports)
    out["ground_y"] = round(ground_y, 1)

    if frame_h and ground_y >= frame_h - 3:
        out["contact"] = "cropped"
        out["reasons"].append("support_at_frame_edge")
        return out

    if lowest_name in _SUPPORT:
        out["contact"] = "grounded"
        out["reasons"].append("support_is_lowest")
        # both support joints on one level?
        ys = [y for _, y in supports]
        span = _span(skel)
        # "uneven" is a fraction of body height — a raw-pixel
        # default would fabricate the tolerance.
        if len(ys) > 1 and span > 0 \
                and max(ys) - min(ys) > 0.08 * span:
            out["reasons"].append("uneven_support")
    else:
        out["contact"] = "airborne"
        out["reasons"].append(
            f"{lowest_name} below support line")
    return out


def _span(skel: Skeleton) -> float:
    # observed-only span: a predicted head/foot is prior fill
    return observed_body_span(skel)


def clearance(skel: Skeleton) -> Optional[float]:
    """Vertical gap between ground line and the lowest OBSERVED body
    point (0 when grounded) — a predicted joint below the line is a
    guess, not floating body."""
    r = estimate(skel)
    if r["ground_y"] is None:
        return None
    lo = max(j.y for j in skel.joints.values()
             if j.state == "observed")
    return round(lo - r["ground_y"], 1)

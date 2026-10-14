"""Local pixel sampling at joint positions — skin vs cloth cues.

Color statistics in a small window around each joint tell whether the
limb there reads as bare skin or covered (clothing/hair/prosthetic).
This is the raw material for gendered/anatomical knowledge: a
"skin-like" forearm and a dark-covered one carry different meaning.
Every sample reports pixel counts and means — a joint over background
just reports few pixels, not a guess.
"""

from __future__ import annotations

from typing import Dict, Optional

from .bitmap import Bitmap
from .skeleton import Skeleton

_SKIN = (200, 150, 120)  # rough mid skin tone anchor


def _skin_like(r: int, g: int, b: int) -> bool:
    # skin tends r > g > b with warm range; also darker warm tones count
    return r > 95 and r > g > b and (r - g) > 10 and (max(r, g, b)
                                                    - min(r, g, b)) > 12


def sample(bmp: Bitmap, x: float, y: float, radius: int = 4,
           mask: Optional[list] = None) -> dict:
    """Color stats in a square window around (x, y)."""
    n = skin = 0
    sr = sg = sb = 0
    x0, y0 = int(x) - radius, int(y) - radius
    for yy in range(y0, y0 + radius * 2 + 1):
        for xx in range(x0, x0 + radius * 2 + 1):
            if not (0 <= xx < bmp.width and 0 <= yy < bmp.height):
                continue
            if mask is not None and not mask[yy][xx]:
                continue
            p = bmp.get(xx, yy)
            if not p or p[3] < 128:
                continue
            r, g, b = p[0], p[1], p[2]
            n += 1
            sr += r
            sg += g
            sb += b
            if _skin_like(r, g, b):
                skin += 1
    if not n:
        return {"pixels": 0}
    return {
        "pixels": n,
        "mean_rgb": (round(sr / n, 1), round(sg / n, 1),
                     round(sb / n, 1)),
        "skin_ratio": round(skin / n, 3),
        "brightness": round((sr + sg + sb) / (3 * n), 1),
    }


def joints_report(bmp: Bitmap, skel: Skeleton,
                  mask: Optional[list] = None,
                  radius: int = 4) -> Dict[str, dict]:
    """Sample a window around every OBSERVED joint; label bare vs
    covered. Predicted joints are skipped: their position is prior
    fill, so the sampled pixels are real but the joint attribution
    is a guess — "wrist_l: covered" would be a fabricated label."""
    out: Dict[str, dict] = {}
    for name, j in skel.joints.items():
        if j.state != "observed":
            continue
        s = sample(bmp, j.x, j.y, radius, mask)
        s = dict(s)
        if s.get("pixels", 0) == 0:
            s["cover"] = "no_pixels"
        elif s.get("skin_ratio", 0) >= 0.5:
            s["cover"] = "skin_like"
        else:
            s["cover"] = "covered"
        out[name] = s
    return out


def body_cover_summary(report: Dict[str, dict]) -> dict:
    """How much of the body reads bare skin vs covered."""
    skin = covered = none = 0
    for s in report.values():
        c = s.get("cover")
        if c == "skin_like":
            skin += 1
        elif c == "covered":
            covered += 1
        else:
            none += 1
    return {"skin_like_joints": skin, "covered_joints": covered,
            "no_pixels_joints": none}

"""Bone-level silhouette coverage — do bones stay inside the body?

`evid` audits whether each *joint* sits on the mask. This audits the
*segments between them*: a bone is a physical body part, so the line
shoulder→elbow should pass through foreground pixels. A bone that
crosses background is anatomically impossible — either the joint
positions are wrong or the body isn't what the mask says.

`check(skel, mask)` walks each bone at ~2px steps and reports:

- per-bone {covered_fraction, gaps: [{start_t, end_t, length_px}]}
  where a gap is a contiguous run of background crossings
- `broken` — observed bones whose largest gap exceeds GAP_PX and
  whose endpoints are both observed (a predicted bone crossing
  background is normal: prediction interpolates through empty space)
- `verdict`: covered | gaps | insufficient

Only bones with two observed endpoints are judged; a bone anchored
by a predicted joint is recorded but never counted as broken.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .landmarks import BONES
from .skeleton import OBSERVED, Skeleton

STEP_PX = 2.0     # sampling density along the bone
GAP_PX = 6.0      # a background run longer than this breaks the bone
EDGE = 1          # treat this many px outside the mask as uncovered


def _inside(mask: List[bytearray], x: int, y: int) -> bool:
    if y < 0 or y >= len(mask) or x < 0 or x >= len(mask[0]):
        return False
    return bool(mask[y][x])


def _walk(p0: Tuple[float, float], p1: Tuple[float, float],
          mask: List[bytearray]) -> Tuple[int, List[Tuple[float, float]]]:
    """Sample the segment; return (total samples, off-mask runs)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    n = max(2, int(length / STEP_PX) + 1)
    off: List[Tuple[float, float]] = []
    in_gap = False
    for i in range(n):
        t = i / (n - 1)
        x = int(round(p0[0] + dx * t))
        y = int(round(p0[1] + dy * t))
        if not _inside(mask, x, y):
            if not in_gap:
                off.append([t, t])
                in_gap = True
            else:
                off[-1][1] = t
        else:
            in_gap = False
    return n, [(g[0], g[1]) for g in off]


def check(skel: Skeleton, mask: List[bytearray]) -> Dict:
    """Audit bone coverage against a foreground mask."""
    bones: Dict[str, dict] = {}
    broken: List[str] = []
    for a, b in BONES:
        ja, jb = skel.joints.get(a), skel.joints.get(b)
        if not ja or not jb:
            continue
        measured = ja.state == OBSERVED and jb.state == OBSERVED
        n, gaps = _walk((ja.x, ja.y), (jb.x, jb.y), mask)
        length = max(1e-6, ((jb.x - ja.x) ** 2
                            + (jb.y - ja.y) ** 2) ** 0.5)
        gap_list = [{
            "start_t": round(g[0], 3), "end_t": round(g[1], 3),
            "length_px": round((g[1] - g[0]) * length, 2),
        } for g in gaps]
        worst = max((g["length_px"] for g in gap_list), default=0.0)
        covered = 1.0 - sum(g["length_px"] for g in gap_list) / length
        name = "%s-%s" % (a, b)
        bones[name] = {
            "measured": measured,
            "covered_fraction": round(covered, 3),
            "gaps": gap_list,
            "broken": measured and worst > GAP_PX,
        }
        if bones[name]["broken"]:
            broken.append(name)

    if not bones:
        return {"verdict": "insufficient",
                "reason": "no bones with both endpoints placed"}
    observed_broken = broken
    verdict = "gaps" if observed_broken else "covered"
    return {
        "verdict": verdict,
        "bones": bones,
        "broken": broken,
        "n_bones": len(bones),
        "note": "only bones with two observed endpoints count as "
                "broken — predicted bones legitimately cross space",
    }


def uncovered(result: Dict) -> List[str]:
    """Bone names flagged broken."""
    return result["broken"]

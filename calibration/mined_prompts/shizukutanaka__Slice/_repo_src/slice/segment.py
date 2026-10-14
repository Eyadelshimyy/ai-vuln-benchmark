"""Per-pixel part segmentation — DensePose-style labeling, evidence only.

Every foreground pixel is assigned to the part of the nearest bone
segment. The result is a dense "which body part is this pixel" map —
the lightweight analogue of DensePose/part-affinity labeling, driven
by the estimated skeleton. Parts whose bones are missing get zero
pixels: the map labels evidence, it never invents it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .bitmap import Bitmap
from .landmarks import BONES
from .skeleton import Skeleton

# bone (joint_a, joint_b) -> anatomical part label
PART_OF = {
    ("head", "neck"): "head",
    ("neck", "chest"): "torso",
    ("chest", "pelvis"): "torso",
    ("neck", "shoulder_l"): "torso",
    ("neck", "shoulder_r"): "torso",
    ("pelvis", "hip_l"): "torso",
    ("pelvis", "hip_r"): "torso",
    ("shoulder_l", "elbow_l"): "upper_arm_l",
    ("elbow_l", "wrist_l"): "forearm_l",
    ("shoulder_r", "elbow_r"): "upper_arm_r",
    ("elbow_r", "wrist_r"): "forearm_r",
    ("hip_l", "knee_l"): "thigh_l",
    ("knee_l", "ankle_l"): "shin_l",
    ("ankle_l", "foot_l"): "foot_l",
    ("hip_r", "knee_r"): "thigh_r",
    ("knee_r", "ankle_r"): "shin_r",
    ("ankle_r", "foot_r"): "foot_r",
}

PARTS = sorted(set(PART_OF.values()))


def _seg_dist(px: float, py: float, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2


def label_map(bmp: Bitmap, skel: Skeleton, mask: List[bytearray],
              ) -> List[Optional[str]]:
    """Per-pixel part name for every foreground pixel, else None."""
    segs = []
    for a, b in BONES:
        ja, jb = skel.joints.get(a), skel.joints.get(b)
        if (ja and jb and ja.state == "observed"
                and jb.state == "observed"):
            # predicted endpoints place the bone by prior — pixels
            # labeled off it would wear a fabricated part name
            segs.append((PART_OF[(a, b)], (ja.x, ja.y), (jb.x, jb.y)))
    w, h = bmp.width, bmp.height
    out: List[Optional[str]] = [None] * (w * h)
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            best, best_d = None, float("inf")
            for part, pa, pb in segs:
                d = _seg_dist(x + 0.5, y + 0.5, pa, pb)
                if d < best_d:
                    best, best_d = part, d
            out[y * w + x] = best
    return out


def summary(bmp: Bitmap, skel: Skeleton, mask: List[bytearray],
            ) -> Dict[str, object]:
    """Pixel counts per part — how much evidence covers each body part."""
    labels = label_map(bmp, skel, mask)
    counts: Dict[str, int] = {}
    total = 0
    for v in labels:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
            total += 1
    parts = {
        p: {"pixels": counts.get(p, 0),
            "fraction": round(counts.get(p, 0) / total, 4) if total else 0.0}
        for p in PARTS
    }
    return {"total_pixels": total, "parts": parts}

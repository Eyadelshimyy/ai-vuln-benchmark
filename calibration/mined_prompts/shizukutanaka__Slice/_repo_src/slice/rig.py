"""Rig export — the skeleton as an animation rig.

A rig is bones with parents, lengths, and directions — the structure
retargeting and 3D toolchains (GLTF/BVH/Blender) consume. `build`
emits each BONES edge as a rig bone rooted at the parent joint; bones
missing either endpoint are omitted rather than zeroed, so a rig
honestly reflects what the skeleton knows.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .landmarks import BONES
from .skeleton import Skeleton

# Which rig bone each joint is the head of (bone named by its child).
_NAME = {
    "neck": "neck",
    "shoulder_l": "clavicle_l", "shoulder_r": "clavicle_r",
    "chest": "spine", "pelvis": "hips",
    "elbow_l": "upper_arm_l", "elbow_r": "upper_arm_r",
    "wrist_l": "forearm_l", "wrist_r": "forearm_r",
    "hip_l": "hip_l", "hip_r": "hip_r",
    "knee_l": "thigh_l", "knee_r": "thigh_r",
    "ankle_l": "shin_l", "ankle_r": "shin_r",
    "foot_l": "foot_l", "foot_r": "foot_r",
    "head": "head",
}


def build(skel: Skeleton) -> List[dict]:
    """Rig bones [{name, parent, head:[x,y], tail:[x,y], length, dir}]."""
    out: List[dict] = []
    for pa, ch in BONES:
        # BONES stores ("head","neck") crownward; a rig grows rootward
        # (pelvis is the root) so the bone runs neck→head instead.
        if pa == "head":
            pa, ch = "neck", "head"
        a, b = skel.point(pa), skel.point(ch)
        if not a or not b:
            continue
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        pa_j, ch_j = skel.joints[pa], skel.joints[ch]
        out.append({
            "name": _NAME.get(ch, ch),
            "parent": pa,
            "head": [a[0], a[1]],
            "tail": [b[0], b[1]],
            "length": round(L, 2),
            "dir": [round(dx / L, 4), round(dy / L, 4)] if L else [0, 0],
            # a bone over predicted endpoints is guessed structure —
            # downstream rigs must not read it as observed anatomy
            "state": ("predicted" if "predicted" in
                      (pa_j.state, ch_j.state) else pa_j.state),
            "confidence": round(min(pa_j.confidence,
                                    ch_j.confidence), 3),
        })
    return out


def hierarchy(rig: List[dict]) -> Dict[str, List[str]]:
    """{parent_joint: [child bone names]} — the rig's tree shape."""
    tree: Dict[str, List[str]] = {}
    for b in rig:
        tree.setdefault(b["parent"], []).append(b["name"])
    return tree


def total_bone_length(rig: List[dict]) -> float:
    return round(sum(b["length"] for b in rig), 2)

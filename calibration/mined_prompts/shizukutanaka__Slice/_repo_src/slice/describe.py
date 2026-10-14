"""Human-readable description of an analyzed figure.

Turns the skeleton plus classification labels into short clauses —
"standing, facing right, left arm not directly observed".
The knowledge engine's NLG layer: every clause states only what the
data actually says (observed vs predicted, never guessing).
"""

from __future__ import annotations

from typing import List, Optional

from .skeleton import OBSERVED, Skeleton

_POSE_EN = {
    "stand": "standing",
    "walk": "walking",
    "run": "running",
    "sit": "sitting",
    "squat": "squatting",
    "crouch": "crouching",
    "lie": "lying down",
    "invert": "upside down",
    "bend": "bending forward",
    "arms_up": "with both arms raised",
    "t_pose": "in a T-pose",
    "unknown": "of indeterminate pose",
}

_FACING_EN = {
    "front": "facing the camera",
    "side": "in profile",
    "left": "facing left",
    "right": "facing right",
    "three-quarter": "in three-quarter view",
    "unknown": "facing unknown",
}

_GROUPS = [
    ("arm_l", ["wrist_l", "elbow_l"]),
    ("arm_r", ["wrist_r", "elbow_r"]),
    ("leg_l", ["ankle_l", "knee_l"]),
    ("leg_r", ["ankle_r", "knee_r"]),
]
_GROUP_EN = {"arm_l": "left arm", "arm_r": "right arm",
             "leg_l": "left leg", "leg_r": "right leg"}


def clauses(skel: Skeleton, pose: Optional[str] = None) -> List[str]:
    out: List[str] = []

    pose = pose if pose is not None else getattr(skel, "pose", None)
    if pose:
        out.append(_POSE_EN.get(pose, pose))

    ori = skel.orientation or {}
    facing = ori.get("facing")
    if facing and facing != "unknown":
        out.append(_FACING_EN.get(facing, f"facing {facing}"))

    for gname, keys in _GROUPS:
        joints = [skel.joints.get(k) for k in keys]
        obs = [j for j in joints
               if j is not None and j.state == OBSERVED]
        if len(obs) == len(joints) and joints:
            out.append(f"{_GROUP_EN[gname]} observed")
        elif not obs:
            out.append(f"{_GROUP_EN[gname]} not directly observed")

    obs_n = sum(1 for j in skel.joints.values() if j.state == OBSERVED)
    out.append(f"{obs_n} of {len(skel.joints)} joints observed")
    return out


def describe(skel: Skeleton, pose: Optional[str] = None) -> str:
    c = clauses(skel, pose)
    if not c:
        return "no figure data"
    text = "; ".join(c)
    return text[0].upper() + text[1:]

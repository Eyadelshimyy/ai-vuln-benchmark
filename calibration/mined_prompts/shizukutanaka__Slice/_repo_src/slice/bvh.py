"""BVH export — the skeleton in the standard animation format.

BVH (Biovision Hierarchy) is the text format Blender, MotionBuilder
and every retarget tool reads: a HIERARCHY of joints with offsets
relative to their parent, plus a MOTION section of per-frame channel
values. We emit a single-frame BVH — offsets carry the real 2D pose
(z=0), channels are zeroed (no rotation data yet — honesty over
pretending).
"""

from __future__ import annotations

from typing import List

from .skeleton import Skeleton

ROOT = "pelvis"

# joint -> children (BONES restated rootward from pelvis)
_CHILDREN = {
    "pelvis": ["chest", "hip_l", "hip_r"],
    "chest": ["neck"],
    "neck": ["head", "shoulder_l", "shoulder_r"],
    "shoulder_l": ["elbow_l"], "elbow_l": ["wrist_l"],
    "shoulder_r": ["elbow_r"], "elbow_r": ["wrist_r"],
    "hip_l": ["knee_l"], "knee_l": ["ankle_l"], "ankle_l": ["foot_l"],
    "hip_r": ["knee_r"], "knee_r": ["ankle_r"], "ankle_r": ["foot_r"],
}


def _walk(joint: str, skel: Skeleton) -> List[str]:
    out = [joint]
    for c in _CHILDREN.get(joint, []):
        if skel.point(c) is not None:
            out.extend(_walk(c, skel))
    return out


def export(skel: Skeleton, name: str = "SliceFigure") -> str:
    """Single-frame BVH text; raises ValueError if the root is missing."""
    if skel.point(ROOT) is None:
        raise ValueError("pelvis joint required for BVH export")

    lines: List[str] = ["HIERARCHY", f"ROOT {name}"]

    def emit(joint: str, depth: int) -> List[str]:
        ind = "\t" * depth
        out: List[str] = []
        p = skel.point(joint)
        kids = [c for c in _CHILDREN.get(joint, [])
                if skel.point(c) is not None]
        for c in kids:
            cp = skel.point(c)
            out.append(f"{ind}JOINT {c}")
            out.append(f"{ind}{{")
            out.append(f"{ind}\tOFFSET {cp[0] - p[0]:.2f} "
                       f"{cp[1] - p[1]:.2f} 0.00")
            out.append(f"{ind}\tCHANNELS 3 "
                       "Zrotation Xrotation Yrotation")
            gk = [g for g in _CHILDREN.get(c, [])
                  if skel.point(g) is not None]
            if not gk:
                out.append(f"{ind}\tEnd Site")
                out.append(f"{ind}\t{{")
                out.append(f"{ind}\t\tOFFSET 0.00 0.00 0.00")
                out.append(f"{ind}\t}}")
            out.extend(emit(c, depth + 1))
            out.append(f"{ind}}}")
        return out

    root = skel.point(ROOT)
    lines.append("{")
    lines.append(f"\tOFFSET {root[0]:.2f} {root[1]:.2f} 0.00")
    lines.append("\tCHANNELS 6 Xposition Yposition Zposition "
                 "Zrotation Xrotation Yrotation")
    lines.extend(emit(ROOT, 1))
    lines.append("}")

    n_joints = len(_walk(ROOT, skel)) - 1
    lines.append("MOTION")
    lines.append("Frames: 1")
    lines.append("Frame Time: 0.033333")
    vals = [f"{root[0]:.2f}", f"{root[1]:.2f}", "0.00", "0.00",
            "0.00", "0.00"]
    vals += ["0.00"] * (3 * n_joints)
    lines.append(" ".join(vals))
    return "\n".join(lines) + "\n"


def report(skel: Skeleton) -> dict:
    """Provenance for the export BVH cannot carry.

    The BVH format has no per-joint metadata channel, so `export`
    writes predicted joints as ordinary offsets — indistinguishable
    from measured ones downstream. This sidecar lists which joints
    in the exported tree rest on prediction rather than evidence.
    """
    names = _walk(ROOT, skel)
    predicted = sorted(n for n in names
                       if skel.joints[n].state != "observed")
    return {
        "joints": names,
        "n_joints": len(names),
        "predicted": predicted,
        "state": "estimated",
        "basis": "joints listed under `predicted` have BVH offsets "
                 "computed from inferred, not measured, positions",
    }

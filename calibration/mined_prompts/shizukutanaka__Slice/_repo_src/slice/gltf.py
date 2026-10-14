"""glTF 2.0 export — the second 3D bridge after `slice.bvh`.

Exports the skeleton as a glTF node hierarchy rooted at the pelvis:
every joint becomes a node whose `translation` is parent-relative
(BVH's OFFSET convention), and every node carries the honesty
contract in `extras` — `state`, `confidence`, `basis` — so a
downstream tool can still tell observed from predicted.

Coordinates: 2D joints lift into the XY plane with Y flipped
(image space is Y-down; glTF viewers expect Y-up) and Z = 0.
A real depth channel belongs to Phase 3 (`slice.lift`); here the
flatness is stated once in `asset.extras` instead of faking it.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .landmarks import MIRROR
from .skeleton import Skeleton

# Kinematic tree rooted at the pelvis (joint -> parent).
_PARENT: Dict[str, str] = {
    "chest": "pelvis", "neck": "chest", "head": "neck",
    "shoulder_l": "neck", "elbow_l": "shoulder_l", "wrist_l": "elbow_l",
    "shoulder_r": "neck", "elbow_r": "shoulder_r", "wrist_r": "elbow_r",
    "hip_l": "pelvis", "knee_l": "hip_l", "ankle_l": "knee_l",
    "foot_l": "ankle_l",
    "hip_r": "pelvis", "knee_r": "hip_r", "ankle_r": "knee_r",
    "foot_r": "ankle_r",
    "spine": "pelvis",
}

_ROOT = "pelvis"


def _pos(j) -> List[float]:
    # image (x right, y down) -> glTF (x right, y up), depth 0
    return [round(j.x, 3), round(-j.y, 3), 0.0]


def to_gltf(skel: Skeleton, *, name: str = "skeleton") -> dict:
    """Build a glTF document dict for the skeleton.

    Joints missing from the skeleton are absent as nodes; a node whose
    parent chain has no present ancestor is promoted to a scene root,
    which `scene.nodes` records honestly.
    """
    order: List[str] = [n for n in _walk() if n in skel.joints]
    index = {n: i for i, n in enumerate(order)}
    nodes: List[dict] = []
    children_of: Dict[Optional[str], List[int]] = {}
    for n in order:
        j = skel.joints[n]
        parent = _nearest_parent(skel, n)
        pp = skel.point(parent) if parent else None
        if pp is None:
            tr = _pos(j)
            children_of.setdefault(None, []).append(index[n])
        else:
            tr = [round(j.x - pp[0], 3), round(-(j.y - pp[1]), 3), 0.0]
            children_of.setdefault(parent, []).append(index[n])
        nodes.append({
            "name": n,
            "translation": tr,
            "children": children_of.get(n, []),
            "extras": {
                "state": j.state,
                "confidence": round(j.confidence, 3),
                "basis": j.basis,
            },
        })
    # children lists were filled after nodes were appended — patch in
    for parent, kids in children_of.items():
        if parent is not None:
            nodes[index[parent]]["children"] = kids
    return {
        "asset": {
            "version": "2.0",
            "generator": "slice",
            "extras": {
                "schema": "slice.skeleton/gltf",
                "note": ("2D lift: XY plane, Y flipped from image space, "
                         "Z=0 for all joints — no depth was observed"),
            },
        },
        "scene": 0,
        "scenes": [{"name": name,
                    "nodes": sorted(children_of.get(None, []))}],
        "nodes": nodes,
    }


def _walk() -> List[str]:
    """Pelvis-first preorder over the full vocabulary."""
    out, stack = [], [_ROOT]
    kids: Dict[str, List[str]] = {}
    for child, parent in _PARENT.items():
        kids.setdefault(parent, []).append(child)
    while stack:
        n = stack.pop(0)
        out.append(n)
        stack.extend(kids.get(n, []))
    return out


def _nearest_parent(skel: Skeleton, name: str) -> Optional[str]:
    p = _PARENT.get(name)
    while p is not None and p not in skel.joints:
        p = _PARENT.get(p)
    return p


def export(skel: Skeleton, *, name: str = "skeleton") -> str:
    """glTF document as a JSON string."""
    return json.dumps(to_gltf(skel, name=name), indent=2)


def mirrored(skel: Skeleton) -> dict:
    """Convenience: the same document for the mirrored skeleton —
    left/right swapped, x negated. Kept tiny because MIRROR is
    vocabulary data, not glTF logic."""
    from .skeleton import Joint
    out = Skeleton(skel.image_width, skel.image_height,
                   joints={}, orientation=dict(skel.orientation),
                   body_model=dict(skel.body_model),
                   centroid=skel.centroid)
    for n, j in skel.joints.items():
        out.set(Joint(MIRROR.get(n, n), -j.x, j.y, j.confidence,
                      j.state, j.basis))
    return to_gltf(out)

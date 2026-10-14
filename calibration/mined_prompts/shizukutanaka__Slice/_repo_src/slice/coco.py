"""COCO keypoints export — the dataset-standard annotation format.

COCO keypoint annotations encode each joint as (x, y, v) where the
visibility flag is 0=not labeled, 1=labeled but not visible,
2=labeled and visible. Slice's observed/predicted/missing joint
states map onto that flag almost exactly: observed→2, predicted→1,
absent→0 — so a Slice skeleton becomes a valid COCO annotation whose
visibility semantics already carry our honesty contract.

Slice uses 19 joints; COCO uses 17. Joints without a COCO
counterpart (chest, pelvis, feet, neck) are listed under
`unmapped_joints` rather than dropped silently.
"""

from __future__ import annotations

from typing import Dict, List

from .skeleton import OBSERVED, Skeleton

# COCO keypoint order -> our joint name (None = no Slice counterpart)
COCO_JOINTS = [
    ("nose", "head"),
    ("left_eye", None), ("right_eye", None),
    ("left_ear", None), ("right_ear", None),
    ("left_shoulder", "shoulder_l"),
    ("right_shoulder", "shoulder_r"),
    ("left_elbow", "elbow_l"), ("right_elbow", "elbow_r"),
    ("left_wrist", "wrist_l"), ("right_wrist", "wrist_r"),
    ("left_hip", "hip_l"), ("right_hip", "hip_r"),
    ("left_knee", "knee_l"), ("right_knee", "knee_r"),
    ("left_ankle", "ankle_l"), ("right_ankle", "ankle_r"),
]

_SLICE_ONLY = ["neck", "chest", "pelvis", "foot_l", "foot_r"]


def to_coco(skel: Skeleton, image_id: int = 0,
            annotation_id: int = 0, category_id: int = 1) -> Dict:
    """Build a COCO-format annotation dict from a skeleton."""
    flat: List[float] = []
    n_labeled = 0
    confs: List[float] = []
    for _coco_name, ours in COCO_JOINTS:
        j = skel.joints.get(ours) if ours else None
        if j is None:
            flat += [0, 0, 0]
            continue
        v = 2 if j.state == OBSERVED else 1
        flat += [round(j.x, 1), round(j.y, 1), v]
        n_labeled += 1
        confs.append(j.confidence)
    unmapped = [n for n in _SLICE_ONLY if n in skel.joints]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "keypoints": flat,
        "num_keypoints": n_labeled,
        "score": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "unmapped_joints": unmapped,
    }


def coco_skeleton() -> List[List[int]]:
    """The COCO limb connectivity list (1-based index pairs)."""
    return [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]


def categories() -> List[Dict]:
    """COCO `categories` block describing our person keypoints."""
    return [{
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": [n for n, _ in COCO_JOINTS],
        "skeleton": coco_skeleton(),
    }]

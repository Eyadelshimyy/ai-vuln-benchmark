"""Human landmark vocabulary and the bone graph connecting them.

Names follow the SRS: head / neck / shoulders / elbows / wrists /
chest / pelvis / knees / ankles / foot tips, plus a spine midpoint used
only for mirroring predictions.
"""

JOINTS = [
    "head", "neck",
    "shoulder_l", "shoulder_r",
    "elbow_l", "elbow_r",
    "wrist_l", "wrist_r",
    "chest", "pelvis",
    "hip_l", "hip_r",
    "knee_l", "knee_r",
    "ankle_l", "ankle_r",
    "foot_l", "foot_r",
    "spine",
]

BONES = [
    ("head", "neck"),
    ("neck", "shoulder_l"), ("neck", "shoulder_r"),
    ("neck", "chest"), ("chest", "pelvis"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("pelvis", "hip_l"), ("pelvis", "hip_r"),
    ("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("ankle_l", "foot_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"), ("ankle_r", "foot_r"),
]

MIRROR = {
    "shoulder_l": "shoulder_r", "shoulder_r": "shoulder_l",
    "elbow_l": "elbow_r", "elbow_r": "elbow_l",
    "wrist_l": "wrist_r", "wrist_r": "wrist_l",
    "hip_l": "hip_r", "hip_r": "hip_l",
    "knee_l": "knee_r", "knee_r": "knee_l",
    "ankle_l": "ankle_r", "ankle_r": "ankle_l",
    "foot_l": "foot_r", "foot_r": "foot_l",
}

# Kinematic tree rooted at the pelvis (the SMPL convention): every
# joint's position is defined relative to its parent, which is what
# prediction walks and what a future 3D lift needs.
ROOT = "pelvis"
PARENT = {
    "hip_l": "pelvis", "hip_r": "pelvis",
    "chest": "pelvis",
    "neck": "chest",
    "head": "neck",
    "shoulder_l": "neck", "shoulder_r": "neck",
    "elbow_l": "shoulder_l", "elbow_r": "shoulder_r",
    "wrist_l": "elbow_l", "wrist_r": "elbow_r",
    "knee_l": "hip_l", "knee_r": "hip_r",
    "ankle_l": "knee_l", "ankle_r": "knee_r",
    "foot_l": "ankle_l", "foot_r": "ankle_r",
    # "spine" is a derived midpoint, not a kinematic node.
}


def chain(name: str) -> list:
    """Joint names from the root down to `name` (inclusive)."""
    path = [name]
    while path[-1] != ROOT:
        path.append(PARENT[path[-1]])
    path.reverse()
    return path

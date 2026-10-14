"""Prediction Engine: fill joints the silhouette could not provide.

Two honest strategies, in order:

1. symmetry — mirror the observed counterpart across the spine axis
   (side views, self-occlusion, cropping).
2. anatomy prior — place the joint by statistical limb lengths
   hanging from its parent (nothing visible at all).

Every inserted joint is state=predicted with modest confidence; callers
and the UI must never render it as observed fact.
"""

from __future__ import annotations

from typing import List, Optional

from .anatomy import BODY_MODELS, DEFAULT_MODEL
from .landmarks import MIRROR
from .skeleton import OBSERVED, PREDICTED, Joint, Skeleton

# parent -> (child joint suffix, prior key, fraction of chain length)
_CHAIN = {
    "shoulder": [("elbow", "upper_arm_ratio"), ("wrist", "forearm_ratio")],
    "hip": [("knee", "thigh_ratio"), ("ankle", "shin_ratio")],
}


def _spine_x(skel: Skeleton, y: float) -> float:
    spine = skel.get("spine")
    if spine:
        return spine.x
    neck, pelvis = skel.get("neck"), skel.get("pelvis")
    if neck and pelvis:
        t = (y - neck.y) / max(1e-6, pelvis.y - neck.y)
        return neck.x + (pelvis.x - neck.x) * min(max(t, 0), 1)
    # the component centroid is measured evidence — an off-centre
    # figure mirrored about the IMAGE centre lands the counterpart
    # on the wrong side of the body
    if skel.centroid is not None:
        return skel.centroid[0]
    return skel.image_width / 2


def complete(skel: Skeleton, model: Optional[str] = None) -> List[Joint]:
    """Insert predicted joints for anything missing. Returns the new joints."""
    model_name = (model or (skel.body_model or {}).get("name")
                  or DEFAULT_MODEL)
    prior = BODY_MODELS.get(model_name, BODY_MODELS[DEFAULT_MODEL])
    added: List[Joint] = []

    # The body span that scales every prior distance must come from
    # observed joints only — a predicted extremity is prior geometry,
    # so letting it extend the span inflates the very table the new
    # joints are then drawn from (round-trip fabrication).
    ys = [j.y for j in skel.joints.values() if j.state == OBSERVED] \
        or [j.y for j in skel.joints.values()]
    top = min(ys, default=0.0)
    bottom = max(ys, default=skel.image_height)
    body_h = max(1.0, bottom - top)

    def predict(name: str) -> Optional[Joint]:
        peer = MIRROR.get(name)
        if peer and peer in skel.joints and skel.joints[peer].state == OBSERVED:
            src = skel.joints[peer]
            sx = _spine_x(skel, src.y)
            j = Joint(name, 2 * sx - src.x, src.y,
                      round(src.confidence * 0.5, 3), PREDICTED,
                      f"mirrored from {peer}")
            skel.set(j)
            return j
        return _prior_joint(skel, name, prior, body_h, model_name)

    # Anchors first (shoulders/hips), then distal chain.
    for base in ("shoulder_l", "shoulder_r", "hip_l", "hip_r",
                 "head", "neck", "chest", "pelvis", "spine"):
        if base not in skel.joints:
            j = predict(base) or _prior_joint(skel, base, prior,
                                               body_h, model_name)
            if j:
                added.append(j)
    for stem, chain in _CHAIN.items():
        for side in ("l", "r"):
            parent = f"{stem}_{side}"
            if parent not in skel.joints:
                continue
            for child, key in chain:
                cname = f"{child}_{side}"
                if cname not in skel.joints:
                    j = predict(cname)
                    if j is None:
                        continue
                    added.append(j)
    for side in ("l", "r"):
        ankle = f"ankle_{side}"
        if f"foot_{side}" not in skel.joints and ankle in skel.joints:
            a = skel.joints[ankle]
            j = Joint(f"foot_{side}", a.x, min(a.y + body_h * 0.03,
                                             skel.image_height - 1),
                      0.25, PREDICTED, "foot below ankle")
            skel.set(j)
            added.append(j)
    return added


def _prior_joint(skel: Skeleton, name: str, prior: dict,
                 body_h: float, model_name: str) -> Optional[Joint]:
    """Place a joint from anatomy priors when no evidence exists."""
    anchors = {
        "head": ("neck", 0, -prior["head_ratio"]),
        "neck": ("chest", 0, -prior["head_ratio"] * 0.5),
        "chest": ("pelvis", 0, -prior["torso_ratio"] * 0.5),
        "pelvis": ("spine", 0, prior["torso_ratio"] * 0.5),
        "spine": ("chest", 0, prior["torso_ratio"] * 0.25),
    }
    if name in anchors:
        ref, dxr, dyr = anchors[name]
        r = skel.get(ref)
        if r:
            j = Joint(name, r.x + dxr * body_h, r.y + dyr * body_h,
                      0.2, PREDICTED, f"prior off {ref} ({model_name})")
            skel.set(j)
            return j

    child = name.split("_")[0]
    for stem, chain in _CHAIN.items():
        names = [c[0] for c in chain]
        if child not in names:
            continue
        for side in ("l", "r"):
            parent = f"{stem}_{side}"
            if name != f"{child}_{side}" or parent not in skel.joints:
                continue
            p = skel.joints[parent]
            idx = names.index(child)
            frac = sum(prior[chain[i][1]] for i in range(idx + 1))
            total = sum(prior[k] for _, k in chain)
            distal = next(
                (skel.joints[f"{chain[i][0]}_{side}"]
                 for i in range(idx + 1, len(chain))
                 if f"{chain[i][0]}_{side}" in skel.joints), None)
            if distal:
                # An endpoint further down the chain is known: place the
                # joint between parent and it, at the prior fraction of
                # the whole chain. Better than a blind straight drop.
                j = Joint(
                    name, p.x + (distal.x - p.x) * (frac / total),
                    p.y + (distal.y - p.y) * (frac / total),
                    0.3, PREDICTED,
                    f"interpolated {parent}-{distal.name} ({model_name})")
                skel.set(j)
                return j
            # arms angle slightly outward, legs drop straight down
            out = prior["shoulder_ratio"] * 0.4 if stem == "shoulder" else 0
            sign = -1 if side == "l" else 1
            j = Joint(name, p.x + sign * out * body_h,
                      p.y + frac * body_h,
                      0.2, PREDICTED, f"prior off {parent} ({model_name})")
            skel.set(j)
            return j
    return None

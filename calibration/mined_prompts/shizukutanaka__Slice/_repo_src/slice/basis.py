"""Basis vocabulary — canonical registry of joint provenance strings.

Every Joint carries a `basis` string saying *what evidence produced it*.
Until now each module invented its own phrasing; this module is the
shared vocabulary (AUDIT P4-22). It does NOT rename existing strings —
stored Knowledge documents must stay readable — it classifies them into
evidence categories and defines the prefixes new code should use.

Categories (weakest → strongest evidence):
    "observation"    measured directly from pixels
    "mirror"         copied across the body axis from an observed peer
    "interpolation"  placed between two observed anchors
    "prior"          placed by anatomical statistics alone
    "transform"      produced by a spatial/pose transform (retarget etc.)
    "unknown"        unrecognised wording — flagged, not rejected

New basis strings should start with one of the PREFIXES below so
`category()` can classify them mechanically.
"""

from __future__ import annotations

from typing import Dict, Optional

# Canonical prefixes. A basis beginning with one of these belongs to
# the mapped category. Order matters: first match wins.
PREFIXES = [
    ("interpolated", "interpolation"),
    ("mirrored", "mirror"),
    ("prior", "prior"),
    ("retargeted", "transform"),
    ("transformed", "transform"),
    ("scaled", "transform"),
    ("normalized", "transform"),
]

# Fixed bases emitted by the silhouette estimator — each names the
# pixel feature it was measured from.
OBSERVED_BASES = {
    "top blob centroid", "widest upper row", "widest hip-band row",
    "crotch split row", "arm blob extremity", "arm blob mid-extent",
    "leg run at knee height", "leg run at bottom",
    "merged leg run", "legs not separable",
    "axis midpoint", "midpoint shoulders-pelvis",
    "clavicle midpoint below shoulder row",
    "silhouette bottom",
}

# Observed-state joints stamped from a prior, not pixels — still
# honest about it (the wording says "prior").
OBSERVED_PRIOR_BASES = {"head height prior"}


def category(basis: Optional[str]) -> str:
    """Classify a basis string into an evidence category."""
    if not basis:
        return "unknown"
    low = basis.strip().lower()
    for prefix, cat in PREFIXES:
        if low.startswith(prefix):
            return cat
    if low in OBSERVED_PRIOR_BASES:
        return "prior"
    if low in OBSERVED_BASES or any(
            low.startswith(b) for b in OBSERVED_BASES):
        return "observation"
    return "unknown"


def explain(joint) -> Dict[str, str]:
    """{basis, category} for a Joint — provenance made queryable."""
    b = getattr(joint, "basis", "") or ""
    return {"basis": b, "category": category(b),
            "state": getattr(joint, "state", "unknown")}


def audit(skel) -> Dict[str, object]:
    """Provenance census of a skeleton: count per category and any
    joints whose basis wording is outside the vocabulary."""
    counts: Dict[str, int] = {}
    unknown = []
    for name, j in skel.joints.items():
        cat = category(j.basis)
        counts[cat] = counts.get(cat, 0) + 1
        if cat == "unknown" and j.basis:
            unknown.append(name)
    return {
        "by_category": counts,
        "unknown_basis": sorted(unknown),
        "note": ("category is derived from basis wording, not from "
                 "joint.state — an observed joint may still rest on a "
                 "prior (e.g. 'head height prior')"),
    }

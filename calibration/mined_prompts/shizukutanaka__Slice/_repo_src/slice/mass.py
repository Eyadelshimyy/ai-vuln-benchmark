"""Body mass estimate — silhouette area to kilograms.

Volume ≈ silhouette area × an assumed mean body depth, and mass
≈ volume × 1.0 kg/L (soft tissue density). Pixel area converts to
cm² via a height prior (adult ≈ 170 cm, child ≈ 110, deformed ≈
150). Every number here is a prior-stacked estimate — the output
says `state: "estimated"` and reports the assumed depth so a
consumer can judge the chain of assumptions, not just the kg.
"""

from __future__ import annotations

from typing import Optional

from .skeleton import Skeleton

_HEIGHT_CM = {"adult": 170.0, "child": 110.0, "deformed": 150.0}
_DEPTH_RATIO = 0.28   # mean depth ≈ 28% of mean silhouette width
_DENSITY = 1.04       # kg/L soft tissue


def estimate(skel: Skeleton, area_px: float,
             mean_width_px: Optional[float] = None) -> Optional[dict]:
    """{kg, volume_l, height_cm, state} — None without a body span."""
    hj = skel.joints.get("head")
    top = (hj.x, hj.y) if hj and hj.state == "observed" else None
    lo = max((j.y for j in skel.joints.values()
              if j.state == "observed"), default=0.0)
    if top is None or lo - top[1] <= 0 or area_px <= 0:
        return None

    model = (skel.body_model or {}).get("name", "adult")
    height_cm = _HEIGHT_CM.get(model, 170.0)
    cm_per_px = height_cm / (lo - top[1])

    if mean_width_px is None:
        # crude: area / body height ≈ mean width
        mean_width_px = area_px / (lo - top[1])
    depth_cm = mean_width_px * cm_per_px * _DEPTH_RATIO
    width_cm = mean_width_px * cm_per_px

    area_cm2 = area_px * cm_per_px * cm_per_px
    volume_l = area_cm2 * depth_cm / 1000.0  # cm³ → litres
    kg = volume_l * _DENSITY
    return {"kg": round(kg, 1), "volume_l": round(volume_l, 1),
            "height_cm": height_cm, "depth_cm": round(depth_cm, 1),
            "width_cm": round(width_cm, 1),
            "state": "estimated", "model": model}


def bmi(skel: Skeleton, area_px: float) -> Optional[float]:
    """kg / m² — same priors, so equally estimated."""
    m = estimate(skel, area_px)
    if not m:
        return None
    h_m = m["height_cm"] / 100.0
    return round(m["kg"] / (h_m * h_m), 1)

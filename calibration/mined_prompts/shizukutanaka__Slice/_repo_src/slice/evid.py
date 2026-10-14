"""Per-joint evidence localization — is the observed joint on the mask?

`fit` asks whether the skeleton explains the silhouette globally;
`locate` asks the sharper question per joint: a joint marked
`observed` claims pixel evidence supports it — that evidence lives
on or inside the foreground mask. A joint floating *outside* the
mask (negative distance to the nearest foreground pixel) is an
observed claim with no observed support — a projection error, a
wrong component, or an outright fabrication.

Uses `distfield.distance_transform` twice: once on the mask
(distance inside the foreground to its boundary), once on the
inverted mask (distance in background to the foreground). Each
joint lands in one of:

- `interior`   — safely inside the silhouette (boundary > BAND px)
- `on_boundary`— within BAND px of the edge (boundary features like
                 shoulders and wrist ends live here)
- `off_mask`   — outside the silhouette entirely (flagged)

`off_mask` joints keep their state — this layer reports, it does
not reclassify — but they are listed explicitly so a caller can
down-weight them.
"""

from __future__ import annotations

from typing import Dict, List

from .distfield import distance_transform
from .skeleton import OBSERVED, Skeleton

BAND_PX = 1.5


def _invert(mask: List[bytearray]) -> List[bytearray]:
    return [bytearray(0 if v else 1 for v in row) for row in mask]


def locate(skel: Skeleton, mask: List[bytearray]) -> Dict:
    """Classify each joint by its position relative to the mask.

    `mask` is the estimator-frame foreground mask (same coordinate
    system as `skel`). Returns {joints: {name: {zone, dist_px}},
    off_mask: [...], observed_on_mask_fraction, state, basis}.
    """
    h = len(mask)
    w = len(mask[0]) if h else 0
    din = distance_transform(mask)
    dout = distance_transform(_invert(mask))

    joints: Dict[str, Dict] = {}
    off_mask: List[str] = []
    obs_total = obs_on = 0
    for name, j in sorted(skel.joints.items()):
        x, y = int(j.x), int(j.y)
        if not (0 <= x < w and 0 <= y < h):
            zone, d = "off_mask", None
        elif mask[y][x]:
            d = din[y * w + x]
            zone = "interior" if d > BAND_PX else "on_boundary"
        else:
            d = -dout[y * w + x]
            zone = "off_mask"
        joints[name] = {"zone": zone,
                        "dist_px": round(d, 2) if d is not None else None,
                        "state": j.state}
        if zone == "off_mask":
            off_mask.append(name)
        if j.state == OBSERVED:
            obs_total += 1
            if zone != "off_mask":
                obs_on += 1

    return {
        "joints": joints,
        "off_mask": sorted(off_mask),
        "observed_on_mask_fraction":
            round(obs_on / obs_total, 3) if obs_total else None,
        "state": "measured",
        "basis": "chamfer distance to mask boundary, band %.1fpx"
                 % BAND_PX,
    }


def unsupported(result: Dict) -> List[str]:
    """Observed joints positioned outside the mask — flagged."""
    return sorted(n for n, v in result["joints"].items()
                  if v["zone"] == "off_mask" and v["state"] == OBSERVED)

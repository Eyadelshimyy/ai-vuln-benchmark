"""2D skeleton overlay rendering on a Bitmap (no dependencies).

Observed joints/bones render blue, predicted render orange — the same
honesty color code the web viewer uses. The state distinction is also
dual-encoded in shape so it survives colour-blindness and greyscale
printing: observed = solid line + filled disc, predicted = dashed
line + hollow ring.
"""

from __future__ import annotations

from typing import Tuple

from .bitmap import Bitmap, encode_png
from .landmarks import BONES
from .skeleton import OBSERVED, Skeleton

BLUE = (30, 120, 255, 255)
ORANGE = (255, 150, 0, 255)
WHITE = (255, 255, 255, 255)
GREEN = (80, 220, 120, 255)  # facing indicator — not a state color


def _line(bmp: Bitmap, x0, y0, x1, y1, rgba):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        for ox in (-1, 0):
            for oy in (-1, 0):
                bmp.set(x0 + ox, y0 + oy, rgba)
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _disc(bmp: Bitmap, cx, cy, r, rgba):
    for y in range(int(cy) - r, int(cy) + r + 1):
        for x in range(int(cx) - r, int(cx) + r + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                bmp.set(x, y, rgba)


def _ring(bmp: Bitmap, cx, cy, r, rgba):
    """Hollow circle of radius r, ~1px stroke — predicted marker."""
    for y in range(int(cy) - r, int(cy) + r + 1):
        for x in range(int(cx) - r, int(cx) + r + 1):
            d = (x - cx) ** 2 + (y - cy) ** 2
            if (r - 1.6) ** 2 <= d <= r * r:
                bmp.set(x, y, rgba)


def _line_dashed(bmp: Bitmap, x0, y0, x1, y1, rgba):
    """Bresenham line drawn in 5-on / 3-off dashes — predicted bone."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    step = 0
    while True:
        if step % 8 < 5:
            for ox in (-1, 0):
                for oy in (-1, 0):
                    bmp.set(x0 + ox, y0 + oy, rgba)
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
        step += 1


def overlay(bmp: Bitmap, skel: Skeleton) -> Bitmap:
    """Return a copy of bmp with the skeleton drawn on it."""
    out = Bitmap(bmp.width, bmp.height, bytearray(bmp.data))
    _draw(out, skel, None)
    return out


def _draw(out: Bitmap, skel: Skeleton,
          tint: Tuple[int, int, int, int] = None) -> None:
    """Draw one skeleton onto `out` (in place).

    `tint` recolours the person for multi-person overlays: observed
    parts get the tint, predicted parts get it at half intensity, so
    the observed/predicted contract survives the colour coding.
    """
    if tint is None:
        strong, faint = BLUE, ORANGE
    else:
        strong = tint
        faint = tuple(min(255, int(c * 0.55) + 90) for c in tint[:3])
        faint = (*faint, 255)
    sx = out.width / skel.image_width
    sy = out.height / skel.image_height

    def pt(name):
        j = skel.get(name)
        return (j.x * sx, j.y * sy) if j else None

    for a, b in BONES:
        pa, pb = pt(a), pt(b)
        if not pa or not pb:
            continue
        both_obs = (skel.get(a).state == OBSERVED
                    and skel.get(b).state == OBSERVED)
        color = strong if both_obs else faint
        draw = _line if both_obs else _line_dashed
        draw(out, pa[0], pa[1], pb[0], pb[1], color)
    for j in skel.joints.values():
        r = max(2, out.width // 160)
        if j.state == OBSERVED:
            _disc(out, j.x * sx, j.y * sy, r, strong)
            _disc(out, j.x * sx, j.y * sy, 1, WHITE)
        else:
            _ring(out, j.x * sx, j.y * sy, r + 1, faint)

    # Facing arrow above the head when the estimator saw a side profile.
    facing = skel.orientation.get("facing")
    head = pt("head")
    if facing in ("left", "right") and head:
        sgn = -1 if facing == "left" else 1
        ay = max(2, head[1] - out.height * 0.06)
        ln = max(6, out.width * 0.08)
        tail_x, tip_x = head[0] - sgn * ln, head[0] + sgn * ln * 0.4
        _line(out, tail_x, ay, tip_x, ay, GREEN)
        head_sz = ln * 0.25
        _line(out, tip_x, ay, tip_x - sgn * head_sz, ay - head_sz * 0.6,
              GREEN)
        _line(out, tip_x, ay, tip_x - sgn * head_sz, ay + head_sz * 0.6,
              GREEN)


# per-person hues for overlay_multi — far apart on the wheel so two
# neighbours never read as the same figure
PEOPLE_TINTS = (
    (30, 120, 255, 255),   # blue
    (220, 40, 110, 255),   # magenta
    (0, 170, 110, 255),    # green
    (230, 150, 0, 255),    # amber
    (140, 60, 220, 255),   # violet
)


def overlay_multi(bmp: Bitmap, skels) -> Bitmap:
    """Draw every detected person on one image, each in its own hue.

    Person identity is colour; the observed/predicted contract is
    intensity (bright = observed, washed-out = predicted) — readable
    even where the hues would clash.
    """
    out = Bitmap(bmp.width, bmp.height, bytearray(bmp.data))
    for i, skel in enumerate(skels):
        _draw(out, skel, PEOPLE_TINTS[i % len(PEOPLE_TINTS)])
    return out


def overlay_multi_png(bmp: Bitmap, skels) -> bytes:
    return encode_png(overlay_multi(bmp, skels))


def overlay_png(bmp: Bitmap, skel: Skeleton) -> bytes:
    return encode_png(overlay(bmp, skel))

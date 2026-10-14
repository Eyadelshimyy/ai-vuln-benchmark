"""Person-likeness of a foreground blob — check the assumption.

The estimator's silent assumption is that a large foreground
component *is* a person. A lamp post, a cardboard box, or a dog
all survive the size gates and come out wearing a full skeleton —
wrongly confident `observed` joints on a non-human shape.

`assess(comp)` scores a component mask (rows of 0/1 bytes) on four
person-shape signals, each reported with its measured value:

- `aspect`: height/width ratio — standing people are tall
  (fallen/sitting people break this; it only weakens the score,
  never vetoes)
- `fill`: occupancy inside the bounding box — people are lumpy,
  boxes are dense
- `head_mass`: foreground fraction in the top fifth, in a narrow
  band — heads concentrate mass near the top centre
- `symmetry`: left/right mirrored overlap of the row profile —
  frontal people are roughly bilateral

Each signal maps to [0,1]; `score` is the mean. `person_like`
requires score ≥ `PERSON_LIKE_MIN` **and** every signal above
`SIGNAL_FLOOR` — a rectangle is tall and perfectly symmetric, so
mean alone can't reject it; its zero fill must veto. The verdict
is advisory — a low score means "the blob doesn't look like a
person shape", never "no person present" (poses outside the
signal assumptions lose points, and the caller decides).
"""

from __future__ import annotations

from typing import Dict, List

PERSON_LIKE_MIN = 0.5
SIGNAL_FLOOR = 0.1


def _rows(comp: List[bytearray]) -> List[List[int]]:
    out = []
    for row in comp:
        xs = [x for x, v in enumerate(row) if v]
        out.append(xs)
    return out


def _bbox(rows) -> tuple:
    ys = [y for y, xs in enumerate(rows) if xs]
    if not ys:
        return None
    xs = [x for y in ys for x in rows[y]]
    return min(xs), min(ys), max(xs), max(ys)


def assess(comp: List[bytearray]) -> Dict:
    """Score a component mask for person-likeness.

    Returns {person_like, score, signals:{...each measured value},
    state, basis}. Empty mask → person_like None / unmeasurable.
    """
    rows = _rows(comp)
    b = _bbox(rows)
    if b is None:
        return {"state": "unmeasurable",
                "basis": "mask has no foreground",
                "person_like": None, "score": None, "signals": {}}
    x0, y0, x1, y1 = b
    w, h = x1 - x0 + 1, y1 - y0 + 1
    area = sum(len(r) for r in rows)

    # aspect: height/width; people 1.2–4.0 → full credit, fades out
    # on both sides — a lamp post is taller than any person
    aspect = h / w
    if aspect < 1.2:
        a_sig = max(0.0, (aspect - 0.4) / 0.8)
    elif aspect <= 4.0:
        a_sig = 1.0
    else:
        a_sig = max(0.0, 1.0 - (aspect - 4.0) / 4.0)

    # fill: area/bbox — people are lumpy, not dense (<0.75) and not
    # empty (a 5% scatter of pixels is dust, not a body): full credit
    # inside the plausible band, fading on both sides
    fill = area / (w * h)
    if fill < 0.15:
        f_sig = max(0.0, fill / 0.15)
    elif fill <= 0.75:
        f_sig = 1.0
    else:
        f_sig = max(0.0, 1.0 - (fill - 0.75) / 0.2)

    # head_mass: share of area in top fifth, within centre 40% of
    # its own span at that height
    band = range(y0, y0 + max(1, h // 5))
    band_area = sum(len(rows[y]) for y in band)
    if band_area:
        xs = [x for y in band for x in rows[y]]
        bx0, bx1 = min(xs), max(xs)
        cx = (bx0 + bx1) / 2
        span = max(1.0, bx1 - bx0)
        centre = sum(1 for x in xs if abs(x - cx) <= span * 0.2)
        h_sig = min(1.0, (band_area / area) / 0.25) * (centre / band_area)
        h_sig = min(1.0, h_sig * 2.0)  # both halves needed → rescale
    else:
        h_sig = 0.0

    # symmetry: overlap of mirrored row spans about the bbox centre
    cx = (x0 + x1) / 2
    ov = tot = 0
    for xs in rows:
        if not xs:
            continue
        lo, hi = xs[0], xs[-1]
        mlo, mhi = 2 * cx - hi, 2 * cx - lo  # mirrored span
        ov += max(0, min(hi, mhi) - max(lo, mlo))
        tot += hi - lo + 1
    s_sig = ov / tot if tot else 0.0

    signals = {
        "aspect": {"value": round(aspect, 2), "score": round(a_sig, 2)},
        "fill": {"value": round(fill, 2), "score": round(f_sig, 2)},
        "head_mass": {"value": round(band_area / area, 2),
                      "score": round(h_sig, 2)},
        "symmetry": {"value": round(s_sig, 2), "score": round(s_sig, 2)},
    }
    score = (a_sig + f_sig + h_sig + s_sig) / 4
    floor = min(a_sig, f_sig, h_sig, s_sig)
    return {
        "person_like": score >= PERSON_LIKE_MIN
        and floor >= SIGNAL_FLOOR,
        "weakest_signal": min(signals, key=lambda k: signals[k]["score"]),
        "score": round(score, 3),
        "signals": signals,
        "state": "measured",
        "basis": "component shape features vs person-shape priors "
                 "(advisory; poses outside the priors score low)",
    }

"""Pose Engine: silhouette-based stick-figure extraction.

This is a deterministic, model-free estimator — Slice's baseline port.
It segments the figure from the background, reads the row/column profile
of the silhouette, and locates anatomical landmarks by profile features
(head blob, shoulder line, crotch split, arm protrusions).

Any joint whose image evidence is missing is left absent; the Prediction
Engine fills those in later, keeping observed vs predicted separate.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

from .anatomy import BODY_MODELS, DEFAULT_MODEL, select_model
from .bitmap import Bitmap
from .skeleton import (OBSERVED, Joint, Skeleton,
                       NECK_CLAVICLE_BASIS)


class PoseEstimator:
    """Port: anything that turns a Bitmap into a Skeleton."""

    name = "base"
    version = "0"

    def estimate(self, bmp: Bitmap, model: str = DEFAULT_MODEL) -> Skeleton:
        raise NotImplementedError


def _row_runs(mask: List[bytearray], y: int, w: int) -> List[Tuple[int, int]]:
    runs, start = [], -1
    row = mask[y]
    for x in range(w):
        if row[x] and start < 0:
            start = x
        elif not row[x] and start >= 0:
            runs.append((start, x - 1))
            start = -1
    if start >= 0:
        runs.append((start, w - 1))
    return runs


def _reaches_bottom(mask: List[bytearray], y: int,
                    run: Tuple[int, int], bottom: int, w: int) -> bool:
    """True when the run's x-band stays connected down to the bottom row.

    Legs reach the floor line; a dangling arm run ends mid-frame even
    when it x-overlaps a foot below it (wide feet, crouch/seated poses).
    Overlapping runs merge on the way down — an arm resting on a leg
    honestly counts as grounded.
    """
    x0, x1 = run
    for yy in range(y + 1, bottom + 1):
        nxt = [r for r in _row_runs(mask, yy, w)
               if r[1] >= x0 and r[0] <= x1]
        if not nxt:
            return False
        x0 = min(r[0] for r in nxt)
        x1 = max(r[1] for r in nxt)
    return True


def _oriented_masks(mask: List[bytearray], w: int, h: int):
    """Yield (deg, rotated mask, rw, rh) for 90°/-90°/180°.

    A lying figure is upright in exactly one of the first two; an
    inverted (headstand) figure in the third."""
    cw = [bytearray(h) for _ in range(w)]
    ccw = [bytearray(h) for _ in range(w)]
    r180 = [bytearray(w) for _ in range(h)]
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x]:
                cw[x][h - 1 - y] = 1
                ccw[w - 1 - x][y] = 1
                r180[h - 1 - y][w - 1 - x] = 1
    yield 90, cw, h, w
    yield -90, ccw, h, w
    yield 180, r180, w, h


def _unrotate(pt: Tuple[float, float], deg: int,
              w: int, h: int) -> Tuple[float, float]:
    """Map a joint found on a rotated mask back to image coords."""
    x, y = pt
    if deg == 90:
        return y, h - 1 - x
    if deg == -90:
        return w - 1 - y, x
    return w - 1 - x, h - 1 - y


def _head_band_width(mask: List[bytearray], w: int, h: int) -> int:
    """Widest single run in the top head band — the head blob is the
    densest wide structure at the figure's upright end; thin legs or
    a horizontal torso score far lower."""
    best = 0
    for y in range(min(h, max(1, int(h * 0.2)))):
        for r in _row_runs(mask, y, w):
            best = max(best, r[1] - r[0] + 1)
    return best


class HeuristicPoseEstimator(PoseEstimator):
    name = "heuristic-silhouette"
    version = "0.1.0"

    def __init__(self, max_dim: int = 512, bg_threshold: int = 40,
                 adaptive: bool = False, reject_shadow: bool = False,
                 clean: bool = False, threshold_offset: float = 0):
        self.max_dim = max_dim
        self.bg_threshold = bg_threshold
        self.adaptive = adaptive
        self.reject_shadow = reject_shadow
        self.clean = clean
        # added to whichever threshold the mask ends up using — the
        # perturbation knob `stability.probe` needs, since shifting
        # bg_threshold alone does not move an Otsu split
        self.threshold_offset = threshold_offset
        # (value, "otsu"|"fixed") from the last _mask call —
        # diagnostic surface, not part of the skeleton contract
        self.last_threshold = None
        self.last_shadow_removed = None

    # -- segmentation ----------------------------------------------------

    def _background(self, bmp: Bitmap) -> Tuple[int, int, int]:
        """Most common quantized color along the image border."""
        counts: dict = {}
        fallback: dict = {}
        w, h = bmp.width, bmp.height
        for x in range(0, w, 4):
            for y in (0, h - 1):
                r, g, b, a = bmp.get(x, y)
                key = (r // 32, g // 32, b // 32)
                fallback[key] = fallback.get(key, 0) + 1
                if a >= 128:
                    counts[key] = counts.get(key, 0) + 1
        for y in range(0, h, 4):
            for x in (0, w - 1):
                r, g, b, a = bmp.get(x, y)
                key = (r // 32, g // 32, b // 32)
                fallback[key] = fallback.get(key, 0) + 1
                if a >= 128:
                    counts[key] = counts.get(key, 0) + 1
        # transparent pixels carry no colour information; prefer
        # opaque samples, keep the old behaviour when none exist
        use = counts or fallback
        q = max(use, key=use.get)
        return q[0] * 32 + 16, q[1] * 32 + 16, q[2] * 32 + 16

    def _background_bands(self, bmp: Bitmap, n: int = 6
                          ) -> List[Tuple[int, int, int]]:
        """Per-band background estimates along the y axis.

        A single global mode breaks on gradient walls (the top of the
        frame and the bottom have different bg colors). Sampling the
        side borders per band lets the mask threshold track a smooth
        vertical drift; a band with no opaque border samples falls
        back to the global estimate.
        """
        w, h = bmp.width, bmp.height
        bands: List[Tuple[int, int, int]] = []
        for b in range(n):
            y0, y1 = b * h // n, (b + 1) * h // n
            counts: dict = {}
            for y in range(y0, y1, 4):
                for x in (0, w - 1):
                    r, g, bl, a = bmp.get(x, y)
                    if a < 128:
                        continue
                    key = (r // 32, g // 32, bl // 32)
                    counts[key] = counts.get(key, 0) + 1
            # the outer bands also see the top/bottom edge
            edge_y = y0 if b == 0 else (y1 - 1 if b == n - 1 else None)
            if edge_y is not None:
                for x in range(0, w, 4):
                    r, g, bl, a = bmp.get(x, edge_y)
                    if a < 128:
                        continue
                    key = (r // 32, g // 32, bl // 32)
                    counts[key] = counts.get(key, 0) + 1
            if counts:
                q = max(counts, key=counts.get)
                bands.append((q[0] * 32 + 16, q[1] * 32 + 16,
                              q[2] * 32 + 16))
            else:
                bands.append(self._background(bmp))
        return bands

    def _mask(self, bmp: Bitmap) -> List[bytearray]:
        if self.adaptive:
            mask = self._mask_adaptive(bmp, self._background(bmp))
        else:
            thr = self.bg_threshold + self.threshold_offset
            self.last_threshold = (thr, "fixed")
            mask = self._mask_fixed(bmp, self._background_bands(bmp), thr)
        if self.reject_shadow:
            from . import shadow
            shadow_m = shadow.shadow_pixels(bmp, self._background(bmp),
                                            mask)
            mask, self.last_shadow_removed = shadow.remove(
                mask, shadow_m)
        if self.clean:
            from . import morph
            mask = morph.clean(mask)
        return mask

    @staticmethod
    def _mask_fixed(bmp: Bitmap, bg_rgb,
                    thr: float) -> List[bytearray]:
        """Per-channel threshold mask. `bg_rgb` is one (r, g, b) or a
        list of per-band estimates from `_background_bands`."""
        bands = bg_rgb if isinstance(bg_rgb, list) else [bg_rgb]
        w, h = bmp.width, bmp.height
        n = len(bands)
        mask = [bytearray(w) for _ in range(h)]
        d = bmp.data
        for y in range(h):
            br, bg, bb = bands[min(n - 1, y * n // h)]
            row = mask[y]
            base = y * w * 4
            for x in range(w):
                i = base + x * 4
                if d[i + 3] < 128:
                    continue
                if (abs(d[i] - br) > thr or abs(d[i + 1] - bg) > thr
                        or abs(d[i + 2] - bb) > thr):
                    row[x] = 1
        return mask

    def _mask_adaptive(self, bmp: Bitmap,
                       bg_rgb) -> List[bytearray]:
        """Otsu-thresholded mask: per-pixel distance to background
        decides foreground, with the split chosen by the histogram.
        When no bimodal split exists, falls back to the fixed
        per-channel rule (method recorded on `last_threshold`)."""
        from . import adapt
        dist = adapt.distances(bmp, bg_rgb)
        thr, method = adapt.threshold(dist, fallback=self.bg_threshold)
        if method == "otsu":
            thr += self.threshold_offset
        self.last_threshold = (thr, method)
        if method != "otsu":
            return self._mask_fixed(bmp, self._background_bands(bmp),
                                    self.bg_threshold +
                                    self.threshold_offset)
        w, h = bmp.width, bmp.height
        mask = [bytearray(w) for _ in range(h)]
        d = bmp.data
        k = 0
        for y in range(h):
            row = mask[y]
            for x in range(w):
                if dist[k] > thr:
                    row[x] = 1
                k += 1
        return mask

    def _label_components(self, mask: List[bytearray], w: int,
                          h: int) -> Tuple[list, dict]:
        """4-connected component labelling. Returns (labels, sizes)
        where sizes maps label -> pixel count."""
        labels = [[0] * w for _ in range(h)]
        sizes: dict = {}
        label = 0
        for y0 in range(h):
            for x0 in range(w):
                if not mask[y0][x0] or labels[y0][x0]:
                    continue
                label += 1
                size = 0
                q = deque([(x0, y0)])
                labels[y0][x0] = label
                while q:
                    x, y = q.popleft()
                    size += 1
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if (0 <= nx < w and 0 <= ny < h and mask[ny][nx]
                                and not labels[ny][nx]):
                            labels[ny][nx] = label
                            q.append((nx, ny))
                sizes[label] = size
        return labels, sizes

    def _component_mask(self, labels, label: int, w: int,
                        h: int) -> List[bytearray]:
        comp = [bytearray(w) for _ in range(h)]
        for y in range(h):
            lr, cr = labels[y], comp[y]
            for x in range(w):
                if lr[x] == label:
                    cr[x] = 1
        return comp

    def _largest_component(self, mask: List[bytearray], w: int, h: int
                           ) -> Tuple[List[bytearray], int]:
        labels, sizes = self._label_components(mask, w, h)
        if not sizes:
            return [bytearray(w) for _ in range(h)], 0
        best = max(sizes, key=sizes.get)
        return self._component_mask(labels, best, w, h), sizes[best]

    # -- profile features -------------------------------------------------

    def _profile(self, mask, w, h):
        rows = []   # (min_x, max_x, count) per row, None if empty
        for y in range(h):
            xs = [x for x in range(w) if mask[y][x]]
            rows.append((xs[0], xs[-1], len(xs)) if xs else None)
        return rows

    def _bbox(self, rows, w, h):
        ys = [y for y, r in enumerate(rows) if r]
        if not ys:
            return None
        top, bot = ys[0], ys[-1]
        l = min(r[0] for r in rows if r)
        r = max(r[1] for r in rows if r)
        return l, top, r, bot

    def _centroid(self, mask, w, h) -> Tuple[float, float]:
        sx = sy = n = 0
        for y in range(h):
            row = mask[y]
            for x in range(w):
                if row[x]:
                    sx += x
                    sy += y
                    n += 1
        return (sx / n, sy / n) if n else (w / 2, h / 2)

    def _widest_row(self, rows, y0: int, y1: int) -> int:
        best, best_w = y0, -1
        for y in range(max(0, y0), min(len(rows), y1)):
            r = rows[y]
            if r and r[2] > best_w:
                best, best_w = y, r[2]
        return best

    def _crotch_row(self, mask, w, y0: int, y1: int, torso_runs) -> Optional[int]:
        """First row below the torso band whose mask splits into two runs."""
        for y in range(max(0, y0), min(y1, len(mask) - 1)):
            runs = _row_runs(mask, y, w)
            torso_w = torso_runs[1] - torso_runs[0] if torso_runs else 0
            # A split is a gap between *adjacent* runs; comparing the
            # first and last run spans everything in between and calls
            # "arm | torso | arm" a crotch at chest height. The gap
            # threshold scales with torso width but stays ≤4px — a
            # real leg gap doesn't grow with frame size, and the
            # estimator's own downscale shrinks it further.
            if any(runs[i + 1][0] - runs[i][1]
                   >= max(2, min(4, torso_w * 0.08))
                   for i in range(len(runs) - 1)):
                return y
        return None

    # -- main ---------------------------------------------------------------

    def estimate(self, bmp: Bitmap, model: str = DEFAULT_MODEL) -> Skeleton:
        small = bmp.downscale(self.max_dim)
        w, h = small.width, small.height
        mask = self._mask(small)
        thr = self.last_threshold  # capture this call's decision — a
        comp, size = self._largest_component(mask, w, h)  # later call
        sk = self._estimate_oriented(small, comp, size, w, h, model)
        # may overwrite the shared attribute, so carry it on the
        # skeleton instead of reading it back
        sk.mask_threshold = thr
        return sk

    def _estimate_oriented(self, small, comp, size, w, h,
                           model: str) -> Skeleton:
        """Estimate one component with the orientation retry.

        The upright scan silently mismeasures a lying or inverted
        figure — sometimes even passing the consistency audit with
        fabricated joints. Estimate all four orientations and keep
        the most consistent skeleton with the strongest head-band
        evidence; joints map back to image space and record the
        rotation in their basis."""
        sk = self._estimate_component(small, comp, size, w, h, model)
        # No early exit when the upright scan starves: a lying figure
        # can fail every upright gate (body_h < 24px) yet skeletonize
        # cleanly once rotated — the retry exists for exactly that
        # case. With no base skeleton the adoption rule stays strict:
        # audit of an empty skeleton counts as one issue, so only a
        # fully clean rotated candidate qualifies.
        from . import consistency
        base_issues = len(consistency.audit(sk, model))
        base_headw = _head_band_width(comp, w, h)
        best, best_deg = sk, 0
        best_key = None
        for deg, comp_r, rw, rh in _oriented_masks(comp, w, h):
            cand = self._estimate_component(small, comp_r, size,
                                            rw, rh, model)
            if not cand.joints:
                continue
            issues = len(consistency.audit(cand, model))
            headw = _head_band_width(comp_r, rw, rh)
            # Rotate only on strict evidence: fewer audit issues or a
            # decisively stronger head band — a sideways blob (wide
            # hand, dress hem) only looks head-sized and must not flip
            # an upright figure.
            # a rotated candidate that finds fewer joints than the
            # upright scan is regression, not reorientation; and
            # rotating without stronger head-band evidence just
            # re-labels feet as head — an audit-clean fabrication on
            # the rotated mask must not flip real upright evidence
            qualifies = (len(cand.joints) >= len(sk.joints)
                         and issues <= base_issues
                         and headw > base_headw * 1.3)
            if not qualifies:
                continue
            key = (issues, -len(cand.joints), -headw)
            if best_key is None or key < best_key:
                best_key, best, best_deg = key, cand, deg
        if not best_deg:
            return sk
        for j in best.joints.values():
            j.x, j.y = _unrotate((j.x, j.y), best_deg, w, h)
            j.basis = (j.basis + "; " if j.basis else "") \
                + f"estimated on {best_deg}deg-rotated mask"
        if best.centroid is not None:
            best.centroid = _unrotate(best.centroid, best_deg, w, h)
        # orientation cues were read in the rotated frame — disclose
        # the rotation and, for 180°, mirror the signed left/right
        # cues back (rotated-x flips, so "left" there is "right" here)
        ori = dict(best.orientation or {})
        ori["estimated_on_rotated_deg"] = best_deg
        if best_deg == 180:
            if ori.get("facing") == "left":
                ori["facing"] = "right"
            elif ori.get("facing") == "right":
                ori["facing"] = "left"
            ori["head_shift"] = -ori.get("head_shift", 0.0)
        best.orientation = ori
        best.image_width, best.image_height = w, h
        return best

    def estimate_multi(self, bmp: Bitmap, model: str = DEFAULT_MODEL,
                       top_k: int = 4,
                       min_fraction: float = 0.005) -> List[Skeleton]:
        """One skeleton per large foreground component, biggest first.

        The honest-contract multi-person path (AUDIT P0-1): each
        connected component is estimated independently — two people
        whose silhouettes touch merge into one component and remain a
        single (wrong) skeleton; the API reports what the pixels
        support, it does not guess at occluded overlap. Components
        that fail the single-person gates (too small, too short) yield
        no skeleton rather than a noisy one.
        """
        small = bmp.downscale(self.max_dim)
        w, h = small.width, small.height
        mask = self._mask(small)
        thr = self.last_threshold  # captured once — one mask per call
        labels, sizes = self._label_components(mask, w, h)
        out: List[Skeleton] = []
        for lab in sorted(sizes, key=sizes.get, reverse=True)[:top_k]:
            if sizes[lab] < w * h * min_fraction:
                break
            comp = self._component_mask(labels, lab, w, h)
            sk = self._estimate_oriented(small, comp, sizes[lab],
                                         w, h, model)
            sk.mask_threshold = thr
            if sk.joints:
                out.append(sk)
        return out

    def estimate_split(self, bmp: Bitmap, model: str = DEFAULT_MODEL,
                       top_k: int = 4, min_dist: float = 5.0
                       ) -> List[Skeleton]:
        """Like `estimate_multi`, but each big component is first
        offered to `slice.split`'s distance-field watershed — a
        claimed split for silhouettes that touch. Sub-regions get the
        same component gates and an independent skeleton; the split
        hypothesis is still recorded on every joint's basis
        ("split region"). Fewer cores than seeds, or one whole
        component, both pass through unchanged — the pixels decide.
        """
        from . import split as _split
        small = bmp.downscale(self.max_dim)
        w, h = small.width, small.height
        mask = self._mask(small)
        thr = self.last_threshold  # captured once — one mask per call
        labels, sizes = self._label_components(mask, w, h)
        out: List[Skeleton] = []
        for lab in sorted(sizes, key=sizes.get, reverse=True)[:top_k]:
            if sizes[lab] < w * h * 0.005:
                break
            comp = self._component_mask(labels, lab, w, h)
            # splitting is only claimed with positive evidence:
            # >=2 cores in the head band (top 20% of the bbox) that
            # are horizontally distinct — a single person's head and
            # chest don't count, two people's heads do.
            peaks = _split.peaks(comp, min_dist=min_dist,
                                 top_k=top_k)
            rows = self._profile(comp, w, h)
            bbox = self._bbox(rows, w, h)
            head_line = (bbox[1] + (bbox[3] - bbox[1]) * 0.2
                         if bbox else h)
            spread = (bbox[2] - bbox[0]) * 0.25 if bbox else 0
            heads = [(x, y) for x, y, _ in peaks if y <= head_line]
            heads = [hpt for i, hpt in enumerate(heads)
                     if all(abs(hpt[0] - o[0]) > spread
                            for o in heads[:i])]
            if len(heads) < 2:
                subs = [comp]
            else:
                subs, _ = _split.split(comp, seeds=heads)
            for sub in subs:
                size = sum(sum(r) for r in sub)
                sk = self._estimate_component(small, sub, size,
                                              w, h, model)
                sk.mask_threshold = thr
                if not sk.joints:
                    continue
                for j in sk.joints.values():
                    j.basis = (j.basis + "; split region"
                               if j.basis else "split region")
                out.append(sk)
        return out

    def _estimate_component(self, small, comp, size, w, h,
                            model: str) -> Skeleton:
        sk = Skeleton(image_width=w, image_height=h)
        if size < w * h * 0.005:
            return sk  # no person-sized foreground
        rows = self._profile(comp, w, h)
        bbox = self._bbox(rows, w, h)
        if not bbox:
            return sk
        left, top, right, bottom = bbox
        body_h = bottom - top + 1
        body_w = right - left + 1
        if body_h < 24:
            return sk

        applied_model = model if model in BODY_MODELS else DEFAULT_MODEL
        prior = BODY_MODELS[applied_model]
        head_h = max(4.0, body_h * prior["head_ratio"])
        sk.centroid = self._centroid(comp, w, h)

        def put(name, x, y, conf, basis):
            sk.set(Joint(name, round(x, 2), round(y, 2),
                         round(min(max(conf, 0.0), 1.0), 3), OBSERVED, basis))

        # Head: centroid of the top head_h band.
        hx0, hx1, hn = 0, 0, 0
        for y in range(top, min(bottom, int(top + head_h)) + 1):
            r = rows[y]
            if r:
                span = (r[0] + r[1]) / 2 * r[2]
                hx0 += span
                hn += r[2]
                hx1 = y
        head_cx = hx0 / hn if hn else (left + right) / 2
        head_cy = top + head_h / 2
        put("head", head_cx, head_cy, 0.85, "top blob centroid")

        neck_band = top + head_h

        # Shoulders: widest row in the upper body band.
        sh_row = self._widest_row(rows, int(neck_band),
                                  int(top + body_h * 0.35))
        sr = rows[sh_row] or (left, right, body_w)
        put("shoulder_l", sr[0], sh_row, 0.8, "widest upper row")
        put("shoulder_r", sr[1], sh_row, 0.8, "widest upper row")

        # Neck = clavicle midpoint: just below the shoulder line, not
        # the head-band bottom (that lands at the chin — a ~head-height
        # systematic bias measured by the bias profile).
        neck_y = sh_row + max(2, int(head_h * 0.15))
        put("neck", (sr[0] + sr[1]) / 2, neck_y, 0.7,
            NECK_CLAVICLE_BASIS)

        # Pelvis / hips: crotch split, else widest row in the hip band.
        # Torso column = the mask run under the spine at chest height;
        # arm protrusions are the runs lying outside it.
        chest_y = (sh_row + int(top + body_h * 0.55)) / 2
        cx_spine = (sr[0] + sr[1]) / 2
        torso_run = next(
            (r for r in _row_runs(comp, int(chest_y), w)
             if r[0] <= cx_spine <= r[1]),
            (min(sr[0], sr[1]), max(sr[0], sr[1])))
        # Horizontal arms (T-pose, raised arms): when the widest row
        # spans far beyond the torso column it is an arm strip, not
        # shoulders. Shoulders sit at the torso edge, wrist at the
        # strip tip, elbow at their midpoint.
        torso_w = torso_run[1] - torso_run[0]
        horizontal_arms = set()
        if torso_w > 0 and sr[1] - sr[0] > torso_w * 1.6 + 4:
            for side, sign in (("l", -1), ("r", 1)):
                edge = torso_run[0] if sign < 0 else torso_run[1]
                tip = sr[0] if sign < 0 else sr[1]
                if abs(tip - edge) > torso_w * 0.5:
                    put(f"shoulder_{side}", edge, sh_row, 0.7,
                        "torso edge at arm strip")
                    put(f"elbow_{side}", (edge + tip) / 2, sh_row, 0.6,
                        "arm strip midpoint")
                    put(f"wrist_{side}", tip, sh_row, 0.65,
                        "arm strip tip")
                    horizontal_arms.add(side)
        crotch = self._crotch_row(comp, w, int(top + body_h * 0.45),
                                  int(top + body_h * 0.72), torso_run)
        if crotch is None:
            hip_row = self._widest_row(rows, int(top + body_h * 0.45),
                                       int(top + body_h * 0.65))
            crotch = hip_row + max(2, int(head_h * 0.4))
            hip_conf = 0.55
            hip_basis = "widest hip-band row"
        else:
            hip_row = crotch - 1
            hip_conf = 0.75
            hip_basis = "crotch split row"
        # Hip width = the torso-column run at the hip row, not the
        # row's outermost pixels — arms dangling beside the torso
        # would inflate "hip" to arm-to-arm span.
        hip_run = next(
            (r for r in _row_runs(comp, hip_row, w)
             if r[0] <= cx_spine <= r[1]),
            rows[hip_row] or torso_run)
        hr = hip_run
        put("pelvis", (hr[0] + hr[1]) / 2, hip_row, hip_conf, hip_basis)
        put("hip_l", hr[0], hip_row, hip_conf, hip_basis)
        put("hip_r", hr[1], hip_row, hip_conf, hip_basis)

        chest_y = (sh_row + hip_row) / 2
        cr = rows[int(chest_y)] or torso_run
        put("chest", (cr[0] + cr[1]) / 2, chest_y, 0.65,
            "midpoint shoulders-pelvis")
        # The spine joint is the mirroring axis for predictions and
        # the curvature reference for spine.curve — it must sit on
        # the torso axis, not the head centroid: in profile the head
        # leans toward the faced side (head_shift), which would pull
        # the axis off the body exactly when mirrored prediction is
        # needed most. Measure the torso column at spine height —
        # the axis itself tilts with the torso, so a chest-height
        # run alone still misses it.
        spine_y = (neck_y + hip_row) / 2
        spine_run = next(
            (r for r in _row_runs(comp, int(spine_y), w)
             if r[0] <= cx_spine <= r[1]),
            torso_run)
        put("spine", (spine_run[0] + spine_run[1]) / 2, spine_y, 0.6,
            "axis midpoint")

        # Legs: below the crotch, split the row runs.
        legs_split = False
        for side, take in (("l", 0), ("r", -1)):
            knee_y = crotch + (bottom - crotch) * 0.55
            kr = _row_runs(comp, min(int(knee_y), h - 1), w)
            ar = _row_runs(comp, bottom, w)
            if len(kr) >= 2 and len(ar) >= 2:
                legs_split = True
                # Anchor each leg to its foot run — picking the
                # leftmost/rightmost knee-row run grabs a dangling
                # arm that still reaches below knee height.
                arun = ar[take] if len(ar) > abs(take) else ar[0]
                fcx = (arun[0] + arun[1]) / 2
                krun = next(
                    (r for r in kr if r[0] <= fcx <= r[1]), kr[take])
                put(f"knee_{side}", (krun[0] + krun[1]) / 2, knee_y, 0.7,
                    "leg run at knee height")
                put(f"ankle_{side}", (arun[0] + arun[1]) / 2, bottom - 1,
                    0.7, "leg run at bottom")
                put(f"foot_{side}", (arun[0] + arun[1]) / 2, bottom,
                    0.6, "silhouette bottom")
            elif len(ar) >= 1:
                run = ar[0] if side == "l" else ar[-1]
                # Merged legs: place both on the merged run at low conf.
                put(f"ankle_{side}", (run[0] + run[1]) / 2, bottom - 1,
                    0.35, "merged leg run")
                put(f"foot_{side}", (run[0] + run[1]) / 2, bottom,
                    0.3, "merged leg run")
        if not legs_split:
            for side in ("l", "r"):
                put(f"knee_{side}", (hr[0] + hr[1]) / 2,
                    crotch + (bottom - crotch) * 0.55, 0.3,
                    "legs not separable")

        # Arms: silhouette protrusions beside the torso column, tracked
        # down past the hips so dangling hands are still found.
        crotch_y = crotch
        # Feet anchor the legs: a run below the torso is leg when its
        # x-band still reaches the bottom row — x-overlap alone wrongly
        # claims dangling arms beside wide feet (crouch/seated) as leg.
        for side, sign in (("l", -1), ("r", 1)):
            if side in horizontal_arms:
                continue  # strip joints already placed
            shoulder = sk.get(f"shoulder_{side}")
            tx = torso_run[0] if sign < 0 else torso_run[1]
            cand = []
            band = None  # x-range of the arm beside the torso
            # Start below the head band, not at the shoulder row —
            # arms raised above shoulder line (V-pose) live in the
            # rows between neck and shoulders that a shoulder-anchored
            # scan never reaches.
            for y in range(int(top + head_h), bottom + 1):
                runs = _row_runs(comp, y, w)
                if y <= crotch_y:
                    for run in runs:
                        if (sign < 0 and run[1] < tx - 2) \
                                or (sign > 0 and run[0] > tx + 2):
                            cand += [(x, y) for x in range(run[0], run[1] + 1)]
                else:
                    # Below the torso, legs are the runs whose band still
                    # reaches the bottom row; any other run on
                    # the arm's side inside the arm band is a limb. The
                    # band grows with accepted runs so arms drifting
                    # outward stay tracked.
                    if len(runs) <= 2 or band is None:
                        continue
                    for run in runs:
                        mid = (run[0] + run[1]) / 2
                        if (sign < 0) != (mid < cx_spine):
                            continue
                        if _reaches_bottom(comp, y, run, bottom, w):
                            continue  # leg — connected to the floor line
                        if run[1] < band[0] - 4 or run[0] > band[1] + 4:
                            continue
                        cand += [(x, y)
                                 for x in range(run[0], run[1] + 1)]
                        band = (min(band[0], run[0]),
                                max(band[1], run[1]))
                if cand and y <= crotch_y:
                    band = (min(p[0] for p in cand),
                            max(p[0] for p in cand))
            if cand and shoulder:
                far = max(cand,
                          key=lambda p: (p[0] - shoulder.x) ** 2
                          + (p[1] - shoulder.y) ** 2)
                # Elbow rides at the upper-arm fraction of the
                # *measured* shoulder→wrist extent. Using the prior
                # arm_len here placed elbows ~30px too high whenever
                # the silhouette arm ran longer than the ratio says.
                reach = ((far[0] - shoulder.x) ** 2
                         + (far[1] - shoulder.y) ** 2) ** 0.5
                frac = prior["upper_arm_ratio"] / (
                    prior["upper_arm_ratio"]
                    + prior["forearm_ratio"])
                elbow = max(
                    cand,
                    key=lambda p: -abs(((p[0] - shoulder.x) ** 2
                                       + (p[1] - shoulder.y) ** 2) ** 0.5
                                     - reach * frac))
                conf = min(0.85, 0.4 + len(cand) / (body_h * 8))
                put(f"wrist_{side}", far[0], far[1], conf,
                    "arm blob extremity")
                put(f"elbow_{side}", elbow[0], elbow[1], conf - 0.05,
                    "arm blob mid-extent")

        # Orientation + body model selection.
        # Symmetry is measured around the torso axis, not the image
        # centre — a front-facing person framed off-centre reads
        # symmetric about their own axis but lopsided about w//2,
        # which flipped facing to three-quarter/profile purely by
        # composition.
        sym = self._symmetry(comp, w, int(crotch), bottom,
                             int(cx_spine))
        aspect = body_w / body_h
        sh_span = abs(sr[1] - sr[0]) + 1
        torso_cx = (torso_run[0] + torso_run[1]) / 2
        # Facing direction: the topmost silhouette row (top of the head)
        # leans toward the faced side in profile; row centroid is cleaner
        # than the whole-head centroid, which the neck mass cancels out.
        top_run = rows[top] if 0 <= top < len(rows) else None
        head_shift = ((top_run[0] + top_run[1]) / 2 - torso_cx
                      if top_run else head_cx - torso_cx)
        profile = sh_span < body_h * 0.12 or (not legs_split and aspect < 0.3)
        if legs_split and sym > 0.75:
            facing, fconf = "front", min(0.8, sym)
        elif profile:
            # Direction from where the head blob leans off the torso axis.
            if head_shift < -head_h * 0.1:
                facing = "left"
            elif head_shift > head_h * 0.1:
                facing = "right"
            else:
                facing = "side"
            fconf = 0.55
        else:
            facing, fconf = "three-quarter", 0.4
        sk.orientation = {"facing": facing, "confidence": round(fconf, 3),
                          "symmetry": round(sym, 3),
                          "head_shift": round(head_shift, 2)}

        measured_head_ratio = (hx1 - top + 1) / body_h if hn else prior["head_ratio"]
        mname, mconf = select_model(measured_head_ratio)
        sk.body_model = {"name": mname,
                         "label": BODY_MODELS[mname]["label"],
                         "confidence": mconf,
                         "measured_head_ratio": round(measured_head_ratio, 3),
                         # the prior table actually used to place the
                         # observed joints — may differ from `name`
                         # (measured selection) and from the table
                         # predict.complete applies to missing joints
                         "prior": applied_model,
                         "state": "estimated"}
        return sk

    def _symmetry(self, mask, w, y0, y1, cx) -> float:
        """Left/right run-width agreement across the lower body,
        measured around the body axis `cx`."""
        diffs = 0
        n = 0
        for y in range(y0, min(y1, len(mask))):
            row = mask[y]
            lw = sum(row[x] for x in range(0, cx))
            rw = sum(row[x] for x in range(cx, w))
            if lw + rw == 0:
                continue
            diffs += abs(lw - rw) / (lw + rw)
            n += 1
        return 1 - (diffs / n) if n else 0.0

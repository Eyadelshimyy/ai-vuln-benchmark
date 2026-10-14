"""Contact splitting — best-effort sub-division of one component.

`estimate_multi` is honest about touching people: merged silhouettes
are one connected component, so it reports a single skeleton rather
than guessing. This module adds a *claimed* split for that case:

  1. `distance_transform` finds each pixel's depth inside the figure.
     Local maxima are round-ish cores — a head, a chest, a shoulder.
  2. `peaks` returns the strongest of them (non-max suppressed).
  3. `split` grows each peak by multi-source BFS until the foreground
     is assigned — a watershed without barriers. Whichever core a
     pixel is geodesically closest to owns it.

The result is a hypothesis, not a measurement: callers receive the
seed list with each mask so they can weight or reject the split.
Peaks are body-part cores (head, chest, hip), NOT persons — one
person may yield several cores, and two fused people may share one.
Two people standing apart stay untouched (separate components
upstream); this is only for the fused case.
"""

from __future__ import annotations

from collections import deque
from typing import List, Sequence, Tuple

from .distfield import distance_transform

# pixels shallower than this can't be a body-part core
MIN_PEAK_DIST = 5.0
# two peaks closer than this are the same blob
NMS_RADIUS_FACTOR = 1.6

Mask = List[bytearray]


def _empty(w: int, h: int) -> Mask:
    return [bytearray(w) for _ in range(h)]


def peaks(mask: Mask, *, min_dist: float = MIN_PEAK_DIST,
          top_k: int = 8) -> List[Tuple[int, int, float]]:
    """Distance-field ridge cores, strongest first.

    A pixel counts when `d >= every neighbour` — ties allowed, since
    real silhouettes produce flat plateaus, not point maxima.
    8-connected candidate pixels are merged into one cluster whose
    seed sits on the plateau's deepest pixel (centroid on ties);
    surviving clusters then get non-max suppressed: a weaker peak
    within `NMS_RADIUS_FACTOR * min_dist` px of a stronger one is
    dropped.
    """
    if not mask or not mask[0]:
        return []
    h, w = len(mask), len(mask[0])
    dist = distance_transform(mask)
    is_max = [bytearray(w) for _ in range(h)]
    for y in range(h):
        for x in range(w):
            d = dist[y * w + x]
            if d < min_dist:
                continue
            top = True
            for dy in (-1, 0, 1):
                ny = y + dy
                if not 0 <= ny < h:
                    continue
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    if 0 <= nx < w and dist[ny * w + nx] > d:
                        top = False
                        break
                if not top:
                    break
            if top:
                is_max[y][x] = 1
    # merge 8-connected candidate pixels -> one cluster
    seen = [bytearray(w) for _ in range(h)]
    clusters = []
    for y in range(h):
        for x in range(w):
            if not is_max[y][x] or seen[y][x]:
                continue
            cells = []
            q = deque([(x, y)])
            seen[y][x] = 1
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < w and 0 <= ny < h
                                and is_max[ny][nx] and not seen[ny][nx]):
                            seen[ny][nx] = 1
                            q.append((nx, ny))
            clusters.append(cells)
    cand = []
    for cells in clusters:
        best = max(dist[cy * w + cx] for cx, cy in cells)
        deep = [(cx, cy) for cx, cy in cells
                if dist[cy * w + cx] == best]
        sx = sum(c[0] for c in deep) // len(deep)
        sy = sum(c[1] for c in deep) // len(deep)
        cand.append((sx, sy, best))
    cand.sort(key=lambda p: -p[2])
    radius2 = (min_dist * NMS_RADIUS_FACTOR) ** 2
    out: List[Tuple[int, int, float]] = []
    for x, y, d in cand:
        if len(out) >= top_k:
            break
        if all((x - sx) ** 2 + (y - sy) ** 2 >= radius2
               for sx, sy, _ in out):
            out.append((x, y, d))
    return out


def split(mask: Mask, *, seeds: Sequence[Tuple[int, int]] = (),
          top_k: int = 4, min_dist: float = MIN_PEAK_DIST
          ) -> Tuple[List[Mask], List[Tuple[int, int, float]]]:
    """Assign each foreground pixel to its geodesically nearest seed.

    Multi-source BFS over the silhouette: every pixel inherits the
    label of the first seed that reaches it, so regions grow along
    the body rather than in straight lines. Returns
    `(masks, used_seeds)` — one mask per seed, or `([], [])` for an
    empty input. With no seeds found the input comes back whole:
    `([mask], [])` — nothing to split into.
    """
    if not mask or not mask[0]:
        return [], []
    h, w = len(mask), len(mask[0])
    if not seeds:
        found = peaks(mask, min_dist=min_dist, top_k=top_k)
    else:
        # snap caller-given seeds to the nearest foreground pixel;
        # a seed that hits background is dropped, not forced
        found = []
        for x, y in seeds:
            p = _nearest_fg(mask, w, h, x, y)
            if p is not None:
                found.append((p[0], p[1], -1.0))
    if len(found) <= 1:
        return [mask], found
    labels = [[0] * w for _ in range(h)]
    q = deque()
    for i, (x, y, _) in enumerate(found, 1):
        if mask[y][x]:
            labels[y][x] = i
            q.append((x, y))
    while q:
        x, y = q.popleft()
        lab = labels[y][x]
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= nx < w and 0 <= ny < h and mask[ny][nx]
                    and not labels[ny][nx]):
                labels[ny][nx] = lab
                q.append((nx, ny))
    masks = [_empty(w, h) for _ in found]
    for y in range(h):
        for x in range(w):
            lab = labels[y][x]
            if lab:
                masks[lab - 1][y][x] = 1
    return masks, found


def _nearest_fg(mask: Mask, w: int, h: int,
                x: int, y: int) -> Tuple[int, int]:
    """BFS outward from (x, y) to the closest foreground pixel."""
    if 0 <= x < w and 0 <= y < h and mask[y][x]:
        return (x, y)
    seen = { (x, y) }
    q = deque([(x, y)])
    while q:
        cx, cy = q.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy),
                       (cx, cy - 1), (cx, cy + 1)):
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in seen:
                continue
            if mask[ny][nx]:
                return (nx, ny)
            seen.add((nx, ny))
            q.append((nx, ny))
        if len(seen) > 4096:
            break
    return None


def unassigned(mask: Mask, masks: Sequence[Mask]) -> int:
    """Foreground pixels no seed claimed — should be zero after split."""
    h, w = len(mask), len(mask[0])
    n = 0
    for y in range(h):
        for x in range(w):
            if mask[y][x] and not any(m[y][x] for m in masks):
                n += 1
    return n

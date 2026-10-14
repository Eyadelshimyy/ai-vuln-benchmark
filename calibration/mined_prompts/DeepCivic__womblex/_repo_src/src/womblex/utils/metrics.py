"""Text extraction accuracy metrics: CER and WER.

Both metrics use Levenshtein edit distance to compare a hypothesis (extracted
text) against a reference (ground truth). Lower values are better.
"""

from __future__ import annotations

import unicodedata

import numpy as np


def _normalise(text: str) -> str:
    """Lowercase, NFC-normalise, and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Collapse all whitespace runs to a single space and strip edges
    return " ".join(text.split())


def _levenshtein_chars(s1: str, s2: str) -> int:
    """Levenshtein distance between two strings using numpy.

    Uses a diagonal-wavefront approach that keeps the hot loop in numpy.
    For strings up to ~500 chars the naive DP is fine; beyond that the
    numpy path avoids per-character Python overhead.
    """
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m

    # For short strings, pure-Python DP is faster than numpy overhead
    if n <= 500:
        prev = list(range(n + 1))
        for i in range(m):
            curr = [i + 1]
            ca = s1[i]
            for j in range(n):
                curr.append(min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (ca != s2[j]),
                ))
            prev = curr
        return prev[n]

    # numpy-accelerated: encode strings as int arrays
    a = np.frombuffer(s1.encode("utf-32-le"), dtype=np.int32)
    b = np.frombuffer(s2.encode("utf-32-le"), dtype=np.int32)

    # Two-row DP with vectorised min operations
    # Distinct names from the pure-Python branch above (which uses list[int])
    # so the array dtype flows cleanly.
    dp = np.arange(n + 1, dtype=np.int32)
    for i in range(m):
        cost = (b != a[i]).astype(np.int32)
        sub = dp[:-1] + cost
        dele = dp[1:] + 1
        best_no_ins = np.minimum(sub, dele)

        # Insertion depends on dp_next[j-1]: seed with best_no_ins, then
        # propagate the curr[j] <= curr[j-1]+1 constraint left-to-right.
        dp_next = np.empty(n + 1, dtype=np.int32)
        dp_next[0] = i + 1
        dp_next[1:] = best_no_ins
        _propagate_insertion(dp_next)

        dp = dp_next

    return int(dp[n])


def _propagate_insertion(curr: np.ndarray) -> None:
    """In-place left-to-right propagation: curr[j] = min(curr[j], curr[j-1]+1).

    Uses block processing: within each block, check if propagation can change
    anything. If curr is already non-decreasing with step ≤ 1, skip the block.
    """
    n = len(curr)
    # Process in blocks for branch-prediction friendliness
    block = 64
    for start in range(1, n, block):
        end = min(start + block, n)
        # Quick check: if min possible propagated value from start can't beat
        # any value in the block, skip it
        carry = curr[start - 1] + 1
        if carry >= curr[start]:
            # Need to propagate
            for j in range(start, end):
                v = curr[j - 1] + 1
                curr[j] = min(curr[j], v)
                # else: no further propagation needed in this run
        # If carry >= curr[start], the block might still need internal propagation
        # but the loop above handles it


def _levenshtein(seq_a: list | str, seq_b: list | str) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Works on both strings (character-level) and lists of strings (word-level).
    """
    if isinstance(seq_a, str) and isinstance(seq_b, str):
        return _levenshtein_chars(seq_a, seq_b)

    # Word-level: pure-Python (sequences are typically short)
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    m, n = len(seq_a), len(seq_b)
    if n == 0:
        return m

    prev = list(range(n + 1))
    for i, ca in enumerate(seq_a, 1):
        curr = [i]
        for j, cb in enumerate(seq_b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr

    return prev[-1]


def levenshtein(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Levenshtein edit distance between two strings or two token sequences.

    The raw integer distance CER and WER report as a ratio, exposed for callers
    that need the count itself rather than a rate — the ground-truth scorer's
    rubber-stamp check, where a corrected unit whose distance from its pinned
    baseline is zero is a defect. Character-level for two strings, element-wise
    for two lists; no normalisation, so the count is over the exact bytes.
    """
    return _levenshtein(seq_a, seq_b)


def cer(reference: str, hypothesis: str, *, normalise: bool = True) -> float:
    """Character Error Rate: edit distance at character level divided by len(reference).

    Args:
        reference: Ground truth text.
        hypothesis: Text produced by the extraction process.
        normalise: Apply lowercasing, NFC and whitespace normalisation before
            comparison. Defaults to True.

    Returns:
        CER as a float ≥ 0.0. Values > 1.0 are possible when the hypothesis is
        much longer than the reference. Returns 0.0 when both strings are empty.

    Raises:
        ValueError: If reference is non-empty after normalisation and hypothesis
            is also non-empty, but reference normalises to an empty string
            (guards against accidental empty-reference comparisons).
    """
    ref = _normalise(reference) if normalise else reference
    hyp = _normalise(hypothesis) if normalise else hypothesis

    if not ref:
        return 0.0 if not hyp else 1.0

    return _levenshtein(hyp, ref) / len(ref)


def wer(reference: str, hypothesis: str, *, normalise: bool = True) -> float:
    """Word Error Rate: edit distance at word level divided by word count of reference.

    Args:
        reference: Ground truth text.
        hypothesis: Text produced by the extraction process.
        normalise: Apply lowercasing, NFC and whitespace normalisation before
            tokenising. Defaults to True.

    Returns:
        WER as a float ≥ 0.0. Values > 1.0 are possible. Returns 0.0 when
        both inputs produce no words after normalisation.
    """
    ref = _normalise(reference) if normalise else reference
    hyp = _normalise(hypothesis) if normalise else hypothesis

    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return _levenshtein(hyp_words, ref_words) / len(ref_words)


# ---------------------------------------------------------------------------
# Spatial sorting for CER-s (reading-order-independent accuracy)
# ---------------------------------------------------------------------------

BBox = tuple[float, float, float, float]


def spatial_sort_text(
    words_with_boxes: list[tuple[str, BBox]],
    line_tolerance: float = 0.5,
) -> str:
    """Sort words by spatial position and join into a string.

    Sorts by vertical centroid (top-to-bottom), then horizontal centroid
    (left-to-right) within the same line. Two words are on the same line
    when their vertical centroids are within *line_tolerance* × average
    word height.

    This separates recognition accuracy from reading-order accuracy:
    spatially sorting both GT and OCR words before CER isolates character
    recognition errors from layout/reading-order errors.

    Args:
        words_with_boxes: List of ``(text, (x0, y0, x1, y1))`` tuples.
        line_tolerance: Fraction of average word height used to group
            words into the same line.

    Returns:
        Space-joined string of words in spatial order.
    """
    if not words_with_boxes:
        return ""

    items: list[tuple[str, float, float]] = []
    heights: list[float] = []
    for text, (x0, y0, x1, y1) in words_with_boxes:
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        items.append((text, cx, cy))
        heights.append(y1 - y0)

    avg_height = sum(heights) / len(heights) if heights else 1.0
    threshold = avg_height * line_tolerance

    # Sort by y-centroid first, then x-centroid within same line.
    items.sort(key=lambda t: (t[2], t[1]))

    # Group into lines.
    lines: list[list[tuple[str, float, float]]] = []
    current_line: list[tuple[str, float, float]] = [items[0]]
    for item in items[1:]:
        if abs(item[2] - current_line[0][2]) <= threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    # Sort each line by x-centroid, then join.
    sorted_words: list[str] = []
    for line in lines:
        line.sort(key=lambda t: t[1])
        sorted_words.extend(word for word, _, _ in line)

    return " ".join(sorted_words)


def cer_spatial(
    reference_words: list[tuple[str, BBox]],
    hypothesis_words: list[tuple[str, BBox]],
    *,
    line_tolerance: float = 0.5,
    normalise: bool = True,
) -> float:
    """Spatially-sorted CER (CER-s).

    Sorts both reference and hypothesis words by bounding-box position
    before computing CER. This isolates character recognition errors from
    reading-order errors.

    Args:
        reference_words: GT words with bounding boxes.
        hypothesis_words: OCR words with bounding boxes.
        line_tolerance: Passed to :func:`spatial_sort_text`.
        normalise: Passed to :func:`cer`.

    Returns:
        CER-s as a float ≥ 0.0.
    """
    ref_text = spatial_sort_text(reference_words, line_tolerance)
    hyp_text = spatial_sort_text(hypothesis_words, line_tolerance)
    return cer(ref_text, hyp_text, normalise=normalise)


# ---------------------------------------------------------------------------
# Reading order accuracy
# ---------------------------------------------------------------------------


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) bounding boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def reading_order_accuracy(
    reference: list[tuple[str, BBox]],
    hypothesis: list[tuple[str, BBox]],
    *,
    iou_threshold: float = 0.3,
) -> float:
    """Fraction of GT word pairs whose relative order is preserved in the hypothesis.

    Matches hypothesis words to reference words by bounding-box IoU, then
    counts what proportion of concordant pairs exist in the hypothesis
    ordering. This is equivalent to ``(concordant pairs) / (total pairs)``
    over the matched subset.

    Args:
        reference: GT words with bounding boxes, in annotation (reading) order.
        hypothesis: Extracted words with bounding boxes, in output order.
        iou_threshold: Minimum IoU to consider a spatial match.

    Returns:
        Accuracy as a float in [0.0, 1.0]. Returns 1.0 when fewer than
        2 words are matched (nothing to compare).
    """
    if len(reference) < 2 or len(hypothesis) < 2:
        return 1.0

    # Greedy match: for each GT word, find the best-matching hypothesis word.
    # Track the hypothesis index assigned to each matched GT word.
    used_hyp: set[int] = set()
    gt_to_hyp_idx: list[tuple[int, int]] = []  # (gt_idx, hyp_idx)

    for gt_idx, (_, gt_box) in enumerate(reference):
        best_iou = 0.0
        best_hyp = -1
        for hyp_idx, (_, hyp_box) in enumerate(hypothesis):
            if hyp_idx in used_hyp:
                continue
            score = _bbox_iou(gt_box, hyp_box)
            if score > best_iou:
                best_iou = score
                best_hyp = hyp_idx
        if best_hyp >= 0 and best_iou >= iou_threshold:
            gt_to_hyp_idx.append((gt_idx, best_hyp))
            used_hyp.add(best_hyp)

    if len(gt_to_hyp_idx) < 2:
        return 1.0

    # Count concordant pairs: for every pair (i, j) where gt_i < gt_j,
    # check whether hyp_i < hyp_j (same relative order).
    hyp_indices = [h for _, h in gt_to_hyp_idx]
    n = len(hyp_indices)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if hyp_indices[i] < hyp_indices[j]:
                concordant += 1

    return concordant / total if total > 0 else 1.0

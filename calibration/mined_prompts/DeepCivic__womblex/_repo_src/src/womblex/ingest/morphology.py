"""Page-image morphology helpers used during document-type detection.

Self-contained signals computed from rendered page pixmaps — handwriting
detection (ruled lines, glyph regularity, stroke-width variance), and OCR
confidence sampling. Imported by `detect.py` and `page_profile.py`.

Kept separate from `detect.py` to keep that file under the 750-line cap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from womblex.ingest.pdf.types import Page


def _page_to_grayscale(page: Page, dpi: int = 72) -> tuple:
    """Convert a PDF page to grayscale numpy array.

    Returns (gray_image, width, height) tuple.
    """
    import cv2

    img = page.render(dpi=dpi)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    return gray, width, height


def _has_ruled_lines(page: Page, dpi: int = 72) -> bool:
    """Detect ruled/lined paper patterns (notebook paper).

    Looks for evenly-spaced horizontal lines spanning most of page width.
    Requires multiple lines with consistent spacing to distinguish from
    email separators or table borders.
    """
    import cv2
    import numpy as np

    gray, width, _height = _page_to_grayscale(page, dpi)

    # Threshold to binary (invert so lines are white)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect long horizontal lines (ruled paper lines span most of page width)
    min_line_width = int(width * 0.4)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width, 1))
    ruled_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Find rows with significant horizontal line content
    row_sums = np.sum(ruled_lines > 0, axis=1)
    threshold = min_line_width * 0.5
    line_rows = np.where(row_sums > threshold)[0]

    if len(line_rows) < 10:
        return False

    # Check for even spacing (notebook paper has consistent line spacing)
    spacings = np.diff(line_rows)
    # Filter out tiny gaps (noise) - real line spacing is 20+ pixels at 72dpi
    spacings = spacings[spacings > 15]
    if len(spacings) < 5:
        return False
    # Check if spacing is consistent (std dev < 30% of mean)
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    return bool(mean_spacing > 0 and (std_spacing / mean_spacing) < 0.3)


def _analyze_glyph_regularity(page: Page, dpi: int = 150) -> float | None:
    """Analyze bounding box regularity of text glyphs using connected components.

    Typed text has uniform glyph heights and consistent horizontal spacing.
    Handwritten text has high variance in both dimensions.

    Returns regularity score 0-1 (high = typed, low = handwritten) or None if
    insufficient glyphs detected.
    """
    import cv2
    import numpy as np

    gray, width, height = _page_to_grayscale(page, dpi)

    # Adaptive threshold to handle varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Find connected components (individual glyphs/characters)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)

    # Filter components by size to get likely text glyphs
    # Exclude background (label 0) and very small/large components
    min_area = 20  # Minimum pixels for a glyph
    max_area = (width * height) * 0.01  # Max 1% of page
    min_height = 5
    max_height = height * 0.1  # Max 10% of page height

    glyph_heights = []

    for i in range(1, num_labels):  # Skip background
        _x, _y, w, h, area = stats[i]
        if min_area <= area <= max_area and min_height <= h <= max_height:
            # Aspect ratio filter: glyphs are roughly square-ish to tall
            aspect = w / h if h > 0 else 0
            if 0.1 <= aspect <= 3.0:
                glyph_heights.append(h)

    if len(glyph_heights) < 50:
        # Not enough glyphs to make a determination
        return None

    heights = np.array(glyph_heights)

    # Use mode-based analysis: typed text clusters tightly around dominant font size
    from collections import Counter
    counter = Counter(heights)
    mode_height = counter.most_common(1)[0][0]

    # Count glyphs within 25% of mode height
    tolerance = max(mode_height * 0.25, 3)  # At least 3 pixels tolerance
    near_mode = np.sum(np.abs(heights - mode_height) <= tolerance)
    mode_ratio = near_mode / len(heights)

    # Map mode_ratio [0.3, 0.7] to regularity [0.0, 1.0]
    # Typed text: 60-80% near mode; Handwritten: 30-50% near mode
    regularity = max(0.0, min(1.0, (mode_ratio - 0.3) / 0.4))

    return float(regularity)


def _analyze_stroke_width_variance(page: Page, dpi: int = 150) -> float | None:
    """Analyze stroke width consistency using morphological operations.

    Typed text has consistent stroke widths (from uniform font rendering).
    Handwritten text has variable stroke widths (pen pressure, angle changes).

    Returns consistency score 0-1 (high = typed, low = handwritten) or None if
    insufficient strokes detected.
    """
    import cv2
    import numpy as np

    gray, _width, _height = _page_to_grayscale(page, dpi)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Skeletonize to get stroke centerlines via iterative thinning
    skeleton = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(10):
        eroded = cv2.erode(skeleton, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(skeleton, temp)
        skeleton = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # Distance transform — distance at skeleton points ≈ stroke width
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    skeleton_points = skeleton > 0
    stroke_widths = dist_transform[skeleton_points]

    # Filter out noise (very small values)
    stroke_widths = stroke_widths[stroke_widths > 1.0]

    if len(stroke_widths) < 100:
        return None

    # Coefficient of variation: typed 0.1-0.25, handwritten 0.3-0.6+
    mean_width = np.mean(stroke_widths)
    std_width = np.std(stroke_widths)
    cv = std_width / mean_width if mean_width > 0 else 1.0

    # Map CV range [0.1, 0.5] to consistency [1.0, 0.0]
    consistency = max(0.0, min(1.0, 1.0 - (cv - 0.1) / 0.4))

    return consistency


def _has_handwriting_signals(page: Page, dpi: int = 150) -> bool:
    """Detect handwriting indicators on a scanned page.

    Combines ruled-line detection, glyph regularity, and stroke-width
    variance. Returns True if handwriting is likely present.
    """
    if _has_ruled_lines(page, dpi=72):
        return True

    glyph_regularity = _analyze_glyph_regularity(page, dpi)
    stroke_consistency = _analyze_stroke_width_variance(page, dpi)

    if glyph_regularity is not None and stroke_consistency is not None:
        # Both scores low → handwriting
        if glyph_regularity < 0.4 and stroke_consistency < 0.4:
            return True
        # One very low → strong signal
        if glyph_regularity < 0.25 or stroke_consistency < 0.25:
            return True
    # Only one score available — a single low reading has to carry the call, so
    # the threshold sits between the "both low" and "one very low" gates above.
    elif (
        (glyph_regularity is not None and glyph_regularity < 0.3)
        or (stroke_consistency is not None and stroke_consistency < 0.3)
    ):
        return True

    return False


def _sample_ocr_confidence(
    page: Page, dpi: int = 150
) -> tuple[float | None, list[float] | None]:
    """Sample OCR confidence on a page using PaddleOCR.

    Returns (avg_confidence_0_100, region_confidences_0_1).
    Typed text typically scores avg >= 85, handwriting scores 40-70.
    """
    try:
        from womblex.ingest.paddle_ocr import get_paddle_reader

        img = page.render(dpi=dpi)

        reader = get_paddle_reader("eng")
        results = reader.readtext(img)

        region_confidences = [
            float(conf) for _bbox, text, conf in results if text.strip()
        ]

        if not region_confidences:
            return None, None

        avg_100 = (sum(region_confidences) / len(region_confidences)) * 100
        return avg_100, region_confidences
    except Exception:
        return None, None

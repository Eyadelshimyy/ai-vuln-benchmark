"""CV2-based heuristics for document analysis.

Spatial, structural, and morphological operations on image geometry.
All functions take numpy arrays (grayscale or RGB) and return analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SkewAnalysis:
    """Result of skew angle detection."""
    angle: float  # Degrees, positive = clockwise
    confidence: float  # 0-1, based on number of detected lines


@dataclass
class TableGridAnalysis:
    """Result of table/grid detection."""
    has_grid: bool
    horizontal_lines: int
    vertical_lines: int
    cell_count: int  # Approximate intersections


@dataclass
class ContourComplexity:
    """Result of contour complexity analysis."""
    regularity: float  # 0-1, high = regular shapes (typed), low = irregular (handwritten)
    avg_vertices: float  # Average vertices per contour after approximation


def detect_skew_angle(gray: np.ndarray, min_line_length: int = 100) -> SkewAnalysis:
    """Detect document skew angle using Hough line transform.

    Args:
        gray: Grayscale image
        min_line_length: Minimum line length to consider

    Returns:
        SkewAnalysis with angle and confidence
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    if lines is None or len(lines) < 5:
        return SkewAnalysis(angle=0.0, confidence=0.0)

    # Calculate angles of all detected lines. OpenCV 4 returns segments as
    # (N, 1, 4); OpenCV 5 flattens to (N, 4) — reshape covers both.
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        if x2 - x1 != 0:  # Avoid division by zero
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Only consider near-horizontal lines (within 45 degrees)
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return SkewAnalysis(angle=0.0, confidence=0.0)

    # Use median to be robust against outliers
    median_angle = np.median(angles)

    # Confidence based on consistency of angles
    angle_std = np.std(angles)
    confidence = max(0.0, min(1.0, 1.0 - angle_std / 10))

    return SkewAnalysis(angle=median_angle, confidence=confidence)


def detect_table_grid(gray: np.ndarray, min_line_ratio: float = 0.3) -> TableGridAnalysis:
    """Detect table/grid structures using morphological line extraction.

    Args:
        gray: Grayscale image
        min_line_ratio: Minimum line length as ratio of image dimension

    Returns:
        TableGridAnalysis with grid detection results
    """
    height, width = gray.shape[:2]
    min_h_length = int(width * min_line_ratio)
    min_v_length = int(height * min_line_ratio)

    # Threshold and invert
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_length, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_length))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Count lines by finding contours
    h_contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_count = len(h_contours)
    v_count = len(v_contours)

    # Find intersections (approximate cell count)
    intersections = cv2.bitwise_and(h_lines, v_lines)
    intersection_contours, _ = cv2.findContours(
        intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # A grid needs at least 2 horizontal and 2 vertical lines
    has_grid = h_count >= 2 and v_count >= 2

    return TableGridAnalysis(
        has_grid=has_grid,
        horizontal_lines=h_count,
        vertical_lines=v_count,
        cell_count=len(intersection_contours)
    )


def analyze_contour_complexity(gray: np.ndarray, epsilon_ratio: float = 0.02) -> ContourComplexity:
    """Analyze contour complexity to distinguish typed vs handwritten.

    Args:
        gray: Grayscale image
        epsilon_ratio: Approximation accuracy as ratio of contour perimeter

    Returns:
        ContourComplexity with regularity score
    """
    # Threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 10:
        return ContourComplexity(regularity=0.5, avg_vertices=0.0)

    # Filter by size (likely text glyphs)
    height, width = gray.shape[:2]
    min_area = 20
    max_area = (width * height) * 0.01

    vertex_counts = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Approximate contour
            perimeter = cv2.arcLength(contour, True)
            epsilon = epsilon_ratio * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertex_counts.append(len(approx))

    if len(vertex_counts) < 10:
        return ContourComplexity(regularity=0.5, avg_vertices=0.0)

    avg_vertices = np.mean(vertex_counts)
    std_vertices = np.std(vertex_counts)

    # Typed text: consistent vertex counts (low CV)
    # Handwritten: variable vertex counts (high CV)
    cv = std_vertices / avg_vertices if avg_vertices > 0 else 1.0

    # Map CV to regularity score
    regularity = max(0.0, min(1.0, 1.0 - (cv - 0.2) / 0.6))

    return ContourComplexity(regularity=float(regularity), avg_vertices=float(avg_vertices))


def calculate_blur_score(gray: np.ndarray) -> float | None:
    """Calculate blur score using Laplacian variance.

    Higher variance = sharper image, lower variance = blurrier.

    Args:
        gray: Grayscale image (8-bit, single-channel)

    Returns:
        Blur score (float) where higher = sharper, or None if invalid input
    """
    if gray is None or gray.size == 0:
        return None

    if gray.ndim != 2:
        return None

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    return variance


def segment_text_photo_regions(
    gray: np.ndarray,
    block_size: int = 16
) -> np.ndarray | None:
    """Segment image into text vs photo regions using local edge density.

    Args:
        gray: Grayscale image
        block_size: Size of analysis blocks (pixels)

    Returns:
        Binary mask (bool array) where True = text region, False = photo region,
        or None if invalid input
    """
    if gray is None or gray.size == 0:
        return None

    height, width = gray.shape[:2]

    if height < block_size or width < block_size:
        return None

    # Calculate Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Calculate local edge density per block
    blocks_y = height // block_size
    blocks_x = width // block_size

    edge_density = np.zeros((blocks_y, blocks_x), dtype=float)
    entropy_map = np.zeros((blocks_y, blocks_x), dtype=float)

    for i in range(blocks_y):
        for j in range(blocks_x):
            y1, y2 = i * block_size, (i + 1) * block_size
            x1, x2 = j * block_size, (j + 1) * block_size

            block_mag = magnitude[y1:y2, x1:x2]
            block_gray = gray[y1:y2, x1:x2]

            # Edge density: mean gradient magnitude
            edge_density[i, j] = np.mean(block_mag)

            # Local entropy via histogram
            hist, _ = np.histogram(block_gray.flatten(), bins=16, range=(0, 256))
            hist = hist.astype(float) / hist.sum() if hist.sum() > 0 else hist
            # Entropy calculation
            nonzero = hist[hist > 0]
            entropy_map[i, j] = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) > 0 else 0

    # Text regions: high edge density AND moderate entropy
    # Photo regions: lower edge density OR very high entropy (smooth gradients)

    edge_threshold = np.median(edge_density) * 1.2
    entropy_threshold = np.median(entropy_map) * 0.8

    # Text = high edges, not too smooth
    text_mask_blocks = (edge_density > edge_threshold) & (entropy_map > entropy_threshold)

    # Upscale to original resolution
    text_mask = np.zeros((height, width), dtype=bool)
    for i in range(blocks_y):
        for j in range(blocks_x):
            y1, y2 = i * block_size, (i + 1) * block_size
            x1, x2 = j * block_size, (j + 1) * block_size
            text_mask[y1:y2, x1:x2] = text_mask_blocks[i, j]

    return text_mask

"""NumPy-based heuristics for document analysis.

Statistical operations on pixel arrays for preprocessing decisions.
All functions take numpy arrays and return analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HistogramAnalysis:
    """Result of intensity histogram analysis."""
    noise_floor: float  # 0-1, amount of low-level noise
    is_scanned: bool  # True if histogram suggests scanned document
    peak_count: int  # Number of distinct intensity peaks
    dynamic_range: float  # 0-1, spread of intensities used


@dataclass
class OtsuAnalysis:
    """Result of OTSU threshold analysis."""
    threshold: int  # Optimal threshold value (0-255)
    variance_ratio: float  # Between-class / within-class variance
    is_bimodal: bool  # True if clear text/background separation


def analyze_histogram(gray: np.ndarray, num_bins: int = 256) -> HistogramAnalysis:
    """Analyze intensity histogram for scan vs native detection.

    Scanned documents have characteristic histogram shapes:
    - More noise in shadow regions
    - Less sharp peaks than native PDFs
    - Wider distribution of background values

    Args:
        gray: Grayscale image
        num_bins: Number of histogram bins

    Returns:
        HistogramAnalysis with detection results
    """
    hist, _bin_edges = np.histogram(gray.flatten(), bins=num_bins, range=(0, 256))
    hist = hist.astype(float) / hist.sum()  # Normalize

    # Noise floor: amount of signal in very dark region (0-30)
    noise_floor = float(np.sum(hist[:30]))

    # Dynamic range: spread of intensities used
    nonzero_bins = np.where(hist > 0.001)[0]
    if len(nonzero_bins) > 0:
        dynamic_range = (nonzero_bins[-1] - nonzero_bins[0]) / 255
    else:
        dynamic_range = 0.0

    # Peak detection: find local maxima
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01:
            peaks.append(i)

    # Scanned documents typically have:
    # - Higher noise floor (scanner noise)
    # - Wider dynamic range (paper texture)
    # - Fewer sharp peaks (blurring)
    is_scanned = noise_floor > 0.02 or (dynamic_range > 0.7 and len(peaks) < 3)

    return HistogramAnalysis(
        noise_floor=noise_floor,
        is_scanned=is_scanned,
        peak_count=len(peaks),
        dynamic_range=dynamic_range
    )


def analyze_otsu_threshold(gray: np.ndarray) -> OtsuAnalysis:
    """Analyze OTSU threshold characteristics.

    OTSU finds optimal threshold for bimodal distributions.
    The threshold value and variance ratio indicate document type.

    Args:
        gray: Grayscale image

    Returns:
        OtsuAnalysis with threshold characteristics
    """
    # Calculate histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()

    if total == 0:
        return OtsuAnalysis(threshold=128, variance_ratio=0.0, is_bimodal=False)

    # OTSU's method
    sum_total = np.sum(np.arange(256) * hist)
    sum_bg = 0.0
    weight_bg = 0.0
    max_variance = 0.0
    best_threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    # Calculate within-class variance for ratio
    below = gray[gray <= best_threshold]
    above = gray[gray > best_threshold]

    var_below = np.var(below) if len(below) > 0 else 0
    var_above = np.var(above) if len(above) > 0 else 0

    weight_below = len(below) / total
    weight_above = len(above) / total

    within_variance = weight_below * var_below + weight_above * var_above

    if within_variance > 0:
        variance_ratio = max_variance / within_variance
    else:
        variance_ratio = 0.0

    # Bimodal if high variance ratio (clear separation)
    is_bimodal = variance_ratio > 1.0

    return OtsuAnalysis(
        threshold=best_threshold,
        variance_ratio=variance_ratio,
        is_bimodal=is_bimodal
    )

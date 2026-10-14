"""Redaction utility helpers.

Low-level helpers for redaction pre-processing. Not called by extraction
strategies directly — use from the redaction operation only.
"""

from __future__ import annotations

import numpy as np

from womblex.redact.detector import RedactionDetector, RedactionInfo


def pre_ocr_mask(
    image: np.ndarray,
    page: int,
    detector: RedactionDetector,
) -> tuple[np.ndarray, list[RedactionInfo]]:
    """Detect and mask redactions on a page image before OCR.

    Returns the masked image and the list of detected redactions.
    Not called by extraction strategies — available for tooling that
    pre-processes images before passing them to an OCR engine.
    """
    redactions = detector.detect(image, page=page)
    if redactions:
        image = detector.mask(image, redactions)
    return image, redactions

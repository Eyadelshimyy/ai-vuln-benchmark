"""Backend protocols for pluggable ingest components.

Each protocol defines the minimal interface a backend must satisfy.
Concrete implementations (PaddleOCRReader, PPDocLayoutAnalyzer,
preprocess_for_ocr) already conform — these protocols formalise the
contracts so that alternative backends (document-trained layout models,
dedicated HTR recognisers) can be injected without changing strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# OCR reader
# ---------------------------------------------------------------------------


@dataclass
class OCRRegionResult:
    """A single text region from OCR, with four-corner bbox."""

    bbox: list[list[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float  # 0-1


@dataclass
class OCRPageResult:
    """Page-level OCR output.

    Region-based engines (PaddleOCR, Tesseract, EasyOCR, cloud APIs that
    return word/line boxes) populate ``regions``. LLM/VLM-based engines
    (Mistral OCR via Bedrock, Ollama, Gemini) typically return
    prose/markdown with reading order already resolved — they populate
    ``markdown`` and set ``reading_order_native=True``.

    Strategies normalise both shapes into the existing TextBlock pipeline.
    """

    regions: list[OCRRegionResult] = field(default_factory=list)
    markdown: str | None = None
    reading_order_native: bool = False
    confidence: float = 1.0  # page-level confidence, 0-1


@runtime_checkable
class OCRReader(Protocol):
    """Protocol for OCR backends.

    Backends must implement ``read_page``. ``readtext`` is optional —
    region-based engines provide it for callers that want raw EasyOCR-style
    tuples (e.g. sub-page OCR in the orchestrator's `_apply_native_page`).
    LLM-based engines that return only markdown can omit ``readtext`` entirely.
    """

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """OCR an entire page image.

        Returns either region-based output (``regions`` populated) or
        markdown-based output (``markdown`` populated with
        ``reading_order_native=True``), depending on the engine.
        """
        ...


# ---------------------------------------------------------------------------
# Layout analyser
# ---------------------------------------------------------------------------


@dataclass
class LayoutRegionResult:
    """A layout region with bounding box, label, and block type."""

    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) pixel coords
    label: str  # raw model class name
    block_type: str  # mapped womblex block_type
    confidence: float


#: The ``block_type`` vocabulary a layout model must emit: the element kinds
#: a page region can become. Anything else is a conformance failure.
LAYOUT_BLOCK_TYPES: frozenset[str] = frozenset({
    "paragraph", "heading", "list_item", "caption", "header", "footer",
    "footnote", "signature", "figure", "table",
})


def check_layout_regions(regions: list[LayoutRegionResult]) -> None:
    """Raise ``ValueError`` unless every region speaks the womblex vocabulary
    with a well-formed box, a confidence in [0, 1] and top-to-bottom order."""
    for r in regions:
        x0, y0, x1, y1 = r.bbox
        if r.block_type not in LAYOUT_BLOCK_TYPES:
            raise ValueError(
                f"layout block_type {r.block_type!r} (label {r.label!r}) is not "
                f"one of {sorted(LAYOUT_BLOCK_TYPES)}"
            )
        if not (x0 < x1 and y0 < y1) or not 0.0 <= r.confidence <= 1.0:
            raise ValueError(f"malformed layout region: {r}")
    if [r.bbox[1] for r in regions] != sorted(r.bbox[1] for r in regions):
        raise ValueError("layout regions are not sorted top-to-bottom")


@runtime_checkable
class LayoutAnalyzer(Protocol):
    """Protocol for layout analysis backends.

    Any class with an ``analyze`` method returning LayoutRegionResult-compatible
    objects satisfies this protocol; ``check_layout_regions`` is the
    conformance check. PPDocLayoutAnalyzer is the default.
    """

    def analyze(
        self, img: np.ndarray, conf_threshold: float = 0.3
    ) -> list[LayoutRegionResult]:
        """Detect layout regions in *img*.

        Returns regions sorted top-to-bottom by y-coordinate.
        """
        ...


# ---------------------------------------------------------------------------
# Image preprocessor
# ---------------------------------------------------------------------------


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for image preprocessing backends.

    Any callable with signature ``(img) -> (grayscale, steps)`` satisfies
    this protocol.  ``preprocess_for_ocr`` in ``paddle_ocr.py`` is the
    default implementation.
    """

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Preprocess *img* for OCR.

        Returns ``(grayscale_image, steps_applied)`` where *steps_applied*
        is e.g. ``["deskew", "otsu_binarise"]``.
        """
        ...

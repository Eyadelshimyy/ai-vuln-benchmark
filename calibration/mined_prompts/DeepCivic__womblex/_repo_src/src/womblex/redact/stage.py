"""Redaction operation.

Runs as a separate pass after extraction. Renders PDF pages as images,
detects black-box redaction regions, and applies the configured mode
to the affected page text.

Modes:
- ``flag``:     Set ``has_redaction=True`` on affected chunks (no text change).
- ``blackout``: Replace affected page text with ``<REDACTED>`` markers.
- ``delete``:   Clear affected page text entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from womblex.config import RedactionConfig
from womblex.redact.detector import RedactionDetector, RedactionInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from womblex.ingest.elements import Element
    from womblex.ingest.extract import ExtractionResult, PageResult
    from womblex.ingest.layout_step import PageLayout
    from womblex.ingest.pdf.types import Page
    from womblex.process.chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class RedactionReport:
    """Summary of redactions detected across a document."""

    page_redactions: dict[int, list[RedactionInfo]] = field(default_factory=dict)
    # Raster pages the layout filter was asked of but had no usable layout for.
    unfiltered_pages: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.page_redactions.values())

    @property
    def affected_pages(self) -> list[int]:
        return sorted(self.page_redactions.keys())


def build_detector(config: RedactionConfig) -> RedactionDetector:
    """Build a RedactionDetector from config."""
    return RedactionDetector(
        threshold=config.threshold,
        min_area_ratio=config.min_area_ratio,
        max_area_ratio=config.max_area_ratio,
    )


_VECTOR_MIN_WIDTH_PT = 3.0   # filters narrow vertical separator lines (manifest column rules)
_VECTOR_MIN_HEIGHT_PT = 8.0  # filters glyph-rendering small filled rects (body glyphs ≤ 7pt tall)

# Layout regions whose ``block_type`` indicates image / tabular content
# where raster contour redaction false-positives originate (02737-class
# scanned_mixed CRM forms with dark form-field backgrounds and embedded
# chart regions).
_LAYOUT_EXCLUSION_BLOCK_TYPES = frozenset({"figure", "table"})


def detect_redactions(
    path: Path,
    page_count: int,
    detector: RedactionDetector,
    dpi: int = 150,
    use_layout_filter: bool = True,
    layout: Mapping[int, PageLayout] | None = None,
) -> RedactionReport:
    """Detect redacted regions per page; prefer vector ops, fall back to raster.

    For each page:

    - First check ``page.drawings()`` for filled near-black rectangles
      (matches native-PDF vector-drawn redactions; no area threshold).
    - If none found, rasterise the page at *dpi* and run the CV2 contour
      detector (handles raster overlays and scanned pages). When
      *use_layout_filter* is true, pass the page's figure/table layout
      regions (from *layout*, the batch's layout step) as exclusion zones
      to the contour detector — suppresses raster false positives on dark
      form-field backgrounds and embedded chart regions (02737-class
      scanned_mixed CRM forms). The filter is best-effort: a page with no
      usable layout (``error`` status, or none recorded) is detected with no
      exclusion, listed on ``report.unfiltered_pages`` and warned once per document.

    Bboxes are returned in pixel coordinates at *dpi* regardless of which
    path produced them, so consumers see a single coord system.

    Args:
        path: Path to the PDF file.
        page_count: Number of pages to scan (from extraction metadata).
        detector: Configured RedactionDetector instance.
        dpi: Resolution for page rendering / coord scaling.
        use_layout_filter: Drop contour hits inside figure / table layout
            regions on raster-fallback pages. Best-effort; falls back to the
            raw raster pass where a page has no layout.
        layout: Per-page layout regions by 0-based page number; where absent,
            the filter has nothing to read.

    Returns:
        RedactionReport with per-page detection results.
    """
    from womblex.ingest.pdf import open_document

    report = RedactionReport()
    try:
        with open_document(path) as doc:
            pages_to_scan = min(page_count, len(doc))
            scale = dpi / 72.0  # PDF coord (72 DPI) → pixel coord at *dpi*
            for page_num in range(pages_to_scan):
                page = doc[page_num]

                vector_redactions = _detect_vector_redactions(page, page_num, scale)
                if vector_redactions:
                    report.page_redactions[page_num] = vector_redactions
                    continue

                img = page.render(dpi=dpi)
                exclude_rects = (
                    _layout_exclude_rects(img, (layout or {}).get(page_num))
                    if use_layout_filter else None
                )
                if use_layout_filter and exclude_rects is None:
                    report.unfiltered_pages.append(page_num)
                raster_redactions = detector.detect(
                    img, page=page_num, exclude_rects=exclude_rects,
                )
                if raster_redactions:
                    report.page_redactions[page_num] = raster_redactions
    except Exception as e:
        logger.warning("Redaction detection failed for %s: %s", path, e)

    if report.unfiltered_pages:
        logger.warning(
            "layout filter unavailable, redaction ran without exclusion zones: doc=%s "
            "pages=%s (no usable layout; `run-stage layout` with redaction enabled "
            "records it)", path.name, report.unfiltered_pages,
        )
    return report


def _layout_exclude_rects(
    img: np.ndarray,
    page_layout: PageLayout | None,
) -> list[tuple[int, int, int, int]] | None:
    """Figure/table bboxes in *img* pixels from a page's persisted layout.

    ``None`` means the page has no usable layout (none recorded, or the layout
    step failed on it): detection runs without exclusion zones and the page is
    recorded on the report. An ``empty`` page is ``[]``, a genuine "nothing to
    exclude".
    """
    if page_layout is None or page_layout.status == "error":
        return None

    height, width = img.shape[:2]
    rects: list[tuple[int, int, int, int]] = []
    for region in page_layout.regions:
        if region.block_type not in _LAYOUT_EXCLUSION_BLOCK_TYPES:
            continue
        b = region.bbox
        rects.append((
            int(b.x * width), int(b.y * height),
            int((b.x + b.width) * width), int((b.y + b.height) * height),
        ))
    return rects


def _detect_vector_redactions(page: Page, page_num: int, scale: float) -> list[RedactionInfo]:
    """Enumerate filled near-black rectangles from ``page.drawings()``.

    Bboxes converted from PDF coords (72 DPI) to pixel coords using *scale*
    so all ``RedactionInfo.bbox`` values share one coord system regardless of
    which detection path produced them.
    """
    out: list[RedactionInfo] = []
    for d in page.drawings():
        if not d.filled or not _is_near_black_fill(d.fill):
            continue
        rect = d.rect
        if rect.width < _VECTOR_MIN_WIDTH_PT or rect.height < _VECTOR_MIN_HEIGHT_PT:
            continue
        x1 = int(rect.x0 * scale)
        y1 = int(rect.y0 * scale)
        x2 = int(rect.x1 * scale)
        y2 = int(rect.y1 * scale)
        out.append(RedactionInfo(
            bbox=(x1, y1, x2, y2),
            page=page_num,
            area_px=(x2 - x1) * (y2 - y1),
        ))
    return out


def _is_near_black_fill(fill) -> bool:
    """Treat fill as near-black if max channel ≤ 0.1 (CMYK: K ≥ 0.9 + others ≤ 0.1)."""
    if fill is None:
        return False
    if isinstance(fill, (int, float)):
        return fill <= 0.1
    if len(fill) == 1:
        return bool(fill[0] <= 0.1)
    if len(fill) == 3:
        return bool(max(fill) <= 0.1)
    if len(fill) == 4:
        return bool(fill[3] >= 0.9 and max(fill[:3]) <= 0.1)
    return False


def apply_text_redaction(
    pages: list[PageResult],
    report: RedactionReport,
    mode: str,
) -> list[PageResult]:
    """Modify page text based on the redaction mode.

    ``flag`` makes no text changes — use ``annotate_chunks`` instead.
    ``blackout`` prepends ``<REDACTED>`` to affected page text.
    ``delete`` clears affected page text entirely.

    Args:
        pages: Per-page extraction results (mutated in-place).
        report: Detected redaction regions.
        mode: One of ``flag``, ``blackout``, ``delete``.

    Returns:
        The (mutated) pages list.
    """
    if mode == "flag" or not report.total:
        return pages

    affected = set(report.affected_pages)
    for page in pages:
        if page.page_number not in affected:
            continue
        if mode == "blackout":
            page.text = f"<REDACTED>\n{page.text}" if page.text else "<REDACTED>"
        elif mode == "delete":
            page.text = ""

    return pages


def annotate_chunks(
    chunks: list[TextChunk],
    report: RedactionReport,
) -> list[TextChunk]:
    """Mark chunks whose source pages contain redacted regions.

    Sets ``chunk.has_redaction = True`` for any chunk overlapping an
    affected page. Does not modify chunk text.
    """
    if not report.total:
        return chunks

    affected = set(report.affected_pages)
    for chunk in chunks:
        if hasattr(chunk, "source_pages") and chunk.source_pages:
            if any(p in affected for p in chunk.source_pages):
                chunk.has_redaction = True
        elif hasattr(chunk, "page_number") and chunk.page_number in affected:
            chunk.has_redaction = True

    return chunks


def annotate_elements(
    elements: list[Element],
    report: RedactionReport,
) -> list[Element]:
    """Set ``meta['has_redaction']='true'`` on elements whose page is in *report*.

    Page-level propagation: every element on an affected page is flagged.
    Avoids the pixel-coord (report bboxes at detection DPI) vs PDF-coord
    (element bboxes at 72 DPI) conversion that bbox-level overlap would
    require. Mutates elements in place; returns the same list.
    """
    if not report.total:
        return elements

    affected = report.page_redactions
    for element in elements:
        if element.page is not None and element.page in affected:
            element.meta["has_redaction"] = "true"
    return elements


def annotate_extraction(
    extraction: ExtractionResult,
    report: RedactionReport,
) -> ExtractionResult:
    """Annotate an ExtractionResult with redaction metadata.

    Adds per-page warning strings so downstream consumers know which
    pages had redacted content detected.
    """
    if not report.total:
        return extraction

    for page_num, redactions in report.page_redactions.items():
        extraction.warnings.append(
            f"page {page_num}: {len(redactions)} redacted region(s) detected"
        )

    return extraction

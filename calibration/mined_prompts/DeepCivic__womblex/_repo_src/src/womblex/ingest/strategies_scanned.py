"""OCR primitives for scanned pages, consumed by the orchestrator.

Covers SCANNED_MACHINEWRITTEN, SCANNED_HANDWRITTEN, SCANNED_MIXED,
HYBRID, and IMAGE document types — all requiring OCR (PaddleOCR by
default, or an LLM/VLM engine such as Mistral OCR via AWS Bedrock).

These are page-level primitives, not document-level strategies: every
one of the types above is dispatched per page by
``orchestrator.extract_with_plan``. Images are no exception — the PDF seam
opens one as a single-page document, so it reaches the same
``_apply_ocr_page`` path a scanned PDF page does.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from womblex.ingest.elements import TEXT_KINDS
from womblex.ingest.extract import (
    TableData,
    TextBlock,
    _normalise_bbox,
    _normalise_rect,
    _ocr_text_block,
)
from womblex.ingest.interfaces.protocols import OCRRegionResult
from womblex.ingest.ocr_tables import reconstruct_table, regions_in_rect, span_from_region
from womblex.ingest.paddle_ocr import (
    get_ocr_reader,
    is_llm_engine,
    preprocess_for_ocr,
)
from womblex.ingest.table_grid import Span, cluster_x_centroids, rows_from_spans

if TYPE_CHECKING:
    from womblex.ingest.layout_step import PageLayout
    from womblex.ingest.pdf.types import Page, Rect, Word

logger = logging.getLogger(__name__)

# A full-page OCR block whose layout kind is a non-text region (figure,
# caption, …) is reclassified to a text paragraph once its OCR yields at
# least this many words.  Full-page document scans produce a paragraph+ of
# coherent text and must reach chunking (the non-text kinds are excluded
# from TEXT_KINDS); genuine figures yield only incidental words.  See K9-fig.
_OCR_TEXT_KIND_MIN_WORDS = 5


# ---------------------------------------------------------------------------
# Shared OCR helpers
# ---------------------------------------------------------------------------


def _spatial_sort_regions(regions, line_tolerance: float = 0.5) -> str:
    """Order OCR regions row-major and join into newline-delimited text.

    Within a row (regions whose y-centroids cluster within ``line_tolerance``
    × average height) regions are joined left-to-right with spaces.  Rows are
    joined with newlines.  Fixes the column-mixing failure mode where paddle
    returns table cells in detection order.
    """
    spans = [span_from_region(r) for r in regions if r.text.strip()]
    rows, _avg_h = rows_from_spans(spans, line_tolerance=line_tolerance)
    return "\n".join(" ".join(s.text for s in row) for row in rows)


def _table_aware_text(
    regions,
    line_tolerance: float = 0.5,
    table_min_cols: int = 3,
    table_min_start_rows: int = 2,
) -> str:
    """Spatial-sort regions, then re-emit detected table blocks column-major.

    Detection uses a two-phase rule:

    - **Start**: ``table_min_start_rows`` consecutive rows each with
      ``table_min_cols`` or more items establish the table and its column
      structure (cluster x-centroids of all items in the start rows).
    - **Continue**: subsequent rows are absorbed into the table while every
      item falls near one of the established column centres. A row with a
      single item, or any item too far from any column centre, ends the run
      — the next iteration starts re-evaluating from there.

    Detected blocks are emitted column-major (each column joined into its
    own paragraph). Everything else uses the row-major reading order from
    ``_spatial_sort_regions``.
    """
    spans = [span_from_region(r) for r in regions if r.text.strip()]
    rows, avg_h = rows_from_spans(spans, line_tolerance=line_tolerance)
    if not rows:
        return ""

    out_blocks: list[str] = []
    i = 0
    while i < len(rows):
        end = _find_table_end(rows, i, table_min_cols, table_min_start_rows, avg_h)
        if end > i + 1:  # at least 2 rows constitute a meaningful table
            out_blocks.append(_emit_columns(rows[i:end], avg_h))
            i = end
        else:
            out_blocks.append(" ".join(s.text for s in rows[i]))
            i += 1
    return "\n".join(out_blocks)


def _find_table_end(
    rows: list[list[Span]], start: int, min_cols: int, min_start_rows: int, avg_h: float,
) -> int:
    """Return the row index just past a detected table run, or ``start`` if none."""
    if start + min_start_rows > len(rows):
        return start
    if not all(len(rows[start + k]) >= min_cols for k in range(min_start_rows)):
        return start

    start_items = [s for k in range(min_start_rows) for s in rows[start + k]]
    col_gap = max(avg_h * 3.0, 30.0)
    col_centers = cluster_x_centroids([s.cx for s in start_items], col_gap)
    if len(col_centers) < min_cols:
        return start  # the start rows didn't cluster into enough distinct columns

    fit_dist = max(avg_h * 5.0, 50.0)
    j = start + min_start_rows
    while j < len(rows):
        row = rows[j]
        # Continuation requires multiple items AND every item near a column.
        if len(row) < 2:
            break
        if not all(min(abs(s.cx - c) for c in col_centers) <= fit_dist for s in row):
            break
        j += 1
    return j


def _emit_columns(table_rows: list[list[Span]], avg_h: float) -> str:
    """Emit a detected table block column-major, with blank lines between columns."""
    all_items = [s for row in table_rows for s in row]
    if not all_items:
        return ""
    col_gap = max(avg_h * 3.0, 30.0)
    col_centers = cluster_x_centroids([s.cx for s in all_items], col_gap)
    if not col_centers:
        return " ".join(s.text for s in all_items)

    columns: list[list[Span]] = [[] for _ in col_centers]
    for s in all_items:
        idx = min(range(len(col_centers)), key=lambda k: abs(s.cx - col_centers[k]))
        columns[idx].append(s)

    parts: list[str] = []
    for col in columns:
        col.sort(key=lambda s: s.cy)
        parts.append(" ".join(s.text for s in col))
    return "\n\n".join(p for p in parts if p)


def _ocr_page(
    page: Page,
    dpi: int,
    lang: str,
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> tuple[str, float, list[str], bool, list, tuple[int, int]]:
    """OCR a page and return text plus per-region detections.

    Returns ``(text, confidence_0_100, steps, reading_order_native,
    regions, (pix_width, pix_height))`` where ``regions`` is the
    per-detection list from the OCR engine (empty for LLM-OCR engines
    that resolve reading order natively and only emit markdown). Region
    bboxes are in image-pixel coords at *dpi*; callers normalise by the
    returned ``pix_width`` / ``pix_height`` to get 0-1 floats.

    Region-based engines (paddleocr) get the standard preprocess pipeline
    (deskew + binarise). LLM/VLM engines (mistral-ocr, ollama) skip
    preprocessing because they ingest the colour render directly and
    resolve reading order themselves.
    """
    import cv2

    from womblex.ingest.heuristics_cv2 import calculate_blur_score

    img = page.render(dpi=dpi)
    pix_dims = (int(img.shape[1]), int(img.shape[0]))

    # Pre-OCR blur check (cheap, useful for any engine)
    pre_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = calculate_blur_score(pre_gray)

    reader = get_ocr_reader(engine=engine, lang=lang, **(engine_options or {}))

    # LLM/VLM engines work better on the unmodified colour render; binarised
    # output throws off models trained on natural document images.
    if is_llm_engine(engine):
        ocr_input = img
        steps = ["llm_native_input"]
    else:
        ocr_input, steps = preprocess_for_ocr(img)

    if blur is not None and blur < 50:
        steps.append("low_blur_warning")
        logger.warning("blurry page: doc=%s page=%d blur_score=%.1f", page.doc_name, page.number, blur)

    page_result = reader.read_page(ocr_input)

    if page_result.reading_order_native and page_result.markdown is not None:
        text = page_result.markdown.strip()
        avg_conf = page_result.confidence * 100.0
    else:
        text = _table_aware_text(page_result.regions)
        avg_conf = page_result.confidence * 100.0

    if avg_conf < 40.0:
        logger.warning("low OCR confidence: doc=%s page=%d confidence=%.1f", page.doc_name, page.number, avg_conf)

    return (
        text, avg_conf, steps, page_result.reading_order_native,
        page_result.regions, pix_dims,
    )


def _word_inside(word: Word, rect: Rect) -> bool:
    """Check if a word's midpoint falls inside a rect."""
    cx = (word[0] + word[2]) / 2
    cy = (word[1] + word[3]) / 2
    return bool(rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1)


def _ocr_region_block_type(
    text: str,
    layout_kind: str = "figure",
    min_words: int = _OCR_TEXT_KIND_MIN_WORDS,
) -> str:
    """Reclassify an OCR'd region by text volume when its layout kind is non-text.

    Full-page scans collapse a whole page's OCR into one block whose kind is the
    dominant layout region's — sometimes ``figure``. A figure block holds a
    paragraph+ of coherent text yet is excluded from chunking (figure ∉
    TEXT_KINDS), silently losing document content. When the OCR yields >=
    ``min_words`` words the block is promoted to ``paragraph``; sparser output
    (page numbers, bare logos) keeps its ``layout_kind``. Text kinds (incl.
    ``caption``) and ``table`` pass through unchanged.
    """
    if layout_kind in TEXT_KINDS or layout_kind == "table":
        return layout_kind
    return "paragraph" if len(text.split()) >= min_words else layout_kind


# max_overlapping_words=2 tolerates incidental native glyphs (e.g. a caption
# label) inside an image rect; anything more means the text layer already
# covers the region.  min_rect_size=50 px filters icons/bullets that won't
# OCR meaningfully at 72 dpi.
def _ocr_image_regions(
    page: Page,
    native_words: list[Word],
    dpi: int,
    lang: str,
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    max_overlapping_words: int = 2,
    min_rect_size: int = 50,
) -> tuple[list[TextBlock], list[str]]:
    """OCR image rects on a native page that have no overlapping native text.

    Sub-page conditional OCR — used by the orchestrator's `_apply_native_page`
    operation when a page has a native text layer but also embeds image content
    (e.g. a redacted form scan inserted into an otherwise-native report). Only
    image rects with
    at most ``max_overlapping_words`` native words inside are OCR'd, so
    pages where the native text layer already covers everything do not
    incur OCR cost.
    """
    blocks: list[TextBlock] = []
    steps: list[str] = []

    pw, ph = page.rect.width, page.rect.height
    if pw <= 0 or ph <= 0:
        return blocks, steps

    image_rects = [image.rect for image in page.images()]
    if not image_rects:
        return blocks, steps

    candidate_rects: list[Rect] = []
    for rect in image_rects:
        if rect.width < min_rect_size or rect.height < min_rect_size:
            continue
        overlap = sum(1 for w in native_words if _word_inside(w, rect))
        if overlap <= max_overlapping_words:
            candidate_rects.append(rect)
    if not candidate_rects:
        return blocks, steps

    reader = get_ocr_reader(engine=engine, lang=lang, **(engine_options or {}))

    for rect in candidate_rects:
        try:
            img = page.render(dpi=dpi, clip=rect)
            page_result = reader.read_page(img)
        except Exception as exc:
            logger.warning(
                "subpage OCR failed: page=%d err=%s", page.number, exc,
            )
            continue

        if page_result.reading_order_native and page_result.markdown is not None:
            text = page_result.markdown.strip()
        else:
            text = "\n".join(r.text for r in page_result.regions if r.text.strip())
        text = text.strip()
        if not text:
            continue

        blocks.append(TextBlock(
            text=text,
            position=_normalise_rect(rect, pw, ph),
            block_type=_ocr_region_block_type(text),
            confidence=page_result.confidence,
        ))
        steps.append("subpage_ocr")

    return blocks, steps


def _markdown_page_block(
    page: Page, text: str, conf: float,
) -> list[TextBlock]:
    """Wrap LLM-derived markdown as a single page-spanning paragraph block.

    LLM engines resolve reading order natively, so per-region layout
    sorting is skipped. The full page text becomes one block at the
    page rect — downstream chunking/normalisation operates as usual.
    """
    if not text.strip():
        return []
    pw, ph = page.rect.width, page.rect.height
    return [TextBlock(
        text=text.strip(),
        position=_normalise_bbox((0, 0, pw, ph), pw, ph),
        block_type="paragraph",
        confidence=conf / 100.0 if conf > 1.0 else conf,
    )]


def _layout_blocks_and_tables(
    page: Page,
    dpi: int,
    text: str,
    conf: float,
    ocr_regions: Sequence[OCRRegionResult] | None = None,
    ocr_pix_dims: tuple[int, int] | None = None,
    page_deskewed: bool = False,
    page_layout: PageLayout | None = None,
) -> tuple[list[TextBlock], list[TableData], list[OCRRegionResult]]:
    """Turn a page's persisted layout regions into typed TextBlocks and tables.

    ``page_layout`` is the batch's layout step output for this page. Without
    one, or when it is ``empty`` / ``error``, the page falls back to a single
    paragraph block of its OCR text.

    ``ocr_regions`` / ``ocr_pix_dims`` carry the per-detection OCR output
    that produced *text*, in image-pixel coords at *dpi* — the raw
    material for reconstructing cells inside a detected table rect. They
    are supplied together or not at all: regions without their render
    dimensions cannot be mapped from the layout's normalised boxes, so they
    are dropped. Only region-based engines (paddleocr) supply them; LLM/VLM
    engines resolve reading order natively, return no regions, and are
    dispatched to ``_markdown_page_block`` instead, so they never reach
    this function.

    ``page_deskewed`` reports whether ``preprocess_for_ocr`` rotated the
    image before OCR: the region coords are then in deskewed space while
    this pass renders the raw page, so the two are not comparable and
    round 1 refuses reconstruction on such pages (A2).

    Returns ``(blocks, tables, consumed_regions)`` — the third element is
    the OCR regions absorbed by a reconstructed table, which the caller
    excludes from form-pair extraction so a colon-bearing cell does not
    land in both a form element and the table.
    """
    blocks: list[TextBlock] = []
    tables: list[TableData] = []
    consumed: list[OCRRegionResult] = []

    try:
        if page_layout is None or page_layout.status != "ok":
            raise RuntimeError("no layout regions available")

        # Layout renders at the OCR dpi, so its page is the OCR render; the
        # normalised boxes map onto the OCR render's own pixel dimensions.
        if ocr_pix_dims is not None:
            layout_dims = (int(ocr_pix_dims[0]), int(ocr_pix_dims[1]))
        else:
            img = page.render(dpi=dpi)
            layout_dims = (int(img.shape[1]), int(img.shape[0]))
        width, height = float(layout_dims[0]), float(layout_dims[1])

        # Cell regions need their render dimensions to be placed at all.
        cell_source = list(ocr_regions or ())
        if cell_source and ocr_pix_dims is None:
            logger.warning(
                "OCR render dimensions missing, dropping cell regions: page=%d", page.number,
            )
            cell_source = []
        # Deskew rotates the OCR input, so region coords leave the frame the
        # layout was found in (A2). Mapping the layout rect into deskewed space
        # is deferred to the round that targets scans.
        if cell_source and page_deskewed:
            logger.debug(
                "deskewed page, refusing table reconstruction: page=%d", page.number,
            )
            cell_source = []

        regions = page_layout.regions
        if not regions:
            raise RuntimeError("no layout regions detected")

        for region in regions:
            rx0, ry0 = region.bbox.x * width, region.bbox.y * height
            rx1 = (region.bbox.x + region.bbox.width) * width
            ry1 = (region.bbox.y + region.bbox.height) * height
            pos = _normalise_bbox((rx0, ry0, rx1, ry1), width, height)

            if region.block_type == "table":
                table = None
                if cell_source:
                    table = reconstruct_table(
                        cell_source, (rx0, ry0, rx1, ry1), dpi, region.confidence,
                        pix_dims=layout_dims,
                    )
                logger.debug(
                    "layout table region: page=%d confidence=%.2f reconstructed=%s",
                    page.number, region.confidence, table is not None,
                )
                if table is not None:
                    tables.append(table)
                    consumed.extend(regions_in_rect(cell_source, (rx0, ry0, rx1, ry1)))
                    continue
                # Refused: keep the placeholder so the region is still visible
                # to the fallback's "no non-table block has text" test.
                blocks.append(TextBlock(
                    text="[TABLE]",
                    position=pos,
                    block_type="table",
                    confidence=region.confidence,
                ))
            else:
                blocks.append(TextBlock(
                    text="",  # layout region text not yet segmented from OCR output
                    position=pos,
                    block_type=region.block_type,
                    confidence=region.confidence,
                ))

        # Layout-derived blocks carry no text, so this fires on essentially
        # every layout-successful page: the page's OCR text collapses onto one
        # block. Where a table reconstructed, that text is rebuilt from the
        # regions *outside* its rect — otherwise the chunker sees the table
        # twice, once as narrative and once as markdown.
        if consumed or (blocks and not any(
            b.text.strip() for b in blocks if b.block_type != "table"
        )):
            page_text = text
            if consumed:
                taken = {id(r) for r in consumed}
                page_text = _table_aware_text(
                    [r for r in cell_source if id(r) not in taken],
                )
            block = _ocr_text_block(page, page_text, conf)
            if block:
                # The whole page's OCR text is being collapsed into this one
                # block. If the dominant region is a non-text kind (figure) but
                # the page yielded substantial prose, it is a full-page scan,
                # not a figure — tag it paragraph so it is not silently dropped
                # from chunking. K9-fig. Where a table reconstructed, what is
                # left is by construction the non-table remainder, so the
                # dominant region's kind no longer describes it.
                if consumed:
                    block_type = "paragraph"
                else:
                    dominant = max(regions, key=lambda r: r.bbox.width * r.bbox.height)
                    block_type = _ocr_region_block_type(block.text, dominant.block_type)
                block = TextBlock(
                    text=block.text,
                    position=block.position,
                    block_type=block_type,
                    confidence=block.confidence,
                )
                return [block], tables, consumed
            if consumed:
                # A page that is only a table: no narrative left over. Emitting
                # the empty layout placeholders would add blank paragraphs.
                return [], tables, consumed

    except (FileNotFoundError, Exception):
        # Precision-first: a partially-built page must not reach the caller
        # with tables emitted but the narrative not yet subtracted — that is
        # the double-count A3 exists to prevent. Reconstruction added a throw
        # site inside the region loop, so discard the partial page entirely
        # and let the full-text fallback below rebuild it.
        blocks, tables, consumed = [], [], []

    if not blocks:
        block = _ocr_text_block(page, text, conf)
        if block:
            blocks = [block]

    return blocks, tables, consumed

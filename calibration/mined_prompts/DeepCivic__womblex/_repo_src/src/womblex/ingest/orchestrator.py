"""Plan-driven extraction orchestrator (Phase 2).

Walks per-page `PageProfile`s and dispatches page-level operations,
collapsing the previous document-level strategy switch
(NativeNarrative / NativeWithStructured / Structured / Hybrid /
ScannedMachinewritten / ScannedHandwritten / ScannedMixed) into one
loop with mode-based dispatch.

Per-page operations are the same primitives Phase 1 produced:
- native pages: extract_page_text + _build_text_blocks + _extract_forms
  + _extract_tables_from_page + _extract_images_from_page
- OCR pages: _ocr_page + _extract_form_pairs_from_lines + either
  _markdown_page_block (markdown engines) or _layout_blocks_and_tables
- mixed-typed docs: per-page typed/handwritten classification on OCR
  pages
- scanned-machinewritten: table-grid fallback when layout pass finds
  none

The orchestrator accepts a doc-level type hint (from `summarise_doc_type`
or the legacy detector) for two narrow uses: enabling the mixed-typed
classification and the scanned-machinewritten table-grid fallback. All
other dispatch is page-level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from womblex.ingest.detect import DocumentProfile, DocumentType
from womblex.ingest.elements import TEXT_KINDS, Element, ElementKind, FieldEntry
from womblex.ingest.extract import (
    ExtractionMetadata,
    ExtractionResult,
    FormField,
    ImageData,
    PageResult,
    TableData,
    TextBlock,
    _build_text_blocks,
    _emit_table_column_major,
    _extract_form_pairs_from_lines,
    _extract_form_pairs_from_regions,
    _extract_forms,
    _extract_images_from_page,
    _extract_tables_from_page,
    _find_native_tables,
    _page_to_gray,
    _text_coverage,
)
from womblex.ingest.grid_projection import extract_page_text
from womblex.ingest.page_profile import PageProfile, qualify_for_spreadsheet_print
from womblex.ingest.views import table_to_element

if TYPE_CHECKING:
    from womblex.ingest.layout_step import LayoutSettings, PageLayout
    from womblex.ingest.pdf.types import Document, Page

logger = logging.getLogger(__name__)


@dataclass
class _PageAccum:
    """Accumulator across a single page's operations."""

    page_number: int
    text: str = ""
    method: str = ""
    confidence: float = 0.0
    blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    forms: list[FormField] = field(default_factory=list)
    images: list[ImageData] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


def _apply_native_page(
    page: Page,
    profile: PageProfile,
    accum: _PageAccum,
    *,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
) -> None:
    """Extract a native text-layer page; also OCR sizable embedded image
    regions that aren't already covered by the text layer.

    `_ocr_image_regions` self-protects: only fires on image rects ≥50 px
    in both dimensions with ≤2 overlapping native words, so all-native
    pages without embedded images pay only metadata-access overhead.
    """
    from womblex.ingest.strategies_scanned import _ocr_image_regions

    # Detect tables. Ruled tables (lines strategy, confidence ≥ 0.8) drive
    # prose-region exclusion + column-major re-emission, because they
    # represent reliable cell structure that the row-major prose path
    # would otherwise interleave (e.g. the 3-column rules-of-the-Law
    # table on Compliance Notice page 1).
    #
    # Text-strategy hits (whitespace-aligned columns, confidence ≈ 0.6)
    # stay in `tables` but do NOT drive prose partitioning — they
    # over-fire on ordinary multi-column prose where word x-positions
    # happen to cluster. The spreadsheet-print path handles the
    # whitespace-aligned manifest case separately.
    table_records = _find_native_tables(page)
    ruled_table_records = [r for r in table_records if r[0].confidence >= 0.8]
    ruled_rects = [rect for _td, rect, _cells in ruled_table_records]
    native_text = extract_page_text(page, exclude_rects=ruled_rects or None)
    if ruled_table_records:
        ruled_table_records.sort(key=lambda r: r[1].y0)
        column_blocks = [
            _emit_table_column_major(cells)
            for _td, _rect, cells in ruled_table_records
        ]
        column_blocks = [b for b in column_blocks if b]
        if column_blocks:
            joined_tables = "\n\n".join(column_blocks)
            native_text = (
                f"{native_text}\n\n{joined_tables}" if native_text else joined_tables
            )
    accum.text = native_text
    accum.method = "native"
    accum.confidence = 95.0
    accum.blocks.extend(_build_text_blocks(page))
    accum.tables.extend(td for td, _rect, _cells in table_records)
    accum.forms.extend(_extract_forms(page))
    accum.images.extend(_extract_images_from_page(page))

    # Only attempt sub-page OCR if the page actually has images. Skips the
    # paddle-reader cold-start for native docs without any embedded images.
    if profile.image_count == 0:
        return

    native_words = page.words(dehyphenate=False)
    sub_blocks, sub_steps = _ocr_image_regions(
        page, native_words, dpi, lang,
        engine=engine, engine_options=engine_options or {},
    )
    if sub_blocks:
        accum.text = native_text + "\n\n" + "\n\n".join(b.text for b in sub_blocks)
        accum.method = "native+ocr"
        accum.blocks.extend(sub_blocks)
        accum.steps.extend(sub_steps)


def _apply_ocr_page(
    page: Page,
    profile: PageProfile,
    accum: _PageAccum,
    *,
    dpi: int,
    lang: str,
    engine: str,
    engine_options: dict,
    doc_type: DocumentType,
    page_layout: PageLayout | None = None,
) -> None:
    # Imported lazily to keep startup paths free of OCR deps.
    from womblex.ingest.strategies_scanned import (
        _layout_blocks_and_tables,
        _markdown_page_block,
        _ocr_page,
    )

    text, conf, steps, native_order, regions, pix_dims = _ocr_page(
        page, dpi, lang, engine, engine_options,
    )
    accum.text = text
    accum.method = "ocr"
    accum.confidence = conf
    accum.steps.extend(steps)

    # Table structure is a region-based concern: LLM/VLM engines resolve
    # reading order natively and emit markdown with no regions, so there is
    # nothing to reconstruct cells from and the layout pass is bypassed
    # entirely. Only the region-based (paddleocr) branch carries `regions` +
    # `pix_dims` through to the layout pass. See
    # docs/decisions.md “Table-cell reconstruction on OCR pages”.
    consumed: list = []
    if native_order:
        accum.blocks.extend(_markdown_page_block(page, text, conf))
        page_tables: list[TableData] = []
    else:
        page_blocks, page_tables, consumed = _layout_blocks_and_tables(
            page, dpi, text, conf,
            ocr_regions=regions, ocr_pix_dims=pix_dims,
            # A2: deskew rotated the OCR input, so the region coords no longer
            # share the layout render's frame — refuse reconstruction there.
            page_deskewed="deskew" in steps,
            page_layout=page_layout,
        )
        accum.blocks.extend(page_blocks)

    # K2′: prefer per-region extraction (preserves bbox) when the OCR engine
    # returned per-detection results. LLM engines that resolve reading order
    # natively yield no regions — fall back to bbox-less line extraction.
    # Regions a reconstructed table absorbed are excluded, so a colon-bearing
    # cell doesn't land in both a form element and the table (A3).
    if regions:
        pw, ph = pix_dims
        taken = {id(r) for r in consumed}
        form_regions = [r for r in regions if id(r) not in taken]
        accum.forms.extend(
            _extract_form_pairs_from_regions(form_regions, float(pw), float(ph)),
        )
    else:
        accum.forms.extend(_extract_form_pairs_from_lines(text))

    # Mixed-doc tagging: classify content_type per page on OCR pages.
    if doc_type == DocumentType.SCANNED_MIXED:
        from womblex.ingest.heuristics_cv2 import analyze_contour_complexity

        gray = _page_to_gray(page, dpi=dpi)
        complexity = analyze_contour_complexity(gray)
        is_typed = complexity.regularity > 0.5
        content_type = "typed" if is_typed else "handwritten"
        accum.blocks = [
            TextBlock(text=b.text, position=b.position, block_type=content_type, confidence=b.confidence)
            for b in accum.blocks
        ]
        accum.steps.append(f"mixed:{content_type}")

    accum.tables.extend(page_tables)

    # Scanned-machinewritten table-grid fallback (parity with the legacy
    # ScannedMachinewrittenExtractor): if layout pass produced no tables
    # and the page rendered as a grid, retry via the native table finder.
    if (
        not page_tables
        and doc_type == DocumentType.SCANNED_MACHINEWRITTEN
    ):
        from womblex.ingest.heuristics_cv2 import detect_table_grid

        gray = _page_to_gray(page, dpi=dpi)
        grid = detect_table_grid(gray)
        if grid.has_grid:
            accum.tables.extend(_extract_tables_from_page(page))


# Map TextBlock.block_type values onto Element kinds.
# 'table' and mixed-doc tags become 'paragraph' (the block is text from a
# table-region or typed/handwritten region — not a structured table or a
# distinct kind). Unknown values fall through to 'paragraph'.
_BLOCK_TYPE_TO_KIND: dict[str, ElementKind] = {
    "paragraph": "paragraph",
    "heading": "heading",
    "list_item": "list_item",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
    "footnote": "footnote",
    "signature": "signature",
    "figure": "figure",
    "table": "paragraph",
    "typed": "paragraph",
    "handwritten": "paragraph",
}


def _block_to_element(b: TextBlock, page: int, extractor: str, order: int) -> Element:
    kind = _BLOCK_TYPE_TO_KIND.get(b.block_type, "paragraph")
    meta: dict[str, str] = {}
    if b.block_type not in TEXT_KINDS and b.block_type not in ("figure",):
        meta["block_type"] = b.block_type
    return Element(
        order=order, kind=kind, extractor=extractor,
        page=page, bbox=b.position,
        text=b.text, confidence=b.confidence,
        meta=meta,
    )


def _form_to_element(forms: list[FormField], page: int, extractor: str, order: int) -> Element:
    """Group one page's form fields into a single kind='form' element.

    The element's bbox uses the first field's position as a placeholder;
    per-field bbox is not preserved on the element model.
    """
    fields = [FieldEntry(name=f.field_name, value=f.value) for f in forms]
    bbox = forms[0].position if forms else None
    conf = sum(f.confidence for f in forms) / len(forms) if forms else 0.0
    return Element(
        order=order, kind="form", extractor=extractor,
        page=page, bbox=bbox,
        fields=fields, confidence=conf,
    )


def _image_to_element(im: ImageData, page: int, extractor: str, order: int) -> Element:
    return Element(
        order=order, kind="image", extractor=extractor,
        page=page, bbox=im.position,
        alt_text=im.alt_text, confidence=im.confidence,
    )


def _accum_to_elements(
    accum: _PageAccum, start_order: int, *, include_tables: bool,
) -> tuple[list[Element], int]:
    """Convert one page's accumulator to ordered elements.

    Blocks, tables and images are sorted by their y, x position so reading
    order survives within a page. Forms collapse to one form element per
    page. ``include_tables=False`` is set when spreadsheet-print owns the
    table column document-wide.
    """
    extractor = "native_text" if accum.method.startswith("native") else "ocr_paddle"
    placed: list[tuple[float, float, str, object]] = []
    for b in accum.blocks:
        placed.append((b.position.y, b.position.x, "block", b))
    if include_tables:
        for t in accum.tables:
            placed.append((t.position.y, t.position.x, "table", t))
    for im in accum.images:
        placed.append((im.position.y, im.position.x, "image", im))
    placed.sort(key=lambda r: (r[0], r[1]))

    elements: list[Element] = []
    order = start_order
    for _y, _x, kind, obj in placed:
        if kind == "block":
            elements.append(_block_to_element(cast("TextBlock", obj), accum.page_number, extractor, order))
        elif kind == "table":
            elements.append(table_to_element(cast("TableData", obj), accum.page_number, extractor, order))
        elif kind == "image":
            elements.append(_image_to_element(cast("ImageData", obj), accum.page_number, "figure_image", order))
        order += 1

    if accum.forms:
        # Forms attach at the page's first form-field y; sort-place into the
        # stream by inserting after the last element above that y. Simpler
        # heuristic: append at the page's end. Downstream cares about page,
        # not within-page-order of forms.
        elements.append(_form_to_element(accum.forms, accum.page_number, "form", order))
        order += 1

    return elements, order


def extract_with_plan(
    doc: Document,
    profiles: list[PageProfile],
    doc_type: DocumentType,
    *,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    filename: str = "",
    spreadsheet_print: dict | None = None,
    layout: LayoutSettings | None = None,
) -> ExtractionResult:
    """Execute a per-page extraction plan and merge results.

    Dispatch is page-level via ``profile.has_text_layer``. The doc-level
    ``doc_type`` is consumed for two narrow concerns: mixed-typed /
    handwritten tagging on OCR pages, and the scanned-machinewritten
    table-grid fallback. All other behaviour is page-driven.

    Returns an ``ExtractionResult`` with ``elements`` ordered across the
    whole document (within-page reading order preserved by sorting on
    ``(y, x)``) plus ``pages`` carrying per-page concatenated text.

    ``spreadsheet_print`` config dict (optional):
    - ``metadata_location``: ``"both"`` | ``"table"`` | ``"document"`` (default ``"both"``)
    - ``filename_hints``: tuple of substrings to fast-trigger the qualifier

    ``layout`` runs the one layout step (after profiling, before any page is
    extracted), carries its regions on the result, and feeds each OCR page its
    own regions. Omitted, no layout runs and OCR pages fall back to whole-page text.
    """
    opts = engine_options or {}
    layout_outcome = None
    if layout is not None:
        from womblex.ingest.layout_step import run_layout_step

        layout_outcome = run_layout_step(doc, profiles, layout, engine)
    page_layouts = {pl.page: pl for pl in layout_outcome.pages} if layout_outcome else {}
    pages: list[PageResult] = []
    all_elements: list[Element] = []
    next_order = 0
    confidences: list[float] = []
    combined_steps: list[str] = []
    native_count = ocr_count = 0
    document_metadata: dict[str, str] = {}

    # Spreadsheet-print fast path: if the qualifier hits, manifest-style
    # tables replace per-page table extraction document-wide.
    sp_cfg = spreadsheet_print or {}
    sp_hints = tuple(sp_cfg.get("filename_hints", ())) or None
    sp_loc = sp_cfg.get("metadata_location", "both")
    sp_qualifier_kwargs: dict = {}
    if sp_hints:
        sp_qualifier_kwargs["filename_hints"] = sp_hints
    spreadsheet_tables: list[TableData] = []
    is_spreadsheet_print = False
    if doc_type in (DocumentType.STRUCTURED, DocumentType.NATIVE_WITH_STRUCTURED) and \
            qualify_for_spreadsheet_print(profiles, filename, **sp_qualifier_kwargs):
        from womblex.ingest.spreadsheet_print import extract_spreadsheet_print
        spreadsheet_tables, document_metadata = extract_spreadsheet_print(
            doc, metadata_location=sp_loc,
        )
        is_spreadsheet_print = bool(spreadsheet_tables)

    for page in doc:
        profile = profiles[page.number]
        accum = _PageAccum(page_number=page.number)

        if profile.has_text_layer:
            native_count += 1
            _apply_native_page(
                page, profile, accum,
                dpi=dpi, lang=lang, engine=engine, engine_options=opts,
            )
        else:
            ocr_count += 1
            _apply_ocr_page(
                page, profile, accum,
                dpi=dpi, lang=lang, engine=engine, engine_options=opts,
                doc_type=doc_type, page_layout=page_layouts.get(page.number),
            )

        pages.append(PageResult(page_number=page.number, text=accum.text, method=accum.method))
        if page.number > 0:
            all_elements.append(Element(
                order=next_order, kind="page_break",
                extractor="orchestrator", page=page.number,
            ))
            next_order += 1
        page_elements, next_order = _accum_to_elements(
            accum, next_order, include_tables=not is_spreadsheet_print,
        )
        all_elements.extend(page_elements)
        confidences.append(accum.confidence)
        combined_steps.extend(accum.steps)

    if is_spreadsheet_print:
        # Manifest tables append after all per-page elements; they have no
        # natural per-page anchor (they span pages) so they tail the stream.
        for t in spreadsheet_tables:
            all_elements.append(table_to_element(t, None, "spreadsheet_print", next_order))
            next_order += 1
        combined_steps.append("spreadsheet_print")

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    coverage = _text_coverage(pages)
    unique_steps = sorted(set(combined_steps))

    content_mix: dict[str, float] = {}
    total = native_count + ocr_count
    if total > 0:
        content_mix = {"native": native_count / total, "scanned": ocr_count / total}
    if is_spreadsheet_print:
        content_mix["spreadsheet_print"] = 1.0

    # Native pages contribute 95 (treated as percent); OCR pages contribute
    # paddle's 0–100 confidence. Normalise to 0–1 for the metadata field.
    confidence_01 = avg_conf / 100.0 if avg_conf > 1 else avg_conf

    return ExtractionResult(
        pages=pages,
        elements=all_elements,
        layout=layout_outcome,
        method=doc_type.value,
        document_metadata=document_metadata,
        metadata=ExtractionMetadata(
            extraction_strategy=doc_type.value,
            confidence=confidence_01,
            processing_time=0.0,
            page_count=len(doc),
            text_coverage=coverage,
            preprocessing_steps=unique_steps,
            content_mix=content_mix,
        ),
    )


def extract_pdf_with_plan(
    doc: Document,
    profile: DocumentProfile,
    *,
    dpi: int = 200,
    lang: str = "eng",
    engine: str = "paddleocr",
    engine_options: dict | None = None,
    filename: str = "",
    spreadsheet_print: dict | None = None,
    layout: LayoutSettings | None = None,
) -> ExtractionResult:
    """Convenience: profile pages, summarise type, run the orchestrator."""
    from womblex.ingest.page_profile import profile_pages, summarise_doc_type

    profiles = profile_pages(doc)
    doc_type = summarise_doc_type(profiles, profile)
    logger.debug(
        "plan: pages=%d native=%d ocr=%d tables=%d forms=%d type=%s",
        len(profiles),
        sum(1 for p in profiles if p.has_text_layer),
        sum(1 for p in profiles if p.needs_ocr),
        sum(1 for p in profiles if p.has_table_signal),
        sum(1 for p in profiles if p.has_form_signal),
        doc_type.value,
    )
    return extract_with_plan(
        doc, profiles, doc_type,
        dpi=dpi, lang=lang, engine=engine, engine_options=engine_options,
        filename=filename, spreadsheet_print=spreadsheet_print, layout=layout,
    )

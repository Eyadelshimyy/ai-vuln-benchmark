"""The batch's one layout step: select pages, run the analyser once per page.

Runs after page profiling and before OCR and redaction detection, because it
needs each page's route. Its output is carried on ``ExtractionResult.layout``
and written to ``*.layout_regions.parquet`` by ``store.output.write_results``.
OCR reads each page's regions for table rects and block types, and redaction
reads them for its exclusion zones.

Boxes are stored normalised (0-1, top-left), converted from the pixels of the
render the analyser saw, so any later consumer maps them onto its own render.

A page that fails is a status row, not an error: the document continues and
the failure is on record. Only an unknown model name raises, as a
configuration error must not read as a missing model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from womblex.ingest.elements import BBox

if TYPE_CHECKING:
    from womblex.config import WomblexConfig
    from womblex.ingest.page_profile import PageProfile
    from womblex.ingest.pdf.types import Document, Page
    from womblex.store.layout_output import LayoutFingerprint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayoutSettings:
    """What the step needs from the configuration, resolved once per batch."""

    model: str
    options: dict
    page_scope: str
    dpi: int
    redaction_filter: bool
    fingerprint: LayoutFingerprint

    @classmethod
    def from_config(cls, config: WomblexConfig) -> LayoutSettings:
        from womblex.store.layout_output import layout_fingerprint

        return cls(
            model=config.layout.model,
            options=dict(config.layout.options),
            page_scope=config.layout.page_scope,
            dpi=config.extraction.ocr.dpi,
            redaction_filter=config.redaction.enabled and config.redaction.use_layout_filter,
            fingerprint=layout_fingerprint(config),
        )


@dataclass
class LayoutRegion:
    bbox: BBox  # normalised 0-1, top-left
    label: str
    block_type: str
    confidence: float


@dataclass
class PageLayout:
    """One analysed page: ``ok`` with regions, ``empty``, or ``error`` with a reason."""

    page: int  # 0-based, as on elements
    status: str
    regions: list[LayoutRegion] = field(default_factory=list)
    error: str = ""


@dataclass
class LayoutOutcome:
    """A document's layout: the pages analysed, and what produced them.

    ``pages`` is empty for a document the step does not apply to (DOCX,
    spreadsheets, text) and for one with no page in scope; the fingerprint is
    still recorded, as what extraction consumed.
    """

    fingerprint: LayoutFingerprint
    redaction_consumed: bool
    pages: list[PageLayout] = field(default_factory=list)


def _is_markdown_engine(engine: str) -> bool:
    import womblex.ingest.paddle_ocr  # noqa: F401  (registers the built-ins)
    from womblex.utils.model_registry import SLOT_OCR, resolve

    try:
        return bool(resolve(SLOT_OCR, engine).traits.get("markdown"))
    except ValueError:
        return False  # the unknown engine is reported by OCR itself


def select_pages(
    doc: Document,
    profiles: list[PageProfile],
    settings: LayoutSettings,
    engine: str,
) -> list[int]:
    """The pages to analyse under ``settings.page_scope``.

    ``consumers``: OCR-routed pages (a markdown engine bypasses layout, so it
    contributes none) plus, when redaction's layout filter is on, pages with
    no vector redaction. ``all``: every page.
    """
    if settings.page_scope == "all":
        return [p.number for p in doc]
    from womblex.redact.stage import _detect_vector_redactions

    ocr_layout = not _is_markdown_engine(engine)
    scale = settings.dpi / 72.0
    chosen: list[int] = []
    for page in doc:
        wanted = ocr_layout and not profiles[page.number].has_text_layer
        if not wanted and settings.redaction_filter:
            try:
                wanted = not _detect_vector_redactions(page, page.number, scale)
            except Exception as e:
                # Unreadable drawings must not fail extraction; analysing the
                # page is the safe direction for redaction's filter.
                logger.warning("layout scope: page=%d drawings unreadable: %s", page.number, e)
                wanted = True
        if wanted:
            chosen.append(page.number)
    return chosen


def _analyse_page(page: Page, analyzer: object, dpi: int) -> PageLayout:
    from womblex.ingest.interfaces.protocols import check_layout_regions

    img = page.render(dpi=dpi)
    height, width = float(img.shape[0]), float(img.shape[1])
    found = analyzer.analyze(img)  # type: ignore[attr-defined]
    check_layout_regions(found)
    if not found:
        return PageLayout(page.number, "empty")
    regions = []
    for r in found:
        x0, y0, x1, y1 = r.bbox
        nx0, ny0 = max(0.0, x0 / width), max(0.0, y0 / height)
        nx1, ny1 = min(1.0, x1 / width), min(1.0, y1 / height)
        if nx1 <= nx0 or ny1 <= ny0:
            continue  # wholly off the page
        regions.append(LayoutRegion(
            BBox(nx0, ny0, nx1 - nx0, ny1 - ny0), r.label, r.block_type, float(r.confidence),
        ))
    return PageLayout(page.number, "ok", regions) if regions else PageLayout(page.number, "empty")


def run_layout_step(
    doc: Document,
    profiles: list[PageProfile],
    settings: LayoutSettings,
    engine: str,
) -> LayoutOutcome:
    """Analyse the selected pages once each and return the document's layout."""
    outcome = LayoutOutcome(settings.fingerprint, settings.redaction_filter)
    selected = select_pages(doc, profiles, settings, engine)
    if selected:
        outcome.pages = analyse_pages(doc, selected, settings.model, settings.options, settings.dpi)
    return outcome


def analyse_pages(
    doc: Document, numbers: list[int], model: str, options: dict, dpi: int,
) -> list[PageLayout]:
    """Analyse *numbers* with the named model, one ``PageLayout`` per page.

    A model that cannot be built or a page that fails is an ``error`` row, not
    an exception; only an unknown model name raises.
    """
    from womblex.ingest.paddle_ocr import get_layout_analyzer
    from womblex.utils.model_registry import SLOT_LAYOUT, resolve

    resolve(SLOT_LAYOUT, model)  # an unknown name is a config error: raise
    analyzer: object | None = None
    build_error = ""
    try:
        analyzer = get_layout_analyzer(model, **options)
    except Exception as e:
        build_error = f"{type(e).__name__}: {e}"
        logger.warning("layout model %r unavailable: %s", model, build_error)

    pages: list[PageLayout] = []
    for number in numbers:
        if analyzer is None:
            pages.append(PageLayout(number, "error", error=build_error))
            continue
        try:
            pages.append(_analyse_page(doc[number], analyzer, dpi))
        except Exception as e:
            logger.warning("layout failed: page=%d error=%s", number, e)
            pages.append(PageLayout(number, "error", error=f"{type(e).__name__}: {e}"))
    return pages


__all__ = [
    "LayoutOutcome",
    "LayoutRegion",
    "LayoutSettings",
    "PageLayout",
    "analyse_pages",
    "run_layout_step",
    "select_pages",
]

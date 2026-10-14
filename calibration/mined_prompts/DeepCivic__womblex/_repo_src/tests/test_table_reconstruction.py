"""Table-cell reconstruction on OCR'd pages (#17).

Round 1. A0 pinned the scope (region-based engines) and the plumbing that
carries OCR regions into the layout pass. A1 added the shared grid module
(``table_grid``) and the OCR feeder (``ocr_tables.reconstruct_table``).
A3 wires the feeder into the layout pass on the OCR-PDF path — a detected
table region whose cells reconstruct becomes a ``TableData``, the page
narrative is rebuilt from the regions outside its rect, and the absorbed
regions are withheld from form-pair extraction. A2 refuses reconstruction
outright on deskewed pages. A4 closed as a no-op: the legacy
``ImageExtractor`` it named was unreachable — ``extract_text`` routes
standalone images through the orchestrator too — so it was deleted rather
than wired up. ``TestImageDocumentsRouteThroughTheOrchestrator`` pins that.
"""

from __future__ import annotations

import logging

import pytest
from PIL import Image

from tests._pdf_builders import PdfBuilder
from womblex.ingest.elements import BBox
from womblex.ingest.interfaces.protocols import (
    LayoutRegionResult,
    OCRPageResult,
    OCRRegionResult,
)
from womblex.ingest.layout_step import LayoutRegion, PageLayout
from womblex.ingest.ocr_tables import (
    reconstruct_table,
    regions_in_rect,
    span_from_region,
)
from womblex.ingest.page_profile import PageProfile
from womblex.ingest.pdf.types import Page
from womblex.ingest.strategies_scanned import (
    _layout_blocks_and_tables,
    _spatial_sort_regions,
    _table_aware_text,
)
from womblex.ingest.table_grid import Span, rows_from_spans


def _region(x0: int, y0: int, x1: int, y1: int, text: str = "cell") -> OCRRegionResult:
    return OCRRegionResult(
        bbox=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        text=text,
        confidence=0.9,
    )


def _ocr_profile() -> PageProfile:
    """A scanned page profile — no text layer, OCR required."""
    return PageProfile(
        page_number=0, width=612.0, height=792.0,
        char_count=0, image_count=1, vector_drawings=0,
        has_text_layer=False, needs_ocr=True,
        has_table_signal=True, has_form_signal=False,
        has_handwriting_signal=False,
    )


class _StubAnalyzer:
    def __init__(self, regions: list[LayoutRegionResult]) -> None:
        self._regions = regions

    def analyze(self, img) -> list[LayoutRegionResult]:
        return self._regions


def _layout_for(
    regions: list[LayoutRegionResult], dims: tuple[int, int] = (1700, 2200),
) -> PageLayout:
    """The layout step's output for a page: pixel boxes, normalised by the render."""
    w, h = dims
    return PageLayout(0, "ok", [
        LayoutRegion(
            BBox(r.bbox[0] / w, r.bbox[1] / h, (r.bbox[2] - r.bbox[0]) / w, (r.bbox[3] - r.bbox[1]) / h),
            r.label, r.block_type, r.confidence,
        )
        for r in regions
    ])


_PLUMBING_LAYOUT = _layout_for([
    LayoutRegionResult(bbox=(0, 0, 400, 400), label="Table", block_type="table", confidence=0.96),
    LayoutRegionResult(bbox=(0, 500, 400, 700), label="Text", block_type="paragraph", confidence=0.90),
])


@pytest.fixture
def blank_page(tmp_path):
    with PdfBuilder(tmp_path / "blank.pdf").page(612, 792).text(72, 100, "Some page text").open() as doc:
        yield doc[0]


class TestRegionsInRect:
    """Centroid containment — the OCR-quad → table-rect intersection."""

    def test_selects_by_centroid(self) -> None:
        inside = _region(100, 100, 200, 130)
        outside = _region(100, 500, 200, 530)
        got = regions_in_rect([inside, outside], (50.0, 50.0, 300.0, 300.0))
        assert got == [inside]

    def test_straddling_region_follows_its_middle(self) -> None:
        # Spans the rect's bottom edge; centroid (y=295) is inside.
        straddles_in = _region(100, 260, 200, 330)
        # Mostly below; centroid (y=315) is outside.
        straddles_out = _region(100, 280, 200, 350)
        got = regions_in_rect([straddles_in, straddles_out], (50.0, 50.0, 300.0, 300.0))
        assert got == [straddles_in]

    def test_blank_regions_dropped(self) -> None:
        assert regions_in_rect([_region(100, 100, 200, 130, "   ")], (0.0, 0.0, 300.0, 300.0)) == []

    def test_no_regions(self) -> None:
        assert regions_in_rect([], (0.0, 0.0, 300.0, 300.0)) == []


class TestLayoutPassPlumbing:
    """A0 — the layout pass accepts OCR regions and maps layout onto their render."""

    def test_regions_are_optional(self, blank_page: Page) -> None:
        """Callers without regions (legacy, tests) keep today's behaviour."""
        blocks, tables, consumed = _layout_blocks_and_tables(
            blank_page, 200, "page text", 90.0, page_layout=_PLUMBING_LAYOUT,
        )
        # With no cell source there is nothing to reconstruct, so the fallback
        # still collapses the page — table content included — onto one block.
        assert len(blocks) == 1
        assert blocks[0].text == "page text"
        assert tables == []
        assert consumed == []

    def test_matching_render_dims_keep_regions(
        self, blank_page: Page, caplog,
    ) -> None:
        height, width = blank_page.render(dpi=200).shape[:2]
        with caplog.at_level(logging.WARNING, logger="womblex.ingest.strategies_scanned"):
            _blocks, tables, _consumed = _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[_region(10, 10, 100, 40)],
                ocr_pix_dims=(width, height), page_layout=_PLUMBING_LAYOUT,
            )
        assert "dropping cell regions" not in caplog.text
        # One stray region inside the rect is not a grid — the gates refuse.
        assert tables == []

    def test_debug_log_reports_the_reconstruction_outcome(
        self, blank_page: Page, caplog,
    ) -> None:
        """Every detected table region logs whether cells came out of it."""
        height, width = blank_page.render(dpi=200).shape[:2]
        with caplog.at_level(logging.DEBUG, logger="womblex.ingest.strategies_scanned"):
            _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[_region(10, 10, 100, 40)],
                ocr_pix_dims=(width, height), page_layout=_PLUMBING_LAYOUT,
            )
        assert (
            "layout table region: page=0 confidence=0.96 reconstructed=False"
            in caplog.text
        )

    def test_regions_without_dims_are_dropped(
        self, blank_page: Page, caplog,
    ) -> None:
        """Unverifiable is treated as non-comparable: regions need their dims."""
        with caplog.at_level(logging.WARNING, logger="womblex.ingest.strategies_scanned"):
            _blocks, tables, _consumed = _layout_blocks_and_tables(
                blank_page, 200, "page text", 90.0,
                ocr_regions=[_region(10, 10, 100, 40)], page_layout=_PLUMBING_LAYOUT,
            )
        assert "dropping cell regions" in caplog.text
        assert tables == []


class TestOcrPageScoping:
    """A0 — only the region-based branch reaches the layout pass."""

    def _patch_ocr_page(
        self, monkeypatch, *, native_order: bool, regions: list,
        steps: list[str] | None = None,
    ) -> None:
        def _fake_ocr_page(page, dpi, lang, engine, engine_options):
            return ("page text", 90.0, list(steps or []), native_order, regions, (1700, 2200))

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._ocr_page", _fake_ocr_page,
        )

    def test_region_engine_forwards_regions_and_dims(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        from womblex.ingest.detect import DocumentType
        from womblex.ingest.orchestrator import _apply_ocr_page, _PageAccum

        regions = [_region(10, 10, 100, 40)]
        self._patch_ocr_page(monkeypatch, native_order=False, regions=regions)

        seen: dict = {}

        def _fake_layout(
            page, dpi, text, conf,
            ocr_regions=None, ocr_pix_dims=None, page_deskewed=False, **_layout,
        ):
            seen["regions"] = ocr_regions
            seen["pix_dims"] = ocr_pix_dims
            seen["deskewed"] = page_deskewed
            return [], [], []

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._layout_blocks_and_tables", _fake_layout,
        )

        accum = _PageAccum(page_number=0)
        _apply_ocr_page(
            blank_page, _ocr_profile(), accum,
            dpi=200, lang="eng", engine="paddleocr", engine_options={},
            doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        )
        assert seen["regions"] == regions
        assert seen["pix_dims"] == (1700, 2200)

    def test_llm_engine_bypasses_layout_pass(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        """LLM/VLM engines emit markdown with no regions — nothing to reconstruct."""
        from womblex.ingest.detect import DocumentType
        from womblex.ingest.orchestrator import _apply_ocr_page, _PageAccum

        self._patch_ocr_page(monkeypatch, native_order=True, regions=[])

        def _must_not_run(*args, **kwargs):
            raise AssertionError("layout pass reached on a native-reading-order engine")

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._layout_blocks_and_tables", _must_not_run,
        )

        accum = _PageAccum(page_number=0)
        _apply_ocr_page(
            blank_page, _ocr_profile(), accum,
            dpi=200, lang="eng", engine="mistral-ocr", engine_options={},
            doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        )
        assert accum.tables == []


class TestSharedGridHelpers:
    """A1 — one algorithm in ``table_grid``, consumed by both feeders."""

    def test_span_from_region_reduces_quad_to_bounds(self) -> None:
        r = OCRRegionResult(
            bbox=[[10, 20], [110, 25], [108, 55], [12, 50]],
            text="cell", confidence=0.9,
        )
        s = span_from_region(r)
        assert (s.x_left, s.y_top, s.x_right, s.y_bottom) == (10, 20, 110, 55)
        assert s.text == "cell"

    def test_rows_from_spans_clusters_by_centroid(self) -> None:
        spans = [
            Span(y_top=0, y_bottom=10, x_left=50, x_right=60, text="b"),
            Span(y_top=1, y_bottom=11, x_left=10, x_right=20, text="a"),
            Span(y_top=30, y_bottom=40, x_left=10, x_right=20, text="c"),
        ]
        rows, avg_h = rows_from_spans(spans)
        assert [[s.text for s in row] for row in rows] == [["a", "b"], ["c"]]
        assert avg_h == 10.0

    def test_rows_from_spans_empty(self) -> None:
        assert rows_from_spans([]) == ([], 0.0)

    def test_spatial_sort_reads_row_major(self) -> None:
        regions = [
            _region(400, 100, 500, 130, "B"),
            _region(100, 100, 200, 130, "A"),
            _region(100, 300, 200, 330, "C"),
        ]
        assert _spatial_sort_regions(regions) == "A B\nC"

    def test_table_aware_text_emits_table_runs_column_major(self) -> None:
        regions = [
            _region(100 + 300 * c, 100 + 100 * r, 220 + 300 * c, 130 + 100 * r,
                    f"{'abc'[c]}{r + 1}")
            for r in range(3) for c in range(3)
        ]
        assert _table_aware_text(regions) == "a1 a2 a3\n\nb1 b2 b3\n\nc1 c2 c3"


def _grid_regions(
    n_body_rows: int = 3,
    n_cols: int = 4,
    *,
    col_pitch: int = 300,
    row_pitch: int = 100,
    conf: float = 0.9,
) -> list[OCRRegionResult]:
    """A clean grid at 200 dpi spacing: header row H1..Hn plus body cells r{r}c{c}."""
    regions = []
    for r in range(n_body_rows + 1):
        for c in range(n_cols):
            x0, y0 = 100 + col_pitch * c, 100 + row_pitch * r
            regions.append(OCRRegionResult(
                bbox=[[x0, y0], [x0 + 120, y0], [x0 + 120, y0 + 30], [x0, y0 + 30]],
                text=f"H{c + 1}" if r == 0 else f"r{r}c{c + 1}",
                confidence=conf,
            ))
    return regions


class TestReconstructTable:
    """A1 — the OCR feeder produces a TableData, or refuses (never a partial)."""

    # Tall enough to contain every grid these tests build — a rect that
    # clips the last row changes which gate fires and hides the one under test.
    RECT = (50.0, 50.0, 1300.0, 900.0)
    DIMS = (1700, 2200)

    def test_clean_grid_reconstructs(self) -> None:
        table = reconstruct_table(_grid_regions(), self.RECT, 200, 0.96, pix_dims=self.DIMS)
        assert table is not None
        assert table.headers == ["H1", "H2", "H3", "H4"]
        assert table.rows == [
            ["r1c1", "r1c2", "r1c3", "r1c4"],
            ["r2c1", "r2c2", "r2c3", "r2c4"],
            ["r3c1", "r3c2", "r3c3", "r3c4"],
        ]

    def test_lineage_confidence_and_producer(self) -> None:
        """A5 — confidence from constituent regions, producer marker set."""
        table = reconstruct_table(
            _grid_regions(conf=0.8), self.RECT, 200, 0.96, pix_dims=self.DIMS,
        )
        assert table is not None
        assert table.confidence == pytest.approx(0.8)
        assert table.context["producer"] == "table_grid"

    def test_detector_confidence_caps_region_confidence(self) -> None:
        table = reconstruct_table(
            _grid_regions(conf=0.9), self.RECT, 200, 0.5, pix_dims=self.DIMS,
        )
        assert table is not None
        assert table.confidence == pytest.approx(0.5)

    def test_position_normalised_by_pix_dims(self) -> None:
        table = reconstruct_table(_grid_regions(), self.RECT, 200, 0.96, pix_dims=self.DIMS)
        assert table is not None
        assert table.position.x == pytest.approx(50 / 1700)
        assert table.position.y == pytest.approx(50 / 2200)
        assert table.position.width == pytest.approx(1250 / 1700)
        assert table.position.height == pytest.approx(850 / 2200)

    def test_refuses_too_few_columns(self) -> None:
        got = reconstruct_table(
            _grid_regions(n_cols=2), self.RECT, 200, 0.96, pix_dims=self.DIMS,
        )
        assert got is None

    @pytest.mark.parametrize("n_body_rows", [1, 2])
    def test_refuses_too_few_rows(self, n_body_rows: int) -> None:
        """Two body rows also refuse — the column-population floor bites first."""
        got = reconstruct_table(
            _grid_regions(n_body_rows=n_body_rows), self.RECT, 200, 0.96, pix_dims=self.DIMS,
        )
        assert got is None

    def test_blank_leading_cell_does_not_merge_into_the_header(self) -> None:
        """The continuation rule must not fold the first body row into row 0.

        A blank leading cell (indented or grouped rows) is ordinary in real
        tables; absorbing that row into the header loses it silently.
        """
        regions = [r for r in _grid_regions(n_body_rows=4) if r.text != "r1c1"]
        table = reconstruct_table(regions, self.RECT, 200, 0.96, pix_dims=self.DIMS)
        assert table is not None
        assert table.headers == ["H1", "H2", "H3", "H4"]
        assert table.rows[0] == ["", "r1c2", "r1c3", "r1c4"]
        assert len(table.rows) == 4

    def test_refuses_when_no_header_text_recovered(self) -> None:
        """A grid whose header band lands in no column has no usable headers."""
        regions = [
            r for r in _grid_regions(n_body_rows=4)
            if not r.text.startswith("H")
        ] + [_region(55, 100, 95, 130, "stray header")]
        assert reconstruct_table(regions, self.RECT, 200, 0.96, pix_dims=self.DIMS) is None

    def test_refuses_poor_column_fit(self) -> None:
        """Spans the column model can't place trip the assignment gate."""
        # Strays sit inside the rect but left of column 1 beyond tolerance.
        regions = _grid_regions() + [
            _region(55, 200, 95, 230, "stray1"),
            _region(55, 300, 95, 330, "stray2"),
        ]
        assert reconstruct_table(regions, self.RECT, 200, 0.96, pix_dims=self.DIMS) is None

    def test_refuses_empty_rect(self) -> None:
        regions = _grid_regions()
        # A rect nowhere near the regions holds no centroids.
        assert reconstruct_table(regions, (1400.0, 50.0, 1600.0, 500.0), 200, 0.96,
                                 pix_dims=self.DIMS) is None


# The layout rect enclosing every grid `_grid_regions` builds, and a narrative
# band well below it. A 612×792 pt page renders to 1700×2200 px at 200 dpi, so
# both sit inside the page.
_TABLE_RECT = (50.0, 50.0, 1300.0, 900.0)
_NARRATIVE_RECT = (50.0, 950.0, 1300.0, 1300.0)
_RECON_REGIONS = [
    LayoutRegionResult(bbox=_TABLE_RECT, label="Table", block_type="table", confidence=0.96),
    LayoutRegionResult(bbox=_NARRATIVE_RECT, label="Text", block_type="paragraph", confidence=0.90),
]
_RECON_LAYOUT = _layout_for(_RECON_REGIONS)


def _narrative_regions() -> list[OCRRegionResult]:
    return [
        _region(100, 1000, 600, 1040, "Narrative line one"),
        _region(100, 1100, 600, 1140, "Narrative line two"),
    ]


class TestLayoutPassReconstruction:
    """A3 — a detected table region becomes a TableData on the OCR-PDF path."""

    DIMS = (1700, 2200)

    def _run(self, page: Page, regions: list[OCRRegionResult]):
        return _layout_blocks_and_tables(
            page, 200, "whole page OCR text", 90.0,
            ocr_regions=regions, ocr_pix_dims=self.DIMS, page_layout=_RECON_LAYOUT,
        )

    def test_table_region_yields_cells(self, blank_page: Page) -> None:
        blocks, tables, consumed = self._run(
            blank_page, _grid_regions() + _narrative_regions(),
        )
        assert len(tables) == 1
        assert tables[0].headers == ["H1", "H2", "H3", "H4"]
        assert tables[0].rows[0] == ["r1c1", "r1c2", "r1c3", "r1c4"]
        assert tables[0].context["producer"] == "table_grid"
        # The 16 grid regions are absorbed; the narrative pair is not.
        assert len(consumed) == 16
        assert blocks and blocks[0].block_type == "paragraph"

    def test_narrative_is_rebuilt_from_the_complement(
        self, blank_page: Page,
    ) -> None:
        """The chunker must not see the table twice — as prose and as markdown."""
        blocks, tables, _consumed = self._run(
            blank_page, _grid_regions() + _narrative_regions(),
        )
        assert tables
        assert len(blocks) == 1
        assert blocks[0].text == "Narrative line one\nNarrative line two"
        assert "H1" not in blocks[0].text
        assert "r1c1" not in blocks[0].text

    def test_table_only_page_emits_no_narrative_block(
        self, blank_page: Page,
    ) -> None:
        blocks, tables, consumed = self._run(blank_page, _grid_regions())
        assert len(tables) == 1
        assert consumed
        assert blocks == []

    def test_refusal_keeps_todays_full_text_fallback(
        self, blank_page: Page,
    ) -> None:
        """Below the gates the page behaves exactly as it did before A3."""
        sparse = [_region(100, 100, 300, 140, "not a grid")] + _narrative_regions()
        blocks, tables, consumed = self._run(blank_page, sparse)
        assert tables == []
        assert consumed == []
        assert len(blocks) == 1
        assert blocks[0].text == "whole page OCR text"

    def test_reconstructor_failure_falls_back_to_the_whole_page(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        """A throw mid-loop must not leave tables emitted but text unsubtracted."""
        def _boom(*args, **kwargs):
            raise ValueError("grid inference blew up")

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned.reconstruct_table", _boom,
        )
        blocks, tables, consumed = self._run(
            blank_page, _grid_regions() + _narrative_regions(),
        )
        assert tables == []
        assert consumed == []
        assert len(blocks) == 1
        assert blocks[0].text == "whole page OCR text"

    def test_deskewed_page_refuses(self, blank_page: Page, caplog) -> None:
        """A2 — deskew rotated the OCR input out of the layout render's frame."""
        with caplog.at_level(logging.DEBUG, logger="womblex.ingest.strategies_scanned"):
            blocks, tables, consumed = _layout_blocks_and_tables(
                blank_page, 200, "whole page OCR text", 90.0,
                ocr_regions=_grid_regions() + _narrative_regions(),
                ocr_pix_dims=self.DIMS, page_layout=_RECON_LAYOUT,
                page_deskewed=True,
            )
        assert "deskewed page, refusing table reconstruction" in caplog.text
        assert tables == []
        assert consumed == []
        assert blocks[0].text == "whole page OCR text"


class TestOrchestratorTableWiring:
    """A3 — the orchestrator surfaces reconstructed tables and de-duplicates."""

    DIMS = (1700, 2200)

    def _apply(self, page: Page, monkeypatch, regions, steps=()):
        from womblex.ingest.detect import DocumentType
        from womblex.ingest.orchestrator import _apply_ocr_page, _PageAccum

        def _fake_ocr_page(page, dpi, lang, engine, engine_options):
            return ("whole page OCR text", 90.0, list(steps), False, regions, self.DIMS)

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned._ocr_page", _fake_ocr_page,
        )
        accum = _PageAccum(page_number=0)
        _apply_ocr_page(
            page, _ocr_profile(), accum,
            dpi=200, lang="eng", engine="paddleocr", engine_options={},
            doc_type=DocumentType.SCANNED_MACHINEWRITTEN, page_layout=_RECON_LAYOUT,
        )
        return accum

    def test_table_reaches_the_accumulator(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        accum = self._apply(
            blank_page, monkeypatch, _grid_regions() + _narrative_regions(),
        )
        assert len(accum.tables) == 1
        assert accum.tables[0].headers == ["H1", "H2", "H3", "H4"]

    def test_page_text_stays_verbatim(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        """Narrative subtraction is an element-stream concern.

        ``PageResult.text`` is what the page says, table included — it feeds
        text-coverage and the CER metrics, which compare against a transcript
        of the whole page.
        """
        accum = self._apply(
            blank_page, monkeypatch, _grid_regions() + _narrative_regions(),
        )
        assert accum.text == "whole page OCR text"

    def test_consumed_regions_do_not_become_form_fields(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        """A colon-bearing cell must not land in both a form and the table."""
        regions = _grid_regions() + _narrative_regions()
        # Give one body cell a label/value shape the form extractor would pair.
        pairish = next(r for r in regions if r.text == "r2c2")
        pairish.text = "Owner: Smith"
        accum = self._apply(blank_page, monkeypatch, regions)
        assert accum.tables
        assert not any(f.value == "Smith" for f in accum.forms)

    def test_deskew_step_is_forwarded_as_the_refusal_signal(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        """A2 — the orchestrator reads the refusal off ``_ocr_page``'s steps."""
        accum = self._apply(
            blank_page, monkeypatch,
            _grid_regions() + _narrative_regions(),
            steps=("deskew", "binarise"),
        )
        assert accum.tables == []


class TestReconstructedTableDownstream:
    """A5 — the conventions hold once a reconstructed table reaches the stream.

    A1 set the two provenance fields; A3 is the first path that actually puts
    one through ``_table_to_element``, so these claims stop being structural
    and start being observed.
    """

    def _elements(self, blank_page: Page, monkeypatch):
        from womblex.ingest.orchestrator import _accum_to_elements

        accum = TestOrchestratorTableWiring()._apply(
            blank_page, monkeypatch, _grid_regions() + _narrative_regions(),
        )
        elements, _next = _accum_to_elements(accum, 0, include_tables=True)
        return elements

    def test_projects_to_a_cellified_table_element(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        elements = self._elements(blank_page, monkeypatch)
        tables = [e for e in elements if e.kind == "table"]
        assert len(tables) == 1
        el = tables[0]
        assert el.header_rows == [0]
        assert [c.value for c in el.cells if c.row == 0] == ["H1", "H2", "H3", "H4"]
        # Lineage: distinguishable from a native-finder table in the parquet,
        # via the existing context_* → meta copy, with no schema change.
        assert el.meta["context_producer"] == "table_grid"

    def test_narrative_element_holds_no_table_text(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        elements = self._elements(blank_page, monkeypatch)
        paragraphs = [e for e in elements if e.kind == "paragraph"]
        assert len(paragraphs) == 1
        assert paragraphs[0].text == "Narrative line one\nNarrative line two"

    def test_chunker_sees_the_table_exactly_once(
        self, blank_page: Page, monkeypatch,
    ) -> None:
        from womblex.process.chunker import collect_tables_from_elements

        elements = self._elements(blank_page, monkeypatch)
        tables = collect_tables_from_elements(elements)
        assert len(tables) == 1
        _page, _elem_order, markdown = tables[0]
        assert "H1" in markdown
        assert markdown.count("r1c1") == 1


class _StubReader:
    """An OCR reader returning fixed regions, as a region-based engine would."""

    def __init__(self, regions: list[OCRRegionResult]) -> None:
        self._regions = regions

    def read_page(self, img) -> OCRPageResult:
        return OCRPageResult(regions=self._regions, confidence=0.9)


class TestImageDocumentsRouteThroughTheOrchestrator:
    """A4, as resolved: standalone images were never a separate path.

    ``extract_text`` sends every non-(SPREADSHEET|DOCX|TEXT) document —
    ``IMAGE`` included — to ``extract_pdf_with_plan``, because the PDF seam opens
    an image as a one-page document. The legacy ``ImageExtractor`` that A4
    was written to fix was unreachable, so it was deleted instead of wired
    up. These tests pin the routing, so a future change that reintroduces a
    bypass fails here rather than silently losing table reconstruction on
    every image input.
    """

    def _profile(self):
        from womblex.ingest.detect import DocumentProfile, DocumentType

        return DocumentProfile(
            doc_type=DocumentType.IMAGE,
            page_count=1, has_text_layer=False, text_coverage=0.0,
            has_images=True, has_tables=True, has_handwriting_signals=False,
            ocr_confidence=None, glyph_regularity=None,
            stroke_consistency=None, confidence=0.9,
        )

    def _png(self, tmp_path):
        """A 612×792 pt page rendered at 200 dpi — 1700×2200 px, matching the stub rects."""
        with PdfBuilder(tmp_path / "scan.pdf").page(612, 792).text(72, 100, "x").open() as doc:
            pixels = doc[0].render(dpi=200)
        out = tmp_path / "scan.png"
        Image.fromarray(pixels).save(out)
        return out

    def _extract(self, tmp_path, monkeypatch):
        from womblex.ingest.extract import extract_text

        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned.get_ocr_reader",
            lambda **kw: _StubReader(_grid_regions() + _narrative_regions()),
        )
        monkeypatch.setattr(
            "womblex.ingest.paddle_ocr.get_layout_analyzer",
            lambda *_a, **_k: _StubAnalyzer(_RECON_REGIONS),
        )
        # Deskew would refuse reconstruction (A2); this fixture has no skew,
        # so hold preprocessing to the identity and keep the test about routing.
        monkeypatch.setattr(
            "womblex.ingest.strategies_scanned.preprocess_for_ocr",
            lambda img: (img, []),
        )
        from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
        from womblex.ingest.layout_step import LayoutSettings

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
        )
        return extract_text(
            self._png(tmp_path), self._profile(), dpi=200,
            layout=LayoutSettings.from_config(config),
        )[0]

    def test_get_extractor_refuses_image(self) -> None:
        """The dead IMAGE case is gone — routing it here would be the bug."""
        from womblex.ingest.extract import get_extractor

        with pytest.raises(ValueError, match="SPREADSHEET/DOCX/TEXT"):
            get_extractor(self._profile())

    def test_image_document_yields_a_reconstructed_table(
        self, tmp_path, monkeypatch,
    ) -> None:
        result = self._extract(tmp_path, monkeypatch)
        tables = [e for e in result.elements if e.kind == "table"]
        assert len(tables) == 1
        el = tables[0]
        assert el.header_rows == [0]
        assert [c.value for c in el.cells if c.row == 0] == ["H1", "H2", "H3", "H4"]
        assert el.meta["context_producer"] == "table_grid"

    def test_image_document_narrative_holds_no_table_text(
        self, tmp_path, monkeypatch,
    ) -> None:
        result = self._extract(tmp_path, monkeypatch)
        paragraphs = [e for e in result.elements if e.kind == "paragraph"]
        assert len(paragraphs) == 1
        assert paragraphs[0].text == "Narrative line one\nNarrative line two"

"""Tests for womblex.redact — detection, masking, stage logic, and utils.

The RedactionDetector operates on numpy image arrays.  Positive-case
tests (detecting known black boxes) need controlled inputs with exact
geometry, so ``redacted_image`` is constructed inline.  Negative-case
tests (no spurious detections) use a synthetic scanned page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from tests._pdf_builders import PdfBuilder
from tests._synthetic import SCANS_DIR
from womblex.cli.redact import cmd_annotate_redactions, cmd_redact
from womblex.config import RedactionConfig
from womblex.ingest.elements import Element, ElementKind
from womblex.ingest.pdf.types import Rect
from womblex.redact import RedactionDetector, RedactionInfo
from womblex.redact.batch import (
    REDACTIONS_SCHEMA,
    annotate_redactions_for_shards,
    validate_redactions_against_labels,
)
from womblex.redact.stage import (
    RedactionReport,
    annotate_chunks,
    annotate_elements,
    annotate_extraction,
    apply_text_redaction,
    build_detector,
    detect_redactions,
)
from womblex.redact.utils import pre_ocr_mask

if TYPE_CHECKING:
    from womblex.process.chunker import TextChunk

@pytest.fixture
def redacted_image() -> np.ndarray:
    """RGB image with two black rectangles at known coordinates."""
    img = np.full((400, 600, 3), 240, dtype=np.uint8)
    img[80:120, 100:500] = 0   # wide bar
    img[200:260, 150:350] = 0  # narrower bar
    return img


@pytest.fixture
def clean_image() -> np.ndarray:
    """A synthetic scanned page (a heading and one line) as grayscale — no redaction boxes."""
    return np.array(Image.open(SCANS_DIR / "page-sparse.png").convert("L"))


# ---------------------------------------------------------------------------
# RedactionDetector.detect
# ---------------------------------------------------------------------------


class TestRedactionDetect:
    def setup_method(self) -> None:
        self.detector = RedactionDetector()

    def test_detects_black_rectangles(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        # We drew two black boxes in the fixture
        assert len(redactions) == 2

    def test_no_redactions_on_clean_image(self, clean_image: np.ndarray) -> None:
        redactions = self.detector.detect(clean_image, page=0)
        assert len(redactions) == 0

    def test_redaction_bboxes_are_reasonable(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            assert x1 < x2
            assert y1 < y2
            assert r.area_px > 0

    def test_page_number_is_set(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=7)
        for r in redactions:
            assert r.page == 7

    def test_grayscale_input(self) -> None:
        """Detector should handle grayscale images too."""
        gray = np.full((400, 600), 240, dtype=np.uint8)
        gray[100:150, 100:400] = 0  # black bar
        redactions = self.detector.detect(gray, page=0)
        assert len(redactions) >= 1

    def test_rejects_tiny_contours(self) -> None:
        """Very small dark spots should not count as redactions."""
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        # A single pixel is too small
        img[200, 300] = 0
        redactions = self.detector.detect(img, page=0)
        assert len(redactions) == 0

    def test_rejects_full_page_black(self) -> None:
        """An entirely black image should not count as a redaction."""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        redactions = self.detector.detect(img, page=0)
        # The contour covers > max_area_ratio so should be rejected
        assert len(redactions) == 0

    def test_custom_threshold(self) -> None:
        """Darker threshold should still detect very dark boxes."""
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        # Box with pixel value 30 (very dark)
        img[100:160, 100:400] = 30
        detector = RedactionDetector(threshold=40)
        redactions = detector.detect(img, page=0)
        assert len(redactions) >= 1

    def test_near_threshold_not_detected(self) -> None:
        """Boxes lighter than threshold should not be detected."""
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        # Box with value 80, which is above default threshold of 50
        img[100:160, 100:400] = 80
        detector = RedactionDetector(threshold=50)
        redactions = detector.detect(img, page=0)
        assert len(redactions) == 0

    def test_exclude_rects_drops_centred_candidates(self, redacted_image: np.ndarray) -> None:
        """A candidate whose centre falls inside an exclude rect is dropped."""
        # The wide bar in the fixture is at rows 80-120, cols 100-500 → centre (300, 100).
        # The narrower bar is at rows 200-260, cols 150-350 → centre (250, 230).
        # Exclude the wide-bar region; expect only the narrower bar to survive.
        exclude = [(50, 50, 550, 150)]  # covers wide bar's centre, not narrower bar's
        redactions = self.detector.detect(redacted_image, page=0, exclude_rects=exclude)
        assert len(redactions) == 1
        # Surviving bar should be the narrower one (y around 200-260)
        _x1, y1, _x2, _y2 = redactions[0].bbox
        assert 180 <= y1 <= 220

    def test_exclude_rects_none_is_noop(self, redacted_image: np.ndarray) -> None:
        """exclude_rects=None matches the un-filtered detect()."""
        baseline = self.detector.detect(redacted_image, page=0)
        with_none = self.detector.detect(redacted_image, page=0, exclude_rects=None)
        assert len(baseline) == len(with_none) == 2

    def test_exclude_rects_empty_is_noop(self, redacted_image: np.ndarray) -> None:
        """exclude_rects=[] matches the un-filtered detect()."""
        baseline = self.detector.detect(redacted_image, page=0)
        with_empty = self.detector.detect(redacted_image, page=0, exclude_rects=[])
        assert len(baseline) == len(with_empty) == 2


# ---------------------------------------------------------------------------
# RedactionDetector.mask
# ---------------------------------------------------------------------------


class TestRedactionMask:
    def setup_method(self) -> None:
        self.detector = RedactionDetector()

    def test_mask_replaces_with_white(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        masked = self.detector.mask(redacted_image, redactions)

        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            region = masked[y1:y2, x1:x2]
            # All pixels in the masked region should be white (255)
            assert np.all(region == 255)

    def test_mask_preserves_non_redacted_areas(self, redacted_image: np.ndarray) -> None:
        redactions = self.detector.detect(redacted_image, page=0)
        masked = self.detector.mask(redacted_image, redactions)

        # Non-redacted background should remain unchanged
        assert np.array_equal(masked[0, 0], redacted_image[0, 0])

    def test_mask_does_not_modify_original(self, redacted_image: np.ndarray) -> None:
        original = redacted_image.copy()
        redactions = self.detector.detect(redacted_image, page=0)
        self.detector.mask(redacted_image, redactions)

        # Original should be unchanged
        assert np.array_equal(redacted_image, original)

    def test_mask_with_no_redactions(self, clean_image: np.ndarray) -> None:
        masked = self.detector.mask(clean_image, [])
        assert np.array_equal(masked, clean_image)

    def test_mask_grayscale(self) -> None:
        gray = np.full((400, 600), 200, dtype=np.uint8)
        gray[100:150, 100:400] = 0

        redactions = [RedactionInfo(bbox=(100, 100, 400, 150), page=0, area_px=300 * 50)]
        detector = RedactionDetector()
        masked = detector.mask(gray, redactions)

        assert np.all(masked[100:150, 100:400] == 255)
        # Untouched area preserved
        assert masked[0, 0] == 200


# ---------------------------------------------------------------------------
# RedactionInfo dataclass
# ---------------------------------------------------------------------------


class TestRedactionInfo:
    def test_fields(self) -> None:
        r = RedactionInfo(bbox=(10, 20, 100, 50), page=3, area_px=2700)
        assert r.bbox == (10, 20, 100, 50)
        assert r.page == 3
        assert r.area_px == 2700


# ---------------------------------------------------------------------------
# Backwards-compatible import from ingest.redaction
# ---------------------------------------------------------------------------


class TestBackwardsCompatImport:
    def test_import_from_ingest_redaction(self) -> None:
        from womblex.ingest.redaction import RedactionDetector as LegacyDetector
        from womblex.ingest.redaction import RedactionInfo as LegacyInfo

        assert LegacyDetector is RedactionDetector
        assert LegacyInfo is RedactionInfo


# ---------------------------------------------------------------------------
# Stage logic: build_detector
# ---------------------------------------------------------------------------


class TestBuildDetector:
    def test_uses_config_values(self) -> None:
        cfg = RedactionConfig(threshold=30, min_area_ratio=0.01, max_area_ratio=0.5)
        detector = build_detector(cfg)
        assert detector.threshold == 30
        assert detector.min_area_ratio == 0.01
        assert detector.max_area_ratio == 0.5


# ---------------------------------------------------------------------------
# Utils: pre_ocr_mask
# ---------------------------------------------------------------------------


class TestPreOcrMask:
    def test_masks_redactions(self, redacted_image: np.ndarray) -> None:
        detector = RedactionDetector()
        masked, redactions = pre_ocr_mask(redacted_image, page=0, detector=detector)
        assert len(redactions) == 2
        for r in redactions:
            x1, y1, x2, y2 = r.bbox
            assert np.all(masked[y1:y2, x1:x2] == 255)

    def test_no_redactions_returns_original(self, clean_image: np.ndarray) -> None:
        detector = RedactionDetector()
        masked, redactions = pre_ocr_mask(clean_image, page=0, detector=detector)
        assert len(redactions) == 0
        assert np.array_equal(masked, clean_image)


# ---------------------------------------------------------------------------
# Stage logic: RedactionReport
# ---------------------------------------------------------------------------


class TestRedactionReport:
    def test_empty_report(self) -> None:
        report = RedactionReport()
        assert report.total == 0
        assert report.affected_pages == []

    def test_report_with_data(self) -> None:
        r1 = RedactionInfo(bbox=(0, 0, 100, 50), page=0, area_px=5000)
        r2 = RedactionInfo(bbox=(0, 0, 200, 30), page=2, area_px=6000)
        report = RedactionReport(page_redactions={0: [r1], 2: [r2]})
        assert report.total == 2
        assert report.affected_pages == [0, 2]


# ---------------------------------------------------------------------------
# Stage logic: apply_text_redaction
# ---------------------------------------------------------------------------


class TestApplyTextRedaction:
    def _make_pages(self, texts: list[str]) -> list:
        from womblex.ingest.extract import PageResult

        return [PageResult(page_number=i, text=t, method="test") for i, t in enumerate(texts)]

    def _make_report(self, *pages: int) -> RedactionReport:
        return RedactionReport(
            page_redactions={
                p: [RedactionInfo(bbox=(0, 0, 100, 50), page=p, area_px=5000)]
                for p in pages
            }
        )

    def test_flag_mode_no_change(self) -> None:
        pages = self._make_pages(["hello", "world"])
        report = self._make_report(0)
        result = apply_text_redaction(pages, report, mode="flag")
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_blackout_mode_prepends_marker(self) -> None:
        pages = self._make_pages(["sensitive text", "clean text"])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text.startswith("<REDACTED>")
        assert pages[1].text == "clean text"

    def test_blackout_mode_empty_page(self) -> None:
        pages = self._make_pages([""])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text == "<REDACTED>"

    def test_delete_mode_clears_page(self) -> None:
        pages = self._make_pages(["sensitive text", "clean text"])
        report = self._make_report(0)
        apply_text_redaction(pages, report, mode="delete")
        assert pages[0].text == ""
        assert pages[1].text == "clean text"

    def test_empty_report_no_change(self) -> None:
        pages = self._make_pages(["hello"])
        report = RedactionReport()
        apply_text_redaction(pages, report, mode="blackout")
        assert pages[0].text == "hello"


# ---------------------------------------------------------------------------
# Stage logic: annotate_chunks
# ---------------------------------------------------------------------------


class TestAnnotateChunks:
    def _make_chunk(self, index: int, **kwargs: object) -> TextChunk:
        from womblex.process.chunker import TextChunk

        chunk = TextChunk(
            text=f"chunk {index}",
            start_char=0,
            end_char=7,
            chunk_index=index,
        )
        for k, v in kwargs.items():
            setattr(chunk, k, v)
        return chunk

    def _make_report(self, pages: dict[int, int]) -> RedactionReport:
        """Build a report with *n* redactions per page."""
        page_redactions: dict[int, list[RedactionInfo]] = {}
        for page, count in pages.items():
            page_redactions[page] = [
                RedactionInfo(bbox=(0, 0, 100, 50), page=page, area_px=5000)
                for _ in range(count)
            ]
        return RedactionReport(page_redactions=page_redactions)

    def test_empty_report_returns_unchanged(self) -> None:
        chunks = [self._make_chunk(0), self._make_chunk(1)]
        report = RedactionReport()
        result = annotate_chunks(chunks, report)
        assert result is chunks
        assert chunks[0].has_redaction is False
        assert chunks[1].has_redaction is False

    def test_flags_chunk_with_source_pages(self) -> None:
        chunk = self._make_chunk(0, source_pages=[0, 1])
        report = self._make_report({1: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True

    def test_skips_chunk_with_unaffected_source_pages(self) -> None:
        chunk = self._make_chunk(0, source_pages=[2, 3])
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is False

    def test_flags_chunk_with_page_number(self) -> None:
        chunk = self._make_chunk(0, page_number=5)
        report = self._make_report({5: 2})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True

    def test_skips_chunk_with_unaffected_page_number(self) -> None:
        chunk = self._make_chunk(0, page_number=3)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is False

    def test_source_pages_takes_precedence_over_page_number(self) -> None:
        """When both attributes exist, source_pages is checked first."""
        chunk = self._make_chunk(0, source_pages=[0], page_number=5)
        report = self._make_report({5: 1})
        annotate_chunks([chunk], report)
        # source_pages=[0] doesn't overlap affected page 5, so no flag
        assert chunk.has_redaction is False

    def test_chunk_without_page_attrs_is_skipped(self) -> None:
        chunk = self._make_chunk(0)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is False

    def test_mixed_chunks(self) -> None:
        c0 = self._make_chunk(0, source_pages=[1])
        c1 = self._make_chunk(1, page_number=2)
        c2 = self._make_chunk(2, page_number=9)
        report = self._make_report({1: 1, 2: 1})
        annotate_chunks([c0, c1, c2], report)
        assert c0.has_redaction is True
        assert c1.has_redaction is True
        assert c2.has_redaction is False

    def test_empty_source_pages_falls_through(self) -> None:
        """source_pages=[] is falsy — should fall through to page_number."""
        chunk = self._make_chunk(0, source_pages=[], page_number=0)
        report = self._make_report({0: 1})
        annotate_chunks([chunk], report)
        assert chunk.has_redaction is True


# ---------------------------------------------------------------------------
# Stage logic: annotate_elements
# ---------------------------------------------------------------------------


class TestAnnotateElements:
    def _make_element(self, order: int, page: int | None, kind: ElementKind = "paragraph") -> Element:
        return Element(order=order, kind=kind, extractor="test", page=page)

    def _make_report(self, pages: dict[int, int]) -> RedactionReport:
        page_redactions: dict[int, list[RedactionInfo]] = {}
        for page, count in pages.items():
            page_redactions[page] = [
                RedactionInfo(bbox=(0, 0, 100, 50), page=page, area_px=5000)
                for _ in range(count)
            ]
        return RedactionReport(page_redactions=page_redactions)

    def test_empty_report_returns_unchanged(self) -> None:
        elements = [self._make_element(0, 0), self._make_element(1, 1)]
        result = annotate_elements(elements, RedactionReport())
        assert result is elements
        assert "has_redaction" not in elements[0].meta
        assert "has_redaction" not in elements[1].meta

    def test_flags_elements_on_affected_page(self) -> None:
        e = self._make_element(0, page=2)
        annotate_elements([e], self._make_report({2: 3}))
        assert e.meta["has_redaction"] == "true"

    def test_skips_elements_on_unaffected_page(self) -> None:
        e = self._make_element(0, page=1)
        annotate_elements([e], self._make_report({2: 1}))
        assert "has_redaction" not in e.meta

    def test_skips_elements_with_no_page(self) -> None:
        e = self._make_element(0, page=None)
        annotate_elements([e], self._make_report({0: 1}))
        assert "has_redaction" not in e.meta

    def test_mixed_elements(self) -> None:
        e0 = self._make_element(0, page=0)  # affected
        e1 = self._make_element(1, page=1)  # not affected
        e2 = self._make_element(2, page=2)  # affected (different page)
        report = self._make_report({0: 2, 2: 5})
        annotate_elements([e0, e1, e2], report)
        assert e0.meta["has_redaction"] == "true"
        assert "has_redaction" not in e1.meta
        assert e2.meta["has_redaction"] == "true"


# ---------------------------------------------------------------------------
# Stage logic: annotate_extraction
# ---------------------------------------------------------------------------


class TestAnnotateExtraction:
    def test_adds_warnings(self) -> None:
        from womblex.ingest.extract import ExtractionResult

        extraction = ExtractionResult(pages=[], method="test")
        r1 = RedactionInfo(bbox=(0, 0, 100, 50), page=0, area_px=5000)
        report = RedactionReport(page_redactions={0: [r1]})

        annotate_extraction(extraction, report)
        assert len(extraction.warnings) == 1
        assert "page 0" in extraction.warnings[0]
        assert "1 redacted region" in extraction.warnings[0]

    def test_no_warnings_for_empty_report(self) -> None:
        from womblex.ingest.extract import ExtractionResult

        extraction = ExtractionResult(pages=[], method="test")
        report = RedactionReport()

        annotate_extraction(extraction, report)
        assert len(extraction.warnings) == 0


# ---------------------------------------------------------------------------
# Vector-direct redaction detection (via page.drawings())
# ---------------------------------------------------------------------------


class TestVectorRedactionPath:
    """Vector path catches native-drawn black-fill rectangles without
    rasterising, and without the area threshold that filters small bars
    out of the raster contour detector."""

    def _build(
        self, tmp_path: Path, rect: Rect, colour: tuple[float, float, float] = (0, 0, 0),
    ) -> Path:
        return PdfBuilder(tmp_path / "vec.pdf").page().rect(rect, colour).save()  # A4

    def test_detects_native_vector_black_fill(self, tmp_path: Path) -> None:
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 300, 220))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 in report.page_redactions
        assert len(report.page_redactions[0]) >= 1

    def test_filters_thin_vector_lines(self, tmp_path: Path) -> None:
        """A 1pt-tall horizontal underline should not register (min-side filter)."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 300, 201))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions

    def test_filters_narrow_vertical_separators(self, tmp_path: Path) -> None:
        """A 0.4pt-wide vertical line (table column separator) should not register."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(150, 100, 150.4, 800))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions

    def test_filters_glyph_sized_fills(self, tmp_path: Path) -> None:
        """A 5pt × 6pt fill (body-glyph rendering) should not register."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 105, 206))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions

    def test_vector_bbox_in_pixel_coords(self, tmp_path: Path) -> None:
        """Vector path scales PDF coords (72dpi) to detection dpi for consistency."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 200, 230))
        config = RedactionConfig(dpi=150)
        report = detect_redactions(pdf_path, 1, build_detector(config), dpi=config.dpi)
        regions = report.page_redactions[0]
        assert len(regions) == 1
        x1, _y1, x2, _y2 = regions[0].bbox
        # 100pt * 150/72 ≈ 208; 200pt * 150/72 ≈ 416
        assert x1 == int(100 * 150 / 72)
        assert x2 == int(200 * 150 / 72)

    def test_ignores_non_black_fills(self, tmp_path: Path) -> None:
        """A light-grey filled rectangle should not register (header shading etc)."""
        from womblex.config import RedactionConfig
        pdf_path = self._build(tmp_path, Rect(100, 200, 300, 220), (0.8, 0.8, 0.8))
        report = detect_redactions(pdf_path, 1, build_detector(RedactionConfig()))
        assert 0 not in report.page_redactions


# ---------------------------------------------------------------------------
# Batch operations: annotate_redactions_for_shards + validate_redactions_against_labels
# ---------------------------------------------------------------------------


class TestRedactionBatch:
    def test_redactions_schema_columns(self) -> None:
        names = REDACTIONS_SCHEMA.names
        assert names == ["source_hash", "elem_order", "has_redaction"]

    def test_annotate_returns_empty_summary_for_empty_shard_dir(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)
        assert summary == {}

    def test_validate_returns_empty_for_empty_labels_dir(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        pdf_dir = tmp_path / "pdfs"
        labels_dir.mkdir()
        pdf_dir.mkdir()

        summaries = validate_redactions_against_labels(labels_dir, pdf_dir)
        assert summaries == []

    def test_annotate_writes_empty_sidecar_when_no_redactions(self, tmp_path: Path) -> None:
        """A batch whose source PDF has no detectable redactions should still produce
        a redactions.parquet (empty) so downstream consumers can rely on the file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        # Programmatic PDF with no redactions (a blank page)
        PdfBuilder(pdf_dir / "blank.pdf").page().save()

        # Minimal manifest + elements
        manifest_tbl = pa.table({"source_hash": ["h1"], "filename": ["blank.pdf"]})
        pq.write_table(manifest_tbl, shard_dir / "batch-0001._manifest.parquet")
        elements_tbl = pa.table({
            "source_hash": ["h1"],
            "elem_order": pa.array([0], type=pa.int32()),
            "page": pa.array([0], type=pa.int32()),
        })
        pq.write_table(elements_tbl, shard_dir / "batch-0001.elements.parquet")

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)

        out_path = shard_dir / "batch-0001.redactions.parquet"
        assert out_path.exists()
        sidecar = pq.read_table(out_path)
        assert sidecar.num_rows == 0
        assert sidecar.schema.names == ["source_hash", "elem_order", "has_redaction"]
        assert summary == {"h1": 0}

    def test_annotate_writes_rows_for_affected_elements(self, tmp_path: Path) -> None:
        """When a PDF has a detectable redaction and the manifest has elements on that
        page, the sidecar carries one row per affected element."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        # PDF with a fat black rectangle on page 0
        PdfBuilder(pdf_dir / "redacted.pdf").page(600, 400).rect(Rect(100, 100, 300, 140)).save()

        manifest_tbl = pa.table({"source_hash": ["h2"], "filename": ["redacted.pdf"]})
        pq.write_table(manifest_tbl, shard_dir / "batch-0002._manifest.parquet")
        elements_tbl = pa.table({
            "source_hash": ["h2", "h2", "h2"],
            "elem_order": pa.array([0, 1, 2], type=pa.int32()),
            "page": pa.array([0, 0, 1], type=pa.int32()),  # 2 on affected page, 1 not
        })
        pq.write_table(elements_tbl, shard_dir / "batch-0002.elements.parquet")

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)

        sidecar = pq.read_table(shard_dir / "batch-0002.redactions.parquet")
        assert sidecar.num_rows == 2  # elem_orders 0 and 1 are on page 0
        assert set(sidecar.column("elem_order").to_pylist()) == {0, 1}
        assert all(sidecar.column("has_redaction").to_pylist())
        assert summary["h2"] >= 1

    def test_annotate_resumes_from_checkpoint(self, tmp_path: Path) -> None:
        """A batch listed in the checkpoint should be skipped on the second run;
        new batches should still be processed."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()
        checkpoint_path = tmp_path / "redactions_checkpoint.json"

        # Build a single batch + matching blank PDF
        PdfBuilder(pdf_dir / "blank.pdf").page().save()
        pq.write_table(
            pa.table({"source_hash": ["h1"], "filename": ["blank.pdf"]}),
            shard_dir / "batch-0001._manifest.parquet",
        )
        pq.write_table(
            pa.table({
                "source_hash": ["h1"],
                "elem_order": pa.array([0], type=pa.int32()),
                "page": pa.array([0], type=pa.int32()),
            }),
            shard_dir / "batch-0001.elements.parquet",
        )

        # First run: processes the batch, writes checkpoint
        annotate_redactions_for_shards(shard_dir, pdf_dir, checkpoint_path=checkpoint_path)
        assert checkpoint_path.exists()
        state = json.loads(checkpoint_path.read_text())
        assert "batch-0001" in state["processed_batches"]

        # Delete the sidecar to detect re-processing; second run should skip
        (shard_dir / "batch-0001.redactions.parquet").unlink()
        annotate_redactions_for_shards(shard_dir, pdf_dir, checkpoint_path=checkpoint_path)
        assert not (shard_dir / "batch-0001.redactions.parquet").exists()

    def test_annotate_skips_missing_pdf(self, tmp_path: Path) -> None:
        """A manifest pointing to a non-existent PDF should be skipped gracefully."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_dir = tmp_path / "shards"
        pdf_dir = tmp_path / "pdfs"
        shard_dir.mkdir()
        pdf_dir.mkdir()

        manifest_tbl = pa.table({"source_hash": ["h3"], "filename": ["nonexistent.pdf"]})
        pq.write_table(manifest_tbl, shard_dir / "batch-0003._manifest.parquet")
        elements_tbl = pa.table({
            "source_hash": ["h3"],
            "elem_order": pa.array([0], type=pa.int32()),
            "page": pa.array([0], type=pa.int32()),
        })
        pq.write_table(elements_tbl, shard_dir / "batch-0003.elements.parquet")

        summary = annotate_redactions_for_shards(shard_dir, pdf_dir)
        # Missing PDF → no summary entry, no crash, empty sidecar still written
        assert "h3" not in summary
        assert (shard_dir / "batch-0003.redactions.parquet").exists()


# ---------------------------------------------------------------------------
# CLI surface: `womblex redact --shards` (I3) + `annotate-redactions` alias
# ---------------------------------------------------------------------------


def _seed_redaction_shards(tmp_path: Path) -> tuple[Path, Path]:
    """Build a one-batch shard dir + a PDF with a detectable redaction on page 0.

    Returns ``(shard_dir, pdf_dir)``. Mirrors the engine-level fixtures above.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir = tmp_path / "shards"
    pdf_dir = tmp_path / "pdfs"
    shard_dir.mkdir()
    pdf_dir.mkdir()

    PdfBuilder(pdf_dir / "redacted.pdf").page(600, 400).rect(Rect(100, 100, 300, 140)).save()

    pq.write_table(
        pa.table({"source_hash": ["h1"], "filename": ["redacted.pdf"]}),
        shard_dir / "batch-0001._manifest.parquet",
    )
    pq.write_table(
        pa.table({
            "source_hash": ["h1", "h1"],
            "elem_order": pa.array([0, 1], type=pa.int32()),
            "page": pa.array([0, 0], type=pa.int32()),
        }),
        shard_dir / "batch-0001.elements.parquet",
    )
    return shard_dir, pdf_dir


def _redact_shards_args(shard_dir: Path, pdf_dir: Path | None, **overrides) -> argparse.Namespace:
    base = {
        "shards": shard_dir, "config": None, "pdfs": pdf_dir, "output": None,
        "checkpoint": None, "dpi": 150, "max_area_ratio": 0.05, "limit": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCmdRedactShards:
    def test_writes_redactions_sidecar(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        shard_dir, pdf_dir = _seed_redaction_shards(tmp_path)
        assert cmd_redact(_redact_shards_args(shard_dir, pdf_dir)) == 0

        sidecar = shard_dir / "batch-0001.redactions.parquet"
        assert sidecar.exists()
        tbl = pq.read_table(sidecar)
        assert tbl.schema.names == ["source_hash", "elem_order", "has_redaction"]
        assert set(tbl.column("elem_order").to_pylist()) == {0, 1}

    def test_requires_pdfs_with_shards(self, tmp_path: Path) -> None:
        shard_dir, _ = _seed_redaction_shards(tmp_path)
        assert cmd_redact(_redact_shards_args(shard_dir, None)) == 1

    def test_rejects_shard_dir_without_manifests(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        pdfs = tmp_path / "pdfs"
        empty.mkdir()
        pdfs.mkdir()
        assert cmd_redact(_redact_shards_args(empty, pdfs)) == 1

    def test_rejects_nonexistent_shard_dir(self, tmp_path: Path) -> None:
        pdfs = tmp_path / "pdfs"
        pdfs.mkdir()
        assert cmd_redact(_redact_shards_args(tmp_path / "nope", pdfs)) == 1

    def test_honours_output_dir(self, tmp_path: Path) -> None:
        shard_dir, pdf_dir = _seed_redaction_shards(tmp_path)
        out = tmp_path / "sidecars"
        assert cmd_redact(_redact_shards_args(shard_dir, pdf_dir, output=out)) == 0
        assert (out / "batch-0001.redactions.parquet").exists()
        assert not (shard_dir / "batch-0001.redactions.parquet").exists()

    def test_dispatches_to_config_branch_when_shards_none(self, tmp_path: Path) -> None:
        # When --shards is absent, cmd_redact routes to the E2E --config branch
        # (load_config → FileNotFoundError for a missing path). The argparse-level
        # "exactly one of --shards/--config required" enforcement is verified
        # separately via the CLI smoke test, not reachable from a hand-built Namespace.
        args = _redact_shards_args(tmp_path / "x", tmp_path / "y")
        args.shards = None
        args.config = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            cmd_redact(args)


class TestAnnotateRedactionsAlias:
    def test_alias_produces_same_sidecar_as_redact_shards(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        shard_dir, pdf_dir = _seed_redaction_shards(tmp_path)
        alias_args = argparse.Namespace(
            shards=shard_dir, pdfs=pdf_dir, output=None,
            checkpoint=None, dpi=150, max_area_ratio=0.05,
        )
        assert cmd_annotate_redactions(alias_args) == 0

        sidecar = shard_dir / "batch-0001.redactions.parquet"
        assert sidecar.exists()
        tbl = pq.read_table(sidecar)
        assert set(tbl.column("elem_order").to_pylist()) == {0, 1}

    def test_alias_rejects_missing_dirs(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            shards=tmp_path / "nope", pdfs=tmp_path / "nope2", output=None,
            checkpoint=None, dpi=150, max_area_ratio=0.05,
        )
        assert cmd_annotate_redactions(args) == 1


class TestLayoutFilterOnRasterPages:
    """The filter reads the layout step's regions; it never runs an analyser."""

    @staticmethod
    def _scan(tmp_path: Path) -> Path:
        px = np.full((400, 600, 3), 240, dtype=np.uint8)
        px[80:120, 100:500] = 0  # one dark bar, no vector redaction anywhere
        return (
            PdfBuilder(tmp_path / "scan.pdf").page(600, 400)
            .image(Rect(0, 0, 600, 400), Image.fromarray(px)).save()
        )

    @staticmethod
    def _layout(status: str, block_type: str = "figure"):
        from womblex.ingest.elements import BBox
        from womblex.ingest.layout_step import LayoutRegion, PageLayout

        regions = [LayoutRegion(BBox(0.1, 0.1, 0.8, 0.3), "x", block_type, 0.9)] if status == "ok" else []
        return {0: PageLayout(0, status, regions, error="boom" if status == "error" else "")}

    def _detect(self, tmp_path: Path, layout):
        return detect_redactions(
            self._scan(tmp_path), 1, build_detector(RedactionConfig()), dpi=72, layout=layout,
        )

    def test_a_figure_region_suppresses_the_hit(self, tmp_path: Path) -> None:
        assert self._detect(tmp_path, self._layout("ok")).total == 0

    def test_a_paragraph_region_does_not(self, tmp_path: Path) -> None:
        assert self._detect(tmp_path, self._layout("ok", "paragraph")).total == 1

    @pytest.mark.parametrize("case", ["none", "no-row", "error"])
    def test_no_layout_runs_unfiltered_and_says_so(self, tmp_path: Path, case, caplog) -> None:
        layout = {"none": None, "no-row": {}, "error": self._layout("error")}[case]
        with caplog.at_level("WARNING", logger="womblex.redact.stage"):
            report = self._detect(tmp_path, layout)
        assert (report.total, report.unfiltered_pages) == (1, [0])
        assert caplog.text.count("doc=scan.pdf pages=[0]") == 1  # one warning per document

    def test_a_usable_page_is_not_unfiltered(self, tmp_path: Path) -> None:
        assert self._detect(tmp_path, self._layout("empty")).unfiltered_pages == []

    def test_shards_read_the_batch_layout_sidecar(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        from womblex.ingest.elements import BBox
        from womblex.ingest.layout_step import LayoutRegion, PageLayout
        from womblex.store.layout_output import (
            LayoutFingerprint,
            layout_rows,
            unfiltered_redaction_pages,
            write_layout_regions,
        )

        shards, pdfs = tmp_path / "shards", tmp_path / "pdfs"
        shards.mkdir()
        pdfs.mkdir()
        self._scan(tmp_path).rename(pdfs / "scan.pdf")
        pq.write_table(pa.table({"source_hash": ["h1"], "filename": ["scan.pdf"]}),
                       shards / "batch-0001._manifest.parquet")
        pq.write_table(pa.table({
            "source_hash": ["h1"], "elem_order": pa.array([0], type=pa.int32()),
            "page": pa.array([0], type=pa.int32()),
        }), shards / "batch-0001.elements.parquet")
        config = RedactionConfig(dpi=72)

        assert annotate_redactions_for_shards(shards, pdfs, config) == {"h1": 1}
        assert unfiltered_redaction_pages(shards) == [("h1", 0)]  # no sidecar, no layout

        fp = LayoutFingerprint("m", "d", "o", 72, "consumers")
        page = PageLayout(0, "ok", [LayoutRegion(BBox(0.1, 0.1, 0.8, 0.3), "x", "figure", 0.9)])
        write_layout_regions(
            layout_rows([("h1", [page])]), shards / "batch-0001.parquet", fp, redaction_consumed=True,
        )
        (shards / "batch-0001.redactions.parquet").unlink()
        assert annotate_redactions_for_shards(shards, pdfs, config) == {"h1": 0}
        assert unfiltered_redaction_pages(shards) == []

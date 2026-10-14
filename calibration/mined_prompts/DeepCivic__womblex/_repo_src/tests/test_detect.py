"""Tests for womblex.ingest.detect — document type detection.

Tests use controlled inputs for classification logic and real benchmark
fixtures for integration tests.
"""

from pathlib import Path

import pytest
from PIL import Image

from tests._pdf_builders import PdfBuilder
from tests._synthetic import SCANS_DIR
from womblex.config import DetectionConfig
from womblex.ingest.detect import (
    DocumentType,
    _classify,
    _has_table_structure,
    detect_document_type,
)
from womblex.ingest.pdf.types import Rect

# ---------------------------------------------------------------------------
# _has_table_structure
# ---------------------------------------------------------------------------


class TestHasTableStructure:
    def test_pipe_delimited_table(self) -> None:
        text = (
            "Name       | Date       | Status\n"
            "Alice      | 2024-01-01 | Active\n"
            "Bob        | 2024-01-02 | Inactive\n"
        )
        assert _has_table_structure(text) is True

    def test_tab_delimited_table(self) -> None:
        text = "Name\tDate\tStatus\nAlice\t2024-01-01\tActive\nBob\t2024-01-02\tInactive\n"
        assert _has_table_structure(text) is True

    def test_plain_prose_is_not_table(self) -> None:
        text = (
            "This is a plain paragraph with no table structure whatsoever. "
            "It contains only normal prose text."
        )
        assert _has_table_structure(text) is False

    def test_empty_string(self) -> None:
        assert _has_table_structure("") is False


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class TestClassify:
    """Test the classification function directly with controlled inputs."""

    def setup_method(self) -> None:
        self.config = DetectionConfig()

    def test_empty_document(self) -> None:
        profile = _classify(0, 0, 0, 0, None, None, None, 0, self.config)
        assert profile.doc_type == DocumentType.UNKNOWN
        assert profile.page_count == 0

    def test_all_text_no_structure(self) -> None:
        """Pure text, no tables or images → NATIVE_NARRATIVE."""
        profile = _classify(
            text_pages=10, image_pages=0, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=self.config,
        )
        assert profile.doc_type == DocumentType.NATIVE_NARRATIVE
        assert profile.has_text_layer is True
        assert profile.text_coverage == 1.0

    def test_text_with_tables(self) -> None:
        """Text with table signals → NATIVE_WITH_STRUCTURED."""
        profile = _classify(
            text_pages=10, image_pages=0, table_signals=5,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=self.config,
        )
        assert profile.doc_type == DocumentType.NATIVE_WITH_STRUCTURED

    def test_text_with_images(self) -> None:
        """Text with embedded images → NATIVE_WITH_STRUCTURED."""
        profile = _classify(
            text_pages=10, image_pages=5, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=self.config,
        )
        assert profile.doc_type == DocumentType.NATIVE_WITH_STRUCTURED

    def test_scanned_machinewritten_high_morphology(self) -> None:
        """No text layer, all images, high morphology regularity → SCANNED_MACHINEWRITTEN."""
        profile = _classify(
            text_pages=0, image_pages=5, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=0.75, stroke_consistency=0.70,
            total_pages=5, config=self.config,
        )
        assert profile.doc_type == DocumentType.SCANNED_MACHINEWRITTEN

    def test_scanned_machinewritten_high_ocr_fallback(self) -> None:
        """No text layer, no morphology, high OCR confidence → SCANNED_MACHINEWRITTEN."""
        profile = _classify(
            text_pages=0, image_pages=5, table_signals=0,
            handwriting_signals=0, ocr_confidence=85.0,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=5, config=self.config,
        )
        assert profile.doc_type == DocumentType.SCANNED_MACHINEWRITTEN

    def test_scanned_low_morphology_goes_handwritten(self) -> None:
        """No text layer, low morphology regularity → SCANNED_HANDWRITTEN."""
        profile = _classify(
            text_pages=0, image_pages=5, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=0.25, stroke_consistency=0.30,
            total_pages=5, config=self.config,
        )
        assert profile.doc_type == DocumentType.SCANNED_HANDWRITTEN

    def test_scanned_no_signals_defaults_machinewritten(self) -> None:
        """No text layer, no morphology, no OCR → default to SCANNED_MACHINEWRITTEN."""
        profile = _classify(
            text_pages=0, image_pages=5, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=5, config=self.config,
        )
        assert profile.doc_type == DocumentType.SCANNED_MACHINEWRITTEN

    def test_scanned_handwritten(self) -> None:
        """No text, all images, handwriting on most pages → SCANNED_HANDWRITTEN."""
        profile = _classify(
            text_pages=0, image_pages=10, table_signals=0,
            handwriting_signals=9, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=self.config,
        )
        assert profile.doc_type == DocumentType.SCANNED_HANDWRITTEN
        assert profile.has_handwriting_signals is True

    def test_scanned_mixed(self) -> None:
        """No text, images, some handwriting pages → SCANNED_MIXED."""
        profile = _classify(
            text_pages=0, image_pages=10, table_signals=0,
            handwriting_signals=3, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=self.config,
        )
        assert profile.doc_type == DocumentType.SCANNED_MIXED

    def test_hybrid_partial_text(self) -> None:
        """Some pages with text, some without, below threshold → HYBRID."""
        profile = _classify(
            text_pages=2, image_pages=8, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=self.config,
        )
        # 2/10 = 0.2 < 0.3 threshold, but > 0.1 lower bound → HYBRID
        assert profile.doc_type == DocumentType.HYBRID

    def test_confidence_is_between_zero_and_one(self) -> None:
        for text_p in range(0, 11, 3):
            profile = _classify(
                text_p, 10 - text_p, 0, 0, 85.0, 0.7, 0.7, 10, self.config
            )
            assert 0.0 <= profile.confidence <= 1.0

    def test_custom_thresholds(self) -> None:
        """With high min_text_coverage, partial text falls below threshold."""
        config = DetectionConfig(min_text_coverage=0.8)
        profile = _classify(
            text_pages=5, image_pages=5, table_signals=0,
            handwriting_signals=0, ocr_confidence=None,
            glyph_regularity=None, stroke_consistency=None,
            total_pages=10, config=config,
        )
        # 0.5 < 0.8 threshold, has text and images → HYBRID (mixed native + scanned)
        assert profile.doc_type == DocumentType.HYBRID


# ---------------------------------------------------------------------------
# detect_document_type (integration with real fixtures)
# ---------------------------------------------------------------------------


class TestDetectDocumentType:
    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        # A backend may raise its own error rather than the builtin; accept
        # either so the test survives a change of extractor without going blind.
        with pytest.raises((FileNotFoundError, RuntimeError)):
            detect_document_type(tmp_path / "nope.pdf")

    def test_returns_document_profile(self, tmp_path: Path) -> None:
        """A scanned form embedded in a PDF profiles as an image-only scan."""
        img = Image.open(SCANS_DIR / "form-koala-rescue-intake.png").convert("RGB")
        pdf = PdfBuilder(tmp_path / "form.pdf").page().image(Rect(0, 0, 595, 842), img).save()
        profile = detect_document_type(pdf)
        assert profile.page_count == 1 and not profile.has_text_layer

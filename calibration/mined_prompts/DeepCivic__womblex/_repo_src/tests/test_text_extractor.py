"""Tests for plain text file extraction."""

from pathlib import Path

from tests._synthetic import AUDIT_TRANSCRIPT
from womblex.ingest.detect import DetectionConfig, DocumentType, detect_file_type
from womblex.ingest.extract import extract_text

_TXT_FILE = AUDIT_TRANSCRIPT


class TestTextDetection:
    def test_txt_detected_as_text(self, tmp_path: Path) -> None:
        txt = tmp_path / "sample.txt"
        txt.write_text("Hello world", encoding="utf-8")
        profile = detect_file_type(txt, DetectionConfig())
        assert profile.doc_type == DocumentType.TEXT
        assert profile.confidence == 1.0
        assert profile.page_count == 1

    def test_non_txt_not_affected(self, tmp_path: Path) -> None:
        """Other extensions still route correctly."""
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        profile = detect_file_type(csv, DetectionConfig())
        assert profile.doc_type != DocumentType.TEXT


class TestTextExtraction:
    def test_simple_text_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Line one\nLine two\n", encoding="utf-8")
        profile = detect_file_type(txt)
        results = extract_text(txt, profile)

        assert len(results) == 1
        r = results[0]
        assert r.error is None
        assert r.method == "text"
        assert "Line one" in r.full_text
        assert "Line two" in r.full_text
        assert r.metadata.extraction_strategy == "text"
        assert r.metadata.confidence == 1.0

    def test_empty_text_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "empty.txt"
        txt.write_text("", encoding="utf-8")
        profile = detect_file_type(txt)
        results = extract_text(txt, profile)

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].full_text == ""

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        """Files with non-UTF-8 encoding fall back to latin-1."""
        txt = tmp_path / "latin.txt"
        txt.write_bytes("caf\xe9".encode("latin-1"))
        profile = detect_file_type(txt)
        results = extract_text(txt, profile)

        assert len(results) == 1
        assert "caf" in results[0].full_text

    def test_real_transcript_fixture(self) -> None:
        """Extract from the synthetic audit report transcript."""

        profile = detect_file_type(_TXT_FILE)
        assert profile.doc_type == DocumentType.TEXT

        results = extract_text(_TXT_FILE, profile)
        assert len(results) == 1
        r = results[0]
        assert r.error is None
        assert len(r.full_text) > 1000  # the transcript is several pages
        assert r.method == "text"

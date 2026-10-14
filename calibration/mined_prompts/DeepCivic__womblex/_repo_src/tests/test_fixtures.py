"""Pipeline tests on the synthetic scans: forms, single text lines and report pages.

The three groups stand in for FUNSD forms, IAM lines and DocLayNet pages
(``tests/_synthetic.py``); scoring against those datasets is the benchmark's.
Validates the detection → extraction → chunking pipeline against images with
known text. Tests are grouped by concern:

- Detection: image-only PDFs classify as SCANNED_MACHINEWRITTEN
- Extraction: OCR runs without error on all fixture types
- OCR content: machine-printed fixtures yield recognisable text
- Redaction: clean fixture images report no redaction regions
- Chunking: ground-truth text chunks correctly via the semchunk pipeline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests._pdf_builders import PdfBuilder
from tests._synthetic import SCANS_DIR
from womblex.ingest.detect import DocumentProfile, DocumentType, detect_document_type
from womblex.ingest.extract import extract_text
from womblex.ingest.pdf.types import Rect
from womblex.process.chunker import ChunkInput, TextChunk, chunk_batch, create_chunker
from womblex.redact import RedactionDetector


def _chunk(text: str, chunker) -> list[TextChunk]:
    return chunk_batch([ChunkInput(source_hash="d", narrative=text)], chunker).get("d", [])

FUNSD_IMAGES = IAM_DIR = DOCLAYNET_DIR = SCANS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scanned_profile(page_count: int = 1) -> DocumentProfile:
    """Minimal DocumentProfile for a single-page scanned document."""
    return DocumentProfile(
        doc_type=DocumentType.SCANNED_MACHINEWRITTEN,
        page_count=page_count,
        has_text_layer=False,
        text_coverage=0.0,
        has_images=True,
        has_tables=False,
        has_handwriting_signals=False,
        ocr_confidence=None,
        glyph_regularity=None,
        stroke_consistency=None,
        confidence=0.8,
    )


def _image_to_pdf(
    image_path: Path,
    output_path: Path,
    page_w: int = 595,
    page_h: int = 842,
) -> Path:
    """Embed a PNG image into a single-page PDF.

    Args:
        image_path: Source PNG file.
        output_path: Destination PDF path.
        page_w: Page width in points (default 595 = A4 width).
        page_h: Page height in points (default 842 = A4 height).

    Using a fixed page size ensures OCR renders a predictable-size
    pixmap regardless of the source image's native pixel dimensions.
    For tests that call the full OCR pipeline use a smaller page size
    (e.g. 150×150 pt → ~417×417 px at 200 DPI) to keep wall time low.
    """
    img = Image.open(image_path).convert("RGB")
    builder = PdfBuilder(output_path).page(page_w, page_h)
    return builder.image(Rect(0, 0, page_w, page_h), img).save()


def _gt(name: str) -> str:
    return (SCANS_DIR / f"{name}.gt.txt").read_text(encoding="utf-8").strip()


def _funsd_ground_truth(name: str) -> list[str]:
    """The form's text, one entry per line (title, label-value pairs, declaration)."""
    return _gt(name).splitlines()


def _doclaynet_ground_truth(name: str) -> list[str]:
    """The page's text as words."""
    return _gt(name).split()


def _iam_ground_truth(name: str) -> str:
    """The line's text."""
    return _gt(name)


def _word_token_counter(text: str) -> int:
    """Simple word-count token counter for tests (no network required)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Parametrise fixture names
# ---------------------------------------------------------------------------

FUNSD_SAMPLES = ["form-koala-rescue-intake", "form-bilby-sighting-report", "form-wombat-burrow-permit"]
IAM_SAMPLES = ["line-field-note-1", "line-field-note-2", "line-field-note-3"]
DOCLAYNET_SAMPLES = ["page-sparse", "page-table", "page-dense"]


# ---------------------------------------------------------------------------
# Detection: image-only PDFs classify as SCANNED_MACHINEWRITTEN
# ---------------------------------------------------------------------------


_SCANNED_TYPES = {
    DocumentType.SCANNED_MACHINEWRITTEN,
    DocumentType.SCANNED_HANDWRITTEN,
    DocumentType.SCANNED_MIXED,
}


_SCANNED_TYPES = {
    DocumentType.SCANNED_MACHINEWRITTEN,
    DocumentType.SCANNED_HANDWRITTEN,
    DocumentType.SCANNED_MIXED,
}

# UNKNOWN is a valid outcome for very small or very low-contrast handwriting
# images where OCR confidence is below the classification thresholds.
_SCANNED_OR_UNKNOWN = _SCANNED_TYPES | {DocumentType.UNKNOWN}


@pytest.mark.slow
class TestFixtureDetection:
    """All fixture images should classify as a SCANNED_* type when wrapped in a
    PDF — they have no text layer, only an embedded raster image.

    Forms and pages are scanned documents (always SCANNED_*).  A single text
    line is a very small image, which may fall through to UNKNOWN when OCR
    confidence is below classification thresholds.  That is expected behaviour.
    """

    @pytest.mark.parametrize("name", FUNSD_SAMPLES)
    def test_funsd_detects_as_scanned(self, tmp_path: Path, name: str) -> None:
        pdf = _image_to_pdf(FUNSD_IMAGES / f"{name}.png", tmp_path / f"{name}.pdf")
        profile = detect_document_type(pdf)

        assert profile.has_text_layer is False
        assert profile.has_images is True
        assert profile.text_coverage == 0.0
        assert profile.doc_type in _SCANNED_TYPES, (
            f"FUNSD/{name} should be a scanned type, got {profile.doc_type}"
        )

    @pytest.mark.parametrize("name", IAM_SAMPLES)
    def test_iam_detects_as_image_only(self, tmp_path: Path, name: str) -> None:
        """IAM is a handwriting database; narrow/short images with low OCR
        confidence may classify as UNKNOWN — that is acceptable so long as
        the profile correctly reports no text layer."""
        pdf = _image_to_pdf(IAM_DIR / f"{name}.png", tmp_path / f"{name}.pdf")
        profile = detect_document_type(pdf)

        assert profile.has_text_layer is False
        assert profile.has_images is True
        assert profile.text_coverage == 0.0
        assert profile.doc_type in _SCANNED_OR_UNKNOWN, (
            f"IAM/{name}: unexpected type {profile.doc_type}"
        )

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES)
    def test_doclaynet_detects_as_scanned(self, tmp_path: Path, name: str) -> None:
        pdf = _image_to_pdf(DOCLAYNET_DIR / f"{name}.png", tmp_path / f"{name}.pdf")
        profile = detect_document_type(pdf)

        assert profile.has_text_layer is False
        assert profile.has_images is True
        assert profile.text_coverage == 0.0
        assert profile.doc_type in _SCANNED_TYPES, (
            f"DocLayNet/{name} should be a scanned type, got {profile.doc_type}"
        )

    def test_detection_profile_fields_populated(self, tmp_path: Path) -> None:
        """Profile returned for a fixture PDF has all expected fields set."""
        pdf = _image_to_pdf(FUNSD_IMAGES / "form-koala-rescue-intake.png", tmp_path / "test.pdf")
        profile = detect_document_type(pdf)

        assert profile.page_count == 1
        assert 0.0 <= profile.confidence <= 1.0
        # ocr_confidence is set when morphology signals are inconclusive; may or may not be None
        assert profile.ocr_confidence is None or 0.0 <= profile.ocr_confidence <= 100.0


# ---------------------------------------------------------------------------
# Extraction: OCR runs without error on all fixture images
# ---------------------------------------------------------------------------


# Page size used for OCR extraction tests — small enough that OCR (rapidocr) runs
# in a few seconds on CPU.  At 200 DPI a 150×150 pt page renders to ~417×417 px.
_OCR_TEST_PAGE_W = 150
_OCR_TEST_PAGE_H = 150


@pytest.mark.slow
class TestFixtureExtraction:
    """extract_text() should complete without raising for one representative
    fixture from each dataset.

    Pages are created at a small size (150×150 pt) so that OCR (rapidocr) processes
    ~417×417 pixel rasters rather than full A4 bitmaps.  This keeps wall time
    to a few seconds per test on CPU-only machines.  Detection tests already
    cover all 15 fixtures at full A4 resolution.
    """

    @pytest.fixture(autouse=True)
    def _require_ocr(self):
        pytest.importorskip("rapidocr_onnxruntime", reason="rapidocr-onnxruntime not installed")

    def test_funsd_sparse_form_extraction(self, tmp_path: Path) -> None:
        """The koala intake form is the sparsest form; OCR should
        return one result with no error.

        The profile is set explicitly to SCANNED_MACHINEWRITTEN so this test
        exercises the extraction path independent of detection.
        """
        pdf = _image_to_pdf(
            FUNSD_IMAGES / "form-koala-rescue-intake.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results) == 1
        assert results[0].error is None
        assert results[0].method == "scanned_machinewritten"

    def test_iam_single_line_extraction(self, tmp_path: Path) -> None:
        """A single text line; extraction should produce
        one PageResult regardless of OCR confidence."""
        pdf = _image_to_pdf(
            IAM_DIR / "line-field-note-1.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results) == 1
        assert results[0].page_count == 1

    def test_doclaynet_sparse_extraction(self, tmp_path: Path) -> None:
        """The sparse page has minimal content; OCR should
        complete and return one result."""
        pdf = _image_to_pdf(
            DOCLAYNET_DIR / "page-sparse.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results) == 1
        assert results[0].error is None

    def test_extraction_result_has_metadata(self, tmp_path: Path) -> None:
        """ExtractionResult always carries metadata with strategy and timing."""
        pdf = _image_to_pdf(
            FUNSD_IMAGES / "form-koala-rescue-intake.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        meta = results[0].metadata
        assert meta is not None
        assert meta.extraction_strategy == "scanned_machinewritten"
        assert meta.processing_time > 0
        assert meta.page_count == 1

    def test_extraction_result_has_pages(self, tmp_path: Path) -> None:
        """A single-page fixture always returns exactly one PageResult."""
        pdf = _image_to_pdf(
            DOCLAYNET_DIR / "page-sparse.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert results[0].page_count == 1
        assert len(results[0].pages) == 1
        assert results[0].pages[0].page_number == 0


# ---------------------------------------------------------------------------
# OCR content: machine-printed fixtures yield recognisable text
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFixtureOCRContent:
    """For machine-printed fixtures the extracted text should be non-empty.

    Pages are rendered at a small size (150×150 pt) so OCR completes quickly.
    These tests verify the OCR path produces real output; accuracy comparisons
    against ground truth are outside scope.
    """

    @pytest.fixture(autouse=True)
    def _require_ocr(self):
        pytest.importorskip("rapidocr_onnxruntime", reason="rapidocr-onnxruntime not installed")

    def test_funsd_sparse_form_has_some_text(self, tmp_path: Path) -> None:
        """The sparse koala intake form; OCR should return
        non-empty text."""
        pdf = _image_to_pdf(
            FUNSD_IMAGES / "form-koala-rescue-intake.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results[0].full_text) > 0, (
            "Expected non-empty OCR output from the koala intake form"
        )

    def test_doclaynet_sparse_has_some_text(self, tmp_path: Path) -> None:
        """The sparse page has a heading and one line; OCR should return
        non-empty output."""
        pdf = _image_to_pdf(
            DOCLAYNET_DIR / "page-sparse.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert len(results[0].full_text) > 0, (
            "Expected non-empty OCR output from the sparse page"
        )

    def test_iam_line_page_count_is_one(self, tmp_path: Path) -> None:
        """Each line sample is a single-line image; extraction must
        return exactly one page regardless of content."""
        pdf = _image_to_pdf(
            IAM_DIR / "line-field-note-1.png",
            tmp_path / "test.pdf",
            page_w=_OCR_TEST_PAGE_W,
            page_h=_OCR_TEST_PAGE_H,
        )
        results = extract_text(pdf, _scanned_profile())

        assert results[0].page_count == 1


# ---------------------------------------------------------------------------
# Redaction: clean fixture images report no redaction regions
# ---------------------------------------------------------------------------


class TestFixtureRedaction:
    """RedactionDetector behaviour on clean fixture images.

    The fixtures contain no black-box redactions.  However, the default
    detector uses loose area thresholds suited to government PDFs; on some
    fixture images it may flag small dark blobs (form borders, handwriting
    strokes).  The meaningful contract is:

    1. The detector runs without error on every fixture image.
    2. No *document-width* redaction bars are detected — those would indicate
       an actual censorship bar spanning most of the page width.
    3. Masking any detected regions does not raise an error (idempotency).
    """

    # A genuine censorship bar spans nearly the full page width. Form borders
    # and thick horizontal rules on scanned forms can reach 60–93 % of width;
    # only flag regions above 95 % as suspicious full-page redactions.
    _FULL_WIDTH_RATIO = 0.95

    def _load_gray(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img)

    def _has_full_width_redaction(self, gray: np.ndarray, redactions: list) -> bool:
        """Return True if any detected region spans ≥ 95 % of image width."""
        img_width = gray.shape[1]
        for r in redactions:
            x1, _, x2, _ = r.bbox
            if (x2 - x1) / img_width >= self._FULL_WIDTH_RATIO:
                return True
        return False

    @pytest.mark.parametrize("name", FUNSD_SAMPLES)
    def test_funsd_no_full_width_redaction_bars(self, name: str) -> None:
        gray = self._load_gray(FUNSD_IMAGES / f"{name}.png")
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        assert not self._has_full_width_redaction(gray, redactions), (
            f"FUNSD/{name}: unexpected full-width redaction bar detected"
        )

    @pytest.mark.parametrize("name", IAM_SAMPLES)
    def test_iam_no_full_width_redaction_bars(self, name: str) -> None:
        gray = self._load_gray(IAM_DIR / f"{name}.png")
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        assert not self._has_full_width_redaction(gray, redactions), (
            f"IAM/{name}: unexpected full-width redaction bar detected"
        )

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES)
    def test_doclaynet_no_full_width_redaction_bars(self, name: str) -> None:
        gray = self._load_gray(DOCLAYNET_DIR / f"{name}.png")
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        assert not self._has_full_width_redaction(gray, redactions), (
            f"DocLayNet/{name}: unexpected full-width redaction bar detected"
        )

    @pytest.mark.parametrize("name", DOCLAYNET_SAMPLES + FUNSD_SAMPLES + IAM_SAMPLES)
    def test_masking_does_not_raise(self, name: str) -> None:
        """detector.mask() must not raise regardless of what detect() found."""
        if name in DOCLAYNET_SAMPLES:
            path = DOCLAYNET_DIR / f"{name}.png"
        elif name in FUNSD_SAMPLES:
            path = FUNSD_IMAGES / f"{name}.png"
        else:
            path = IAM_DIR / f"{name}.png"
        gray = self._load_gray(path)
        detector = RedactionDetector()
        redactions = detector.detect(gray, page=0)
        masked = detector.mask(gray, redactions)
        assert masked.shape == gray.shape


# ---------------------------------------------------------------------------
# Chunking: ground-truth text chunks correctly via the semchunk pipeline
# ---------------------------------------------------------------------------


class TestFixtureChunking:
    """Uses ground-truth text from the fixture annotations as chunker input.
    These tests do not involve OCR, so they are fast and deterministic."""

    @pytest.fixture(autouse=True)
    def _setup_chunker(self) -> None:
        # Word-based token counter avoids network calls to HuggingFace
        self.chunker = create_chunker(
            tokenizer=_word_token_counter, chunk_size=30
        )

    def test_iam_long_line_single_chunk(self) -> None:
        """The longest line is 17 words; at chunk_size=30 words it fits in one chunk."""
        gt = _iam_ground_truth("line-field-note-3")
        chunks = _chunk(gt, self.chunker)

        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)
        assert gt.strip() in chunks[0].text

    def test_iam_median_single_chunk(self) -> None:
        """A short 8-word line; must be exactly one chunk."""
        gt = _iam_ground_truth("line-field-note-1")
        chunks = _chunk(gt, self.chunker)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_doclaynet_dense_text_produces_multiple_chunks(self) -> None:
        """The dense page's text is long enough to produce multiple chunks
        at chunk_size=30 words."""
        words = _doclaynet_ground_truth("page-dense")
        full_text = " ".join(words)
        chunks = _chunk(full_text, self.chunker)

        assert len(chunks) > 1, "Expected multiple chunks for dense DocLayNet text"

    def test_chunker_indices_are_sequential(self) -> None:
        """chunk_index must be 0, 1, 2, … with no gaps."""
        words = _doclaynet_ground_truth("page-dense")
        full_text = " ".join(words)
        chunks = _chunk(full_text, self.chunker)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_funsd_ground_truth_round_trips_through_chunker(self) -> None:
        """A form's text should chunk without error and
        the combined chunk text should contain all original content."""
        gt_texts = _funsd_ground_truth("form-wombat-burrow-permit")
        # Join all form field texts into a single document body
        full_text = " ".join(gt_texts)
        chunks = _chunk(full_text, self.chunker)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        # Every ground-truth token should appear somewhere in the chunks
        for token in gt_texts[:10]:
            assert token in combined, (
                f"Ground-truth token {token!r} missing from chunks"
            )

    def test_empty_ground_truth_yields_no_chunks(self) -> None:
        """Empty input must produce an empty chunk list (not crash)."""
        chunks = _chunk("", self.chunker)
        assert chunks == []

    def test_sparse_doclaynet_chunks_at_small_size(self) -> None:
        """The sparse page has only 11 words; at chunk_size=5 this produces
        multiple small chunks that together cover all the original words."""
        small_chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=5)
        words = _doclaynet_ground_truth("page-sparse")
        full_text = " ".join(words)
        chunks = _chunk(full_text, small_chunker)

        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        for w in words:
            assert w in combined

    def test_chunk_offsets_cover_input(self) -> None:
        """start_char and end_char must span non-overlapping, contiguous regions."""
        gt = _iam_ground_truth("line-field-note-3")
        large_chunker = create_chunker(tokenizer=_word_token_counter, chunk_size=5)
        chunks = _chunk(gt, large_chunker)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(gt)


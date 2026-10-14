"""Tests for womblex.pii — PII detection, cleaning, stage logic, and config.


No mocks — all tests exercise real code paths. Tests that need context

validation use high-confidence honorific matches (which bypass the

Sentence Transformers model) or explicitly accept model loading time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from womblex.config import PIIConfig, WomblexConfig
from womblex.pii.cleaner import _COMMON_WORDS, _HONORIFIC_RE, _TITLE_CASE_RE, PIICleaner
from womblex.pii.stage import clean_chunks, clean_extraction

# ---------------------------------------------------------------------------

# Helpers — lightweight stand-ins for extraction/chunk types

# ---------------------------------------------------------------------------



@dataclass

class _Page:

    page_number: int

    text: str

    method: str = "native"



@dataclass

class _Extraction:

    pages: list[_Page] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)


    @property

    def full_text(self) -> str:

        return "\n\n".join(p.text for p in self.pages if p.text)



@dataclass

class _Chunk:

    text: str

    start_char: int = 0

    end_char: int = 0

    chunk_index: int = 0

    content_type: str = "narrative"



# ---------------------------------------------------------------------------

# Regex pattern tests

# ---------------------------------------------------------------------------



class TestHonorificsPattern:

    def test_mr_dot_full_name(self) -> None:

        assert _HONORIFIC_RE.search("Mr. John Smith attended the meeting")


    def test_mrs_no_dot(self) -> None:

        assert _HONORIFIC_RE.search("Mrs Brown lodged the complaint")


    def test_ms_dot(self) -> None:

        assert _HONORIFIC_RE.search("Ms. Jane Doe was interviewed")


    def test_dr_prefix(self) -> None:

        assert _HONORIFIC_RE.search("Dr. Williams reviewed the case")


    def test_no_match_lowercase_name(self) -> None:

        assert not _HONORIFIC_RE.search("Mr. john smith")


    def test_no_match_bare_title(self) -> None:

        assert not _HONORIFIC_RE.search("The Minister said")



class TestTitleCasePattern:

    def test_two_caps_words(self) -> None:

        assert _TITLE_CASE_RE.search("John Smith was present")


    def test_three_caps_words(self) -> None:

        assert _TITLE_CASE_RE.search("Mary Jane Watson confirmed")


    def test_no_match_single_cap_word(self) -> None:

        assert not _TITLE_CASE_RE.search("The worker attended")


    def test_no_match_all_lower(self) -> None:

        assert not _TITLE_CASE_RE.search("the worker attended")



class TestCommonWords:

    def test_months_excluded(self) -> None:

        assert "January" in _COMMON_WORDS

        assert "December" in _COMMON_WORDS


    def test_states_excluded(self) -> None:

        assert "Victoria" in _COMMON_WORDS

        assert "Queensland" in _COMMON_WORDS


    def test_dept_names_excluded(self) -> None:

        assert "Department" in _COMMON_WORDS

        assert "Government" in _COMMON_WORDS



# ---------------------------------------------------------------------------

# PIICleaner — high-confidence matches (no model loading needed)

# ---------------------------------------------------------------------------



class TestPIICleanerHighConfidence:

    """Honorific-based detections don't require context scoring."""


    def test_replaces_honorific_name(self) -> None:

        cleaner = PIICleaner(context_similarity_threshold=0.5)

        text = "Mr. John Smith attended the hearing."

        cleaned, count = cleaner.clean(text)

        assert count == 1

        assert "<PERSON>" in cleaned

        assert "John Smith" not in cleaned


    def test_replaces_multiple_honorifics(self) -> None:

        cleaner = PIICleaner(context_similarity_threshold=0.5)

        text = "Dr. Jane Doe and Mrs. Alice Brown were present."

        cleaned, count = cleaner.clean(text)

        assert count == 2

        assert cleaned.count("<PERSON>") == 2


    def test_empty_text_returns_unchanged(self) -> None:

        cleaner = PIICleaner()

        cleaned, count = cleaner.clean("")
        assert cleaned == ""

        assert count == 0


    def test_text_without_pii_unchanged(self) -> None:

        cleaner = PIICleaner()

        text = "The report was submitted on time."

        cleaned, count = cleaner.clean(text)

        assert count == 0

        assert cleaned == text


    def test_replacement_preserves_surrounding_text(self) -> None:

        cleaner = PIICleaner()

        text = "On 1 January, Ms. Helen Ward signed the form."

        cleaned, _count = cleaner.clean(text)

        assert "On 1 January," in cleaned

        assert "signed the form." in cleaned


    def test_overlapping_spans_kept_once(self) -> None:

        """If a title-case match overlaps a high-confidence span, keep only one."""

        cleaner = PIICleaner(context_similarity_threshold=0.0)

        text = "Dr. Sarah Connor attended."

        _cleaned, count = cleaner.clean(text)

        assert count == 1



# ---------------------------------------------------------------------------

# PIICleaner — common-word exclusion

# ---------------------------------------------------------------------------



class TestPIICleanerCommonWordExclusion:

    def test_all_common_words_not_replaced(self) -> None:

        """Title-case sequences of only common words produce no replacements."""

        cleaner = PIICleaner(context_similarity_threshold=0.0)

        text = "The incident occurred in New South Wales last Monday."

        cleaned, _count = cleaner.clean(text)

        # "New South Wales" should not be replaced — all common words

        assert "New South Wales" in cleaned


    def test_department_terms_not_replaced(self) -> None:

        cleaner = PIICleaner(context_similarity_threshold=0.0)

        text = "The Australian Government Department issued the report."

        cleaned, _count = cleaner.clean(text)

        assert "Australian Government Department" in cleaned



# ---------------------------------------------------------------------------

# PIICleaner — context validation (uses real text patterns)

# ---------------------------------------------------------------------------



class TestPIICleanerContextValidation:

    """Tests that validate threshold behaviour using honorific patterns.


    Honorific matches have score 0.9, so threshold behaviour can be tested

    by varying the threshold and checking whether the match passes.
    """


    def test_high_threshold_still_accepts_honorific(self) -> None:

        """Honorific matches (score 0.9) always pass any threshold <= 0.9."""

        cleaner = PIICleaner(context_similarity_threshold=0.9)

        text = "Mr. James Wilson attended the interview."

        cleaned, count = cleaner.clean(text)

        assert count == 1

        assert "<PERSON>" in cleaned


    def test_text_without_names_produces_no_candidates(self) -> None:

        cleaner = PIICleaner(context_similarity_threshold=0.0)

        text = "The report was filed on time and processed correctly."

        cleaned, count = cleaner.clean(text)

        assert count == 0

        assert cleaned == text



# ---------------------------------------------------------------------------

# Stage — clean_extraction

# ---------------------------------------------------------------------------



class TestCleanExtraction:

    def test_replaces_page_text(self) -> None:

        extraction = _Extraction(

            pages=[_Page(0, "Mr. John Smith attended."), _Page(1, "No names here.")]

        )

        cleaner = PIICleaner(context_similarity_threshold=0.5)

        count = clean_extraction(extraction, cleaner)  # type: ignore[arg-type]

        assert count == 1

        assert "<PERSON>" in extraction.pages[0].text

        assert extraction.pages[1].text == "No names here."


    def test_empty_pages_skipped(self) -> None:

        extraction = _Extraction(pages=[_Page(0, "")])

        cleaner = PIICleaner()

        count = clean_extraction(extraction, cleaner)  # type: ignore[arg-type]

        assert count == 0


    def test_multiple_pages_with_names(self) -> None:

        extraction = _Extraction(

            pages=[

                _Page(0, "Mr. John Smith attended."),

                _Page(1, "Dr. Jane Doe signed."),

                _Page(2, "No names here."),

            ]

        )

        cleaner = PIICleaner(context_similarity_threshold=0.5)

        count = clean_extraction(extraction, cleaner)  # type: ignore[arg-type]

        assert count == 2



# ---------------------------------------------------------------------------

# Stage — clean_chunks

# ---------------------------------------------------------------------------



class TestCleanChunks:

    def test_replaces_chunk_text(self) -> None:

        chunks = [_Chunk("Mr. John Smith attended."), _Chunk("No names here.")]

        cleaner = PIICleaner(context_similarity_threshold=0.5)

        count = clean_chunks(chunks, cleaner)  # type: ignore[arg-type]

        assert count == 1

        assert "<PERSON>" in chunks[0].text

        assert chunks[1].text == "No names here."


    def test_empty_chunks_list(self) -> None:

        cleaner = PIICleaner()

        count = clean_chunks([], cleaner)

        assert count == 0


    def test_empty_chunk_text_skipped(self) -> None:

        chunks = [_Chunk("")]

        cleaner = PIICleaner()

        count = clean_chunks(chunks, cleaner)  # type: ignore[arg-type]

        assert count == 0



# ---------------------------------------------------------------------------

# Config

# ---------------------------------------------------------------------------



class TestPIIConfig:

    def test_defaults(self) -> None:

        cfg = PIIConfig()

        assert cfg.enabled is False

        assert cfg.entities == ["PERSON"]

        assert cfg.person_types == ["natural"]

        assert cfg.pipeline_point == "post_chunk"

        assert cfg.context_similarity_threshold == 0.35

        assert cfg.model == "all-MiniLM-L6-v2"


    def test_custom_values(self) -> None:

        cfg = PIIConfig(

            enabled=True,

            entities=["PERSON"],

            pipeline_point="post_extraction",

            context_similarity_threshold=0.7,

            model="all-MiniLM-L6-v2",

        )

        assert cfg.enabled is True

        assert cfg.pipeline_point == "post_extraction"

        assert cfg.context_similarity_threshold == 0.7


    def test_threshold_clamped(self) -> None:

        with pytest.raises(ValidationError):

            PIIConfig(context_similarity_threshold=1.5)


    def test_pipeline_config_has_pii(self) -> None:

        cfg = WomblexConfig(

            dataset={"name": "test"},

            paths={"input_root": "/tmp", "output_root": "/tmp", "checkpoint_dir": "/tmp"},

        )

        assert hasattr(cfg, "pii")

        assert isinstance(cfg.pii, PIIConfig)

        assert cfg.pii.enabled is False



# ---------------------------------------------------------------------------

# run_pii_cleaning integration

# ---------------------------------------------------------------------------



class TestRunPIICleaning:

    def _make_config(self, pipeline_point: str = "post_chunk") -> WomblexConfig:

        cfg = WomblexConfig(

            dataset={"name": "test"},

            paths={"input_root": "/tmp", "output_root": "/tmp", "checkpoint_dir": "/tmp"},

        )

        cfg.pii.enabled = True

        cfg.pii.pipeline_point = pipeline_point

        return cfg


    def test_disabled_returns_unchanged(self) -> None:

        from womblex.operations import DocumentResult, run_pii_cleaning


        cfg = WomblexConfig(

            dataset={"name": "test"},

            paths={"input_root": "/tmp", "output_root": "/tmp", "checkpoint_dir": "/tmp"},

        )

        cfg.pii.enabled = False

        results = [DocumentResult(path=Path("/tmp/a.pdf"), doc_id="a")]

        out = run_pii_cleaning(results, cfg)
        assert out is results


    def test_post_chunk_cleans_chunks(self) -> None:

        from womblex.operations import DocumentResult, run_pii_cleaning


        cfg = self._make_config("post_chunk")

        extraction = _Extraction(pages=[_Page(0, "text")])


        dr = DocumentResult(path=Path("/tmp/a.pdf"), doc_id="a", status="completed")

        dr.extraction = extraction  # type: ignore[assignment]

        dr.chunks = [_Chunk("Mr. John Smith attended.")]


        run_pii_cleaning([dr], cfg)

        assert "<PERSON>" in dr.chunks[0].text


    def test_post_extraction_cleans_pages(self) -> None:

        from womblex.operations import DocumentResult, run_pii_cleaning


        cfg = self._make_config("post_extraction")

        extraction = _Extraction(pages=[_Page(0, "Mr. John Smith attended.")])


        dr = DocumentResult(path=Path("/tmp/a.pdf"), doc_id="a", status="completed")

        dr.extraction = extraction  # type: ignore[assignment]


        run_pii_cleaning([dr], cfg)

        assert "<PERSON>" in extraction.pages[0].text


    def test_error_docs_skipped(self) -> None:

        from womblex.operations import DocumentResult, run_pii_cleaning


        cfg = self._make_config()

        dr = DocumentResult(path=Path("/tmp/a.pdf"), doc_id="a", status="error")

        out = run_pii_cleaning([dr], cfg)

        assert len(out) == 1

    def test_post_enrichment_without_enrichment_raises(self) -> None:

        from womblex.operations import DocumentResult, PreconditionError, run_pii_cleaning


        cfg = self._make_config("post_enrichment")

        dr = DocumentResult(path=Path("/tmp/a.pdf"), doc_id="a", status="completed")

        dr.extraction = _Extraction(pages=[_Page(0, "text")])  # type: ignore[assignment]

        dr.chunks = [_Chunk("Mr. John Smith attended.")]


        # Graph-driven PII requested but enrichment never ran → misuse, not a no-op.
        with pytest.raises(PreconditionError, match="post_enrichment"):

            run_pii_cleaning([dr], cfg)

    # post_enrichment pipeline routing tests are in test_pii_enrichment.py


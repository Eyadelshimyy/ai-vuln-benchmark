"""Tests for enrichment-enhanced PII cleaning — graph entities, person_types, ADDRESS.

Uses real dataclasses from analyse.models and real PIICleaner instances.
No mocks — tests exercise actual code paths.
"""

from __future__ import annotations

from womblex.analyse.models import Location, Person, Span
from womblex.config import PIIConfig
from womblex.pii.cleaner import _ADDRESS_RE, PIICleaner
from womblex.pii.stage import _extract_known_spans, clean_enriched_chunks
from womblex.process.chunker import TextChunk

# ---------------------------------------------------------------------------
# Helpers — minimal enrichment result with only the fields PII stage reads
# ---------------------------------------------------------------------------


class _EnrichmentResult:
    """Lightweight enrichment result for PII tests.

    Only carries the fields that ``_extract_known_spans`` accesses.
    """

    def __init__(
        self,
        text: str,
        persons: list[Person] | None = None,
        locations: list[Location] | None = None,
    ) -> None:
        self.text = text
        self.persons = persons or []
        self.locations = locations or []
        self.emails: list = []
        self.phone_numbers: list = []
        self.id_numbers: list = []


# ---------------------------------------------------------------------------
# Enrichment candidates — _enrichment_candidates static method
# ---------------------------------------------------------------------------


class TestEnrichmentCandidates:
    def test_span_within_text(self) -> None:
        candidates = PIICleaner._enrichment_candidates(
            text_len=30, text_offset=0,
            known_spans=[(5, 15, "PERSON")],
        )
        assert len(candidates) == 1
        assert candidates[0].start == 5
        assert candidates[0].end == 15
        assert candidates[0].entity_type == "PERSON"
        assert candidates[0].score == 0.95

    def test_span_outside_text_discarded(self) -> None:
        candidates = PIICleaner._enrichment_candidates(
            text_len=30, text_offset=100,
            known_spans=[(5, 15, "PERSON")],
        )
        assert len(candidates) == 0

    def test_span_clipped_to_text_boundary(self) -> None:
        candidates = PIICleaner._enrichment_candidates(
            text_len=20, text_offset=10,
            known_spans=[(5, 25, "PERSON")],
        )
        assert len(candidates) == 1
        assert candidates[0].start == 0
        assert candidates[0].end == 15

    def test_offset_adjusts_positions(self) -> None:
        candidates = PIICleaner._enrichment_candidates(
            text_len=50, text_offset=100,
            known_spans=[(110, 120, "PERSON")],
        )
        assert len(candidates) == 1
        assert candidates[0].start == 10
        assert candidates[0].end == 20

    def test_multiple_entity_types(self) -> None:
        candidates = PIICleaner._enrichment_candidates(
            text_len=100, text_offset=0,
            known_spans=[
                (5, 15, "PERSON"),
                (30, 55, "EMAIL"),
                (70, 85, "PHONE_NUMBER"),
            ],
        )
        assert len(candidates) == 3
        types = {c.entity_type for c in candidates}
        assert types == {"PERSON", "EMAIL", "PHONE_NUMBER"}


# ---------------------------------------------------------------------------
# clean_with_known_spans — combined detection
# ---------------------------------------------------------------------------


class TestCleanWithKnownSpans:
    def test_enrichment_span_replaces(self) -> None:
        cleaner = PIICleaner(context_similarity_threshold=0.5)
        text = "Jane Smith attended the hearing."
        cleaned, count = cleaner.clean_with_known_spans(
            text, known_spans=[(0, 10, "PERSON")], text_offset=0,
        )
        assert count >= 1
        assert "<PERSON>" in cleaned
        assert "Jane Smith" not in cleaned

    def test_enrichment_plus_regex_deduplicates(self) -> None:
        """Overlapping regex and enrichment spans produce one replacement."""
        cleaner = PIICleaner(context_similarity_threshold=0.5)
        text = "Mr. John Smith attended."
        cleaned, _count = cleaner.clean_with_known_spans(
            text, known_spans=[(4, 14, "PERSON")], text_offset=0,
        )
        assert "<PERSON>" in cleaned
        assert cleaned.count("<PERSON>") == 1

    def test_empty_known_spans_falls_back_to_regex(self) -> None:
        cleaner = PIICleaner(context_similarity_threshold=0.5)
        text = "Mr. John Smith attended."
        cleaned, count = cleaner.clean_with_known_spans(
            text, known_spans=[], text_offset=0,
        )
        assert count == 1
        assert "<PERSON>" in cleaned

    def test_empty_text(self) -> None:
        cleaner = PIICleaner()
        cleaned, count = cleaner.clean_with_known_spans("", [], 0)
        assert cleaned == ""
        assert count == 0


# ---------------------------------------------------------------------------
# Stage — clean_enriched_chunks (real models, no mocks)
# ---------------------------------------------------------------------------


class TestCleanEnrichedChunks:
    def test_natural_person_cleaned(self) -> None:
        text = "Mr. James Wilson attended the hearing on Monday."
        enrichment = _EnrichmentResult(
            text=text,
            persons=[
                Person(
                    id="p1", name=Span(4, 16), type="natural", role="witness",
                    mentions=[Span(4, 16)],
                ),
            ],
        )
        chunk = TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0)
        cleaner = PIICleaner(context_similarity_threshold=0.5)
        count = clean_enriched_chunks(
            [chunk], enrichment, cleaner, entities={"PERSON"},  # type: ignore[arg-type]
        )
        assert count >= 1
        assert "<PERSON>" in chunk.text
        assert "James Wilson" not in chunk.text

    def test_corporate_person_excluded_by_default(self) -> None:
        enrichment = _EnrichmentResult(
            text="Acme Corp signed the contract.",
            persons=[
                Person(
                    id="p1", name=Span(0, 9), type="corporate", role="seller",
                    mentions=[Span(0, 9)],
                ),
            ],
        )
        spans = _extract_known_spans(enrichment, {"PERSON"})  # type: ignore[arg-type]
        assert len(spans) == 0

    def test_natural_person_included(self) -> None:
        enrichment = _EnrichmentResult(
            text="Jane Smith attended.",
            persons=[
                Person(
                    id="p1", name=Span(0, 10), type="natural", role="witness",
                    mentions=[Span(0, 10)],
                ),
            ],
        )
        spans = _extract_known_spans(enrichment, {"PERSON"})  # type: ignore[arg-type]
        assert len(spans) == 1
        assert spans[0] == (0, 10, "PERSON")

    def test_empty_enrichment_no_replacements(self) -> None:
        text = "No entities here at all."
        enrichment = _EnrichmentResult(text=text)
        chunk = TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0)
        cleaner = PIICleaner(entities=["PERSON"])
        count = clean_enriched_chunks(
            [chunk], enrichment, cleaner,  # type: ignore[arg-type]
        )
        assert count == 0
        assert chunk.text == text


# ---------------------------------------------------------------------------
# person_types configuration
# ---------------------------------------------------------------------------


class TestPersonTypesFilter:
    def test_default_filters_natural_only(self) -> None:
        enrichment = _EnrichmentResult(
            text="Jane Smith and Acme Corp attended.",
            persons=[
                Person(id="p1", name=Span(0, 10), type="natural",
                       role="witness", mentions=[Span(0, 10)]),
                Person(id="p2", name=Span(15, 24), type="corporate",
                       role="seller", mentions=[Span(15, 24)]),
            ],
        )
        spans = _extract_known_spans(enrichment, {"PERSON"})  # type: ignore[arg-type]
        assert len(spans) == 1
        assert spans[0] == (0, 10, "PERSON")

    def test_include_corporate(self) -> None:
        enrichment = _EnrichmentResult(
            text="Jane Smith and Acme Corp attended.",
            persons=[
                Person(id="p1", name=Span(0, 10), type="natural",
                       role="witness", mentions=[Span(0, 10)]),
                Person(id="p2", name=Span(15, 24), type="corporate",
                       role="seller", mentions=[Span(15, 24)]),
            ],
        )
        spans = _extract_known_spans(
            enrichment, {"PERSON"}, person_types={"natural", "corporate"},  # type: ignore[arg-type]
        )
        assert len(spans) == 2

    def test_politic_included_when_configured(self) -> None:
        enrichment = _EnrichmentResult(
            text="The Commonwealth of Australia was a party.",
            persons=[
                Person(id="p1", name=Span(4, 30), type="politic",
                       role="other", mentions=[Span(4, 30)]),
            ],
        )
        spans = _extract_known_spans(
            enrichment, {"PERSON"}, person_types={"politic"},  # type: ignore[arg-type]
        )
        assert len(spans) == 1

    def test_config_person_types_default(self) -> None:
        cfg = PIIConfig()
        assert cfg.person_types == ["natural"]

    def test_config_person_types_custom(self) -> None:
        cfg = PIIConfig(person_types=["natural", "corporate"])
        assert "corporate" in cfg.person_types

    def test_corporate_person_cleaned_when_type_included(self) -> None:
        """End-to-end: corporate person cleaned when person_types includes corporate."""
        text = "Mr. James Brown of Acme Corp attended."
        enrichment = _EnrichmentResult(
            text=text,
            persons=[
                Person(id="p1", name=Span(4, 15), type="natural",
                       role="witness", mentions=[Span(4, 15)]),
                Person(id="p2", name=Span(19, 28), type="corporate",
                       role="seller", mentions=[Span(19, 28)]),
            ],
        )
        chunk = TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0)
        cleaner = PIICleaner(context_similarity_threshold=0.5)
        count = clean_enriched_chunks(
            [chunk], enrichment, cleaner,  # type: ignore[arg-type]
            entities={"PERSON"},
            person_types={"natural", "corporate"},
        )
        assert count >= 2
        assert "Acme Corp" not in chunk.text


# ---------------------------------------------------------------------------
# Address regex pattern
# ---------------------------------------------------------------------------


class TestAddressRegex:
    def test_simple_street_address(self) -> None:
        m = _ADDRESS_RE.search("Located at 100 George Street in Sydney.")
        assert m is not None
        assert "100 George Street" in m.group()

    def test_address_with_suburb_state_postcode(self) -> None:
        m = _ADDRESS_RE.search("Office at 42 Collins Street, Melbourne VIC 3000 is open.")
        assert m is not None
        assert "3000" in m.group()

    def test_unit_prefix(self) -> None:
        m = _ADDRESS_RE.search("Level 5, 100 George Street is the office.")
        assert m is not None
        assert "Level 5" in m.group()

    def test_abbreviated_street_type(self) -> None:
        m = _ADDRESS_RE.search("Send mail to 15 Park Rd please.")
        assert m is not None
        assert "15 Park Rd" in m.group()

    def test_po_box_not_matched(self) -> None:
        m = _ADDRESS_RE.search("PO Box 1234, Canberra ACT 2601")
        assert m is None

    def test_plain_number_no_street_type(self) -> None:
        m = _ADDRESS_RE.search("Section 42 Administrative Decisions is relevant.")
        assert m is None

    def test_range_street_number(self) -> None:
        m = _ADDRESS_RE.search("Located at 100-102 Main Street in town.")
        assert m is not None
        assert "100-102 Main Street" in m.group()


# ---------------------------------------------------------------------------
# Address detection in PIICleaner
# ---------------------------------------------------------------------------


class TestPIICleanerAddress:
    def test_address_detected_when_entity_enabled(self) -> None:
        cleaner = PIICleaner(entities=["ADDRESS"])
        text = "The office is at 100 George Street in Sydney."
        cleaned, count = cleaner.clean(text)
        assert count == 1
        assert "<ADDRESS>" in cleaned
        assert "100 George Street" not in cleaned

    def test_address_not_detected_when_entity_disabled(self) -> None:
        cleaner = PIICleaner(entities=["PERSON"])
        text = "The office is at 100 George Street in Sydney."
        cleaned, _count = cleaner.clean(text)
        assert "<ADDRESS>" not in cleaned

    def test_person_and_address_both_detected(self) -> None:
        cleaner = PIICleaner(entities=["PERSON", "ADDRESS"])
        text = "Mr. John Smith lives at 42 Collins Street in Melbourne."
        cleaned, count = cleaner.clean(text)
        assert "<PERSON>" in cleaned
        assert "<ADDRESS>" in cleaned
        assert count == 2


# ---------------------------------------------------------------------------
# ADDRESS in enrichment extraction
# ---------------------------------------------------------------------------


class TestEnrichmentAddress:
    def test_address_location_extracted(self) -> None:
        enrichment = _EnrichmentResult(
            text="Office at 100 George Street, Sydney NSW 2000.",
            locations=[
                Location(
                    id="loc1", name=Span(10, 44), type="address",
                    mentions=[Span(10, 44)],
                ),
            ],
        )
        spans = _extract_known_spans(enrichment, {"ADDRESS"})  # type: ignore[arg-type]
        assert len(spans) == 1
        assert spans[0] == (10, 44, "ADDRESS")

    def test_city_location_not_extracted(self) -> None:
        enrichment = _EnrichmentResult(
            text="The incident occurred in Sydney.",
            locations=[
                Location(
                    id="loc1", name=Span(24, 30), type="city",
                    mentions=[Span(24, 30)],
                ),
            ],
        )
        spans = _extract_known_spans(enrichment, {"ADDRESS"})  # type: ignore[arg-type]
        assert len(spans) == 0

    def test_country_and_address_mixed(self) -> None:
        enrichment = _EnrichmentResult(
            text="Australia. 100 George Street, Sydney.",
            locations=[
                Location(id="loc1", name=Span(0, 9), type="country",
                         mentions=[Span(0, 9)]),
                Location(id="loc2", name=Span(11, 35), type="address",
                         mentions=[Span(11, 35)]),
            ],
        )
        spans = _extract_known_spans(enrichment, {"ADDRESS"})  # type: ignore[arg-type]
        assert len(spans) == 1
        assert spans[0][2] == "ADDRESS"

    def test_enrichment_address_replaces_in_chunk(self) -> None:
        """End-to-end: enrichment address span is replaced in chunk text."""
        text = "Send mail to 100 George Street, Sydney NSW 2000 please."
        enrichment = _EnrichmentResult(
            text=text,
            locations=[
                Location(
                    id="loc1", name=Span(13, 47), type="address",
                    mentions=[Span(13, 47)],
                ),
            ],
        )
        chunk = TextChunk(text=text, start_char=0, end_char=len(text), chunk_index=0)
        cleaner = PIICleaner(entities=["ADDRESS"])
        count = clean_enriched_chunks(
            [chunk], enrichment, cleaner,  # type: ignore[arg-type]
            entities={"ADDRESS"},
        )
        assert count >= 1
        assert "<ADDRESS>" in chunk.text
        assert "George Street" not in chunk.text

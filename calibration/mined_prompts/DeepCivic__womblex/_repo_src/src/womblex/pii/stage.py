"""PII cleaning pipeline stage.

Applies PII cleaning at a configurable pipeline point:

- ``post_extraction``: Cleans page texts on the ExtractionResult before chunking.
  Ensures chunks never contain raw PII, but operates on larger text blocks.
- ``post_chunk``:      Cleans individual chunk texts after chunking.
  Consistent with how the redaction stage annotates chunks.
- ``post_enrichment``: Cleans chunk texts using Isaacus enrichment graph entities
  as high-confidence PII candidates, supplemented by regex detection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from womblex.analyse.models import EnrichmentResult
    from womblex.ingest.extract import ExtractionResult
    from womblex.pii.cleaner import PIICleaner
    from womblex.process.chunker import TextChunk

logger = logging.getLogger(__name__)


def clean_extraction(extraction: ExtractionResult, cleaner: PIICleaner) -> int:
    """Apply PII cleaning to all page texts in an ExtractionResult.

    Mutates ``page.text`` in-place.  The ``full_text`` property is computed
    dynamically from pages, so no separate update is required.

    Args:
        extraction: Extraction result whose page texts will be cleaned.
        cleaner: Configured PIICleaner instance.

    Returns:
        Total number of PII replacements made across all pages.
    """
    total = 0
    for page in extraction.pages:
        if not page.text:
            continue
        cleaned, count = cleaner.clean(page.text)
        if count:
            page.text = cleaned
            total += count
    return total


def clean_chunks(chunks: list[TextChunk], cleaner: PIICleaner) -> int:
    """Apply PII cleaning to all chunk texts.

    Mutates ``chunk.text`` in-place.

    Args:
        chunks: Chunks whose text will be cleaned.
        cleaner: Configured PIICleaner instance.

    Returns:
        Total number of PII replacements made across all chunks.
    """
    total = 0
    for chunk in chunks:
        if not chunk.text:
            continue
        cleaned, count = cleaner.clean(chunk.text)
        if count:
            chunk.text = cleaned
            total += count
    return total


def _extract_known_spans(
    enrichment: EnrichmentResult,
    entities: set[str],
    person_types: set[str] | None = None,
) -> list[tuple[int, int, str]]:
    """Extract PII spans from enrichment entities in full-document coordinates.

    Args:
        enrichment: Isaacus enrichment result.
        entities: Entity types to extract (PERSON, EMAIL, ADDRESS, etc.).
        person_types: Which person types to include (natural, corporate, politic).
            Defaults to ``{"natural"}``.
    """
    allowed_person_types = person_types or {"natural"}
    spans: list[tuple[int, int, str]] = []

    if "PERSON" in entities:
        for person in enrichment.persons:
            if person.type not in allowed_person_types:
                continue
            for mention in person.mentions:
                spans.append((mention.start, mention.end, "PERSON"))

    if "ADDRESS" in entities:
        for location in enrichment.locations:
            if location.type != "address":
                continue
            for mention in location.mentions:
                spans.append((mention.start, mention.end, "ADDRESS"))

    if "EMAIL" in entities:
        for email in enrichment.emails:
            for mention in email.mentions:
                spans.append((mention.start, mention.end, "EMAIL"))

    if "PHONE_NUMBER" in entities:
        for phone in enrichment.phone_numbers:
            for mention in phone.mentions:
                spans.append((mention.start, mention.end, "PHONE_NUMBER"))

    if "ID_NUMBER" in entities:
        for idn in enrichment.id_numbers:
            for mention in idn.mentions:
                spans.append((mention.start, mention.end, "ID_NUMBER"))

    return spans


def clean_enriched_chunks(
    chunks: list[TextChunk],
    enrichment: EnrichmentResult,
    cleaner: PIICleaner,
    entities: set[str] | None = None,
    person_types: set[str] | None = None,
) -> int:
    """Clean chunk texts using enrichment graph entities plus regex detection.

    Enrichment-derived spans (natural persons, addresses, contact info) are
    treated as high-confidence candidates.  Regex detection still runs as a
    fallback for anything the enrichment missed.

    Mutates ``chunk.text`` in-place.

    Args:
        chunks: Chunks whose text will be cleaned.
        enrichment: Isaacus enrichment result for the document.
        cleaner: Configured PIICleaner instance.
        entities: Entity types to extract from enrichment. Defaults to
            ``{"PERSON"}``.
        person_types: Which person types to include (natural, corporate,
            politic). Defaults to ``{"natural"}``.

    Returns:
        Total number of PII replacements made across all chunks.
    """
    target_entities = entities or {"PERSON"}
    known_spans = _extract_known_spans(enrichment, target_entities, person_types)

    total = 0
    for chunk in chunks:
        if not chunk.text:
            continue
        cleaned, count = cleaner.clean_with_known_spans(
            chunk.text, known_spans, text_offset=chunk.start_char
        )
        if count:
            chunk.text = cleaned
            total += count
    return total

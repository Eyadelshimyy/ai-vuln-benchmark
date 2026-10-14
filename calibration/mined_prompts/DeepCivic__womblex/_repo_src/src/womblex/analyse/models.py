"""Data models for Isaacus enrichment results.

Mirrors the Isaacus Legal Graph Schema (ILGS) v1 response structure.
All spans use zero-based, half-open Unicode code-point indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    """A zero-based, half-open Unicode code-point span within document text."""

    start: int
    end: int

    def decode(self, text: str) -> str:
        """Extract the span text from the document."""
        return text[self.start : self.end]

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and self.end >= other.end


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A structurally distinct portion of document content."""

    id: str
    kind: str  # container | unit | item | figure
    type: str | None  # chapter, section, paragraph, etc.
    category: str  # front_matter | scope | main | annotation | back_matter | other
    span: Span
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    level: int = 0
    type_name: Span | None = None
    code: Span | None = None
    title: Span | None = None


@dataclass
class CrossReference:
    """A reference within the document pointing to one or more segments."""

    start: str  # segment id
    end: str  # segment id
    span: Span


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


@dataclass
class Location:
    """A geographical location identified in the document."""

    id: str
    name: Span
    type: str  # country | state | city | address | other
    mentions: list[Span] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class Person:
    """A legal person (natural, corporate, or politic) identified in the document."""

    id: str
    name: Span
    type: str  # natural | corporate | politic
    role: str  # seller, buyer, other, etc.
    mentions: list[Span] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    residence: str | None = None  # location id


@dataclass
class Email:
    """An email address identified in the document belonging to a person."""

    address: str
    person: str  # person id
    mentions: list[Span] = field(default_factory=list)


@dataclass
class Website:
    """A website identified in the document belonging to a person."""

    url: str
    person: str  # person id
    mentions: list[Span] = field(default_factory=list)


@dataclass
class PhoneNumber:
    """A phone number identified in the document belonging to a person."""

    number: str
    person: str  # person id
    mentions: list[Span] = field(default_factory=list)


@dataclass
class IDNumber:
    """An identification number belonging to a person."""

    number: str
    person: str  # person id
    mentions: list[Span] = field(default_factory=list)


@dataclass
class Term:
    """A term assigned a definite meaning within the document."""

    id: str
    name: Span
    meaning: Span
    mentions: list[Span] = field(default_factory=list)


@dataclass
class ExternalDocument:
    """An external document referenced within the document."""

    id: str
    name: Span
    type: str  # statute | regulation | decision | contract | other
    reception: str  # positive | mixed | negative | neutral
    mentions: list[Span] = field(default_factory=list)
    pinpoints: list[Span] = field(default_factory=list)
    jurisdiction: str | None = None


@dataclass
class Quote:
    """A quotation within the document."""

    span: Span
    amending: bool = False
    source_segment: str | None = None
    source_document: str | None = None
    source_person: str | None = None


@dataclass
class DateInfo:
    """A date identified in the document."""

    value: str  # ISO 8601 (YYYY-MM-DD)
    type: str  # creation | signature | effective | expiry | delivery | renewal | payment | birth | death
    mentions: list[Span] = field(default_factory=list)
    person: str | None = None  # person id


# ---------------------------------------------------------------------------
# Top-level enrichment result
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """Complete enrichment result for a single document (ILGS Document)."""

    text: str
    type: str  # statute | regulation | decision | contract | other
    jurisdiction: str | None = None
    title: Span | None = None
    subtitle: Span | None = None

    segments: list[Segment] = field(default_factory=list)
    crossreferences: list[CrossReference] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    persons: list[Person] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    websites: list[Website] = field(default_factory=list)
    phone_numbers: list[PhoneNumber] = field(default_factory=list)
    id_numbers: list[IDNumber] = field(default_factory=list)
    terms: list[Term] = field(default_factory=list)
    external_documents: list[ExternalDocument] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    dates: list[DateInfo] = field(default_factory=list)
    headings: list[Span] = field(default_factory=list)
    junk: list[Span] = field(default_factory=list)

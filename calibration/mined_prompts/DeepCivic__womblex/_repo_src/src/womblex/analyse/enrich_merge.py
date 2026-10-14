"""Merge per-segment enrichment results back into one document result.

Documents past the rate-limit token ceiling are split on structural
boundaries (:func:`womblex.utils.token_packer.split_on_boundaries`) and each
segment is enriched separately. This module stitches the per-segment
:class:`EnrichmentResult` objects back into a single result whose spans index
the full document text — the same offset stitch the enricher performs
internally for >16 K inputs, applied one level up.

Two adjustments per segment ``i`` at char offset ``delta``:

- **offset shift**: every :class:`Span` (entity names, mentions, titles,
  headings, junk, quote/segment spans) is moved by ``delta`` so it indexes
  the full text, and ``Span.decode(full_text)`` returns the right substring.
- **id namespacing**: every entity/segment id — and every reference to one
  (parent, children, residence, person, source_*) — is prefixed with the
  segment index. Segments are enriched independently, so their ids restart at
  ``s1``/``p1`` each time; the prefix keeps them unique and every reference
  points within its own segment, so the prefixed graph stays consistent.
"""

from __future__ import annotations

from womblex.analyse.models import (
    CrossReference,
    DateInfo,
    Email,
    EnrichmentResult,
    ExternalDocument,
    IDNumber,
    Location,
    Person,
    PhoneNumber,
    Quote,
    Segment,
    Span,
    Term,
    Website,
)
from womblex.utils.token_packer import TextSegment


def _span(s: Span | None, delta: int) -> Span | None:
    return Span(start=s.start + delta, end=s.end + delta) if s is not None else None


def _spans(ss: list[Span], delta: int) -> list[Span]:
    return [Span(start=s.start + delta, end=s.end + delta) for s in ss]


def _pid(prefix: str, ref: str | None) -> str | None:
    return f"{prefix}{ref}" if ref else ref


def _pids(prefix: str, refs: list[str]) -> list[str]:
    return [f"{prefix}{r}" for r in refs]


def _merge_one(
    result: EnrichmentResult, delta: int, prefix: str, merged: EnrichmentResult,
) -> None:
    """Shift + prefix one segment's result into ``merged`` (in place)."""
    merged.segments.extend(
        Segment(
            id=f"{prefix}{seg.id}",
            kind=seg.kind, type=seg.type, category=seg.category,
            span=_span(seg.span, delta),  # type: ignore[arg-type]
            parent=_pid(prefix, seg.parent),
            children=_pids(prefix, seg.children),
            level=seg.level,
            type_name=_span(seg.type_name, delta),
            code=_span(seg.code, delta),
            title=_span(seg.title, delta),
        )
        for seg in result.segments
    )
    merged.crossreferences.extend(
        CrossReference(
            start=f"{prefix}{x.start}", end=f"{prefix}{x.end}",
            span=_span(x.span, delta),  # type: ignore[arg-type]
        )
        for x in result.crossreferences
    )
    merged.locations.extend(
        Location(
            id=f"{prefix}{loc.id}", name=_span(loc.name, delta),  # type: ignore[arg-type]
            type=loc.type, mentions=_spans(loc.mentions, delta),
            parent=_pid(prefix, loc.parent), children=_pids(prefix, loc.children),
        )
        for loc in result.locations
    )
    merged.persons.extend(
        Person(
            id=f"{prefix}{p.id}", name=_span(p.name, delta),  # type: ignore[arg-type]
            type=p.type, role=p.role, mentions=_spans(p.mentions, delta),
            parent=_pid(prefix, p.parent), children=_pids(prefix, p.children),
            residence=_pid(prefix, p.residence),
        )
        for p in result.persons
    )
    merged.emails.extend(
        Email(address=e.address, person=f"{prefix}{e.person}", mentions=_spans(e.mentions, delta))
        for e in result.emails
    )
    merged.websites.extend(
        Website(url=w.url, person=f"{prefix}{w.person}", mentions=_spans(w.mentions, delta))
        for w in result.websites
    )
    merged.phone_numbers.extend(
        PhoneNumber(number=p.number, person=f"{prefix}{p.person}", mentions=_spans(p.mentions, delta))
        for p in result.phone_numbers
    )
    merged.id_numbers.extend(
        IDNumber(number=i.number, person=f"{prefix}{i.person}", mentions=_spans(i.mentions, delta))
        for i in result.id_numbers
    )
    merged.terms.extend(
        Term(
            id=f"{prefix}{t.id}", name=_span(t.name, delta),  # type: ignore[arg-type]
            meaning=_span(t.meaning, delta),  # type: ignore[arg-type]
            mentions=_spans(t.mentions, delta),
        )
        for t in result.terms
    )
    merged.external_documents.extend(
        ExternalDocument(
            id=f"{prefix}{x.id}", name=_span(x.name, delta),  # type: ignore[arg-type]
            type=x.type, reception=x.reception, mentions=_spans(x.mentions, delta),
            pinpoints=_spans(x.pinpoints, delta), jurisdiction=x.jurisdiction,
        )
        for x in result.external_documents
    )
    merged.quotes.extend(
        Quote(
            span=_span(q.span, delta),  # type: ignore[arg-type]
            amending=q.amending,
            source_segment=_pid(prefix, q.source_segment),
            source_document=_pid(prefix, q.source_document),
            source_person=_pid(prefix, q.source_person),
        )
        for q in result.quotes
    )
    merged.dates.extend(
        DateInfo(
            value=d.value, type=d.type, mentions=_spans(d.mentions, delta),
            person=_pid(prefix, d.person),
        )
        for d in result.dates
    )
    merged.headings.extend(_spans(result.headings, delta))
    merged.junk.extend(_spans(result.junk, delta))


def merge_segment_results(
    full_text: str,
    segment_results: list[tuple[TextSegment, EnrichmentResult]],
) -> EnrichmentResult:
    """Stitch per-segment results into one whose spans index ``full_text``.

    ``segment_results`` is ``(TextSegment, EnrichmentResult)`` in document
    order; each result's spans are shifted by the segment's ``start_char``
    and its ids namespaced by segment index. Document-level scalars
    (``type``/``jurisdiction``) come from the first segment; ``title`` /
    ``subtitle`` from the first segment that carries one (shifted).
    """
    if not segment_results:
        return EnrichmentResult(text=full_text, type="other")

    first = segment_results[0][1]
    merged = EnrichmentResult(
        text=full_text, type=first.type, jurisdiction=first.jurisdiction,
    )
    for i, (seg, result) in enumerate(segment_results):
        delta = seg.start_char
        prefix = f"{i}:"
        if merged.title is None and result.title is not None:
            merged.title = _span(result.title, delta)
        if merged.subtitle is None and result.subtitle is not None:
            merged.subtitle = _span(result.subtitle, delta)
        _merge_one(result, delta, prefix, merged)
    return merged


__all__ = ["merge_segment_results"]

"""Unit tests for offset-merging split-document enrichment results."""

from __future__ import annotations

from womblex.analyse.enrich_merge import merge_segment_results
from womblex.analyse.models import (
    EnrichmentResult,
    Location,
    Person,
    Segment,
    Span,
)
from womblex.utils.token_packer import TextSegment


def _seg_result(text: str, person_start: int, person_len: int) -> EnrichmentResult:
    """A one-person, one-segment result with spans local to ``text``."""
    return EnrichmentResult(
        text=text,
        type="decision",
        jurisdiction="new_south_wales",
        segments=[Segment(
            id="s1", kind="unit", type="paragraph", category="main",
            span=Span(0, len(text)),
        )],
        persons=[Person(
            id="p1", name=Span(person_start, person_start + person_len),
            type="natural", role="other",
            mentions=[Span(person_start, person_start + person_len)],
            residence="l1",
        )],
        locations=[Location(id="l1", name=Span(0, 4), type="address", mentions=[Span(0, 4)])],
    )


def test_merge_shifts_spans_into_full_text():
    full = "Alice went home.\n\nBob went away too."
    # segment 0 = full[0:16], segment 1 = full[18:]
    seg0 = TextSegment(text=full[0:16], start_char=0, end_char=16, tokens=3)
    seg1 = TextSegment(text=full[18:], start_char=18, end_char=len(full), tokens=4)
    r0 = _seg_result(seg0.text, person_start=0, person_len=5)   # "Alice"
    r1 = _seg_result(seg1.text, person_start=0, person_len=3)   # "Bob"

    merged = merge_segment_results(full, [(seg0, r0), (seg1, r1)])

    assert merged.text == full
    assert len(merged.persons) == 2
    # spans now index the FULL text — decode returns the right surface forms
    assert merged.persons[0].name.decode(full) == "Alice"
    assert merged.persons[1].name.decode(full) == "Bob"
    assert merged.persons[1].name.start == 18  # shifted by seg1.start_char


def test_merge_namespaces_ids_and_references():
    full = "x" * 40
    seg0 = TextSegment(text=full[0:20], start_char=0, end_char=20, tokens=1)
    seg1 = TextSegment(text=full[20:], start_char=20, end_char=40, tokens=1)
    merged = merge_segment_results(
        full,
        [(seg0, _seg_result(full[0:20], 0, 4)), (seg1, _seg_result(full[20:], 0, 4))],
    )
    # ids restart at p1/l1/s1 per segment; the prefix keeps them unique
    person_ids = {p.id for p in merged.persons}
    assert person_ids == {"0:p1", "1:p1"}
    # residence reference is prefixed to match its own segment's location id
    for p in merged.persons:
        assert p.residence in {"0:l1", "1:l1"}
    loc_ids = {loc.id for loc in merged.locations}
    assert loc_ids == {"0:l1", "1:l1"}
    # each person's residence points at a real location id
    for p in merged.persons:
        assert p.residence in loc_ids


def test_merge_empty_is_safe():
    merged = merge_segment_results("abc", [])
    assert merged.text == "abc"
    assert merged.persons == []

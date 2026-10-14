"""Tests for token-aware text chunking helpers."""

from __future__ import annotations

import pytest
from ellements.core.chunking import (
    TextProcessor,
    _chunk_tokens,
    count_tokens,
    map_reduce,
    sequential_refine,
    simple_truncate,
)


class TestChunkTokens:
    def test_no_overlap(self):
        tokens = list(range(100))
        chunks = _chunk_tokens(tokens, 30, 0)
        assert len(chunks) == 4
        assert chunks[0] == list(range(0, 30))
        assert chunks[3] == list(range(90, 100))

    def test_with_overlap(self):
        tokens = list(range(100))
        chunks = _chunk_tokens(tokens, 30, 10)
        assert chunks[0] == list(range(0, 30))
        assert chunks[0][-10:] == chunks[1][:10]

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError):
            _chunk_tokens(list(range(100)), 30, 30)

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size"):
            _chunk_tokens(list(range(100)), 0, 0)

    def test_overlap_must_not_be_negative(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            _chunk_tokens(list(range(100)), 30, -1)

    def test_empty_input(self):
        assert _chunk_tokens([], 30, 0) == []


class TestMapReduce:
    def test_basic_map_reduce(self):
        big_text = "word " * 2000
        result = map_reduce(
            big_text, max_tokens=50, mapper=lambda t: "x", chunk_size=40
        )
        assert "x" in result

    def test_custom_reducer(self):
        big_text = "word " * 2000
        result = map_reduce(
            big_text,
            max_tokens=50,
            mapper=lambda t: "item",
            reducer=lambda rs: " + ".join(rs),
            chunk_size=40,
        )
        assert " + " in result

    def test_short_text_bypass(self):
        assert (
            map_reduce("short", max_tokens=1000, mapper=lambda t: "NEVER")
            == "short"
        )

    def test_non_shrinking_reducer_raises(self):
        big_text = "word " * 2000
        with pytest.raises(ValueError, match="mapper/reducer combination"):
            map_reduce(big_text, max_tokens=50, mapper=lambda t: t, chunk_size=40)


class TestSequentialRefine:
    def test_basic_refinement(self):
        long_text = "data " * 500
        calls = []
        sequential_refine(
            long_text,
            max_tokens=50,
            refiner=lambda a, c: (calls.append(1), a + ".")[1],
            chunk_size=30,
        )
        assert calls

    def test_with_initial_processor(self):
        long_text = "data " * 500
        ip_calls = []
        result = sequential_refine(
            long_text,
            max_tokens=5000,
            refiner=lambda a, c: a + "+",
            chunk_size=30,
            initial_processor=lambda t: (ip_calls.append(1), "INIT")[1],
        )
        assert len(ip_calls) == 1
        assert result.startswith("INIT")

    def test_short_text_with_initial_processor(self):
        result = sequential_refine(
            "tiny",
            max_tokens=1000,
            refiner=lambda a, c: a,
            initial_processor=lambda t: f"PROCESSED({t})",
        )
        assert result == "PROCESSED(tiny)"


class TestTextProcessor:
    def test_map_reduce_with_mapper_and_reducer(self):
        p = TextProcessor(
            max_tokens=50, strategy="map_reduce", chunk_size=40, chunk_overlap=5
        )
        p.set_mapper(lambda t: "x")
        p.set_reducer(lambda rs: ",".join(rs))
        assert "," in p.process("word " * 2000)

    def test_map_reduce_without_mapper_raises(self):
        p = TextProcessor(max_tokens=10, strategy="map_reduce")
        with pytest.raises(ValueError, match="mapper"):
            p.process("text " * 2000)

    def test_sequential_without_refiner_raises(self):
        p = TextProcessor(max_tokens=10, strategy="sequential")
        with pytest.raises(ValueError, match="refiner"):
            p.process("text " * 2000)

    def test_simple_truncate_adds_suffix(self):
        result = simple_truncate("word " * 2000, max_tokens=50)
        assert "truncated" in result


def test_count_tokens_basic():
    assert count_tokens("hello world") > 0

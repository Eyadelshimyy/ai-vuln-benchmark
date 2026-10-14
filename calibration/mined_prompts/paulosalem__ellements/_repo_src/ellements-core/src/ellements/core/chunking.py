"""Token-aware text **chunking** primitives.

Everything here is LLM-dependency-free and operates on plain strings:

- :func:`count_tokens` — token accounting via tiktoken (cached
  encoder lookup so callers can call it freely on hot paths).
- :func:`simple_truncate` — direct character/token truncation with an
  optional suffix.
- :func:`map_reduce` — chunked fan-out (one mapper call per chunk)
  followed by a single reducer.
- :func:`sequential_refine` — rolling synthesis where each chunk
  refines an accumulating draft.
- :class:`TextProcessor` — strategy-based orchestration of the above
  with a single ``process()`` entry point.

These primitives accept arbitrary mapper/reducer/refiner callables so
callers plug in their own LLM-backed (or non-LLM-backed) processing
without this module having an opinion.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Literal

import tiktoken

TruncationStrategy = Literal["simple", "map_reduce", "sequential"]

_DEFAULT_MODEL = "gpt-5-mini"
_Mapper = Callable[[str], str]
_Reducer = Callable[[list[str]], str]
_Refiner = Callable[[str, str], str]
_InitialProcessor = Callable[[str], str]

__all__ = [
    "TextProcessor",
    "TruncationStrategy",
    "count_tokens",
    "map_reduce",
    "sequential_refine",
    "simple_truncate",
]


@functools.lru_cache(maxsize=32)
def _get_encoding(model: str = _DEFAULT_MODEL) -> tiktoken.Encoding:
    """Return the tiktoken encoding for *model*, falling back to ``cl100k_base``."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def _resolve_chunk_size(max_tokens: int, chunk_size: int | None) -> int:
    _validate_positive("max_tokens", max_tokens)
    if chunk_size is None:
        return max_tokens
    _validate_positive("chunk_size", chunk_size)
    return chunk_size


def _chunk_tokens(
    tokens: Sequence[int],
    chunk_size: int,
    chunk_overlap: int = 0,
) -> list[list[int]]:
    """Split a token sequence into chunks with optional overlap."""
    _validate_positive("chunk_size", chunk_size)
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if not tokens:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[list[int]] = []
    for start in range(0, len(tokens), stride):
        chunk = list(tokens[start : start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def count_tokens(text: str, model: str = _DEFAULT_MODEL) -> int:
    """Return the number of tokens in *text* under the given model's tokenizer."""
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


def simple_truncate(
    text: str,
    max_tokens: int,
    model: str = _DEFAULT_MODEL,
    suffix: str = "\n\n[... truncated ...]",
) -> str:
    """Truncate *text* to at most *max_tokens* tokens, appending *suffix*."""
    _validate_positive("max_tokens", max_tokens)
    encoding = _get_encoding(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    suffix_tokens = len(encoding.encode(suffix))
    keep_tokens = max_tokens - suffix_tokens
    truncated_tokens = tokens[:max_tokens] if keep_tokens <= 0 else tokens[:keep_tokens]
    truncated_text: str = encoding.decode(truncated_tokens)
    return truncated_text + suffix if keep_tokens > 0 else truncated_text


def map_reduce(
    text: str,
    max_tokens: int,
    mapper: _Mapper,
    reducer: _Reducer | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Process *text* via a map-reduce pipeline until it fits the token budget."""
    resolved_chunk_size = _resolve_chunk_size(max_tokens, chunk_size)
    reduce_results = reducer or (lambda results: "\n\n".join(results))
    encoding = _get_encoding(model)
    current = text

    while True:
        current_tokens = encoding.encode(current)
        if len(current_tokens) <= max_tokens:
            return current

        token_chunks = _chunk_tokens(current_tokens, resolved_chunk_size, chunk_overlap)
        mapped_results = [mapper(encoding.decode(chunk)) for chunk in token_chunks]
        current = reduce_results(mapped_results)
        current_token_count = len(encoding.encode(current))

        if current_token_count <= max_tokens:
            return current
        if current_token_count >= len(current_tokens):
            raise ValueError(
                "map_reduce requires the mapper/reducer combination to reduce "
                "the text when the input exceeds max_tokens"
            )


def sequential_refine(
    text: str,
    max_tokens: int,
    refiner: _Refiner,
    chunk_size: int | None = None,
    chunk_overlap: int = 0,
    initial_processor: _InitialProcessor | None = None,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Process *text* chunk-by-chunk while refining an accumulated result."""
    resolved_chunk_size = _resolve_chunk_size(max_tokens, chunk_size)
    encoding = _get_encoding(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return initial_processor(text) if initial_processor is not None else text

    token_chunks = _chunk_tokens(tokens, resolved_chunk_size, chunk_overlap)
    chunks = [encoding.decode(chunk) for chunk in token_chunks]

    accumulated = (
        initial_processor(chunks[0]) if initial_processor is not None else chunks[0]
    )
    for chunk in chunks[1:]:
        accumulated = refiner(accumulated, chunk)

    if len(encoding.encode(accumulated)) > max_tokens:
        return simple_truncate(accumulated, max_tokens, model)
    return accumulated


class TextProcessor:
    """Configurable orchestrator over the chunking primitives."""

    def __init__(
        self,
        max_tokens: int = 4000,
        strategy: TruncationStrategy = "simple",
        model: str = _DEFAULT_MODEL,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> None:
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._mapper: _Mapper | None = None
        self._reducer: _Reducer | None = None
        self._refiner: _Refiner | None = None
        self._initial_processor: _InitialProcessor | None = None

    def set_mapper(self, mapper: _Mapper) -> None:
        """Bind the mapper used by the ``map_reduce`` strategy."""
        self._mapper = mapper

    def set_reducer(self, reducer: _Reducer) -> None:
        """Bind the reducer used by the ``map_reduce`` strategy."""
        self._reducer = reducer

    def set_refiner(self, refiner: _Refiner) -> None:
        """Bind the refiner used by the ``sequential`` strategy."""
        self._refiner = refiner

    def set_initial_processor(self, processor: _InitialProcessor) -> None:
        """Bind the initial processor used for the first chunk in sequential mode."""
        self._initial_processor = processor

    def process(self, text: str) -> str:
        """Process *text* according to the configured strategy."""
        if self.strategy == "simple":
            return simple_truncate(text, self.max_tokens, self.model)

        if self.strategy == "map_reduce":
            if self._mapper is None:
                raise ValueError(
                    "map_reduce strategy requires a mapper. Call set_mapper() first."
                )
            return map_reduce(
                text=text,
                max_tokens=self.max_tokens,
                mapper=self._mapper,
                reducer=self._reducer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                model=self.model,
            )

        if self.strategy == "sequential":
            if self._refiner is None:
                raise ValueError(
                    "sequential strategy requires a refiner. Call set_refiner() first."
                )
            return sequential_refine(
                text=text,
                max_tokens=self.max_tokens,
                refiner=self._refiner,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                initial_processor=self._initial_processor,
                model=self.model,
            )

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in *text* using the processor's configured model."""
        return count_tokens(text, self.model)

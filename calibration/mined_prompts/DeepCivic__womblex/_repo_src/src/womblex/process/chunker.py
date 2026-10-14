"""Text chunking using semchunk.

Adapter boundary (audited I5, 2026-05-30). semchunk 3.x owns all
chunking; Womblex is a thin adapter handling only what semchunk can't
know about — parquet I/O, element-stream → ``ChunkInput`` projection,
source-hash plumbing, and ``<REDACTED>`` cross-boundary repair.
semchunk's own parameters *are* the feature surface; this module adds
no toggle that re-exposes a semchunk feature under a Womblex name (the
dead ``ChunkingConfig.batch`` flag was removed in I5 — chunk_batch
always batches across the whole input list).

Pass-through to semchunk:

- :func:`create_chunker` wraps ``semchunk.chunkerify`` and exposes
  every creation-time parameter (``tokenizer`` → ``tokenizer_or_token_counter``,
  ``chunk_size``, ``chunking_model``, ``isaacus_client``,
  ``tokenizer_kwargs``, ``memoize``, ``cache_maxsize``, ``max_token_chars``).
  ``chunking_model`` is semchunk 4's AI-chunking lever and stays ``None``
  by default — composable, so callers using a non-Kanon tokeniser keep the
  offline token-based split untouched.
- :func:`chunk_batch` is the single entry point used by every caller
  (E2E :func:`womblex.operations.run_chunking`, per-stage
  :mod:`womblex.process.chunk_stage`). It flattens narratives across
  every input document into one semchunk call, and table markdowns into
  another, so ``processes``, ``overlap`` and ``progress`` pass straight
  through and parallelise across the whole batch instead of being thrown
  away per-document. ``offsets=True`` is pinned (Womblex needs char
  offsets for page mapping); every other ``__call__`` argument is the
  caller's via ``ChunkingConfig``.

Womblex-only surface (no semchunk equivalent):

- :class:`TextChunk` — Womblex's chunk schema (offsets, content type,
  page span, redaction flag, token count).
- :func:`table_to_markdown` — TableData → markdown projection.
- :func:`reassemble_narrative` / :func:`collect_tables_from_elements` /
  :func:`build_chunk_input` — element-stream → ``ChunkInput`` projection.
- :func:`element_spans` / :func:`chunks_in_document_order` — the narrative
  offset map, and the interleave it enables. The two projections above are
  disjoint, so document order survives only as the ``elem_order`` anchor on
  table chunks; these turn that anchor back into a position.
- :func:`_repair_redaction_splits` — heal ``<REDACTED>`` markers split
  across a chunk boundary; semchunk has no opinion about our marker.

AI chunking (semchunk 4) and enrichment reuse: when ``chunking_model``
is set, semchunk picks chunk boundaries from an ILGS Document. To avoid
enriching the same narrative twice (once at chunk time, once in the
``enrich`` stage), :func:`chunk_batch` accepts ``narrative_overrides``
mapping ``source_hash`` → a pre-enriched ``Document`` persisted by the
enrich stage (``*.enrichment_doc.parquet``). An override is used only when
its ``.text`` is byte-identical to the doc's reassembled narrative — the
coordinate-space guard from ``docs/decisions.md``; on mismatch (or no
sidecar) the doc falls back to passing the string and letting semchunk
self-enrich. The composable default (``chunking_model=None``) keeps the
offline token split, so non-Kanon callers are unaffected.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import semchunk

from womblex.ingest.elements import TEXT_KINDS, Element
from womblex.ingest.views import _element_to_table_data, _sheets_to_table_data
from womblex.utils.model_registry import SLOT_TOKENIZER, record_use, register, resolve

logger = logging.getLogger(__name__)

NARRATIVE_JOIN = "\n\n"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TextChunk:
    """A single chunk of text with offset, content-type and locality metadata."""

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    content_type: str = "narrative"  # "narrative" | "table"
    has_redaction: bool = False
    page_start: int | None = None
    page_end: int | None = None
    # Document-order anchor for table chunks: the `elem_order` of the table
    # element this chunk came from. Always None for narrative chunks — they
    # straddle multiple elements, which is why chunks otherwise join to
    # elements by offset overlap rather than by elem_order. Lets a consumer
    # reconstruct narrative ↔ table document order, which the two disjoint
    # projections (narrative string vs per-table markdown) otherwise lose.
    elem_order: int | None = None
    # Length of `text` in the units the chunker budgeted with — its own
    # token counter, not a re-estimate. Set by chunk_batch after the
    # redaction-split repair, so a merged chunk reports the merged text.
    # None on a TextChunk built outside chunk_batch.
    token_count: int | None = None


@dataclass
class ChunkInput:
    """One document's input to :func:`chunk_batch`.

    ``narrative`` is the concatenated text of the doc's text-bearing
    elements in ``elem_order``. ``page_breaks`` is a sorted list of
    ``(end_char_exclusive, page_number)`` pairs covering ``narrative``;
    empty for sources without page semantics (DOCX, spreadsheets).
    ``tables`` is one ``(page, elem_order, markdown)`` per table — ``page``
    and ``elem_order`` may both be ``None`` (spreadsheet sheets).
    """

    source_hash: str
    narrative: str
    page_breaks: list[tuple[int, int]] = field(default_factory=list)
    tables: list[tuple[int | None, int | None, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokeniser slot
# ---------------------------------------------------------------------------

DEFAULT_TOKENIZER = "kanon-2-tokenizer"


def _kanon_tokenizer(**_: object) -> str:
    return "isaacus/kanon-2-tokenizer"


def _huggingface_tokenizer(name: str, **_: object) -> str:
    """Any Hugging Face tokeniser, named by option: ``{"name": "org/tok"}``."""
    return name


# A factory returns what ``create_chunker`` accepts: a Hugging Face id (or
# vendored name) string, or a ``(str) -> int`` token counter.
register(
    SLOT_TOKENIZER, DEFAULT_TOKENIZER, _kanon_tokenizer,
    aliases=("isaacus/kanon-2-tokenizer",),
)
register(SLOT_TOKENIZER, "huggingface", _huggingface_tokenizer, aliases=("hf",))


def resolve_tokenizer(
    name: str, options: Mapping[str, Any] | None = None
) -> str | Callable[[str], int]:
    """The registered tokeniser *name* built with *options* passed unchanged."""
    entry = resolve(SLOT_TOKENIZER, name)
    result = entry.factory(**(options or {}))
    record_use(entry)
    return result  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Chunker factory
# ---------------------------------------------------------------------------


def create_chunker(
    tokenizer: str | Callable[[str], int],
    chunk_size: int | None = None,
    *,
    chunking_model: str | None = None,
    isaacus_client: object | None = None,
    tokenizer_kwargs: dict | None = None,
    memoize: bool = True,
    cache_maxsize: int | None = None,
    max_token_chars: int | None = None,
) -> semchunk.Chunker:
    """Construct a semchunk chunker.

    Args:
        tokenizer: HuggingFace tokeniser identifier string, or a callable
            token counter ``(str) -> int``.
        chunk_size: Maximum tokens per chunk. ``None`` passes through to
            semchunk, which derives the size from the tokeniser's
            ``model_max_length`` (only valid for a real tokeniser, not a
            callable token counter).
        chunking_model: Isaacus enrichment model (e.g. ``"kanon-2-enricher"``)
            enabling semchunk 4's AI chunking — boundaries follow the
            enricher's structure spans instead of the purely token/recursive
            split. ``None`` (default) keeps the offline token-based algorithm,
            so non-Kanon callers are unaffected. When set, semchunk calls the
            Isaacus API per document at chunk time (needs the SDK + a key);
            see the AI-chunking note in the module docstring on avoiding a
            second enrichment pass.
        isaacus_client: An ``isaacus.Isaacus`` instance for AI chunking. When
            ``None`` and ``chunking_model`` is set, semchunk constructs one
            from ``ISAACUS_API_KEY``. Ignored unless ``chunking_model`` is set.
        tokenizer_kwargs: Extra keyword arguments forwarded to the tokeniser /
            token counter (semchunk 4).
        memoize: Cache token counts for repeated substrings.
        cache_maxsize: Upper bound on memoization cache entries.
            ``None`` = unbounded.
        max_token_chars: Max chars per token estimate for optimisation.
    """
    if isinstance(tokenizer, str):
        # Prefer a vendored copy (e.g. _models/kanon-2-tokenizer) so chunking is
        # offline — no Hugging Face round-trip; fall back to the hub id when not
        # vendored. A callable token counter is passed through untouched.
        from womblex.utils.models import resolve_local_model_path

        local = resolve_local_model_path(tokenizer.split("/")[-1])
        if not isinstance(local, str):
            tokenizer = str(local)
    return semchunk.chunkerify(
        tokenizer,
        chunk_size=chunk_size,
        chunking_model=chunking_model,
        # `object | None` on the signature keeps this module SDK-free (callers
        # without the isaacus extra pass None); semchunk's stub names the
        # concrete client, so the narrowing happens here at the boundary.
        isaacus_client=cast("Any", isaacus_client),
        tokenizer_kwargs=tokenizer_kwargs,
        memoize=memoize,
        cache_maxsize=cache_maxsize,
        max_token_chars=max_token_chars,
    )


# ---------------------------------------------------------------------------
# Table to markdown
# ---------------------------------------------------------------------------


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Convert table data to a markdown table string."""
    if not headers and not rows:
        return ""

    cols = headers if headers else ([""] * len(rows[0]) if rows else [])
    lines: list[str] = []

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        padded = list(row) + [""] * (len(cols) - len(row))
        lines.append("| " + " | ".join(padded[: len(cols)]) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Redaction-split repair
# ---------------------------------------------------------------------------


def _repair_redaction_splits(chunks: list[TextChunk]) -> list[TextChunk]:
    """Merge chunks where a ``<REDACTED>`` marker was split across a boundary.

    Safe with overlap: if the marker is already complete in both chunks
    (because overlap duplicated it), the repair won't trigger. Only
    fires when one chunk ends with a prefix of ``<REDACTED>`` and the
    next starts with the suffix.
    """
    if not chunks:
        return chunks

    # Cross-file coupling: this literal MUST match the blackout marker emitted
    # by redact/stage.py and stay coherent with the PII tag style in
    # pii/cleaner.py. See docs/decisions.md "PII / redaction marker
    # convention" before changing.
    marker = "<REDACTED>"
    repaired: list[TextChunk] = []

    i = 0
    while i < len(chunks):
        chunk = chunks[i]

        needs_merge = False
        if i + 1 < len(chunks):
            for length in range(1, len(marker)):
                suffix = marker[:length]
                if chunk.text.endswith(suffix) and chunks[i + 1].text.startswith(
                    marker[length:]
                ):
                    needs_merge = True
                    break

        if needs_merge:
            nxt = chunks[i + 1]
            merged_text = chunk.text + nxt.text
            repaired.append(
                TextChunk(
                    text=merged_text,
                    start_char=chunk.start_char,
                    end_char=nxt.end_char,
                    chunk_index=len(repaired),
                    content_type=chunk.content_type,
                    has_redaction=True,
                    page_start=chunk.page_start,
                    page_end=nxt.page_end if nxt.page_end is not None else chunk.page_end,
                    elem_order=chunk.elem_order,
                )
            )
            i += 2
        else:
            chunk.chunk_index = len(repaired)
            repaired.append(chunk)
            i += 1

    return repaired


# ---------------------------------------------------------------------------
# Page lookup
# ---------------------------------------------------------------------------


def _page_for_offset(
    page_breaks: list[tuple[int, int]], offset: int,
) -> int | None:
    """Return the page number covering ``offset``, or ``None`` if past end.

    ``page_breaks`` is sorted by ``end_char_exclusive``; an offset falls
    on the first page whose end exceeds it.
    """
    if not page_breaks:
        return None
    ends = [end for end, _ in page_breaks]
    idx = bisect.bisect_right(ends, offset)
    if idx >= len(page_breaks):
        return None
    return page_breaks[idx][1]


# ---------------------------------------------------------------------------
# Batch chunking — single entry point
# ---------------------------------------------------------------------------


_REDACTED_MARKER = "<REDACTED>"


def chunk_batch(
    inputs: list[ChunkInput],
    chunker: semchunk.Chunker,
    *,
    overlap: float | None = None,
    processes: int = 1,
    progress: bool = False,
    narrative_overrides: dict[str, object] | None = None,
) -> dict[str, list[TextChunk]]:
    """Chunk every doc's narrative + tables in two semchunk calls.

    All narrative texts across ``inputs`` are flattened into one
    ``chunker(...)`` invocation (with ``overlap``); all table markdowns
    into another (no overlap — tables are self-contained). semchunk's
    ``processes`` and ``progress`` arguments therefore parallelise
    across the entire batch, not per document.

    ``narrative_overrides`` maps ``source_hash`` → a pre-enriched semchunk
    input (an ILGS ``Document``) for semchunk-4 AI-chunking reuse. An
    override is used **only** when its ``.text`` is byte-identical to that
    doc's reassembled ``narrative`` (the coordinate-space guard from
    ``docs/decisions.md``); on any mismatch the plain narrative string is
    chunked instead, so a stale or differently-cleaned Document silently
    falls back to self-enrich rather than desyncing offsets. Tables always
    chunk in token mode.

    Returns ``{source_hash: list[TextChunk]}`` with ``chunk_index``
    re-sequenced per doc, ``has_redaction`` populated from the chunk
    text, ``page_start`` / ``page_end`` resolved from per-doc page
    breaks (or the table's page for table chunks), and ``token_count``
    measured by ``chunker.token_counter``.
    """
    if not inputs:
        return {}

    overrides = narrative_overrides or {}
    narrative_texts: list[object] = []  # str | ILGSDocument (mixed; semchunk handles both)
    narrative_owners: list[int] = []
    table_texts: list[str] = []
    table_owners: list[tuple[int, int | None, int | None]] = []

    for i, doc in enumerate(inputs):
        if doc.narrative.strip():
            narrative_texts.append(_resolve_narrative_input(doc, overrides))
            narrative_owners.append(i)
        for page, elem_order, md in doc.tables:
            if md.strip():
                table_texts.append(md)
                table_owners.append((i, page, elem_order))

    out: dict[str, list[TextChunk]] = {doc.source_hash: [] for doc in inputs}

    if narrative_texts:
        n_chunks_all, n_offsets_all = _chunker_batch(
            chunker, narrative_texts,
            overlap=overlap, processes=processes, progress=progress,
        )
        for owner_idx, chunks, offsets in zip(
            narrative_owners, n_chunks_all, n_offsets_all,
        ):
            doc = inputs[owner_idx]
            for text, (start, end) in zip(chunks, offsets):
                out[doc.source_hash].append(
                    _build_narrative_chunk(
                        text=text, start=start, end=end,
                        page_breaks=doc.page_breaks,
                    )
                )

    if table_texts:
        t_chunks_all, t_offsets_all = _chunker_batch(
            chunker, table_texts,
            overlap=None, processes=processes, progress=False,
        )
        for (owner_idx, table_page, table_elem_order), chunks, offsets in zip(
            table_owners, t_chunks_all, t_offsets_all,
        ):
            doc = inputs[owner_idx]
            for text, (start, end) in zip(chunks, offsets):
                out[doc.source_hash].append(
                    TextChunk(
                        text=text,
                        start_char=start,
                        end_char=end,
                        chunk_index=0,
                        content_type="table",
                        has_redaction=_REDACTED_MARKER in text,
                        page_start=table_page,
                        page_end=table_page,
                        elem_order=table_elem_order,
                    )
                )

    for src, doc_chunks in out.items():
        out[src] = _count_tokens(
            _repair_redaction_splits(doc_chunks), chunker.token_counter,
        )

    return out


def _count_tokens(
    chunks: list[TextChunk], token_counter: Callable[[str], int],
) -> list[TextChunk]:
    """Stamp each chunk with its length under the chunker's own counter.

    Runs after ``_repair_redaction_splits`` so a merged chunk reports the
    merged text rather than the sum of two pre-merge counts. ``token_counter``
    is the callable semchunk budgeted with, so the number is the one it
    compared against ``chunk_size``.

    **The count saturates above budget.** ``chunkerify`` derives
    ``max_token_chars`` from the tokeniser's vocabulary whenever it exposes one
    — so for a real tokeniser, not just an explicit
    ``create_chunker(max_token_chars=...)``, the counter it builds short-circuits
    to ``chunk_size + 1`` for text over budget rather than tokenising it
    (measured: a 193-token string counts 65 under a 64-token budget). Every
    chunk semchunk emits is at or under budget, so every count is exact; the
    one chunk that can exceed it is a repaired one, whose count is then a lower
    bound. Hence the rule on the column: **exact at or below ``chunk_size``, a
    floor above it.** Recovering the true figure for a repaired chunk means
    tokenising it again with a second counter, which is the re-estimate this
    column exists not to be.
    """
    for chunk in chunks:
        chunk.token_count = token_counter(chunk.text)
    return chunks


def _resolve_narrative_input(doc: ChunkInput, overrides: dict[str, object]) -> object:
    """Return the semchunk input for ``doc``: a reused Document or the string.

    The byte-identity guard: an override is honoured only when its ``.text``
    equals ``doc.narrative`` exactly (same coordinate space). Otherwise the
    plain narrative string is returned and semchunk self-enriches (if a
    ``chunking_model`` is set) — the per-doc fallback in ``docs/decisions.md``.
    """
    override = overrides.get(doc.source_hash)
    if override is None:
        return doc.narrative
    override_text = getattr(override, "text", None)
    if override_text == doc.narrative:
        return override
    logger.warning(
        "chunk reuse guard: persisted Document.text for %s does not match the "
        "reassembled narrative (len %s vs %s); falling back to self-enrich.",
        doc.source_hash,
        None if override_text is None else len(override_text),
        len(doc.narrative),
    )
    return doc.narrative


def _chunker_batch(
    chunker: semchunk.Chunker,
    texts: Sequence[object],
    *,
    overlap: float | None,
    processes: int,
    progress: bool,
) -> tuple[list[list[str]], list[list[tuple[int, int]]]]:
    """Call semchunk on a list of texts; always returns ``(chunks, offsets)``.

    ``texts`` may mix ``str`` and ILGS ``Document`` items (AI-chunking reuse);
    semchunk dispatches on type at runtime — verified live in docs/decisions.md.
    """
    chunks, offsets = chunker(
        texts,  # type: ignore[arg-type]  # str | Document items; semchunk handles both
        offsets=True,
        overlap=overlap,
        processes=processes,
        progress=progress,
    )
    return cast(
        "tuple[list[list[str]], list[list[tuple[int, int]]]]", (chunks, offsets),
    )


def _build_narrative_chunk(
    *, text: str, start: int, end: int,
    page_breaks: list[tuple[int, int]],
) -> TextChunk:
    page_start = _page_for_offset(page_breaks, start)
    page_end = _page_for_offset(page_breaks, max(start, end - 1))
    return TextChunk(
        text=text,
        start_char=start,
        end_char=end,
        chunk_index=0,
        content_type="narrative",
        has_redaction=_REDACTED_MARKER in text,
        page_start=page_start,
        page_end=page_end,
    )


# ---------------------------------------------------------------------------
# Element-stream reassembly
# ---------------------------------------------------------------------------


def _narrative_pieces(
    elements: list[Element],
) -> Iterator[tuple[Element, str, int, int]]:
    """Yield ``(element, piece, start, end)`` for the narrative-bearing elements.

    The one place the narrative's coordinate space is defined —
    :func:`reassemble_narrative` and :func:`element_spans` are both views
    over it, so the two cannot drift. ``start`` includes the
    ``NARRATIVE_JOIN`` that separates an element from its predecessor.
    """
    cursor = 0
    first = True
    for e in elements:
        if e.kind not in TEXT_KINDS or not e.text:
            continue
        piece = e.text if first else NARRATIVE_JOIN + e.text
        first = False
        start, cursor = cursor, cursor + len(piece)
        yield e, piece, start, cursor


def reassemble_narrative(
    elements: list[Element],
) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate TEXT_KINDS element texts and emit page-break offsets.

    Returns ``(text, [(end_offset_exclusive, page_number), ...])``. An
    offset in ``[prev_end, curr_end)`` resolves to ``curr_page``; offsets
    past the final entry have no page. Elements with no ``page``
    contribute to the text but not to any page span.

    Elements are assumed already sorted by ``order``.
    """
    parts: list[str] = []
    spans: list[list] = []

    for e, piece, start, end in _narrative_pieces(elements):
        parts.append(piece)
        if e.page is None:
            continue
        if spans and spans[-1][2] == e.page:
            spans[-1][1] = end
        else:
            spans.append([start, end, e.page])

    page_breaks = [(end, page) for _, end, page in spans]
    return "".join(parts), page_breaks


def element_spans(elements: list[Element]) -> list[tuple[int, int, int]]:
    """``(elem_order, start, end)`` per element contributing to the narrative.

    The offset map :func:`reassemble_narrative` does not return. It places
    an element in the narrative coordinate space, which is what recovering
    narrative ↔ table document order needs: a table chunk's only positional
    anchor is its source element's ``elem_order``, and nothing else maps
    that to a narrative offset (chunks otherwise join to elements by
    offset overlap, never by ``elem_order``). See
    :func:`chunks_in_document_order`.

    Spans are contiguous, sorted by ``elem_order``, and cover the whole
    narrative; the last ``end`` is its length. Elements are assumed
    already sorted by ``order``, and must be the *same* element stream
    (same ``text_source`` overlay) the narrative was reassembled from —
    a different overlay is a different coordinate space.
    """
    return [(e.order, start, end) for e, _piece, start, end in _narrative_pieces(elements)]


def collect_tables_from_elements(
    elements: list[Element],
) -> list[tuple[int | None, int | None, str]]:
    """Materialise ``(page, elem_order, markdown)`` per table for :func:`chunk_batch`.

    Mirrors :pyattr:`ExtractionResult.tables`: one entry per
    ``kind='table'`` element followed by one synthetic entry per
    spreadsheet sheet. ``page`` and ``elem_order`` are both ``None`` for
    sheets — a sheet aggregates many ``sheet_cell`` elements rather than
    sitting at one position, and a spreadsheet has no narrative to be
    ordered against, so the anchor would be meaningless there.
    """
    return [(t.page, t.elem_order, t.markdown) for t in table_texts(elements)]


@dataclass(frozen=True)
class TableText:
    """One table's markdown with the handle that locates it in the source.

    ``elem_order`` names a ``kind='table'`` element; ``sheet`` names a
    spreadsheet sheet (which aggregates many ``sheet_cell`` elements and has
    no single position). Exactly one of the two is set. ``markdown`` is the
    string the chunker budgets and the enricher reads, so offsets into a table
    chunk and into an enriched table share this coordinate space.
    """

    page: int | None
    elem_order: int | None
    sheet: str | None
    markdown: str


def table_texts(elements: list[Element]) -> list[TableText]:
    """One :class:`TableText` per non-empty table element, then per spreadsheet sheet."""
    out: list[TableText] = []
    for e in elements:
        if e.kind != "table":
            continue
        td = _element_to_table_data(e)
        md = table_to_markdown(td.headers, td.rows)
        if md.strip():
            out.append(TableText(e.page, e.order, None, md))

    for sheet_td in _sheets_to_table_data(elements):
        md = table_to_markdown(sheet_td.headers, sheet_td.rows)
        if md.strip():
            out.append(TableText(None, None, sheet_td.context.get("sheet"), md))
    return out


def build_chunk_input(
    source_hash: str,
    elements: list[Element],
    *,
    include_tables: bool = True,
) -> ChunkInput:
    """Convenience: build a :class:`ChunkInput` from an ordered element list."""
    narrative, page_breaks = reassemble_narrative(elements)
    tables = collect_tables_from_elements(elements) if include_tables else []
    return ChunkInput(
        source_hash=source_hash,
        narrative=narrative,
        page_breaks=page_breaks,
        tables=tables,
    )


# ---------------------------------------------------------------------------
# Document order
# ---------------------------------------------------------------------------


def _table_anchor(elem_order: int, spans: Sequence[tuple[int, int, int]]) -> int:
    """Narrative offset a table element sits at: where the next element begins.

    A table past every narrative element anchors at the narrative's end.
    """
    idx = bisect.bisect_right(spans, elem_order, key=lambda s: s[0])
    if idx < len(spans):
        return spans[idx][1]  # start of the first element after the table
    return spans[-1][2] if spans else 0


def _document_order_key(
    row: Mapping[str, Any], spans: Sequence[tuple[int, int, int]],
) -> tuple[int, int, int, int]:
    """Sort key: (position, rank at that position, element, offset in element).

    Rank orders a table (0) before the narrative chunk (1) it is anchored
    at — the table element precedes that text — and puts sheets (2) last.
    The final slot keeps one table's own chunks in markdown order without
    relying on the caller's input order; sheets have no such handle (the
    chunks sidecar carries nothing that tells one sheet from another), so
    they hold their input order and take 0.
    """
    narrative_end = spans[-1][2] if spans else 0
    if row["content_type"] == "narrative":
        return (row["start_char"], 1, 0, 0)
    elem_order = row["elem_order"]
    if elem_order is None:  # spreadsheet sheet — no narrative to be ordered against
        return (narrative_end, 2, 0, 0)
    return (_table_anchor(elem_order, spans), 0, elem_order, row["start_char"])


def chunks_in_document_order[Row: Mapping[str, Any]](
    chunks: Sequence[Row], spans: Sequence[tuple[int, int, int]],
) -> list[Row]:
    """Sort chunk rows into document order — narrative and tables interleaved.

    The two chunk projections carry their positions in two coordinate
    spaces: a narrative chunk's ``start_char`` indexes the reassembled
    narrative, while a table chunk's only anchor is the ``elem_order`` of
    the element it came from. *spans* — :func:`element_spans` over the
    same element stream, under the same ``text_source`` overlay the chunks
    were produced under — is what makes the two comparable.

    *chunks* are ``CHUNKS_SCHEMA`` rows (``content_type``, ``start_char``,
    ``elem_order``); one document's, since offsets are per source_hash.
    A table anchored at a narrative chunk's start sorts before it — the
    table element precedes that text in the document. Spreadsheet sheets
    (null ``elem_order``) have no narrative to be ordered against and sort
    last, in the order they arrive — nothing in the chunks sidecar tells
    one sheet from another, so pass the rows in ``chunk_index`` order to
    keep each sheet's chunks together. A table chunk read from a shard
    written before the anchor column existed is null too, so it sorts
    there as well — that shard holds no position to recover.
    """
    return sorted(chunks, key=lambda r: _document_order_key(r, spans))

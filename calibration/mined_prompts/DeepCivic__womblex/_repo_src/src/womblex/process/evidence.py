"""Build and check evidence references (:mod:`womblex.store.evidence`).

:class:`EvidenceIndex` holds one document's elements under one element-text
layer and answers two questions about a span: *where is it* (the ``*_ref``
methods return an evidence reference at the level the location supports) and
*does the receipt hold* (:meth:`EvidenceIndex.holds`, at that level's own
precision). The narrative, the element offset map and the table markdowns all
come from ``chunker`` — the one place those coordinate spaces are defined — so
a reference cannot drift from the chunks and mentions it joins to.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from womblex.ingest.elements import BBox, Element
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.chunker import TableText, element_spans, reassemble_narrative, table_texts
from womblex.process.text_overlay import MissingOverlayError, apply_overlay, load_overlay
from womblex.store.evidence import (
    CELL_LAYER,
    CHUNK,
    DOCUMENT,
    ELEMENT,
    FAILS,
    HOLDS,
    SPAN,
    TABLE_LAYER,
    UNCHECKED,
    evidence,
)

logger = logging.getLogger(__name__)


def _bbox(b: BBox | None) -> dict[str, float] | None:
    return None if b is None else {"x": b.x, "y": b.y, "width": b.width, "height": b.height}


class EvidenceIndex:
    """One document's anchors under one element-text layer.

    ``elements`` must already carry the layer's text (``apply_overlay``).
    """

    def __init__(self, elements: list[Element], text_layer: str = "elements") -> None:
        self.text_layer = text_layer
        self.narrative, _ = reassemble_narrative(elements)
        self._spans = element_spans(elements)
        self._by_order = {e.order: e for e in elements}
        self._tables: list[TableText] = table_texts(elements)

    def _located(
        self, level: str, elem: Element | None, *, sheet: str | None = None,
        cell: tuple[int, int] | None = None, bbox: BBox | None = None, **fields: Any,
    ) -> dict[str, Any]:
        return evidence(
            level,
            elem_order=elem.order if elem is not None else None,
            page=elem.page if elem is not None else None,
            bbox=_bbox(bbox or (elem.bbox if elem is not None else None)),
            sheet=sheet,
            cell_row=cell[0] if cell else None, cell_col=cell[1] if cell else None,
            **fields,
        )

    def _cell_box(self, elem: Element, cell: tuple[int, int] | None) -> BBox | None:
        if elem.kind != "table" or cell is None:
            return None
        return next((c.bbox for c in elem.cells or ()
                     if (c.row, c.col) == cell and c.bbox is not None), None)

    def narrative_ref(self, start: int, end: int) -> dict[str, Any] | None:
        """Span evidence for ``narrative[start:end]``; ``None`` outside the narrative."""
        if not 0 <= start <= end <= len(self.narrative):
            return None
        idx = bisect.bisect_right(self._spans, start, key=lambda s: s[1]) - 1
        order = self._spans[idx][0] if idx >= 0 and self._spans else None
        elem = self._by_order.get(order) if order is not None else None
        return self._located(
            SPAN, elem, char_start=start, char_end=end, text_layer=self.text_layer)

    def table_ref(self, table: TableText, start: int, end: int) -> dict[str, Any] | None:
        """Span evidence for ``table.markdown[start:end]``."""
        if not 0 <= start <= end <= len(table.markdown):
            return None
        elem = self._by_order.get(table.elem_order) if table.elem_order is not None else None
        return self._located(
            SPAN, elem, sheet=table.sheet, char_start=start, char_end=end,
            text_layer=TABLE_LAYER)

    def cell_ref(
        self, elem_order: int, start: int, end: int, *,
        cell: tuple[int, int] | None = None, sheet: str | None = None,
    ) -> dict[str, Any] | None:
        """Span evidence for ``[start:end]`` of one cell's value; ``None`` outside it."""
        value = self._cell_text(elem_order, cell)
        if value is None or not 0 <= start <= end <= len(value):
            return None
        elem = self._by_order[elem_order]
        return self._located(
            SPAN, elem, sheet=sheet, cell=cell, bbox=self._cell_box(elem, cell),
            char_start=start, char_end=end, text_layer=CELL_LAYER)

    def element_ref(
        self, elem_order: int, *, cell: tuple[int, int] | None = None, sheet: str | None = None,
    ) -> dict[str, Any] | None:
        """Element evidence: the span lies somewhere in this element (or one of its cells)."""
        elem = self._by_order.get(elem_order)
        if elem is None or (cell is not None and self._cell_text(elem_order, cell) is None):
            return None
        return self._located(
            ELEMENT, elem, sheet=sheet, cell=cell, bbox=self._cell_box(elem, cell))

    def document_ref(self) -> dict[str, Any]:
        """Document evidence: the span lies somewhere in this document."""
        return evidence(DOCUMENT, text_layer=self.text_layer)

    def resolve_table(
        self, *, elem_order: int | None, sheet: str | None = None,
        chunk_text: str | None = None, start: int = 0,
    ) -> TableText | None:
        """The table a chunk or mention belongs to.

        A table element is named by ``elem_order``. A sheet has no element
        position, so it is named by ``sheet`` when known, else found as the sheet
        whose markdown holds ``chunk_text`` at ``start``.
        """
        for t in self._tables:
            if elem_order is not None:
                if t.elem_order == elem_order:
                    return t
            elif t.elem_order is None and (
                (sheet is not None and t.sheet == sheet)
                or (sheet is None and chunk_text is not None
                    and t.markdown[start:start + len(chunk_text)] == chunk_text)
            ):
                return t
        return None

    def source_text(self, ref: Mapping[str, Any]) -> str | None:
        """The whole text a ``span`` reference's offsets index, or ``None`` if not found."""
        layer = ref.get("text_layer")
        if layer == TABLE_LAYER:
            t = self.resolve_table(elem_order=ref.get("elem_order"), sheet=ref.get("sheet"))
            return t.markdown if t is not None else None
        if layer == CELL_LAYER:
            return self._cell_text(ref.get("elem_order"), (ref.get("cell_row"), ref.get("cell_col")))
        return self.narrative

    def _cell_text(self, order: int | None, cell: tuple[Any, Any] | None) -> str | None:
        elem = self._by_order.get(order) if order is not None else None
        if elem is None:
            return None
        if elem.kind == "sheet_cell":
            return elem.value or ""
        return next((c.value or "" for c in elem.cells or () if (c.row, c.col) == cell), None)

    def _element_text(self, ref: Mapping[str, Any]) -> str | None:
        """The text an ``element`` reference confines a span to."""
        order = ref.get("elem_order")
        elem = self._by_order.get(order) if order is not None else None
        if elem is None:
            return None
        if ref.get("cell_row") is not None:
            return self._cell_text(order, (ref.get("cell_row"), ref.get("cell_col")))
        if elem.kind == "table":
            t = self.resolve_table(elem_order=elem.order)
            return t.markdown if t is not None else None
        if elem.kind == "sheet_cell":
            return elem.value or ""
        return elem.text or ""

    def holds(self, ref: Mapping[str, Any], text: str) -> bool | None:
        """Whether the receipt holds at its own precision; ``None`` if it cannot be checked here.

        ``span``: the source slice equals ``text``. ``element``: ``text`` lies in
        that element or cell. ``document``: ``text`` lies in the narrative or a
        table markdown. A ``chunk`` receipt needs the chunk's text, which this
        index does not hold, so it is ``None`` here; so is a narrative offset or
        document receipt under another element-text layer than this index's.
        """
        level = ref.get("anchor_level")
        layer = ref.get("text_layer")
        if layer not in (None, TABLE_LAYER, CELL_LAYER) and layer != self.text_layer:
            return None
        if level == SPAN:
            source = self.source_text(ref)
            start, end = ref.get("char_start"), ref.get("char_end")
            if source is None or start is None or end is None:
                return False
            return source[start:end] == text
        if level == ELEMENT:
            source = self._element_text(ref)
            return source is not None and text in source
        if level == DOCUMENT:
            return text in self.narrative or any(text in t.markdown for t in self._tables)
        return None


class EvidenceIndexes:
    """A batch's per-document indexes, built on demand.

    Elements load once; each ``(source_hash, layer)`` index is built when first
    asked for, applying that layer's overlay. ``cell`` and ``table_markdown``
    references read the verbatim layer — overlays only rewrite text kinds.
    """

    def __init__(self, base_path: Path, text_source: str = "elements") -> None:
        self.base_path = base_path
        self.text_source = text_source
        try:
            self._elements = _load_elements(base_path)
        except FileNotFoundError:
            logger.warning(
                "evidence: %s has no elements or table-cells sidecar; its spans cannot be "
                "located or checked beyond the chunk", base_path.stem)
            self._elements = {}
        self._overlays: dict[str, dict[tuple[str, int], str] | None] = {}
        self._cache: dict[tuple[str, str], EvidenceIndex] = {}

    def get(self, source_hash: str, layer: str | None = None) -> EvidenceIndex | None:
        layer = layer or self.text_source
        if layer in (CELL_LAYER, TABLE_LAYER):
            layer = "elements"
        key = (source_hash, layer)
        if key not in self._cache:
            elements = self._elements.get(source_hash)
            if elements is None:
                return None
            if layer not in self._overlays:
                self._overlays[layer] = load_overlay(self.base_path, layer, required=True)
            elems = [replace(e) for e in elements]
            apply_overlay(source_hash, elems, self._overlays[layer])
            self._cache[key] = EvidenceIndex(elems, layer)
        return self._cache[key]


#: Rows named in the warning; the counts are always complete.
_REPORT_LIMIT = 5

#: ``(source_hash, text_layer)`` -> that document's index, or ``None`` if its elements are absent.
IndexLookup = Callable[[str, str | None], EvidenceIndex | None]
#: A ``chunk`` row's chunk text, or ``None`` if the chunk is not to hand.
ChunkText = Callable[[Mapping[str, Any]], str | None]


@dataclass
class EvidenceReport:
    """Outcome of checking receipts against their sources.

    ``unchecked`` counts receipts whose source or text is absent; ``n_mismatched``
    those that fail their own check. Both are written and labelled, never refused.
    """

    checked: int = 0
    unchecked: int = 0
    n_mismatched: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_mismatched == 0

    def merge(self, other: EvidenceReport) -> None:
        self.checked += other.checked
        self.unchecked += other.unchecked
        self.n_mismatched += other.n_mismatched
        self.mismatches.extend(other.mismatches[: _REPORT_LIMIT - len(self.mismatches)])


def check_rows(
    rows: Iterable[MutableMapping[str, Any]], lookup: IndexLookup, *, text_key: str, label: str,
    chunk_text: ChunkText | None = None,
) -> EvidenceReport:
    """Check each row's receipt at its own precision and label it in ``evidence_check``.

    A row without a valid ``anchor_level`` is a null receipt and fails. A
    ``chunk`` receipt is checked against ``chunk_text(row)``; every other level
    against the document's index. A missing source makes the row unchecked, as
    does a missing text (e.g. ``mention_text`` on files before contract 1.5):
    empty text lies in any source, so it cannot pass a check.
    """
    report = EvidenceReport()
    for row in rows:
        text = row[text_key] or ""
        level = row.get("anchor_level")
        verdict: bool | None
        if not text and level in (SPAN, ELEMENT, CHUNK, DOCUMENT):
            verdict = None
        elif level not in (SPAN, ELEMENT, CHUNK, DOCUMENT):
            verdict = False
        elif level == CHUNK:
            source = chunk_text(row) if chunk_text is not None else None
            verdict = None if source is None else text in source
        else:
            index = lookup(row["source_hash"], row.get("text_layer"))
            verdict = None if index is None else index.holds(row, text)
        row["evidence_check"] = UNCHECKED if verdict is None else HOLDS if verdict else FAILS
        if verdict is None:
            report.unchecked += 1
            continue
        report.checked += 1
        if not verdict:
            report.n_mismatched += 1
            if len(report.mismatches) < _REPORT_LIMIT:
                report.mismatches.append(
                    f"{label} {row['source_hash'][:12]} [{level}:{row.get('text_layer')}:"
                    f"{row.get('char_start')}-{row.get('char_end')}] {text[:40]!r}"
                )
    return report


def stamp_evidence(
    rows: list[dict[str, Any]], lookup: IndexLookup, *, text_key: str, label: str, base: Path,
    chunk_text: ChunkText | None = None,
) -> EvidenceReport:
    """Check every receipt, label each row's ``evidence_check`` and warn about the rest.

    Called before the sidecar is written. Nothing is refused: a receipt that
    fails, or cannot be checked, is written with its label so the run completes
    and says what it could not verify.
    """
    report = check_rows(rows, lookup, text_key=text_key, label=label, chunk_text=chunk_text)
    if report.unchecked:
        logger.warning(
            "%s: %s has %d receipt(s) whose source or text is absent; written as unchecked",
            label, base.stem, report.unchecked,
        )
    if not report.ok:
        logger.warning(
            "%s: %s has %d of %d receipt(s) that fail their own check; written as fails."
            " First: %s",
            label, base.stem, report.n_mismatched, report.checked + report.unchecked,
            "; ".join(report.mismatches),
        )
    return report


def chunk_text_from(texts: Mapping[tuple[str, int], str]) -> ChunkText:
    """A :data:`ChunkText` over ``{(source_hash, chunk_index): text}``."""
    return lambda row: texts.get((row["source_hash"], row["chunk_index"]))


def _lookup_in(indexes: EvidenceIndexes) -> IndexLookup:
    """``indexes.get``, with a text layer whose sidecar is missing read as no index."""
    def lookup(source_hash: str, layer: str | None) -> EvidenceIndex | None:
        try:
            return indexes.get(source_hash, layer)
        except MissingOverlayError:
            return None
    return lookup


def verify_shards(shard_dir: Path) -> EvidenceReport:
    """Re-check every money, entity-link and PII receipt in ``shard_dir`` against its source.

    Each row's check is recomputed, so a file written before ``evidence_check``
    existed, or whose source has changed since, is judged afresh. A receipt whose
    source (the elements, a text layer, the chunk) is absent counts as unchecked.
    """
    from womblex.store.entity_links_output import read_entity_links
    from womblex.store.money_output import read_money_spans
    from womblex.store.output import read_chunks
    from womblex.store.pii_output import read_pii_spans

    readers: tuple[tuple[str, str, Callable[[Path], Any]], ...] = (
        ("money_spans", "text", read_money_spans),
        ("entity_links", "mention_text", read_entity_links),
        ("pii_spans", "text", read_pii_spans),
    )
    total = EvidenceReport()
    for base in _batch_bases(shard_dir):
        lookup: IndexLookup | None = None
        for label, text_key, read in readers:
            try:
                rows = read(base).to_pylist()
            except FileNotFoundError:
                continue
            lookup = lookup or _lookup_in(EvidenceIndexes(base))
            texts: dict[tuple[str, int], str] = {}
            if label == "pii_spans":
                try:
                    texts = {(r["source_hash"], r["chunk_index"]): r["text"] or ""
                             for r in read_chunks(base).to_pylist()}
                except FileNotFoundError:
                    pass
            total.merge(check_rows(
                rows, lookup, text_key=text_key, label=f"{label}:{base.stem}",
                chunk_text=chunk_text_from(texts)))
    return total

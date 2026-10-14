"""Reviewer-facing rendering of a ground-truth segment.

A *segment* (:mod:`womblex.process.segmenter`) is a contiguous run of
elements from one source document; the renderer turns that run into the
markdown a human reads and corrects. Ground truth is keyed to element
identity, not to this rendering — reading order and formatting are
presentation and are regenerable, the corrected text is not — so the
renderer is a pure, order-preserving projection of the element stream,
inventing nothing beyond a closed set of structural delimiters.

The closed structural set is exactly three things and nothing else:

- one **blank line** between elements,
- one **GitHub-flavoured markdown table** (pipe rows plus one header
  separator row) per ``table`` element, and
- one ``label: value`` line per field of a ``form`` element.

Narrative-bearing elements (:data:`~womblex.ingest.elements.TEXT_KINDS`)
render verbatim via the same reassembly the chunker uses
(:func:`~womblex.process.chunker.reassemble_narrative`), so a reviewed
unit and a chunked one cannot drift into two coordinate spaces. Tables
and forms reuse :func:`womblex.process.segmenter.element_text` — the very
projection a segment's token budget is measured over (GFM markdown for a
table, ``label: value`` lines for a form) — so a segment is rendered in
the shape it was measured in, with no second copy of that logic to drift.
Tables and forms are interleaved with narrative at their real position in
the element stream (elements arrive in ``order``, so a walk of the stream
*is* document order), never appended. Images, page breaks and spreadsheet
cells are outside this closed set and contribute nothing.

Guarantees held here, each pinned by a test:

- **Order.** Rendered output follows element order exactly — narrative,
  tables and forms alike. :func:`rendered_order` returns the ``order``
  sequence as it appears in the output, which a test compares against the
  input sequence.
- **Invent nothing.** The only bytes emitted beyond element / cell / field
  content are the closed structural set above. A test strips that set from
  the rendered output and asserts the remainder is a subsequence of the
  element text, so any other invented character fails.
- **Verbatim.** Redaction markers present in the element stream — in
  narrative, a table cell, or a form field — survive unchanged because
  nothing here rewrites text.
- **Determinism.** The render is byte-identical on re-run — in one process
  and across processes under a randomised hash seed — because it is a pure
  projection over an ordered list that iterates no mapping in output order
  (:func:`~womblex.ingest.views._element_to_table_data` orders its rows with
  ``sorted``) and reads no clock, path or process state. Cell and field
  values are ``str`` by schema, so table numbers render verbatim with no
  default-``repr`` path. Pinned by the determinism tests (U4).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from womblex.ingest.elements import Element
from womblex.process.chunker import NARRATIVE_JOIN, element_spans, reassemble_narrative
from womblex.process.segmenter import element_text

#: The renderer's own version, stamped onto a ground-truth unit's sidecar
#: (``derivation.renderer_version``). Bump it whenever the closed structural set
#: or the projection would change the bytes emitted for an unchanged element
#: stream — the signal that a re-derived baseline may differ from a corrected one.
RENDERER_VERSION = "renderer-1"

__all__ = ["RENDERER_VERSION", "baseline_digest", "render_elements", "rendered_order"]


def baseline_digest(text: str) -> str:
    """``sha256:…`` over the baseline bytes a reviewer corrects.

    Digested as UTF-8, the same encoding the baseline is written in, so the
    stored digest is over exactly the bytes on disk. The ``sha256:`` prefix
    matches the run stamp's ``config_digest`` convention, so a reader can tell
    the algorithm from the value without a second field.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_elements(elements: Sequence[Element]) -> str:
    """Render an ordered element run to reviewer-facing markdown.

    ``elements`` is a segment's element slice, assumed already sorted by
    ``order`` (a segment's range is contiguous and ordered by
    construction). Narrative-bearing elements render verbatim, tables as
    GitHub-flavoured markdown and forms as ``label: value`` lines, each
    separated from its neighbour by one blank line and interleaved at its
    real position in the stream. Kinds outside the closed structural set
    (images, page breaks, sheet cells) contribute nothing.

    The narrative projection is :func:`reassemble_narrative`'s text and the
    table projection is :func:`table_to_markdown` — reused rather than
    reimplemented, so the bytes a reviewer corrects are the bytes the
    chunker and the segmenter's budget see.
    """
    pieces: list[str] = []
    run: list[Element] = []
    for e in elements:
        structural = _structural_piece(e)
        if structural is None:
            # Narrative or a kind outside the closed set; reassemble_narrative
            # keeps the former and drops the latter, so accumulate and let it
            # decide rather than filtering here.
            run.append(e)
            continue
        _flush_narrative(run, pieces)
        run = []
        pieces.append(structural)
    _flush_narrative(run, pieces)
    return NARRATIVE_JOIN.join(pieces)


def rendered_order(elements: Sequence[Element]) -> list[int]:
    """The element ``order`` values in the sequence they appear in the output.

    The handle a reading-order check compares against the source element
    sequence without parsing the rendered markdown back apart: the
    ``order`` of each element that contributes to the output, in output
    order. That is every narrative-bearing element with text plus every
    table / form element that renders a non-empty piece. Elements that
    contribute nothing (empty text kinds, images, page breaks, sheet
    cells, an empty table) do not appear, exactly as they do not in the
    output.

    Contribution here uses the same predicates as :func:`render_elements`
    — :func:`element_spans` for narrative, :func:`_structural_piece` for
    tables and forms — so the two cannot disagree about what was rendered.
    """
    ordered = list(elements)
    narrative_orders = {order for order, _start, _end in element_spans(ordered)}
    out: list[int] = []
    for e in ordered:
        if e.order in narrative_orders or _structural_piece(e) is not None:
            out.append(e.order)
    return out


def _flush_narrative(run: list[Element], pieces: list[str]) -> None:
    """Append the narrative reassembly of ``run`` to ``pieces`` if non-empty."""
    if not run:
        return
    text, _page_breaks = reassemble_narrative(run)
    if text:
        pieces.append(text)


def _structural_piece(element: Element) -> str | None:
    """The markdown a table or form element contributes, or ``None``.

    Reuses :func:`~womblex.process.segmenter.element_text` — the very
    projection a segment's token budget is measured over — so a table or
    form renders in the shape it was budgeted in, with no second copy of
    the table-to-markdown or ``label: value`` logic to drift. ``None`` for
    every other kind (narrative is rendered via :func:`reassemble_narrative`;
    images, page breaks and sheet cells are outside the closed set) and for
    a table or form that projects to nothing (no cells, no fields), which
    then contributes no piece and no blank line.
    """
    if element.kind in ("table", "form"):
        text = element_text(element)
        return text if text.strip() else None
    return None

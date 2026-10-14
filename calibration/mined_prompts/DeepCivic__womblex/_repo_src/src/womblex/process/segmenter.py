"""Ground-truth segmentation: element ranges under a token budget.

A *segment* is a contiguous run of elements from one source document,
small enough for a human to review in one sitting. It is the unit
ground truth is keyed to, so its boundaries must be a function of the
element stream and the configuration alone — never of a stage's output.
Deriving them from chunk boundaries would make the ground-truth unit a
function of the chunker, and the chunker is one of the things being
measured against it.

Two constraints, applied in that order:

- a **page ceiling** partitions the stream into runs no wider than N
  pages (a reviewer holds a handful of pages in view, not forty), and
- a **token budget** packs each run into segments via
  :func:`womblex.utils.token_packer.pack_by_tokens` — the same greedy
  grouping the Isaacus request packer uses, including its solo path for
  an item that does not fit on its own.

Splits happen only between elements, so a table and its cells, or a
form and its fields, are always wholly inside one segment: they nest
within a single :class:`~womblex.ingest.elements.Element`.

Segments of one document tile it — gapless and non-overlapping — which
is what lets a future re-cut of boundaries be arithmetic over existing
corrections rather than a re-review. That rests on one document's stream
being contiguous, so a filtered stream is refused rather than quietly
segmented into ranges that do not tile.

The budget is in tokens because that is what the packer counts; the
human-facing word count is recorded alongside rather than converted at
the boundary, so no rounding sits between the two.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

from womblex.config import SegmentationConfig
from womblex.ingest.elements import TEXT_KINDS, Element
from womblex.ingest.views import _element_to_table_data
from womblex.process.chunker import table_to_markdown
from womblex.utils.token_packer import pack_by_tokens

CountFn = Callable[[list[str]], list[int]]

# One `label: value` line per form field — the form projection the
# reviewer-facing renderer is specified to emit, so a segment is budgeted
# against the shape a reviewer is handed rather than a second one.
FIELD_JOIN = "\n"


@dataclass(frozen=True, slots=True)
class Segment:
    """One reviewable run of elements, and what it measured.

    ``element_range`` and ``page_range`` are half-open, over ``elem_order``
    and zero-based page indices respectively.

    ``page_range`` is ``None`` when no element in the segment carries a
    page — a fact about the segment, not the source. A paged document can
    hold page-less elements (spreadsheet-print manifest tables span pages
    and so anchor to none), so this does not mean the source is unpaged;
    a caller filling a ground-truth sidecar, whose ``page_range`` separates
    "no pages exist" from "none was produced", decides that from the whole
    document.

    ``oversize`` marks the one segment shape the budget cannot honour: a
    single element whose own token count exceeds the budget, emitted alone
    because there is no boundary inside an element to split at.
    """

    element_range: tuple[int, int]
    page_range: tuple[int, int] | None
    tokens: int
    words: int
    oversize: bool


def element_text(element: Element) -> str:
    """The text an element contributes to its segment's measured size.

    Narrative kinds contribute verbatim text, a table its markdown and a
    form its ``label: value`` lines, so a segment is budgeted against what
    a reviewer reads. Structural kinds (page breaks, sheet metadata)
    contribute nothing but are still carried — dropping them would gap the
    tiling.
    """
    if element.kind in TEXT_KINDS:
        return element.text or ""
    if element.kind == "table":
        td = _element_to_table_data(element)
        return table_to_markdown(td.headers, td.rows)
    if element.kind == "form":
        return FIELD_JOIN.join(f"{f.name}: {f.value}" for f in element.fields or [])
    if element.kind in ("image", "figure"):
        return element.alt_text or ""
    if element.kind == "sheet_cell":
        return element.value or ""
    return ""


def segment_elements(
    elements: Sequence[Element],
    count_fn: CountFn,
    config: SegmentationConfig,
) -> list[Segment]:
    """Cut an ordered element stream into budgeted, page-bounded segments.

    ``count_fn`` maps texts to exact token counts (e.g.
    ``TokenCounter().count_batch``). Passed in rather than constructed here,
    so the caller picks the tokeniser its budget is expressed in and
    segmentation stays offline-testable.

    Raises ``ValueError`` if the stream is not one document's elements,
    contiguous and in order (segment ranges would not tile), or if an
    element exceeds the budget on its own under ``oversize='error'``.
    """
    ordered = list(elements)
    _check_contiguous(ordered)
    if not ordered:
        return []

    by_order = {e.order: e for e in ordered}
    segments: list[Segment] = []

    for run in _page_runs(ordered, config.page_ceiling):
        items = [(str(e.order), element_text(e)) for e in run]
        for group in pack_by_tokens(
            items, count_fn, max_items=len(items), token_budget=config.token_budget
        ):
            members = [by_order[int(item.key)] for item in group.items]
            oversize = group.total_tokens > config.token_budget
            if oversize and config.oversize == "error":
                raise ValueError(
                    f"element {members[0].order} ({members[0].kind}) is "
                    f"{group.total_tokens} tokens, over the {config.token_budget}-token "
                    "budget, and has no internal boundary to split at"
                )
            segments.append(
                Segment(
                    element_range=(members[0].order, members[-1].order + 1),
                    page_range=_page_range(members),
                    tokens=group.total_tokens,
                    words=sum(len(item.text.split()) for item in group.items),
                    oversize=oversize,
                )
            )

    return segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_contiguous(elements: Sequence[Element]) -> None:
    """Refuse a stream whose segments could not tile the document.

    Every producer increments ``order`` only on append, so one document's
    stream is contiguous: a gap means the caller filtered it, and segments
    cut from a subset report ranges that silently fail to tile.
    """
    for prev, curr in pairwise(elements):
        if curr.order <= prev.order:
            raise ValueError(
                "element stream must be sorted by strictly increasing order; "
                f"saw {prev.order} then {curr.order}"
            )
        if curr.order != prev.order + 1:
            raise ValueError(
                f"element stream must be contiguous; order jumps {prev.order} "
                f"to {curr.order}. Segment the document's whole element stream, "
                "not a filtered subset — segments of a subset cannot tile it"
            )


def _page_runs(elements: Sequence[Element], page_ceiling: int) -> Iterator[list[Element]]:
    """Partition into runs spanning at most ``page_ceiling`` pages.

    A page-less element never widens the span, so an unpaged source yields
    one run, bounded by the token budget alone.
    """
    run: list[Element] = []
    low: int | None = None
    high: int | None = None

    for e in elements:
        if e.page is not None:
            new_low = e.page if low is None else min(low, e.page)
            new_high = e.page if high is None else max(high, e.page)
            if run and new_high - new_low + 1 > page_ceiling:
                yield run
                run, low, high = [], e.page, e.page
            else:
                low, high = new_low, new_high
        run.append(e)

    if run:
        yield run


def _page_range(members: Sequence[Element]) -> tuple[int, int] | None:
    pages = [e.page for e in members if e.page is not None]
    if not pages:
        return None
    return (min(pages), max(pages) + 1)


__all__ = ["Segment", "element_text", "segment_elements"]

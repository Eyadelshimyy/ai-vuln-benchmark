"""Spatial text reconstruction via column projection.

Reconstructs multi-column reading order from per-word bounding boxes by
projecting x-coordinates onto a 1D occupancy histogram, identifying
vertical gutters (sustained empty bands), and segmenting the page into
columns.  Within each column, words are clustered into lines and rendered
to a whitespace-aligned character grid.

The algorithm is purely geometric — no model, no training data — and
operates directly on the seam's ``page.words()`` tuples.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from womblex.ingest.pdf.types import Word

if TYPE_CHECKING:
    from womblex.ingest.pdf.types import Page, Rect

logger = logging.getLogger(__name__)

# Only the first five positions of a word are used; trailing fields
# (block_no, line_no, word_no) are ignored, so callers may pass the seam's
# full `Word` unchanged.
WordTuple = tuple[float, float, float, float, str] | Word


@dataclass
class ColumnRegion:
    """A vertical column of text reconstructed from word positions.

    Words are stored sorted top-to-bottom (then left-to-right within ties),
    ready for line clustering during rendering.
    """

    x0: float
    x1: float
    words: list[WordTuple] = field(default_factory=list)


def project_to_columns(
    words: Sequence[WordTuple],
    page_width: float,
    min_gutter_width: float = 0.02,
    edge_margin: float = 0.05,
    min_words: int = 10,
) -> list[ColumnRegion]:
    """Segment words into vertical columns by projecting onto an x-axis histogram.

    Args:
        words: word tuples ``(x0, y0, x1, y1, text, ...)``.
        page_width: Page width in PDF points.
        min_gutter_width: Minimum gutter width as a fraction of page width to
            qualify as a column boundary (0.02 ≈ 12 pt on letter-size pages).
        edge_margin: Outer margin (fraction of page width) where unoccupied
            runs are treated as page margins, not column gutters.
        min_words: Below this count the page is treated as too sparse for
            reliable column detection and a single column is returned.

    Returns:
        Columns left-to-right.  Single-column or sparse pages return one region.
        Empty input returns an empty list.
    """
    if not words or page_width <= 0:
        return []

    if len(words) < min_words:
        return [_build_column(words, 0.0, float(page_width))]

    bins = max(int(page_width), 1)
    occupied = np.zeros(bins, dtype=bool)
    for w in words:
        b0 = max(0, int(w[0]))
        b1 = min(bins, int(w[2]) + 1)
        if b1 > b0:
            occupied[b0:b1] = True

    min_gutter_px = max(2, int(min_gutter_width * page_width))
    edge_px = int(edge_margin * page_width)

    # Find runs of unoccupied bins via np.diff on the boolean mask.
    transitions = np.diff(occupied.astype(np.int8))
    gap_starts = np.where(transitions == -1)[0] + 1
    gap_ends = np.where(transitions == 1)[0] + 1
    if not occupied[0]:
        gap_starts = np.insert(gap_starts, 0, 0)
    if not occupied[-1]:
        gap_ends = np.append(gap_ends, bins)

    gutters: list[tuple[int, int]] = []
    for g_start, g_end in zip(gap_starts, gap_ends):
        run_len = g_end - g_start
        if run_len >= min_gutter_px and g_start > edge_px and g_end < bins - edge_px:
            gutters.append((int(g_start), int(g_end)))

    if not gutters:
        return [_build_column(words, 0.0, float(page_width))]

    columns: list[ColumnRegion] = []
    prev_end = 0.0
    for g_start, g_end in gutters:
        columns.append(_build_column(words, prev_end, float(g_start)))
        prev_end = float(g_end)
    columns.append(_build_column(words, prev_end, float(page_width)))

    return [c for c in columns if c.words]


def _build_column(
    words: Sequence[WordTuple], x0: float, x1: float,
) -> ColumnRegion:
    """Build a ColumnRegion containing words whose midpoint x falls in [x0, x1)."""
    in_col = [w for w in words if x0 <= (w[0] + w[2]) / 2 < x1]
    in_col.sort(key=lambda w: (w[1], w[0]))
    return ColumnRegion(x0=x0, x1=x1, words=in_col)


def render_spatial_text(columns: list[ColumnRegion]) -> str:
    """Render columns as whitespace-aligned text in left-to-right reading order.

    Each column is rendered independently; columns are joined by a blank line
    so downstream chunkers treat them as separate sections.
    """
    parts = [_render_column(c) for c in columns if c.words]
    return "\n\n".join(p for p in parts if p)


def _render_column(col: ColumnRegion) -> str:
    """Render a single column to whitespace-aligned text."""
    if not col.words:
        return ""

    heights = [w[3] - w[1] for w in col.words if w[3] > w[1]]
    line_tol = median(heights) * 0.6 if heights else 5.0

    char_widths: list[float] = []
    for w in col.words:
        text_len = len(w[4])
        if text_len > 0 and w[2] > w[0]:
            char_widths.append((w[2] - w[0]) / text_len)
    char_w = median(char_widths) if char_widths else 5.0

    lines: list[list[WordTuple]] = []
    for w in col.words:
        wy_center = (w[1] + w[3]) / 2
        if lines:
            last_line = lines[-1]
            last_y = sum((lw[1] + lw[3]) / 2 for lw in last_line) / len(last_line)
            if abs(wy_center - last_y) <= line_tol:
                last_line.append(w)
                continue
        lines.append([w])

    rendered: list[str] = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        buf: list[str] = []
        cursor = 0
        for w in line:
            target = max(0, int((w[0] - col.x0) / char_w))
            if target > cursor:
                buf.append(" " * (target - cursor))
                cursor = target
            buf.append(w[4])
            cursor += len(w[4])
            buf.append(" ")
            cursor += 1
        rendered.append("".join(buf).rstrip())

    return "\n".join(rendered)


def extract_page_text(
    page: Page,
    *,
    exclude_rects: Sequence[Rect] | None = None,
) -> str:
    """Extract page text, using grid projection for multi-column layouts.

    For single-column or sparse pages, emits the page's blocks joined with
    blank lines so paragraph breaks survive — `page.get_text("text")`
    flat-joins blocks with single newlines and loses paragraph structure
    that the block_type classifier later needs.

    ``exclude_rects`` (when supplied) drops words whose midpoint falls
    inside any rect, and excludes any block whose bbox intersects them.
    Used by the orchestrator's native path to splice out detected-table
    regions before prose emission so cells aren't read row-major.
    """
    page_width = page.rect.width
    words = page.words()
    if not words:
        return ""

    if exclude_rects:
        words = [w for w in words if not _word_in_any_rect(w, exclude_rects)]
        if not words:
            return ""

    columns = project_to_columns(words, page_width)
    if len(columns) >= 2:
        return render_spatial_text(columns)
    return _render_blocks_with_breaks(page, exclude_rects=exclude_rects)


def _word_in_any_rect(
    word: WordTuple, rects: Sequence[Rect],
) -> bool:
    """Test if a word's midpoint falls inside any of the given rects."""
    cx = (word[0] + word[2]) / 2
    cy = (word[1] + word[3]) / 2
    for r in rects:
        if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1:
            return True
    return False


def _render_blocks_with_breaks(
    page: Page, *, exclude_rects: Sequence[Rect] | None = None,
) -> str:
    """Render single-column page text block-by-block with ``\\n\\n`` separators.

    `page.text_blocks()` is already split at paragraph-shaped boundaries;
    joining with blank lines preserves the structure that the downstream
    block-type classifier later annotates.
    """
    parts: list[str] = []
    for block in page.text_blocks():
        if block.kind != "text":
            continue
        if exclude_rects:
            # Drop block if its centre falls inside any exclusion rect —
            # using centre rather than full overlap allows narrow text
            # bands (e.g. the "Section / 165(1)" cell-internal lines)
            # to still drop while leaving adjacent prose intact.
            cx = (block.bbox.x0 + block.bbox.x1) / 2
            cy = (block.bbox.y0 + block.bbox.y1) / 2
            if any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in exclude_rects):
                continue
        text = block.text.rstrip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)

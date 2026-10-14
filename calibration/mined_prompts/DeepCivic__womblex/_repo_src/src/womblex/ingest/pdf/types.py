"""Backend-neutral PDF types — the seam every extractor is written against.

No third-party imports at runtime, deliberately: importing this module must not
pull in a PDF backend or numpy. ``numpy`` appears under ``TYPE_CHECKING`` only,
for ``Page.render``'s annotation.

Two conventions the adapters own, so no caller has to:

- **Coordinates** are PDF points, origin at the page's top-left, y growing
  downward (MuPDF's convention). A backend whose native convention differs —
  pdfium's y grows upward — converts in its adapter.
- **Shapes are typed, not backend-shaped.** ``text_dict`` yields ``Block``
  rather than a nested dict, ``find_tables`` yields ``FoundTable`` rather than
  something with an ``extract()`` method, ``render`` yields an array rather
  than a pixmap. ``Word`` is the one exception: it stays tuple-shaped because
  ``grid_projection`` slices it positionally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

#: A 2-D affine transform in PDF order: ``(a, b, c, d, e, f)``.
Matrix = tuple[float, float, float, float, float, float]

#: "lines" follows ruled borders; "text" infers a grid from alignment and
#: over-fires, which is why callers post-filter it.
TableStrategy = Literal["lines", "text"]
DrawingKind = Literal["fill", "stroke", "fill_stroke"]
BlockKind = Literal["text", "image"]


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in points, normalised so x0<=x1 and y0<=y1."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def transform(self, matrix: Matrix) -> Rect:
        """Apply an affine transform, re-normalising the corners.

        A rotation swaps which corner is which, so the transformed corners are
        sorted back into a normalised rectangle rather than left in place.
        """
        a, b, c, d, e, f = matrix
        xs, ys = [], []
        for x, y in ((self.x0, self.y0), (self.x1, self.y0), (self.x0, self.y1), (self.x1, self.y1)):
            xs.append(a * x + c * y + e)
            ys.append(b * x + d * y + f)
        return Rect(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def of(cls, values: tuple[float, float, float, float]) -> Rect:
        """Build from a 4-tuple, normalising the corners."""
        x0, y0, x1, y1 = values
        return cls(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


#: MuPDF's ``fz_round_rect`` tolerance: an edge within this of a whole pixel
#: rounds to it, so 595.2pt at 150dpi is 1240px rather than 1241.
_ROUND_TOLERANCE = 0.001


def render_box(page: Rect, dpi: int, clip: Rect | None = None) -> tuple[int, int, int, int]:
    """The pixel box ``(x0, y0, x1, y1)`` a render of *page* at *dpi* covers.

    MuPDF's rule (``fz_round_rect``), which every backend reproduces so a
    render's shape does not depend on the backend: the page is rounded up to
    whole pixels and a clip is rounded outward, both within a 0.001px
    tolerance, then cut to the page. A clip off the page leaves an empty axis
    (``x1 <= x0`` or ``y1 <= y0``) rather than raising.
    """
    scale = dpi / 72
    width, height = _round_up(page.width * scale), _round_up(page.height * scale)
    if clip is None:
        return 0, 0, width, height
    return (
        max(0, _round_down(clip.x0 * scale)), max(0, _round_down(clip.y0 * scale)),
        min(width, _round_up(clip.x1 * scale)), min(height, _round_up(clip.y1 * scale)),
    )


def _round_down(value: float) -> int:
    return math.floor(value + _ROUND_TOLERANCE)


def _round_up(value: float) -> int:
    return math.ceil(value - _ROUND_TOLERANCE)


class Word(NamedTuple):
    """One word and its box, shaped like MuPDF's 8-tuple.

    Positional on purpose: `grid_projection` indexes and slices words, so
    keeping the layout leaves that code untouched by the seam.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_no: int
    line_no: int
    word_no: int


@dataclass(frozen=True)
class Span:
    """A run of characters sharing one font and size."""

    text: str
    bbox: Rect
    size: float
    font: str
    bold: bool


@dataclass(frozen=True)
class Line:
    """One line of text, as its spans."""

    bbox: Rect
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class Block:
    """A block of a page's text.

    The two passes report different granularity, so they fill this
    differently: ``text_dict`` fills ``lines``, ``text_blocks`` fills ``text``
    with the block's paragraph text. ``kind`` is ``"image"`` for the image
    blocks MuPDF interleaves into its dict output; those carry neither.
    """

    bbox: Rect
    number: int
    kind: BlockKind = "text"
    text: str = ""
    lines: tuple[Line, ...] = ()


@dataclass(frozen=True)
class FoundTable:
    """A table a backend's finder located.

    ``row_count``/``col_count`` are the finder's own counts, not derived from
    ``rows``, because callers gate on them before reading cells.
    """

    bbox: Rect
    row_count: int
    col_count: int
    rows: tuple[tuple[str | None, ...], ...] = ()
    cell_boxes: tuple[tuple[Rect | None, ...], ...] = ()


@dataclass(frozen=True)
class Drawing:
    """One vector drawing operation. ``fill`` is RGB in 0..1, or None."""

    kind: DrawingKind
    rect: Rect
    fill: tuple[float, float, float] | None = None

    @property
    def filled(self) -> bool:
        return self.kind in ("fill", "fill_stroke")


@dataclass(frozen=True)
class Widget:
    """An AcroForm field."""

    field_name: str
    field_value: str
    rect: Rect


@dataclass(frozen=True)
class PageImage:
    """One drawn instance of an image.

    An *instance*: an image placed twice on a page yields two of these.
    ``xref`` identifies the shared resource behind them, so a caller wanting
    resources rather than draws groups by it instead of counting rows. MuPDF's
    own ``get_images()`` conflates the two — it lists one entry per draw, all
    carrying the same xref — which is the trap this type exists to close.
    """

    rect: Rect
    xref: int = 0


class Page(Protocol):
    """One page, as the extractors consume it."""

    @property
    def number(self) -> int:
        """Zero-based index within the document."""
        ...

    @property
    def rect(self) -> Rect: ...

    @property
    def rotation(self) -> int:
        """Degrees: 0, 90, 180 or 270."""
        ...

    @property
    def rotation_matrix(self) -> Matrix:
        """Maps unrotated coordinates onto the rotated page."""
        ...

    @property
    def doc_name(self) -> str:
        """The containing document's name, for logs and errors."""
        ...

    def plain_text(self, *, dehyphenate: bool = True) -> str:
        """The page's text in reading order."""
        ...

    def text_dict(self) -> list[Block]:
        """Blocks with their lines and spans, whitespace preserved."""
        ...

    def words(self, *, dehyphenate: bool = True) -> list[Word]: ...

    def text_blocks(self, *, dehyphenate: bool = True) -> list[Block]:
        """Paragraph-shaped blocks carrying text rather than spans."""
        ...

    def find_tables(self, *, strategy: TableStrategy = "lines") -> list[FoundTable]:
        """Expensive — callers gate it behind a cheaper signal."""
        ...

    def render(self, *, dpi: int, clip: Rect | None = None) -> np.ndarray:
        """Rasterise to an ``(h, w, 3)`` uint8 RGB array; alpha is dropped."""
        ...

    def images(self) -> list[PageImage]: ...

    def drawings(self) -> list[Drawing]: ...

    def widgets(self) -> list[Widget]: ...


class Document(Protocol):
    """A PDF, or an image a backend presents as a one-page PDF."""

    @property
    def page_count(self) -> int: ...

    @property
    def name(self) -> str: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Page: ...

    def __iter__(self) -> Iterator[Page]: ...

    def select(self, pages: list[int]) -> None:
        """Keep only *pages*, in the order given. In-memory only."""
        ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...

"""Block grouping on the pdfium backend follows MuPDF's measured rules.

MuPDF starts a block where the baseline pitch exceeds about 1.5 font sizes, keeps a
row's fragments (table cells) together, and does not break on a size or weight
change alone. The expected values below were measured on PyMuPDF 1.27.2.2 before
it was removed: lines per block, per page, in block order. The spacing fixture
varies leading and paragraph gap so the rule is held at more than one pitch.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest

from tests._synthetic import AUDIT_PDF, FOI_INDEX_PDF, NOTICE_PDF, SCHEDULE_PDF, SPACING_PDF
from womblex.ingest.detect import detect_file_type
from womblex.ingest.extract import extract_text
from womblex.ingest.pdf import open_document

_FOI_PAGE = [7, 5, 1] + [11] * 36

#: Lines per block, per page, in block order.
PARITY = {
    AUDIT_PDF: [
        [1, 4, 4, 4, 1], [1, 4, 4, 4, 1], [1] + [4] * 10 + [1], [1, 4, 4, 4, 1], [1, 4, 3, 4, 1], [1, 3, 4, 3, 1],
    ],
    SPACING_PDF: [[1, 12], [1, 4, 3, 3, 3], [1, 3, 3, 3, 3], [1, 3, 3, 3, 3], [1, 3, 3, 3, 3], [1, 12]],
    NOTICE_PDF: [[1, 6, 3, 1, 3], [1, 4, 4, 4, 4, 4], [1, 4, 4, 4, 4, 4]],
    SCHEDULE_PDF: [[1, 1] + [9] * 45],
    FOI_INDEX_PDF: [[2, 2, 7, 5, 1] + [11] * 36, _FOI_PAGE, _FOI_PAGE, _FOI_PAGE],
}

#: Element kinds and counts over the whole extraction.
KINDS = {
    AUDIT_PDF: {"header": 6, "paragraph": 24, "footer": 6, "page_break": 5, "table": 1, "heading": 1},
    SPACING_PDF: {"header": 6, "heading": 6, "page_break": 5, "paragraph": 12},
    FOI_INDEX_PDF: {"heading": 12, "paragraph": 144, "footer": 2, "page_break": 3, "table": 1},
}


def _lines_per_block(path: Path) -> list[list[int]]:
    return [[len(b.lines) for b in page.text_dict()] for page in open_document(path)]


@pytest.mark.parametrize("path", PARITY, ids=lambda p: p.name)
def test_blocks_hold_the_same_lines_as_mupdf(path: Path) -> None:
    assert _lines_per_block(path) == PARITY[path]


@pytest.mark.parametrize("path", KINDS, ids=lambda p: p.name)
def test_element_kinds_and_counts_match_mupdf(path: Path) -> None:
    kinds = collections.Counter(e.kind for r in extract_text(path, detect_file_type(path)) for e in r.elements)
    assert dict(kinds) == KINDS[path]


def _draw(path: Path, texts: list[tuple[float, float, str]], *, angle: int = 0, rotate_page: int = 0) -> Path:
    from reportlab.pdfgen.canvas import Canvas

    canvas = Canvas(str(path), pagesize=(842, 595), invariant=1)
    if rotate_page:
        canvas.setPageRotation(rotate_page)
    canvas.translate(400, 300)
    canvas.rotate(angle)
    canvas.setFont("Helvetica", 7)
    for x, y, text in texts:
        canvas.drawString(x, y, text)
    canvas.save()
    return path


#: (x, y, text) in the text's own frame, y up: header cells, a two-line header,
#: an indented second line, and lines either side of the baseline.
_SHAPES = {
    "two-line header under the last cell": [(0, 0, "AAAA"), (50, 0, "BBBB"), (100, 0, "Sub"), (100, -8, "code"), (150, 0, "Next")],
    "two-line header under the first cell": [(0, 0, "AAAA"), (0, -8, "a2"), (50, 0, "BBBB")],
    "line below, same start": [(0, 0, "AAAAAAAA"), (0, -9, "BBBB")],
    "line below, indented": [(0, 0, "AAAAAAAAAAAAAAAAAAAA"), (6, -8.4, "BBBBBBBB")],
    "cell after, 3pt up": [(0, 0, "AAAA"), (50, 3, "BBBB")],
    "cell after, 6pt down": [(0, 0, "AAAA"), (50, -6, "BBBB")],
}

#: Axis-aligned only: how MuPDF joins an overprint and a line above is measured there.
_AXIS_SHAPES = {
    "overprinted line": [(0, 0, "AAAAAAAA"), (0, 3, "BBBB")],
    "overprinted line, indented": [(0, 0, "AAAAAAAA"), (10, 3, "BBBB")],
    "line above": [(0, 0, "AAAAAAAA"), (0, 8, "BBBB")],
}

#: Lines per block as MuPDF groups each shape, by angle.
_EXPECTED = {
    "two-line header under the last cell": {0: [4, 1], 5: [3, 1, 1], 90: [3, 2], 180: [5], 270: [4, 1]},
    "two-line header under the first cell": {0: [2, 1], 5: [1, 1, 1], 90: [1, 2], 180: [3], 270: [2, 1]},
    "line below, same start": {0: [2], 5: [1, 1], 90: [1, 1], 180: [2], 270: [2]},
    "line below, indented": {0: [1, 1], 5: [1, 1], 90: [1, 1], 180: [2], 270: [2]},
    "cell after, 3pt up": {0: [2], 5: [2], 90: [2], 180: [2], 270: [2]},
    "cell after, 6pt down": {0: [1, 1], 5: [1, 1], 90: [1, 1], 180: [2], 270: [2]},
    "overprinted line": {0: [2], 90: [2], 180: [2], 270: [2]},
    "overprinted line, indented": {0: [2], 90: [2], 180: [2], 270: [2]},
    "line above": {0: [2], 90: [2], 180: [2], 270: [1, 1]},
}


@pytest.mark.parametrize("angle", [0, 5, 90, 180, 270])
@pytest.mark.parametrize("shape", _SHAPES)
def test_header_shapes_group_as_mupdf_does(shape: str, angle: int, tmp_path: Path) -> None:
    path = _draw(tmp_path / "shape.pdf", _SHAPES[shape], angle=angle)
    assert _lines_per_block(path) == [_EXPECTED[shape][angle]]


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("shape", _AXIS_SHAPES)
def test_overprints_and_lines_above_group_as_mupdf_does(shape: str, angle: int, tmp_path: Path) -> None:
    path = _draw(tmp_path / "shape.pdf", _AXIS_SHAPES[shape], angle=angle)
    assert _lines_per_block(path) == [_EXPECTED[shape][angle]]


def test_a_rotated_page_with_upright_text_is_grouped_in_the_unrotated_frame(tmp_path: Path) -> None:
    path = _draw(tmp_path / "rotated.pdf", _SHAPES["two-line header under the last cell"], angle=90, rotate_page=90)
    assert _lines_per_block(path) == [[3, 2]]


@pytest.mark.parametrize("angle", [90, 270])
def test_vertical_text_keeps_content_order(angle: int, tmp_path: Path) -> None:
    # Drawn out of position order: pdfium sorts vertical text by position, MuPDF does not.
    path = _draw(tmp_path / "order.pdf", [(0, -9, "L0"), (0, -18, "L1"), (0, 0, "L2")], angle=angle)
    lines = [
        "".join(s.text for s in line.spans)
        for page in open_document(path) for b in page.text_dict() for line in b.lines
    ]
    assert lines == ["L0", "L1", "L2"]

"""Markdown extraction: block-level structure via markdown-it-py + GFM tables.

Thin adapter over ``markdown-it-py``: the parser is used only to find block
boundaries (headings, paragraphs, list items, tables). ``Token.content`` is
the parser's raw pre-inline-parse text for a block — markdown syntax inside
it (``**bold**`` etc) is left untouched, matching the verbatim text policy at
the extraction boundary.

Source lines the parser emits no block for (YAML front matter, link
reference definitions) are kept as verbatim paragraphs rather than dropped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from markdown_it import MarkdownIt
from markdown_it.token import Token

from womblex.ingest.elements import Cell, Element, ElementKind
from womblex.ingest.extract import ExtractionMetadata, ExtractionResult, PageResult

_MD = MarkdownIt("commonmark").enable("table")

# CommonMark has no front matter: unhandled, `---` parses as a rule and the
# YAML body plus closing `---` as a setext heading.
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n.*?\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)


def read_markdown(path: Path) -> str:
    """File text with any UTF-8 BOM removed and line endings as the parser sees them."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")


class MarkdownExtractor:
    """Extract elements from a Markdown file in document order.

    Headings, paragraphs, list items and GFM pipe tables become
    ``heading`` / ``paragraph`` / ``list_item`` / ``table`` elements,
    interleaved as they appear in the source — the same document-order
    handling ``DocxExtractor`` gives prose alongside tables.
    """

    def extract_path(self, path: Path) -> ExtractionResult:
        elements = markdown_elements(read_markdown(path))

        page_text = "\n\n".join(e.text for e in elements if e.text)
        return ExtractionResult(
            pages=[PageResult(page_number=0, text=page_text, method="markdown")],
            elements=elements,
            method="markdown",
            metadata=ExtractionMetadata(
                extraction_strategy="markdown",
                confidence=0.9,
                processing_time=0.0,
                page_count=1,
                text_coverage=1.0 if page_text else 0.0,
            ),
        )


def markdown_elements(text: str) -> list[Element]:
    """Ordered elements for Markdown *text* (``\\n`` line endings)."""
    lines = text.split("\n")
    elements: list[Element] = []
    covered = 0

    front = _FRONT_MATTER.match(text)
    if front:
        _add(elements, "paragraph", front.group(0).strip(), 0.85)
        # Blank the lines rather than cut them so token line maps stay aligned.
        block = front.group(0)
        covered = block.rstrip("\n").count("\n") + 1
        text = "\n" * block.count("\n") + text[front.end():]

    tokens = _MD.parse(text)
    list_depth = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.level == 0 and token.map:
            _add_uncovered(elements, lines, covered, token.map[0])
            covered = max(covered, token.map[1])
        if token.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
            i += 1
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
            i += 1
        elif token.type == "heading_open":
            _add(elements, "heading", tokens[i + 1].content.strip(), 0.9)
            i += 3
        elif token.type == "paragraph_open":
            kind = cast("ElementKind", "list_item" if list_depth > 0 else "paragraph")
            _add(elements, kind, tokens[i + 1].content.strip(), 0.9)
            i += 3
        elif token.type in ("fence", "code_block", "html_block"):
            _add(elements, "paragraph", token.content.strip(), 0.85)
            i += 1
        elif token.type == "table_open":
            end = _matching_close(tokens, i, "table_close")
            cells, header_rows = _table_cells(tokens[i:end + 1])
            if cells:
                elements.append(Element(
                    order=len(elements), kind="table", extractor="markdown",
                    cells=cells, header_rows=header_rows, confidence=0.85,
                ))
            i = end + 1
        else:
            i += 1
    _add_uncovered(elements, lines, covered, len(lines))
    return elements


def _add(elements: list[Element], kind: ElementKind, text: str, confidence: float) -> None:
    if text:
        elements.append(Element(
            order=len(elements), kind=kind, extractor="markdown",
            text=text, confidence=confidence,
        ))


def _add_uncovered(elements: list[Element], lines: list[str], start: int, end: int) -> None:
    """One verbatim paragraph per blank-line-separated run of *lines[start:end]*."""
    run: list[str] = []
    for line in [*lines[start:end], ""]:
        if line.strip():
            run.append(line)
        elif run:
            _add(elements, "paragraph", "\n".join(run).strip(), 0.85)
            run = []


def _matching_close(tokens: list[Token], start: int, close_type: str) -> int:
    for j in range(start + 1, len(tokens)):
        if tokens[j].type == close_type:
            return j
    return len(tokens) - 1


def _table_cells(tokens: list[Token]) -> tuple[list[Cell], list[int]]:
    """Cells + header row indices for one ``table_open``..``table_close`` span."""
    cells: list[Cell] = []
    header_rows: list[int] = []
    in_header = False
    row = -1
    col = 0
    for token in tokens:
        if token.type == "thead_open":
            in_header = True
        elif token.type == "thead_close":
            in_header = False
        elif token.type == "tr_open":
            row += 1
            col = 0
            if in_header:
                header_rows.append(row)
        elif token.type == "inline":
            cells.append(Cell(row=row, col=col, value=token.content.strip()))
            col += 1
    return cells, header_rows

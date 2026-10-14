"""Source spans retained on every semantic node."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpan:
    """Byte and line/column range in a single file.

    Lines and columns are 1-indexed. Bytes are 0-indexed half-open.
    """

    start_byte: int
    end_byte: int
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def contains_byte(self, index: int) -> bool:
        return self.start_byte <= index < self.end_byte

    def identity_key(self) -> str:
        return f"{self.start_byte}:{self.end_byte}:{self.start_line}:{self.start_column}"


@dataclass(frozen=True)
class SourceMap:
    """Map coordinates from parser input back onto original repository source.

    Some parsers require a synthetic prefix (for example PHP ``<?php ``). The
    prefix may exist only in parser input. Reported spans always refer to the
    original file.
    """

    original: str
    prefix: str = ""

    @property
    def prefix_bytes(self) -> int:
        if not self.prefix:
            return 0
        return len(self.prefix.encode("utf-8"))

    def remap(self, span: SourceSpan) -> SourceSpan:
        shift = self.prefix_bytes
        if shift <= 0:
            return span
        prefix_cols = len(self.prefix)
        start_byte = max(0, span.start_byte - shift)
        end_byte = max(start_byte, span.end_byte - shift)
        start_column = span.start_column
        end_column = span.end_column
        if span.start_line == 1:
            start_column = max(1, span.start_column - prefix_cols)
            if span.start_byte < shift:
                start_column = 1
        if span.end_line == 1:
            end_column = max(1, span.end_column - prefix_cols)
            if span.end_byte <= shift:
                end_column = 1
        return SourceSpan(
            start_byte=start_byte,
            end_byte=end_byte,
            start_line=span.start_line,
            start_column=start_column,
            end_line=span.end_line,
            end_column=end_column,
        )


def span_from_ts_node(node: object) -> SourceSpan:
    """Build a span from a tree-sitter node (duck-typed)."""
    start_point = getattr(node, "start_point")
    end_point = getattr(node, "end_point")
    return SourceSpan(
        start_byte=int(getattr(node, "start_byte")),
        end_byte=int(getattr(node, "end_byte")),
        start_line=int(start_point[0]) + 1,
        start_column=int(start_point[1]) + 1,
        end_line=int(end_point[0]) + 1,
        end_column=int(end_point[1]) + 1,
    )


def span_from_lineno(
    source: str,
    start_line: int,
    end_line: int | None = None,
    *,
    start_column: int = 1,
    end_column: int | None = None,
) -> SourceSpan:
    """Approximate a span from 1-indexed line numbers (CPython AST fallback)."""
    lines = source.splitlines(keepends=True) or [""]
    n = len(lines)
    start_i = max(0, start_line - 1)
    end_i = max(start_i, (end_line or start_line) - 1)
    start_byte = sum(len(lines[i]) for i in range(start_i))
    if end_i < n:
        end_byte = sum(len(lines[i]) for i in range(end_i + 1))
    else:
        end_byte = len(source.encode("utf-8", errors="replace"))
    last = lines[end_i] if end_i < n else ""
    return SourceSpan(
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=max(1, start_line),
        start_column=max(1, start_column),
        end_line=max(1, end_line or start_line),
        end_column=end_column if end_column is not None else max(1, len(last.rstrip("\n"))),
    )

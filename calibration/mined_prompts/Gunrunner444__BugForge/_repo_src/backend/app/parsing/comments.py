"""Comment stripping that respects string literals."""

from __future__ import annotations


def strip_comments(
    source: str,
    *,
    line_comment: str | tuple[str, ...] | None,
    block_comment: tuple[str, str] | None,
    string_delims: tuple[str, ...] = ("'", '"', "`"),
) -> str:
    """Replace comments with spaces so line numbers stay aligned."""
    out: list[str] = []
    i = 0
    n = len(source)
    in_string: str | None = None
    escape = False
    block_start, block_end = block_comment or ("", "")
    line_markers = _line_markers(line_comment)

    while i < n:
        ch = source[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_string == "`" and ch == "$" and i + 1 < n and source[i + 1] == "{":
                # Template interpolation: leave `${` in place and continue; the
                # caller scans interpolations as code. Nested comments inside
                # interpolations are handled by treating `{...}` as code.
                out.append("{")
                i += 2
                inner, consumed = _strip_interpolation(
                    source[i:],
                    line_markers=line_markers,
                    block_start=block_start,
                    block_end=block_end,
                    string_delims=string_delims,
                )
                out.append(inner)
                i += consumed
                continue
            elif ch == in_string:
                in_string = None
            i += 1
            continue
        if block_start and source.startswith(block_start, i):
            end = source.find(block_end, i + len(block_start)) if block_end else -1
            chunk_end = end + len(block_end) if end != -1 else n
            for c in source[i:chunk_end]:
                out.append("\n" if c == "\n" else " ")
            i = chunk_end
            continue
        marker = _starts_with_any(source, i, line_markers)
        if marker:
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch in string_delims:
            in_string = ch
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _line_markers(line_comment: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if line_comment is None:
        return ()
    if isinstance(line_comment, str):
        return (line_comment,) if line_comment else ()
    return tuple(sorted((m for m in line_comment if m), key=len, reverse=True))


def _starts_with_any(source: str, index: int, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if source.startswith(marker, index):
            return marker
    return None


def _strip_interpolation(
    source: str,
    *,
    line_markers: tuple[str, ...],
    block_start: str,
    block_end: str,
    string_delims: tuple[str, ...],
) -> tuple[str, int]:
    """Strip comments inside a `${...}` region; return (text, chars_consumed)."""
    depth = 1
    i = 0
    n = len(source)
    in_string: str | None = None
    escape = False
    out: list[str] = []
    while i < n and depth:
        ch = source[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_string == "`" and ch == "$" and i + 1 < n and source[i + 1] == "{":
                out.append("{")
                i += 2
                inner, consumed = _strip_interpolation(
                    source[i:],
                    line_markers=line_markers,
                    block_start=block_start,
                    block_end=block_end,
                    string_delims=string_delims,
                )
                out.append(inner)
                i += consumed
                continue
            elif ch == in_string:
                in_string = None
            i += 1
            continue
        if block_start and source.startswith(block_start, i):
            end = source.find(block_end, i + len(block_start)) if block_end else -1
            chunk_end = end + len(block_end) if end != -1 else n
            for c in source[i:chunk_end]:
                out.append("\n" if c == "\n" else " ")
            i = chunk_end
            continue
        marker = _starts_with_any(source, i, line_markers)
        if marker:
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch in string_delims:
            in_string = ch
            out.append(ch)
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(ch)
                i += 1
                break
        out.append(ch)
        i += 1
    return "".join(out), i

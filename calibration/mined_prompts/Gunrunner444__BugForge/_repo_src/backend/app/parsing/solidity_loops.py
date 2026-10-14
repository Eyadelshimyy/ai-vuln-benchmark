"""Loop bound and state-growth classification for one Solidity loop statement.

The bound is the loop header, not a search of the surrounding function for
``[N]`` or ``[]``. A loop is bounded only when that header is clearly capped.
"""

from __future__ import annotations

import re

from app.parsing.solidity_cfg import split_loop

_EXTERNAL = re.compile(r"\.(call|delegatecall|staticcall|send|transfer|safeTransferFrom)\s*[\(\{]")
_GROWTH = re.compile(
    r"\.push\s*\(|"
    r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])+(?:\s*\.\s*[A-Za-z_]\w*)*\s*(?:\+=|-=|=)(?!=)"
)


def loop_header(loop_text: str) -> str:
    try:
        header, _body, _style = split_loop(loop_text)
    except ValueError:
        return loop_text
    return header


def loop_body(loop_text: str) -> str:
    try:
        _header, body, _style = split_loop(loop_text)
    except ValueError:
        return loop_text
    return body


def loop_is_bounded(loop_text: str, function_text: str) -> bool:
    """True only when the header itself shows a fixed or explicitly capped bound."""
    header = loop_header(loop_text)
    if re.search(r"<\s*\d+\b", header):
        return True
    if re.search(r"\bmin\s*\(", header) and re.search(r"\b(?:\d+|[A-Z][A-Z0-9_]*)\b", header):
        return True
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\.\s*length\b", header)
    if names and all(_fixed_collection(function_text, name) for name in names):
        return True
    return False


def loop_has_external_call(loop_text: str) -> bool:
    return bool(_EXTERNAL.search(loop_body(loop_text)))


def loop_grows_state(loop_text: str) -> bool:
    return bool(_GROWTH.search(loop_body(loop_text)))


def _fixed_collection(function_text: str, name: str) -> bool:
    pattern = rf"\[(\d+)\]\s*(?:memory|calldata|storage)?\s*{re.escape(name)}\b"
    return bool(re.search(pattern, function_text))

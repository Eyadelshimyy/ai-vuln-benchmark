"""Tree-sitter parser cache.

Grammars are loaded once per process from the pinned language pack. A Parser
instance is created per parse because tree-sitter parsers are not thread-safe.
Source size is bounded. Analysis never downloads grammars from the network.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.core.config import settings
from app.parsing.span import SourceSpan, span_from_ts_node

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_LANGUAGES: dict[str, object] = {}
_UNAVAILABLE: set[str] = set()

# Tested pair (see tests/test_semantic/test_parser_deps.py):
#   tree-sitter==0.26.0
#   tree-sitter-language-pack==1.20.0
# Official BugForge languages are bundled in the pack. Analysis must not fetch
# additional grammars over the network while walking an untrusted repository.
TS_LANGUAGE_IDS: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "ruby": "ruby",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "php": "php",
    "kotlin": "kotlin",
    "swift": "swift",
    "csharp": "csharp",
    "shell": "bash",
    "html": "html",
    "css": "css",
    "scss": "scss",
    "sql": "sql",
    "solidity": "solidity",
}

# Grammars exist in the pack but BugForge does not yet meet the full-analysis
# contract for these languages (quality catalog, taint vocab, fixtures).
DETECTION_ONLY_TS_IDS: dict[str, str] = {
    "r": "r",
    "scala": "scala",
    "dart": "dart",
    "lua": "lua",
    "elixir": "elixir",
}

MAX_PARSE_BYTES = 2 * 1024 * 1024
MAX_WALK_NODES = 40_000
MAX_NESTING = 256

TESTED_TREE_SITTER = "0.26.0"
TESTED_LANGUAGE_PACK = "1.20.0"


@dataclass(frozen=True)
class TreeSitterTree:
    language_id: str
    ts_language: str
    source_bytes: bytes
    tree: object
    truncated: bool = False


@dataclass(frozen=True)
class NativeParserStatus:
    language_id: str
    available: bool
    ts_language: str | None
    reason: str = ""


def treesitter_available(language_id: str) -> bool:
    ts_id = TS_LANGUAGE_IDS.get(language_id)
    if ts_id is None:
        return False
    return _language(ts_id) is not None


def native_parser_status(language_id: str) -> NativeParserStatus:
    ts_id = TS_LANGUAGE_IDS.get(language_id)
    if ts_id is None:
        return NativeParserStatus(
            language_id=language_id,
            available=False,
            ts_language=None,
            reason="no bundled grammar mapping",
        )
    if _language(ts_id) is None:
        return NativeParserStatus(
            language_id=language_id,
            available=False,
            ts_language=ts_id,
            reason="native parser unavailable",
        )
    return NativeParserStatus(
        language_id=language_id,
        available=True,
        ts_language=ts_id,
        reason="native parser available",
    )


def reset_treesitter_cache() -> None:
    """Test helper. Does not download grammars."""
    with _LOCK:
        _LANGUAGES.clear()
        _UNAVAILABLE.clear()


def _language(ts_id: str) -> object | None:
    with _LOCK:
        if ts_id in _UNAVAILABLE:
            return None
        cached = _LANGUAGES.get(ts_id)
        if cached is not None:
            return cached
        try:
            from tree_sitter_language_pack import get_language

            lang = get_language(ts_id)
        except Exception as exc:  # pragma: no cover - environment-specific
            logger.warning("Tree-sitter grammar %s unavailable: %s", ts_id, exc)
            _UNAVAILABLE.add(ts_id)
            return None
        _LANGUAGES[ts_id] = lang
        return lang


def parse_treesitter(
    language_id: str,
    source: str,
    *,
    ts_language: str | None = None,
) -> TreeSitterTree | None:
    ts_id = ts_language or TS_LANGUAGE_IDS.get(language_id)
    if ts_id is None:
        return None
    lang = _language(ts_id)
    if lang is None:
        return None
    encoded = source.encode("utf-8", errors="replace")
    truncated = False
    limit = min(MAX_PARSE_BYTES, settings.max_file_size_bytes)
    if len(encoded) > limit:
        encoded = encoded[:limit]
        truncated = True
    try:
        from tree_sitter import Parser

        parser = Parser(lang)  # type: ignore[arg-type]
        tree = parser.parse(encoded)
    except Exception as exc:
        logger.debug("Tree-sitter parse failed for %s: %s", language_id, exc)
        return None
    return TreeSitterTree(
        language_id=language_id,
        ts_language=ts_id,
        source_bytes=encoded,
        tree=tree,
        truncated=truncated,
    )


def collect_error_spans(root: object) -> tuple[SourceSpan, ...]:
    spans: list[SourceSpan] = []
    stack = [(root, 0)]
    visited = 0
    while stack and visited < MAX_WALK_NODES:
        node, depth = stack.pop()
        visited += 1
        ntype = getattr(node, "type", "")
        if ntype == "ERROR" or getattr(node, "is_missing", False):
            try:
                spans.append(span_from_ts_node(node))
            except Exception:
                continue
        if depth >= MAX_NESTING:
            continue
        children = getattr(node, "children", None) or ()
        for child in children:
            stack.append((child, depth + 1))
    return tuple(spans[:200])

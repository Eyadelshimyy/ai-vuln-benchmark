"""Observability primitives for ellements.

This subpackage centralises everything related to *seeing what
happened* inside an ellements run:

- :mod:`.events` — the event vocabulary
  (:class:`LLMRequestEvent`, :class:`LLMResponseEvent`,
  :class:`LLMErrorEvent`, :class:`AgentEvent`).
- :mod:`.observer` — the :class:`LLMObserver` Protocol.
- :mod:`.jsonl_logger` — :class:`JsonlPromptLogger`, the canonical
  observer implementation, and its :class:`LoggedCall` record schema.
- :mod:`.markdown_formatter` — :func:`format_log_markdown`, a
  human-readable view over any ``.jsonl`` log file.

This ``__init__`` re-exports the full public surface so callers can
``from ellements.core.observability import …`` without reaching into
specific submodules.
"""

from .events import (
    AgentEvent,
    LLMErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
)
from .jsonl_logger import JsonlPromptLogger, LoggedCall
from .markdown_formatter import format_log_markdown
from .observer import LLMObserver

__all__ = [
    "AgentEvent",
    "JsonlPromptLogger",
    "LLMErrorEvent",
    "LLMObserver",
    "LLMRequestEvent",
    "LLMResponseEvent",
    "LoggedCall",
    "format_log_markdown",
]

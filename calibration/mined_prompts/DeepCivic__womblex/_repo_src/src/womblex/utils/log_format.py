"""Structured (JSON-lines) logging, stdlib only.

``--log-format json`` (or ``WOMBLEX_LOG_FORMAT=json``) swaps the text formatter
for :class:`JsonFormatter`: one JSON object per record, carrying the run
context a log aggregator filters on — ``run_id``, ``job_id``, ``stage`` and
``source_hash``. A caller supplies those either per record (``extra=``) or for
a whole block with :func:`log_context`, which is how the worker tags every line
a job emits without threading identifiers through the stages.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

CONTEXT_KEYS = ("run_id", "job_id", "stage", "source_hash")

_context: ContextVar[dict[str, Any] | None] = ContextVar("womblex_log_context", default=None)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach *fields* (``CONTEXT_KEYS`` only; ``None`` skipped) to records in the block."""
    unknown = set(fields) - set(CONTEXT_KEYS)
    if unknown:
        raise ValueError(f"unknown log context keys: {sorted(unknown)}")
    merged = {**(_context.get() or {}), **{k: v for k, v in fields.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


class JsonFormatter(logging.Formatter):
    """One JSON object per record; context keys from ``extra=`` win over the block's."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context.get() or {})
        for key in CONTEXT_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)

"""
Structured logging and tracing setup.

Call configure_logging() once at startup (done in main.py).

Usage in modules:
    from qsf.common.logging import get_logger
    logger = get_logger(__name__)
    logger.info("fetch_news: %d articles", count)

Trace context is set per-request by TraceMiddleware and flows automatically
into every log line via structlog's contextvars.
"""
import contextvars
import logging
import os

import structlog


def get_logger(name: str):
    return structlog.get_logger(name)


def configure_logging() -> None:
    json_mode = os.getenv("LOG_FORMAT", "").lower() == "json"
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    renderer = structlog.processors.JSONRenderer() if json_mode else structlog.dev.ConsoleRenderer()

    shared_pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Structlog native loggers (our modules): process through the shared chain,
    # then hand off to stdlib via render_to_log_kwargs so the ProcessorFormatter
    # below does the single final render for all log lines.
    structlog.configure(
        processors=shared_pre_chain + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.stdlib.render_to_log_kwargs,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Single renderer for all logs — both our structlog lines and third-party stdlib
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_pre_chain,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)


# ---------------------------------------------------------------------------
# Langfuse client (disabled if LANGFUSE_PUBLIC_KEY is not set)
# ---------------------------------------------------------------------------

def _init_langfuse():
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse()
    except Exception:
        return None


_langfuse = _init_langfuse()


def get_langfuse():
    """Return the Langfuse singleton, or None if not configured."""
    return _langfuse


# ContextVar holds the active Langfuse trace for the current request.
# Set by main.py before pipeline.invoke(); read by ingestion/nlp modules.
_CURRENT_TRACE: contextvars.ContextVar = contextvars.ContextVar("langfuse_trace", default=None)


def set_current_trace(trace) -> None:
    _CURRENT_TRACE.set(trace)


def get_current_trace():
    """Return the Langfuse trace for the current request, or None."""
    return _CURRENT_TRACE.get()


def flush_langfuse() -> None:
    if _langfuse is not None:
        _langfuse.flush()

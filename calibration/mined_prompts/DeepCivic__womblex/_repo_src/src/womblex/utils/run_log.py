"""Capture a batch's ``womblex`` log lines to a file alongside its shards.

The per-document failure lines that explain a run — ``operations/extract.py``'s
``Detection failed: doc=… error=…`` and ``Extraction failed: doc=… error=…`` —
already exist, but they go to stderr, so they die in ``docker logs`` on whichever
worker claimed the batch and never reach the console. :func:`capture_batch_log`
tees them into one file per batch, which the worker (and ``womblex run``) then
publishes next to the shards for the Dashboard to serve.

Additive by construction: it *attaches* a ``FileHandler`` to the ``womblex``
logger for the batch's duration and detaches it after, leaving the existing
stderr handler untouched — so ``docker logs`` output is unchanged. The handler
is always removed, on success or failure, so nothing leaks across batches.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: The logger every stage logs through. The whole codebase uses
#: ``logging.getLogger(__name__)`` under the ``womblex`` package (and the CLI
#: uses ``getLogger("womblex")`` directly), so attaching here captures all of
#: them without touching the root logger and picking up dependency chatter.
_ROOT_LOGGER_NAME = "womblex"

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


@contextmanager
def capture_batch_log(path: Path, *, level: int = logging.INFO) -> Iterator[Path]:
    """Tee ``womblex`` log records into *path* for the duration of the block.

    Yields *path*. The file is created (parents made) on entry and always
    flushed and closed on exit, so it is complete and publishable whether the
    block returns or raises — the failing case is the one that matters, since
    it is the failure lines the operator needs. The handler is removed in a
    ``finally``, so a raising block never leaves it attached to leak the next
    batch's records into this file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    # The `womblex` logger may propagate to a root that has its own level; set
    # the handler's own level rather than the logger's so we neither raise the
    # global threshold nor miss records the logger already passes.
    logger.addHandler(handler)
    try:
        yield path
    finally:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


__all__ = ["capture_batch_log"]

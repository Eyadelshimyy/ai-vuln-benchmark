"""Process-wide logging for the deployed entrypoints.

Both services write one line per event to stderr, where the host collects it.
The level comes from ``GVAS_LOG_LEVEL``; an unknown name falls back to INFO
rather than failing the deploy, since logging is observability, not correctness.
"""

import logging
import sys

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def resolve_level(name: str) -> int:
    level = logging.getLevelNamesMapping().get(name.strip().upper())
    return level if level is not None else logging.INFO


def configure_logging(level_name: str) -> int:
    level = resolve_level(level_name)
    logging.basicConfig(level=level, format=FORMAT, stream=sys.stderr, force=True)
    return level

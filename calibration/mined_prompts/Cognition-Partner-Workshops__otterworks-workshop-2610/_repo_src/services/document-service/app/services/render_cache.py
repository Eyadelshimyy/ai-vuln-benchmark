"""In-process memo of rendered document exports.

Added for the exports dashboard so repeated downloads of the same document
version do not re-render. Entries are keyed by document, version and format.
"""

from __future__ import annotations

import structlog

from app.models.document import Document
from app.telemetry import RENDER_CACHE_ENTRIES, SERVICE

logger = structlog.get_logger()


class RenderCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, int, str], tuple[str, str, dict[str, str]]] = {}

    def _key(self, document: Document, fmt: str) -> tuple[str, int, str]:
        return (str(document.id), document.version, fmt)

    def get(self, document: Document, fmt: str) -> tuple[str, str] | None:
        hit = self._entries.get(self._key(document, fmt))
        if hit is None:
            return None
        body, content_type, _ = hit
        return body, content_type

    def put(self, document: Document, fmt: str, body: str, content_type: str) -> None:
        # Keep the source alongside the render so a later diff view can show
        # what changed between versions without another round trip.
        source = {"title": document.title, "content": document.content}
        self._entries[self._key(document, fmt)] = (body, content_type, source)
        RENDER_CACHE_ENTRIES.labels(SERVICE).set(len(self._entries))

    def clear(self) -> None:
        self._entries.clear()
        RENDER_CACHE_ENTRIES.labels(SERVICE).set(0)

    def __len__(self) -> int:
        return len(self._entries)


render_cache = RenderCache()

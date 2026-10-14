"""The PDF seam: one entry point onto the pdfium backend.

`types.py` holds the backend-neutral vocabulary the extractors are written
against. The backend lives in its own `_*.py` and is imported only when a
document is opened, so `import womblex` — and importing this package — loads no
PDF library.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from womblex.ingest.pdf.types import Document


def open_document(path: Path) -> Document:
    """Open *path* as a `Document`, loading the pdfium backend on the way."""
    module = import_module("womblex.ingest.pdf._pdfium_doc")
    return cast("Document", module.open_document(path))


__all__ = ["open_document"]

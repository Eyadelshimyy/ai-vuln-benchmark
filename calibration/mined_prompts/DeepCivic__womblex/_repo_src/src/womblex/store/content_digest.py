"""Content digest of a document's element stream.

The manifest's ``content_digest`` is the determinism handle: for a given
``source_hash`` + ``womblex.version`` + ``config_digest`` + ``womblex.models``
two extractions carry the same digest, whatever bytes the Parquet files
happen to have. It covers what a consumer reads — kind, order, text, table
cells and form fields, spreadsheet cells (sheet / row / col / value and
their formatting), alt text and element meta — and nothing that varies run
to run. Geometry, confidence and extractor name are left out.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from womblex.ingest.elements import Element


def _element_payload(e: Element) -> list[object]:
    return [
        e.kind,
        e.order,
        e.text,
        [[c.row, c.col, c.value, c.rowspan, c.colspan, c.value_type] for c in e.cells or ()],
        e.header_rows,
        [[f.name, f.value, f.field_type] for f in e.fields or ()],
        e.alt_text,
        [e.sheet, e.row, e.col, e.value, e.value_type, e.formula, e.number_format, e.merge_range],
        sorted(e.meta.items()),
    ]


def content_digest(elements: Iterable[Element]) -> str:
    """SHA-256 hex over the ordered elements' content fields (see module docstring)."""
    payload = json.dumps(
        [_element_payload(e) for e in elements],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["content_digest"]

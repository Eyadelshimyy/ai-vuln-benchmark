"""Document text reads for the `/v1` API (`docs/service-api.md`).

One document's text out of the run's shards, by layer. A layer's sensitivity is
the one its file role carries (:mod:`womblex.store.contract`), so the gate is
the contract's own: a ``masked`` or ``none`` layer needs ``read``, anything
else ``read_raw``. Rows are read in place with the ``source_hash`` predicate
pushed into the Parquet reader, as the console's chunk inspector does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pyarrow.parquet as pq

from womblex.store.contract import Sensitivity, sensitivity_for
from womblex.store.output import CHUNKS_SUFFIX
from womblex.store.pii_output import CLEAN_TEXT_SUFFIX
from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

ELEMENTS_SUFFIX = ".elements.parquet"


@dataclass(frozen=True)
class Layer:
    name: str
    role: str
    suffix: str
    columns: tuple[str, ...]
    order: str

    @property
    def sensitivity(self) -> Sensitivity:
        return sensitivity_for(self.role)


#: The text layers a caller can ask for; ``masked`` is the default.
LAYERS: dict[str, Layer] = {
    "masked": Layer("masked", "clean_text", CLEAN_TEXT_SUFFIX,
                    ("chunk_index", "content_type", "text", "n_masked"), "chunk_index"),
    "chunks": Layer("chunks", "chunks", CHUNKS_SUFFIX,
                    ("chunk_index", "content_type", "text"), "chunk_index"),
    "elements": Layer("elements", "elements", ELEMENTS_SUFFIX,
                      ("elem_order", "kind", "page", "text"), "elem_order"),
}


def document_rows(store_uri: str, run_id: str, source_hash: str, layer: Layer) -> list[dict]:
    """Rows of *layer* for one document, in document order; empty when it has none."""
    store = RemoteStore.from_uri(store_uri)
    rows: list[dict] = []
    for key in store.list_files(f"runs/{run_id}/documents", f"*{layer.suffix}"):
        try:
            table = pq.read_table(
                f"{store.root}/{key}", filesystem=store.fs, columns=["source_hash", *layer.columns],
                filters=[("source_hash", "=", source_hash)],
            )
        except Exception as e:  # one unreadable shard narrows the answer, it does not fail it
            logger.warning("api text: skipping unreadable shard %s: %s", key, e)
            continue
        rows.extend({c: r[c] for c in layer.columns} for r in table.to_pylist())
    rows.sort(key=lambda r: r[layer.order])
    return rows

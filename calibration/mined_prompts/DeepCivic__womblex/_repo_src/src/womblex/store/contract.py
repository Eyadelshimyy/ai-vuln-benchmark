"""Which contract a file was written under, and how sensitive its content is.

Two footer keys ride on every pipeline Parquet, whether or not the run can be
named (unlike the run stamp, which is written only for a run that can):

- ``womblex.contract_version`` — the on-disk contract the file conforms to.
  It is not the package version: a release that changes no schema leaves it
  alone, and a schema change bumps it (additive column → minor; rename or
  removal → major, with a reader backfill like ``_CHUNKS_BACKFILL``).
- ``womblex.sensitivity`` — whether the file carries document text with PII
  unmasked (``raw``), masked (``masked``), or no document text at all
  (``none``). A consumer handing files onward reads this instead of knowing
  every role. A role this module does not know reads as ``raw``: an
  unclassified file is treated as the sensitive kind, never the safe one.

``none`` is a claim about text, not about derivation: embeddings are computed
from unmasked chunks and are labelled ``none`` because they carry no text.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from womblex.store.source_provenance import NAMESPACE

CONTRACT_VERSION = "2.0"
CONTRACT_VERSION_KEY = f"{NAMESPACE}.contract_version"
SENSITIVITY_KEY = f"{NAMESPACE}.sensitivity"

Sensitivity = Literal["raw", "masked", "none"]

#: Sensitivity by file role — the segment before ``.parquet``
#: (``batch-0001.clean_text.parquet`` → ``clean_text``; run-root
#: ``manifest.parquet`` → ``manifest``).
ROLE_SENSITIVITY: dict[str, Sensitivity] = {
    # Verbatim or near-verbatim document text.
    "elements": "raw",
    "table_cells": "raw",
    "form_fields": "raw",
    "chunks": "raw",
    "normalised_text": "raw",
    "spellfix_text": "raw",
    "spellfix_corrections": "raw",
    "enrichment_entities": "raw",
    "enrichment_doc": "raw",
    "graph_edges": "raw",
    "entity_links": "raw",
    "pii_spans": "raw",
    # Corpus-declared provenance columns: content unknown, so not claimed safe.
    "provenance": "raw",
    # PII-masked text — the layer meant to be handed onward.
    "clean_text": "masked",
    # Identifiers, offsets, numbers, vectors — no document text.
    "_manifest": "none",
    "manifest": "none",
    "source_index": "none",
    "embeddings": "none",
    "enrichment_meta": "none",
    "chunk_quality": "none",
    "redactions": "none",
    "money_spans": "none",
    "money_columns": "none",
    "layout_regions": "none",
}


def role_of(path: Path) -> str:
    """The role segment of a Parquet file name (see :data:`ROLE_SENSITIVITY`)."""
    stem = Path(path).name.removesuffix(".parquet")
    return stem.rsplit(".", 1)[-1]


def sensitivity_for(role: str) -> Sensitivity:
    """The sensitivity of *role*; ``raw`` for a role this module does not know."""
    return ROLE_SENSITIVITY.get(role, "raw")


def contract_footer(path: Path, *, role: str | None = None) -> dict[bytes, bytes]:
    """Footer keys for a file at *path*; *role* overrides the one its name implies."""
    return {
        CONTRACT_VERSION_KEY.encode(): CONTRACT_VERSION.encode(),
        SENSITIVITY_KEY.encode(): sensitivity_for(role or role_of(path)).encode(),
    }


def read_footer_contract(metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    """Decode the contract keys out of a Parquet footer; ``{}`` for a file written before them."""
    if not metadata:
        return {}
    names = ((CONTRACT_VERSION_KEY, "contract_version"), (SENSITIVITY_KEY, "sensitivity"))
    return {name: metadata[key.encode()].decode() for key, name in names if key.encode() in metadata}


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_VERSION_KEY",
    "ROLE_SENSITIVITY",
    "SENSITIVITY_KEY",
    "Sensitivity",
    "contract_footer",
    "read_footer_contract",
    "role_of",
    "sensitivity_for",
]

"""womblex — document extraction and normalisation for Australian government data.

The names in ``__all__`` are the stable Python API (see ``docs/contract.md``).
They resolve lazily, so ``import womblex`` stays cheap.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.5.12"

#: public name -> module that defines it
_EXPORTS: dict[str, str] = {
    "CONTRACT_VERSION": "womblex.store.contract",
    "extract_text": "womblex.ingest.extract",
    "run_extraction": "womblex.operations",
    "run_redaction": "womblex.operations",
    "run_chunking": "womblex.operations",
    "run_enrichment": "womblex.operations",
    "run_pii_cleaning": "womblex.operations",
    "normalise_shards": "womblex.process.normalise_stage",
    "spellfix_shards": "womblex.process.spellfix_stage",
    "chunk_shards": "womblex.process.chunk_stage",
    "money_shards": "womblex.process.money_stage",
    "quality_shards": "womblex.process.quality_stage",
    "enrich_shards": "womblex.analyse.enrich_stage",
    "embed_shards": "womblex.analyse.embed_stage",
    "link_shards": "womblex.link.stage",
    "pii_shards": "womblex.pii.pii_stage",
    "build_bundle": "womblex.store.egress",
    "write_run_manifest": "womblex.store.run_manifest",
    "read_results": "womblex.store.output",
}

__all__ = [
    "CONTRACT_VERSION",
    "__version__",
    "build_bundle",
    "chunk_shards",
    "embed_shards",
    "enrich_shards",
    "extract_text",
    "link_shards",
    "money_shards",
    "normalise_shards",
    "pii_shards",
    "quality_shards",
    "read_results",
    "run_chunking",
    "run_enrichment",
    "run_extraction",
    "run_pii_cleaning",
    "run_redaction",
    "spellfix_shards",
    "write_run_manifest",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'womblex' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

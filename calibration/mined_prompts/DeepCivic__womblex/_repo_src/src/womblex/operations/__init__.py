"""Independent operations for document processing.

Each function is standalone with clear input/output contracts. There is no
orchestrator — callers compose operations directly.

Operations (one module each):

- :mod:`~womblex.operations.extract`  — ``run_extraction``
- :mod:`~womblex.operations.redact`   — ``run_redaction``
- :mod:`~womblex.operations.chunk`    — ``run_chunking``
- :mod:`~womblex.operations.pii`      — ``run_pii_cleaning``
- :mod:`~womblex.operations.enrich`   — ``run_enrichment`` / ``enrich_batch``
- :mod:`~womblex.operations.persist`  — ``write_batch_parquet`` / ``write_batch_enrichment``
- :mod:`~womblex.operations.models`   — ``DocumentResult`` / ``BatchResult``

This package was split from a single ``operations.py`` (>750-line cap); the
flat import surface ``from womblex.operations import run_extraction`` etc. is
preserved by the re-exports below.

Standalone ingest paths (G-NAF PSV, SHP) have their own modules under
``ingest/`` and do not use these operations.
"""

from womblex.operations.chunk import run_chunking
from womblex.operations.enrich import enrich_batch, run_enrichment
from womblex.operations.extract import run_extraction
from womblex.operations.models import BatchResult, DocumentResult, PreconditionError
from womblex.operations.persist import write_batch_enrichment, write_batch_parquet
from womblex.operations.pii import run_pii_cleaning
from womblex.operations.redact import run_redaction

__all__ = [
    "BatchResult",
    "DocumentResult",
    "PreconditionError",
    "enrich_batch",
    "run_chunking",
    "run_enrichment",
    "run_extraction",
    "run_pii_cleaning",
    "run_redaction",
    "write_batch_enrichment",
    "write_batch_parquet",
]

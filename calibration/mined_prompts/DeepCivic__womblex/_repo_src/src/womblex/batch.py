"""Shared per-batch pipeline body.

``cmd_run`` (local, single-process) and the cloud worker (distributed) must
process a batch of documents *identically* — same stages, same sequencing,
same shard layout — or the two execution modes would silently diverge. This
module is the single home for that sequencing: extraction → optional redaction
detection → write one ``batch-NNNN.parquet`` shard (and its sidecars).

Extraction is strictly extraction: it produces an extracted, true-to-source
version of the input document and nothing else. Chunking, PII, enrichment,
embedding, money and the rest are *downstream* stages with their own
``run-stage`` contracts — they never run in-batch. The one thing that stays
is redaction *detection*: flagging where the source itself has redacted
regions is part of representing the document true to source, not a transform
applied on top of it (see CLAUDE.md, "Redaction is a post-extraction
concern").

Deliberately stateless: it does no checkpointing and no cumulative-size
bookkeeping. Those are caller concerns — the local runner uses a
``CheckpointManager`` + cross-batch size check, the distributed worker uses the
Postgres job queue as its checkpoint. Keeping them out keeps this body reusable
by both without a race.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.config import WomblexConfig
from womblex.operations import (
    BatchResult,
    run_extraction,
    run_redaction,
    write_batch_parquet,
)
from womblex.store.run_stamp import RunStamp
from womblex.store.source_provenance import IngestProvenance

logger = logging.getLogger(__name__)


@dataclass
class BatchOutcome:
    """Result of processing one batch: the results plus what was persisted."""

    batch: BatchResult
    shard_path: Path
    rows_written: int


def process_batch(
    batch_files: list[Path],
    config: WomblexConfig,
    *,
    batch_num: int,
    shard_dir: Path,
    provenance: IngestProvenance | None = None,
    stamp: RunStamp | None = None,
) -> BatchOutcome:
    """Extract *batch_files* (plus optional redaction detection) and write one shard.

    Mirrors the inner loop of ``cmd_run``. This is extraction only: it runs
    ``run_extraction`` and, when ``config.redaction.enabled``, ``run_redaction``
    (redaction *detection* is true-to-source, not a transform). Chunking, PII
    and the other downstream stages are ``run-stage`` contracts and are never
    run here — so ``config.chunking.enabled`` / ``config.pii.enabled`` mean
    "this stage is in the pipeline", not "run it inside extraction". A worker
    fed the same config produces byte-identical shards. Returns a
    :class:`BatchOutcome`; the caller decides how to verify, checkpoint, or
    publish.

    ``provenance`` declares where the documents came from — the ingest root
    and each document's path under it — and is stamped onto the shard's
    manifest columns and every shard file's footer. The caller supplies it
    because only the caller can know it: the local runner reads the root the
    config declares, while the worker's documents arrive in a scratch dir
    whose paths say nothing about the store keys they came from. Omitted, the
    shard goes unstamped — this body never invents a root of its own.

    ``stamp`` names the run the shard belongs to and reaches the same footer,
    for the same reason and on the same terms: the caller holds the run id,
    and an omitted stamp writes no run keys rather than inventing one.
    """
    results = run_extraction(batch_files, config)
    if config.redaction.enabled:
        results = run_redaction(results, config)

    batch = BatchResult(results=results)
    shard_path = shard_dir / f"batch-{batch_num:04d}.parquet"
    rows_written = sum(
        1 for r in batch.results if r.status == "completed" and r.extraction is not None
    )
    write_batch_parquet(batch, shard_path, provenance=provenance, stamp=stamp)
    return BatchOutcome(batch=batch, shard_path=shard_path, rows_written=rows_written)


__all__ = ["BatchOutcome", "process_batch"]

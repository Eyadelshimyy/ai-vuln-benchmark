"""Per-stage text normalisation over an existing shard directory.

Consumes ``*.elements.parquet`` (+ ``*._manifest.parquet`` for doc ids) and
writes a ``*.normalised_text.parquet`` sibling per batch: one row per
``TEXT_KINDS`` element with its normalised text (verbatim passthrough where
nothing fired). Tables / forms / images are left to the element stream — this
op cleans narrative text only.

Mirrors :mod:`womblex.process.chunk_stage`: per-stage ``CheckpointManager``,
skip-existing on resume, batch-level isolation. Reuses ``_batch_bases`` and
``_load_elements`` from the chunk stage so element materialisation stays in
one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.config import NormaliseConfig
from womblex.ingest.elements import TEXT_KINDS
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.process.normalise import NormaliseTransforms, normalise_text
from womblex.store.checkpoint import CheckpointManager
from womblex.store.normalise_output import (
    normalised_text_path_for,
    write_normalised_text,
)
from womblex.store.output import read_manifest

logger = logging.getLogger(__name__)


@dataclass
class NormaliseStageResult:
    batches_written: int
    elements_normalised: int
    elements_changed: int


def normalise_shards(
    shard_dir: Path,
    config: NormaliseConfig,
    *,
    checkpoint_mgr: CheckpointManager | None = None,
) -> NormaliseStageResult:
    """Normalise narrative text for every batch; write ``*.normalised_text.parquet``."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("normalise_shards: no batches found in %s", shard_dir)
        return NormaliseStageResult(0, 0, 0)

    transforms = NormaliseTransforms(
        unicode_hygiene=config.unicode_hygiene,
        collapse_whitespace=config.collapse_whitespace,
        despace_page_marker=config.despace_page_marker,
        substitutions=dict(config.substitutions),
    )

    batches_written = 0
    elements_normalised = 0
    elements_changed = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("normalise_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        rows, n_changed = _normalise_batch(base, transforms)

        write_normalised_text(rows, base)
        batches_written += 1
        elements_normalised += len(rows)
        elements_changed += n_changed

        if checkpoint_mgr is not None:
            doc_ids = _doc_ids(base)
            if doc_ids:
                checkpoint_mgr.update(
                    doc_ids=doc_ids,
                    succeeded=len(doc_ids),
                    failed=0,
                    batch_num=int(base.stem.replace("batch-", "") or 0),
                )

        logger.info(
            "normalise_shards: %s normalised %d elements (%d changed)",
            base.stem, len(rows), n_changed,
        )

    return NormaliseStageResult(
        batches_written=batches_written,
        elements_normalised=elements_normalised,
        elements_changed=elements_changed,
    )


def _normalise_batch(
    base_path: Path, transforms: NormaliseTransforms,
) -> tuple[list[dict], int]:
    """Read a batch's elements and produce normalised-text rows for TEXT_KINDS."""
    elements_by_hash = _load_elements(base_path)
    rows: list[dict] = []
    n_changed = 0
    for source_hash, elements in elements_by_hash.items():
        for e in elements:
            if e.kind not in TEXT_KINDS or not e.text:
                continue
            text, n = normalise_text(e.text, e.kind, transforms)
            if n:
                n_changed += 1
            rows.append({
                "source_hash": source_hash,
                "elem_order": e.order,
                "kind": e.kind,
                "page": e.page,
                "text": text,
                "n_changes": n,
            })
    return rows, n_changed


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not normalised_text_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["NormaliseStageResult", "normalise_shards"]

"""Per-stage OCR character-confusion repair over a shard directory.

Consumes ``*.elements.parquet`` and writes two siblings per batch:
``*.spellfix_text.parquet`` (the repaired element-text overlay, verbatim
passthrough where nothing fired) and ``*.spellfix_corrections.parquet`` (the
audit trail). The raw ``*.elements.parquet`` is never modified.

Repair lands at the **element** layer — the shared source the chunk and
enrichment branches both fork from — so downstream stages compose on it by
selecting ``text_source='spellfix'`` (see :mod:`womblex.process.text_overlay`).
Mirrors :mod:`womblex.process.normalise_stage`: per-stage ``CheckpointManager``,
skip-existing on resume, batch-level isolation. Offline (no API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from womblex.config import SpellfixConfig
from womblex.ingest.elements import TEXT_KINDS
from womblex.process.chunk_stage import _batch_bases, _load_elements
from womblex.store.checkpoint import CheckpointManager
from womblex.store.output import read_manifest
from womblex.store.spellfix_output import (
    spellfix_text_path_for,
    write_spellfix_corrections,
    write_spellfix_text,
)

logger = logging.getLogger(__name__)


@dataclass
class SpellfixStageResult:
    batches_written: int
    elements_repaired: int
    corrections_applied: int


def spellfix_shards(
    shard_dir: Path,
    config: SpellfixConfig,
    *,
    checkpoint_mgr: CheckpointManager | None = None,
) -> SpellfixStageResult:
    """Repair every batch's element text; write overlay + corrections siblings."""
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")

    # Built up front so an unknown name, bad options or a missing dictionary
    # directory fail before the first document, not mid-batch.
    from womblex.process.spellfix import load_dictionary

    load_dictionary(config.dict_name, config.dict_options)

    bases = _batch_bases(shard_dir)
    if not bases:
        logger.warning("spellfix_shards: no batches found in %s", shard_dir)
        return SpellfixStageResult(0, 0, 0)

    batches_written = 0
    elements_repaired = 0
    corrections_applied = 0

    for base in bases:
        if checkpoint_mgr is not None and _all_docs_checkpointed(base, checkpoint_mgr):
            logger.info("spellfix_shards: skipping %s (all docs checkpointed)", base.stem)
            continue

        text_rows, audit_rows, n_changed = _repair_batch(base, config)

        write_spellfix_text(text_rows, base)
        write_spellfix_corrections(audit_rows, base)
        batches_written += 1
        elements_repaired += n_changed
        corrections_applied += len(audit_rows)

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
            "spellfix_shards: %s repaired %d elements (%d corrections)",
            base.stem, n_changed, len(audit_rows),
        )

    return SpellfixStageResult(
        batches_written=batches_written,
        elements_repaired=elements_repaired,
        corrections_applied=corrections_applied,
    )


def _repair_batch(base_path: Path, config: SpellfixConfig) -> tuple[list[dict], list[dict], int]:
    """Read a batch's elements; return ``(text_rows, audit_rows, n_changed)``.

    Chains on top of the normalise layer: if ``*.normalised_text.parquet`` exists
    it is overlaid onto the elements before repair, so the resulting
    ``*.spellfix_text.parquet`` is the terminal cleaned+repaired layer.
    """
    from womblex.process.spellfix import repair_text
    from womblex.process.text_overlay import apply_overlay, load_overlay

    elements_by_hash = _load_elements(base_path)
    normalised = load_overlay(base_path, "normalised", warn_if_missing=False)
    text_rows: list[dict] = []
    audit_rows: list[dict] = []
    n_changed = 0
    for source_hash, elements in elements_by_hash.items():
        apply_overlay(source_hash, elements, normalised)
        for e in elements:
            if e.kind not in TEXT_KINDS or not e.text:
                continue
            fixed, corrections = repair_text(
                e.text,
                general_edits=config.general_edits,
                dict_name=config.dict_name,
                dict_options=config.dict_options or None,
            )
            text_rows.append({
                "source_hash": source_hash,
                "elem_order": e.order,
                "kind": e.kind,
                "page": e.page,
                "text": fixed,
                "n_changes": len(corrections),
            })
            if corrections:
                n_changed += 1
            for c in corrections:
                audit_rows.append({
                    "source_hash": source_hash,
                    "elem_order": e.order,
                    "kind": e.kind,
                    "offset": c.offset,
                    "original": c.original,
                    "corrected": c.corrected,
                    "method": c.method,
                })
    return text_rows, audit_rows, n_changed


def _doc_ids(base_path: Path) -> list[str]:
    try:
        return list(read_manifest(base_path).column("doc_id").to_pylist())
    except Exception:
        return []


def _all_docs_checkpointed(base_path: Path, mgr: CheckpointManager) -> bool:
    if not spellfix_text_path_for(base_path).exists():
        return False
    doc_ids = _doc_ids(base_path)
    return bool(doc_ids) and all(d in mgr.state.processed_ids for d in doc_ids)


__all__ = ["SpellfixStageResult", "spellfix_shards"]

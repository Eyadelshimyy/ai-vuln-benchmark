"""Per-stage layout over an existing shard directory (``run-stage layout``).

Re-runs the batch's layout step against the source documents and replaces each
batch's ``*.layout_regions.parquet``. Elements, tables and redaction results
stay as extracted: nothing reads layout after extraction, so a rerun is a way to
measure a model, not to change what was built (applying a new model to tables or
redaction is a re-extract). The sidecar's own fingerprint, compared with the
elements footer's, says whether the two still agree
(:func:`womblex.store.layout_output.layout_fingerprint_status`).

Skip rule: a batch whose sidecar already carries this config's fingerprint is
left alone; a changed model, option, dpi or page scope reruns it. ``force``
reruns regardless.

Sources come back through :class:`SourceResolver`, the first downstream stage to
need them, or, on a distributed run, through a ``source_fetcher`` the runner
supplies: it stages one document from the ingest store and checks its hash, and
is called only for a batch that will actually be analysed. A batch is written only if every document in it was analysed, so a
sidecar is never half old model and half new; one that could not be analysed is
logged with its source hash and the batch is counted failed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from womblex.process.chunk_stage import _batch_bases
from womblex.store.layout_output import (
    layout_regions_path_for,
    layout_rows,
    read_footer_layout_fingerprint,
    write_layout_regions,
)
from womblex.store.output import read_manifest
from womblex.store.run_stamp import sidecar_footer

if TYPE_CHECKING:
    from womblex.config import WomblexConfig
    from womblex.ingest.layout_step import LayoutSettings
    from womblex.store.source_resolver import SourceResolver

logger = logging.getLogger(__name__)


@dataclass
class LayoutStageResult:
    batches_written: int = 0
    batches_skipped: int = 0
    batches_failed: int = 0
    documents: int = 0


def _seam_extensions() -> frozenset[str]:
    """The extensions the PDF seam opens: the only documents layout applies to."""
    from womblex.cli._shared import IMAGE_EXTENSIONS

    return frozenset({".pdf", *IMAGE_EXTENSIONS})


def layout_shards(
    shard_dir: Path,
    config: WomblexConfig,
    *,
    source_root: str | Path | None = None,
    source_fetcher: Callable[[dict], Path] | None = None,
    force: bool = False,
) -> LayoutStageResult:
    """Rerun layout for every batch in *shard_dir* whose sidecar is out of date."""
    from womblex.ingest.layout_step import LayoutSettings
    from womblex.store.source_resolver import SourceResolver

    if not shard_dir.is_dir():
        raise FileNotFoundError(f"shard directory not found: {shard_dir}")
    bases = _batch_bases(shard_dir)
    result = LayoutStageResult()
    if not bases:
        logger.warning("layout_shards: no batches found in %s", shard_dir)
        return result

    settings = LayoutSettings.from_config(config)
    resolver: SourceResolver | None = None

    for base in bases:
        sidecar = layout_regions_path_for(base)
        if not force and sidecar.exists() and read_footer_layout_fingerprint(
            pq.read_metadata(str(sidecar)).metadata,
        ) == settings.fingerprint:
            logger.info("layout_shards: skipping %s (fingerprint unchanged)", base.stem)
            result.batches_skipped += 1
            continue
        try:
            if resolver is None and source_fetcher is None:
                resolver = SourceResolver.for_run(shard_dir, root=source_root)
            rows, documents = _analyse_batch(base, settings, config, resolver, source_fetcher)
        except Exception as e:  # one batch must not stop the run
            logger.error("layout_shards: %s failed: %s", base.stem, e)
            result.batches_failed += 1
            continue
        if rows is None:
            result.batches_failed += 1
            continue
        # Redaction ran at extraction, against the regions these replace.
        write_layout_regions(
            rows, base, settings.fingerprint,
            redaction_consumed=False,
            metadata=sidecar_footer(base, "layout"),
        )
        result.batches_written += 1
        result.documents += documents
    return result


def _analyse_batch(
    base: Path, settings: LayoutSettings, config: WomblexConfig,
    resolver: SourceResolver | None, fetcher: Callable[[dict], Path] | None,
) -> tuple[list[dict] | None, int]:
    """``(rows, documents analysed)`` for one batch; rows are ``None`` if any document failed."""
    from womblex.ingest.layout_step import run_layout_step
    from womblex.ingest.page_profile import profile_pages
    from womblex.ingest.pdf import open_document

    seam = _seam_extensions()
    manifest = read_manifest(base).to_pylist()
    per_document: list[tuple[str, list]] = []
    failed = 0
    seen: set[str] = set()
    for row in manifest:
        source_hash = row["source_hash"]
        if source_hash in seen or row["status"] == "error" or row["ext"] not in seam:
            continue
        seen.add(source_hash)
        try:
            path = _source_path(row, resolver, fetcher)
            with open_document(path) as doc:
                outcome = run_layout_step(
                    doc, profile_pages(doc), settings, config.extraction.ocr.engine,
                )
        except Exception as e:
            logger.error("layout_shards: doc=%s source_hash=%s failed: %s", row["doc_id"], source_hash, e)
            failed += 1
            continue
        per_document.append((source_hash, outcome.pages))
    if failed:
        logger.error(
            "layout_shards: %s left as it was, %d document(s) not analysed", base.stem, failed,
        )
        return None, 0
    return layout_rows(per_document), len(per_document)


def _source_path(
    row: dict, resolver: SourceResolver | None, fetcher: Callable[[dict], Path] | None,
) -> Path:
    """The local file for a manifest row: the fetcher's, else the resolver's, else an error."""
    if fetcher is not None:
        return fetcher(row)
    assert resolver is not None
    resolution = resolver.resolve(row["source_hash"])
    if not resolution.ok or resolution.path is None:
        raise FileNotFoundError(f"not resolved ({resolution.status}): {resolution.detail}")
    return resolution.path


__all__ = ["LayoutStageResult", "layout_shards"]

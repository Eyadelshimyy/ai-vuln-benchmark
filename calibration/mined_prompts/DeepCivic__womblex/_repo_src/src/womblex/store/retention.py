"""Run directories: generate, list, describe, and purge.

Each pipeline run writes its outputs under ``<output_root>/<run_id>/`` and
its checkpoint state under ``<checkpoint_dir>/<run_id>/``. Multiple runs
co-exist for stage-vs-stage comparison; retention controls how many old
runs are kept on disk.

:func:`describe_run` summarises a single run — document count, which stages
have run, timestamps — for a run selector. It reads only artefacts the
pipeline already writes and never mutates a run.

Two policies:

- ``rolling`` — keep the ``keep`` most-recent runs (including current),
  purge older. Default ``keep=2`` (current + previous).
- ``keep_all`` — disable auto-purge; the user manages purges manually.

The current run is *always* preserved regardless of policy or position.
Retention is normally applied at the start of a new run, before any
output is written.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from womblex.pipeline_order import sort_by_pipeline
from womblex.store.embed_output import EMBEDDINGS_SUFFIX
from womblex.store.enrichment_output import ENRICHMENT_ENTITIES_SUFFIX
from womblex.store.entity_links_output import ENTITY_LINKS_SUFFIX
from womblex.store.money_output import MONEY_SPANS_SUFFIX
from womblex.store.normalise_output import NORMALISED_TEXT_SUFFIX
from womblex.store.output import CHUNKS_SUFFIX, ELEMENTS_SUFFIX, read_manifest
from womblex.store.pii_output import PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SUFFIX
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.store.spellfix_output import SPELLFIX_TEXT_SUFFIX

logger = logging.getLogger(__name__)

#: Stage name -> the sidecar suffix that stage writes, in pipeline order.
#: Presence of a matching file in a shard dir is what "this stage has run"
#: means on disk. Redaction is deliberately absent: it rewrites element text
#: in memory and leaves no sidecar of its own to detect, and so is
#: `graph-refresh`, which rewrites enrich's two sidecars in place rather than
#: writing one of its own.
#:
#: Ordered by `sort_by_pipeline`, not by how it is spelled here: every reader
#: of this dict presents its keys as the run's stage list (the console's
#: lifecycle-checkpoint switcher renders them directly), and hand-keeping a
#: literal in order is exactly what drifted — this ran `chunk` before
#: `enrich`, backwards from the sequence the README prescribes.
STAGE_SUFFIXES: dict[str, str] = sort_by_pipeline({
    "extract": ELEMENTS_SUFFIX,
    "normalise": NORMALISED_TEXT_SUFFIX,
    "spellfix": SPELLFIX_TEXT_SUFFIX,
    "chunk": CHUNKS_SUFFIX,
    "quality": CHUNK_QUALITY_SUFFIX,
    "money": MONEY_SPANS_SUFFIX,
    "pii": PII_SPANS_SUFFIX,
    "link": ENTITY_LINKS_SUFFIX,
    "enrich": ENRICHMENT_ENTITIES_SUFFIX,
    "embed": EMBEDDINGS_SUFFIX,
})


# A run id becomes a path segment under a run root. Anything that can re-root
# or climb out of that join is refused rather than sanitised: real run ids are
# single segments (``generate_run_id`` emits ``run-YYYYMMDDTHHMMSSZ``), so
# there is nothing legitimate to rewrite.
_UNSAFE_IN_RUN_ID = ("/", "\\", "\x00")


def is_safe_run_id(run_id: str) -> bool:
    """True if *run_id* is a single path segment safe to join onto a root."""
    return (
        bool(run_id)
        and run_id not in {".", ".."}
        and not any(c in run_id for c in _UNSAFE_IN_RUN_ID)
    )


def generate_run_id() -> str:
    """Generate a sortable, filesystem-safe run id from current UTC time.

    Format: ``run-YYYYMMDDTHHMMSSZ`` — sorts lexically by creation order
    and parses back as an ISO-8601 basic-format timestamp.
    """
    return "run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def list_runs(output_root: Path) -> list[Path]:
    """Return run directories under ``output_root``, sorted oldest-first.

    A "run directory" is any subdirectory whose name starts with ``run-``
    (matching the auto-generated ``run-YYYYMMDDTHHMMSSZ`` convention).
    Subdirectories with other names are preserved — they are not subject
    to auto-retention. This protects legacy / hand-named output dirs
    (e.g. ``output/documents/`` from a pre-run_id layout, or
    ``output/baseline-snapshot/``) from being purged when a new run lands.

    To bring a hand-named run under the retention policy, name it with a
    ``run-`` prefix (either via ``dataset.run_id`` in config or
    ``--run-id`` on the CLI).

    Sort key is directory name (so timestamp-default ids sort by
    creation order) with mtime as a secondary key for user-supplied
    ``run-*`` names.
    """
    if not output_root.is_dir():
        return []
    runs = [
        p for p in output_root.iterdir()
        if p.is_dir() and p.name.startswith("run-")
    ]
    runs.sort(key=lambda p: (p.name, p.stat().st_mtime))
    return runs


def most_recent_run(output_root: Path) -> Path | None:
    """Return the newest run directory under ``output_root``, or None."""
    runs = list_runs(output_root)
    return runs[-1] if runs else None


@dataclass(frozen=True)
class RunDescription:
    """Summary of one run directory, for a run selector (console or CLI).

    ``document_count`` is ``None`` when there is no manifest to count yet —
    an empty or not-yet-extracted run — which is distinct from a run that
    genuinely extracted zero documents. ``stages`` names only the stages with
    an on-disk sidecar to show for themselves (see :data:`STAGE_SUFFIXES`), so
    a run that also redacted still reports ``("extract",)``. Timestamps are
    ISO-8601 UTC derived from file mtimes, since a run directory carries no
    timestamp of its own.
    """

    run_id: str
    document_count: int | None
    stages: tuple[str, ...]
    created_at: str | None
    updated_at: str | None


def describe_run(run_dir: Path) -> RunDescription:
    """Describe a run root (``<output_root>/<run_id>/``) — the shape :func:`list_runs` returns.

    Reads only artefacts the pipeline already writes: the consolidated
    ``manifest.parquet`` when a run has been finalised, else the per-batch
    shard manifests under ``documents/``.
    """
    shard_dir = run_dir / "documents"
    created_at, updated_at = _run_timestamps(run_dir)
    return RunDescription(
        run_id=run_dir.name,
        document_count=_document_count(run_dir, shard_dir),
        stages=_stages_present(shard_dir),
        created_at=created_at,
        updated_at=updated_at,
    )


def _document_count(run_dir: Path, shard_dir: Path) -> int | None:
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        return int(pq.ParquetFile(str(manifest_path)).metadata.num_rows)
    if not shard_dir.is_dir():
        return None
    return int(read_manifest(shard_dir).num_rows)


def _stages_present(shard_dir: Path) -> tuple[str, ...]:
    if not shard_dir.is_dir():
        return ()
    return tuple(
        stage for stage, suffix in STAGE_SUFFIXES.items() if any(shard_dir.glob(f"*{suffix}"))
    )


def _run_timestamps(run_dir: Path) -> tuple[str | None, str | None]:
    mtimes = [p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()]
    if not mtimes:
        return None, None
    return (
        datetime.fromtimestamp(min(mtimes), tz=UTC).isoformat(),
        datetime.fromtimestamp(max(mtimes), tz=UTC).isoformat(),
    )


def apply_retention(
    output_root: Path,
    checkpoint_dir: Path,
    *,
    current_run_id: str,
    policy: str,
    keep: int,
) -> list[Path]:
    """Purge old run directories per policy.

    Args:
        output_root: Parent directory holding per-run subdirs.
        checkpoint_dir: Parent of per-run checkpoint subdirs (purged in lockstep).
        current_run_id: Name of the active run. Always preserved.
        policy: ``rolling`` (default) or ``keep_all``.
        keep: Number of runs to retain under ``rolling`` (>= 1).

    Returns:
        List of purged run directories (empty under ``keep_all`` or if
        nothing exceeds the retention window).

    Raises:
        ValueError: Unknown policy or ``keep < 1``.
    """
    if policy == "keep_all":
        return []
    if policy != "rolling":
        raise ValueError(f"unknown retention policy: {policy!r}")
    if keep < 1:
        raise ValueError(f"retention.keep must be >= 1, got {keep}")

    runs = list_runs(output_root)
    # Newest-first
    runs.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)

    others = [p for p in runs if p.name != current_run_id]
    # Reserve one slot for the current run (whether or not it's on disk yet)
    retain_others = max(keep - 1, 0)
    to_purge = others[retain_others:]

    purged: list[Path] = []
    for run_dir in to_purge:
        logger.info("retention: purging old run %s", run_dir)
        shutil.rmtree(run_dir)
        ckpt = checkpoint_dir / run_dir.name
        if ckpt.is_dir():
            shutil.rmtree(ckpt)
        purged.append(run_dir)
    return purged

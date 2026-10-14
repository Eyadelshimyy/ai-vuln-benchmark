"""Thin pyarrow readers over shard sidecars, local and store-backed.

Local reads go straight through ``store/retention.py`` / ``store/output.py``.
Remote reads stage the (small) manifest into a temp dir and hand it to the
same local readers — the pattern ``cli/cloud.py``'s ``finalize`` already
uses, so there is one code path for the parquet logic.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from womblex.store.enrichment_output import (
    ENRICHMENT_ENTITIES_SUFFIX,
    ENTITY_SCHEMA,
    LEGACY_HASH_COLUMN,
)
from womblex.store.money_output import (
    MONEY_SPANS_SCHEMA,
    MONEY_SPANS_SUFFIX,
    money_spans_from_legacy,
)
from womblex.store.output import (
    _SHARD_SUFFIX,
    CHUNKS_SCHEMA,
    CHUNKS_SUFFIX,
    read_manifest,
)
from womblex.store.pii_output import PII_SPANS_SCHEMA, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SCHEMA, CHUNK_QUALITY_SUFFIX
from womblex.store.retention import STAGE_SUFFIXES, RunDescription, describe_run, list_runs
from womblex.store.run_manifest import RUN_MANIFEST_FILENAME
from womblex.store.shard_audit import ARCHIVE_SUFFIX, audit_shard_directory
from womblex.ui import presets as presets_mod
from womblex.ui.deps import UISettings
from womblex.ui.presets import Preset

if TYPE_CHECKING:
    from womblex.store.remote import RemoteStore

logger = logging.getLogger(__name__)

#: A run log's filename, validated before any path join. Batch logs are named
#: ``batch-NNNN.log`` by :func:`womblex.utils.run_log.capture_batch_log`'s
#: callers (the worker and ``cmd_run``); anything else is refused rather than
#: joined, the same containment discipline :func:`is_safe_run_id` applies to a
#: run id. The console has no auth, so a name that reaches a store/filesystem
#: join could otherwise probe outside the run's ``logs/`` prefix.
_LOG_NAME_RE = re.compile(r"batch-\d{4}\.log\Z")


def is_safe_log_name(name: str) -> bool:
    """True if *name* is a batch log filename safe to join onto a ``logs/`` prefix."""
    return bool(_LOG_NAME_RE.fullmatch(name))


class StoreUnreachable(Exception):
    """The configured object store could not be opened or reached.

    Raised by :func:`_open_store` when ``RemoteStore.from_uri`` fails — a
    missing backend package (``ImportError: Install s3fs to access S3`` when
    ``womblex[cloud]`` is not installed), bad credentials, or an unreachable
    endpoint. It carries the underlying message so the operator sees the same
    actionable cause the Resources card's *Test connection* reports.

    A store fault is not a bug in a request — it is a deployment that cannot
    reach its own run source — so the read routes map this to **503** rather
    than letting the raw exception surface as an opaque **500**. This mirrors
    the dashboard's queue handling, where an unreachable queue is a reported
    ``queue_error``, not a crash.
    """


def list_run_summaries(settings: UISettings) -> list[RunDescription]:
    """All runs the settings can see, newest run_id first."""
    if settings.is_remote:
        runs = _list_remote_runs(cast(str, settings.store_uri))
    else:
        runs = [describe_run(p) for p in list_runs(cast(Path, settings.output_root))]
    return sorted(runs, key=lambda d: d.run_id, reverse=True)


def get_manifest_rows(settings: UISettings, run_id: str) -> list[dict] | None:
    """Manifest rows (one per document) for run_id, or None if it doesn't exist."""
    if settings.is_remote:
        return _remote_manifest_rows(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return list(_local_manifest_table(run_dir).to_pylist())


def get_stage_presence(settings: UISettings, run_id: str, stage: str) -> list[str] | None:
    """``source_hash`` values with a ``stage`` sidecar row in run_id.

    None if the run doesn't exist, else a (possibly empty) sorted list —
    the lifecycle-checkpoint switcher's data, since a lifecycle checkpoint
    *is* sidecar presence. Reads only the ``source_hash`` column of each
    sidecar, so this stays cheap even for large chunks /
    enrichment sidecars.
    """
    suffix = STAGE_SUFFIXES[stage]
    if settings.is_remote:
        return _remote_stage_presence(cast(str, settings.store_uri), run_id, suffix)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    shard_dir = run_dir / "documents"
    if not shard_dir.is_dir():
        return []
    return _scan_stage_presence(shard_dir, suffix)


def get_shard_audit(settings: UISettings, run_id: str) -> dict | None:
    """Cross-batch structural audit for run_id — the verify-shards action's data."""
    if settings.is_remote:
        return _remote_shard_audit(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    return audit_shard_directory(run_dir / "documents").as_dict()


# The Chunk Inspector's overlay sidecars, each filtered to one document:
# (response key, suffix, canonical schema) — both taken from the same store
# module so a renamed suffix cannot drift from the schema it describes. All five
# join on `source_hash`; the enrichment one named it `document_id` before 0.6.0.
_CHUNK_DETAIL_SIDECARS: tuple[tuple[str, str, pa.Schema], ...] = (
    ("chunks", CHUNKS_SUFFIX, CHUNKS_SCHEMA),
    ("entities", ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA),
    ("pii_spans", PII_SPANS_SUFFIX, PII_SPANS_SCHEMA),
    ("money_spans", MONEY_SPANS_SUFFIX, MONEY_SPANS_SCHEMA),
    ("quality", CHUNK_QUALITY_SUFFIX, CHUNK_QUALITY_SCHEMA),
)


def get_chunk_detail(settings: UISettings, run_id: str, source_hash: str) -> dict | None:
    """Chunks + overlay sidecars for one document — the Chunk Inspector's data.

    None if run_id doesn't exist. An empty ``chunks`` list is a legitimate
    answer (chunk stage not yet run, or source_hash absent from this run) —
    the same "present, not missing" shape as :func:`get_stage_presence`.
    """
    if settings.is_remote:
        return _remote_chunk_detail(cast(str, settings.store_uri), run_id, source_hash)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    shard_dir = run_dir / "documents"  # glob yields nothing if it doesn't exist yet
    paths = {
        suffix: [str(p) for p in sorted(shard_dir.glob(f"*{suffix}"))]
        for _key, suffix, _schema in _CHUNK_DETAIL_SIDECARS
    }
    return _chunk_detail(paths, source_hash)


# ---------------------------------------------------------------------------
# Run logs
# ---------------------------------------------------------------------------
#
# Batch logs are the per-document failure lines `operations/extract.py` already
# emits, teed to `runs/<run_id>/logs/batch-NNNN.log` by the worker and
# `cmd_run` (via `utils/run_log.capture_batch_log`). They read with the same
# local/remote fork every other reader uses. `None` means the run does not
# exist (→ 404); an empty list means the run exists but predates this change
# (→ an explained empty state, not a 404).


def list_run_logs(settings: UISettings, run_id: str) -> list[dict] | None:
    """Batch logs for run_id (``name``, ``size``, ``modified``), or None if absent.

    Newest batch first — the batch an operator is most likely watching. A run
    with no ``logs/`` prefix (every run written before this change) lists
    empty rather than 404-ing.
    """
    if settings.is_remote:
        return _remote_run_logs(cast(str, settings.store_uri), run_id)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return []
    entries = [
        _log_entry(p.name, p.stat().st_size, p.stat().st_mtime)
        for p in logs_dir.iterdir()
        if p.is_file() and is_safe_log_name(p.name)
    ]
    return sorted(entries, key=lambda e: e["name"], reverse=True)


def read_run_log(settings: UISettings, run_id: str, name: str) -> str | None:
    """The text of one batch log, or None if the run, or that log, is absent.

    *name* is validated by :func:`is_safe_log_name` before any join — an unsafe
    or absent name both return None, which the route renders as one 404 carrying
    the available list (the caller cannot distinguish rejected from absent, so
    the endpoint cannot be used to probe outside the ``logs/`` prefix).
    """
    if not is_safe_log_name(name):
        return None
    if settings.is_remote:
        return _remote_run_log(cast(str, settings.store_uri), run_id, name)
    run_dir = cast(Path, settings.output_root) / run_id
    if not run_dir.is_dir():
        return None
    log_path = run_dir / "logs" / name
    if not log_path.is_file():
        return None
    return log_path.read_text(encoding="utf-8", errors="replace")


def _log_entry(name: str, size: int, mtime: float) -> dict:
    from datetime import UTC, datetime

    return {
        "name": name,
        "size": size,
        "modified": datetime.fromtimestamp(mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Operator-saved presets (docs/ui-plan.md merge 9)
# ---------------------------------------------------------------------------
#
# Presets get the same local-vs-store treatment as other console
# state, for the same reason: a store-backed (compose) console mounts its run source read-only, so
# saving to a local ``presets_dir`` would 409 there. The store branch writes /
# lists / deletes under the store's own ``presets/`` prefix (a sibling of
# ``runs/``), and the container needs no writable mount.
# ``womblex.ui.presets`` owns the *format* (filename, bytes, parsing); these
# functions own *where* it sits, the same split the rest of this module keeps.


def list_all_presets(settings: UISettings) -> list[dict]:
    """Built-in presets, then any operator-saved ones (a saved name shadows).

    Reads saved presets from the store's ``presets/`` prefix in remote mode,
    or from ``presets_dir`` locally (empty when none is configured).
    """
    return presets_mod.merge_saved(_read_saved_presets(settings))


def get_any_preset(settings: UISettings, name: str) -> Preset | None:
    """One preset by name (a saved one shadowing a built-in), or ``None``."""
    if not presets_mod.is_safe_preset_name(name):
        return presets_mod.PRESETS.get(name)
    return presets_mod.resolve_one(_read_saved_presets(settings), name)


def save_preset(
    settings: UISettings,
    *,
    name: str,
    description: str,
    formats: tuple[str, ...],
    config: dict,
) -> Preset:
    """Save one preset; return the saved :class:`~womblex.ui.presets.Preset`.

    The overlay is validated before either branch writes (a preset that would
    not load is refused, not stored). Raises ``ValueError`` on an unsafe name,
    ``pydantic.ValidationError`` on a non-loadable overlay — the route maps both
    to 400. Callers gate on :attr:`UISettings.presets_writable` first (409).
    """
    record, overlay = presets_mod.build_preset_record(
        name=name, description=description, formats=formats, config=config,
    )
    body = presets_mod.serialise_preset_record(record)
    filename = presets_mod.preset_filename(name)
    if settings.is_remote:
        store = _open_store(cast(str, settings.store_uri))
        with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
            local = Path(tmp) / filename
            local.write_text(body, encoding="utf-8")
            store.upload_file(local, f"{presets_mod.PRESETS_DIRNAME}/{filename}")
    else:
        presets_dir = cast(Path, settings.presets_dir)
        presets_dir.mkdir(parents=True, exist_ok=True)
        (presets_dir / filename).write_text(body, encoding="utf-8")
    return presets_mod.Preset(
        name=name, description=description, formats=formats, config=overlay, source="saved",
    )


def delete_saved_preset(settings: UISettings, name: str) -> bool:
    """Delete a saved preset; True if removed, False if absent (or a built-in).

    A built-in is code, so a name matching only a built-in — or an unsafe name
    — returns False untouched; only a file/object under the presets location is
    ever removed.
    """
    if not presets_mod.is_safe_preset_name(name):
        return False
    filename = presets_mod.preset_filename(name)
    if settings.is_remote:
        store = _open_store(cast(str, settings.store_uri))
        key = f"{presets_mod.PRESETS_DIRNAME}/{filename}"
        if not store.exists(key):
            return False
        store.delete(key)
        return True
    presets_dir = cast(Path, settings.presets_dir)
    path = presets_dir / filename
    if not path.is_file():
        return False
    path.unlink()
    return True


def _read_saved_presets(settings: UISettings) -> dict[str, Preset]:
    """Every operator-saved preset, keyed by name — store prefix or local dir.

    A file that will not parse or whose overlay will not load is skipped (not
    fatal): one corrupt preset must not blank the dropdown, the same
    skip-and-continue the sidecar readers apply.
    """
    saved: dict[str, Preset] = {}
    if settings.is_remote:
        store = _open_store(cast(str, settings.store_uri))
        for key in store.list_files(presets_mod.PRESETS_DIRNAME, "*"):
            filename = key.rsplit("/", 1)[-1]
            name = presets_mod.preset_name_from_filename(filename)
            if name is None:
                continue
            preset = presets_mod.parse_saved_preset(name, store.read_text(key))
            if preset is not None:
                saved[name] = preset
        return saved
    presets_dir = settings.presets_dir
    if presets_dir is None or not presets_dir.is_dir():
        return saved
    for path in sorted(presets_dir.glob("*")):
        name = presets_mod.preset_name_from_filename(path.name)
        if name is None:
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        preset = presets_mod.parse_saved_preset(name, body)
        if preset is not None:
            saved[name] = preset
    return saved


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


def _local_manifest_table(run_dir: Path) -> pa.Table:
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        return pq.read_table(str(manifest_path))
    return read_manifest(run_dir / "documents")


def _conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reindex *table* to *schema*, null-filling columns a drifted shard predates.

    The same compat shim ``store.output._read_chunks_shard`` applies (see
    ``_CHUNKS_BACKFILL``): a run written before a column existed must not
    hand the frontend rows whose key set differs batch to batch.
    """
    for field in schema:
        if field.name not in table.schema.names:
            table = table.append_column(field, pa.nulls(table.num_rows, type=field.type))
    return table.select([f.name for f in schema]).cast(schema)


def _read_filtered(
    paths: list[str],
    schema: pa.Schema,
    value: str,
    *,
    filesystem: object | None = None,
) -> list[dict]:
    """Rows for source_hash *value* across *paths*, conformed to *schema*.

    The predicate is pushed into the parquet reader rather than applied after
    a full read, so a whole-corpus sidecar costs only the row groups whose
    statistics admit *value*. The hash column is resolved per file
    (:func:`_hash_column`) so a pre-0.6.0 enrichment sidecar reads too, and the
    rows come back under the canonical name either way. Same
    skip-unreadable-and-warn policy as :func:`_scan_stage_presence` — one
    corrupt batch narrows a document's overlay data, it doesn't blank the
    screen.
    """
    rows: list[dict] = []
    for p in paths:
        try:
            column = _hash_column(p, filesystem=filesystem)
            table = pq.read_table(p, filesystem=filesystem, filters=[(column, "=", value)])
            if column != "source_hash":
                table = table.rename_columns(
                    ["source_hash" if n == column else n for n in table.schema.names]
                )
            if schema is MONEY_SPANS_SCHEMA:
                table = money_spans_from_legacy(table)
            rows.extend(_conform(table, schema).to_pylist())
        except Exception as e:
            logger.warning("chunk-detail: skipping unreadable sidecar %s: %s", p, e)
    return rows


def _hash_column(path: str, *, filesystem: object | None = None) -> str:
    """``source_hash``, or the pre-0.6.0 name when the file carries that instead.

    One footer read. The store reader's legacy shim on the console's pushdown
    path, which reads parquet directly rather than through it; both go in 0.7.0.
    """
    names = pq.read_schema(path, filesystem=filesystem).names
    return "source_hash" if "source_hash" in names else LEGACY_HASH_COLUMN


def _chunk_detail(
    paths_by_suffix: dict[str, list[str]],
    source_hash: str,
    *,
    filesystem: object | None = None,
) -> dict:
    """Assemble the Chunk Inspector payload from already-located sidecar paths.

    Local and store-backed reads share this body; only path resolution and
    the filesystem differ.
    """
    detail: dict = {}
    for key, suffix, schema in _CHUNK_DETAIL_SIDECARS:
        detail[key] = _read_filtered(
            paths_by_suffix.get(suffix, []), schema, source_hash,
            filesystem=filesystem,
        )
    # chunk_index is re-sequenced per document across narrative *and* table
    # chunks (process/chunker.py), so it totally orders a document's chunks.
    detail["chunks"].sort(key=lambda r: r["chunk_index"])
    # Only narrative-locus money anchors to chunk text; table_cell / sheet_cell
    # spans anchor to the cell sidecars and have no offset to overlay here.
    detail["money_spans"] = [r for r in detail["money_spans"] if r["locus"] == "narrative"]
    _chunk_relative_pii(detail)
    # `value` is decimal128(38,4) — exact by contract. FastAPI's encoder turns
    # a Decimal into a float, which silently loses that exactness, so it goes
    # over the wire as a string.
    for span in detail["money_spans"]:
        span["value"] = None if span["value"] is None else str(span["value"])
    return detail


def _chunk_relative_pii(detail: dict) -> None:
    """Give each PII span its position in its chunk's text.

    A span located to the span has offsets into the source (narrative or table
    markdown); the inspector slices a chunk, so they become relative to the
    chunk's ``start_char``. Any coarser level has no offset: ``-1``, so the front
    end skips it rather than slicing at ``null``.
    """
    starts = {c["chunk_index"]: c["start_char"] for c in detail["chunks"]}
    for span in detail["pii_spans"]:
        origin = starts.get(span["chunk_index"])
        cs, ce = span["char_start"], span["char_end"]
        if span["anchor_level"] != "span" or origin is None or cs is None or ce is None:
            span["start"] = span["end"] = -1
        else:
            span["start"], span["end"] = cs - origin, ce - origin


def _scan_stage_presence(shard_dir: Path, suffix: str) -> list[str]:
    """source_hash values across every ``*<suffix>`` sidecar in *shard_dir*.

    Every sidecar carries them in ``source_hash``; a pre-0.6.0 enrichment
    sidecar carries the same values in the legacy column, resolved per file by
    :func:`_hash_column`. Reading the wrong column is why ``enrich`` presence
    once came back empty.

    An unreadable sidecar is warned about and skipped rather than failing the
    whole screen: presence is an annotation on the documents grid, and one
    corrupt batch should not blank the other batches' answer. ``verify-shards``
    is the surface that reports corruption as such.
    """
    hashes: set[str] = set()
    for p in sorted(shard_dir.glob(f"*{suffix}")):
        if p.name.endswith(ARCHIVE_SUFFIX):
            continue
        try:
            column = _hash_column(str(p))
            col = pq.read_table(str(p), columns=[column]).column(column)
        except Exception as e:
            logger.warning("stage-presence: skipping unreadable sidecar %s: %s", p.name, e)
            continue
        hashes.update(h for h in col.to_pylist() if h)
    return sorted(hashes)


# ---------------------------------------------------------------------------
# Remote (store-backed)
# ---------------------------------------------------------------------------


def _open_store(store_uri: str) -> RemoteStore:
    """Open the store at *store_uri*, or raise :class:`StoreUnreachable`.

    ``RemoteStore.from_uri`` raises ``ImportError`` when the fsspec backend for
    the URI's scheme is not installed (the ``s3fs`` case), and various
    connection errors when the endpoint is set but unreachable. All become one
    catchable, message-preserving :class:`StoreUnreachable` so a store fault
    reads as a legible 503, not an opaque 500.
    """
    from womblex.store.remote import RemoteStore

    try:
        return RemoteStore.from_uri(store_uri)
    except Exception as e:
        logger.warning("store unreachable (%s): %s", store_uri, e)
        raise StoreUnreachable(str(e)) from e


def _list_remote_runs(store_uri: str) -> list[RunDescription]:
    store = _open_store(store_uri)
    return [_describe_remote_run(store, run_id) for run_id in store.list_dirs("runs")]


def _remote_manifest_rows(store_uri: str, run_id: str) -> list[dict] | None:
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    return list(_remote_manifest_table(store, f"runs/{run_id}").to_pylist())


def _describe_remote_run(store: RemoteStore, run_id: str) -> RunDescription:
    prefix = f"runs/{run_id}"
    table = _remote_manifest_table(store, prefix)
    return RunDescription(
        run_id=run_id,
        document_count=table.num_rows,
        stages=_remote_stages_present(store, prefix),
        created_at=None,  # object stores don't expose this uniformly
        updated_at=None,
    )


def _remote_manifest_table(store: RemoteStore, prefix: str) -> pa.Table:
    """Stage *prefix*'s manifest into a temp dir and read it with the local reader.

    Prefers the consolidated ``manifest.parquet`` and falls back to the
    per-batch shard manifests, matching :func:`describe_run`'s local
    precedence — so a distributed run reads correctly both before and after
    ``womblex finalize`` has published the consolidated file.

    Assumes *prefix* names an existing run; callers check that first against
    ``list_dirs("runs")``.
    """
    manifest_key = f"{prefix}/{RUN_MANIFEST_FILENAME}"
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        if store.exists(manifest_key):
            local = store.download_file(manifest_key, tmp_dir / RUN_MANIFEST_FILENAME)
            return pq.read_table(str(local))
        documents_dir = tmp_dir / "documents"
        documents_dir.mkdir()
        manifest_keys = store.list_files(f"{prefix}/documents", "*._manifest.parquet")
        if manifest_keys:
            store.download_to_dir(manifest_keys, documents_dir)
        return read_manifest(documents_dir)


def _remote_stages_present(store: RemoteStore, prefix: str) -> tuple[str, ...]:
    return tuple(
        stage for stage, suffix in STAGE_SUFFIXES.items()
        if store.list_files(f"{prefix}/documents", f"*{suffix}")
    )


def _remote_stage_presence(store_uri: str, run_id: str, suffix: str) -> list[str] | None:
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    keys = store.list_files(f"runs/{run_id}/documents", f"*{suffix}")
    if not keys:
        return []
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        store.download_to_dir(keys, tmp_dir)
        return _scan_stage_presence(tmp_dir, suffix)


def _remote_chunk_detail(store_uri: str, run_id: str, source_hash: str) -> dict | None:
    """Read the overlay sidecars in place, filtered to one document.

    The only reader here that does **not** stage into a temp dir first. The
    others answer whole-run questions about small files (manifests) or must
    read every shard anyway (the audit); this one wants one document out of
    sidecars that span the entire corpus, so staging them would transfer
    every chunk of every document to render one page. Reading through the
    store's own fsspec filesystem pushes the ``source_hash`` predicate into
    the parquet reader instead.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    prefix = f"runs/{run_id}/documents"
    # `list_files` returns store-relative keys; pyarrow needs them rooted the
    # way RemoteStore roots its own filesystem calls.
    paths = {
        suffix: [f"{store.root}/{key}" for key in store.list_files(prefix, f"*{suffix}")]
        for _key, suffix, _schema in _CHUNK_DETAIL_SIDECARS
    }
    return _chunk_detail(paths, source_hash, filesystem=store.fs)


def _remote_shard_audit(store_uri: str, run_id: str) -> dict | None:
    """Stage the four extraction-role shards and audit them locally.

    Restricted to ``_SHARD_SUFFIX`` rather than ``*.parquet`` on purpose:
    the audit reads only those roles, and a run's embeddings / enrichment
    sidecars are the largest files in it — globbing everything would pull
    the whole corpus over the network to count element kinds.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    prefix = f"runs/{run_id}/documents"
    with tempfile.TemporaryDirectory(prefix="womblex-ui-") as tmp:
        tmp_dir = Path(tmp)
        for suffix in _SHARD_SUFFIX.values():
            keys = store.list_files(prefix, f"*{suffix}")
            if keys:
                store.download_to_dir(keys, tmp_dir)
        return audit_shard_directory(tmp_dir).as_dict()


def _remote_run_logs(store_uri: str, run_id: str) -> list[dict] | None:
    """List a remote run's batch logs from its ``logs/`` prefix.

    ``None`` when the run does not exist; an empty list when it does but has no
    ``logs/`` prefix (a run written before this change). fsspec does not expose
    an object's size uniformly per-listing, so a per-key ``info`` fetches size
    and mtime for the (small) set of batch logs.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    keys = store.list_files(f"runs/{run_id}/logs", "*")
    entries: list[dict] = []
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        if not is_safe_log_name(name):
            continue
        entries.append(_remote_log_entry(store, key, name))
    return sorted(entries, key=lambda e: e["name"], reverse=True)


def _remote_log_entry(store: RemoteStore, key: str, name: str) -> dict:
    """Size + modified for one remote log, tolerating a backend that omits either."""
    size = 0
    modified: str | None = None
    try:
        info: dict = store.fs.info(f"{store.root}/{key}")  # type: ignore[attr-defined]
        size = int(info.get("size") or info.get("Size") or 0)
        mtime = info.get("LastModified") or info.get("mtime")
        if mtime is not None and hasattr(mtime, "strftime"):
            modified = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:  # a listing that cannot be `info`'d is still nameable
        logger.warning("run-logs: could not stat %s: %s", key, e)
    return {"name": name, "size": size, "modified": modified}


def _remote_run_log(store_uri: str, run_id: str, name: str) -> str | None:
    """Read one remote batch log in place; None if the run or the log is absent.

    *name* is already validated by :func:`read_run_log`; the run existence
    check gives a hand-typed run id the same 404 the local branch produces.
    """
    store = _open_store(store_uri)
    if run_id not in store.list_dirs("runs"):
        return None
    key = f"runs/{run_id}/logs/{name}"
    if not store.exists(key):
        return None
    return store.read_text(key)

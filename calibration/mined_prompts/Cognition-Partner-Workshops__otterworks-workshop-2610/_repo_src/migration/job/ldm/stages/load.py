"""LOAD: copybook conversion of unload files into stg.* with row-level rejects (CONTRACTS.md §9.2, §9.4).

Two engines share the per-range bookkeeping here (`begin_range` / `finish_range`): the serial engine below converts
and inserts in this process; `execution.load_engine: spark` hands the range to `ldm.engines.spark_load`.
"""

from __future__ import annotations

from pathlib import Path

from ..context import RunContext, TableSpec
from ..convert import convert_record
from ..drivers.base import InsertFailure, KeyRange, Reject, StagedRow
from ..errors import ConfigError, ReconcileError
from ..staging import sha256_file

# ODBC SQLSTATE / SQL Server native error -> reject rule for rows the target itself refused
_NATIVE_RULES = {2601: "DUPLICATE_SOURCE_KEY", 2627: "DUPLICATE_SOURCE_KEY"}
_SQLSTATE_RULES = {
    "23000": "DUPLICATE_SOURCE_KEY",
    "23505": "DUPLICATE_SOURCE_KEY",  # PostgreSQL unique_violation
    "22003": "DECIMAL_OVERFLOW",
    "22007": "DATE_INVALID",
    "22001": "STRING_TRUNCATION",
}


def rule_for(f: InsertFailure) -> str:
    if f.native_error in _NATIVE_RULES:
        return _NATIVE_RULES[f.native_error]
    return _SQLSTATE_RULES.get(f.sqlstate or "", "TARGET_REJECTED")


def field_for(rule: str, key_columns: list[str]) -> str | None:
    if rule == "DUPLICATE_SOURCE_KEY":
        return key_columns[0]
    return None


def failure_reject(f: InsertFailure, row: StagedRow, key_columns: list[str]) -> Reject:
    """The LOAD reject for a row the target refused (rule from SQLSTATE / native error)."""
    rule = rule_for(f)
    return Reject(
        table_name=row.table_name,
        source_key=f.source_key,
        stage="LOAD",
        rule=rule,
        field_name=field_for(rule, key_columns),
        raw_bytes=row.raw_bytes,
        field_bytes=None,
        sqlstate=f.sqlstate,
        native_error=f.native_error,
        error_text=f.error_text[:4000],
        key_range_seq=row.key_range_seq,
    )


def duplicate_reject(table: str, source_key: str, key_column: str, rec: bytes, range_seq: int) -> Reject:
    return Reject(
        table,
        source_key,
        "LOAD",
        "DUPLICATE_SOURCE_KEY",
        key_column,
        rec,
        None,
        "23000",
        2601,
        f"source key {source_key!r} appears more than once in the unload of range {range_seq}",
        range_seq,
    )


def _range_file(ctx: RunContext, ts: TableSpec, rng: KeyRange) -> Path:
    local = Path(rng.local_path) if rng.local_path else ctx.range_dir(ts.name) / f"{rng.range_seq:06d}.dat"
    if not local.exists() or (rng.sha256_hex and sha256_file(local) != rng.sha256_hex):
        if not rng.blob_path or not ctx.blobs.download(rng.blob_path, local):
            raise ConfigError(
                f"{ts.name} range {rng.range_seq}: unload file {local} is missing and blob {rng.blob_path!r} "
                f"is not available"
            )
    if rng.sha256_hex:
        actual = sha256_file(local)
        if actual != rng.sha256_hex:
            raise ConfigError(f"{ts.name} range {rng.range_seq}: checksum {actual} != recorded {rng.sha256_hex}")
    return local


def begin_range(ctx: RunContext, ts: TableSpec, rng: KeyRange) -> tuple[Path, int]:
    """Verify the unload file (checksum, LRECL, .cnt), roll back a half-loaded range, mark it RUNNING.

    Returns (local file, record count).
    """
    local = _range_file(ctx, ts, rng)
    size = local.stat().st_size
    lrecl = ts.config.record_length
    if size % lrecl:
        raise ConfigError(f"{ts.name} range {rng.range_seq}: file size {size} is not a multiple of LRECL {lrecl}")
    if rng.row_count is not None and size // lrecl != rng.row_count:
        raise ConfigError(
            f"{ts.name} range {rng.range_seq}: file has {size // lrecl} records, .cnt says {rng.row_count}"
        )

    if rng.load_status == "RUNNING":
        ctx.log.warn(f"range {rng.range_seq} was interrupted mid-load; replacing its staged rows", ts.name)
        ctx.target.delete_staging_range(ctx.run_id, ctx.namespace, ts.name, rng.range_seq)
        ctx.target.delete_rejects_range(ctx.run_id, ctx.namespace, ts.name, "LOAD", rng.range_seq)
    ctx.target.update_key_range(ctx.run_id, ctx.namespace, ts.name, rng.range_seq, load_status="RUNNING")
    return local, size // lrecl


def finish_range(ctx: RunContext, ts: TableSpec, rng: KeyRange, loaded: int, rejected: int) -> tuple[int, int]:
    ctx.target.update_key_range(ctx.run_id, ctx.namespace, ts.name, rng.range_seq, load_status="DONE")
    ctx.log.info(f"range {rng.range_seq} loaded={loaded} rejected={rejected}", ts.name)
    return loaded, rejected


def load_range(ctx: RunContext, ts: TableSpec, rng: KeyRange) -> tuple[int, int]:
    """Serial engine: return (loaded, rejected) for one key range."""
    m = ctx.manifest
    local, _count = begin_range(ctx, ts, rng)
    data = local.read_bytes()
    lrecl = ts.config.record_length

    loaded = rejected = 0
    seen_keys: set[str] = set()
    batch: list[StagedRow] = []
    rejects: list[Reject] = []

    def flush() -> None:
        nonlocal loaded, rejected
        if not batch:
            return
        failures = ctx.target.insert_staging(ctx.run_id, ctx.namespace, ts.name, batch)
        failed_keys = {f.source_key for f in failures}
        loaded += len(batch) - len(failed_keys)
        by_key = {r.source_key: r for r in batch}
        rejects.extend(failure_reject(f, by_key[f.source_key], ts.config.key_columns) for f in failures)
        rejected += len(failed_keys)
        batch.clear()
        if rejects:
            ctx.target.insert_rejects(ctx.run_id, ctx.namespace, rejects)
            rejects.clear()

    for i in range(0, len(data), lrecl):
        rec = data[i : i + lrecl]
        conv = convert_record(rec, ts.columns)
        if conv.ok and conv.source_key in seen_keys:
            rejects.append(duplicate_reject(ts.name, conv.source_key, ts.config.key_columns[0], rec, rng.range_seq))
            rejected += 1
            continue
        if not conv.ok:
            err = conv.error
            assert err is not None
            rejects.append(
                Reject(
                    ts.name,
                    conv.source_key,
                    "LOAD",
                    err.rule,
                    err.column,
                    rec,
                    err.field_bytes,
                    err.sqlstate,
                    None,
                    err.error[:4000],
                    rng.range_seq,
                )
            )
            rejected += 1
            continue
        seen_keys.add(conv.source_key)
        batch.append(StagedRow(ts.name, conv.source_key, rng.range_seq, rec, conv.target_row()))
        if len(batch) >= m.batch.load_batch_rows:
            flush()
    flush()
    if rejects:
        ctx.target.insert_rejects(ctx.run_id, ctx.namespace, rejects)
    return finish_range(ctx, ts, rng, loaded, rejected)


def load_range_with_engine(ctx: RunContext, ts: TableSpec, rng: KeyRange) -> tuple[int, int]:
    if ctx.manifest.execution.load_engine == "spark":
        from ..engines.spark_load import load_range_spark

        return load_range_spark(ctx, ts, rng)
    return load_range(ctx, ts, rng)


def load_table(ctx: RunContext, ts: TableSpec) -> dict[str, int]:
    ranges = ctx.target.get_key_ranges(ctx.run_id, ctx.namespace, ts.name)
    if not ranges:
        extracted_so_far = ctx.target.get_ledger(ctx.run_id, ctx.namespace)[ts.name].extracted
        if extracted_so_far is None:
            raise ConfigError(f"{ts.name}: no key ranges recorded; run extract first")
        if extracted_so_far != 0:
            raise ConfigError(f"{ts.name}: ledger extracted={extracted_so_far} but no key ranges recorded")
        ctx.log.info("extract selected 0 rows; nothing to load", ts.name)
    pending = [r for r in ranges if r.status != "DONE"]
    if pending:
        raise ConfigError(f"{ts.name}: {len(pending)} key range(s) not extracted (status != DONE)")
    for rng in sorted(ranges, key=lambda r: r.range_seq):
        if rng.load_status == "DONE":
            continue
        load_range_with_engine(ctx, ts, rng)
    ledger = ctx.target.get_ledger(ctx.run_id, ctx.namespace)[ts.name]
    loaded = ctx.target.count_staging(ctx.run_id, ctx.namespace, ts.name)
    rejected = ctx.target.count_rejects(ctx.run_id, ctx.namespace, ts.name, "LOAD")
    ctx.target.update_ledger(ctx.run_id, ctx.namespace, ts.name, loaded=loaded, rejected=rejected)
    extracted = ledger.extracted if ledger.extracted is not None else -1
    if extracted != loaded + rejected:
        raise ReconcileError(f"{ts.name}: extracted {extracted} != loaded {loaded} + rejected {rejected}")
    return {"loaded": loaded, "rejected": rejected}


def run(ctx: RunContext) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    ctx.log.info(f"engine={ctx.manifest.execution.load_engine}")
    for ts in ctx.tables_in_order():
        log_id = ctx.target.stage_log_start(ctx.run_id, ctx.namespace, "LOAD", ts.name)
        try:
            out[ts.name] = load_table(ctx, ts)
        except Exception as e:
            ctx.target.stage_log_finish(log_id, "FAILED", None, str(e)[:4000])
            raise
        ctx.target.stage_log_finish(log_id, "OK", out[ts.name]["loaded"], None)
    return out

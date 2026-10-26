"""PURGE: delete only purge_safe keys from the source, audit first, transactional batches, guarded."""

from __future__ import annotations

import time

from ..context import RunContext, TableSpec
from ..errors import LdmError, PurgeGuardError, SourceError

_sleep = time.sleep

# Source SQLSTATEs for which the rolled-back batch is retried: 57011 transaction log full (Db2 SQL0964C),
# 57033 lock timeout without automatic rollback, 40001 deadlock / lock-timeout rollback. A driver that knows
# better (Oracle: ORA-01555 / ORA-00060 / ORA-30036, no SQLSTATE from the client) sets SourceError.transient.
TRANSIENT_SOURCE_SQLSTATES = frozenset({"57011", "57033", "40001"})


def is_transient(e: LdmError) -> bool:
    if not isinstance(e, SourceError):
        return False
    if e.transient is not None:
        return e.transient
    return e.sqlstate in TRANSIENT_SOURCE_SQLSTATES


def _purge_batch_with_retry(ctx: RunContext, ts: TableSpec, chunk: list[str], batch_no: int) -> int:
    m = ctx.manifest
    cfg = ts.config
    attempts = m.batch.purge_retry_attempts
    for attempt in range(attempts + 1):
        ctx.target.insert_purge_audit(ctx.run_id, ctx.namespace, ts.name, chunk, batch_no)
        try:
            return ctx.source.purge_batch(
                cfg.schema_, cfg.name, ts.key_column, chunk, ctx.run_id, ctx.namespace, batch_no
            )
        except LdmError as e:
            ctx.target.set_purge_audit_status(ctx.run_id, ctx.namespace, ts.name, chunk, "ROLLED_BACK")
            if attempt >= attempts or not is_transient(e):
                raise
            delay = min(m.batch.purge_retry_backoff_s * 2**attempt, m.batch.purge_retry_max_backoff_s)
            ctx.log.warn(
                f"batch {batch_no} rolled back by source (SQLCODE={e.sqlcode} SQLSTATE={e.sqlstate}); "
                f"retry {attempt + 1}/{attempts} in {delay:g}s",
                ts.name,
            )
            _sleep(delay)
    raise AssertionError("unreachable")


def guard(ctx: RunContext, ts: TableSpec) -> tuple[list[str], int]:
    """Return (purge_safe keys, ledger validated) or raise PurgeGuardError when they disagree."""
    keys = ctx.target.purge_safe_keys(ctx.run_id, ctx.namespace, ts.name)
    ledger = ctx.target.get_ledger(ctx.run_id, ctx.namespace)[ts.name]
    intended = len(keys)
    validated = ledger.validated if ledger.validated is not None else -1
    ctx.target.update_ledger(
        ctx.run_id, ctx.namespace, ts.name, purge_intended=intended, purge_dry_run=not ctx.manifest.purge
    )
    if intended != validated:
        raise PurgeGuardError(
            f"{ts.name}: purge guard: {intended} purge_safe key(s) but ledger validated={validated}; nothing deleted"
        )
    return keys, validated


def purge_table(ctx: RunContext, ts: TableSpec, keys: list[str], validated: int) -> dict[str, int]:
    m = ctx.manifest
    cfg = ts.config
    intended = len(keys)
    if not m.purge:
        ctx.log.info(
            f"DRY RUN: would delete {intended} row(s) from {cfg.qualified_name} (overlay purge: false)",
            ts.name,
        )
        ctx.target.update_ledger(ctx.run_id, ctx.namespace, ts.name, purged=0)
        return {"purge_intended": intended, "purged": 0, "dry_run": 1}

    already = ctx.target.purged_keys(ctx.run_id, ctx.namespace, ts.name)
    todo = [k for k in keys if k not in already]
    if already:
        ctx.log.info(f"resuming: {len(already)} key(s) already PURGED in mig.purge_audit", ts.name)
    if todo:
        committed = ctx.source.audited_keys(ctx.run_id, cfg.name, todo)
        if committed:
            recovered = [k for k in todo if k in committed]
            ctx.target.insert_purge_audit(ctx.run_id, ctx.namespace, ts.name, recovered, 0)
            ctx.target.set_purge_audit_status(ctx.run_id, ctx.namespace, ts.name, recovered, "PURGED")
            ctx.log.info(
                f"resuming: {len(recovered)} key(s) already committed to MIGAUDIT.PURGE_AUDIT; marked PURGED", ts.name
            )
            todo = [k for k in todo if k not in committed]
    batch_rows = m.batch.purge_batch_rows
    batch_no = 0
    for i in range(0, len(todo), batch_rows):
        batch_no += 1
        chunk = todo[i : i + batch_rows]
        try:
            deleted = _purge_batch_with_retry(ctx, ts, chunk, batch_no)
        except LdmError as e:
            ctx.target.update_ledger(
                ctx.run_id,
                ctx.namespace,
                ts.name,
                purged=len(ctx.target.purged_keys(ctx.run_id, ctx.namespace, ts.name)),
            )
            ctx.log.error(f"batch {batch_no} rolled back: {e}", ts.name)
            raise
        ctx.target.set_purge_audit_status(ctx.run_id, ctx.namespace, ts.name, chunk, "PURGED")
        ctx.log.info(f"batch {batch_no}: deleted {deleted} row(s)", ts.name)
    purged = len(ctx.target.purged_keys(ctx.run_id, ctx.namespace, ts.name))
    ctx.target.update_ledger(ctx.run_id, ctx.namespace, ts.name, purged=purged)
    if purged != validated:
        raise PurgeGuardError(f"{ts.name}: purged {purged} != validated {validated} after all batches")
    return {"purge_intended": intended, "purged": purged, "dry_run": 0}


def run(ctx: RunContext) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ts in ctx.tables_in_order():
        if ts.config.role != "data":
            ctx.target.update_ledger(
                ctx.run_id,
                ctx.namespace,
                ts.name,
                purge_intended=0,
                purged=0,
                purge_dry_run=not ctx.manifest.purge,
            )
            out[ts.name] = {"purge_intended": 0, "purged": 0}
    data_tables = list(reversed(ctx.data_tables()))  # children before parents
    # every table's guard is evaluated before the first DELETE, so a guard failure deletes nothing anywhere
    guarded: dict[str, tuple[list[str], int]] = {}
    for ts in data_tables:
        log_id = ctx.target.stage_log_start(ctx.run_id, ctx.namespace, "PURGE", ts.name)
        try:
            guarded[ts.name] = guard(ctx, ts)
        except Exception as e:
            ctx.target.stage_log_finish(log_id, "FAILED", None, str(e)[:4000])
            raise
        ctx.target.stage_log_finish(log_id, "OK", None, f"guard ok: intended={guarded[ts.name][1]}")
    for ts in data_tables:
        log_id = ctx.target.stage_log_start(ctx.run_id, ctx.namespace, "PURGE", ts.name)
        try:
            out[ts.name] = purge_table(ctx, ts, *guarded[ts.name])
        except Exception as e:
            ctx.target.stage_log_finish(log_id, "FAILED", None, str(e)[:4000])
            raise
        ctx.target.stage_log_finish(log_id, "OK", out[ts.name]["purged"], None)
    return out

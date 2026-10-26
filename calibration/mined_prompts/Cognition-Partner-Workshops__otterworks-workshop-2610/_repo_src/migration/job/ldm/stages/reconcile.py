"""RECONCILE: ledger + rejects + purge audit -> mig.runs closure and JSON/CSV/HTML evidence files."""

from __future__ import annotations

import glob
import os
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..context import RunContext
from ..drivers.base import LedgerRow
from ..errors import EXIT_OK, EXIT_RECONCILE, ConfigError, ReconcileError
from ..report import (
    ClassTotalEntry,
    FailureEntry,
    Report,
    SessionEntry,
    TableRow,
    dec8,
    to_csv,
    to_html,
    to_json,
)


def load_session_links(ctx: RunContext) -> list[tuple[str, str]]:
    """Per-unit `migration/sessions/<unit>.yaml` files (sorted by name) followed by DEVIN_SESSION_LINKS."""
    links: list[tuple[str, str]] = []
    pattern = str(ctx.loaded.resolve(ctx.manifest.report.sessions_glob))
    for path in sorted(glob.glob(pattern)):
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
        if not isinstance(doc, list):
            raise ConfigError(f"{path}: expected a YAML list of {{label, url}} entries")
        for entry in doc:
            if not isinstance(entry, dict) or "label" not in entry or "url" not in entry:
                raise ConfigError(f"{path}: every entry needs 'label' and 'url'")
            links.append((str(entry["label"]), str(entry["url"])))
    raw = ctx.env.get("DEVIN_SESSION_LINKS") or os.environ.get("DEVIN_SESSION_LINKS") or ""
    for item in filter(None, (p.strip() for p in raw.split(","))):
        label, sep, url = item.partition("=")
        if not sep or not url:
            raise ConfigError(f"DEVIN_SESSION_LINKS entry {item!r} is not 'label=url'")
        links.append((label.strip(), url.strip()))
    return links


def table_closes(row: LedgerRow, purge_enabled: bool) -> tuple[bool, str | None]:
    """Same arithmetic as mig.v_reconciliation_tables.closes, with the reason when it fails."""
    ext = row.extracted if row.extracted is not None else -1
    loaded = row.loaded if row.loaded is not None else -1
    rejected = row.rejected or 0
    validated = row.validated or 0
    vfailed = row.validate_failed or 0
    purged = row.purged or 0
    intended = row.purge_intended if row.purge_intended is not None else -1
    if ext != (row.loaded or 0) + rejected:
        return False, f"extracted {ext} != loaded {row.loaded or 0} + rejected {rejected}"
    if loaded != validated + vfailed:
        return False, f"loaded {loaded} != validated {validated} + validate_failed {vfailed}"
    if row.table_role == "reference":
        if purged != 0:
            return False, f"reference table purged {purged} != 0"
        return True, None
    if purge_enabled:
        if (row.purged if row.purged is not None else -1) != validated:
            return False, f"purged {row.purged} != validated {validated}"
    elif intended != validated or purged != 0:
        return False, f"dry run: purge_intended {intended} != validated {validated} or purged {purged} != 0"
    return True, None


def build_report(ctx: RunContext, closes: bool) -> Report:
    ledger = ctx.target.get_ledger(ctx.run_id, ctx.namespace)
    tables: list[TableRow] = []
    for row in sorted(ledger.values(), key=lambda r: r.table_order):
        tables.append(
            TableRow(
                table=row.table_name,
                extracted=row.extracted or 0,
                loaded=row.loaded or 0,
                validated=row.validated or 0,
                purged=row.purged or 0,
                failed=(row.rejected or 0) + (row.validate_failed or 0),
            )
        )
    failures: list[FailureEntry] = []
    for f in sorted(ctx.target.failures(ctx.run_id, ctx.namespace), key=lambda f: (f.table_name, f.source_key)):
        failures.append(
            FailureEntry(
                table=f.table_name,
                source_key=f.source_key.rstrip(" "),
                rule=f.rule,
                stage=f.stage,
                field=f.field_name,
                sqlstate=f.sqlstate,
                native_error=f.native_error,
                error=f.error_text,
            )
        )
    sessions = [SessionEntry(label=lbl, url=url) for lbl, url in load_session_links(ctx)]
    class_totals = [
        ClassTotalEntry(
            {
                "table": c.table_name,
                "class": c.retention_class,
                "source_count": c.source_count,
                "target_count": c.target_count,
                "source_sum": dec8(c.source_sum),
                "target_sum": dec8(c.target_sum),
                "status": c.status,
            }
        )
        for c in ctx.target.get_class_totals(ctx.run_id, ctx.namespace)
    ]
    return Report(
        run_id=ctx.run_id,
        namespace=ctx.namespace,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        tables=tables,
        failures=failures,
        sessions=sessions,
        closes=closes,
        class_totals=class_totals,
    )


def write_report_files(ctx: RunContext, report: Report) -> dict[str, Path]:
    out_dir = ctx.local_dir / ctx.blob_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = {
        "reconciliation.json": (to_json(report), "application/json"),
        "reconciliation.csv": (to_csv(report), "text/csv; charset=utf-8"),
        "reconciliation.html": (to_html(report), "text/html; charset=utf-8"),
    }
    paths: dict[str, Path] = {}
    for name, (text, ctype) in rendered.items():
        p = out_dir / name
        data = text.encode("utf-8")
        p.write_bytes(data)
        ctx.blobs.upload_bytes(data, f"{ctx.blob_prefix}{name}", ctype)
        paths[name] = p
    return paths


def run(ctx: RunContext) -> dict[str, dict[str, int]]:
    log_id = ctx.target.stage_log_start(ctx.run_id, ctx.namespace, "RECONCILE", None)
    try:
        sessions = load_session_links(ctx)
        ctx.target.insert_run_sessions(ctx.run_id, ctx.namespace, sessions)
        ledger = ctx.target.get_ledger(ctx.run_id, ctx.namespace)
        purge_enabled = ctx.manifest.purge
        closes = True
        for row in sorted(ledger.values(), key=lambda r: r.table_order):
            ok, why = table_closes(row, purge_enabled)
            if not ok:
                closes = False
                ctx.log.error(f"does not close: {why}", row.table_name)
        report = build_report(ctx, closes)
        paths = write_report_files(ctx, report)
        for p in paths.values():
            ctx.log.info(f"wrote {p}")
        exit_code = EXIT_OK if closes else EXIT_RECONCILE
        if closes:
            ctx.target.delete_staging_run(ctx.run_id, ctx.namespace)
        ctx.target.set_run_status(ctx.run_id, ctx.namespace, "CLOSED" if closes else "FAILED", exit_code)
    except Exception as e:
        ctx.target.stage_log_finish(log_id, "FAILED", None, str(e)[:4000])
        raise
    ctx.target.stage_log_finish(log_id, "OK" if closes else "FAILED", None, None if closes else "run does not close")
    out = {t["table"]: dict(t) for t in report["tables"]}
    for v in out.values():
        v.pop("table", None)
    if not closes:
        raise ReconcileError("run does not close: see table lines above", tables={k: dict(v) for k, v in out.items()})
    return {k: {kk: int(vv) for kk, vv in v.items()} for k, v in out.items()}

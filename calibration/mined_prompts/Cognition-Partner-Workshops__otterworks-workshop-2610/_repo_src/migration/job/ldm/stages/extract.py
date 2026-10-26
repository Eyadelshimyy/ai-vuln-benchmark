"""EXTRACT: manifest selection -> key ranges -> fixed-width unload files with .cnt/.sha256 (CONTRACTS.md §9.2)."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from ..context import RunContext, TableSpec
from ..convert import encode_record
from ..drivers.base import KeyRange, SelectionSpec
from ..errors import ConfigError, LdmError
from ..staging import sha256_file

UNLOAD01_RE = re.compile(r"^UNLOAD01 ROWS=(\d+) BYTES=(\d+) SHA256=([0-9a-fA-F]{64})\s*$", re.MULTILINE)


class UnloadError(LdmError):
    """The unload command failed; exit 1 after the range is marked FAILED."""


def render_predicate(selection: SelectionSpec, table_alias: str = "T") -> str:
    """The selection predicate as SQL, for the unload wrapper (LDM_SELECT_WHERE) and for Db2 SELECTs."""
    if selection.all_rows:
        where = "1 = 1"
    else:
        classes = ", ".join("'" + c.replace("'", "''") + "'" for c in selection.classes)
        where = (
            f"{table_alias}.{selection.class_column} IN ({classes}) "
            f"AND {table_alias}.{selection.last_access_column} < TIMESTAMP('{selection.last_access_before}')"
        )
    # selection.parent is deliberately NOT applied here: children are selected by their own predicate and
    # the parent relation is checked in VALIDATE (ORPHAN_PARENT_NOT_SELECTED).
    return where


def plan_ranges(table: str, keys: list[str], range_rows: int) -> list[KeyRange]:
    ranges: list[KeyRange] = []
    for seq, start in enumerate(range(0, len(keys), range_rows), start=1):
        chunk = keys[start : start + range_rows]
        ranges.append(KeyRange(table_name=table, range_seq=seq, key_from=chunk[0], key_to=chunk[-1]))
    return ranges


def _range_paths(ctx: RunContext, table: str, seq: int) -> tuple[Path, str]:
    fname = f"{seq:06d}.dat"
    local = ctx.range_dir(table) / fname
    blob = f"{ctx.blob_prefix}{table}/{fname}"
    return local, blob


def _builtin_unload(
    ctx: RunContext, ts: TableSpec, selection: SelectionSpec, rng: KeyRange, out: Path
) -> tuple[int, int, str]:
    cfg = ts.config
    columns = cfg.columns
    h = hashlib.sha256()
    rows = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for row in ctx.source.fetch_range(cfg.schema_, cfg.name, columns, selection, rng.key_from, rng.key_to):
            rec = encode_record(row, ts.columns, cfg.record_length)
            f.write(rec)
            h.update(rec)
            rows += 1
    return rows, rows * cfg.record_length, h.hexdigest()


def _command_unload(
    ctx: RunContext, ts: TableSpec, selection: SelectionSpec, rng: KeyRange, out: Path
) -> tuple[int, int, str]:
    m = ctx.manifest
    subs = {
        "table": ts.name,
        "key_from": rng.key_from,
        "key_to": rng.key_to,
        "out_file": str(out),
        "namespace": ctx.namespace,
        "run_id": ctx.run_id,
    }
    argv = [a.format(**subs) for a in m.source.unload_command]
    if argv and not os.path.isabs(argv[0]) and (ctx.loaded.repo_root / argv[0]).exists():
        argv[0] = str(ctx.loaded.repo_root / argv[0])
    env = dict(os.environ)
    env.update(ctx.env)
    env["LDM_SELECT_WHERE"] = render_predicate(selection)
    env["LDM_TABLE_SCHEMA"] = ts.config.schema_
    env["LDM_KEY_COLUMN"] = ts.key_column
    env["LDM_LRECL"] = str(ts.config.record_length)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(argv, env=env, capture_output=True, text=True, cwd=str(ctx.loaded.repo_root))
    except (FileNotFoundError, PermissionError) as e:
        raise ConfigError(
            f"unload_command {argv[0]!r} is not executable ({e.strerror}); install the UNLOAD01 wrapper "
            "or set LDM_UNLOAD_MODE=builtin to use the built-in fixed-width writer"
        ) from e
    for line in proc.stderr.splitlines():
        ctx.log.info(f"unload: {line}", ts.name)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["(no output)"]
        raise UnloadError(f"unload command exited {proc.returncode} for {ts.name} range {rng.range_seq}: {tail[0]}")
    matches = list(UNLOAD01_RE.finditer(proc.stdout))
    m01 = matches[-1] if matches else None
    if m01 is None:
        raise UnloadError(f"unload command printed no 'UNLOAD01 ROWS=.. BYTES=.. SHA256=..' line for {ts.name}")
    return int(m01.group(1)), int(m01.group(2)), m01.group(3).lower()


def run_range(ctx: RunContext, ts: TableSpec, selection: SelectionSpec, rng: KeyRange) -> KeyRange:
    local, blob = _range_paths(ctx, ts.name, rng.range_seq)
    ctx.target.update_key_range(
        ctx.run_id,
        ctx.namespace,
        ts.name,
        rng.range_seq,
        status="RUNNING",
        attempt=rng.attempt + 1,
        local_path=str(local),
    )
    mode = ctx.env.get("LDM_UNLOAD_MODE") or os.environ.get("LDM_UNLOAD_MODE") or "command"
    use_builtin = mode == "builtin" or not ctx.manifest.source.unload_command
    try:
        if use_builtin:
            rows, nbytes, digest = _builtin_unload(ctx, ts, selection, rng, local)
        else:
            rows, nbytes, digest = _command_unload(ctx, ts, selection, rng, local)
        lrecl = ts.config.record_length
        actual_size = local.stat().st_size if local.exists() else -1
        if nbytes != rows * lrecl or actual_size != nbytes:
            raise UnloadError(
                f"{ts.name} range {rng.range_seq}: UNLOAD01 reports ROWS={rows} BYTES={nbytes} but LRECL={lrecl} "
                f"and file size is {actual_size}"
            )
        actual_sha = sha256_file(local)
        if actual_sha != digest:
            raise UnloadError(f"{ts.name} range {rng.range_seq}: SHA-256 mismatch {actual_sha} != reported {digest}")
        local.with_suffix(".cnt").write_text(f"{rows}\n", encoding="ascii")
        local.with_suffix(".sha256").write_text(f"{digest}  {local.name}\n", encoding="ascii")
        for p in (local, local.with_suffix(".cnt"), local.with_suffix(".sha256")):
            ctx.blobs.upload(p, blob.rsplit(".dat", 1)[0] + p.suffix)
    except LdmError as e:
        ctx.target.update_key_range(
            ctx.run_id, ctx.namespace, ts.name, rng.range_seq, status="FAILED", error_text=str(e)[:4000]
        )
        raise
    ctx.target.update_key_range(
        ctx.run_id,
        ctx.namespace,
        ts.name,
        rng.range_seq,
        status="DONE",
        row_count=rows,
        byte_count=nbytes,
        sha256_hex=digest,
        blob_path=blob,
        local_path=str(local),
        error_text=None,
    )
    ctx.log.info(
        f"range {rng.range_seq} [{rng.key_from.strip()}..{rng.key_to.strip()}] rows={rows} sha256={digest[:12]}",
        ts.name,
    )
    return KeyRange(ts.name, rng.range_seq, rng.key_from, rng.key_to, "DONE", rows, nbytes, digest, blob, str(local))


def extract_table(ctx: RunContext, ts: TableSpec) -> dict[str, int]:
    m = ctx.manifest
    selection = ctx.selection_for(ts.config)
    ranges = ctx.target.get_key_ranges(ctx.run_id, ctx.namespace, ts.name)
    if not ranges:
        keys = ctx.source.select_keys(ts.config.schema_, ts.name, selection)
        if len(set(keys)) != len(keys):
            raise ConfigError(f"{ts.name}: source key column {ts.key_column} is not unique within the selection")
        ranges = plan_ranges(ts.name, keys, m.batch.extract_range_rows)
        ctx.target.insert_key_ranges(ctx.run_id, ctx.namespace, ranges)
        ctx.log.info(f"selected {len(keys)} keys -> {len(ranges)} range(s) of <= {m.batch.extract_range_rows}", ts.name)
    else:
        done = sum(1 for r in ranges if r.status == "DONE")
        ctx.log.info(f"resuming: {done}/{len(ranges)} range(s) already DONE", ts.name)
    extracted = 0
    for rng in sorted(ranges, key=lambda r: r.range_seq):
        if rng.status == "DONE":
            extracted += rng.row_count or 0
            continue
        rng = run_range(ctx, ts, selection, rng)
        extracted += rng.row_count or 0
    ctx.target.update_ledger(ctx.run_id, ctx.namespace, ts.name, extracted=extracted, extract_files=len(ranges))
    return {"extracted": extracted, "extract_files": len(ranges)}


def run(ctx: RunContext) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ts in ctx.tables_in_order():
        log_id = ctx.target.stage_log_start(ctx.run_id, ctx.namespace, "EXTRACT", ts.name)
        try:
            out[ts.name] = extract_table(ctx, ts)
        except Exception as e:
            ctx.target.stage_log_finish(log_id, "FAILED", None, str(e)[:4000])
            raise
        ctx.target.stage_log_finish(log_id, "OK", out[ts.name]["extracted"], None)
    return out

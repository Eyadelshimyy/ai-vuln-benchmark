"""VALIDATE: business hash, parent check, per-class counts and sums; purge_safe only for fully clean rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal

from ..context import RunContext, TableSpec
from ..convert import convert_record
from ..drivers.base import ClassAggregate, ClassTotalRow, Reject, StagedRow, ValidationRow
from ..errors import ReconcileError
from ..hashing import hash_expression, render_source_column, render_target_column, source_hash

RULE_HASH = "HASH_MISMATCH"
RULE_ARCHIVE_CONFLICT = "ARCHIVE_CONFLICT"
RULE_ORPHAN = "ORPHAN_PARENT_NOT_SELECTED"
RULE_CLASS_COUNT = "CLASS_COUNT_MISMATCH"
RULE_CLASS_TOTAL = "CLASS_TOTAL_MISMATCH"


@dataclass
class _Candidate:
    row: StagedRow
    source_values: dict[str, object]
    source_hash: bytes
    target_hash: bytes | None
    source_class: str | None = None
    target_class: str | None = None
    rule: str | None = None
    field_name: str | None = None
    error: str | None = None


@dataclass
class _ClassSide:
    count: int = 0
    sums: dict[str, Decimal] = field(default_factory=dict)


def _successor_map(ctx: RunContext, ts: TableSpec) -> dict[str, str]:
    """Db2 policy code -> system-of-record class from the staged reference table (this run)."""
    ct = ts.config.class_totals
    if not ct or not ct.source_class_resolution:
        return {}
    res = ct.source_class_resolution
    lookup = ctx.table(res.lookup_table)
    out: dict[str, str] = {}
    for batch in ctx.target.iter_staging(
        ctx.run_id, ctx.namespace, lookup.name, ctx.manifest.batch.validate_batch_rows
    ):
        for r in batch:
            code = str(r.values[res.lookup_key]).rstrip(" ")
            succ = str(r.values[res.successor_column] or "").rstrip(" ")
            if succ:
                out[code] = succ
    return out


def _source_class(raw_class: str, spec_value_map: dict[str, str], successors: dict[str, str]) -> str:
    code = raw_class.rstrip(" ")
    if code in successors:
        return successors[code]
    return spec_value_map.get(code, code)


def _differing_column(c: _Candidate, ts: TableSpec) -> str:
    for name in ts.config.hash_columns:
        spec = ts.by_name[name]
        if render_source_column(c.source_values[name], spec) != render_target_column(c.row.values, spec):
            return name
    return ts.config.hash_columns[0]


def _persist(ctx: RunContext, ts: TableSpec, candidates: list[_Candidate]) -> None:
    rows: list[ValidationRow] = []
    rejects: list[Reject] = []
    for c in candidates:
        ok = c.rule is None
        rows.append(
            ValidationRow(
                ts.name,
                c.row.source_key,
                "VALIDATED" if ok else "FAILED",
                c.rule,
                c.source_hash,
                c.target_hash,
                purge_safe=ok and ts.config.role == "data",
                error_text=c.error,
            )
        )
        if not ok:
            rejects.append(
                Reject(
                    ts.name,
                    c.row.source_key,
                    "VALIDATE",
                    c.rule or "",
                    c.field_name,
                    c.row.raw_bytes,
                    None,
                    None,
                    None,
                    c.error or "",
                )
            )
    ctx.target.write_validation(ctx.run_id, ctx.namespace, rows)
    keys = [c.row.source_key for c in candidates]
    ctx.target.delete_rejects_keys(ctx.run_id, ctx.namespace, ts.name, "VALIDATE", keys)
    if rejects:
        ctx.target.insert_rejects(ctx.run_id, ctx.namespace, rejects)


def _iter_candidates(ctx: RunContext, ts: TableSpec) -> Iterator[list[_Candidate]]:
    """One key-contiguous staging batch at a time: converted source values + both business hashes."""
    hash_cols = ts.config.hash_columns
    tsql = hash_expression(ctx.manifest.target.provider, hash_cols, ts.columns)
    for batch in ctx.target.iter_staging(ctx.run_id, ctx.namespace, ts.name, ctx.manifest.batch.validate_batch_rows):
        if not batch:
            continue
        target_hashes = ctx.target.target_hashes(
            ctx.run_id, ctx.namespace, ts.name, tsql, batch[0].source_key, batch[-1].source_key
        )
        out: list[_Candidate] = []
        for row in batch:
            conv = convert_record(row.raw_bytes, ts.columns)
            if not conv.ok:  # cannot happen for a staged row; treat as a hash failure rather than crash
                err = conv.error
                assert err is not None
                out.append(_Candidate(row, {}, b"", None, rule=RULE_HASH, field_name=err.column, error=err.error))
                continue
            out.append(
                _Candidate(
                    row,
                    dict(conv.source_text),
                    source_hash(conv.source_text, hash_cols, ts.columns),
                    target_hashes.get(row.source_key),
                )
            )
        yield out


@dataclass
class _RowChecks:
    """Per-row rules (hash, archive conflict, parent) - everything that does not need the whole table."""

    ts: TableSpec
    archived: dict[str, bytes]
    parents_present: set[str] | None

    def apply(self, c: _Candidate) -> None:
        cfg = self.ts.config
        if c.rule:
            return
        if c.target_hash != c.source_hash:
            c.rule = RULE_HASH
            c.field_name = _differing_column(c, self.ts)
            c.error = (
                f"business hash source={c.source_hash.hex()[:16]} target="
                f"{(c.target_hash or b'').hex()[:16] or 'missing'} differs at {c.field_name}"
            )
            return
        # an already-archived row with the same key must carry the same business hash (idempotent re-run),
        # otherwise purging the source would leave a conflicting archive copy
        prior = self.archived.get(c.row.source_key)
        if prior is not None and prior != c.source_hash:
            c.rule = RULE_ARCHIVE_CONFLICT
            c.field_name = cfg.key_columns[0]
            c.error = (
                f"arch.{self.ts.name} already holds this key with business hash {prior.hex()[:16]} != "
                f"source {c.source_hash.hex()[:16]}; not promoted, not purge-safe"
            )
            return
        parent = cfg.selection.parent
        if parent and self.parents_present is not None and c.row.source_key not in self.parents_present:
            c.rule = RULE_ORPHAN
            c.field_name = parent.columns[0]
            pk = "|".join(str(c.row.values[col]).rstrip(" ") for col in parent.columns)
            c.error = f"parent {parent.table}({pk}) is neither validated in this run nor present in arch.{parent.table}"


def _row_checks(ctx: RunContext, ts: TableSpec) -> _RowChecks:
    cfg = ts.config
    archived = ctx.target.archived_hashes(ctx.run_id, ctx.namespace, ts.name, cfg.key_columns)
    present: set[str] | None = None
    if cfg.selection.parent:
        parent = cfg.selection.parent
        present = ctx.target.parents_present(
            ctx.run_id, ctx.namespace, ts.name, parent.columns, parent.table, parent.references
        )
    return _RowChecks(ts, archived, present)


@dataclass
class _ClassMismatch:
    rule: str
    detail: str


def _classify(c: _Candidate, ts: TableSpec, successors: dict[str, str]) -> None:
    ct = ts.config.class_totals
    assert ct is not None
    class_spec = ts.by_name[ct.class_column]
    raw = c.source_values.get(ct.class_column)
    c.source_class = _source_class(str(raw or ""), class_spec.value_map, successors)
    c.target_class = str(c.row.values[ct.class_column] or "").rstrip(" ")


def _class_mismatches(ctx: RunContext, ts: TableSpec, src: dict[str, _ClassSide]) -> dict[str, _ClassMismatch]:
    """Compare per-class source aggregates with the target's, persist mig.class_totals; returns class -> mismatch."""
    ct = ts.config.class_totals
    assert ct is not None
    tgt: dict[str, ClassAggregate] = ctx.target.class_aggregates(
        ctx.run_id, ctx.namespace, ts.name, ct.class_column, ct.sum_columns
    )
    sum_col = ct.sum_columns[0] if ct.sum_columns else None
    class_rows: list[ClassTotalRow] = []
    mismatches: dict[str, _ClassMismatch] = {}
    for cls in sorted(set(src) | set(tgt)):
        s, t = src.get(cls, _ClassSide()), tgt.get(cls, ClassAggregate())
        s_sum = s.sums.get(sum_col, Decimal(0)) if sum_col else None
        t_sum = t.sums.get(sum_col, Decimal(0)) if sum_col else None
        count_ok = s.count == t.count
        sums_ok = all(s.sums.get(col, Decimal(0)) == t.sums.get(col, Decimal(0)) for col in ct.sum_columns)
        status = "MATCH" if count_ok and sums_ok else "MISMATCH"
        class_rows.append(ClassTotalRow(ts.name, cls, s.count, t.count, s_sum, t_sum, status))
        if status == "MISMATCH":
            mismatches[cls] = _ClassMismatch(
                RULE_CLASS_COUNT if not count_ok else RULE_CLASS_TOTAL,
                f"class {cls}: source count={s.count} target count={t.count}"
                if not count_ok
                else f"class {cls}: source SUM({sum_col})={s_sum} target SUM({sum_col})={t_sum}",
            )
    ctx.target.write_class_totals(ctx.run_id, ctx.namespace, class_rows)
    for r in class_rows:
        ctx.log.info(
            f"class {r.retention_class}: count {r.source_count}/{r.target_count} sum {r.source_sum}/{r.target_sum} "
            f"{r.status}",
            ts.name,
        )
    return mismatches


def _fail_class(c: _Candidate, ts: TableSpec, mm: _ClassMismatch) -> None:
    ct = ts.config.class_totals
    assert ct is not None
    c.rule = mm.rule
    c.field_name = ct.class_column
    c.error = (
        f"{mm.detail}; row class {c.source_class} (system of record) vs {c.target_class} (target)"
        if c.source_class != c.target_class
        else mm.detail
    )


def validate_table(ctx: RunContext, ts: TableSpec) -> dict[str, int]:
    """Two streaming passes over staging so memory is bounded by validate_batch_rows, not by the table.

    Pass 1: per-row rules (hash, archive conflict, parent) are decided and persisted batch by batch while
    per-class source aggregates accumulate. Pass 2 (only when a class disagrees): re-stream the rows of the
    mismatching classes and flip the ones that passed pass 1 to FAILED.
    """
    cfg = ts.config
    ct = cfg.class_totals
    checks = _row_checks(ctx, ts)
    successors = _successor_map(ctx, ts) if ct else {}
    src: dict[str, _ClassSide] = defaultdict(_ClassSide)
    moved_classes: set[str] = set()

    for batch in _iter_candidates(ctx, ts):
        for c in batch:
            checks.apply(c)
            if ct:
                _classify(c, ts, successors)
                assert c.source_class is not None and c.target_class is not None
                if c.source_class != c.target_class:
                    moved_classes.update((c.source_class, c.target_class))
                side = src[c.source_class]
                side.count += 1
                for col in ct.sum_columns:
                    v = c.source_values[col]
                    assert isinstance(v, Decimal)
                    side.sums[col] = side.sums.get(col, Decimal(0)) + v
        _persist(ctx, ts, batch)

    if ct:
        mismatches = _class_mismatches(ctx, ts, src)
        if mismatches:
            # Rows whose class differs between the two sides explain a mismatch; if no row of the class moved,
            # every target row of that class is affected. Rows that already failed a per-row rule keep that rule.
            for batch in _iter_candidates(ctx, ts):
                flipped: list[_Candidate] = []
                for c in batch:
                    checks.apply(c)
                    if c.rule:
                        continue
                    _classify(c, ts, successors)
                    assert c.source_class is not None and c.target_class is not None
                    if c.source_class != c.target_class:
                        hit = mismatches.get(c.source_class) or mismatches.get(c.target_class)
                    elif c.target_class not in moved_classes:
                        hit = mismatches.get(c.target_class)
                    else:
                        hit = None
                    if hit is not None:
                        _fail_class(c, ts, hit)
                        flipped.append(c)
                if flipped:
                    _persist(ctx, ts, flipped)

    promoted = ctx.target.promote(ctx.run_id, ctx.namespace, ts.name, cfg.key_columns)
    validated = ctx.target.count_validation(ctx.run_id, ctx.namespace, ts.name, "VALIDATED")
    failed = ctx.target.count_validation(ctx.run_id, ctx.namespace, ts.name, "FAILED")
    ctx.target.update_ledger(ctx.run_id, ctx.namespace, ts.name, validated=validated, validate_failed=failed)
    ledger = ctx.target.get_ledger(ctx.run_id, ctx.namespace)[ts.name]
    loaded = ledger.loaded if ledger.loaded is not None else -1
    if loaded != validated + failed:
        raise ReconcileError(f"{ts.name}: loaded {loaded} != validated {validated} + validate_failed {failed}")
    ctx.log.info(f"validated={validated} failed={failed} promoted={promoted}", ts.name)
    return {"validated": validated, "validate_failed": failed}


def run(ctx: RunContext) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ts in ctx.tables_in_order():
        log_id = ctx.target.stage_log_start(ctx.run_id, ctx.namespace, "VALIDATE", ts.name)
        try:
            out[ts.name] = validate_table(ctx, ts)
        except Exception as e:
            ctx.target.stage_log_finish(log_id, "FAILED", None, str(e)[:4000])
            raise
        ctx.target.stage_log_finish(log_id, "OK", out[ts.name]["validated"], None)
    return out

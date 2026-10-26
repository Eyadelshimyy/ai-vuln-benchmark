"""In-memory source/target drivers for unit tests and dry local runs (LDM_DRIVER=fake)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal

from ..convert import Value
from ..errors import PurgeGuardError, SourceError
from ..hashing import target_hash
from ..typemap import ColumnSpec
from .base import (
    ClassAggregate,
    ClassTotalRow,
    FailureRow,
    InsertFailure,
    KeyRange,
    LedgerRow,
    PurgeAuditRow,
    Reject,
    SelectionSpec,
    StagedRow,
    TargetValue,
    ValidationRow,
)


def _text(v: Value) -> str:
    if isinstance(v, bytes):
        return v.decode("latin-1")
    return "" if v is None else str(v)


class FakeSource:
    """Rows are dicts column -> source value (str keeps CHAR padding, bytes for FOR BIT DATA)."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Value]]] = defaultdict(list)
        self.purge_audit: list[tuple[str, str, str, int, str]] = []  # (run_id, table, key, batch_no, status)
        self.fail_delete_for: set[str] = set()  # keys whose DELETE raises a SQLCODE
        self.short_delete_for: set[str] = set()  # keys that "vanish" so the deleted count is short
        self.connected = False

    def add_rows(self, schema: str, table: str, rows: Sequence[dict[str, Value]]) -> None:
        self.tables[f"{schema}.{table}"].extend(rows)

    def rows(self, schema: str, table: str) -> list[dict[str, Value]]:
        return self.tables[f"{schema}.{table}"]

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def _selected(self, qualified: str, selection: SelectionSpec) -> list[dict[str, Value]]:
        rows = self.tables[qualified]
        if selection.all_rows:
            out = list(rows)
        else:
            assert selection.class_column and selection.last_access_column and selection.last_access_before
            out = [
                r
                for r in rows
                if _text(r[selection.class_column]).rstrip(" ") in selection.classes
                and _text(r[selection.last_access_column]) < selection.last_access_before
            ]
        return out

    def _key(self, row: dict[str, Value], selection: SelectionSpec) -> str:
        return "|".join(_text(row[c]) for c in selection.key_columns)

    def select_keys(self, schema: str, table: str, selection: SelectionSpec) -> list[str]:
        return sorted(self._key(r, selection) for r in self._selected(f"{schema}.{table}", selection))

    def fetch_range(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        selection: SelectionSpec,
        key_from: str,
        key_to: str,
    ) -> Iterator[dict[str, Value]]:
        rows = self._selected(f"{schema}.{table}", selection)
        for r in sorted(rows, key=lambda r: self._key(r, selection)):
            k = self._key(r, selection)
            if key_from <= k <= key_to:
                yield {c: r[c] for c in columns}

    def purge_batch(
        self,
        schema: str,
        table: str,
        key_column: str,
        keys: Sequence[str],
        run_id: str,
        namespace: str,
        batch_no: int,
    ) -> int:
        qualified = f"{schema}.{table}"
        audit = [(run_id, table, k, batch_no, "INTENDED") for k in keys]
        for k in keys:
            if k in self.fail_delete_for:
                raise SourceError(-911, "40001", f"deadlock or timeout deleting {table} key {k.strip()}")
        wanted = {k.rstrip(" ") for k in keys if k not in self.short_delete_for}
        remaining = [r for r in self.tables[qualified] if _text(r[key_column]).rstrip(" ") not in wanted]
        deleted = len(self.tables[qualified]) - len(remaining)
        if deleted != len(keys):
            raise PurgeGuardError(
                f"{table} batch {batch_no}: DELETE removed {deleted} rows but {len(keys)} were intended; rolled back"
            )
        self.tables[qualified] = remaining
        self.purge_audit.extend((r, t, k, b, "PURGED") for r, t, k, b, _ in audit)
        return deleted

    def audited_keys(self, run_id, table, keys) -> set[str]:
        wanted = set(keys)
        return {k for r, t, k, _, _ in self.purge_audit if r == run_id and t == table and k in wanted}


@dataclass
class _Run:
    status: str = "RUNNING"
    exit_code: int | None = None
    purge_enabled: bool = False
    manifest_sha: str = ""
    sessions: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class TableShape:
    key_columns: tuple[str, ...]
    hash_columns: tuple[str, ...]
    specs: list[ColumnSpec]


class FakeTarget:
    def __init__(self) -> None:
        self.ddl_applied: dict[str, str] = {}
        self.readers: dict[str, str] = {}
        self.namespace_scripts: list[tuple[str, str]] = []
        self.runs: dict[tuple[str, str], _Run] = {}
        self.ledger: dict[tuple[str, str], dict[str, LedgerRow]] = {}
        self.stage_log: list[dict[str, object]] = []
        self.key_ranges: dict[tuple[str, str, str], list[KeyRange]] = defaultdict(list)
        self.rejects: dict[tuple[str, str], list[Reject]] = defaultdict(list)
        self.staging: dict[str, list[StagedRow]] = defaultdict(list)  # table -> rows (all runs, all ns)
        self.staging_ns: dict[int, tuple[str, str]] = {}  # stg_id -> (run_id, namespace)
        self.validation: dict[tuple[str, str], list[ValidationRow]] = defaultdict(list)
        self.class_totals: dict[tuple[str, str], list[ClassTotalRow]] = defaultdict(list)
        self.arch: dict[str, dict[tuple[str, ...], dict[str, TargetValue]]] = defaultdict(dict)
        self.arch_hash: dict[str, dict[tuple[str, ...], bytes]] = defaultdict(dict)
        self.purge_audit: dict[tuple[str, str], list[PurgeAuditRow]] = defaultdict(list)
        self.shapes: dict[str, TableShape] = {}
        self.insert_failures: dict[str, InsertFailure] = {}  # planted target-side insert errors by source_key
        self._next_id = 1
        self._next_log = 1
        self.connected = False

    # --- test helpers ---
    def register_table(
        self, table: str, key_columns: Sequence[str], hash_columns: Sequence[str], specs: list[ColumnSpec]
    ):
        self.shapes[table] = TableShape(tuple(key_columns), tuple(hash_columns), specs)

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    # --- init ---
    def applied_ddl(self) -> dict[str, str]:
        return dict(self.ddl_applied)

    def apply_ddl(self, sql_text: str, file_name: str, sha256_hex: str) -> None:
        self.ddl_applied[file_name] = sha256_hex

    def ensure_reader(self, user: str, password: str) -> None:
        self.readers[user] = password

    def apply_sql_in_namespace(self, sql_text: str, namespace: str) -> None:
        self.namespace_scripts.append((namespace, sql_text))

    # --- run / ledger ---
    def ensure_run(self, run_id, namespace, purge_enabled, job_image, manifest_sha) -> str:
        run = self.runs.setdefault((run_id, namespace), _Run(purge_enabled=purge_enabled, manifest_sha=manifest_sha))
        return run.manifest_sha

    def refresh_run(self, run_id, namespace, purge_enabled, job_image) -> None:
        self.runs[(run_id, namespace)].purge_enabled = purge_enabled

    def get_run_status(self, run_id, namespace):
        r = self.runs.get((run_id, namespace))
        return r.status if r else None

    def set_run_status(self, run_id, namespace, status, exit_code) -> None:
        r = self.runs[(run_id, namespace)]
        r.status, r.exit_code = status, exit_code

    def ensure_ledger(self, run_id, namespace, tables) -> None:
        led = self.ledger.setdefault((run_id, namespace), {})
        for name, order, role in tables:
            led.setdefault(name, LedgerRow(table_name=name, table_order=order, table_role=role))

    def get_ledger(self, run_id, namespace):
        return {k: replace(v) for k, v in self.ledger.get((run_id, namespace), {}).items()}

    def update_ledger(self, run_id, namespace, table, **cols) -> None:
        led = self.ledger[(run_id, namespace)]
        unknown = set(cols) - set(LedgerRow.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown ledger columns {sorted(unknown)}")
        led[table] = replace(led[table], **cols)

    def stage_log_start(self, run_id, namespace, stage, table) -> int:
        log_id = self._next_log
        self._next_log += 1
        self.stage_log.append({"id": log_id, "run_id": run_id, "stage": stage, "table": table, "status": "RUNNING"})
        return log_id

    def stage_log_finish(self, log_id, status, rows, message) -> None:
        for e in self.stage_log:
            if e["id"] == log_id:
                e.update(status=status, rows=rows, message=message)

    # --- key ranges ---
    def get_key_ranges(self, run_id, namespace, table):
        return [replace(r) for r in self.key_ranges[(run_id, namespace, table)]]

    def insert_key_ranges(self, run_id, namespace, ranges) -> None:
        for r in ranges:
            self.key_ranges[(run_id, namespace, r.table_name)].append(replace(r))

    def update_key_range(self, run_id, namespace, table, range_seq, **cols) -> None:
        unknown = set(cols) - set(KeyRange.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown key_ranges columns {sorted(unknown)}")
        ranges = self.key_ranges[(run_id, namespace, table)]
        for i, r in enumerate(ranges):
            if r.range_seq == range_seq:
                ranges[i] = replace(r, **cols)

    # --- rejects ---
    def insert_rejects(self, run_id, namespace, rejects) -> None:
        existing = {(r.table_name, r.stage, r.source_key) for r in self.rejects[(run_id, namespace)]}
        for r in rejects:
            if (r.table_name, r.stage, r.source_key) not in existing:
                self.rejects[(run_id, namespace)].append(r)
                existing.add((r.table_name, r.stage, r.source_key))

    def count_rejects(self, run_id, namespace, table, stage=None) -> int:
        return sum(
            1
            for r in self.rejects[(run_id, namespace)]
            if r.table_name == table and (stage is None or r.stage == stage)
        )

    # --- staging ---
    def _stg(self, run_id, namespace, table) -> list[StagedRow]:
        return [r for r in self.staging[table] if self.staging_ns[r.stg_id] == (run_id, namespace)]

    def delete_staging_range(self, run_id, namespace, table, range_seq) -> None:
        self.staging[table] = [
            r
            for r in self.staging[table]
            if not (self.staging_ns[r.stg_id] == (run_id, namespace) and r.key_range_seq == range_seq)
        ]

    def delete_rejects_keys(self, run_id, namespace, table, stage, keys) -> None:
        wanted = set(keys)
        self.rejects[(run_id, namespace)] = [
            r
            for r in self.rejects[(run_id, namespace)]
            if not (r.table_name == table and r.stage == stage and r.source_key in wanted)
        ]

    def delete_rejects_range(self, run_id, namespace, table, stage, range_seq) -> None:
        self.rejects[(run_id, namespace)] = [
            r
            for r in self.rejects[(run_id, namespace)]
            if not (r.table_name == table and r.stage == stage and r.key_range_seq == range_seq)
        ]

    def insert_staging(self, run_id, namespace, table, rows) -> list[InsertFailure]:
        failures: list[InsertFailure] = []
        existing = {r.source_key for r in self.staging[table] if self.staging_ns[r.stg_id][1] == namespace}
        for row in rows:
            if row.source_key in self.insert_failures:
                failures.append(self.insert_failures[row.source_key])
                continue
            if row.source_key in existing:
                failures.append(
                    InsertFailure(
                        row.source_key,
                        "23000",
                        2601,
                        f"Cannot insert duplicate key row in object 'stg.{table}' with unique index "
                        f"'UX_stg_{table}_ns_key'. The duplicate key value is ({namespace}, {row.source_key}).",
                    )
                )
                continue
            stored = replace(row, stg_id=self._next_id, values=dict(row.values))
            self.staging_ns[self._next_id] = (run_id, namespace)
            self._next_id += 1
            self.staging[table].append(stored)
            existing.add(row.source_key)
        return failures

    def count_staging(self, run_id, namespace, table) -> int:
        return len(self._stg(run_id, namespace, table))

    def iter_staging(self, run_id, namespace, table, batch) -> Iterator[list[StagedRow]]:
        rows = sorted(self._stg(run_id, namespace, table), key=lambda r: r.source_key)
        for i in range(0, len(rows), batch):
            yield [replace(r, values=dict(r.values)) for r in rows[i : i + batch]]

    def target_hashes(self, run_id, namespace, table, tsql_expr, key_from, key_to) -> dict[str, bytes]:
        shape = self.shapes[table]
        return {
            r.source_key: target_hash(r.values, shape.hash_columns, shape.specs)
            for r in self._stg(run_id, namespace, table)
            if key_from <= r.source_key <= key_to
        }

    def class_aggregates(self, run_id, namespace, table, class_column, sum_columns) -> dict[str, ClassAggregate]:
        out: dict[str, ClassAggregate] = defaultdict(ClassAggregate)
        for r in self._stg(run_id, namespace, table):
            cls = str(r.values[class_column]).rstrip(" ")
            agg = out[cls]
            agg.count += 1
            for c in sum_columns:
                v = r.values[c]
                assert isinstance(v, Decimal)
                agg.sums[c] = agg.sums.get(c, Decimal(0)) + v
        return dict(out)

    def _validated_keys(self, run_id, namespace, table) -> set[str]:
        return {
            v.source_key
            for v in self.validation[(run_id, namespace)]
            if v.table_name == table and v.status == "VALIDATED"
        }

    def parents_present(self, run_id, namespace, child_table, child_columns, parent_table, parent_columns) -> set[str]:
        validated = self._validated_keys(run_id, namespace, parent_table)
        present: set[tuple[str, ...]] = set()
        for p in self._stg(run_id, namespace, parent_table):
            if p.source_key in validated:
                present.add(tuple(str(p.values[c]).rstrip(" ") for c in parent_columns))
        for row in self.arch[parent_table].values():
            present.add(tuple(str(row[c]).rstrip(" ") for c in parent_columns))
        return {
            c.source_key
            for c in self._stg(run_id, namespace, child_table)
            if tuple(str(c.values[col]).rstrip(" ") for col in child_columns) in present
        }

    # --- validation ---
    def write_validation(self, run_id, namespace, rows) -> None:
        keep = {(r.table_name, r.source_key) for r in rows}
        self.validation[(run_id, namespace)] = [
            v for v in self.validation[(run_id, namespace)] if (v.table_name, v.source_key) not in keep
        ] + list(rows)

    def write_class_totals(self, run_id, namespace, rows) -> None:
        tables = {r.table_name for r in rows}
        self.class_totals[(run_id, namespace)] = [
            c for c in self.class_totals[(run_id, namespace)] if c.table_name not in tables
        ] + list(rows)

    def get_class_totals(self, run_id, namespace):
        return list(self.class_totals[(run_id, namespace)])

    def archived_hashes(self, run_id, namespace, table, key_columns) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for r in self._stg(run_id, namespace, table):
            k = tuple(str(r.values[c]) for c in key_columns)
            if k in self.arch_hash[table]:
                out[r.source_key] = self.arch_hash[table][k]
        return out

    def promote(self, run_id, namespace, table, key_columns) -> int:
        validated = self._validated_keys(run_id, namespace, table)
        hashes = {
            v.source_key: v.source_hash
            for v in self.validation[(run_id, namespace)]
            if v.table_name == table and v.source_hash is not None
        }
        n = 0
        for r in self._stg(run_id, namespace, table):
            if r.source_key not in validated:
                continue
            k = tuple(str(r.values[c]) for c in key_columns)
            if k not in self.arch[table]:
                self.arch[table][k] = dict(r.values)
                self.arch_hash[table][k] = hashes.get(r.source_key, b"")
                n += 1
        return n

    def purge_safe_keys(self, run_id, namespace, table) -> list[str]:
        return sorted(
            v.source_key for v in self.validation[(run_id, namespace)] if v.table_name == table and v.purge_safe
        )

    def count_validation(self, run_id, namespace, table, status) -> int:
        return sum(1 for v in self.validation[(run_id, namespace)] if v.table_name == table and v.status == status)

    # --- purge audit ---
    def insert_purge_audit(self, run_id, namespace, table, keys, batch_no) -> None:
        existing = {a.source_key for a in self.purge_audit[(run_id, namespace)] if a.table_name == table}
        for k in keys:
            if k not in existing:
                self.purge_audit[(run_id, namespace)].append(PurgeAuditRow(table, k, batch_no, "INTENDED"))

    def set_purge_audit_status(self, run_id, namespace, table, keys, status) -> None:
        ks = set(keys)
        rows = self.purge_audit[(run_id, namespace)]
        for i, a in enumerate(rows):
            if a.table_name == table and a.source_key in ks:
                rows[i] = replace(a, status=status)

    def purged_keys(self, run_id, namespace, table) -> set[str]:
        return {
            a.source_key
            for a in self.purge_audit[(run_id, namespace)]
            if a.table_name == table and a.status == "PURGED"
        }

    # --- reconcile ---
    def failures(self, run_id, namespace) -> list[FailureRow]:
        out = [
            FailureRow(
                r.table_name,
                r.source_key,
                r.stage,
                r.rule,
                r.field_name,
                r.sqlstate,
                r.native_error,
                r.error_text,
            )
            for r in self.rejects[(run_id, namespace)]
        ]
        seen = {(f.table_name, f.source_key) for f in out}
        for v in self.validation[(run_id, namespace)]:
            if v.status == "FAILED" and (v.table_name, v.source_key) not in seen:
                out.append(
                    FailureRow(
                        v.table_name,
                        v.source_key,
                        "VALIDATE",
                        v.rule or "",
                        None,
                        None,
                        None,
                        v.error_text or "",
                    )
                )
        return sorted(out, key=lambda f: (f.table_name, f.source_key))

    def insert_run_sessions(self, run_id, namespace, sessions) -> None:
        self.runs[(run_id, namespace)].sessions = list(sessions)

    def delete_staging_run(self, run_id, namespace) -> None:
        for table in list(self.staging):
            self.staging[table] = [r for r in self.staging[table] if self.staging_ns[r.stg_id] != (run_id, namespace)]

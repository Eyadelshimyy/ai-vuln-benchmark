"""Driver protocols. Stages talk only to these; Db2/Azure SQL and the in-memory fakes implement them."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

from ..convert import Timestamp12, Value
from ..typemap import ColumnSpec

TargetValue = Value | Timestamp12 | datetime | date


@dataclass(frozen=True)
class SelectionSpec:
    """Manifest-driven selection predicate (never hard-coded in a stage)."""

    key_columns: tuple[str, ...]
    all_rows: bool
    class_column: str | None = None
    classes: tuple[str, ...] = ()
    last_access_column: str | None = None
    last_access_before: str | None = None
    parent_table: str | None = None  # qualified parent table (schema.name); checked in VALIDATE, not EXTRACT
    parent_child_columns: tuple[str, ...] = ()
    parent_columns: tuple[str, ...] = ()


@dataclass
class LedgerRow:
    table_name: str
    table_order: int
    table_role: str
    extracted: int | None = None
    extract_files: int | None = None
    loaded: int | None = None
    rejected: int | None = None
    validated: int | None = None
    validate_failed: int | None = None
    purge_intended: int | None = None
    purged: int | None = None
    purge_dry_run: bool | None = None


@dataclass
class KeyRange:
    table_name: str
    range_seq: int
    key_from: str
    key_to: str
    status: str = "PLANNED"  # PLANNED | RUNNING | DONE | FAILED
    row_count: int | None = None
    byte_count: int | None = None
    sha256_hex: str | None = None
    blob_path: str | None = None
    local_path: str | None = None
    load_status: str = "PENDING"  # PENDING | RUNNING | DONE
    attempt: int = 0
    error_text: str | None = None


@dataclass(frozen=True)
class Reject:
    table_name: str
    source_key: str
    stage: str
    rule: str
    field_name: str | None
    raw_bytes: bytes | None
    field_bytes: bytes | None
    sqlstate: str | None
    native_error: int | None
    error_text: str
    key_range_seq: int | None = None


@dataclass
class StagedRow:
    table_name: str
    source_key: str
    key_range_seq: int | None
    raw_bytes: bytes
    values: dict[str, TargetValue]
    stg_id: int | None = None


@dataclass(frozen=True)
class InsertFailure:
    source_key: str
    sqlstate: str | None
    native_error: int | None
    error_text: str


@dataclass(frozen=True)
class ValidationRow:
    table_name: str
    source_key: str
    status: str  # VALIDATED | FAILED
    rule: str | None
    source_hash: bytes | None
    target_hash: bytes | None
    purge_safe: bool
    error_text: str | None = None


@dataclass(frozen=True)
class ClassTotalRow:
    table_name: str
    retention_class: str
    source_count: int
    target_count: int
    source_sum: Decimal | None
    target_sum: Decimal | None
    status: str  # MATCH | MISMATCH


@dataclass
class ClassAggregate:
    count: int = 0
    sums: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureRow:
    table_name: str
    source_key: str
    stage: str
    rule: str
    field_name: str | None
    sqlstate: str | None
    native_error: int | None
    error_text: str


@dataclass(frozen=True)
class PurgeAuditRow:
    table_name: str
    source_key: str
    batch_no: int
    status: str  # INTENDED | PURGED | ROLLED_BACK


class SourceDriver(Protocol):
    """A legacy source (Db2). Errors raise ldm.errors.SourceError with SQLCODE/SQLSTATE."""

    def connect(self) -> None: ...
    def close(self) -> None: ...

    def select_keys(self, schema: str, table: str, selection: SelectionSpec) -> list[str]:
        """Selected key values (single key column: the raw CHAR value incl. padding), ordered ascending."""
        ...

    def fetch_range(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        selection: SelectionSpec,
        key_from: str,
        key_to: str,
    ) -> Iterator[dict[str, Value]]:
        """Rows (column -> raw python value) of the selection within [key_from, key_to] ordered by key.

        Used by the built-in fixed-width writer when the manifest's unload_command is not used.
        """
        ...

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
        """One unit of work: INSERT MIGAUDIT.PURGE_AUDIT rows, DELETE the keys, verify the count, COMMIT.

        Rolls back and raises PurgeGuardError if the deleted count differs from len(keys);
        raises SourceError on any Db2 error (after rollback).
        """
        ...

    def audited_keys(self, run_id: str, table: str, keys: Sequence[str]) -> set[str]:
        """Subset of `keys` already committed to MIGAUDIT.PURGE_AUDIT for this run (i.e. already deleted)."""
        ...


class TargetDriver(Protocol):
    """The migration target (Azure SQL) which also hosts mig.*, stg.* and arch.*."""

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def register_table(
        self, table: str, key_columns: Sequence[str], hash_columns: Sequence[str], specs: list[ColumnSpec]
    ) -> None:
        """Tell the driver the target column shape of stg.<table> (needed to bind/read typed columns)."""
        ...

    # --- init ---
    def applied_ddl(self) -> dict[str, str]:
        """mig.schema_version: file_name -> sha256 (empty when the schema does not exist yet)."""
        ...

    def apply_ddl(self, sql_text: str, file_name: str, sha256_hex: str) -> None:
        """Execute a GO-separated T-SQL file and record it in mig.schema_version."""
        ...

    def ensure_reader(self, user: str, password: str) -> None:
        """Create/refresh the contained reader user in role ldm_report_reader."""
        ...

    def apply_sql_in_namespace(self, sql_text: str, namespace: str) -> None:
        """EXEC sp_set_session_context N'ldm.namespace', <namespace>; then the GO-separated script."""
        ...

    # --- run / ledger / stage log ---
    def ensure_run(
        self, run_id: str, namespace: str, purge_enabled: bool, job_image: str | None, manifest_sha: str
    ) -> str:
        """Create the mig.runs row if absent; returns the sha the run was created with. Never alters an existing row."""
        ...

    def refresh_run(self, run_id: str, namespace: str, purge_enabled: bool, job_image: str | None) -> None:
        """Update purge_enabled/job_image of an existing run (only after the manifest digest was accepted)."""
        ...

    def get_run_status(self, run_id: str, namespace: str) -> str | None: ...
    def set_run_status(self, run_id: str, namespace: str, status: str, exit_code: int | None) -> None: ...
    def ensure_ledger(self, run_id: str, namespace: str, tables: Sequence[tuple[str, int, str]]) -> None: ...
    def get_ledger(self, run_id: str, namespace: str) -> dict[str, LedgerRow]: ...
    def update_ledger(self, run_id: str, namespace: str, table: str, **cols: int | bool | None) -> None: ...
    def stage_log_start(self, run_id: str, namespace: str, stage: str, table: str | None) -> int: ...
    def stage_log_finish(self, log_id: int, status: str, rows: int | None, message: str | None) -> None: ...

    # --- key ranges (extract / load restart) ---
    def get_key_ranges(self, run_id: str, namespace: str, table: str) -> list[KeyRange]: ...
    def insert_key_ranges(self, run_id: str, namespace: str, ranges: Sequence[KeyRange]) -> None: ...
    def update_key_range(self, run_id: str, namespace: str, table: str, range_seq: int, **cols: object) -> None: ...

    # --- rejects ---
    def insert_rejects(self, run_id: str, namespace: str, rejects: Sequence[Reject]) -> None: ...
    def count_rejects(self, run_id: str, namespace: str, table: str, stage: str | None = None) -> int: ...
    def delete_rejects_keys(self, run_id: str, namespace: str, table: str, stage: str, keys: Sequence[str]) -> None: ...

    # --- staging ---
    def delete_staging_range(self, run_id: str, namespace: str, table: str, range_seq: int) -> None: ...
    def delete_rejects_range(self, run_id: str, namespace: str, table: str, stage: str, range_seq: int) -> None: ...
    def insert_staging(self, run_id: str, namespace: str, table: str, rows: Sequence[StagedRow]) -> list[InsertFailure]:
        """Insert a batch; on a batch error retry row by row and return the rows that failed."""
        ...

    def count_staging(self, run_id: str, namespace: str, table: str) -> int: ...
    def iter_staging(self, run_id: str, namespace: str, table: str, batch: int) -> Iterator[list[StagedRow]]:
        """Staged rows of the run in batches of `batch`, ordered by source_key (batches are key-contiguous)."""
        ...

    def target_hashes(
        self, run_id: str, namespace: str, table: str, tsql_expr: str, key_from: str, key_to: str
    ) -> dict[str, bytes]:
        """source_key -> HASHBYTES(...) computed on the target for staged rows with key_from <= source_key <= key_to."""
        ...

    def class_aggregates(
        self, run_id: str, namespace: str, table: str, class_column: str, sum_columns: Sequence[str]
    ) -> dict[str, ClassAggregate]: ...

    def parents_present(
        self,
        run_id: str,
        namespace: str,
        child_table: str,
        child_columns: Sequence[str],
        parent_table: str,
        parent_columns: Sequence[str],
    ) -> set[str]:
        """Child source_keys whose parent exists in mig.validation(VALIDATED, this run) or in arch.<parent>."""
        ...

    # --- validation ---
    def write_validation(self, run_id: str, namespace: str, rows: Sequence[ValidationRow]) -> None: ...
    def write_class_totals(self, run_id: str, namespace: str, rows: Sequence[ClassTotalRow]) -> None: ...
    def get_class_totals(self, run_id: str, namespace: str) -> list[ClassTotalRow]: ...
    def archived_hashes(self, run_id: str, namespace: str, table: str, key_columns: Sequence[str]) -> dict[str, bytes]:
        """source_key -> arch.<T>.row_hash for staged rows of this run already archived in this namespace."""
        ...

    def promote(self, run_id: str, namespace: str, table: str, key_columns: Sequence[str]) -> int:
        """INSERT arch.<T> from stg.<T> for VALIDATED rows of this run that are not already archived."""
        ...

    def purge_safe_keys(self, run_id: str, namespace: str, table: str) -> list[str]: ...
    def count_validation(self, run_id: str, namespace: str, table: str, status: str) -> int: ...

    # --- purge audit ---
    def insert_purge_audit(
        self, run_id: str, namespace: str, table: str, keys: Sequence[str], batch_no: int
    ) -> None: ...
    def set_purge_audit_status(
        self, run_id: str, namespace: str, table: str, keys: Sequence[str], status: str
    ) -> None: ...
    def purged_keys(self, run_id: str, namespace: str, table: str) -> set[str]: ...

    # --- reconcile ---
    def failures(self, run_id: str, namespace: str) -> list[FailureRow]: ...
    def insert_run_sessions(self, run_id: str, namespace: str, sessions: Sequence[tuple[str, str]]) -> None: ...
    def delete_staging_run(self, run_id: str, namespace: str) -> None: ...

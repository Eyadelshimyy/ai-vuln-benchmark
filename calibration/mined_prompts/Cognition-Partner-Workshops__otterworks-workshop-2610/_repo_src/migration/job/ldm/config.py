"""Manifest + overlay loading and validation (CONTRACTS.md §3.1, §4)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ConfigError

TOKEN_RE = re.compile(r"^[a-z][a-z0-9]{1,11}-(before|after)$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")

OVERLAY_ONLY_KEYS = {"purge", "migrate", "azure", "execution"}
BASE_TOP_LEVEL_KEYS = {
    "schema_version",
    "run_token",
    "source",
    "target",
    "staging",
    "batch",
    "selection_sets",
    "report",
    "tables",
}
OVERLAY_TOP_LEVEL_KEYS = {
    "namespace",
    "extends",
    "migrate",
    "azure",
    "purge",
    "execution",
    "batch",
    "run_token",
    "source",
    "target",
    "staging",
}
TARGET_PROVIDERS = ("postgresql", "azuresql")


def validate_token(token: str) -> tuple[str, str]:
    """Return (RUN, STATE) or raise ConfigError (exit 4)."""
    if not TOKEN_RE.match(token):
        raise ConfigError(f"namespace token {token!r} does not match ^[a-z][a-z0-9]{{1,11}}-(before|after)$")
    run, state = token.rsplit("-", 1)
    return run, state


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise ConfigError(f"run id {run_id!r} does not match ^[a-z0-9][a-z0-9-]{{2,62}}$")
    return run_id


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConnectionEnv(_Strict):
    host: str
    port: str
    database: str
    user: str
    password: str


class RecordFormat(_Strict):
    recfm: Literal["F"]
    text_encoding: str
    bit_data_encoding: str


class SourceConfig(_Strict):
    driver: Literal["db2", "oracle"]
    connection_env: SourceConnectionEnv
    # argv template for the external unloader; empty = ldm's built-in fixed-width writer over the driver
    unload_command: list[str]
    copybook_dir: str
    record_format: RecordFormat


class TargetConnectionEnv(_Strict):
    """Names of the environment variables holding the target connection. Which ones are required depends on
    target.provider: postgresql reads host/port/database/user/password(/sslmode), azuresql reads server/database/
    user/password/auth_mode/managed_identity_client_id."""

    database: str
    user: str
    password: str
    host: str | None = None
    port: str | None = None
    sslmode: str | None = None
    server: str | None = None
    auth_mode: str | None = None
    managed_identity_client_id: str | None = None


class TargetConfig(_Strict):
    provider: Literal["postgresql", "azuresql"]
    connection_env: TargetConnectionEnv
    ddl_dir: str
    typemap: str

    def check(self) -> None:
        ce = self.connection_env
        if self.provider == "postgresql" and not (ce.host and ce.port):
            raise ConfigError("target.provider postgresql requires connection_env.host and connection_env.port")
        if self.provider == "azuresql" and not (ce.server and ce.auth_mode and ce.managed_identity_client_id):
            raise ConfigError(
                "target.provider azuresql requires connection_env.server, auth_mode and managed_identity_client_id"
            )


class StagingConnectionEnv(_Strict):
    """Blob staging is selected by whichever variable is set at run time: s3 bucket (S3), else Azure storage
    account (Azure Blob), else no blob store (files stay on the local staging volume)."""

    local_dir: str
    bucket: str | None = None
    region: str | None = None
    endpoint_url: str | None = None
    storage_account: str | None = None
    container: str | None = None


class StagingConfig(_Strict):
    connection_env: StagingConnectionEnv
    blob_prefix: str


class BatchConfig(_Strict):
    extract_range_rows: int = Field(gt=0)
    load_batch_rows: int = Field(gt=0)
    validate_batch_rows: int = Field(gt=0)
    purge_batch_rows: int = Field(gt=0)
    # A batch that the source rolls back for transient pressure (Db2 SQL0964C log full, lock timeout,
    # deadlock victim) is retried after an exponential backoff instead of failing the stage.
    purge_retry_attempts: int = Field(default=6, ge=0)
    purge_retry_backoff_s: float = Field(default=5.0, gt=0)
    purge_retry_max_backoff_s: float = Field(default=60.0, gt=0)


class ReportConfig(_Strict):
    sessions_glob: str
    issue_register: str | None = None


class ParentRef(_Strict):
    table: str
    columns: list[str]
    references: list[str]


class Selection(_Strict):
    all: bool | None = None
    retention_classes: str | None = None
    class_column: str | None = None
    last_access_column: str | None = None
    last_access_before: str | None = None
    parent: ParentRef | None = None

    @property
    def is_all(self) -> bool:
        return bool(self.all)

    def check(self) -> None:
        if self.is_all:
            return
        if not (self.retention_classes and self.class_column and self.last_access_column and self.last_access_before):
            raise ConfigError(
                "selection must be {all: true} or {retention_classes, class_column, last_access_column, "
                "last_access_before}"
            )


class SourceClassResolution(_Strict):
    lookup_table: str
    lookup_key: str
    successor_column: str


class ClassTotals(_Strict):
    class_column: str
    sum_columns: list[str] = Field(default_factory=list)
    source_class_resolution: SourceClassResolution | None = None


class TypeOverride(_Strict):
    target_type: str | None = None
    trim: Literal["none", "right"] | None = None
    encoding: str | None = None
    format: str | None = None
    value_map: dict[str, str] | None = None


class TableConfig(_Strict):
    name: str
    schema_: str = Field(alias="schema")
    role: Literal["data", "reference"]
    order: int
    copybook: str
    record_length: int = Field(gt=0)
    field_map: dict[str, str]
    key_columns: list[str]
    hash_columns: list[str]
    selection: Selection
    class_totals: ClassTotals | None = None
    type_overrides: dict[str, TypeOverride] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def columns(self) -> list[str]:
        return list(self.field_map.values())

    @property
    def qualified_name(self) -> str:
        return f"{self.schema_}.{self.name}"


class SparkConfig(_Strict):
    master: str = "local[*]"
    driver_memory: str = "1g"
    shuffle_partitions: int = Field(default=8, gt=0)
    records_per_slice: int = Field(default=5000, gt=0)


class ExecutionConfig(_Strict):
    run_job_in_azure: bool = False
    load_engine: Literal["serial", "spark"] = "serial"
    spark: SparkConfig = Field(default_factory=SparkConfig)


class Manifest(_Strict):
    """The merged (base + overlay) manifest."""

    schema_version: int
    run_token: str
    source: SourceConfig
    target: TargetConfig
    staging: StagingConfig
    batch: BatchConfig
    selection_sets: dict[str, list[str]]
    report: ReportConfig
    tables: list[TableConfig]
    namespace: str
    extends: str
    migrate: bool
    azure: bool
    purge: bool = False
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError("schema_version must be 1")
        return v

    def tables_in_order(self) -> list[TableConfig]:
        return sorted(self.tables, key=lambda t: t.order)

    def table(self, name: str) -> TableConfig:
        for t in self.tables:
            if t.name == name:
                return t
        raise ConfigError(f"unknown table {name!r}")

    def data_tables(self) -> list[TableConfig]:
        return [t for t in self.tables_in_order() if t.role == "data"]

    @property
    def run(self) -> str:
        return self.namespace.rsplit("-", 1)[0]

    @property
    def state(self) -> str:
        return self.namespace.rsplit("-", 1)[1]

    def blob_prefix(self, run_id: str) -> str:
        return self.staging.blob_prefix.format(namespace=self.namespace, run_id=run_id)


class LoadedManifest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    manifest: Manifest
    base_path: Path
    overlay_path: Path
    repo_root: Path
    sha256: str

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.repo_root / p


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Maps merge recursively, lists (and scalars) replace."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"manifest file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def _check_column_references(t: TableConfig) -> None:
    cols = set(t.columns)
    referenced: dict[str, list[str]] = {
        "key_columns": t.key_columns,
        "hash_columns": t.hash_columns,
        "type_overrides": list(t.type_overrides),
    }
    sel = t.selection
    sel.check()
    sel_cols = [c for c in (sel.class_column, sel.last_access_column) if c]
    if sel.parent:
        sel_cols.extend(sel.parent.columns)
    referenced["selection"] = sel_cols
    if t.class_totals:
        referenced["class_totals"] = [t.class_totals.class_column, *t.class_totals.sum_columns]
    for where, names in referenced.items():
        for c in names:
            if c not in cols:
                raise ConfigError(f"table {t.name}: {where} names column {c!r} which is not a value of field_map")
    if len(set(t.key_columns)) != len(t.key_columns) or not t.key_columns:
        raise ConfigError(f"table {t.name}: key_columns must be a non-empty list of distinct columns")


def load_manifest(base: Path, namespace: str) -> LoadedManifest:
    """Load base + overlay <dir(base)>/manifests/<NS>.yaml and validate per §4.1. Every error is exit 4."""
    run, _state = validate_token(namespace)
    base = Path(base)
    base_data = _read_yaml(base)

    bad = OVERLAY_ONLY_KEYS & set(base_data)
    if bad:
        raise ConfigError(f"base manifest {base} contains overlay-only key(s): {sorted(bad)}")
    unknown = set(base_data) - BASE_TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"base manifest {base} has unknown top-level key(s): {sorted(unknown)}")

    overlay_path = base.parent / "manifests" / f"{namespace}.yaml"
    if not overlay_path.exists():
        raise ConfigError(f"overlay missing: {overlay_path}")
    overlay = _read_yaml(overlay_path)
    unknown = set(overlay) - OVERLAY_TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"overlay {overlay_path} has unknown top-level key(s): {sorted(unknown)}")
    for req in ("namespace", "extends", "migrate", "azure", "purge"):
        if req not in overlay:
            raise ConfigError(f"overlay {overlay_path} is missing required key {req!r}")
    if overlay["namespace"] != namespace:
        raise ConfigError(f"overlay.namespace {overlay['namespace']!r} != --namespace {namespace!r}")
    extends = (overlay_path.parent / str(overlay["extends"])).resolve()
    if extends != base.resolve():
        raise ConfigError(f"overlay.extends {overlay['extends']!r} does not resolve to the base manifest {base}")

    merged = deep_merge(base_data, overlay)
    try:
        manifest = Manifest.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(f"manifest invalid: {_format_validation_error(e)}") from e

    if manifest.run_token != run:
        raise ConfigError(f"run_token {manifest.run_token!r} != RUN part {run!r} of namespace {namespace!r}")
    if not manifest.migrate:
        raise ConfigError(f"overlay {overlay_path} has migrate: false; this namespace is never migrated")
    if manifest.target.provider == "azuresql" and not manifest.azure:
        raise ConfigError("target.provider azuresql requires azure: true")
    if manifest.execution.run_job_in_azure and not manifest.azure:
        raise ConfigError("execution.run_job_in_azure requires azure: true")
    manifest.target.check()

    orders = [t.order for t in manifest.tables]
    if len(set(orders)) != len(orders):
        raise ConfigError("tables[].order values must be unique")
    names = [t.name for t in manifest.tables]
    if len(set(names)) != len(names):
        raise ConfigError("tables[].name values must be unique")
    for t in manifest.tables:
        _check_column_references(t)
        if not t.selection.is_all and t.selection.retention_classes not in manifest.selection_sets:
            raise ConfigError(
                f"table {t.name}: selection.retention_classes {t.selection.retention_classes!r} "
                f"is not a selection_sets entry"
            )
        if t.selection.parent and t.selection.parent.table not in names:
            raise ConfigError(f"table {t.name}: selection.parent.table {t.selection.parent.table!r} is unknown")
        if t.class_totals and t.class_totals.source_class_resolution:
            if t.class_totals.source_class_resolution.lookup_table not in names:
                raise ConfigError(f"table {t.name}: class_totals lookup_table is unknown")

    digest = hashlib.sha256(base.read_bytes() + overlay_path.read_bytes()).hexdigest()
    return LoadedManifest(
        manifest=manifest,
        base_path=base,
        overlay_path=overlay_path,
        repo_root=base.resolve().parent.parent,
        sha256=digest,
    )

"""Run context shared by every stage: manifest, drivers, table specs, staging, logging."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import LoadedManifest, Manifest, TableConfig
from .copybook import Copybook, parse_copybook
from .drivers.base import SelectionSpec, SourceDriver, TargetDriver
from .errors import ConfigError
from .staging import BlobStore
from .typemap import ColumnSpec, build_column_specs, load_typemap

if TYPE_CHECKING:
    from .drivers.factory import TargetSpec

DEFAULT_LOCAL_STAGING = "/work/staging"


class Log:
    """'<UTC ts> <LEVEL> <stage> <table> <message>' on stderr (CONTRACTS.md §9.3)."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stderr
        self.stage = "-"

    def _emit(self, level: str, table: str | None, msg: str) -> None:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        print(f"{ts} {level} {self.stage} {table or '-'} {msg}", file=self.stream, flush=True)

    def info(self, msg: str, table: str | None = None) -> None:
        self._emit("INFO", table, msg)

    def warn(self, msg: str, table: str | None = None) -> None:
        self._emit("WARN", table, msg)

    def error(self, msg: str, table: str | None = None) -> None:
        self._emit("ERROR", table, msg)


@dataclass
class TableSpec:
    config: TableConfig
    copybook: Copybook
    columns: list[ColumnSpec]

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def by_name(self) -> dict[str, ColumnSpec]:
        return {c.name: c for c in self.columns}

    @property
    def key_column(self) -> str:
        if len(self.config.key_columns) != 1:
            raise ConfigError(f"table {self.name}: composite keys are not supported by this stage")
        return self.config.key_columns[0]


@dataclass
class RunContext:
    loaded: LoadedManifest
    run_id: str
    source: SourceDriver
    target: TargetDriver
    blobs: BlobStore
    local_dir: Path
    log: Log
    env: dict[str, str] = field(default_factory=dict)
    tables: dict[str, TableSpec] = field(default_factory=dict)
    target_spec: TargetSpec | None = None  # how a Spark task reopens `target`; None for injected drivers

    @property
    def manifest(self) -> Manifest:
        return self.loaded.manifest

    @property
    def namespace(self) -> str:
        return self.manifest.namespace

    @property
    def blob_prefix(self) -> str:
        return self.manifest.blob_prefix(self.run_id)

    def table(self, name: str) -> TableSpec:
        return self.tables[name]

    def tables_in_order(self) -> list[TableSpec]:
        return [self.tables[t.name] for t in self.manifest.tables_in_order()]

    def data_tables(self) -> list[TableSpec]:
        return [self.tables[t.name] for t in self.manifest.data_tables()]

    def selection_for(self, table: TableConfig) -> SelectionSpec:
        m = self.manifest
        sel = table.selection
        parent_table = m.table(sel.parent.table).qualified_name if sel.parent else None
        classes: tuple[str, ...] = ()
        if not sel.is_all:
            assert sel.retention_classes
            classes = tuple(m.selection_sets[sel.retention_classes])
        return SelectionSpec(
            key_columns=tuple(table.key_columns),
            all_rows=sel.is_all,
            class_column=sel.class_column,
            classes=classes,
            last_access_column=sel.last_access_column,
            last_access_before=sel.last_access_before,
            parent_table=parent_table,
            parent_child_columns=tuple(sel.parent.columns) if sel.parent else (),
            parent_columns=tuple(sel.parent.references) if sel.parent else (),
        )

    def range_dir(self, table: str) -> Path:
        return self.local_dir / self.blob_prefix / table


def build_table_specs(loaded: LoadedManifest) -> dict[str, TableSpec]:
    m = loaded.manifest
    typemap = load_typemap(loaded.resolve(m.target.typemap))
    out: dict[str, TableSpec] = {}
    for t in m.tables:
        cb = parse_copybook(loaded.resolve(m.source.copybook_dir) / t.copybook, t.record_length)
        specs = build_column_specs(t, cb, typemap, m.source.record_format.text_encoding)
        out[t.name] = TableSpec(config=t, copybook=cb, columns=specs)
    return out


def require_env(env: dict[str, str], names: list[str], what: str) -> None:
    missing = [n for n in names if not env.get(n)]
    if missing:
        raise ConfigError(f"{what}: missing environment variable(s) {missing}")


def local_staging_dir(m: Manifest, env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get(m.staging.connection_env.local_dir) or DEFAULT_LOCAL_STAGING)

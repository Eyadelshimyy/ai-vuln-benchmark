"""Wires manifest + drivers + staging into a RunContext and runs stage verbs with contract exit codes."""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from .config import RUN_ID_RE, LoadedManifest, load_manifest
from .context import Log, RunContext, build_table_specs, local_staging_dir, require_env
from .drivers.base import SelectionSpec, SourceDriver, TargetDriver, Value
from .drivers.factory import TargetSpec, target_spec
from .errors import EXIT_OK, EXIT_UNEXPECTED, ConfigError, ForeignRunError, LdmError
from .stages import extract, init, load, purge, reconcile, validate
from .staging import AzureBlobStore, BlobStore, NoBlobStore, S3BlobStore

STAGES: dict[str, Callable[[RunContext], dict[str, dict[str, int]]]] = {
    "extract": extract.run,
    "load": load.run,
    "validate": validate.run,
    "purge": purge.run,
    "reconcile": reconcile.run,
}
ALL_ORDER = ("extract", "load", "validate", "purge", "reconcile")
SOURCE_VERBS = ("extract", "purge", "all")
HOSTS = ("eks", "aca", "local")


class UnconfiguredSource:
    """Stands in for the source on target-only verbs (load/validate/reconcile/init), e.g. when run in ACA."""

    def __init__(self, verb: str) -> None:
        self._verb = verb

    def _refuse(self) -> ConfigError:
        return ConfigError(f"verb {self._verb!r} must not touch the source; source credentials were not loaded")

    def connect(self) -> None:
        raise self._refuse()

    def close(self) -> None:
        return None

    def select_keys(self, schema: str, table: str, selection: SelectionSpec) -> list[str]:
        raise self._refuse()

    def fetch_range(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        selection: SelectionSpec,
        key_from: str,
        key_to: str,
    ) -> Iterator[dict[str, Value]]:
        raise self._refuse()

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
        raise self._refuse()

    def audited_keys(self, run_id: str, table: str, keys: Sequence[str]) -> set[str]:
        raise self._refuse()


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise ConfigError(f"--run-id {run_id!r} must match {RUN_ID_RE.pattern}")
    return run_id


def make_source(loaded: LoadedManifest, env: Mapping[str, str]) -> SourceDriver:
    src = loaded.manifest.source
    if src.driver == "db2":
        from .drivers.db2 import Db2Source

        ce = src.connection_env
        require_env(dict(env), [ce.host, ce.port, ce.database, ce.user, ce.password], "source db2")
        return Db2Source(
            host=env[ce.host],
            port=env[ce.port],
            database=env[ce.database],
            user=env[ce.user],
            password=env[ce.password],
        )
    if src.driver == "oracle":
        from .drivers.oracle import OracleSource

        ce = src.connection_env
        require_env(dict(env), [ce.host, ce.port, ce.database, ce.user, ce.password], "source oracle")
        return OracleSource(
            host=env[ce.host],
            port=env[ce.port],
            service=env[ce.database],
            user=env[ce.user],
            password=env[ce.password],
        )
    raise ConfigError(f"source.driver {src.driver!r} has no driver in this build (known: db2, oracle)")


def make_target(loaded: LoadedManifest, env: Mapping[str, str]) -> TargetDriver:
    return target_spec(loaded, env).open()


def make_blobs(loaded: LoadedManifest, env: Mapping[str, str]) -> BlobStore:
    """S3 when the bucket variable is set, else Azure Blob when the storage account is, else local files only."""
    ce = loaded.manifest.staging.connection_env
    bucket = env.get(ce.bucket) if ce.bucket else None
    if bucket:
        return S3BlobStore(
            bucket,
            region=env.get(ce.region) if ce.region else None,
            endpoint_url=env.get(ce.endpoint_url) if ce.endpoint_url else None,
        )
    account = env.get(ce.storage_account) if ce.storage_account else None
    if not account:
        return NoBlobStore()
    if not loaded.manifest.azure:
        raise ConfigError(f"{ce.storage_account} is set but this namespace has azure: false")
    container = env.get(ce.container) if ce.container else None
    if not container:
        raise ConfigError(f"{ce.storage_account} is set but {ce.container} is not")
    return AzureBlobStore(account, container, env.get("AZ_STORAGE_KEY"), env.get("AZURE_CLIENT_ID"))


def build_context(
    manifest_path: Path,
    namespace: str,
    run_id: str,
    env: Mapping[str, str] | None = None,
    verb: str = "all",
    source: SourceDriver | None = None,
    target: TargetDriver | None = None,
    blobs: BlobStore | None = None,
    log: Log | None = None,
) -> RunContext:
    env = dict(os.environ if env is None else env)
    loaded = load_manifest(manifest_path, namespace)
    host = env.get("LDM_HOST") or "local"
    if host not in HOSTS:
        raise ConfigError(f"LDM_HOST={host!r}: expected one of {HOSTS}")
    tables = build_table_specs(loaded)
    if source is None:
        source = make_source(loaded, env) if verb in SOURCE_VERBS else UnconfiguredSource(verb)
    spec: TargetSpec | None = None
    if target is None:
        spec = target_spec(loaded, env)
        target = spec.open()
    ctx = RunContext(
        loaded=loaded,
        run_id=run_id,
        source=source,
        target=target,
        blobs=blobs if blobs is not None else make_blobs(loaded, env),
        local_dir=local_staging_dir(loaded.manifest, env),
        log=log or Log(),
        env=env,
        tables=tables,
        target_spec=spec,
    )
    for ts in ctx.tables_in_order():
        ctx.target.register_table(ts.name, ts.config.key_columns, ts.config.hash_columns, ts.columns)
    return ctx


def prepare_run(ctx: RunContext) -> None:
    """DDL part of init, then mig.runs / mig.run_ledger rows for this run (every stage verb)."""
    ctx.log.stage = "INIT"
    init.apply_ddl(ctx)
    m = ctx.manifest
    job_image = ctx.env.get("LDM_JOB_IMAGE")
    recorded = ctx.target.ensure_run(ctx.run_id, ctx.namespace, m.purge, job_image, ctx.loaded.sha256)
    if recorded != ctx.loaded.sha256:
        raise ForeignRunError(
            f"run {ctx.run_id} in {ctx.namespace} was started with manifest sha256 {recorded[:12]}..., "
            f"current manifest is {ctx.loaded.sha256[:12]}...; a changed manifest needs a new --run-id"
        )
    ctx.target.refresh_run(ctx.run_id, ctx.namespace, m.purge, job_image)
    ctx.target.ensure_ledger(ctx.run_id, ctx.namespace, [(t.name, t.order, t.role) for t in m.tables_in_order()])
    status = ctx.target.get_run_status(ctx.run_id, ctx.namespace)
    if status in ("CLOSED",):
        raise ConfigError(f"run {ctx.run_id} in {ctx.namespace} is already CLOSED; use a new --run-id")
    if status == "FAILED":
        ctx.log.warn(f"run {ctx.run_id} previously FAILED; resuming")
        ctx.target.set_run_status(ctx.run_id, ctx.namespace, "RUNNING", None)


def run_stage(ctx: RunContext, stage: str) -> dict[str, dict[str, int]]:
    ctx.log.stage = stage.upper()
    ctx.log.info(f"start run_id={ctx.run_id} namespace={ctx.namespace} host={ctx.env.get('LDM_HOST') or 'local'}")
    result = STAGES[stage](ctx)
    ctx.log.info("done")
    return result


def execute(ctx: RunContext, verb: str, apply_sql: list[Path] | None = None) -> tuple[int, dict[str, dict[str, int]]]:
    """Run a verb; returns (exit_code, tables). Never raises for LdmError - the code carries it."""
    tables: dict[str, dict[str, int]] = {}
    try:
        if verb in SOURCE_VERBS:
            ctx.source.connect()
        ctx.target.connect()
        if verb == "init":
            ctx.log.stage = "INIT"
            tables = init.run(ctx, apply_sql)
            return EXIT_OK, tables
        prepare_run(ctx)
        stages = ALL_ORDER if verb == "all" else (verb,)
        for stage in stages:
            for name, counts in run_stage(ctx, stage).items():
                tables.setdefault(name, {}).update(counts)
        return EXIT_OK, tables
    except LdmError as e:
        ctx.log.error(f"{type(e).__name__}: {e}")
        for name, counts in e.tables.items():
            tables.setdefault(name, {}).update(counts)
        if e.exit_code != EXIT_OK and verb != "init" and not isinstance(e, ForeignRunError):
            _mark_failed(ctx, e.exit_code)
        return e.exit_code, tables
    except Exception as e:  # noqa: BLE001 - contract: any other error is exit 1 with the traceback on stderr
        ctx.log.error(f"unexpected {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        if verb != "init":
            _mark_failed(ctx, EXIT_UNEXPECTED)
        return EXIT_UNEXPECTED, tables
    finally:
        for drv in (ctx.source, ctx.target):
            try:
                drv.close()
            except Exception:  # noqa: BLE001, S110 - best effort on shutdown
                pass


def _mark_failed(ctx: RunContext, exit_code: int) -> None:
    try:
        if ctx.target.get_run_status(ctx.run_id, ctx.namespace) == "RUNNING":
            ctx.target.set_run_status(ctx.run_id, ctx.namespace, "FAILED", exit_code)
    except Exception as e:  # noqa: BLE001 - the original failure is what the caller must see
        ctx.log.error(f"could not mark run FAILED: {e}")


def emit_result(run_id: str, namespace: str, stage: str, exit_code: int, tables: dict[str, dict[str, int]]) -> None:
    print(
        json.dumps(
            {
                "run_id": run_id,
                "namespace": namespace,
                "stage": stage,
                "exit_code": exit_code,
                "tables": tables,
            }
        )
    )

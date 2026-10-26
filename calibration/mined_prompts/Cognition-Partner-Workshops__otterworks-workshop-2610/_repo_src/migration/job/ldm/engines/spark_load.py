"""LOAD engine on PySpark local mode: convert, route and insert one key range inside the migration Job.

The range's unload file is split into bounded slices of `records_per_slice` fixed-width records. Each Spark task
opens the file, reads only its slice, converts every record with `convert_record` and emits a routed record:
staged row or reject. A shuffle on source_key then resolves duplicates exactly like the serial engine (first
well-formed occurrence wins, later ones are DUPLICATE_SOURCE_KEY rejects, malformed records are rejected with
their own rule). Sink tasks reopen the target from the picklable `TargetSpec`, insert staged rows in
`load_batch_rows` batches, turn target refusals into rejects, and return only their counts; the driver never
materialises rows (no collect / toLocalIterator / toPandas).

Everything runs in the one-shot Job pod: `master` defaults to local[*], `spark.local.dir` lives under the
staging volume, and no Spark service, operator or cluster is involved.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..context import RunContext, TableSpec
from ..convert import convert_record
from ..drivers.base import InsertFailure, KeyRange, Reject, StagedRow
from ..drivers.factory import TargetSpec
from ..errors import ConfigError
from ..stages.load import begin_range, duplicate_reject, failure_reject, finish_range
from ..typemap import ColumnSpec

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

Slice = tuple[int, int]  # (first record index, one past the last)


@dataclass(frozen=True)
class RangeJob:
    """Everything a task needs about the range; small and picklable."""

    run_id: str
    namespace: str
    table: str
    range_seq: int
    path: str
    lrecl: int
    key_columns: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    load_batch_rows: int


@dataclass(frozen=True)
class Routed:
    """One record after conversion. `row` is set for a well-formed record, `reject` otherwise."""

    source_key: str
    index: int
    row: StagedRow | None
    reject: Reject | None


def slices(record_count: int, records_per_slice: int) -> list[Slice]:
    if records_per_slice <= 0:
        raise ConfigError("execution.spark.records_per_slice must be positive")
    return [(i, min(i + records_per_slice, record_count)) for i in range(0, record_count, records_per_slice)]


def read_slice(path: str, lrecl: int, sl: Slice) -> Iterator[tuple[int, bytes]]:
    """Yield (record index, record bytes) for one slice, reading only that slice's bytes."""
    first, last = sl
    with open(path, "rb") as f:
        f.seek(first * lrecl)
        data = f.read((last - first) * lrecl)
    if len(data) != (last - first) * lrecl:
        raise ConfigError(f"{path}: slice {sl} is short ({len(data)} bytes)")
    for i in range(last - first):
        yield first + i, data[i * lrecl : (i + 1) * lrecl]


def convert_slice(job: RangeJob, sl: Slice) -> Iterator[Routed]:
    specs = list(job.columns)
    for index, rec in read_slice(job.path, job.lrecl, sl):
        conv = convert_record(rec, specs)
        if conv.ok:
            row = StagedRow(job.table, conv.source_key, job.range_seq, rec, conv.target_row())
            yield Routed(conv.source_key, index, row, None)
            continue
        err = conv.error
        assert err is not None
        reject = Reject(
            job.table,
            conv.source_key,
            "LOAD",
            err.rule,
            err.column,
            rec,
            err.field_bytes,
            err.sqlstate,
            None,
            err.error[:4000],
            job.range_seq,
        )
        yield Routed(conv.source_key, index, None, reject)


def route_key(job: RangeJob, records: Iterable[Routed]) -> Iterator[Routed]:
    """Resolve one source_key: the first well-formed record (file order) is staged, later ones are duplicates.

    Malformed records keep their own reject and never claim the key (CONTRACTS.md §9.4 rule priority).
    """
    kept = False
    for r in sorted(records, key=lambda r: r.index):
        if r.row is None:
            yield r
        elif not kept:
            kept = True
            yield r
        else:
            reject = duplicate_reject(job.table, r.source_key, job.key_columns[0], r.row.raw_bytes, job.range_seq)
            yield Routed(r.source_key, r.index, None, reject)


def sink_partition(job: RangeJob, spec: TargetSpec, records: Iterable[Routed]) -> Iterator[tuple[int, int]]:
    """Insert one partition's routed records; yields a single (loaded, rejected) pair."""
    target = spec.open()
    target.register_table(job.table, list(job.key_columns), [], list(job.columns))
    target.connect()
    loaded = rejected = 0
    batch: list[StagedRow] = []
    rejects: list[Reject] = []
    try:

        def flush() -> None:
            nonlocal loaded, rejected
            if batch:
                failures: list[InsertFailure] = target.insert_staging(job.run_id, job.namespace, job.table, batch)
                failed = {f.source_key for f in failures}
                by_key = {r.source_key: r for r in batch}
                rejects.extend(failure_reject(f, by_key[f.source_key], list(job.key_columns)) for f in failures)
                loaded += len(batch) - len(failed)
                rejected += len(failed)
                batch.clear()
            if rejects:
                target.insert_rejects(job.run_id, job.namespace, rejects)
                rejects.clear()

        for r in records:
            if r.row is not None:
                batch.append(r.row)
            else:
                assert r.reject is not None
                rejects.append(r.reject)
                rejected += 1
            if len(batch) >= job.load_batch_rows:
                flush()
        flush()
    finally:
        target.close()
    yield loaded, rejected


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def spark_session(ctx: RunContext) -> SparkSession:
    """One local-mode session per Job process, scratch space under the staging volume."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise ConfigError("pyspark is not installed; pip install 'ldm[spark]'") from e
    cfg = ctx.manifest.execution.spark
    if not cfg.master.startswith("local"):
        raise ConfigError(f"execution.spark.master {cfg.master!r}: only local[...] runs inside the migration Job")
    scratch = ctx.local_dir / "spark"
    scratch.mkdir(parents=True, exist_ok=True)
    pkg_root = str(Path(__file__).resolve().parents[2])
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_root, os.environ.get("PYTHONPATH")) if p)
    return (
        SparkSession.builder.master(cfg.master)
        .appName(f"ldm-load-{ctx.namespace}-{ctx.run_id}")
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.local.dir", str(scratch))
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.default.parallelism", str(cfg.shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def run_range(
    session: SparkSession, job: RangeJob, spec: TargetSpec, record_count: int, per_slice: int, sinks: int
) -> tuple[int, int]:
    parts = slices(record_count, per_slice)
    if not parts:
        return 0, 0
    sc = session.sparkContext
    routed = (
        sc.parallelize(parts, len(parts))
        .flatMap(lambda sl: convert_slice(job, sl))
        .groupBy(lambda r: r.source_key, numPartitions=sinks)
        .flatMap(lambda kv: route_key(job, kv[1]))
    )
    return routed.mapPartitions(lambda it: sink_partition(job, spec, it)).reduce(_add)


def load_range_spark(ctx: RunContext, ts: TableSpec, rng: KeyRange) -> tuple[int, int]:
    if ctx.target_spec is None:
        raise ConfigError("execution.load_engine spark needs a reconnectable target (target.provider + env)")
    local, count = begin_range(ctx, ts, rng)
    cfg = ctx.manifest.execution.spark
    job = RangeJob(
        run_id=ctx.run_id,
        namespace=ctx.namespace,
        table=ts.name,
        range_seq=rng.range_seq,
        path=str(local),
        lrecl=ts.config.record_length,
        key_columns=tuple(ts.config.key_columns),
        columns=tuple(ts.columns),
        load_batch_rows=ctx.manifest.batch.load_batch_rows,
    )
    loaded, rejected = run_range(
        spark_session(ctx), job, ctx.target_spec, count, cfg.records_per_slice, cfg.shuffle_partitions
    )
    return finish_range(ctx, ts, rng, loaded, rejected)

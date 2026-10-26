"""Spark LOAD engine pieces that need no database: bounded slicing, conversion + duplicate routing, forbidden APIs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ldm.config import load_manifest
from ldm.context import build_table_specs
from ldm.convert import encode_record
from ldm.engines import spark_load
from ldm.engines.spark_load import RangeJob, Routed, convert_slice, read_slice, route_key, slices
from ldm.errors import ConfigError

from .conftest import make_manifest_tree, policy_row

ENGINE_SRC = Path(spark_load.__file__).read_text(encoding="utf-8")


def _job(tmp_path: Path, records: list[dict], token: str = "sp1") -> tuple[RangeJob, int]:
    manifest = make_manifest_tree(tmp_path, token, load_engine="spark")
    loaded = load_manifest(manifest, f"{token}-after")
    ts = build_table_specs(loaded)["RETNPLCY"]
    data = b"".join(encode_record(r, ts.columns, ts.config.record_length) for r in records)
    path = tmp_path / "000001.dat"
    path.write_bytes(data)
    job = RangeJob(
        run_id="r1",
        namespace=f"{token}-after",
        table="RETNPLCY",
        range_seq=1,
        path=str(path),
        lrecl=ts.config.record_length,
        key_columns=tuple(ts.config.key_columns),
        columns=tuple(ts.columns),
        load_batch_rows=2,
    )
    return job, len(records)


def test_slices_cover_every_record_exactly_once() -> None:
    assert slices(0, 5) == []
    assert slices(5, 5) == [(0, 5)]
    assert slices(12, 5) == [(0, 5), (5, 10), (10, 12)]
    with pytest.raises(ConfigError):
        slices(3, 0)


def test_read_slice_reads_only_its_records(tmp_path: Path) -> None:
    lrecl = 4
    path = tmp_path / "f.dat"
    path.write_bytes(b"".join(f"{i:04d}".encode() for i in range(10)))
    got = list(read_slice(str(path), lrecl, (3, 6)))
    assert got == [(3, b"0003"), (4, b"0004"), (5, b"0005")]
    with pytest.raises(Exception, match="short"):
        list(read_slice(str(path), lrecl, (8, 12)))


def test_route_key_keeps_first_well_formed_record_and_rejects_later_duplicates(tmp_path: Path) -> None:
    rows = [
        policy_row("AAA1", "first", 7, "", "Y"),
        policy_row("BBB1", "other", 1, "", "Y"),
        policy_row("AAA1", "second", 7, "", "Y"),
    ]
    job, n = _job(tmp_path, rows)
    routed = list(convert_slice(job, (0, n)))
    assert [r.source_key for r in routed] == ["AAA1", "BBB1", "AAA1"] and all(r.row is not None for r in routed)
    by_key: dict[str, list[Routed]] = {}
    for r in routed:
        by_key.setdefault(r.source_key, []).append(r)
    # shuffle order is arbitrary: feed the duplicates reversed and expect file order to decide
    out = list(route_key(job, reversed(by_key["AAA1"])))
    assert out[0].row is not None and out[0].row.values["POLICY_DESC"] == "first"
    assert out[1].row is None and out[1].reject is not None and out[1].reject.rule == "DUPLICATE_SOURCE_KEY"
    assert out[1].reject.raw_bytes == routed[2].row.raw_bytes  # type: ignore[union-attr]


def test_convert_slice_rejects_malformed_records_with_conversion_rule(tmp_path: Path) -> None:
    rows = [policy_row("AAA1", "first", 7, "", "Y")]
    job, n = _job(tmp_path, rows)
    rec = bytearray(Path(job.path).read_bytes())
    ts = next(c for c in job.columns if c.name == "EFFECTIVE_TS")
    rec[ts.field.offset] = ord("X")  # TIMESTAMP(12) text with a non-digit
    Path(job.path).write_bytes(bytes(rec))
    (only,) = convert_slice(job, (0, n))
    assert only.row is None and only.reject is not None
    assert (
        only.reject.rule == "TIMESTAMP_INVALID" and only.reject.stage == "LOAD" and only.reject.raw_bytes == bytes(rec)
    )
    assert list(route_key(job, [only])) == [only]  # a malformed record never claims its key


def test_engine_never_materialises_rows_on_the_driver() -> None:
    forbidden = re.compile(r"\.(collect|toLocalIterator|toPandas|binaryFiles)\(")
    code = "\n".join(line for line in ENGINE_SRC.splitlines() if not line.lstrip().startswith(('"""', "#", "no ")))
    body = code.split('"""', 2)[-1]  # skip the module docstring, which names the APIs it avoids
    assert not forbidden.search(body), forbidden.search(body)


def test_spark_session_refuses_non_local_master(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    from .conftest import make_ctx, seed_source

    manifest = make_manifest_tree(tmp_path, "sp2", load_engine="spark")
    overlay = manifest.parent / "manifests" / "sp2-after.yaml"
    overlay.write_text(overlay.read_text(encoding="utf-8") + "  spark:\n    master: spark://cluster:7077\n")
    ctx = make_ctx(tmp_path, seed_source().source, manifest=manifest, namespace="sp2-after")
    with pytest.raises(ConfigError, match="only local"):
        spark_load.spark_session(ctx)


def test_spark_local_pipeline_converts_and_routes(tmp_path: Path) -> None:
    """Convert + shuffle + route on a real local[2] session; only aggregate counts reach the driver."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    rows = [policy_row(f"K{i:03d}", f"d{i}", 1 + i % 3, "", "Y") for i in range(23)]
    rows.insert(7, policy_row("K001", "dup", 1, "", "Y"))
    job, n = _job(tmp_path, rows, "sp3")
    parts = slices(n, 5)
    assert len(parts) == 5
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ldm-test")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.dir", str(tmp_path / "spark"))
        .getOrCreate()
    )
    try:
        sc = session.sparkContext
        routed = (
            sc.parallelize(parts, len(parts))
            .flatMap(lambda sl: convert_slice(job, sl))
            .groupBy(lambda r: r.source_key, numPartitions=3)
            .flatMap(lambda kv: route_key(job, kv[1]))
        )
        staged, rejected = routed.map(lambda r: (1, 0) if r.row is not None else (0, 1)).reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1])
        )
        dup_rules = routed.filter(lambda r: r.reject is not None).map(lambda r: r.reject.rule).countByValue()
    finally:
        session.stop()
    assert (staged, rejected) == (23, 1)
    assert dict(dup_rules) == {"DUPLICATE_SOURCE_KEY": 1}

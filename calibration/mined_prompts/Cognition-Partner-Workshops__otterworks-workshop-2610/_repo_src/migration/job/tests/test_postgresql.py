"""PostgreSQL target + PySpark local-mode LOAD against a real database.

Skipped unless LDM_TEST_PG_HOST is set (CI starts a postgres service container; locally e.g.
`docker run -e POSTGRES_PASSWORD=ldm -e POSTGRES_USER=ldm -e POSTGRES_DB=ldm -p 55432:5432 postgres:16`).
Every test uses a fresh namespace so runs never collide, and drops the namespace's rows afterwards.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from ldm.context import Log
from ldm.drivers.base import Reject, StagedRow
from ldm.drivers.factory import target_spec
from ldm.drivers.postgresql import PostgresTarget
from ldm.runner import build_context, execute, prepare_run
from ldm.stages import extract, load
from ldm.staging import DirectoryBlobStore

from .conftest import Seed, make_manifest_tree, seed_source

PG_ENV = {
    "PG_HOST": os.environ.get("LDM_TEST_PG_HOST", ""),
    "PG_PORT": os.environ.get("LDM_TEST_PG_PORT", "5432"),
    "PG_DATABASE": os.environ.get("LDM_TEST_PG_DATABASE", "ldm"),
    "PG_USER": os.environ.get("LDM_TEST_PG_USER", "ldm"),
    "PG_PASSWORD": os.environ.get("LDM_TEST_PG_PASSWORD", "ldm"),
}

pytestmark = pytest.mark.skipif(not PG_ENV["PG_HOST"], reason="LDM_TEST_PG_HOST not set")


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        **PG_ENV,
        "LOCAL_STAGING_DIR": str(tmp_path / "staging"),
        "LDM_UNLOAD_MODE": "builtin",
        "LDM_HOST": "local",
    }


def _ctx(tmp_path: Path, manifest: Path, token: str, seed: Seed, run_id: str):
    return build_context(
        manifest,
        f"{token}-after",
        run_id,
        env=_env(tmp_path),
        source=seed.source,
        blobs=DirectoryBlobStore(tmp_path / "blobs"),
        log=Log(),
    )


def _staged(target: PostgresTarget, run_id: str, ns: str, table: str) -> dict[str, StagedRow]:
    out: dict[str, StagedRow] = {}
    for batch in target.iter_staging(run_id, ns, table, 500):
        for r in batch:
            out[r.source_key] = r
    return out


def _rejects(target: PostgresTarget, run_id: str, ns: str) -> set[tuple[str, str, str]]:
    return {(f.table_name, f.source_key, f.rule) for f in target.failures(run_id, ns)}


@pytest.fixture
def token() -> str:
    return "p" + uuid.uuid4().hex[:5]


@pytest.fixture
def cleanup():
    """Drop everything a test's namespace wrote (mig/stg/arch rows, purge audit)."""
    targets: list[tuple[PostgresTarget, str]] = []
    yield targets
    for t, ns in targets:
        t.connect()
        try:
            t.drop_namespace(ns)
        finally:
            t.close()


def test_full_run_on_postgresql_serial(tmp_path: Path, token: str, cleanup) -> None:
    seed = seed_source()
    manifest = make_manifest_tree(tmp_path, token)
    ctx = _ctx(tmp_path, manifest, token, seed, "run-serial")
    assert isinstance(ctx.target, PostgresTarget)
    cleanup.append((ctx.target, ctx.namespace))
    code, tables = execute(ctx, "all")
    assert code == 0, tables
    p = seed.planted
    n_load_rejects = sum(len(p[c]) for c in ("MIG-01", "MIG-02", "MIG-03"))
    d = tables["DOCARCH"]
    assert d["extracted"] == seed.docarch_selected
    assert d["loaded"] + d["rejected"] == d["extracted"]
    assert d["rejected"] == n_load_rejects
    assert d["validated"] == d["loaded"] - len(p["MIG-04"]) - len(p["MIG-07"])
    assert tables["RETNPLCY"]["loaded"] == 10
    ctx.target.connect()
    try:
        assert ctx.target.get_run_status(ctx.run_id, ctx.namespace) == "CLOSED"
        rules = {r for _, _, r in _rejects(ctx.target, ctx.run_id, ctx.namespace)}
        assert {
            "CCSID_UNMAPPABLE",
            "DECIMAL_OVERFLOW",
            "DATE_INVALID",
            "HASH_MISMATCH",
            "ORPHAN_PARENT_NOT_SELECTED",
        } <= rules
    finally:
        ctx.target.close()


def test_spark_engine_matches_serial_engine(tmp_path: Path, token: str, cleanup) -> None:
    pytest.importorskip("pyspark")
    seed = seed_source()
    # one namespace per engine: both would otherwise see each other's stg rows as a prior partial run (MIG-06)
    serial_manifest = make_manifest_tree(tmp_path / "s", token + "r", load_engine="serial")
    spark_manifest = make_manifest_tree(tmp_path / "k", token + "a", load_engine="spark")
    # tiny slices/batches so several Spark tasks, shuffles and insert batches are exercised on the seed
    text = spark_manifest.read_text(encoding="utf-8").replace("load_batch_rows: 5000", "load_batch_rows: 7")
    spark_manifest.write_text(text, encoding="utf-8")
    overlay = spark_manifest.parent / "manifests" / f"{token}a-after.yaml"
    overlay.write_text(
        overlay.read_text(encoding="utf-8") + "  spark:\n    records_per_slice: 5\n    shuffle_partitions: 3\n",
        encoding="utf-8",
    )

    results: dict[str, tuple[dict[str, dict[str, StagedRow]], set[tuple[str, str, str]], dict]] = {}
    for engine, manifest in (("serial", serial_manifest), ("spark", spark_manifest)):
        ctx = _ctx(tmp_path / engine[0], manifest, token + engine[2], seed, f"run-{engine}")
        assert isinstance(ctx.target, PostgresTarget)
        cleanup.append((ctx.target, ctx.namespace))
        assert ctx.manifest.execution.load_engine == engine
        ctx.source.connect()
        ctx.target.connect()
        try:
            prepare_run(ctx)
            extract.run(ctx)
            counts = load.run(ctx)
            staged = {t.name: _staged(ctx.target, ctx.run_id, ctx.namespace, t.name) for t in ctx.tables_in_order()}
            rejects = _rejects(ctx.target, ctx.run_id, ctx.namespace)
        finally:
            ctx.source.close()
            ctx.target.close()
        results[engine] = (staged, rejects, counts)

    (s_rows, s_rej, s_counts), (k_rows, k_rej, k_counts) = results["serial"], results["spark"]
    assert k_counts == s_counts
    assert k_rej == s_rej and s_rej
    for table, rows in s_rows.items():
        assert set(k_rows[table]) == set(rows)
        for key, row in rows.items():
            other = k_rows[table][key]
            assert other.raw_bytes == row.raw_bytes
            assert other.values == row.values
            assert other.key_range_seq == row.key_range_seq


def test_spark_engine_rejects_duplicate_keys_like_serial(tmp_path: Path, token: str, cleanup) -> None:
    """A duplicate source key inside one unload keeps the first well-formed record on both engines."""
    pytest.importorskip("pyspark")
    from ldm.convert import encode_record
    from ldm.drivers.base import KeyRange

    from .conftest import policy_row

    outcomes = {}
    for engine in ("serial", "spark"):
        seed = seed_source(generated=0, plant=False)
        manifest = make_manifest_tree(tmp_path / engine, token + engine[2], load_engine=engine)
        ctx = _ctx(tmp_path / engine, manifest, token + engine[2], seed, f"dup-{engine}")
        assert isinstance(ctx.target, PostgresTarget)
        cleanup.append((ctx.target, ctx.namespace))
        ts = ctx.table("RETNPLCY")
        rows = [policy_row("AAA1", "first", 7, "", "Y"), policy_row("AAA1", "second", 7, "", "Y")]
        rows += [policy_row("BBB1", "other", 1, "", "Y")]
        data = b"".join(encode_record(r, ts.columns, ts.config.record_length) for r in rows)
        f = ctx.range_dir(ts.name) / "000001.dat"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        ctx.target.connect()
        try:
            prepare_run(ctx)
            rng = KeyRange(ts.name, 1, "AAA1", "BBB1", status="DONE", row_count=3, local_path=str(f))
            ctx.target.insert_key_ranges(ctx.run_id, ctx.namespace, [rng])
            loaded, rejected = load.load_range_with_engine(ctx, ts, rng)
            staged = _staged(ctx.target, ctx.run_id, ctx.namespace, ts.name)
            rejects: list[Reject | object] = list(ctx.target.failures(ctx.run_id, ctx.namespace))
        finally:
            ctx.target.close()
        outcomes[engine] = (loaded, rejected, {k: v.values["POLICY_DESC"] for k, v in staged.items()}, len(rejects))
    assert outcomes["serial"] == outcomes["spark"]
    loaded, rejected, descs, n_rej = outcomes["spark"]
    assert (loaded, rejected, n_rej) == (2, 1, 1)
    assert descs["AAA1"] == "first"


def test_mig06_prior_run_fixture_plants_duplicates(tmp_path: Path, token: str, cleanup) -> None:
    """`ldm init --apply-sql` with the PostgreSQL MIG-06 fixture is idempotent and makes LOAD reject MIG06-* keys."""
    seed = seed_source()
    manifest = make_manifest_tree(tmp_path, token)
    fixture = manifest.parent / "fixtures" / "mig06_prior_run.postgresql.sql"
    fixture.parent.mkdir()
    fixture.write_bytes((manifest.parent / "source" / "seed" / "fixtures" / fixture.name).read_bytes())

    ctx = _ctx(tmp_path, manifest, token, seed, "run-init")
    assert isinstance(ctx.target, PostgresTarget)
    cleanup.append((ctx.target, ctx.namespace))
    for _ in range(2):
        code, tables = execute(ctx, "init", apply_sql=[fixture])
        assert code == 0, tables
        assert tables["_ddl"]["scripts"] == 1
    ns = ctx.namespace
    ctx.target.connect()
    try:
        assert ctx.target.get_run_status("prior-partial", ns) == "ABANDONED"
        stale = ctx.target.get_key_ranges("prior-partial", ns, "DOCARCH")
        assert [(r.range_seq, r.key_from, r.key_to, r.row_count, r.load_status) for r in stale] == [
            (1, "MIG06-0000000001", "MIG06-0000000005", 5, "RUNNING")
        ]
        planted = _staged(ctx.target, "prior-partial", ns, "DOCARCH")
        assert sorted(planted) == [f"MIG06-{k:010d}" for k in range(1, 6)]
        assert planted["MIG06-0000000001"].values["RETENTION_CLASS"] == "AUD7"
    finally:
        ctx.target.close()

    ctx = _ctx(tmp_path, manifest, token, seed, "run-real")
    code, tables = execute(ctx, "all")
    assert code == 0, tables
    ctx.target.connect()
    try:
        rejects = _rejects(ctx.target, ctx.run_id, ns)
        dupes = {k for t, k, r in rejects if t == "DOCARCH" and r == "DUPLICATE_SOURCE_KEY"}
    finally:
        ctx.target.close()
    assert dupes == {k.strip() for k in seed.planted["MIG-06"]}
    assert tables["DOCARCH"]["rejected"] == sum(len(seed.planted[c]) for c in ("MIG-01", "MIG-02", "MIG-03", "MIG-06"))


def test_target_spec_is_picklable_and_reopens(tmp_path: Path, token: str) -> None:
    import pickle

    manifest = make_manifest_tree(tmp_path, token)
    from ldm.config import load_manifest

    spec = pickle.loads(pickle.dumps(target_spec(load_manifest(manifest, f"{token}-after"), _env(tmp_path))))
    t = spec.open()
    assert isinstance(t, PostgresTarget)
    t.connect()
    t.close()

"""End-to-end runs through every stage with the in-memory drivers: MIG-01..07, guards, resume, arithmetic."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from ldm.drivers.base import InsertFailure
from ldm.drivers.fakes import FakeSource, FakeTarget
from ldm.errors import PurgeGuardError, ReconcileError, SourceError
from ldm.runner import execute, prepare_run
from ldm.stages import extract, load, purge, reconcile, validate

from .conftest import Seed, docarch_row, make_ctx, seed_source


def _run_all(tmp_path: Path, seed: Seed, manifest: Path, target: FakeTarget | None = None, **kw):
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest, **kw)
    code, tables = execute(ctx, "all")
    return ctx, code, tables


def _rules(target: FakeTarget, run_id: str, namespace: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in target.failures(run_id, namespace):
        out.setdefault(f.rule, set()).add(f.source_key)
    return out


@pytest.fixture
def seed() -> Seed:
    return seed_source()


def test_full_run_closes_with_all_seven_failure_classes(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    target = FakeTarget()
    # MIG-06: a partially completed prior run left stg.DOCARCH rows for the same keys in this namespace
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after, run_id="prior-partial")
    prepare_run(ctx)
    ts = ctx.table("DOCARCH")
    prior = [
        docarch_row(k.strip(), "AUD7", f"2016-09-0{i}-11.11.11.111111111111")
        for i, k in enumerate(seed.planted["MIG-06"], start=1)
    ]
    from ldm.convert import convert_record, encode_record
    from ldm.drivers.base import StagedRow

    rows = []
    for r in prior:
        rec = encode_record(r, ts.columns, ts.config.record_length)
        conv = convert_record(rec, ts.columns)
        rows.append(StagedRow("DOCARCH", conv.source_key, 1, rec, conv.target_row()))
    target.insert_staging("prior-partial", "t01-after", "DOCARCH", rows)
    target.set_run_status("prior-partial", "t01-after", "ABANDONED", None)

    ctx, code, tables = _run_all(tmp_path, seed, manifest_after, target)
    assert code == 0, tables
    run, ns = ctx.run_id, ctx.namespace

    p = seed.planted
    n_load_rejects = sum(len(p[c]) for c in ("MIG-01", "MIG-02", "MIG-03", "MIG-06"))
    n_validate_fail = len(p["MIG-04"]) + len(p["MIG-07"])
    d = tables["DOCARCH"]
    assert d["extracted"] == seed.docarch_selected
    assert d["loaded"] == seed.docarch_selected - n_load_rejects
    assert d["rejected"] == n_load_rejects
    assert d["validated"] == d["loaded"] - n_validate_fail
    assert d["purged"] == d["validated"]
    f = tables["FILEAUD"]
    assert f["extracted"] == f["loaded"] == seed.fileaud_selected
    assert f["validate_failed"] == len(p["MIG-05"])
    assert f["purged"] == f["validated"] == seed.fileaud_selected - len(p["MIG-05"])
    r = tables["RETNPLCY"]
    assert r == {**r, "extracted": 10, "loaded": 10, "validated": 10, "purged": 0}

    rules = _rules(target, run, ns)
    assert rules["CCSID_UNMAPPABLE"] == set(p["MIG-01"])
    assert rules["DECIMAL_OVERFLOW"] == set(p["MIG-02"])
    assert rules["DATE_INVALID"] == set(p["MIG-03"])
    assert rules["HASH_MISMATCH"] == set(p["MIG-04"])
    assert rules["ORPHAN_PARENT_NOT_SELECTED"] == set(p["MIG-05"])
    assert rules["DUPLICATE_SOURCE_KEY"] == set(p["MIG-06"])
    assert rules["CLASS_TOTAL_MISMATCH"] == set(p["MIG-07"])
    assert "CLASS_COUNT_MISMATCH" not in rules

    by_key = {(x.source_key, x.rule): x for x in target.failures(run, ns)}
    assert by_key[(p["MIG-02"][0], "DECIMAL_OVERFLOW")].sqlstate == "22003"
    assert by_key[(p["MIG-03"][0], "DATE_INVALID")].sqlstate == "22007"
    assert by_key[(p["MIG-06"][0], "DUPLICATE_SOURCE_KEY")].sqlstate == "23000"
    mig01 = by_key[(p["MIG-01"][0], "CCSID_UNMAPPABLE")]
    assert mig01.sqlstate is None and "X'3F'" in mig01.error_text and mig01.field_name == "OWNER_NAME"

    # MIG-07 headline: table total agrees, FIN7/LGL7 sums disagree by 1e-8, counts agree
    totals = {c.retention_class: c for c in target.get_class_totals(run, ns) if c.table_name == "DOCARCH"}
    assert totals["FIN7"].status == "MISMATCH" and totals["LGL7"].status == "MISMATCH"
    assert totals["FIN7"].source_count == totals["FIN7"].target_count
    assert abs(totals["FIN7"].source_sum - totals["FIN7"].target_sum) == Decimal("0.00000001")
    assert sum(c.source_sum for c in totals.values()) == sum(c.target_sum for c in totals.values())
    assert all(c.status == "MATCH" for k, c in totals.items() if k not in ("FIN7", "LGL7"))

    # failed rows stay in the source; validated rows are gone
    remaining = {str(r["ARCH_KEY"]) for r in seed.source.rows("ARCHIVE", "DOCARCH")}
    for cls in ("MIG-01", "MIG-02", "MIG-03", "MIG-04", "MIG-06", "MIG-07"):
        assert set(p[cls]) <= remaining
    assert not (remaining & target.purged_keys(run, ns, "DOCARCH"))
    assert len(seed.source.rows("ARCHIVE", "RETNPLCY")) == 10
    assert target.runs[(run, ns)].status == "CLOSED"

    # purge audit: every purged key was INTENDED before deletion and is now PURGED
    audit = [a for a in target.purge_audit[(run, ns)] if a.table_name == "DOCARCH"]
    assert len(audit) == d["purged"] and all(a.status == "PURGED" for a in audit)

    # report files under LOCAL_STAGING_DIR/<blob_prefix>, uploaded to the blob store, JSON shape per contract
    out_dir = ctx.local_dir / ctx.blob_prefix
    report = json.loads((out_dir / "reconciliation.json").read_text())
    assert report["closes"] is True and report["run_id"] == run and report["namespace"] == ns
    required = ["run_id", "namespace", "generated_at", "tables", "failures", "sessions", "closes"]
    assert [k for k in report if k in required] == required and set(report) - set(required) <= {"class_totals"}
    assert set(report["failures"][0]) - {"native_error", "issue"} == {
        "table",
        "source_key",
        "rule",
        "stage",
        "field",
        "sqlstate",
        "error",
    }
    ct = {(c["table"], c["class"]): c for c in report["class_totals"]}
    assert ct[("DOCARCH", "FIN7")]["source_sum"] != ct[("DOCARCH", "FIN7")]["target_sum"]
    csv_text = (out_dir / "reconciliation.csv").open(encoding="utf-8", newline="").read()
    assert csv_text.startswith("section,table,extracted,loaded,validated,purged,failed\r\n")
    assert "\r\n\r\nsection,table,source_key,rule,stage,field,sqlstate,error\r\n" in csv_text
    assert {t["table"] for t in report["tables"]} == {"RETNPLCY", "DOCARCH", "FILEAUD"}
    doc = next(t for t in report["tables"] if t["table"] == "DOCARCH")
    assert doc["failed"] == n_load_rejects + n_validate_fail
    assert {f["rule"] for f in report["failures"]} == set(rules)
    assert (out_dir / "reconciliation.csv").exists() and (out_dir / "reconciliation.html").exists()
    assert (tmp_path / "blobs" / ctx.blob_prefix / "reconciliation.json").exists()
    # staging cleared after closure
    assert target.count_staging(run, ns, "DOCARCH") == 0


def test_session_links_from_env_and_yaml(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    env = {"DEVIN_SESSION_LINKS": "job=https://example.test/s/1,source=https://example.test/s/2"}
    ctx, code, _ = _run_all(tmp_path, seed, manifest_after, env=env)
    assert code == 0
    report = json.loads((ctx.local_dir / ctx.blob_prefix / "reconciliation.json").read_text())
    labels = {s["label"]: s["url"] for s in report["sessions"]}
    assert labels["job"] == "https://example.test/s/1" and labels["source"] == "https://example.test/s/2"


def test_dry_run_purges_nothing_and_still_closes(tmp_path: Path, seed: Seed, manifest_dry: Path) -> None:
    before = len(seed.source.rows("ARCHIVE", "DOCARCH"))
    ctx, code, tables = _run_all(tmp_path, seed, manifest_dry, namespace="t02-after")
    assert code == 0
    assert tables["DOCARCH"]["purged"] == 0
    assert tables["DOCARCH"]["purge_intended"] == tables["DOCARCH"]["validated"]
    assert len(seed.source.rows("ARCHIVE", "DOCARCH")) == before
    assert not ctx.target.purge_audit[(ctx.run_id, ctx.namespace)]


def test_purge_guard_stops_when_intended_differs_from_validated(
    tmp_path: Path, seed: Seed, manifest_after: Path
) -> None:
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate):
        stage.run(ctx)
    target = ctx.target
    assert isinstance(target, FakeTarget)
    # tamper: the ledger says one more validated row than purge_safe keys exist
    led = target.get_ledger(ctx.run_id, ctx.namespace)["DOCARCH"]
    target.update_ledger(ctx.run_id, ctx.namespace, "DOCARCH", validated=(led.validated or 0) + 1)
    before = len(seed.source.rows("ARCHIVE", "DOCARCH"))
    with pytest.raises(PurgeGuardError):
        purge.run(ctx)
    assert len(seed.source.rows("ARCHIVE", "DOCARCH")) == before
    assert not target.purged_keys(ctx.run_id, ctx.namespace, "DOCARCH")
    assert not target.purged_keys(ctx.run_id, ctx.namespace, "FILEAUD")


def test_purge_guard_exit_code_is_3(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate):
        stage.run(ctx)
    target = ctx.target
    assert isinstance(target, FakeTarget)
    led = target.get_ledger(ctx.run_id, ctx.namespace)["FILEAUD"]
    target.update_ledger(ctx.run_id, ctx.namespace, "FILEAUD", validated=(led.validated or 0) - 1)
    code, _ = execute(ctx, "purge")
    assert code == 3
    assert target.runs[(ctx.run_id, ctx.namespace)].status == "FAILED"


def test_delete_count_mismatch_rolls_back_batch(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate):
        stage.run(ctx)
    victim = "DA00000000000001".ljust(16)
    seed.source.short_delete_for.add(victim)
    with pytest.raises(PurgeGuardError):
        purge.run(ctx)
    target = ctx.target
    assert isinstance(target, FakeTarget)
    audit = {
        a.source_key: a.status for a in target.purge_audit[(ctx.run_id, ctx.namespace)] if a.table_name == "DOCARCH"
    }
    assert audit[victim] == "ROLLED_BACK"
    assert victim in {str(r["ARCH_KEY"]) for r in seed.source.rows("ARCHIVE", "DOCARCH")}


def test_db2_sqlcode_surfaces_on_purge_after_retries_exhausted(
    tmp_path: Path, seed: Seed, manifest_after: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate):
        stage.run(ctx)
    slept: list[float] = []
    monkeypatch.setattr(purge, "_sleep", slept.append)
    seed.source.fail_delete_for.add("FA000000000000000001".ljust(20))
    code, _ = execute(ctx, "purge")
    assert code == 1
    b = ctx.manifest.batch
    assert slept == [
        min(b.purge_retry_backoff_s * 2**i, b.purge_retry_max_backoff_s) for i in range(b.purge_retry_attempts)
    ]
    target = ctx.target
    assert isinstance(target, FakeTarget)
    failed = [e for e in target.stage_log if e["stage"] == "PURGE" and e["status"] == "FAILED"]
    assert failed and "SQLCODE=-911" in str(failed[0]["message"]) and "SQLSTATE=40001" in str(failed[0]["message"])


def test_transient_source_pressure_retries_batch_then_purges(
    tmp_path: Path, seed: Seed, manifest_after: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate):
        stage.run(ctx)
    source = seed.source
    real = source.purge_batch
    log_full = {"remaining": 2}

    def flaky(schema, table, key_column, keys, run_id, namespace, batch_no):
        if table == "DOCARCH" and log_full["remaining"]:
            log_full["remaining"] -= 1
            raise SourceError(-964, "57011", f"{table} batch {batch_no}: SQL0964C transaction log full")
        return real(schema, table, key_column, keys, run_id, namespace, batch_no)

    monkeypatch.setattr(source, "purge_batch", flaky)
    slept: list[float] = []
    monkeypatch.setattr(purge, "_sleep", slept.append)
    code, tables = execute(ctx, "purge")
    assert code == 0, tables
    assert log_full["remaining"] == 0 and slept == [5.0, 10.0]
    assert tables["DOCARCH"]["purged"] == tables["DOCARCH"]["purge_intended"]
    target = ctx.target
    assert isinstance(target, FakeTarget)
    statuses = {a.status for a in target.purge_audit[(ctx.run_id, ctx.namespace)] if a.table_name == "DOCARCH"}
    assert statuses == {"PURGED"}


def test_reconcile_arithmetic_failure_exits_2(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate, purge):
        stage.run(ctx)
    target = ctx.target
    assert isinstance(target, FakeTarget)
    led = target.get_ledger(ctx.run_id, ctx.namespace)["DOCARCH"]
    target.update_ledger(ctx.run_id, ctx.namespace, "DOCARCH", extracted=(led.extracted or 0) + 1)
    with pytest.raises(ReconcileError):
        reconcile.run(ctx)
    assert target.runs[(ctx.run_id, ctx.namespace)].status == "FAILED"
    report = json.loads((ctx.local_dir / ctx.blob_prefix / "reconciliation.json").read_text())
    assert report["closes"] is False
    # staging is kept for forensics when the run does not close
    assert target.count_staging(ctx.run_id, ctx.namespace, "DOCARCH") > 0
    code, _ = execute(make_ctx(tmp_path, seed.source, target, manifest=manifest_after), "reconcile")
    assert code == 2


def test_load_arithmetic_failure_when_target_loses_rows(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    target = FakeTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    extract.run(ctx)
    # an insert failure without a reject row would break extracted == loaded + rejected; the stage records
    # it as a reject instead and the equation still holds
    target.insert_failures["DA00000000000004".ljust(16)] = InsertFailure(
        "DA00000000000004".ljust(16), "22001", 8152, "String or binary data would be truncated"
    )
    out = load.run(ctx)
    d = out["DOCARCH"]
    assert d["loaded"] + d["rejected"] == seed.docarch_selected
    rules = _rules(target, ctx.run_id, ctx.namespace)
    assert "DA00000000000004".ljust(16) in rules["STRING_TRUNCATION"]


def test_extract_resume_skips_completed_ranges(tmp_path: Path, manifest_after: Path) -> None:
    seed = seed_source(generated=12, children_per_parent=1, plant=False)
    target = FakeTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    # force tiny ranges so several are planned
    ctx.manifest.batch.extract_range_rows = 3
    ts = ctx.table("DOCARCH")
    sel = ctx.selection_for(ts.config)
    keys = seed.source.select_keys("ARCHIVE", "DOCARCH", sel)
    ranges = extract.plan_ranges("DOCARCH", keys, 3)
    assert len(ranges) >= 2
    target.insert_key_ranges(ctx.run_id, ctx.namespace, ranges)
    # complete the first range by hand
    extract.run_range(ctx, ts, sel, ranges[0])
    first = target.get_key_ranges(ctx.run_id, ctx.namespace, "DOCARCH")[0]
    assert first.status == "DONE" and first.attempt == 1
    # a resumed extract must not re-run range 1
    out = extract.extract_table(ctx, ts)
    after = target.get_key_ranges(ctx.run_id, ctx.namespace, "DOCARCH")
    assert after[0].attempt == 1 and after[0].sha256_hex == first.sha256_hex
    assert all(r.status == "DONE" and r.attempt == 1 for r in after)
    assert out["extracted"] == len(keys) and out["extract_files"] == len(ranges)
    # files, .cnt and .sha256 exist per range
    for r in after:
        p = Path(r.local_path or "")
        assert p.exists() and p.with_suffix(".cnt").read_text().strip() == str(r.row_count)
        assert p.with_suffix(".sha256").read_text().split()[0] == r.sha256_hex


def test_load_resume_replaces_interrupted_range(tmp_path: Path, manifest_after: Path) -> None:
    seed = seed_source(generated=9, children_per_parent=1, plant=False)
    target = FakeTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    ctx.manifest.batch.extract_range_rows = 2
    extract.run(ctx)
    ranges = target.get_key_ranges(ctx.run_id, ctx.namespace, "DOCARCH")
    ts = ctx.table("DOCARCH")
    # load range 1 fully, pretend range 2 was interrupted after staging half of it
    load.load_range(ctx, ts, ranges[0])
    load.load_range(ctx, ts, ranges[1])
    target.update_key_range(ctx.run_id, ctx.namespace, "DOCARCH", 2, load_status="RUNNING")
    staged_before = target.count_staging(ctx.run_id, ctx.namespace, "DOCARCH")
    out = load.run(ctx)
    assert out["DOCARCH"]["loaded"] == staged_before + sum(r.row_count or 0 for r in ranges[2:])
    assert out["DOCARCH"]["rejected"] == 0
    assert all(r.load_status == "DONE" for r in target.get_key_ranges(ctx.run_id, ctx.namespace, "DOCARCH"))


def test_closed_run_cannot_be_rerun(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    ctx, code, _ = _run_all(tmp_path, seed, manifest_after)
    assert code == 0
    ctx2 = make_ctx(tmp_path, seed.source, ctx.target, manifest=manifest_after)  # type: ignore[arg-type]
    code2, _ = execute(ctx2, "extract")
    assert code2 == 4


def test_before_overlay_is_rejected_with_exit_4(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    from ldm.errors import ConfigError

    with pytest.raises(ConfigError, match="migrate: false"):
        make_ctx(tmp_path, seed.source, manifest=manifest_after, namespace="t01-before")


def test_stage_log_records_ok_and_host(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    ctx, code, _ = _run_all(tmp_path, seed, manifest_after)
    assert code == 0
    target = ctx.target
    assert isinstance(target, FakeTarget)
    statuses = {e["status"] for e in target.stage_log}
    assert statuses == {"OK"}
    stages = {e["stage"] for e in target.stage_log}
    assert stages == {"EXTRACT", "LOAD", "VALIDATE", "PURGE", "RECONCILE"}


def test_source_error_maps_to_exit_1(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    class Broken(FakeSource):
        def connect(self) -> None:
            from ldm.errors import SourceError

            raise SourceError(-30081, "08001", "communication error")

    src = Broken()
    ctx = make_ctx(tmp_path, src, manifest=manifest_after)
    code, _ = execute(ctx, "extract")
    assert code == 1


def test_archived_row_with_different_hash_blocks_purge(tmp_path: Path, manifest_after: Path) -> None:
    """Idempotent re-runs promote nothing twice; a conflicting archive copy must fail the row, not purge its source."""
    seed = seed_source(generated=6, children_per_parent=1, plant=False)
    target = FakeTarget()
    ctx, code, _ = _run_all(tmp_path, seed, manifest_after, target)
    assert code == 0
    # a second namespace-run selects the same keys again, but one row's charge changed in the source
    src2 = seed_source(generated=6, children_per_parent=1, plant=False).source
    changed = src2.rows("ARCHIVE", "DOCARCH")[0]
    changed["STORAGE_CHARGE"] = Decimal(str(changed["STORAGE_CHARGE"])) + Decimal("1.00000000")
    changed_key = str(changed["ARCH_KEY"])
    ctx2 = make_ctx(tmp_path, src2, target, manifest=manifest_after, run_id="run-2")
    prepare_run(ctx2)
    for stage in (extract, load, validate):
        stage.run(ctx2)
    rules = _rules(target, ctx2.run_id, ctx2.namespace)
    assert rules.get("ARCHIVE_CONFLICT") == {changed_key}
    assert changed_key not in target.purge_safe_keys(ctx2.run_id, ctx2.namespace, "DOCARCH")
    code2, _ = execute(ctx2, "purge")
    assert code2 == 0
    remaining = {str(r["ARCH_KEY"]) for r in src2.rows("ARCHIVE", "DOCARCH")}
    assert changed_key in remaining


def test_purge_resume_recovers_keys_committed_in_source_audit(tmp_path: Path, manifest_after: Path) -> None:
    """Db2 committed the audit+delete but the target status update was lost: resume must not re-delete."""
    seed = seed_source(generated=6, children_per_parent=1, plant=False)
    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate):
        stage.run(ctx)
    target = ctx.target
    assert isinstance(target, FakeTarget)
    ts = ctx.table("FILEAUD")
    keys = target.purge_safe_keys(ctx.run_id, ctx.namespace, "FILEAUD")
    first = keys[:2]
    target.insert_purge_audit(ctx.run_id, ctx.namespace, "FILEAUD", first, 1)
    seed.source.purge_batch("ARCHIVE", "FILEAUD", ts.key_column, first, ctx.run_id, ctx.namespace, 1)
    # simulate: set_purge_audit_status(PURGED) never happened -> rows stay INTENDED on the target
    assert not target.purged_keys(ctx.run_id, ctx.namespace, "FILEAUD")
    out = purge.run(ctx)
    assert out["FILEAUD"]["purged"] == len(keys)
    assert target.purged_keys(ctx.run_id, ctx.namespace, "FILEAUD") == set(keys)
    assert sum(1 for a in seed.source.purge_audit if a[1] == "FILEAUD") == len(keys)


def test_reconcile_cleanup_failure_leaves_run_running(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    class FlakyTarget(FakeTarget):
        def delete_staging_run(self, run_id, namespace) -> None:
            raise RuntimeError("staging cleanup timed out")

    target = FlakyTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    for stage in (extract, load, validate, purge):
        stage.run(ctx)
    with pytest.raises(RuntimeError):
        reconcile.run(ctx)
    assert target.runs[(ctx.run_id, ctx.namespace)].status == "RUNNING"
    # a retry (once cleanup works) closes the run
    target.__class__ = FakeTarget
    ctx2 = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    code, _ = execute(ctx2, "reconcile")
    assert code == 0
    assert target.runs[(ctx.run_id, ctx.namespace)].status == "CLOSED"


def test_empty_selection_loads_zero_rows(tmp_path: Path, manifest_after: Path) -> None:
    src = FakeSource()
    from .conftest import POLICIES, policy_row

    src.add_rows("ARCHIVE", "RETNPLCY", [policy_row(*p) for p in POLICIES])
    ctx = make_ctx(tmp_path, src, manifest=manifest_after)
    prepare_run(ctx)
    extract.run(ctx)
    out = load.run(ctx)
    assert out["DOCARCH"] == {"loaded": 0, "rejected": 0}
    assert out["FILEAUD"] == {"loaded": 0, "rejected": 0}


def test_init_apply_sql_is_confined_to_the_migration_tree(tmp_path: Path, seed: Seed, manifest_after: Path) -> None:
    from ldm.errors import ConfigError
    from ldm.stages import init

    ctx = make_ctx(tmp_path, seed.source, manifest=manifest_after)
    inside = ctx.loaded.repo_root / "extra.sql"
    inside.write_text("SELECT 1;\n", encoding="utf-8")
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_dir.mkdir(exist_ok=True)
    outside = outside_dir / "evil.sql"
    outside.write_text("SELECT 1;\n", encoding="utf-8")
    assert init.run(ctx, [inside])["_ddl"]["scripts"] == 1
    with pytest.raises(ConfigError, match="must live under"):
        init.run(ctx, [outside])
    with pytest.raises(ConfigError, match="not a .sql file"):
        init.run(ctx, [ctx.loaded.repo_root / "manifest.yaml"])


def test_validate_streams_in_small_batches_with_identical_outcome(tmp_path: Path, seed: Seed) -> None:
    """Validation is a bounded-memory stream: a 7-row batch size must classify exactly like one big batch."""
    from .conftest import make_manifest_tree

    def outcome(token: str, batch_rows: int) -> tuple[dict[str, set[str]], dict[str, int], dict[str, int]]:
        manifest = make_manifest_tree(tmp_path, token, purge=True)
        text = manifest.read_text(encoding="utf-8").replace(
            "validate_batch_rows: 20000", f"validate_batch_rows: {batch_rows}"
        )
        manifest.write_text(text, encoding="utf-8")
        target = FakeTarget()
        ctx, code, tables = _run_all(tmp_path, seed_source(), manifest, target, namespace=f"{token}-after")
        assert code == 0, tables
        return _rules(target, ctx.run_id, ctx.namespace), tables["DOCARCH"], tables["FILEAUD"]

    big = outcome("b01", 20000)
    small = outcome("b02", 7)
    assert small == big
    assert {"HASH_MISMATCH", "CLASS_TOTAL_MISMATCH", "ORPHAN_PARENT_NOT_SELECTED"} <= set(small[0])


def test_changed_manifest_cannot_resume_a_run(tmp_path: Path, manifest_after: Path) -> None:
    from ldm.errors import ConfigError

    seed = seed_source(generated=6, children_per_parent=1, plant=False)
    target = FakeTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    extract.run(ctx)
    text = manifest_after.read_text(encoding="utf-8")
    manifest_after.write_text(text.replace("extract_range_rows:", "extract_range_rows: 999 #"), encoding="utf-8")
    ctx2 = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    with pytest.raises(ConfigError, match="new --run-id"):
        prepare_run(ctx2)
    # the recorded digest is immutable: the original manifest still resumes
    manifest_after.write_text(text, encoding="utf-8")
    prepare_run(make_ctx(tmp_path, seed.source, target, manifest=manifest_after))
    # a rejected retry (here: overlay flipped to a dry run) must not touch the original run row at all
    overlay = manifest_after.parent / "manifests" / "t01-after.yaml"
    overlay.write_text(overlay.read_text(encoding="utf-8").replace("purge: true", "purge: false"), encoding="utf-8")
    run = target.runs[(ctx.run_id, ctx.namespace)]
    code, _ = execute(make_ctx(tmp_path, seed.source, target, manifest=manifest_after), "load")
    assert code == 4
    assert run.status == "RUNNING" and run.purge_enabled is True


def test_revalidation_drops_stale_validate_rejects(tmp_path: Path, manifest_after: Path) -> None:
    seed = seed_source(generated=6, children_per_parent=1, plant=False)
    target = FakeTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    extract.run(ctx)
    load.run(ctx)
    run, ns = ctx.run_id, ctx.namespace
    # corrupt one staged DOCARCH value so the first validate flags a HASH_MISMATCH
    key = seed.source.select_keys("ARCHIVE", "DOCARCH", ctx.selection_for(ctx.table("DOCARCH").config))[0]
    victim = next(r for r in target._stg(run, ns, "DOCARCH") if r.source_key == key)
    original = victim.values["OWNER_NAME"]
    victim.values["OWNER_NAME"] = "tampered"
    validate.run(ctx)
    assert _rules(target, run, ns).get("HASH_MISMATCH") == {key}
    # operator fixes the staged value and re-runs validate with the same run id
    victim.values["OWNER_NAME"] = original
    validate.run(ctx)
    assert "HASH_MISMATCH" not in _rules(target, run, ns)
    assert all(v.status == "VALIDATED" for v in target.validation[(run, ns)] if v.source_key == key)


def test_interrupted_revalidation_keeps_reject_evidence(tmp_path: Path, manifest_after: Path) -> None:
    seed = seed_source(generated=6, children_per_parent=1, plant=False)
    target = FakeTarget()
    ctx = make_ctx(tmp_path, seed.source, target, manifest=manifest_after)
    prepare_run(ctx)
    extract.run(ctx)
    load.run(ctx)
    run, ns = ctx.run_id, ctx.namespace
    key = seed.source.select_keys("ARCHIVE", "DOCARCH", ctx.selection_for(ctx.table("DOCARCH").config))[0]
    victim = next(r for r in target._stg(run, ns, "DOCARCH") if r.source_key == key)
    victim.values["OWNER_NAME"] = "tampered"
    validate.run(ctx)
    before = [r for r in target.rejects[(run, ns)] if r.stage == "VALIDATE"]
    assert key in {r.source_key for r in before} and all(r.error_text for r in before)

    # the retry dies before it reaches any DOCARCH row: the earlier reject must survive intact
    def boom(*_args: object, **_kw: object) -> dict[str, bytes]:
        raise RuntimeError("target unreachable")

    target.target_hashes = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        validate.run(ctx)
    assert [r for r in target.rejects[(run, ns)] if r.stage == "VALIDATE"] == before

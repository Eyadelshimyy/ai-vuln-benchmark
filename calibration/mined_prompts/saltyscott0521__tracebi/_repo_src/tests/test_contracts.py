"""
M4 — transform contracts (architecture v2 §2.6).

The sink contract: declared, closed-vocabulary checks run as read-only SQL
against the tables a transform just sank, recorded beside the warehouse
with connector-load-path fingerprints. The claims under test:

- a violated contract RAISES and writes no certificate;
- a satisfied contract records every check with its observed value;
- the recorded fingerprint round-trips: it equals what a fresh connector
  load fingerprints, across the dtype families the connector ships
  (the flaw-4 gate — without this the satisfied/stale join is noise);
- the manifest join classifies satisfied / stale / no_contract, and stale
  NEVER reads green;
- contract status never colors a figure status — two claims, side by side;
- re-run checks say whether TODAY's sink still satisfies the declaration.

The locked language: "the sink satisfied its contract" — never "the
transform was verified". Nothing here checks the pandas above the sink.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel
from tracebi.connectors.duckdb_connector import DuckDBConnector
from tracebi.contracts import (
    ContractViolation,
    check_fingerprints,
    contract,
    contracts_path,
    read_contracts,
    rerun_checks,
    transform_contracts_block,
)
from tracebi.model.dataset import frame_fingerprint


# ── fixtures ───────────────────────────────────────────────────────────────

def _sink(tmp_path, fact=None, dim=None):
    """Write a small star into a tmp warehouse; return its path."""
    wh_path = str(tmp_path / "warehouse.duckdb")
    dim = dim if dim is not None else pd.DataFrame({
        "region_id": [1, 2, 3], "region": ["NE", "SE", "MW"],
    })
    fact = fact if fact is not None else pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "region_id": [1, 2, 1, 3],
        "revenue": [100.0, 250.0, 75.0, 300.0],
        "status": ["shipped", "open", "shipped", "shipped"],
    })
    wh = DuckDBConnector("wh", database=wh_path)
    wh.write(dim, "dim_region")
    wh.write(fact, "fact_orders")
    wh.disconnect()
    return wh_path


@pytest.fixture()
def warehouse(tmp_path):
    return _sink(tmp_path)


def _good_contract(wh_path, name="orders"):
    with contract(name, warehouse=wh_path) as c:
        c.rows("fact_orders", at_least=1, at_most=100)
        c.unique("fact_orders", ["order_id"])
        c.not_null("fact_orders", ["order_id", "revenue"])
        c.foreign_key("fact_orders", "region_id",
                      refers_to=("dim_region", "region_id"))
        c.values("fact_orders", "status", within=["shipped", "open"])


# ── the closed vocabulary ──────────────────────────────────────────────────

class TestFingerprintDrift:
    """`check_fingerprints` catches the failure mode `rerun_checks` misses: a
    table changed out-of-band since the certificate was written, which can
    still pass every declared check."""

    def test_unchanged_warehouse_matches_the_certificate(self, warehouse):
        _good_contract(warehouse)
        rows = check_fingerprints(warehouse)
        assert rows and all(r["matches"] for r in rows)
        assert {r["table"] for r in rows} == {"fact_orders", "dim_region"}

    def test_out_of_band_row_is_caught_though_checks_still_pass(self, warehouse):
        _good_contract(warehouse)
        # Insert a row straight into the warehouse, bypassing the transform —
        # it still satisfies every declared check (unique id, valid status,
        # valid FK, row count in bounds), so rerun_checks stays green...
        import duckdb
        con = duckdb.connect(warehouse)
        con.execute("INSERT INTO fact_orders VALUES (5, 2, 999.0, 'shipped')")
        con.close()
        assert all(r["passed_now"] for r in rerun_checks(warehouse))
        # ...but the fingerprint no longer matches: drift is detected.
        drifted = [r for r in check_fingerprints(warehouse) if not r["matches"]]
        assert [r["table"] for r in drifted] == ["fact_orders"]
        assert drifted[0]["current"] != drifted[0]["recorded"]

    def test_a_dropped_table_reads_as_missing(self, warehouse):
        _good_contract(warehouse)
        import duckdb
        con = duckdb.connect(warehouse)
        con.execute("DROP TABLE fact_orders")
        con.close()
        rows = {r["table"]: r for r in check_fingerprints(warehouse)}
        assert rows["fact_orders"]["current"] is None
        assert rows["fact_orders"]["matches"] is False

class TestChecks:
    def test_a_satisfied_contract_passes_and_writes_the_record(self, warehouse):
        _good_contract(warehouse)
        data = read_contracts(warehouse)
        rec = data["transforms"]["orders"]
        assert all(c["passed"] for c in rec["checks"])
        assert len(rec["checks"]) == 5
        by_kind = {c["check"]: c for c in rec["checks"]}
        assert by_kind["rows"]["observed"] == 4
        assert by_kind["unique"]["observed"] == 0
        assert by_kind["foreign_key"]["observed"] == 0

    def test_rows_bounds_violation(self, warehouse):
        with pytest.raises(ContractViolation, match="rows"):
            with contract("orders", warehouse=warehouse) as c:
                c.rows("fact_orders", at_least=1000)

    def test_unique_violation(self, tmp_path):
        wh = _sink(tmp_path, fact=pd.DataFrame({
            "order_id": [1, 1], "region_id": [1, 1],
            "revenue": [1.0, 2.0], "status": ["open", "open"],
        }))
        with pytest.raises(ContractViolation, match="unique"):
            with contract("orders", warehouse=wh) as c:
                c.unique("fact_orders", ["order_id"])

    def test_not_null_violation(self, tmp_path):
        wh = _sink(tmp_path, fact=pd.DataFrame({
            "order_id": [1, 2], "region_id": [1, None],
            "revenue": [1.0, 2.0], "status": ["open", "open"],
        }))
        with pytest.raises(ContractViolation, match="not_null"):
            with contract("orders", warehouse=wh) as c:
                c.not_null("fact_orders", ["region_id"])

    def test_foreign_key_orphan_violation(self, tmp_path):
        wh = _sink(tmp_path, fact=pd.DataFrame({
            "order_id": [1, 2], "region_id": [1, 99],   # 99 is an orphan
            "revenue": [1.0, 2.0], "status": ["open", "open"],
        }))
        with pytest.raises(ContractViolation, match="foreign_key"):
            with contract("orders", warehouse=wh) as c:
                c.foreign_key("fact_orders", "region_id",
                              refers_to=("dim_region", "region_id"))

    def test_values_outside_the_closed_set(self, warehouse):
        with pytest.raises(ContractViolation, match="values"):
            with contract("orders", warehouse=warehouse) as c:
                c.values("fact_orders", "status", within=["shipped"])

    def test_reconcile_within_and_outside_tolerance(self, tmp_path):
        wh_path = str(tmp_path / "warehouse.duckdb")
        wh = DuckDBConnector("wh", database=wh_path)
        wh.write(pd.DataFrame({"k": [1, 2], "v": [10.0, 20.0]}), "a")
        wh.write(pd.DataFrame({"k": [1, 2], "v": [10.005, 20.0]}), "b")
        wh.disconnect()
        with contract("rec", warehouse=wh_path) as c:      # within tolerance
            c.reconcile("a", "v", against=("b", "v"), by="k", tolerance=0.01)
        with pytest.raises(ContractViolation, match="reconcile"):
            with contract("rec", warehouse=wh_path) as c:
                c.reconcile("a", "v", against=("b", "v"), by="k",
                            tolerance=0.0001)

    def test_reconcile_catches_a_key_missing_entirely(self, tmp_path):
        wh_path = str(tmp_path / "warehouse.duckdb")
        wh = DuckDBConnector("wh", database=wh_path)
        wh.write(pd.DataFrame({"k": [1, 2], "v": [10.0, 20.0]}), "a")
        wh.write(pd.DataFrame({"k": [1], "v": [10.0]}), "b")
        wh.disconnect()
        with pytest.raises(ContractViolation, match="reconcile"):
            with contract("rec", warehouse=wh_path) as c:
                c.reconcile("a", "v", against=("b", "v"), by="k",
                            tolerance=1000.0)


class TestViolationSemantics:
    def test_all_failures_are_reported_not_just_the_first(self, warehouse):
        with pytest.raises(ContractViolation) as err:
            with contract("orders", warehouse=warehouse) as c:
                c.rows("fact_orders", at_least=1000)
                c.values("fact_orders", "status", within=["shipped"])
        msg = str(err.value)
        assert "rows" in msg and "values" in msg
        assert "2 of 2" in msg

    def test_a_violated_contract_writes_no_certificate(self, warehouse):
        with pytest.raises(ContractViolation):
            with contract("orders", warehouse=warehouse) as c:
                c.rows("fact_orders", at_least=1000)
        assert read_contracts(warehouse) is None

    def test_language_is_sink_satisfied_never_transform_verified(self, warehouse):
        with pytest.raises(ContractViolation) as err:
            with contract("orders", warehouse=warehouse) as c:
                c.rows("fact_orders", at_least=1000)
        assert "sink did not satisfy" in str(err.value)
        assert "verified" not in str(err.value).lower()

    def test_an_exception_inside_the_block_propagates_unswallowed(self, warehouse):
        with pytest.raises(ZeroDivisionError):
            with contract("orders", warehouse=warehouse):
                1 / 0
        assert read_contracts(warehouse) is None

    def test_an_empty_block_writes_nothing(self, warehouse):
        with contract("orders", warehouse=warehouse):
            pass
        assert read_contracts(warehouse) is None


class TestRecord:
    def test_fingerprints_cover_every_touched_table_including_fk_targets(
            self, warehouse):
        with contract("orders", warehouse=warehouse) as c:
            c.foreign_key("fact_orders", "region_id",
                          refers_to=("dim_region", "region_id"))
        rec = read_contracts(warehouse)["transforms"]["orders"]
        assert set(rec["tables"]) == {"fact_orders", "dim_region"}

    def test_two_transforms_merge_into_one_certificate(self, warehouse):
        _good_contract(warehouse, name="orders")
        with contract("regions", warehouse=warehouse) as c:
            c.unique("dim_region", ["region_id"])
        data = read_contracts(warehouse)
        assert set(data["transforms"]) == {"orders", "regions"}

    def test_record_is_plain_json(self, warehouse):
        _good_contract(warehouse)
        with open(contracts_path(warehouse), encoding="utf-8") as f:
            data = json.load(f)                # round-trips as plain JSON
        assert data["schema_version"] == 1


# ── stated methodology: note= travels the certificate ──────────────────────

TRANSFORM_NOTE = "unkeyed rows dropped and counted; regions normalised"
CHECK_NOTE = "an empty fact means the raw pull failed, not an empty book"


def _noted_contract(wh_path, name="orders"):
    with contract(name, warehouse=wh_path, note=TRANSFORM_NOTE) as c:
        c.rows("fact_orders", at_least=1, note=CHECK_NOTE)
        c.unique("fact_orders", ["order_id"])
        c.foreign_key("fact_orders", "region_id",
                      refers_to=("dim_region", "region_id"))


class TestStatedMethodology:
    """note= is the author's STATED methodology — prose recorded verbatim,
    never a verified claim, never an input to any check."""

    def test_notes_are_recorded_in_the_certificate(self, warehouse):
        _noted_contract(warehouse)
        rec = read_contracts(warehouse)["transforms"]["orders"]
        assert rec["note"] == TRANSFORM_NOTE
        by_kind = {c["check"]: c for c in rec["checks"]}
        assert by_kind["rows"]["note"] == CHECK_NOTE
        # Beside params, never inside them — a note must not become a
        # replayable argument.
        assert "note" not in by_kind["rows"]["params"]
        # Omitted when unset, like other optional fields.
        assert "note" not in by_kind["unique"]

    def test_an_unnoted_contract_record_carries_no_note(self, warehouse):
        _good_contract(warehouse)
        rec = read_contracts(warehouse)["transforms"]["orders"]
        assert "note" not in rec
        assert all("note" not in c for c in rec["checks"])

    def test_notes_survive_rerun_untouched(self, warehouse):
        _noted_contract(warehouse)
        rows = rerun_checks(warehouse)
        assert all(r["passed_now"] for r in rows)
        by_kind = {r["check"]: r for r in rows}
        assert by_kind["rows"]["note"] == CHECK_NOTE
        assert "note" not in by_kind["unique"]

    def test_a_crashed_rerun_keeps_the_recorded_note_beside_the_error(
            self, warehouse):
        _noted_contract(warehouse)
        import duckdb
        con = duckdb.connect(warehouse)
        con.execute('DROP TABLE "fact_orders"')
        con.close()
        by_kind = {r["check"]: r for r in rerun_checks(warehouse)}
        assert by_kind["rows"]["passed_now"] is False
        assert by_kind["rows"]["error"]                  # the diagnostic
        assert by_kind["rows"]["note"] == CHECK_NOTE     # untouched

    def test_transform_note_joins_the_manifest_block(self, warehouse, tmp_path):
        _noted_contract(warehouse)
        manifest = _render(tmp_path, _model_over(warehouse))
        tc = manifest["transform_contracts"]
        assert tc["fact_orders"]["status"] == "satisfied"
        assert tc["fact_orders"]["note"] == TRANSFORM_NOTE
        # The fk-target table is covered by the same record and carries the
        # same transform-level statement.
        assert tc["dim_region"]["note"] == TRANSFORM_NOTE

    def test_an_unnoted_record_joins_without_a_note_key(
            self, warehouse, tmp_path):
        _good_contract(warehouse)
        manifest = _render(tmp_path, _model_over(warehouse))
        assert "note" not in manifest["transform_contracts"]["fact_orders"]

    def test_a_stale_entry_still_carries_the_stated_note(
            self, warehouse, tmp_path):
        """Stated methodology is prose, not status: it rides on stale
        entries exactly as on satisfied ones, and colors neither."""
        _noted_contract(warehouse)
        model = _model_over(warehouse)
        conn = model.connectors()[0]
        extra = conn.load("fact_orders")
        extra.loc[len(extra)] = [5, 2, 999.0, "open"]
        conn.write(extra, "fact_orders")
        manifest = _render(tmp_path, model)
        tc = manifest["transform_contracts"]
        assert tc["fact_orders"]["status"] == "stale"
        assert tc["fact_orders"]["note"] == TRANSFORM_NOTE


# ── the flaw-4 gate: connector-load-path fingerprint round-trip ────────────

class TestFingerprintRoundTrip:
    def test_recorded_fingerprint_equals_a_fresh_connector_load(self, tmp_path):
        """write → contract fingerprint == fingerprint(load()) — twice.

        The frame deliberately spans the connector's dtype families: ints,
        floats with NaN, strings with None, dates, booleans, and a nullable
        Int64 — any normalization the write/read round trip performs must
        land inside BOTH sides of the satisfied/stale comparison.
        """
        df = pd.DataFrame({
            "k": [1, 2, 3],
            "amount": [1.5, float("nan"), 2.25],
            "label": ["a", None, "c"],
            "day": pd.to_datetime(["2026-01-01", "2026-01-02",
                                   "2026-01-03"]).date,
            "flag": [True, False, True],
            "spread_bps": pd.array([597, None, 484], dtype="Int64"),
        })
        wh_path = str(tmp_path / "warehouse.duckdb")
        wh = DuckDBConnector("wh", database=wh_path)
        wh.write(df, "t")
        wh.disconnect()

        with contract("t", warehouse=wh_path) as c:
            c.rows("t", exactly=3)
        recorded = read_contracts(wh_path)["transforms"]["t"]["tables"]["t"]

        fresh = DuckDBConnector("fresh", database=wh_path)
        assert frame_fingerprint(fresh.load("t")) == recorded
        assert frame_fingerprint(fresh.load("t")) == recorded  # stable
        fresh.disconnect()


# ── the manifest join ──────────────────────────────────────────────────────

def _model_over(wh_path):
    m = DataModel("wh_model")
    m.add_connector(DuckDBConnector("wh", database=wh_path))
    m.add_table("fact_orders", connector="wh", source="fact_orders")
    m.add_table("dim_region", connector="wh", source="dim_region")
    m.add_relationship("o2r", left_table="fact_orders",
                       right_table="dim_region", left_key="region_id")
    m.add_dimension("dim_r", table_name="dim_region",
                    key_col="region_id", attributes=["region"])
    m.add_fact("f", table_name="fact_orders", measures=["revenue"],
               foreign_keys={"dim_r": "region_id"})
    m.add_measure("total", column="revenue", agg="sum")
    m.connect()
    return m


def _package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "pkg",
        "data": {
            "kpi": {"model": "wh_model",
                    "query": {"fact": "f", "measures": ["total"]}},
            "by_region": {"model": "wh_model",
                          "query": {"fact": "f",
                                    "measures": {"revenue": "sum"},
                                    "dimensions": ["dim_r.region"]}},
        },
    }))
    (pkg / "template.html").write_text(
        "<html><head><title>x</title></head><body>"
        '<div data-tb-figure="value" data-tb-binding="kpi" '
        'data-tb-cell="total" id="fig-kpi"></div>'
        '<table data-tb-figure="table" data-tb-binding="by_region" '
        'id="fig-tbl"></table>'
        "</body></html>"
    )
    return pkg


def _render(tmp_path, model):
    from tracebi.reports.template_package import TemplatePackage
    pkg = TemplatePackage(str(_package(tmp_path)))
    out = tmp_path / "out.html"
    manifest = pkg.render({"wh_model": model}, str(out)).to_dict()
    return manifest


class TestManifestJoin:
    def test_satisfied_when_the_certificate_matches_the_tables(
            self, warehouse, tmp_path):
        _good_contract(warehouse)
        manifest = _render(tmp_path, _model_over(warehouse))
        tc = manifest["transform_contracts"]
        assert tc["fact_orders"]["status"] == "satisfied"
        assert tc["dim_region"]["status"] == "satisfied"   # fk target covered
        assert tc["fact_orders"]["transform"] == "orders"
        assert tc["fact_orders"]["checks"] == 5

    def test_stale_after_a_resink_and_stale_never_reads_green(
            self, warehouse, tmp_path):
        _good_contract(warehouse)
        model = _model_over(warehouse)
        # Re-sink the fact AFTER its contract was checked. The model's
        # read-only handle must be released first: one file, one process,
        # one configuration at a time.
        conn = model.connectors()[0]
        extra = conn.load("fact_orders")
        extra.loc[len(extra)] = [5, 2, 999.0, "open"]
        conn.write(extra, "fact_orders")
        manifest = _render(tmp_path, model)
        tc = manifest["transform_contracts"]
        assert tc["fact_orders"]["status"] == "stale"
        assert tc["dim_region"]["status"] == "satisfied"   # untouched table

    def test_no_contract_when_nothing_certifies_the_input(
            self, warehouse, tmp_path):
        manifest = _render(tmp_path, _model_over(warehouse))
        tc = manifest["transform_contracts"]
        assert tc["fact_orders"] == {"status": "no_contract"}
        assert tc["dim_region"] == {"status": "no_contract"}

    def test_contract_status_never_colors_a_figure_status(
            self, warehouse, tmp_path):
        """Two claims, side by side, never blended: with a STALE contract
        the figures still verify green — the report honestly reproduces
        from the warehouse as it now is; the contract block separately says
        the warehouse outgrew its certificate."""
        from tracebi.verify import REPRODUCES, verify_manifest
        _good_contract(warehouse)
        model = _model_over(warehouse)
        conn = model.connectors()[0]
        extra = conn.load("fact_orders")
        extra.loc[len(extra)] = [5, 2, 999.0, "open"]
        conn.write(extra, "fact_orders")
        manifest = _render(tmp_path, model)
        assert manifest["transform_contracts"]["fact_orders"]["status"] == "stale"
        result = verify_manifest(manifest, {"wh_model": model}, strict=True)
        assert result["exit_code"] == 0
        assert all(f["status"] == REPRODUCES for f in result["figures"])

    def test_memory_connectors_join_as_no_contract(self, tmp_path):
        from tracebi import MemoryConnector
        df = pd.DataFrame({"region": ["NE"], "revenue": [1.0]})
        m = DataModel("wh_model")
        m.add_connector(MemoryConnector("mem", tables={"t": df}))
        m.add_table("t", connector="mem", source="t")
        m.add_dimension("dim_r", table_name="t", key_col="region",
                        attributes=["region"])
        m.add_fact("f", table_name="t", measures=["revenue"],
                   foreign_keys={})
        m.add_measure("total", column="revenue", agg="sum")
        m.connect()
        manifest = _render(tmp_path, m)
        assert manifest["transform_contracts"]["t"] == {"status": "no_contract"}


# ── re-running the declaration ─────────────────────────────────────────────

class TestRerunChecks:
    def test_a_still_satisfied_sink_passes_now(self, warehouse):
        _good_contract(warehouse)
        rows = rerun_checks(warehouse)
        assert len(rows) == 5
        assert all(r["passed_now"] for r in rows)

    def test_a_mutated_sink_fails_now(self, warehouse):
        _good_contract(warehouse)
        wh = DuckDBConnector("wh", database=warehouse)
        broken = wh.load("fact_orders")
        broken["status"] = "mystery"          # outside the declared set
        wh.write(broken, "fact_orders")
        wh.disconnect()
        by_kind = {r["check"]: r for r in rerun_checks(warehouse)}
        assert by_kind["values"]["passed_now"] is False
        assert by_kind["values"]["observed_now"] == 4
        assert by_kind["rows"]["passed_now"] is True

    def test_a_dropped_table_is_a_result_not_a_crash(self, warehouse):
        _good_contract(warehouse)
        wh = DuckDBConnector("wh", database=warehouse)
        wh.connection  # ensure file exists check passes
        wh.disconnect()
        import duckdb
        con = duckdb.connect(warehouse)
        con.execute('DROP TABLE "fact_orders"')
        con.close()
        rows = rerun_checks(warehouse)
        fact_rows = [r for r in rows if r["table"] == "fact_orders"]
        assert fact_rows
        assert all(r["passed_now"] is False for r in fact_rows)
        # Diagnostics ride in "error" — "note" is reserved for the recorded
        # stated methodology, which a re-run passes through untouched.
        assert all(r.get("error") for r in fact_rows)
        assert all("note" not in r for r in fact_rows)

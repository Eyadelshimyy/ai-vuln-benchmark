"""
The verify loop — input fingerprints at load, manifest schema_version, and
``tracebi verify`` / the ``verify_manifest`` gateway tool.

The trust thesis rests on receipts someone can check. These tests close the
loop end to end: a rendered manifest re-verifies (REPRODUCES), a mutated
source table is diagnosed (SOURCE DRIFT via input fingerprints), a tampered
result fingerprint with unmoved inputs is the alarming case (UNEXPLAINED),
and python-authored ad hoc data is honestly UNVERIFIABLE — with the CLI
exit codes distinguishing the three outcomes.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector, model_registry, verify
from tracebi.cli import main as cli_main
from tracebi.mcp_server import gateway_render_spec, gateway_verify_manifest
from tracebi.model.dataset import DataSet, LineageNode, frame_fingerprint
from tracebi.reports.report import Report, TableSection
from tracebi.verify import (
    MISMATCH_UNKNOWN,
    REPRODUCES,
    SOURCE_DRIFT,
    UNEXPLAINED,
    UNVERIFIABLE,
    load_models,
    verify_manifest,
)


ORDERS = pd.DataFrame({
    "order_id":    [1, 2, 3, 4, 5, 6],
    "customer_id": [1, 2, 1, 3, 2, 1],
    "revenue":     [100.0, 250.0, 75.0, 300.0, 125.0, 50.0],
    "status":      ["shipped", "open", "shipped", "shipped", "open", "shipped"],
})
CUSTOMERS = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "region":      ["NE", "SE", "MW"],
})


@pytest.fixture()
def vf_model():
    """A fresh star schema per test, registered under a name only this
    module uses. Fresh per test so one test's source mutation cannot leak
    into another."""
    connector = MemoryConnector("vf_mem", tables={
        "orders": ORDERS.copy(), "customers": CUSTOMERS.copy(),
    })
    model = DataModel("vf_demo")
    model.add_connector(connector)
    model.add_table("orders", connector="vf_mem", source="orders")
    model.add_table("customers", connector="vf_mem", source="customers")
    model.add_dimension(
        "dim_customer", table_name="customers",
        key_col="customer_id", attributes=["region"],
    )
    model.add_fact(
        "fact_orders", table_name="orders",
        measures=["revenue"], foreign_keys={"dim_customer": "customer_id"},
    )
    model.connect()
    model_registry.register(model)
    return model


@pytest.fixture()
def empty_models_dir(tmp_path):
    """A models dir with no files, so verify sees only registry models and
    never auto-discovers the repository's own models/ as a side effect."""
    d = tmp_path / "no_models"
    d.mkdir()
    return d


def _spec():
    return {
        "name": "VF Spec",
        "sections": [{
            "type": "table",
            "title": "Revenue by region",
            "data": {
                "model": "vf_demo",
                "query": {
                    "fact": "fact_orders",
                    "measures": {"revenue": "sum"},
                    "dimensions": ["dim_customer.region"],
                },
            },
        }],
    }


def _render(tmp_path):
    out = gateway_render_spec(_spec(), output_dir=str(tmp_path))
    assert out["ok"], out.get("errors")
    return tmp_path / "vf-spec.manifest.json"


def _connector_of(model):
    return model.connectors()[0]


# ── (a) Input fingerprints at query time ───────────────────────────────────

def test_query_lineage_records_input_fingerprints(vf_model):
    ds = vf_model.query(
        fact="fact_orders", measures={"revenue": "sum"},
        dimensions=["dim_customer.region"],
    )
    inputs = {
        n.metadata["input"]["table"]: n.metadata["input"]
        for n in ds.lineage if n.operation == "load"
    }
    assert set(inputs) == {"orders", "customers"}
    assert inputs["orders"]["fingerprint"] == frame_fingerprint(ORDERS)
    assert inputs["orders"]["rows"] == len(ORDERS)
    assert inputs["customers"]["fingerprint"] == frame_fingerprint(CUSTOMERS)


def test_rendered_manifest_carries_input_fingerprints(vf_model, tmp_path):
    manifest = json.loads(_render(tmp_path).read_text(encoding="utf-8"))
    section = next(
        s for s in manifest["sections"] if s.get("dataset_fingerprint")
    )
    recorded = [
        n["metadata"]["input"]
        for n in section["dataset_lineage"]
        if n.get("metadata", {}).get("input")
    ]
    assert {i["table"] for i in recorded} == {"orders", "customers"}
    assert all(i["fingerprint"] and i["rows"] for i in recorded)


# ── (b) Manifest schema version ────────────────────────────────────────────

# ── Render → verify round trip ─────────────────────────────────────────────

def test_mutated_source_is_diagnosed_as_source_drift(
    vf_model, tmp_path, empty_models_dir, capsys,
):
    manifest_path = _render(tmp_path)
    drifted = ORDERS.copy()
    drifted.loc[0, "revenue"] = 999.0
    _connector_of(vf_model).write(drifted, "orders")

    rc = cli_main([
        "verify", str(manifest_path), "--models-dir", str(empty_models_dir),
    ])
    assert rc == 2, "diagnosed drift is exit 2, not the alarming exit 1"
    err = capsys.readouterr().err
    assert "SOURCE DRIFT" in err
    assert "orders" in err, "the drifted table must be named"

    result = verify_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        load_models(empty_models_dir),
    )
    (section,) = result["sections"]
    assert section["status"] == SOURCE_DRIFT
    orders_input = next(
        i for i in section["inputs"] if i["table"] == "orders"
    )
    assert not orders_input["match"], "the input fingerprint diff is the diagnosis"
    customers_input = next(
        i for i in section["inputs"] if i["table"] == "customers"
    )
    assert customers_input["match"]


def test_tampered_fingerprint_is_unexplained(
    vf_model, tmp_path, empty_models_dir, capsys,
):
    manifest_path = _render(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    section = next(
        s for s in manifest["sections"] if s.get("dataset_fingerprint")
    )
    section["dataset_fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rc = cli_main([
        "verify", str(manifest_path), "--models-dir", str(empty_models_dir),
    ])
    assert rc == 1
    assert "UNEXPLAINED" in capsys.readouterr().err

    result = verify_manifest(manifest, load_models(empty_models_dir))
    (checked,) = result["sections"]
    assert checked["status"] == UNEXPLAINED
    assert all(i["match"] for i in checked["inputs"])


def test_engine_version_change_is_named_in_the_unexplained_detail(
    vf_model, tmp_path, empty_models_dir,
):
    """Data unmoved, result differs, and the recorded engine version differs
    from the one re-running now: verify names the version change as the likely
    cause instead of a blank 'something changed'."""
    manifest = json.loads(_render(tmp_path).read_text(encoding="utf-8"))
    section = next(
        s for s in manifest["sections"] if s.get("dataset_fingerprint")
    )
    # Force a mismatch with unmoved inputs (as the tamper test does)…
    section["dataset_fingerprint"] = "0" * 64
    # …and record an engine version different from the one re-running now.
    touched = False
    for node in section["dataset_lineage"]:
        md = dict(node.get("metadata") or {})
        if md.get("engine_version"):
            md["engine_version"] = "0.0.1-recorded"
            node["metadata"] = md
            touched = True
    assert touched, "no recorded engine_version to tamper — stamp regressed"

    result = verify_manifest(manifest, load_models(empty_models_dir))
    (checked,) = result["sections"]
    assert checked["status"] == UNEXPLAINED
    assert all(i["match"] for i in checked["inputs"])
    assert "0.0.1-recorded" in checked["detail"]
    assert "engine version" in checked["detail"].lower()


def test_python_authored_section_is_unverifiable(
    vf_model, tmp_path, empty_models_dir, capsys,
):
    """Ad hoc data has no recorded query — honest UNVERIFIABLE, exit 0."""
    adhoc = DataSet(
        df=pd.DataFrame({"metric": ["a"], "value": [1.0]}),
        name="adhoc",
        lineage=[LineageNode(operation="load", description="hand-built frame")],
    )
    report = Report("VF Python Report").add(
        TableSection(title="Ad hoc", dataset=adhoc)
    )
    manifest_path = tmp_path / "vf-python.manifest.json"
    report.build_manifest("html", "(in-memory)").save(str(manifest_path))

    rc = cli_main([
        "verify", str(manifest_path), "--models-dir", str(empty_models_dir),
    ])
    assert rc == 0, "unverifiable alone is not a failure"
    captured = capsys.readouterr()
    assert "UNVERIFIABLE" in captured.out
    assert "no recorded query" in captured.out
    assert captured.err == "", (
        "a run that exits 0 must write nothing to stderr — CI wrappers "
        "that treat stderr as failure would fail a receipt this command "
        "just called fine"
    )


def test_post_query_transform_is_unverifiable(vf_model, empty_models_dir):
    """A dataset transformed after the query cannot be reproduced by the
    query alone — claiming drift or unexplained would be a guess."""
    ds = vf_model.query(
        fact="fact_orders", measures={"revenue": "sum"},
        dimensions=["dim_customer.region"],
    ).sort("revenue", ascending=False)
    report = Report("VF Transformed").add(
        TableSection(title="Sorted", dataset=ds)
    )
    manifest = report.build_manifest("html", "(in-memory)").to_dict()

    result = verify_manifest(manifest, load_models(empty_models_dir))
    (section,) = result["sections"]
    assert section["status"] == UNVERIFIABLE
    assert "transformed after the recorded query" in section["detail"]
    assert result["exit_code"] == 0


def test_pre_input_fingerprint_manifest_is_honestly_unknown(
    vf_model, tmp_path, empty_models_dir,
):
    """A mismatch in a manifest rendered before input fingerprints existed
    cannot be split into drift vs unexplained — say so, and stay loud."""
    manifest = json.loads(_render(tmp_path).read_text(encoding="utf-8"))
    section = next(
        s for s in manifest["sections"] if s.get("dataset_fingerprint")
    )
    for node in section["dataset_lineage"]:
        node.get("metadata", {}).pop("input", None)
    section["dataset_fingerprint"] = "0" * 64

    result = verify_manifest(manifest, load_models(empty_models_dir))
    (checked,) = result["sections"]
    assert checked["status"] == MISMATCH_UNKNOWN
    assert "no input fingerprints recorded" in checked["detail"]
    assert result["exit_code"] == 1, "an undiagnosable mismatch must stay loud"


# ── CLI edges ──────────────────────────────────────────────────────────────

def test_cli_verify_missing_manifest_fails_loudly(tmp_path, capsys):
    rc = cli_main(["verify", str(tmp_path / "nope.manifest.json")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_cli_verify_rejects_invalid_json(tmp_path, capsys):
    bad = tmp_path / "bad.manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    rc = cli_main(["verify", str(bad)])
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err


# ── The gateway tool ───────────────────────────────────────────────────────

def test_gateway_verify_manifest_dict_and_path(vf_model, tmp_path):
    manifest_path = _render(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for arg in (manifest, str(manifest_path)):
        result = gateway_verify_manifest(arg)
        assert result["ok"], result
        assert result["report_name"] == "VF Spec"
        assert result["schema_version"] == 2   # spec renders via the artifact path
        assert result["summary"][REPRODUCES] == 1
        (section,) = result["sections"]
        assert section["status"] == REPRODUCES
        assert section["expected_fingerprint"] == section["actual_fingerprint"]
        assert section["model"] == "vf_demo"
        assert section["query_spec"]["fact"] == "fact_orders"
        json.dumps(result)  # the payload must survive the wire


def test_gateway_verify_detects_drift_structurally(vf_model, tmp_path):
    manifest_path = _render(tmp_path)
    drifted = ORDERS.copy()
    drifted.loc[0, "revenue"] = 999.0
    _connector_of(vf_model).write(drifted, "orders")

    result = gateway_verify_manifest(str(manifest_path))
    assert not result["ok"]
    assert result["exit_code"] == 2
    assert result["summary"][SOURCE_DRIFT] == 1
    (section,) = result["sections"]
    assert section["status"] == SOURCE_DRIFT


def test_gateway_verify_manifest_bad_inputs():
    missing = gateway_verify_manifest("/no/such/manifest.json")
    assert not missing["ok"]
    assert any("not found" in e for e in missing["errors"])

    not_a_manifest = gateway_verify_manifest(42)
    assert not not_a_manifest["ok"]
    assert not_a_manifest["errors"]


# ─────────────────────────────────────────────
# Review-pass fixes (adversarial findings)
# ─────────────────────────────────────────────

def test_model_repointing_is_model_changed_not_source_drift(
    vf_model, tmp_path, empty_models_dir
):
    """A table remapped to a different source is a governance event (exit 1),
    never a benign data refresh (exit 2) — the review's major finding."""
    from tracebi.verify import MODEL_CHANGED, load_models, verify_manifest

    import json as _json
    manifest = _json.loads(_render(tmp_path).read_text(encoding="utf-8"))

    # Repoint: same model name, same fingerprint-different data, but the
    # table now loads from a *different source table* on the connector.
    other = ORDERS.copy()
    other.loc[0, "revenue"] = 99999.0
    connector = MemoryConnector("vf_mem2", tables={
        "orders_v2": other, "customers": CUSTOMERS.copy(),
    })
    model = DataModel("vf_demo")
    model.add_connector(connector)
    model.add_table("orders", connector="vf_mem2", source="orders_v2")
    model.add_table("customers", connector="vf_mem2", source="customers")
    model.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
    model.add_fact("fact_orders", table_name="orders",
                   measures=["revenue"], foreign_keys={"dim_customer": "customer_id"})
    model.connect()
    model_registry.register(model)

    result = verify_manifest(manifest, load_models(empty_models_dir))
    statuses = [s["status"] for s in result["sections"]]
    assert MODEL_CHANGED in statuses
    assert result["exit_code"] == 1
    section = next(s for s in result["sections"] if s["status"] == MODEL_CHANGED)
    assert "orders" in section["detail"]


def test_malformed_lineage_nodes_do_not_crash_verification(vf_model, empty_models_dir):
    """Corrupt receipts are exactly what verify gets pointed at."""
    from tracebi.verify import UNVERIFIABLE, load_models, verify_manifest

    manifest = {
        "report_name": "corrupt", "schema_version": 1,
        "sections": [{
            "section_type": "table", "title": "t",
            "dataset_fingerprint": "x",
            "dataset_lineage": ["bogus", 42, {"metadata": "not-a-dict"}],
        }],
    }
    result = verify_manifest(manifest, load_models(empty_models_dir))
    assert result["sections"][0]["status"] == UNVERIFIABLE


def test_gateway_verify_survives_corrupt_manifest():
    from tracebi.mcp_server import gateway_verify_manifest

    out = gateway_verify_manifest({"sections": "not-a-list"})
    assert isinstance(out, dict)
    assert "ok" in out


def test_newer_schema_version_refuses_to_guess(vf_model, empty_models_dir):
    from tracebi.verify import load_models, verify_manifest

    result = verify_manifest(
        {"report_name": "future", "schema_version": 99, "sections": []},
        load_models(empty_models_dir),
    )
    assert not result["ok"]
    assert result["exit_code"] == 1
    assert "schema_version 99" in result["error"]


# ─────────────────────────────────────────────
# "Nothing was verified" is not "everything reproduced"
# ─────────────────────────────────────────────

def test_empty_receipt_is_never_an_ok_receipt(
    vf_model, tmp_path, empty_models_dir, capsys,
):
    """A manifest with no data-bearing section proves nothing. Verifying it
    checked zero numbers, so no consumer — library, CLI or gateway — may
    answer the way it answers a receipt that reproduced."""
    manifest = {
        "report_name": "Empty Receipt", "schema_version": 1,
        "sections": [{"section_type": "text", "title": "Just prose"}],
    }
    result = verify_manifest(manifest, load_models(empty_models_dir))
    assert result["sections"] == []
    assert not result["ok"], "zero sections checked must not read as a pass"
    assert result["exit_code"] == 1
    assert result.get("verdict") == verify.NOTHING_TO_VERIFY

    manifest_path = tmp_path / "empty.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rc = cli_main([
        "verify", str(manifest_path), "--models-dir", str(empty_models_dir),
    ])
    assert rc == 1
    assert "NOTHING VERIFIED" in capsys.readouterr().err

    assert not gateway_verify_manifest(manifest)["ok"]


def test_all_unverifiable_is_not_the_same_answer_as_all_reproduces(
    vf_model, tmp_path, empty_models_dir,
):
    """Hand-transformed sections stay exit 0 — a documented, legitimate
    state — but the receipt-level answer must still say that nothing was
    checked, or a reader cannot tell the two apart."""
    reproduced = verify_manifest(
        json.loads(_render(tmp_path).read_text(encoding="utf-8")),
        load_models(empty_models_dir),
    )
    ds = vf_model.query(
        fact="fact_orders", measures={"revenue": "sum"},
        dimensions=["dim_customer.region"],
    ).sort("revenue", ascending=False)
    unchecked = verify_manifest(
        Report("VF All Unverifiable")
        .add(TableSection(title="Sorted", dataset=ds))
        .build_manifest("html", "(in-memory)").to_dict(),
        load_models(empty_models_dir),
    )

    assert reproduced.get("verdict") == REPRODUCES
    assert unchecked.get("verdict") == UNVERIFIABLE
    assert unchecked.get("verdict") != reproduced.get("verdict"), (
        "a receipt where nothing was checked must not give the same answer "
        "as one that reproduced"
    )
    assert unchecked["summary"][REPRODUCES] == 0
    assert "NOTHING VERIFIED" in unchecked["verdict_detail"]
    assert unchecked["exit_code"] == 0, "unverifiable alone stays a non-failure"


def test_exit_code_ladder_is_pinned(vf_model, tmp_path, empty_models_dir):
    """The ladder is a contract other people's CI depends on. Behaviour
    first — a receipt that checked nothing must not share a rung with one
    that reproduced — then the whole map, so it cannot shift unnoticed."""
    reproduced = verify_manifest(
        json.loads(_render(tmp_path).read_text(encoding="utf-8")),
        load_models(empty_models_dir),
    )
    nothing = verify_manifest(
        {"report_name": "Empty", "schema_version": 1, "sections": []},
        load_models(empty_models_dir),
    )
    for result, code in ((reproduced, 0), (nothing, 1)):
        assert result["exit_code"] == code
        assert result["ok"] is (code == 0), (
            "ok and the exit code must never disagree"
        )

    assert verify.VERDICT_EXIT_CODES == {
        REPRODUCES:                  0,
        UNVERIFIABLE:                0,
        SOURCE_DRIFT:                2,
        verify.NOT_REPRODUCED:       1,
        verify.NOTHING_TO_VERIFY:    1,
        verify.REFUSED_NEWER_SCHEMA: 1,
    }
    assert set(verify.VERDICT_LABELS) == set(verify.VERDICT_EXIT_CODES)


def test_refusing_a_newer_manifest_does_not_claim_nothing_needed_checking(
    vf_model, empty_models_dir,
):
    """Refusing to read a manifest is 'I could not check this', never
    'there was nothing here to check' — the receipt may be full of
    data-bearing sections, and this dict reaches an agent verbatim."""
    manifest = {
        "report_name": "Too New", "schema_version": 99,
        "sections": [{"section_type": "table", "title": "Revenue",
                      "dataset_fingerprint": "abc"}],
    }
    result = verify_manifest(manifest, load_models(empty_models_dir))
    assert not result["ok"]
    assert result["exit_code"] == 1
    assert result.get("verdict") != verify.NOTHING_TO_VERIFY, (
        "this manifest has a data-bearing section; reporting that there was "
        "nothing to check is a false machine-readable answer"
    )
    assert "no data-bearing section" not in result.get("verdict_detail", "")
    assert result["verdict"] == verify.REFUSED_NEWER_SCHEMA
    assert "schema_version 99" in result["error"]
    assert gateway_verify_manifest(manifest)["verdict"] == (
        verify.REFUSED_NEWER_SCHEMA
    ), "the gateway hands this dict to an agent unchanged"


# ─────────────────────────────────────────────
# Metric receipts (report architecture v2 §2.2 — the metric-receipt hole)
# ─────────────────────────────────────────────

def _metrics_spec():
    return {
        "name": "VF KPIs",
        "sections": [{
            "type": "metrics",
            "title": "Totals",
            "data": {
                "model": "vf_demo",
                "query": {
                    "fact": "fact_orders",
                    "measures": {"revenue": "sum"},
                },
            },
            "metrics": [
                {"label": "Revenue", "value": "revenue", "format": "currency0"},
            ],
        }],
    }


def test_metrics_with_data_is_checked_and_reproduces(vf_model, empty_models_dir):
    """A KPI strip resolved from a query used to discard the one-row frame at
    build time, so `verify` printed REPRODUCES for a page whose biggest
    numbers carried no receipt. The frame now stays on the section, the
    manifest records its fingerprint + lineage, and verify classifies it like
    any table section (report architecture v2 §2.2)."""
    from tracebi.spec import ReportSpec

    spec = ReportSpec.from_dict(_metrics_spec())
    # M0 flip ledger: metrics-with-data now counts as data-bearing.
    assert spec.data_coverage() == {
        "total": 1, "with_data_ref": 1, "presentation_only": [],
    }

    report = spec.build({"vf_demo": vf_model})
    manifest = report.build_manifest("html", "(in-memory)").to_dict()
    (section,) = manifest["sections"]
    assert section["section_type"] == "metrics"
    assert section["dataset_fingerprint"]
    assert section["dataset_lineage"]

    result = verify_manifest(manifest, load_models(empty_models_dir))
    (checked,) = result["sections"]
    assert checked["status"] == REPRODUCES
    assert checked["model"] == "vf_demo"
    assert checked["query_spec"]["fact"] == "fact_orders"
    assert result["verdict"] == REPRODUCES
    assert result["summary"][REPRODUCES] == 1


def test_metrics_with_drifted_source_is_diagnosed(vf_model, empty_models_dir):
    """The metric receipt is a real receipt: when the source moves, the KPI
    section is diagnosed as SOURCE DRIFT, not waved through."""
    from tracebi.spec import ReportSpec

    manifest = (
        ReportSpec.from_dict(_metrics_spec())
        .build({"vf_demo": vf_model})
        .build_manifest("html", "(in-memory)")
        .to_dict()
    )
    drifted = ORDERS.copy()
    drifted.loc[0, "revenue"] = 999.0
    _connector_of(vf_model).write(drifted, "orders")

    result = verify_manifest(manifest, load_models(empty_models_dir))
    (section,) = result["sections"]
    assert section["status"] == SOURCE_DRIFT
    assert result["exit_code"] == 2


def test_static_metrics_section_stays_out_of_the_receipt(vf_model, empty_models_dir):
    """A metrics section with literal card values and no `data` query has
    nothing to verify — its manifest shape is unchanged and it is neither
    counted as data-bearing nor classified."""
    from tracebi.spec import ReportSpec

    spec = ReportSpec.from_dict({
        "name": "Static KPIs",
        "sections": [{"type": "metrics", "metrics": [
            {"label": "Target", "value": 500, "format": "currency0"},
        ]}],
    })
    assert spec.data_coverage() == {
        "total": 0, "with_data_ref": 0, "presentation_only": [],
    }
    manifest = spec.build({}).build_manifest("html", "(in-memory)").to_dict()
    (section,) = manifest["sections"]
    assert "dataset_fingerprint" not in section
    result = verify_manifest(manifest, load_models(empty_models_dir))
    assert result["sections"] == []


def test_reproduces_verdict_names_the_sections_it_could_not_check(
    vf_model, tmp_path, empty_models_dir,
):
    """One checked section among many unverifiable ones is exit 0 and
    honestly `reproduces` — nothing failed — but it must not read like a
    receipt where every section was checked."""
    manifest = json.loads(_render(tmp_path).read_text(encoding="utf-8"))
    fully_checked = verify_manifest(manifest, load_models(empty_models_dir))

    manifest["sections"].append({
        "section_type": "table", "title": "Ad hoc",
        "dataset_fingerprint": "x" * 64, "dataset_lineage": [],
    })
    mixed = verify_manifest(manifest, load_models(empty_models_dir))

    assert mixed["summary"][REPRODUCES] == 1
    assert mixed["summary"][UNVERIFIABLE] == 1
    assert mixed["exit_code"] == 0, "nothing failed, so this is not a failure"
    assert mixed.get("verdict") == REPRODUCES
    assert "1 unverifiable" in mixed.get("verdict_detail", ""), (
        "the verdict line is where a reader looks; it must disclose the "
        "section this receipt does not prove"
    )
    assert mixed["verdict_detail"] != fully_checked["verdict_detail"]

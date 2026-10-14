"""Desk lists what needs a person, and opens a receipt that reproduces."""

from __future__ import annotations

import json

import duckdb

from tracebi.desk import review


def test_review_lists_pins_drafts_failures_and_sinks(tmp_path):
    pin_dir = tmp_path / ".tracebi" / "workbench" / "demo"
    pin_dir.mkdir(parents=True)
    (pin_dir / "pins.json").write_text(json.dumps([
        {"id": "fig-1", "note": "look here", "at_seq": 3},
    ]), encoding="utf-8")
    discovery = tmp_path / ".tracebi" / "workbench" / "_discovery"
    discovery.mkdir(parents=True)
    (discovery / "pins.json").write_text(json.dumps([
        {"id": "table", "note": "warehouse", "at_seq": 1},
    ]), encoding="utf-8")

    draft = tmp_path / "reports" / "scratch"
    draft.mkdir(parents=True)
    (draft / "template.html").write_text(
        '<section data-tb-stage="exploration">probe</section>',
        encoding="utf-8",
    )
    clean = tmp_path / "reports" / "published"
    clean.mkdir(parents=True)
    (clean / "template.html").write_text("<p>done</p>", encoding="utf-8")

    output = tmp_path / "output"
    output.mkdir()
    (output / "waiting.html.manifest.json").write_text(json.dumps({
        "report_name": "waiting",
        "sections": [{"title": "note"}],
    }), encoding="utf-8")
    (output / "broken.html.manifest.json").write_text("{", encoding="utf-8")

    warehouse = tmp_path / "data" / "warehouse.duckdb"
    warehouse.parent.mkdir()
    con = duckdb.connect(str(warehouse))
    con.execute("CREATE TABLE certified (id INTEGER)")
    con.execute("INSERT INTO certified VALUES (1)")
    con.execute("CREATE TABLE loose (id INTEGER)")
    con.execute("INSERT INTO loose VALUES (1)")
    con.close()
    (tmp_path / "data" / "warehouse.contracts.json").write_text(json.dumps({
        "transforms": {
            "demo": {"tables": {"certified": "not-the-fingerprint"}, "checks": []},
        },
    }), encoding="utf-8")

    body = review(str(tmp_path), models={})
    notes = {pin["note"] for pin in body["pins"]}
    assert notes == {"look here", "warehouse"}
    assert [d["report"] for d in body["drafts"]] == ["scratch"]
    verdicts = {row["report"]: row["verdict"] for row in body["verdicts"]}
    assert verdicts["waiting"] == "nothing_to_verify"
    assert verdicts["broken"] == "error"
    statuses = {row["table"]: row["status"] for row in body["sinks"]}
    assert statuses["certified"] == "stale"
    assert statuses["loose"] == "no_contract"
    assert body["warehouse"] is True
    assert body["open"] is None


def test_a_reproducing_artifact_is_what_desk_opens(tmp_path):
    from tracebi import DataModel, MemoryConnector
    from tracebi.reports.template_package import TemplatePackage

    orders = __import__("pandas").DataFrame({
        "customer_id": [1], "revenue": [10], "order_id": [1],
    })
    customers = __import__("pandas").DataFrame({
        "customer_id": [1], "region": ["East"],
    })
    model = DataModel("desk_model")
    model.add_connector(MemoryConnector("desk_mem", tables={
        "orders": orders, "customers": customers,
    }))
    model.add_table("orders", connector="desk_mem", source="orders")
    model.add_table("customers", connector="desk_mem", source="customers")
    model.add_dimension(
        "dim_customer", table_name="customers", key_col="customer_id",
        attributes=["region"],
    )
    model.add_fact(
        "fact_orders", table_name="orders", measures=["revenue", "order_id"],
        foreign_keys={"dim_customer": "customer_id"},
    )
    model.add_measure("revenue", column="revenue", agg="sum")
    pkg = tmp_path / "reports" / "kept"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "kept",
        "data": {
            "totals": {
                "model": "desk_model",
                "query": {"fact": "fact_orders", "measures": ["revenue"]},
            },
        },
    }), encoding="utf-8")
    (pkg / "template.html").write_text(
        "<html><head><title>kept</title></head><body>"
        '<div data-tb-figure="value" data-tb-binding="totals" '
        'data-tb-cell="revenue" id="fig-total"></div>'
        "</body></html>",
        encoding="utf-8",
    )
    html = tmp_path / "output" / "kept.html"
    TemplatePackage(str(pkg)).render(
        {"desk_model": model}, str(html), save_manifest=True,
    )
    (tmp_path / "output" / "note.html.manifest.json").write_text(json.dumps({
        "report_name": "note",
        "sections": [{"title": "prose"}],
    }), encoding="utf-8")
    # A reproduces receipt with no HTML beside it is not a review item.
    kept_manifest = tmp_path / "output" / "kept.html.manifest.json"
    (tmp_path / "output" / "ghost.html.manifest.json").write_text(
        kept_manifest.read_text(encoding="utf-8"), encoding="utf-8",
    )

    body = review(str(tmp_path), models={"desk_model": model})
    assert body["open"]["report"] == "kept"
    assert body["open"]["verdict"] == "reproduces"
    assert [row["report"] for row in body["verdicts"]] == ["note"]
    # Builds lists only reports with an HTML file, each with its time and
    # verdict, so the desk can say when a report was built and whether its
    # receipt still reproduces.
    builds = {b["report"]: b for b in body["builds"]}
    assert set(builds) == {"kept"}
    assert builds["kept"]["verdict"] == "reproduces"
    assert builds["kept"]["built_at"].endswith("+00:00")


def test_desk_endpoint_reads_the_project(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.web.api.routers import desk as desk_router

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    (out / "note.html.manifest.json").write_text(json.dumps({
        "report_name": "note",
        "sections": [],
    }), encoding="utf-8")
    app = FastAPI()
    app.include_router(desk_router.router, prefix="/api")
    client = TestClient(app)
    res = client.get("/api/desk")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["verdicts"][0]["verdict"] == "nothing_to_verify"
    assert body["pins"] == []

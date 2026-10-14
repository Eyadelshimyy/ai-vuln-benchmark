"""PDF download: a print of the built package HTML.

The real print launches Chromium and skips when Playwright or its browser
is absent. The download-route test always runs: a missing Playwright is a
structured error, not an unhandled stack.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector


def _model() -> DataModel:
    m = DataModel("pdf_model")
    m.add_connector(MemoryConnector("mem", {
        "orders": pd.DataFrame({
            "order_id": [1, 2, 3],
            "customer_id": [1, 2, 1],
            "revenue": [100.0, 200.0, 300.0],
        }),
        "customers": pd.DataFrame({
            "customer_id": [1, 2],
            "region": ["West", "East"],
        }),
    }))
    m.add_table("orders", connector="mem", source="orders")
    m.add_table("customers", connector="mem", source="customers")
    m.add_dimension("dim_customer", table_name="customers",
                    key_col="customer_id", attributes=["region"])
    m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
               foreign_keys={"dim_customer": "customer_id"})
    m.add_measure("revenue", column="revenue", agg="sum")
    m.connect()
    return m


def _package(reports: Path, model: DataModel) -> Path:
    pkg = reports / "bars"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "bars",
        "data": {"by_region": {
            "model": model.name,
            "query": {"fact": "fact_orders", "measures": ["revenue"],
                      "dimensions": ["dim_customer.region"]},
        }},
        "figures": {"mix": {
            "kind": "chart", "binding": "by_region", "chart_type": "bar",
            "x": "dim_customer.region", "y": "revenue",
        }},
    }), encoding="utf-8")
    (pkg / "template.html").write_text(
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<title>{{ title }}</title></head><body>"
        "<main>{{ figure(\"mix\") }}</main></body></html>\n",
        encoding="utf-8",
    )
    return pkg


def _built_html(tmp_path: Path) -> Path:
    from tracebi.reports.template_package import TemplatePackage

    model = _model()
    pkg = _package(tmp_path / "reports", model)
    html = tmp_path / "bars.html"
    TemplatePackage(str(pkg)).render({model.name: model}, str(html))
    return html


def _chromium_or_skip() -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        pytest.skip(
            f"Chromium is not installed ({type(exc).__name__}: {exc})"
        )


def test_print_built_chart_package(tmp_path: Path) -> None:
    """A built package with a chart prints to a real PDF after the chart is drawn."""
    _chromium_or_skip()
    from playwright.sync_api import sync_playwright

    from tracebi.reports.pdf import print_pdf

    html = _built_html(tmp_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html.resolve().as_uri(), wait_until="load")
        page.wait_for_function(
            "() => document.documentElement.getAttribute('data-tb-ready') === '1'",
            timeout=30_000,
        )
        drawn = page.evaluate(
            """() => {
              const el = document.querySelector('[data-tb-figure="chart"]');
              return !!(el && el.querySelector('canvas, svg:not(.tb-chart-fallback)'));
            }"""
        )
        browser.close()
    assert drawn, "the chart figure was not drawn before printing"

    pdf = tmp_path / "bars.pdf"
    print_pdf(html, pdf)
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 5_000


def test_cli_format_pdf_writes_beside_the_html(tmp_path, monkeypatch) -> None:
    from tracebi import cli

    model = _model()
    _package(tmp_path / "reports", model)
    monkeypatch.setattr(cli, "_load_project_models", lambda: {model.name: model})

    def fake(html_path, pdf_path):
        Path(pdf_path).write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr("tracebi.reports.pdf.print_pdf", fake)
    html_out = tmp_path / "out" / "bars.html"
    rc = cli.main([
        "report", "build", "bars",
        "--reports-dir", str(tmp_path / "reports"),
        "--output", str(html_out),
        "--format", "pdf",
    ])
    assert rc == 0
    assert html_out.is_file()
    pdf = html_out.with_suffix(".pdf")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert (html_out.parent / (html_out.name + ".manifest.json")).is_file()


def test_pdf_download_names_the_install_when_playwright_is_missing(
        tmp_path, monkeypatch) -> None:
    """A missing Playwright is the router's structured detail, install hint included."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import tracebi.model_registry as model_registry
    from tracebi.web.api.registry import registry
    from tracebi.web.api.routers import reports as reports_router

    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)

    model = _model()
    name = "pdf_missing"
    pkg = _package(tmp_path / "pkg", model)
    model_registry.register(model)

    def factory():
        from tracebi.reports.template_package import TemplatePackage
        return TemplatePackage(str(pkg))

    factory._tracebi_package_dir = str(pkg)
    registry.add_report(name, factory)
    app = FastAPI()
    app.include_router(reports_router.router, prefix="/api")
    client = TestClient(app)
    try:
        response = client.get(f"/api/reports/{name}/download?format=pdf")
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["exception_type"] == "ImportError"
        assert "tracebi[pdf]" in detail["message"]
        assert "python -m playwright install chromium" in detail["message"]
        assert isinstance(detail["traceback"], str)
    finally:
        registry._report_factories.pop(name, None)
        model_registry._registry._models.pop(model.name, None)

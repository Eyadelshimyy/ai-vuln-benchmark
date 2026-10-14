"""
Agent gateway tests — the ``gateway_*`` functions in ``tracebi/mcp_server.py``.

The gateway's promise is the *stamp*: every query response carries the
resolved query, the lineage chain, and a fingerprint of the full result.
These tests hold it to that — a stamped response must re-verify, a preview
cap must not change the fingerprint, and JSON must survive the trip —
because an agent will quote these numbers to a person who never sees this
code.

The MCP registration layer itself is only exercised when the optional
``mcp`` package is installed (skipped otherwise); the operations are plain
functions precisely so the suite does not depend on it.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi import model_registry
from tracebi.mcp_server import (
    gateway_context,
    gateway_model_info,
    gateway_models,
    gateway_query,
    gateway_render_spec,
    gateway_validate_spec,
)


@pytest.fixture()
def gateway_model():
    """A small star schema registered under a name no other test uses."""
    orders = pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 5, 6],
        "customer_id": [1, 2, 1, 3, 2, 1],
        "revenue":     [100.0, 250.0, 75.0, 300.0, 125.0, 50.0],
        "status":      ["shipped", "open", "shipped", "shipped", "open", "shipped"],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "region":      ["NE", "SE", "MW"],
    })
    connector = MemoryConnector("gw_mem", tables={
        "orders": orders, "customers": customers,
    })
    model = DataModel("gw_demo")
    model.add_connector(connector)
    model.add_table("orders", connector="gw_mem", source="orders")
    model.add_table("customers", connector="gw_mem", source="customers")
    model.add_relationship(
        "orders_to_customers",
        left_table="orders", right_table="customers", left_key="customer_id",
    )
    model.add_dimension(
        "dim_customer", table_name="customers",
        key_col="customer_id", attributes=["region"],
    )
    model.add_fact(
        "fact_orders", table_name="orders",
        measures=["revenue"], foreign_keys={"dim_customer": "customer_id"},
    )
    model.add_measure(
        "total_revenue", column="revenue", agg="sum", format="currency",
    )
    model.connect()
    model_registry.register(model)
    return model


def _query(**overrides):
    kwargs = dict(
        model="gw_demo",
        fact="fact_orders",
        measures={"revenue": "sum"},
        dimensions=["dim_customer.region"],
    )
    kwargs.update(overrides)
    return gateway_query(**kwargs)


# ── The stamp ──────────────────────────────────────────────────────────────

def test_query_is_stamped(gateway_model):
    out = _query()
    assert out["fingerprint"], "a stamped response must carry a fingerprint"
    assert out["lineage"], "a stamped response must carry a lineage chain"
    assert out["query"]["fact"] == "fact_orders"
    assert out["row_count"] == 3          # NE, SE, MW
    assert not out["truncated"]


def test_having_is_applied_and_echoed_by_the_gateway(gateway_model):
    """having must reach the query over MCP — the primary agent surface —
    not be dropped. An impossibly high threshold yields no groups, and the
    stamped query echoes having so what the agent cites is what ran."""
    out = _query(having={"revenue": {"gte": 10 ** 12}})
    assert out["row_count"] == 0, "having must filter groups over the gateway"
    assert out["query"].get("having") == {"revenue": {"gte": 10 ** 12}}, (
        "the stamped query must echo having, not silently drop it"
    )


def test_stamp_reverifies(gateway_model):
    """The recorded query, re-run, reproduces the recorded fingerprint."""
    out = _query()
    q = out["query"]
    ds = gateway_model.query(
        fact=q["fact"], measures=q["measures"],
        dimensions=q["dimensions"], filters=q["filters"] or None,
        aggregate=q["aggregate"], allow_fanout=q["allow_fanout"],
    )
    assert ds.fingerprint() == out["fingerprint"]


def test_preview_cap_does_not_change_the_fingerprint(gateway_model):
    """rows is transport; the stamp covers the full result."""
    full = _query()
    capped = _query(preview_rows=1)
    assert capped["rows_returned"] == 1
    assert capped["truncated"]
    assert capped["row_count"] == full["row_count"]
    assert capped["fingerprint"] == full["fingerprint"]


def test_preview_rows_is_clamped_to_the_hard_cap(gateway_model):
    from tracebi.mcp_server import _ROW_HARD_CAP
    out = _query(preview_rows=10_000)
    assert out["rows_returned"] <= _ROW_HARD_CAP


def test_filters_travel_through(gateway_model):
    out = _query(filters={"status": "shipped"}, dimensions=[])
    assert out["row_count"] == 1
    total = out["rows"][0]["revenue"]
    assert total == pytest.approx(100.0 + 75.0 + 300.0 + 50.0)
    assert out["query"]["filters"] == {"status": "shipped"}


def test_brief_context_carries_a_report_json_example_that_builds(tmp_path):
    """A gateway agent sees one complete report.json, and that file builds."""
    from tracebi.mcp_server import authoring_guide, gateway_context
    from tracebi.reports.template_package import TemplatePackage

    pres = gateway_context(brief=True)["presentation"]
    block = pres["report_json"]
    example = block["example"]
    assert example["name"] and example["data"] and example["figures"]
    assert "libs" not in example  # the build adds D3 for the chart
    assert "optional" in block["libs"] and "automatically" in block["libs"]
    guide = authoring_guide()
    assert json.dumps(example, indent=2) in guide
    assert block["libs"] in guide

    pkg = tmp_path / "sales_by_region"
    pkg.mkdir()
    (pkg / "report.json").write_text(json.dumps(example), encoding="utf-8")
    placed = "\n".join(
        f'{{{{ figure("{name}") }}}}' for name in example["figures"])
    (pkg / "template.html").write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>t</title></head><body>\n" + placed + "\n</body></html>\n",
        encoding="utf-8",
    )
    orders = pd.DataFrame({
        "order_id": [1, 2], "customer_id": [1, 2], "revenue": [100.0, 50.0],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2], "region": ["NE", "SE"],
    })
    model = DataModel("sales").add_connector(MemoryConnector("m", tables={
        "orders": orders, "customers": customers,
    }))
    model.add_table("orders", connector="m", source="orders")
    model.add_table("customers", connector="m", source="customers")
    model.add_dimension("dim_customer", table_name="customers",
                        key_col="customer_id", attributes=["region"])
    model.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={"dim_customer": "customer_id"})
    model.connect()
    out = tmp_path / "sales_by_region.html"
    TemplatePackage(str(pkg)).render({"sales": model}, str(out))
    html = out.read_text(encoding="utf-8")
    assert 'data-tb-figure="value"' in html
    assert 'data-tb-figure="chart"' in html
    assert (tmp_path / "sales_by_region.html.manifest.json").is_file()


def test_order_by_tiebreak_is_noted_and_the_fingerprint_is_unchanged(gateway_model):
    """Tie-break keys stay in the receipt. The note says the gateway added them."""
    import anyio

    from tracebi.mcp_server import build_server

    out = _query(order_by=["-revenue"])
    cols = [o["column"] for o in out["query"]["order_by"]]
    assert cols[0] == "revenue"
    assert "dim_customer.region" in cols[1:]
    assert out["binding"]["query"]["order_by"] == out["query"]["order_by"]
    assert "tie-break" in out["order_by_note"]
    direct = gateway_model.query(
        fact="fact_orders", measures={"revenue": "sum"},
        dimensions=["dim_customer.region"], order_by=["-revenue"],
    )
    assert direct.fingerprint() == out["fingerprint"]
    plain = _query()
    assert "order_by" not in plain["query"]
    assert "order_by_note" not in plain

    pytest.importorskip("mcp")
    server = build_server()

    async def call():
        return await server.call_tool("query_model", {
            "model": "gw_demo", "fact": "fact_orders",
            "measures": {"revenue": "sum"},
            "dimensions": ["dim_customer.region"],
            "order_by": ["-revenue"],
        })

    payload = anyio.run(call).model_dump(by_alias=True, exclude_none=True)
    sc = payload["structuredContent"]
    assert sc["fingerprint"] == out["fingerprint"]
    assert "tie-break" in sc["order_by_note"]
    assert sc["binding"]["query"]["order_by"] == sc["query"]["order_by"]


def test_describe_model_names_a_share_base_and_a_ratio_pair(monkeypatch):
    """A share measure says which measure it is a share of."""
    m = DataModel("share_demo")
    m.add_measure("fair_value", column="fair_value", agg="sum")
    m.add_measure("cost", column="cost", agg="sum")
    m.add_measure("mark", ratio=("fair_value", "cost"))
    m.add_measure("fv_share", share="fair_value", format="percent")
    monkeypatch.setattr(
        "tracebi.mcp_server._load_models", lambda: {"share_demo": m})
    by_name = {item["name"]: item for item in gateway_model_info("share_demo")["measures"]}
    assert by_name["fv_share"]["kind"] == "share"
    assert by_name["fv_share"]["share"] == "fair_value"
    assert by_name["mark"]["ratio"] == ["fair_value", "cost"]


def test_query_returns_a_binding_stub(gateway_model):
    out = _query(filters={"status": "shipped"})
    binding = out["binding"]
    assert binding["model"] == "gw_demo"
    assert binding["query"]["fact"] == out["query"]["fact"]
    assert binding["query"]["measures"] == out["query"]["measures"]
    assert binding["query"]["filters"] == {"status": "shipped"}
    assert "rows" not in binding
    # Paste-ready: this object is the value of report.json data.<name>.
    assert set(binding) == {"model", "query"}


def test_include_lineage_false_drops_the_chain_but_keeps_the_stamp(gateway_model):
    lean = _query(include_lineage=False)
    assert lean["ok"] is True
    assert "lineage" not in lean               # the ~600-token chain is dropped
    assert lean["fingerprint"] and lean["query"]   # the citable stamp survives
    assert "lineage" in _query()               # default still carries it


def test_unknown_model_names_the_alternatives(gateway_model):
    # A structured, retryable error — the {ok, errors} envelope every other
    # tool uses — not a raised exception that ends the agent loop.
    out = gateway_query(model="nope", fact="f", measures={"x": "sum"})
    assert out["ok"] is False
    assert any("gw_demo" in e for e in out["errors"])
    assert "fingerprint" not in out          # no half-formed success payload


def test_query_errors_are_the_envelope_not_a_raise(gateway_model):
    # Every bad-input class an agent hits mid-loop returns {ok: False}, with a
    # message that names the fix — never a stack trace.
    bad_fact = _query(fact="nope")
    assert bad_fact["ok"] is False and "fact_orders" in bad_fact["errors"][0]

    bad_dim = _query(dimensions=["dim_ghost.x"])
    assert bad_dim["ok"] is False and "dim_ghost" in bad_dim["errors"][0]

    bad_filter = _query(filters={"ghostcol": "NE"})
    assert bad_filter["ok"] is False and "ghostcol" in bad_filter["errors"][0]


def test_bare_string_measures_gets_a_readable_error(gateway_model):
    # The classic 'dictionary update sequence' leak is replaced by a message
    # that shows both valid measure forms.
    out = _query(measures="revenue")
    assert out["ok"] is False
    msg = out["errors"][0]
    assert "dictionary update sequence" not in msg
    assert "revenue" in msg and ("sum" in msg or "list" in msg)


# ── Contract and schema ────────────────────────────────────────────────────

def test_list_models_reports_a_file_that_failed_to_load(tmp_path, monkeypatch):
    """One good model stays listed; the broken file is under skipped."""
    import sys

    models = tmp_path / "models"
    models.mkdir()
    (models / "good_model.py").write_text(
        "from tracebi import DataModel\nmodel = DataModel('good_model')\n",
        encoding="utf-8",
    )
    (models / "broken_model.py").write_text("model = nope\n", encoding="utf-8")
    monkeypatch.setenv("TRACEBI_MODELS_DIR", str(models))

    reg = model_registry._registry
    saved_default = reg._default
    try:
        listing = gateway_models()
        assert "good_model" in listing["models"]
        assert "broken_model" not in listing["models"]
        broken = next(
            item for item in listing["skipped"]
            if item["file"] == "models/broken_model.py"
        )
        assert broken["error"].startswith("NameError:")
        assert "\n" not in broken["error"]
        described = gateway_model_info("broken_model")
        assert described["error"] == broken["error"]
        assert "not found" not in described["error"]
    finally:
        for stem in ("good_model", "broken_model"):
            reg._models.pop(stem, None)
            reg._paths.pop(stem, None)
            reg._mtime_ns.pop(stem, None)
            for key, origin in list(reg._origin.items()):
                if origin == stem or key == stem:
                    reg._models.pop(key, None)
                    reg._origin.pop(key, None)
            sys.modules.pop(f"tracebi_model_{stem}", None)
        reg._default = saved_default


def test_models_listing_has_no_skipped_files_when_nothing_failed(gateway_model):
    assert gateway_models()["skipped"] == []


def test_models_listing_collapses_aliases(gateway_model, monkeypatch):
    """stem + .name index the same object; the listing shows one model."""
    import tracebi.mcp_server as gw

    monkeypatch.setattr(
        gw, "_load_models",
        lambda: {"gw_demo": gateway_model, "gw_demo_file": gateway_model},
    )
    listing = gw.gateway_models()["models"]
    assert list(listing) == ["gw_demo"]
    assert listing["gw_demo"]["aliases"] == ["gw_demo_file"]


def test_named_measure_queries_through_the_gateway(gateway_model):
    out = _query(measures=["total_revenue"], dimensions=[])
    assert out["row_count"] == 1
    assert out["rows"][0]["total_revenue"] == pytest.approx(900.0)
    assert out["fingerprint"]


# ── Spec validation and rendering ──────────────────────────────────────────

def _spec(fact="fact_orders"):
    return {
        "name": "GW Spec",
        "sections": [{
            "type": "table",
            "title": "Revenue by region",
            "data": {
                "model": "gw_demo",
                "query": {
                    "fact": fact,
                    "measures": {"revenue": "sum"},
                    "dimensions": ["dim_customer.region"],
                },
            },
        }],
    }


def test_validate_paths_a_bad_fact(gateway_model):
    result = gateway_validate_spec(_spec(fact="fact_nope"))
    assert not result["ok"]
    assert any("fact" in e for e in result["errors"])


def test_validate_survives_garbage():
    result = gateway_validate_spec("{not json")
    assert not result["ok"]
    assert result["errors"]


def test_render_refuses_an_invalid_spec(gateway_model, tmp_path):
    out = gateway_render_spec(_spec(fact="fact_nope"), output_dir=str(tmp_path))
    assert not out["ok"]
    assert not list(tmp_path.iterdir()), "no artifact may exist for a refused spec"


# ── fetch_artifact: deliver the rendered bytes over MCP (WALL 2) ────────────

def test_fetch_returns_the_rendered_bytes(gateway_model, tmp_path, monkeypatch):
    """A render/build tool returns a server-side PATH; fetch_artifact delivers
    the actual bytes so a remote agent can send the report or verify it."""
    from tracebi.mcp_server import gateway_fetch_artifact

    monkeypatch.chdir(tmp_path)
    out = gateway_render_spec(_spec(), output_dir="out")
    assert out["ok"], out.get("errors")

    html = gateway_fetch_artifact(out["html_path"])
    assert html["ok"] and html["content_type"] == "text/html"
    assert "data-tb-figure" in html["content"]          # the real artifact
    assert html["bytes"] == len(html["content"].encode("utf-8"))

    manifest = gateway_fetch_artifact(out["manifest_path"])
    assert manifest["ok"] and manifest["content_type"] == "application/json"
    assert json.loads(manifest["content"])["schema_version"] == 2


def test_fetch_is_path_guarded(gateway_model, tmp_path, monkeypatch):
    """Read-only and confined: no traversal, no absolute escape, no arbitrary
    file type, and nothing inside the installed package."""
    from tracebi.mcp_server import gateway_fetch_artifact

    monkeypatch.chdir(tmp_path)
    (tmp_path / "art.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "secret.py").write_text("TOKEN = 'x'", encoding="utf-8")

    assert gateway_fetch_artifact("art.html")["ok"]              # in-root artifact OK
    assert not gateway_fetch_artifact("../escape.html")["ok"]    # traversal
    assert not gateway_fetch_artifact("/etc/hosts")["ok"]        # absolute escape
    assert not gateway_fetch_artifact("secret.py")["ok"]         # wrong extension
    assert not gateway_fetch_artifact("missing.html")["ok"]      # no such file

    import os

    import tracebi
    pkg = os.path.dirname(tracebi.__file__)
    inside_pkg = os.path.join(pkg, "web", "ui", "dist", "index.html")
    assert not gateway_fetch_artifact(inside_pkg)["ok"]          # installed package


# ── build_report: the publish step for the package lane ────────────────────

class TestBuildReport:
    def _package(self, tmp_path, monkeypatch, gateway_model):
        import tracebi.mcp_server as gw

        reports = tmp_path / "reports"
        pkg = reports / "gwpkg"
        pkg.mkdir(parents=True)
        (pkg / "report.json").write_text(json.dumps({
            "name": "gwpkg",
            "data": {"kpi": {"model": "gw_demo",
                             "query": {"fact": "fact_orders",
                                       "measures": ["total_revenue"]}}},
        }))
        (pkg / "template.html").write_text(
            "<html><head><title>g</title></head><body>"
            '<div data-tb-figure="value" data-tb-binding="kpi" '
            'data-tb-cell="total_revenue" id="fig-kpi"></div>'
            '<section data-tb-stage="exploration"><p>scratch</p></section>'
            "</body></html>"
        )
        monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(reports))
        monkeypatch.setattr(gw, "_load_models",
                            lambda: {"gw_demo": gateway_model})
        return pkg

    def test_xlsx_writes_the_tables_and_fetch_returns_them(
            self, gateway_model, tmp_path, monkeypatch):
        """format=xlsx writes a workbook of the report's tables beside the
        HTML receipt. fetch_artifact returns that workbook base64-encoded.
        A path outside the output root is still refused."""
        import base64

        from openpyxl import load_workbook

        from tracebi.mcp_server import gateway_build_report, gateway_fetch_artifact

        self._package(tmp_path, monkeypatch, gateway_model)
        out_dir = tmp_path / "out"
        out = gateway_build_report(
            "gwpkg", output_dir=str(out_dir), format="xlsx")
        assert out["ok"], out.get("errors")
        xlsx = out_dir / "gwpkg.xlsx"
        assert out["xlsx_path"] == str(xlsx)
        assert xlsx.is_file()
        assert not (out_dir / "gwpkg.xlsx.manifest.json").exists()
        assert (out_dir / "gwpkg.html").is_file()
        assert (out_dir / "gwpkg.html.manifest.json").is_file()
        note = out["spreadsheet_note"]
        assert "no receipt" in note and "not verifiable" in note
        assert "output_path" in note and "manifest_path" in note

        book = load_workbook(xlsx)
        cells = [
            c.value for row in book["Report"].iter_rows() for c in row
            if c.value is not None
        ]
        assert "kpi" in cells
        assert "total_revenue" in cells
        assert 900 in cells or 900.0 in cells

        monkeypatch.chdir(tmp_path)
        fetched = gateway_fetch_artifact(str(xlsx))
        assert fetched["ok"], fetched.get("errors")
        assert fetched["encoding"] == "base64"
        assert fetched["content_type"] == (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert base64.b64decode(fetched["content"]) == xlsx.read_bytes()
        assert fetched["bytes"] == xlsx.stat().st_size

        monkeypatch.setenv("TRACEBI_OUTPUT_ROOT", str(out_dir))
        escaped = gateway_build_report(
            "gwpkg", output_dir=str(tmp_path / "elsewhere"), format="xlsx")
        assert not escaped["ok"]
        assert "TRACEBI_OUTPUT_ROOT" in escaped["errors"][0]
        assert not (tmp_path / "elsewhere" / "gwpkg.xlsx").exists()
        outside = tmp_path / "elsewhere" / "gwpkg.xlsx"
        outside.parent.mkdir()
        outside.write_bytes(b"PK\x03\x04not-a-real-book")
        refused = gateway_fetch_artifact(str(outside))
        assert not refused["ok"]

    def test_pdf_writes_a_print_and_fetch_returns_it(
            self, gateway_model, tmp_path, monkeypatch):
        """format=pdf writes a print beside the HTML. fetch returns it base64.
        The note says the PDF carries no receipt."""
        import base64
        from pathlib import Path

        from tracebi.mcp_server import gateway_build_report, gateway_fetch_artifact

        def fake(html_path, pdf_path):
            Path(pdf_path).write_bytes(b"%PDF-1.4\n" + b"x" * 32)

        monkeypatch.setattr("tracebi.reports.pdf.print_pdf", fake)
        self._package(tmp_path, monkeypatch, gateway_model)
        out_dir = tmp_path / "out"
        out = gateway_build_report(
            "gwpkg", output_dir=str(out_dir), format="pdf")
        assert out["ok"], out.get("errors")
        pdf = out_dir / "gwpkg.pdf"
        assert out["pdf_path"] == str(pdf)
        assert pdf.read_bytes().startswith(b"%PDF")
        assert not (out_dir / "gwpkg.pdf.manifest.json").exists()
        assert (out_dir / "gwpkg.html").is_file()
        note = out["pdf_note"]
        assert "no receipt" in note
        assert "output_path" in note and "manifest_path" in note

        monkeypatch.chdir(tmp_path)
        fetched = gateway_fetch_artifact(str(pdf))
        assert fetched["ok"], fetched.get("errors")
        assert fetched["encoding"] == "base64"
        assert fetched["content_type"] == "application/pdf"
        assert base64.b64decode(fetched["content"]) == pdf.read_bytes()

    def test_refuses_writing_into_the_installed_package(self, gateway_model,
                                                        tmp_path, monkeypatch):
        """Always on: an agent must not write into the tracebi package (e.g.
        clobber web/ui/dist/index.html, which a server then serves)."""
        import pathlib

        import tracebi
        from tracebi.mcp_server import gateway_build_report

        self._package(tmp_path, monkeypatch, gateway_model)
        pkg_dist = str(
            pathlib.Path(tracebi.__file__).resolve().parent / "web" / "ui" / "dist")
        out = gateway_build_report("index", output_dir=pkg_dist)
        assert not out["ok"]
        assert "package" in out["errors"][0]

    def test_strict_confinement_refuses_output_outside_the_root(
            self, gateway_model, tmp_path, monkeypatch):
        """Opt-in: with TRACEBI_OUTPUT_ROOT set, an absolute path outside it or
        a traversal that escapes it is refused before any write."""
        from tracebi.mcp_server import gateway_build_report

        self._package(tmp_path, monkeypatch, gateway_model)
        monkeypatch.setenv("TRACEBI_OUTPUT_ROOT", str(tmp_path / "out"))
        for bad in ["/tmp/tracebi-escape-xyz", str(tmp_path / "elsewhere")]:
            out = gateway_build_report("gwpkg", output_dir=bad)
            assert not out["ok"], f"{bad!r} was not refused"
            assert "TRACEBI_OUTPUT_ROOT" in out["errors"][0]

    def test_missing_package_is_a_result_not_a_crash(self, gateway_model,
                                                     tmp_path, monkeypatch):
        from tracebi.mcp_server import gateway_build_report

        self._package(tmp_path, monkeypatch, gateway_model)
        out = gateway_build_report("nope")
        assert not out["ok"]
        assert "no artifact package" in out["errors"][0]


# ── workbench_state: no report means the discovery session ─────────────────

def test_workbench_state_defaults_to_discovery(tmp_path, monkeypatch):
    """Called with no report (the human is running `tracebi dev` with no
    name), the tool returns the project-level discovery state."""
    import tracebi.mcp_server as gw
    from tracebi.mcp_server import gateway_workbench_state

    monkeypatch.chdir(tmp_path)
    # Isolate from the process-global model registry: earlier tests
    # register models whose DuckDB warehouses discovery would (correctly)
    # scan, flipping warehouse.exists in full-suite runs.
    monkeypatch.setattr(gw, "_load_models", lambda: {})
    out = gateway_workbench_state()
    assert out["mode"] == "discovery"
    assert out["name"] == "_discovery"
    assert out["warehouse"]["exists"] is False
    assert out["packages"] == []
    assert out["exhibits"] == [] and out["pins"] == []


def test_workbench_state_refuses_a_path_shaped_name(tmp_path, monkeypatch):
    """A caller-supplied report name must never become a path. Without the
    guard, report='/etc/x' or '../../x' escapes reports/ and collect_state
    reads — and executes report.py from — an attacker-chosen directory. The
    refusal must fire before any filesystem access."""
    import tracebi.mcp_server as gw
    from tracebi.mcp_server import gateway_workbench_state

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gw, "_load_models", lambda: {})
    for payload in ["/etc/passwd", "../../etc/x", "../secrets", ".ssh/config",
                    "finance/../../x", "finance\\..\\x"]:
        out = gateway_workbench_state(report=payload)
        assert "errors" in out, f"{payload!r} was not refused: {out!r}"
        assert "invalid report name" in out["errors"][0]
    # A report in a folder is named by its path, and that is not refused: it
    # reaches the package lookup (and finds nothing here).
    out = gateway_workbench_state(report="finance/weekly")
    assert "invalid report name" not in " ".join(out.get("errors") or [])


def test_workbench_state_with_a_name_stays_package_scoped(tmp_path, monkeypatch):
    """A named report keeps exactly the package behavior — a missing
    package is the errors envelope, never the discovery state."""
    from tracebi.mcp_server import gateway_workbench_state

    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(tmp_path / "reports"))
    out = gateway_workbench_state("nope")
    assert "no artifact package" in out["errors"][0]


# ── MCP registration (only when the optional dep is present) ───────────────

def test_build_server_registers_the_tools(gateway_model):
    pytest.importorskip("mcp")
    import anyio

    from tracebi.mcp_server import build_server

    server = build_server()
    tools = anyio.run(server.list_tools)
    names = {t.name for t in tools}
    # M3 flip: workbench_state joined the surface. Round-2 flip: build_report
    # joined it — the publish step for the package lane, so an MCP-driving
    # agent can finish the loop it iterates in the workbench. fetch_artifact
    # then delivers the rendered bytes a remote agent cannot otherwise reach
    # (thirteen tools). The six draft tools make nineteen.
    assert names == {
        "get_context", "list_models", "describe_model", "describe_table",
        "query_model",
        "validate_report_spec", "render_report_spec", "list_reports",
        "verify_manifest", "workbench_state", "resolve_pin", "build_report",
        "fetch_artifact",
        "list_drafts", "start_draft", "read_draft", "write_draft_file",
        "preview_draft", "publish_draft",
    }


# ── MCP 2.0 protocol features ────────────────────────────────────────────────
# The gateway advertises typed structured output, read-only tool annotations
# (the "read-and-compute only" refusal, in the protocol), reference resources,
# and an authoring prompt.

class TestMcp2Features:
    def _tools(self):
        pytest.importorskip("mcp")
        import anyio
        from tracebi.mcp_server import build_server
        server = build_server()
        return server, {t.name: t for t in anyio.run(server.list_tools)}

    def test_every_tool_advertises_an_output_schema(self, gateway_model):
        _server, tools = self._tools()
        for name, t in tools.items():
            d = t.model_dump(by_alias=True, exclude_none=True)
            assert d.get("outputSchema"), f"{name} has no outputSchema"

    def test_read_tools_are_annotated_read_only(self, gateway_model):
        _server, tools = self._tools()
        read_only = {
            "get_context", "list_models", "describe_model", "describe_table",
            "query_model",
            "validate_report_spec", "list_reports", "verify_manifest",
            "fetch_artifact",
        }
        for name in read_only:
            ann = tools[name].annotations
            assert ann is not None
            assert ann.model_dump(by_alias=True).get("readOnlyHint") is True, name
        # render is the one writer — it must NOT claim read-only, and it is
        # non-destructive (writes only its own artifact, never source data).
        render = tools["render_report_spec"].annotations.model_dump(by_alias=True)
        assert render.get("readOnlyHint") is False
        assert render.get("destructiveHint") is False
        # resolve_pin writes pins.json only — a write, like build_report.
        pin = tools["resolve_pin"].annotations.model_dump(by_alias=True)
        built = tools["build_report"].annotations.model_dump(by_alias=True)
        assert pin.get("readOnlyHint") is False
        assert built.get("readOnlyHint") is False
        assert pin.get("destructiveHint") is False
        # The draft writers say so; reading and previewing a draft write nothing.
        for name in ("start_draft", "write_draft_file", "publish_draft"):
            assert tools[name].annotations.model_dump(by_alias=True)["readOnlyHint"] is False, name
        for name in ("list_drafts", "read_draft", "preview_draft"):
            assert tools[name].annotations.model_dump(by_alias=True)["readOnlyHint"] is True, name

    def test_query_tool_emits_structured_content(self, gateway_model):
        pytest.importorskip("mcp")
        import anyio
        server, _ = self._tools()

        async def call():
            return await server.call_tool("query_model", {
                "model": "gw_demo", "fact": "fact_orders",
                "measures": {"revenue": "sum"},
                "dimensions": ["dim_customer.region"],
            })

        result = anyio.run(call)
        payload = result.model_dump(by_alias=True, exclude_none=True)
        assert payload.get("isError") is not True
        sc = payload.get("structuredContent")
        assert sc and sc.get("fingerprint"), "the stamp must arrive as structured content"
        assert sc.get("row_count") == 3

    def test_resources_and_template_are_registered_and_readable(self, gateway_model):
        pytest.importorskip("mcp")
        import anyio
        server, _ = self._tools()

        static = {str(r.uri) for r in anyio.run(server.list_resources)}
        assert {"tracebi://guide", "tracebi://spec-schema"} <= static
        templates = {t.uri_template for t in anyio.run(server.list_resource_templates)}
        assert "tracebi://models/{name}" in templates

        # the spec-schema resource returns the real ReportSpec JSON Schema
        schema = list(anyio.run(server.read_resource, "tracebi://spec-schema"))[0].content
        assert json.loads(schema).get("$schema")
        # the model template resolves to that model's schema
        model_doc = list(anyio.run(server.read_resource, "tracebi://models/gw_demo"))[0].content
        assert json.loads(model_doc)["name"] == "gw_demo"

    def test_knowledge_resource_returns_the_lesson_and_rejects_an_unknown_slug(
            self, gateway_model):
        """A lesson body is a resource, not a fourteenth tool."""
        pytest.importorskip("mcp")
        import anyio
        from mcp.shared.exceptions import MCPError

        from tracebi.knowledge import get_lesson

        server, tools = self._tools()
        assert len(tools) == 19
        templates = {t.uri_template for t in anyio.run(server.list_resource_templates)}
        assert "tracebi://knowledge/{slug}" in templates
        assert "tracebi://knowledge/{slug}" in server.instructions

        lesson = get_lesson("share-of-total")
        body = list(anyio.run(
            server.read_resource, "tracebi://knowledge/share-of-total"))[0].content
        assert body == f"# {lesson.title}\n\n{lesson.body}\n"
        assert "revenue_share" in body

        with pytest.raises(MCPError, match="No lesson 'no-such-lesson'"):
            anyio.run(server.read_resource, "tracebi://knowledge/no-such-lesson")

    def test_author_report_prompt_walks_the_loop(self, gateway_model):
        pytest.importorskip("mcp")
        import anyio
        server, _ = self._tools()

        prompts = {p.name for p in anyio.run(server.list_prompts)}
        assert "author_report" in prompts

        async def render():
            return await server.get_prompt("author_report", {"question": "revenue by region?"})

        got = anyio.run(render)
        text = got.messages[0].content.text
        assert "revenue by region?" in text
        for step in ("get_context", "query_model", "build_report",
                     "validate_report_spec", "render_report_spec",
                     "verify_manifest"):
            assert step in text, f"the SOP prompt should name {step}"
        # The package lane leads; the spec lane is the fallback.
        assert text.index("build_report") < text.index("render_report_spec")
        # The prompt names the three page structures, one line each. The agent
        # picks by fit; it does not stop to ask, and a gateway-only agent
        # writes the skeleton itself.
        assert text.index("- brief") < text.index("- dashboard") < text.index("- tabbed")
        assert "--layout" in text
        assert "Ask which page structure" not in text
        assert "only the gateway" in text

    def test_instructions_lead_with_the_package_lane(self, gateway_model):
        server, _ = self._tools()
        text = server.instructions
        assert text.index("build_report") < text.index("render_report_spec")
        for name in ("workbench_state", "resolve_pin", "verify_manifest",
                     "answer_question", "address_pins"):
            assert name in text
        assert "verify_manifest(manifest=...)" in text
        assert "fetch_artifact(path=...)" in text
        assert "output_path" in text
        assert "describe_table" in text
        assert "analyst_knowledge.lessons" in text
        assert "tracebi://knowledge/{slug}" in text
        assert "list_models" in text

    def test_tool_descriptions_name_the_argument_that_feeds_the_next_call(
            self, gateway_model):
        _server, tools = self._tools()
        desc = {name: t.description for name, t in tools.items()}
        assert "argument is report" in desc["build_report"]
        assert "output_path" in desc["build_report"]
        assert "verify_manifest(manifest=...)" in desc["build_report"]
        assert "argument is path" in desc["fetch_artifact"]
        assert "build_report's output_path" in desc["fetch_artifact"]
        assert "argument is manifest" in desc["verify_manifest"]
        assert "build_report's manifest_path" in desc["verify_manifest"]
        assert "data.<name>" in desc["query_model"]

    def test_answer_question_and_address_pins_render(self, gateway_model):
        pytest.importorskip("mcp")
        import anyio
        server, _ = self._tools()

        prompts = {p.name for p in anyio.run(server.list_prompts)}
        assert {"author_report", "answer_question", "address_pins"} <= prompts

        async def render(name, args):
            got = await server.get_prompt(name, args)
            return got.messages[0].content.text

        answer = anyio.run(render, "answer_question",
                           {"question": "revenue by region?", "model": "gw_demo"})
        assert "revenue by region?" in answer
        assert "'gw_demo'" in answer
        assert "Never estimate" in answer
        assert "fingerprint" in answer
        assert "Do not build a report unless" in answer
        no_model = anyio.run(render, "answer_question", {"question": "q?"})
        assert "model=" not in no_model

        pins = anyio.run(render, "address_pins", {"report": "weekly"})
        for step in ("workbench_state", "build_report", "resolve_pin",
                     "reports/weekly/"):
            assert step in pins
        assert pins.index("workbench_state") < pins.index("build_report") \
            < pins.index("verify_manifest") < pins.index("resolve_pin")


def test_brief_context_returns_the_presentation_grammar(gateway_model):
    """The function-level payload, both tiers. brief is what agents call
    first; the figure grammar has to be in it."""
    from tracebi.mcp_server import gateway_context

    brief = gateway_context(brief=True)
    attrs = brief["presentation"]["figure_attributes"]
    assert "data-tb-figure" in attrs
    assert "data-tb-format" in attrs
    assert "currency" in brief["number_formats"]
    assert "cheat_sheets" not in brief
    assert "report_sections" not in brief
    assert "dataset_verbs" not in brief
    assert "model" not in brief
    assert "omitted" in brief["brief"]

    full = gateway_context(brief=False)
    assert "cheat_sheets" in full
    assert "report_sections" in full
    assert "presentation" in full
    assert "brief" not in full

    named = gateway_context(brief=True, model="gw_demo")
    assert named["model"]["name"] == "gw_demo"


def test_structured_brief_context_keeps_presentation(gateway_model):
    """The SDK drops any returned key the output schema does not name.
    Omitted schema keys are null on mcp 2.0 and absent on mcp 2.3.
    The grammar must survive either."""
    pytest.importorskip("mcp")
    import anyio
    from tracebi.mcp_server import build_server

    server = build_server()

    async def call(args):
        result = await server.call_tool("get_context", args)
        payload = result.model_dump(by_alias=True, exclude_none=True)
        assert payload.get("isError") is not True
        return payload["structuredContent"]

    brief = anyio.run(call, {"brief": True})
    assert "data-tb-figure" in brief["presentation"]["figure_attributes"]
    assert brief["number_formats"]["currency"]
    assert brief.get("cheat_sheets") is None
    assert brief.get("report_sections") is None
    assert brief.get("dataset_verbs") is None
    assert brief.get("model") is None
    assert brief["brief"]["omitted"] == [
        "cheat_sheets", "report_sections", "dataset_verbs",
    ]

    full = anyio.run(call, {"brief": False})
    assert full["cheat_sheets"]
    assert full["presentation"]["figure_attributes"]["data-tb-figure"]
    assert full.get("brief") is None
    assert full.get("model") is None


def test_a_schema_checking_client_accepts_every_tool(gateway_model, tmp_path,
                                                     monkeypatch):
    """Drive the real server through a schema-validating MCP client.

    The SDK fills every omitted result field with null and advertises that
    on the output schema. A client that checks structured content
    (``ClientSession.call_tool`` → ``validate_tool_result``) rejects a null
    the schema types as string, number, boolean, or array — and a success
    looks the same as a failure. Both the success path and the error
    envelope of every tool must come back as structured content.
    """
    pytest.importorskip("mcp")
    import anyio
    from mcp.client import Client

    from tracebi.mcp_server import build_server

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRACEBI_MODELS_DIR", str(tmp_path / "models"))
    reports = tmp_path / "reports"
    pkg = reports / "gwpkg"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "gwpkg",
        "data": {"kpi": {"model": "gw_demo",
                         "query": {"fact": "fact_orders",
                                   "measures": ["total_revenue"]}}},
    }), encoding="utf-8")
    (pkg / "template.html").write_text(
        "<html><head><title>g</title></head><body>"
        '<div data-tb-figure="value" data-tb-binding="kpi" '
        'data-tb-cell="total_revenue" id="fig-kpi"></div>'
        "</body></html>",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(reports))
    out_dir = tmp_path / "built"
    server = build_server()

    async def drive():
        async with Client(server) as client:
            seen: list[str] = []

            async def call(name, arguments=None):
                result = await client.call_tool(name, arguments or {})
                seen.append(name)
                # A tool that raises is an isError result; the client does
                # not schema-check those. Every normal result — success or
                # the {ok, errors} envelope — must carry structured content.
                if not result.is_error:
                    assert result.structured_content is not None, name
                return result

            ctx = await call("get_context", {"brief": True})
            assert ctx.structured_content.get("tracebi_version")

            listed = await call("list_models")
            assert "gw_demo" in listed.structured_content["models"]

            described = await call("describe_model", {"model": "gw_demo"})
            assert described.structured_content.get("name") == "gw_demo"

            unknown = await call("describe_model", {"model": "no_such_model"})
            if unknown.is_error:
                text = unknown.content[0].text
                assert "Invalid structured content" not in text
                assert "not found" in text
            else:
                assert unknown.structured_content is not None

            tables = await call("describe_table")
            assert tables.structured_content.get("ok") is True
            one_table = await call("describe_table", {"table": "orders"})
            assert "ok" in one_table.structured_content

            query = await call("query_model", {
                "model": "gw_demo",
                "fact": "fact_orders",
                "measures": {"revenue": "sum"},
                "dimensions": ["dim_customer.region"],
            })
            assert query.structured_content.get("ok") is True
            assert query.structured_content.get("fingerprint")

            bad_query = await call("query_model", {
                "model": "gw_demo",
                "fact": "fact_orders",
                "measures": ["no_such_measure"],
            })
            assert bad_query.structured_content.get("ok") is False

            valid = await call("validate_report_spec", {"spec": _spec()})
            assert valid.structured_content.get("ok") is True
            invalid = await call("validate_report_spec",
                                 {"spec": _spec(fact="fact_nope")})
            assert invalid.structured_content.get("ok") is False

            rendered = await call("render_report_spec", {
                "spec": _spec(), "output_dir": str(out_dir),
            })
            assert rendered.structured_content.get("ok") is True
            refused_spec = await call("render_report_spec", {
                "spec": _spec(fact="fact_nope"), "output_dir": str(out_dir),
            })
            assert refused_spec.structured_content.get("ok") is False

            reports_listing = await call("list_reports")
            assert "reports" in reports_listing.structured_content

            built = await call("build_report", {
                "report": "gwpkg", "output_dir": str(out_dir),
            })
            assert built.structured_content.get("ok") is True, (
                built.structured_content)
            html_path = built.structured_content["output_path"]
            manifest_path = built.structured_content["manifest_path"]

            missing = await call("build_report", {
                "report": "missing", "output_dir": str(out_dir),
            })
            assert missing.structured_content.get("ok") is False

            fetched = await call("fetch_artifact", {"path": html_path})
            assert fetched.structured_content.get("ok") is True
            blocked = await call("fetch_artifact", {"path": "/etc/hosts"})
            assert blocked.structured_content.get("ok") is False

            verified = await call("verify_manifest", {"manifest": manifest_path})
            assert verified.structured_content.get("verdict")
            missing_manifest = await call("verify_manifest", {
                "manifest": str(tmp_path / "no-such.manifest.json"),
            })
            assert missing_manifest.structured_content.get("ok") is False

            discovery = await call("workbench_state")
            assert discovery.structured_content.get("mode") == "discovery"
            package_state = await call("workbench_state", {"report": "gwpkg"})
            assert package_state.structured_content.get("name")

            pin = await call("resolve_pin", {
                "report": "gwpkg", "pin_id": "no-such-pin",
            })
            assert pin.structured_content.get("ok") is False

            assert set(seen) == {
                "get_context", "list_models", "describe_model", "describe_table",
                "query_model", "validate_report_spec", "render_report_spec",
                "list_reports", "verify_manifest", "workbench_state",
                "resolve_pin", "build_report", "fetch_artifact",
            }

    anyio.run(drive)

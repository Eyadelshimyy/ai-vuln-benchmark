"""Selection through DataModel, then the offline port that has to match it.

The authored selection is the receipt. A reader selection is the same filter
grammar, conjoined, re-run by the model. The fingerprint equals that query.
A report with no selection block still does not recompute. The worker port
of simple aggregations and ratio matches the model; period_end refuses.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector, QuerySpec
from tracebi.model.dataset import frame_fingerprint
from tracebi.reports.selection import (
    answer_question,
    conjoin_filters,
    evaluate_selection,
    filters_equal,
    keep_cut,
    match_question,
    query_under_selection,
    resolve_question,
)
from tracebi.reports.selection_eval import (
    OfflineRefusal,
    aggregate_frame,
    plan_binding,
)
from tracebi.reports.template_package import TemplatePackage

_EVAL_JS = (
    __import__("pathlib").Path(__file__).parent.parent
    / "tracebi" / "reports" / "assets" / "selection_eval.js"
)


def _model():
    orders = pd.DataFrame({
        "customer_id": [10, 10, 20, 20],
        "revenue": [100, 50, 80, 20],
        "cost": [40, 10, 30, 10],
        "order_id": [1, 2, 3, 4],
    })
    customers = pd.DataFrame({
        "customer_id": [10, 20],
        "region": ["East", "West"],
    })
    m = DataModel("selection_model")
    m.add_connector(MemoryConnector("sel_mem", tables={
        "orders": orders, "customers": customers,
    }))
    m.add_table("orders", connector="sel_mem", source="orders")
    m.add_table("customers", connector="sel_mem", source="customers")
    m.add_dimension(
        "dim_customer", table_name="customers", key_col="customer_id",
        attributes=["region"],
    )
    m.add_fact(
        "fact_orders", table_name="orders",
        measures=["revenue", "cost", "order_id"],
        foreign_keys={"dim_customer": "customer_id"},
    )
    m.add_measure("revenue", column="revenue", agg="sum")
    m.add_measure("cost", column="cost", agg="sum")
    m.add_measure("orders", column="order_id", agg="count")
    m.add_measure("margin", ratio=("revenue", "cost"))
    m.add_measure("balance", period_end=("revenue", "dim_customer.region"))
    return m


def _package(tmp_path, *, selection=None, binding_filters=None):
    pkg = tmp_path / "demo"
    pkg.mkdir()
    query = {
        "fact": "fact_orders",
        "measures": ["revenue", "orders", "margin"],
        "dimensions": ["dim_customer.region"],
    }
    if binding_filters:
        query["filters"] = binding_filters
    document = {
        "name": "demo",
        "data": {
            "by_region": {"model": "selection_model", "query": query},
            "totals": {
                "model": "selection_model",
                "query": {
                    "fact": "fact_orders",
                    "measures": ["revenue", "margin"],
                },
            },
        },
    }
    if selection is not None:
        document["selection"] = selection
    (pkg / "report.json").write_text(json.dumps(document), encoding="utf-8")
    (pkg / "template.html").write_text(
        "<html><head><title>demo</title></head><body>"
        '<div class="tb-kpi" data-tb-figure="value" data-tb-binding="totals" '
        'data-tb-cell="revenue" id="fig-total"></div>'
        '<div data-tb-figure="value" data-tb-binding="totals" '
        'data-tb-cell="margin" id="fig-margin"></div>'
        '<table data-tb-figure="table" data-tb-binding="by_region" '
        'id="fig-regions"></table>'
        '<select data-tb-filter data-tb-binding="by_region" '
        'data-tb-column="dim_customer.region"></select>'
        "</body></html>",
        encoding="utf-8",
    )
    return pkg


class TestConjoin:
    def test_selection_wins_on_the_same_target_and_keeps_the_rest(self):
        merged = conjoin_filters(
            {"dim_customer.region": "East", "status": "open"},
            {"dim_customer.region": "West"},
        )
        assert merged == {"dim_customer.region": "West", "status": "open"}

    def test_having_order_and_limit_stay_on_the_binding(self):
        query = QuerySpec(
            fact="fact_orders", measures=["revenue"],
            filters={"status": "open"},
            having={"revenue": {"gte": 10}},
            order_by=("-revenue",), limit=3,
        )
        out = query_under_selection(query, {"dim_customer.region": "East"})
        assert out.having == {"revenue": {"gte": 10}}
        assert out.limit == 3
        assert out.order_by[0]["column"] == "revenue"
        assert out.filters["status"] == "open"
        assert out.filters["dim_customer.region"] == "East"

    def test_empty_selection_does_not_rewrite_an_unfiltered_query(self):
        query = QuerySpec(fact="fact_orders", measures=["revenue"])
        assert query_under_selection(query, {}) is query
        assert filters_equal({}, None)


class TestPackageSelection:
    def test_empty_filters_opt_in_without_an_extra_cut(self, tmp_path):
        pkg = TemplatePackage(str(_package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )))
        assert pkg.selection == {"model": "selection_model", "filters": {}}

    def test_build_stamps_the_authored_selection(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(
            tmp_path,
            selection={"model": "selection_model",
                       "filters": {"dim_customer.region": "East"}},
        )))
        _report, stamped = pkg.build({"selection_model": model})
        by_name = {sd.name: sd for sd in stamped}
        expected = model.query(
            "fact_orders", ["revenue", "margin"],
            filters={"dim_customer.region": "East"},
        )
        assert by_name["totals"].fingerprint == expected.fingerprint()
        assert by_name["totals"].query_spec["filters"] == {
            "dim_customer.region": "East",
        }


class TestEvaluate:
    def test_fingerprint_equals_the_conjoined_query(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(
            tmp_path,
            selection={"model": "selection_model", "filters": {}},
            binding_filters={"dim_customer.region": "East"},
        )))
        # The reader moves the region control to West. The binding's East
        # predicate loses, because that target is the control.
        result = evaluate_selection(
            pkg, {"selection_model": model},
            {"dim_customer.region": "West"},
        )
        assert result["authored"] is False
        totals = model.query(
            "fact_orders", ["revenue", "margin"],
            filters={"dim_customer.region": "West"},
        )
        fig = next(f for f in result["figures"] if f["id"] == "fig-total")
        assert fig["fingerprint"] == totals.fingerprint()
        assert fig["value"] == pytest.approx(100.0)
        regions = next(f for f in result["figures"] if f["id"] == "fig-regions")
        assert regions["rows"] == [{
            "dim_customer.region": "West",
            "revenue": 100.0,
            "orders": 2,
            "cost": 40.0,
            "margin": 2.5,
        }]
        control = result["controls"][0]
        assert "West" in control["included"]
        # East still has rows under no other cut, so it is selectable.
        assert "East" in control["included"]
        assert control["excluded"] == []

    def test_a_value_with_no_rows_under_the_rest_is_excluded(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )))
        result = evaluate_selection(
            pkg, {"selection_model": model},
            {"dim_customer.region": "East", "revenue": {"gte": 1000}},
        )
        control = result["controls"][0]
        assert control["included"] == []
        assert control["excluded"] == ["East", "West"]

    def test_without_a_block_selection_is_refused(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(tmp_path)))
        with pytest.raises(ValueError, match="no selection block"):
            evaluate_selection(pkg, {"selection_model": model}, {})

    def test_page_without_a_block_keeps_connect_src_none(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(tmp_path)))
        out = tmp_path / "plain.html"
        pkg.render({"selection_model": model}, str(out), save_manifest=False)
        html = out.read_text(encoding="utf-8")
        assert "connect-src 'none'" in html
        assert 'id="tracebi-selection"' not in html

    def test_opted_in_page_can_reach_the_model(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )))
        out = tmp_path / "cut.html"
        pkg.render({"selection_model": model}, str(out), save_manifest=True)
        html = out.read_text(encoding="utf-8")
        assert "connect-src 'self'" in html
        assert 'id="tracebi-selection"' in html
        assert 'id="tracebi-grain"' in html
        from tracebi.verify import verify_file
        manifest = json.loads((tmp_path / "cut.html.manifest.json").read_text())
        checked = verify_file(html, manifest)
        assert checked["ok"], checked["verdict_detail"]


class TestKeep:
    def test_keep_writes_the_cut_and_reproduces(self, tmp_path, monkeypatch):
        model = _model()
        pkg_dir = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        monkeypatch.chdir(tmp_path)
        result = keep_cut(
            str(pkg_dir),
            {"dim_customer.region": "East"},
            {"selection_model": model},
            str(tmp_path / "output" / "demo.html"),
        )
        assert result["verdict"] == "reproduces"
        assert result["ok"] is True
        saved = json.loads((pkg_dir / "report.json").read_text(encoding="utf-8"))
        assert saved["selection"]["filters"] == {"dim_customer.region": "East"}
        manifest = json.loads(
            (tmp_path / "output" / "demo.html.manifest.json").read_text()
        )
        # The authored receipt is the cut that was on screen.
        for section in manifest["sections"]:
            node_specs = []
            for node in section.get("dataset_lineage") or []:
                spec = (node.get("metadata") or {}).get("query_spec")
                if spec:
                    node_specs.append(spec)
            assert node_specs, section
            assert node_specs[-1]["filters"]["dim_customer.region"] == "East"


class TestAsk:
    def test_a_question_names_one_declared_value(self):
        assert match_question("what about west", {
            "dim_customer.region": ["East", "West"],
        }) == {"dim_customer.region": "West"}

    def test_several_values_on_one_column_become_a_list(self):
        assert match_question("east and west", {
            "dim_customer.region": ["East", "West"],
        }) == {"dim_customer.region": ["East", "West"]}

    def test_a_longer_value_wins_over_its_prefix(self):
        assert match_question("what about north america", {
            "dim.region": ["North", "North America"],
        }) == {"dim.region": "North America"}

    def test_two_columns_are_refused(self):
        with pytest.raises(ValueError, match="more than one control"):
            match_question("east", {
                "dim_customer.region": ["East", "West"],
                "dim_store.name": ["East"],
            })

    def test_an_unknown_word_is_refused(self):
        with pytest.raises(ValueError, match="does not name a value"):
            match_question("what about nowhere", {
                "dim_customer.region": ["East", "West"],
            })

    def test_a_key_value_cut_is_returned_as_written(self):
        assert match_question(
            "dim_customer.region=West", {},
        ) == {"dim_customer.region": "West"}

    def test_question_recomputes_with_the_model_fingerprint(self, tmp_path):
        model = _model()
        pkg = TemplatePackage(str(_package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )))
        filters = resolve_question(pkg, {"selection_model": model}, "what about west")
        assert filters == {"dim_customer.region": "West"}
        result = evaluate_selection(pkg, {"selection_model": model}, filters)
        expected = model.query(
            "fact_orders", ["revenue", "margin"],
            filters={"dim_customer.region": "West"},
        )
        fig = next(f for f in result["figures"] if f["id"] == "fig-total")
        assert fig["fingerprint"] == expected.fingerprint()
        assert fig["value"] == pytest.approx(100.0)


class TestEndpoint:
    def test_post_selection_returns_the_query_fingerprint(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tracebi import model_registry
        from tracebi.web.api.routers import reports as reports_router
        from tracebi.web.discovery import _register_template_package

        model = _model()
        model_registry.register(model)
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        registered = _register_template_package(str(pkg), "sel_demo")
        assert registered["status"] == "registered", registered
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        client = TestClient(app)
        res = client.post(
            "/api/reports/sel_demo/selection",
            json={"filters": {"dim_customer.region": "East"}},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        fig = next(f for f in body["figures"] if f["id"] == "fig-total")
        expected = model.query(
            "fact_orders", ["revenue", "margin"],
            filters={"dim_customer.region": "East"},
        )
        assert fig["fingerprint"] == expected.fingerprint()
        bare = client.post("/api/reports/sel_demo/selection", json={"filters": {}})
        assert bare.status_code == 200
        # A report that never opted in is not this package; the empty-filters
        # request on an opted-in package is the authored cut.
        assert bare.json()["authored"] is True

    def test_a_question_posts_the_resolved_cut(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tracebi import model_registry
        from tracebi.web.api.routers import reports as reports_router
        from tracebi.web.discovery import _register_template_package

        model = _model()
        model_registry.register(model)
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        registered = _register_template_package(str(pkg), "ask_demo")
        assert registered["status"] == "registered", registered
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        client = TestClient(app)
        res = client.post(
            "/api/reports/ask_demo/selection",
            json={"question": "what about west"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["filters"] == {"dim_customer.region": "West"}
        fig = next(f for f in body["figures"] if f["id"] == "fig-total")
        expected = model.query(
            "fact_orders", ["revenue", "margin"],
            filters={"dim_customer.region": "West"},
        )
        assert fig["fingerprint"] == expected.fingerprint()
        missed = client.post(
            "/api/reports/ask_demo/selection",
            json={"question": "what about nowhere"},
        )
        assert missed.status_code == 422

    def test_built_report_opens_the_saved_artifact(self, tmp_path, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tracebi.web.api.routers import reports as reports_router
        from tracebi.web.discovery import _register_template_package

        monkeypatch.chdir(tmp_path)
        pkg = _package(tmp_path, selection={"model": "selection_model", "filters": {}})
        registered = _register_template_package(str(pkg), "built_demo")
        assert registered["status"] == "registered", registered
        out = tmp_path / "output"
        out.mkdir()
        (out / "built_demo.html").write_text("<html>saved</html>", encoding="utf-8")
        (out / "built_demo.html.manifest.json").write_text(
            '{"report_name": "built_demo"}', encoding="utf-8",
        )
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        client = TestClient(app)
        res = client.get("/api/reports/built_demo/built")
        assert res.status_code == 200, res.text
        assert "saved" in res.json()["html"]

    def test_last_build_is_kept_when_the_disk_is_read_only(self, tmp_path, monkeypatch):
        """A never-built report builds once on first open; after that, opening
        and downloading serve that build and never render again — even on a
        read-only disk, where the file lives under the system temp dir and
        the path is recorded in the run store."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import tracebi.model_registry as model_registry
        from tracebi.reports.template_package import TemplatePackage
        from tracebi.web.api.routers import reports as reports_router
        from tracebi.web.discovery import _register_template_package

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(reports_router, "_writable_output_html", lambda name: None)
        monkeypatch.setattr(reports_router, "_ARTIFACT_TTL_S", 0.0)
        monkeypatch.setattr(reports_router, "_ARTIFACT_CACHE", {})
        renders = []
        real_render = TemplatePackage.render
        monkeypatch.setattr(TemplatePackage, "render",
                            lambda self, *a, **k: renders.append(1) or real_render(self, *a, **k))
        model_registry.register(_model())
        pkg = _package(tmp_path, selection={"model": "selection_model", "filters": {}})
        assert _register_template_package(str(pkg), "ro_demo")["status"] == "registered"
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        client = TestClient(app)

        first = client.get("/api/reports/ro_demo/built")
        assert first.status_code == 200, first.text
        assert first.json()["retained"] is False
        second = client.get("/api/reports/ro_demo/built").json()
        download = client.get("/api/reports/ro_demo/download?format=html")
        assert len(renders) == 1
        assert second["manifest"]["rendered_at"] == first.json()["manifest"]["rendered_at"]
        assert download.text == first.json()["html"]


class TestNewGrain:
    def _segment_model(self):
        orders = pd.DataFrame({
            "customer_id": [10, 10, 20, 20],
            "revenue": [100, 50, 80, 20],
            "cost": [40, 10, 30, 10],
            "order_id": [1, 2, 3, 4],
        })
        customers = pd.DataFrame({
            "customer_id": [10, 20],
            "region": ["East", "West"],
            "segment": ["Retail", "Wholesale"],
            "tier": ["A", "B"],
        })
        model = DataModel("selection_model")
        model.add_connector(MemoryConnector("seg_mem", tables={
            "orders": orders, "customers": customers,
        }))
        model.add_table("orders", connector="seg_mem", source="orders")
        model.add_table("customers", connector="seg_mem", source="customers")
        model.add_dimension(
            "dim_customer", table_name="customers", key_col="customer_id",
            attributes=["region", "segment", "tier"],
        )
        model.add_fact(
            "fact_orders", table_name="orders",
            measures=["revenue", "cost", "order_id"],
            foreign_keys={"dim_customer": "customer_id"},
        )
        model.add_measure("revenue", column="revenue", agg="sum")
        model.add_measure("cost", column="cost", agg="sum")
        model.add_measure("orders", column="order_id", agg="count")
        model.add_measure("margin", ratio=("revenue", "cost"))
        return model

    def test_a_new_grain_becomes_a_binding_and_reproduces(self, tmp_path):
        model = self._segment_model()
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        pin_dir = tmp_path / ".tracebi" / "workbench" / "demo"
        pin_dir.mkdir(parents=True)
        (pin_dir / "pins.json").write_text(json.dumps([
            {"id": "fig-total", "note": "check the total", "at_seq": 1},
        ]), encoding="utf-8")
        result = answer_question(
            str(pkg), {"selection_model": model}, "what about segment",
            output_html=str(tmp_path / "output" / "demo.html"),
            project_root=str(tmp_path),
        )
        assert result["added_binding"] == "by_segment"
        assert result["verdict"] == "reproduces"
        assert result["pins"][0]["note"] == "check the total"
        saved = json.loads((pkg / "report.json").read_text(encoding="utf-8"))
        binding = saved["data"]["by_segment"]
        assert binding["query"]["dimensions"] == ["dim_customer.segment"]
        assert "filters" not in binding["query"]
        assert saved["selection"]["filters"] == {}
        html = (pkg / "template.html").read_text(encoding="utf-8")
        tag = html[html.index('data-tb-binding="by_segment"') - 40:]
        tag = tag[:tag.index(">") + 1]
        assert not any(ch.isdigit() for ch in tag)
        expected = model.query(
            "fact_orders", ["revenue", "orders", "margin"],
            dimensions=["dim_customer.segment"],
            order_by=[{"column": "revenue", "desc": True}],
        )
        fig = next(f for f in result["figures"] if f["binding"] == "by_segment")
        assert fig["fingerprint"] == expected.fingerprint()
        assert fig["kind"] == "table"
        # The thread cites the fingerprint. The table cells are not a value.
        assert "value" not in fig

    def test_an_unknown_word_stays_refused(self, tmp_path):
        model = self._segment_model()
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        with pytest.raises(ValueError, match="does not name a value"):
            answer_question(
                str(pkg), {"selection_model": model}, "what about nowhere",
                output_html=str(tmp_path / "output" / "demo.html"),
                project_root=str(tmp_path),
            )
        saved = json.loads((pkg / "report.json").read_text(encoding="utf-8"))
        assert "by_segment" not in saved["data"]

    def test_an_existing_grain_is_refused(self, tmp_path):
        model = self._segment_model()
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        with pytest.raises(ValueError, match="already a dimension"):
            answer_question(
                str(pkg), {"selection_model": model}, "what about region",
                output_html=str(tmp_path / "output" / "demo.html"),
                project_root=str(tmp_path),
            )

    def test_two_grains_are_refused(self, tmp_path):
        model = self._segment_model()
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        with pytest.raises(ValueError, match="more than one grain"):
            answer_question(
                str(pkg), {"selection_model": model}, "what about customer",
                output_html=str(tmp_path / "output" / "demo.html"),
                project_root=str(tmp_path),
            )

    def test_a_value_question_does_not_write_a_binding(self, tmp_path):
        model = self._segment_model()
        pkg = _package(
            tmp_path, selection={"model": "selection_model", "filters": {}},
        )
        result = answer_question(
            str(pkg), {"selection_model": model}, "what about west",
            output_html=str(tmp_path / "output" / "demo.html"),
            project_root=str(tmp_path),
        )
        assert "added_binding" not in result
        assert result["filters"] == {"dim_customer.region": "West"}
        saved = json.loads((pkg / "report.json").read_text(encoding="utf-8"))
        assert set(saved["data"]) == {"by_region", "totals"}


class TestOfflinePort:
    def _plan(self, measures, dimensions=(), filters=None, order_by=(), limit=None):
        return plan_binding(_model(), QuerySpec(
            fact="fact_orders", measures=measures, dimensions=tuple(dimensions),
            filters=filters, order_by=order_by, limit=limit,
        ))

    def _grain(self, model, plan):
        sources = []
        for agg in plan["aggs"]:
            if agg["source"] not in sources:
                sources.append(agg["source"])
        spec = QuerySpec(
            fact=plan["fact"],
            measures={src: "sum" for src in sources},
            dimensions=tuple(plan["dimensions"]),
            aggregate=False,
        )
        return model.execute(spec).to_pandas()

    def _js(self, rows, plan, filters):
        node = shutil.which("node")
        if node is None:
            pytest.skip("node not available")
        script = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
const payload = JSON.parse(process.argv[2]);
eval(src);
const result = tracebiSelectionEval(payload.rows, payload.plan, payload.filters);
process.stdout.write(JSON.stringify(result));
"""
        proc = subprocess.run(
            [node, "-e", script, str(_EVAL_JS), json.dumps({
                "rows": rows, "plan": plan, "filters": filters,
            })],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    def _as_frame(self, js_rows, plan, grain):
        from tracebi.reports.selection_eval import _cast_agg, _empty_result
        dims = plan["dimensions"]
        aggs = plan["aggs"]
        ratios = plan["ratios"]
        if not js_rows:
            return _empty_result(grain, dims, aggs, ratios)
        df = pd.DataFrame(js_rows)
        cols = dims + [a["name"] for a in aggs] + [r["name"] for r in ratios]
        df = df[cols]
        for agg in aggs:
            df[agg["name"]] = _cast_agg(df[agg["name"]], agg["func"], grain[agg["source"]])
        for dim in dims:
            df[dim] = df[dim].astype(grain[dim].dtype)
        for ratio in ratios:
            df[ratio["name"]] = df[ratio["name"]].astype("float64")
        return df.reset_index(drop=True)

    def test_simple_and_ratio_match_the_model(self):
        model = _model()
        plan = self._plan(
            ["revenue", "orders", "margin"], ["dim_customer.region"],
        )
        assert plan["offline"] is True
        grain = self._grain(model, plan)
        port = aggregate_frame(grain, plan, {})
        expected = model.query(
            "fact_orders", ["revenue", "orders", "margin"],
            ["dim_customer.region"],
        ).to_pandas()
        assert frame_fingerprint(port) == frame_fingerprint(expected)
        js = self._js(
            json.loads(grain.to_json(orient="records")), plan, {},
        )
        assert js.get("refused") in (None, False) or "rows" in js
        js_frame = self._as_frame(js["rows"], plan, grain)
        assert frame_fingerprint(js_frame) == frame_fingerprint(expected)

    def test_a_one_row_drift_changes_the_fingerprint(self):
        model = _model()
        plan = self._plan(["revenue", "margin"])
        grain = self._grain(model, plan)
        expected = model.query(
            "fact_orders", ["revenue", "margin"],
        ).fingerprint()
        js = self._js(json.loads(grain.to_json(orient="records")), plan, {})
        assert frame_fingerprint(self._as_frame(js["rows"], plan, grain)) == expected
        drifted = json.loads(grain.to_json(orient="records"))
        drifted[0]["revenue"] = drifted[0]["revenue"] + 1
        js_drift = self._js(drifted, plan, {})
        assert frame_fingerprint(
            self._as_frame(js_drift["rows"], plan, grain)
        ) != expected

    def test_selection_filter_matches_the_model(self):
        model = _model()
        plan = self._plan(["revenue", "margin"], ["dim_customer.region"])
        grain = self._grain(model, plan)
        filters = {"dim_customer.region": "East"}
        port = aggregate_frame(grain, plan, filters)
        expected = model.query(
            "fact_orders", ["revenue", "margin"],
            ["dim_customer.region"], filters=filters,
        ).to_pandas()
        assert frame_fingerprint(port) == frame_fingerprint(expected)

    def test_period_end_refuses_offline(self):
        plan = self._plan(["balance"])
        assert plan["offline"] is False
        assert plan["reason"] == "period_end"
        with pytest.raises(OfflineRefusal):
            aggregate_frame(pd.DataFrame({"revenue": [1]}), plan, {})
        js = self._js([], plan, {})
        assert js["refused"] == "period_end"

    def test_top_n_stays_inside_the_selection(self):
        model = _model()
        plan = self._plan(
            ["revenue"], ["dim_customer.region"],
            order_by=("-revenue",), limit=1,
        )
        grain = self._grain(model, plan)
        filters = {"dim_customer.region": ["East", "West"]}
        port = aggregate_frame(grain, plan, filters)
        expected = model.query(
            "fact_orders", ["revenue"], ["dim_customer.region"],
            filters=filters, order_by=["-revenue"], limit=1,
        ).to_pandas()
        assert frame_fingerprint(port) == frame_fingerprint(expected)
        assert len(port) == 1

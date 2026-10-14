"""The reference project, end to end: both transforms sink, then EVERY report
in it — packages, folders, the JSON spec, the scenario —
builds and holds to the receipt bar. The showcase and the housing report are
the maintained demos; a feature that breaks either breaks this test, so the
demos can never quietly stop demoing.

Built once per module (the transforms take seconds); each test reads the
built project.
"""

import json
import os
import re
import shutil

import pytest

from tests.e2e.conftest import REFERENCE_PROJECT, run_cli

REPORTS = {
    "portfolio_model/portfolio_dashboard": "portfolio_model/portfolio_dashboard",
    "portfolio_model/portfolio_book": "portfolio_model/portfolio_book",
    "portfolio_model/portfolio_overview": "portfolio_model/portfolio_overview",
    "portfolio_model/portfolio_concentration": "portfolio_model/portfolio_concentration",
    "portfolio_model/portfolio_showcase": "portfolio_model/portfolio_showcase",
    "housing_model/affordability": "housing_model/affordability",
    "saas_model/mrr_dashboard": "saas_model/mrr_dashboard",
    "saas_model/cohort_brief": "saas_model/cohort_brief",
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from tracebi import model_registry, pipeline_registry

    proj = tmp_path_factory.mktemp("ref") / "portfolio_project"
    shutil.copytree(REFERENCE_PROJECT, proj, ignore=shutil.ignore_patterns(
        "data", "output", ".tracebi", "explorations", "__pycache__", "*.duckdb"))
    saved = (os.getcwd(), model_registry._registry, model_registry._auto_discovered,
             pipeline_registry._registry, pipeline_registry._auto_discovered)
    model_registry._registry = model_registry.ModelRegistry()
    model_registry._auto_discovered = False
    pipeline_registry._registry = pipeline_registry.PipelineRegistry()
    pipeline_registry._auto_discovered = False
    os.chdir(proj)
    try:
        for transform in ("holdings_transform", "saas_transform", "affordability_transform"):
            code, out = run_cli("run-transform", transform)
            assert code == 0, out
        for name in REPORTS:
            code, out = run_cli("report", "build", name)
            assert code == 0, f"{name}: {out}"
        yield proj
    finally:
        os.chdir(saved[0])
        (model_registry._registry, model_registry._auto_discovered,
         pipeline_registry._registry, pipeline_registry._auto_discovered) = saved[1:]


def _manifest(proj, name):
    return proj / "output" / f"{name}.html.manifest.json"


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_every_report_reproduces_and_its_file_is_intact(built, name):
    manifest = _manifest(built, name)
    code, out = run_cli("verify", str(manifest), "--contracts")
    assert code == 0, out
    assert "REPRODUCES" in out
    code, out = run_cli("verify", "--file", str(built / "output" / f"{name}.html"))
    assert code == 0, out
    assert "FILE INTACT" in out


def test_the_working_preview_formats_numbers_like_the_build(built):
    """The dev preview and the Build-mode preview render the working state
    (`render_exploration`); a snapshot is the same render, on disk. It must
    carry the model's declared column formats the build embeds, or a table
    that ships as $35,736,396 and 100.4% previews as 35,736,396.18 and 1.00."""
    import re

    name = "portfolio_model/portfolio_overview"
    code, out = run_cli("report", "snapshot", name)
    assert code == 0, out

    def formats(path):
        html = path.read_text(encoding="utf-8")
        block = re.search(
            r'<script id="tracebi-formats" type="application/json">(.*?)</script>', html)
        assert block, f"{path.name} has no tracebi-formats block"
        return json.loads(block.group(1))

    shipped = formats(built / "output" / f"{name}.html")
    assert shipped["top_issuers"] == {
        "cost_basis": "currency0", "fair_value": "currency0", "mark": "percent"}
    assert formats(built / "output" / f"{name}.snapshot.html") == shipped


def test_every_model_keeps_its_reports_in_its_own_folder_and_has_a_pipeline(built):
    """The convention: reports/<model>/ holds a model's reports, and
    pipelines/<model>.py rebuilds its data and then those reports."""
    from tracebi.pipeline import model_pipeline
    from tracebi.pipeline.model_pipeline import reports_of

    models = sorted(p.stem for p in (built / "models").glob("*.yaml") if not p.stem.startswith("_"))
    assert models == ["housing_model", "portfolio_model", "saas_model"]
    for model in models:
        names = reports_of(model)
        assert names, f"reports/{model}/ has no reports"
        assert (built / "pipelines" / f"{model}.yaml").is_file(), f"{model} has no pipeline"
        # ...and every report in the reference project sits under a model's folder
    everything = {name for names in map(reports_of, models) for name in names}
    assert everything == set(REPORTS), "a report sits outside its model's folder"
    assert model_pipeline  # exported at the top level for project pipelines


@pytest.mark.parametrize("model", ["portfolio_model", "housing_model", "saas_model"])
def test_a_models_pipeline_rebuilds_its_data_then_its_reports(built, model):
    from tracebi.pipeline.model_pipeline import reports_of

    code, out = run_cli("run-pipeline", model)
    assert code == 0, out
    assert "transform" in out and "build" in out
    for name in reports_of(model):
        code, out = run_cli("verify", str(_manifest(built, name)), "--contracts")
        assert code == 0 and "REPRODUCES" in out, f"{name}: {out}"


def test_the_showcase_carries_every_affordance(built):
    html = (built / "output" / "portfolio_model" / "portfolio_showcase.html").read_text()
    from tests.test_presentation_js import assert_built_receipt_rows
    assert_built_receipt_rows(html)
    for marker in ("data-tb-filter", "data-tb-search", "data-tb-download",
                   "tb-tabs", "tb-cols-2", 'id="tracebi-receipt"',
                   "tb-methodology", "data-tb-unverified",
                   "tb-semantic-contract-portfolio_model",
                   'id="tracebi-selection"', 'id="tracebi-grain"',
                   "connect-src 'self'", 'class="tb-total"',
                   "data-tb-sort", "data-tb-bars", "data-tb-direction",
                   "tb-table--freeze", "tb-about-receipt",
                   'id="fig-treemap"', "D3 7.9.0 (ISC)"):
        assert marker in html, f"showcase lost its {marker} affordance"
    assert "Working notes" not in html, "exploration must die at build"

    manifest = json.loads(_manifest(built, "portfolio_model/portfolio_showcase").read_text())
    assert manifest["schema_version"] == 2
    assert len(manifest["figures"]) >= 10
    assert manifest["transform_contracts"] and all(
        r.get("status") == "satisfied" for r in manifest["transform_contracts"].values())
    assert manifest["semantic_contract"]["portfolio_model"]["sha256"]
    assert manifest["methodology"]["transform_notes"]


def test_the_showcase_lineage_traces_each_query_to_only_what_it_used(built):
    """The receipt's lineage: each query names exactly the tables it read, a
    python-derived query is called out and never verifiable, and figures with
    no query behind them are shown apart."""
    from tracebi.reports.lineage_flow import report_flow

    manifest = json.loads(_manifest(built, "portfolio_model/portfolio_showcase").read_text())
    flow = report_flow(manifest)
    node = {n["id"]: n for n in flow["nodes"]}

    assert set(node["binding:top_sector"]["detail"]["tables"]) == {"fact_holdings", "dim_issuer"}
    assert set(node["binding:by_fund"]["detail"]["tables"]) == {"fact_holdings", "dim_fund"}
    assert node["binding:concentration"]["status"] == "derived"
    assert node["binding:concentration"]["detail"]["verifiable"] is False
    assert "figures:unverified" in node and flow["notes"][-1].startswith("A query marked python-derived")
    assert node["transform:holdings"]["status"] == "ok"      # the sink satisfied its contract
    # Every figure is on the graph exactly once.
    listed = [f["id"] for n in flow["nodes"] if n["kind"] in ("figures", "unverified")
              for f in n["detail"]["figures"]]
    assert sorted(listed) == sorted(f["id"] for f in manifest["figures"])


def test_a_showcase_control_recomputes_on_the_model(built):
    """Picking a sector re-runs the stamped queries on the model and the cut
    total matches a direct query — interactivity subsets, never computes."""
    from tracebi.model_registry import ModelRegistry
    from tracebi.reports.selection import evaluate_selection
    from tracebi.reports.template_package import TemplatePackage

    previous = os.getcwd()
    os.chdir(built)
    try:
        registry = ModelRegistry()
        registry.auto_discover(str(built / "models"))
        models = {"portfolio_model": registry.get("portfolio_model")}
        package = TemplatePackage(str(built / "reports" / "portfolio_model" / "portfolio_showcase"))
        base = evaluate_selection(package, models, {})
        control = next(c for c in base["controls"] if c["column"] == "dim_issuer.sector")
        assert len(control["included"]) > 1
        sector = control["included"][0]
        cut = evaluate_selection(package, models, {"dim_issuer.sector": sector})
        model = models["portfolio_model"]
        expected = model.query("fact_holdings", ["fair_value", "positions", "mark"],
                               filters={"dim_issuer.sector": sector})
    finally:
        os.chdir(previous)

    def cell(result, figure_id):
        return next(f for f in result["figures"] if f["id"] == figure_id)

    fair, detail = cell(cut, "kpi-fv"), cell(cut, "tbl-detail")
    assert fair["value"] != cell(base, "kpi-fv")["value"]
    assert 0 < len(detail["rows"]) < len(cell(base, "tbl-detail")["rows"])
    assert all(row["dim_issuer.sector"] == sector for row in detail["rows"])
    # The cut is the model's own answer to that query, byte for byte.
    assert fair["fingerprint"] == expected.fingerprint()
    assert cell(cut, "kpi-mark")["fingerprint"] == expected.fingerprint()


def test_the_housing_scenario_is_recorded_never_receipted(built):
    manifest = json.loads(_manifest(built, "housing_model/affordability").read_text())
    assert [s["name"] for s in manifest["scenarios"]] == ["today"]
    assert all(s["verifiable"] is False for s in manifest["scenarios"])
    ids = {f["id"] for f in manifest["figures"]}
    assert {"chart-ten-year", "ten-worst", "ten-best", "ten-today", "chart-paths",
            "chart-rate", "chart-pti", "chart-share"} <= ids
    assert not any(i.startswith("calc-") for i in ids), \
        "a scenario output must never be recorded as a figure"


def test_explore_speaks_the_models_language(built, monkeypatch):
    """Explore queries the model's named measures, offers a fact only the ones
    it can run, and writes the numbers the way a built report does."""
    from tests.e2e.conftest import release_warehouses, serve_app
    from tracebi.registry import registry

    c = serve_app(monkeypatch, models=True)
    try:
        # A measure is declared once on the model; the fact decides whether it
        # runs. Whatever a fact lists, its query answers — and nothing it
        # leaves out would have.
        info = c.get("/api/models/housing_model").json()
        offered = {f["name"]: set(f["runnable_measures"]) for f in info["facts"]}
        assert offered["fact_housing"] != offered["fact_ten_year"]
        for fact, names in offered.items():
            for measure in (m["name"] for m in info["measures"]):
                answer = c.post("/api/models/housing_model/query",
                                json={"fact": fact, "measures": [measure]})
                assert (answer.status_code == 200) == (measure in names), (
                    f"{fact}.{measure}: offered={measure in names}, query answered {answer.status_code}")

        # Likewise a dimension: a fact joins the ones in its foreign_keys, and
        # grouping by any other cannot run.
        joins = {f["name"]: set(f["foreign_keys"]) for f in info["facts"]}
        assert joins["fact_ten_year"] == {"dim_cohort"}
        for fact in offered:
            measure = sorted(offered[fact])[0]
            for dim in info["dimensions"]:
                answer = c.post("/api/models/housing_model/query", json={
                    "fact": fact, "measures": [measure],
                    "dimensions": [f"{dim['name']}.{dim['attributes'][0]}"]})
                assert (answer.status_code == 200) == (dim["name"] in joins[fact]), (
                    f"{fact} x {dim['name']}: joined={dim['name'] in joins[fact]}, "
                    f"query answered {answer.status_code}")

        # The named-measure query: the raw values are the model's own answer.
        body = {"fact": "fact_holdings", "measures": ["fair_value", "positions", "mark"],
                "dimensions": ["dim_issuer.sector"]}
        got = c.post("/api/models/portfolio_model/query", json=body).json()
        direct = registry.get_model("portfolio_model").query(
            "fact_holdings", body["measures"], dimensions=body["dimensions"])
        assert got["data"] == direct.to_pandas().to_dict(orient="records")

        # ...and the display text carries each measure's declared format.
        for raw, shown in zip(got["data"], got["display"]):
            assert shown["dim_issuer.sector"] == raw["dim_issuer.sector"]
            assert re.fullmatch(r"\$[\d,]+", shown["fair_value"]), shown
            assert re.fullmatch(r"\d+\.\d%", shown["mark"]), shown
            assert shown["positions"] == f"{raw['positions']:,}"

        # It is the very text the built overview report prints for that query.
        html = (built / "output" / "portfolio_model" / "portfolio_overview.html").read_text()
        top = c.post("/api/models/portfolio_model/query", json={
            "fact": "fact_holdings", "measures": ["fair_value", "cost_basis", "mark"],
            "dimensions": ["dim_issuer.issuer"], "order_by": ["-fair_value"], "limit": 8}).json()
        assert len(top["display"]) == 8
        for shown in top["display"]:
            for column in ("fair_value", "cost_basis", "mark"):
                assert f'<td class="tb-num">{shown[column]}</td>' in html

        # A result holding NaN (an empty total) or infinity (a division by zero)
        # is answered with missing values, not a 500: JSON has neither.
        empty = c.post("/api/models/portfolio_model/query", json={
            "fact": "fact_holdings", "measures": ["fair_value", "mark"],
            "filters": {"spread_bps": -1}})
        assert empty.status_code == 200
        assert empty.json()["data"] == [{"fair_value": None, "cost_basis": None, "mark": None}]
        assert empty.json()["display"] == [{"fair_value": "", "cost_basis": "", "mark": ""}]
        blown = c.post("/api/models/portfolio_model/query", json={
            "fact": "fact_holdings", "measures": {
                "fair_value": "sum", "inv": {"expr": "1 / (fair_value - fair_value)", "agg": "sum"}}})
        assert blown.status_code == 200
        assert blown.json()["data"][0]["inv"] is None
        assert blown.json()["data"][0]["fair_value"] > 0
        assert blown.json()["display"][0]["inv"] == ""
    finally:
        release_warehouses()

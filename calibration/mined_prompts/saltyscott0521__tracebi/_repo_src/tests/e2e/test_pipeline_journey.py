"""The pipeline journey: ``tracebi new-pipeline`` scaffolds an empty runner,
the analyst fills in a landing layer and a cleaning layer (as the scaffold's
docstring shows), then runs it from the CLI and from the web Pipelines page, and
reads the run history back.
"""

import pytest

from tests.e2e.conftest import follow_run, run_cli, serve_app

LAYERS = '''
from tracebi import CSVConnector, SQLConnector
from tracebi.etl.bronze import BronzeLayer
from tracebi.etl.silver import SilverLayer

_raw = CSVConnector("raw", directory="inputs")
_db = SQLConnector("orders_db", url=_DB_URL)
runner.register(BronzeLayer(connector=_raw, source="orders.csv", sink=_db,
                            sink_table="orders_bronze"), name="orders_bronze")
runner.register(SilverLayer(source=_db, source_table="orders_bronze", sink=_db,
                            sink_table="orders_silver").deduplicate(subset=["order_id"]),
                name="orders_silver", depends_on="orders_bronze")
'''


@pytest.fixture
def pipeline(scaffolded):
    code, out = run_cli("new-pipeline", "--python", "Orders ETL")
    assert code == 0, out
    path = scaffolded / "pipelines" / "orders_etl.py"
    src = path.read_text()
    marker = "runner = PipelineRunner(db_url=_DB_URL)\n"
    assert marker in src, "the scaffold changed shape; update the edit"
    path.write_text(src.replace(marker, marker + LAYERS))
    return scaffolded


def test_a_pipeline_runs_upstream_first_and_records_its_runs(pipeline):
    code, out = run_cli("run-pipeline", "orders_etl")
    assert code == 0, out
    code, out = run_cli("run-pipeline", "orders_etl", "--status")
    assert code == 0, out
    assert "orders_bronze" in out and "orders_silver" in out
    assert out.count("success") >= 2, out

    from tracebi.pipeline_registry import get_runner
    runner = get_runner("orders_etl")
    [landed] = runner.run_history("orders_bronze", limit=1)
    [cleaned] = runner.run_history("orders_silver", limit=1)
    assert landed["status"] == cleaned["status"] == "success"
    # The sample input has twelve rows, one an exact duplicate of order 1009.
    assert (landed["rows_out"], cleaned["rows_out"]) == (12, 11)


def test_the_refresh_page_runs_a_layer_and_shows_its_history(pipeline, monkeypatch):
    from fastapi.testclient import TestClient

    from tracebi.registry import registry
    from tracebi.web import discovery
    from tracebi.web.api.main import app

    for attr in ("_connectors", "_models", "_report_factories",
                 "_scheduled_factories", "_pipelines"):
        monkeypatch.setattr(registry, attr, {})
    monkeypatch.setattr(discovery, "_live_pipelines", set())
    discovery.register_pipelines("pipelines")
    c = TestClient(app)

    listed = c.get("/api/pipelines").json()
    assert sorted(p["pipeline"] for p in listed) == ["orders_etl", "sample_model"]   # init's own, and this one
    run = c.post("/api/pipelines/orders_etl/layers/orders_bronze/run")
    assert run.status_code == 200, run.text
    history = c.get("/api/pipelines/orders_etl/layers/orders_bronze/history").json()
    assert history["runs"] and history["runs"][0]["status"] == "success"


def test_the_build_step_will_not_rebuild_reports_after_a_failed_transform(scaffolded):
    """`run-pipeline` keeps going after a failure to report them all; the
    model pipeline's build must not turn that into a green rebuild on stale data."""
    code, out = run_cli("run-pipeline", "sample_model")
    assert code == 0, out
    assert (scaffolded / "output" / "sample_model" / "sample_dashboard.html").is_file()

    (scaffolded / "transforms" / "sample_transform.py").write_text("raise SystemExit('broken')\n")   # a script that exits, not raises
    code, out = run_cli("run-pipeline", "sample_model")
    assert code != 0
    assert "transform" in out and "not rebuilding reports" in out


def test_the_refresh_page_follows_a_run_as_a_log(pipeline, monkeypatch):
    c = serve_app(monkeypatch)

    started = c.post("/api/pipelines/orders_etl/runs")
    assert started.status_code == 202, started.text
    run = started.json()
    assert run["status"] == "running" and run["layers"] == ["orders_bronze", "orders_silver"]

    text, done = follow_run(c, "orders_etl", run["run_id"])
    assert done["status"] == "succeeded", text
    # What the runner printed, in the order it ran, each line timed.
    assert text.index("[orders_bronze] Running") < text.index("[orders_bronze] ✓  12 in → 12 out")
    assert "[orders_silver] ✓  12 in → 11 out" in text
    assert text.rstrip().splitlines()[-1].endswith("Run succeeded in " + text.rstrip().rsplit(" in ", 1)[1])
    # Asking again from the end of the log finds nothing new, and says it is done.
    again = c.get(f"/api/pipelines/orders_etl/runs/{run['run_id']}/log", params={"after": done["next"]}).json()
    assert (again["text"], again["done"]) == ("", True)

    # The layers recorded their runs as they always do, and the run is in the store.
    [landed] = c.get("/api/pipelines/orders_etl/layers/orders_bronze/history").json()["runs"][:1]
    assert landed["status"] == "success"
    assert [r["run_id"] for r in c.get("/api/pipelines/orders_etl/runs").json()] == [run["run_id"]]
    assert [r["status"] for r in c.get("/api/runs", params={"kind": "pipeline_run"}).json()] == ["succeeded"]

    # One layer, from its node on the flow.
    one = c.post("/api/pipelines/orders_etl/runs", params={"layer": "orders_silver"}).json()
    assert one["layers"] == ["orders_silver"]
    text, done = follow_run(c, "orders_etl", one["run_id"])
    assert done["status"] == "succeeded" and "[orders_bronze]" not in text


def test_a_run_from_the_command_line_is_in_the_same_list_as_one_from_the_app(pipeline, monkeypatch):
    """`tracebi run-pipeline` used to leave nothing the Refresh page could show:
    a cron job's run was invisible beside the ones started from a button."""
    code, out = run_cli("run-pipeline", "orders_etl")
    assert code == 0, out
    c = serve_app(monkeypatch)

    [ran] = c.get("/api/pipelines/orders_etl/runs").json()
    assert ran["status"] == "succeeded" and ran["layers"] == ["orders_bronze", "orders_silver"]
    assert ran["actor_role"] == "cli" and ran["finished_at"]
    text, done = follow_run(c, "orders_etl", ran["run_id"])
    assert done["status"] == "succeeded" and not done["expired"]
    assert "[orders_silver] ✓  12 in → 11 out" in text
    assert text.rstrip().splitlines()[-1].split("  ", 1)[1].startswith("Run succeeded in")
    assert "[orders_silver] ✓" in out                       # the terminal still gets what it always did

    # `--status` runs nothing, so it records nothing; one layer is a run of its own.
    assert run_cli("run-pipeline", "orders_etl", "--status")[0] == 0
    assert run_cli("run-pipeline", "orders_etl", "--layer", "orders_silver")[0] == 0
    # …and a run from the app lands in the same list, newest first.
    from_app = c.post("/api/pipelines/orders_etl/runs").json()
    follow_run(c, "orders_etl", from_app["run_id"])
    runs = c.get("/api/pipelines/orders_etl/runs").json()
    assert [r["layers"] for r in runs] == [["orders_bronze", "orders_silver"], ["orders_silver"],
                                           ["orders_bronze", "orders_silver"]]
    assert [r["run_id"] for r in runs][0] == from_app["run_id"] and runs[2]["run_id"] == ran["run_id"]


def test_a_failed_command_line_run_is_recorded_as_failed_and_still_runs_the_rest(pipeline, monkeypatch):
    path = pipeline / "pipelines" / "orders_etl.py"
    path.write_text(path.read_text() + '''
def _boom():
    raise ValueError("the warehouse said no")

runner.register_step("boom", _boom, depends_on="orders_silver")
''')
    code, out = run_cli("run-pipeline", "orders_etl")
    assert code == 1 and "[tracebi] boom FAILED: ValueError: the warehouse said no" in out
    c = serve_app(monkeypatch)

    [ran] = c.get("/api/pipelines/orders_etl/runs").json()
    assert ran["status"] == "failed" and "the warehouse said no" in ran["error"]["message"]
    text, done = follow_run(c, "orders_etl", ran["run_id"])
    assert "[orders_silver] ✓" in text and "[tracebi] boom FAILED" in text    # what ran before it, and the failure
    assert text.rstrip().splitlines()[-1].split("  ", 1)[1].startswith("Run failed in")


def test_a_command_line_run_that_is_interrupted_does_not_stay_running(pipeline, monkeypatch):
    """A run stuck at 'running' would keep the Refresh page's Run all disabled for good."""
    path = pipeline / "pipelines" / "orders_etl.py"
    path.write_text(path.read_text() + '''
def _interrupt():
    raise KeyboardInterrupt

runner.register_step("interrupt", _interrupt, depends_on="orders_silver")
''')
    with pytest.raises(KeyboardInterrupt):
        run_cli("run-pipeline", "orders_etl")
    [ran] = serve_app(monkeypatch).get("/api/pipelines/orders_etl/runs").json()
    assert ran["status"] == "failed" and "KeyboardInterrupt" in ran["error"]["exception_type"]


def test_a_refresh_from_the_command_line_does_not_need_the_run_record(pipeline, monkeypatch):
    """The record is a convenience: a state store that cannot be reached must not stop the data refreshing."""
    (pipeline / "blocked").write_text("a file where the store's folder should be")
    monkeypatch.setenv("TRACEBI_STATE_URL", f"sqlite:///{pipeline / 'blocked' / 'state.db'}")
    code, out = run_cli("run-pipeline", "orders_etl")
    assert code == 0, out
    assert "this run is not being recorded" in out and "[orders_silver] ✓" in out


def test_a_failed_run_says_where_and_why_and_a_second_click_joins_the_first(pipeline, monkeypatch):
    path = pipeline / "pipelines" / "orders_etl.py"
    path.write_text(path.read_text() + '''
def _boom():
    import os, time
    print("waiting at the gate")
    give_up = time.monotonic() + 20           # a failing test must not leave this spinning
    while not os.path.exists("gate.txt") and time.monotonic() < give_up:
        time.sleep(0.02)
    raise ValueError("the warehouse said no")

runner.register_step("boom", _boom, depends_on="orders_silver")
''')
    c = serve_app(monkeypatch)

    first = c.post("/api/pipelines/orders_etl/runs").json()
    second = c.post("/api/pipelines/orders_etl/runs").json()
    assert second["run_id"] == first["run_id"] and second["already_running"] is True

    (pipeline / "gate.txt").write_text("open")
    text, done = follow_run(c, "orders_etl", first["run_id"])
    assert done["status"] == "failed"
    assert "waiting at the gate" in text                    # a step's own print() is in the log
    assert "[boom] ✗  the warehouse said no" in text and text.rstrip().splitlines()[-1].split("  ", 1)[1].startswith("Run failed in")
    status = c.get(f"/api/pipelines/orders_etl/runs/{first['run_id']}").json()
    assert "the warehouse said no" in status["error"]["message"]
    # The failed run is over, so the next click starts a fresh one.
    again = c.post("/api/pipelines/orders_etl/runs", params={"layer": "orders_bronze"}).json()
    assert again["run_id"] != first["run_id"] and again["already_running"] is False
    follow_run(c, "orders_etl", again["run_id"])
    # Nothing here is another pipeline's run.
    assert c.get(f"/api/pipelines/orders_etl/runs/{first['run_id']}/log").status_code == 200
    assert c.get("/api/pipelines/orders_etl/runs/nope/log").status_code == 404
    assert c.post("/api/pipelines/nope/runs").status_code == 404
    assert c.post("/api/pipelines/orders_etl/runs", params={"layer": "nope"}).status_code == 404


def test_a_model_pipelines_log_includes_what_its_transform_printed(scaffolded, monkeypatch):
    c = serve_app(monkeypatch)

    # A fresh project's Refresh page has a pipeline for its model.
    [pipeline] = c.get("/api/pipelines").json()
    assert (pipeline["pipeline"], pipeline["model"]) == ("sample_model", "sample_model")
    # What the model's Overview says about it: the transform it runs, and its description.
    assert (pipeline["transform"], pipeline["description"]) == (
        "sample_transform", "Rebuild the sample warehouse, then its reports.")
    run = c.post("/api/pipelines/sample_model/runs").json()
    text, done = follow_run(c, "sample_model", run["run_id"])
    assert done["status"] == "succeeded", text
    # The transform ran through `tracebi run-transform`, whose own output used to be thrown away.
    assert "Running transforms/sample_transform.py" in text
    # Paths read as the project's own, not the machine's.
    assert "→ output/sample_model/sample_dashboard.html\n" in text
    assert text.index("[transform]") < text.index("[build]")
    # The run says which reports it rebuilt, for the page to link to.
    assert c.get(f"/api/pipelines/sample_model/runs/{run['run_id']}").json()["reports"] == [
        "sample_model/sample_dashboard"]

    # The Code tab shows what ran: the pipeline file, then the transform it starts with.
    code = c.get("/api/pipelines/sample_model/source").json()
    assert [(f["label"], f["path"]) for f in code["files"]] == [
        ("pipeline", "pipelines/sample_model.yaml"), ("transform", "transforms/sample_transform.py")]
    assert "transform: sample_transform" in code["files"][0]["content"] and "sink" in code["files"][1]["content"].lower()
    assert c.get("/api/pipelines/nope/source").status_code == 404


def test_a_run_lists_only_the_reports_it_actually_rebuilt(scaffolded, monkeypatch):
    folder = scaffolded / "reports" / "sample_model"               # init's layout: the model's reports
    (folder / "zz_broken").mkdir()                                  # built after the good one, and fails
    (folder / "zz_broken" / "report.json").write_text("{not json")
    c = serve_app(monkeypatch)

    def run(layer=None):
        started = c.post("/api/pipelines/sample_model/runs", params={"layer": layer} if layer else {}).json()
        follow_run(c, "sample_model", started["run_id"])
        return c.get(f"/api/pipelines/sample_model/runs/{started['run_id']}").json()

    stopped = run()
    assert stopped["status"] == "failed" and stopped["reports"] == ["sample_model/sample_dashboard"]
    # A run that never reached the build step rebuilt nothing.
    assert run("transform")["reports"] == []


def test_the_app_refreshes_a_model_while_another_model_holds_the_same_warehouse_open(reference, monkeypatch):
    """portfolio_model and saas_model sink into one warehouse file. Listing the
    models in the app opens it read-only for both; a refresh of one must still
    be able to rewrite it."""
    for transform in ("holdings_transform", "saas_transform", "affordability_transform"):
        code, out = run_cli("run-transform", transform)
        assert code == 0, out
    c = serve_app(monkeypatch, models=True)
    assert c.get("/api/models").status_code == 200

    run = c.post("/api/pipelines/portfolio_model/runs").json()
    text, done = follow_run(c, "portfolio_model", run["run_id"])
    assert done["status"] == "succeeded", text
    assert "[transform] ✓" in text and "[build] ✓" in text
    # Every report of the model, and none of another model's.
    from tracebi.pipeline.model_pipeline import reports_of
    rebuilt = c.get(f"/api/pipelines/portfolio_model/runs/{run['run_id']}").json()["reports"]
    assert len(rebuilt) > 1 and sorted(rebuilt) == reports_of("portfolio_model")


def test_a_model_and_its_connector_show_the_file_that_declares_them(reference, monkeypatch):
    from tracebi import DataModel
    from tracebi.registry import registry

    c = serve_app(monkeypatch, models=True)
    code = c.get("/api/models/portfolio_model/source").json()
    assert [f["path"] for f in code["files"]] == ["models/portfolio_model.yaml"]
    assert "measures:" in code["files"][0]["content"] and not code["files"][0]["truncated"]
    # A connector is declared in the model files that read it: both models here read this warehouse.
    shared = c.get("/api/connectors/warehouse/source").json()
    assert sorted(f["path"] for f in shared["files"]) == ["models/portfolio_model.yaml", "models/saas_model.yaml"]
    # A model registered in code has no file, and says so rather than showing nothing.
    registry.add_model(DataModel("in_code"))
    nothing = c.get("/api/models/in_code/source").json()
    assert nothing["files"] == [] and "Python code" in nothing["hint"]
    assert c.get("/api/models/nope/source").status_code == 404

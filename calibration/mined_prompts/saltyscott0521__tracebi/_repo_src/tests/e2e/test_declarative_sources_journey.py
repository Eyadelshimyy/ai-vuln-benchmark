"""Journey: sources and pipelines are YAML. A scaffolded project reads its
warehouse through connections/warehouse.yaml, a YAML model and a YAML pipeline;
secrets stay out of the files and out of the API."""

from __future__ import annotations

import json

import pytest

from .conftest import follow_run, run_cli

SECRET = "hunter2-SECRETVALUE-9f3"


def serve(monkeypatch):
    """The web app over connections, models and pipelines, as the server starts."""
    from fastapi.testclient import TestClient

    from tracebi.registry import registry
    from tracebi.web import discovery
    from tracebi.web.api.main import app

    for attr in ("_connectors", "_models", "_report_factories",
                 "_scheduled_factories", "_pipelines"):
        monkeypatch.setattr(registry, attr, {})
    for attr in ("_live_pipelines", "_live_models"):
        monkeypatch.setattr(discovery, attr, set())
    monkeypatch.setattr(discovery, "_live_connections", {})
    discovery.register_connections("connections")
    discovery.register_models("models")
    discovery.register_pipelines("pipelines")
    return TestClient(app)


def test_a_scaffolded_project_runs_end_to_end_on_yaml_sources(scaffolded, monkeypatch):
    for rel in ("connections/warehouse.yaml", "models/sample_model.yaml", "pipelines/sample_model.yaml"):
        assert (scaffolded / rel).is_file(), rel
    assert not list((scaffolded / "models").glob("*.py")) and not list((scaffolded / "pipelines").glob("*.py"))

    code, out = run_cli("run-pipeline", "sample_model")
    assert code == 0, out
    assert "[tracebi] 2 layer(s) completed." in out
    assert (scaffolded / "output" / "sample_model" / "sample_dashboard.html").is_file()

    c = serve(monkeypatch)
    # The Refresh page runs the YAML pipeline the way the command line did.
    run = c.post("/api/pipelines/sample_model/runs").json()
    text, done = follow_run(c, "sample_model", run["run_id"])
    assert done["status"] == "succeeded", text
    assert text.index("[transform]") < text.index("[build]")
    assert c.get(f"/api/pipelines/sample_model/runs/{run['run_id']}").json()["reports"] == [
        "sample_model/sample_dashboard"]
    code = c.get("/api/pipelines/sample_model/source").json()
    assert [f["path"] for f in code["files"]] == [
        "pipelines/sample_model.yaml", "transforms/sample_transform.py"]

    # The Sources page lists the declared connection, with where it lives.
    [conn] = [x for x in c.get("/api/connectors").json() if x["name"] == "warehouse"]
    assert conn["declared_in"] == "connections/warehouse.yaml"
    assert conn["storage"]["kind"] == "file" and conn["storage"]["where"].endswith("warehouse.duckdb")
    assert conn["used_by"] == ["sample_model"]
    shown = c.get("/api/connectors/warehouse/source").json()
    assert shown["files"][0]["path"] == "connections/warehouse.yaml"
    assert c.get("/api/connectors/warehouse/warehouse").json()["supported"] is True


def test_a_literal_secret_in_a_connection_is_refused(scaffolded, monkeypatch):
    conns = scaffolded / "connections"
    (conns / "snow.yaml").write_text(
        f"name: snow\ntype: snowflake\naccount: a\nuser: u\npassword: {SECRET}\nwarehouse: w\ndatabase: d\n")
    (conns / "pg.yaml").write_text(f"name: pg\ntype: postgres\nurl: postgresql://u:{SECRET}@h/db\n")
    (conns / "ok.yaml").write_text("name: ok\ntype: sql\nurl: sqlite:///data/ok.db\n")   # no credential: fine

    code, out = run_cli("validate")
    assert code == 1
    assert "snow.yaml" in out and "password: a credential" in out
    assert "pg.yaml" in out and "url: carries a credential" in out
    assert "ok.yaml is well formed" in out
    assert SECRET not in out                       # even the refusal does not echo it

    c = serve(monkeypatch)
    names = [x["name"] for x in c.get("/api/connectors").json()]
    assert "ok" in names and "snow" not in names and "pg" not in names
    failed = {e["file"]: e for e in c.get("/api/discovery").json()["entries"]
              if e["directory"].endswith("connections")}
    assert failed["snow.yaml"]["status"] == "failed" and failed["pg.yaml"]["status"] == "failed"
    assert SECRET not in json.dumps(failed)


def test_an_unset_variable_fails_when_the_connection_is_used_not_when_it_is_found(scaffolded, monkeypatch):
    monkeypatch.delenv("TRACEBI_ELSEWHERE_DB", raising=False)
    (scaffolded / "connections" / "elsewhere.yaml").write_text(
        "name: elsewhere\ntype: duckdb\ndatabase: ${TRACEBI_ELSEWHERE_DB}\n")
    (scaffolded / "models" / "remote.yaml").write_text(
        "name: remote\nconnectors:\n  - {name: elsewhere, connection: elsewhere}\n"
        "tables:\n  - {name: t, connector: elsewhere, source: t}\n")

    # Discovery loads both without complaint, and the Sources page lists the connection.
    c = serve(monkeypatch)
    entries = {e["file"]: e["status"] for e in c.get("/api/discovery").json()["entries"]}
    assert entries.get("elsewhere.yaml") != "failed"
    assert "elsewhere" in [x["name"] for x in c.get("/api/connectors").json()]
    assert c.get("/api/connectors/elsewhere").status_code == 200

    # Using it is what fails, and the message names the variable.
    from tracebi import model_registry
    model = model_registry.get_model("remote")
    with pytest.raises(Exception, match="TRACEBI_ELSEWHERE_DB"):
        model.load("t")
    # Setting it in .env (never loaded into the process environment) is enough.
    other = scaffolded / "other.duckdb"
    import duckdb
    con = duckdb.connect(str(other))
    con.execute("create table t as select 1 as x")
    con.close()
    (scaffolded / ".env").write_text(f'TRACEBI_ELSEWHERE_DB="{other}"\n')
    assert len(model_registry.get_model("remote").load("t").to_pandas()) == 1
    import os
    assert "TRACEBI_ELSEWHERE_DB" not in os.environ


def test_the_connectors_api_never_shows_a_resolved_secret(scaffolded, monkeypatch):
    monkeypatch.setenv("TRACEBI_PG_URL", f"postgresql://analyst:{SECRET}@db.internal:5432/sales")
    monkeypatch.setenv("TRACEBI_SNOW_PASSWORD", SECRET)
    (scaffolded / "connections" / "pg.yaml").write_text(
        "name: pg\ntype: postgres\nurl: ${TRACEBI_PG_URL}\n")
    (scaffolded / "connections" / "snow.yaml").write_text(
        "name: snow\ntype: snowflake\naccount: acme\nuser: u\npassword: ${TRACEBI_SNOW_PASSWORD}\n"
        "warehouse: W\ndatabase: D\n")
    (scaffolded / "connections" / "lit.yaml").write_text(
        "name: lit\ntype: sql\nurl: postgresql://reader@db.internal/sales\n")
    c = serve(monkeypatch)

    bodies = [c.get("/api/connectors").text]
    for name in ("pg", "snow", "lit"):
        bodies += [c.get(f"/api/connectors/{name}").text, c.get(f"/api/connectors/{name}/source").text,
                   c.get(f"/api/connectors/{name}/warehouse").text]
    for body in bodies:
        assert SECRET not in body
    listed = {x["name"]: x for x in c.get("/api/connectors").json()}
    assert listed["pg"]["url"] == "${TRACEBI_PG_URL}"            # the reference, never the value
    assert listed["snow"]["storage"]["where"].startswith("Snowflake acme")
    assert "db.internal" not in c.get("/api/connectors/pg").text  # the host came from the secret too
    assert listed["lit"]["url"] == "postgresql://reader@db.internal/sales"   # a literal, no credential


def test_a_python_pipeline_beats_a_yaml_one_of_the_same_name(scaffolded, monkeypatch):
    from tracebi import pipeline_registry

    (scaffolded / "pipelines" / "sample_model.py").write_text(
        "from tracebi import model_pipeline\n"
        "runner = model_pipeline('sample_model', transform='sample_transform')\n")
    (scaffolded / "pipelines" / "twin.yaml").write_text(
        "name: twin\ntransform: sample_transform\nmodels: [sample_model]\n")
    (scaffolded / "pipelines" / "twin.yml").write_text(
        "name: twin\ntransform: sample_transform\nmodels: [sample_model]\n")

    c = serve(monkeypatch)
    assert pipeline_registry.pipeline_path("sample_model").endswith("sample_model.py")
    assert "twin" not in [p["pipeline"] for p in c.get("/api/pipelines").json()]
    refused = {e["file"]: e["reason"] for e in c.get("/api/discovery").json()["entries"]
               if e["status"] == "failed" and e["directory"].endswith("pipelines")}
    assert "sample_model.py exists" in refused["sample_model.yaml"]
    assert "both exist" in refused["twin.yaml"] and "both exist" in refused["twin.yml"]
    assert run_cli("run-pipeline", "sample_model")[0] == 0           # the Python file ran


def test_a_bad_pipeline_file_says_what_is_wrong(scaffolded, monkeypatch):
    (scaffolded / "pipelines" / "bad.yaml").write_text(
        "name: other\ntransform: ../evil\nmodels: []\nschedule: soon\nextra: 1\n")
    code, out = run_cli("validate")
    assert code == 1
    for part in ("unknown key 'extra'", "transform:",
                 "models: required", "schedule: must be a 5-field cron"):
        assert part in out or part in " ".join(out.split()), part
    c = serve(monkeypatch)
    [entry] = [e for e in c.get("/api/discovery").json()["entries"] if e["file"] == "bad.yaml"]
    assert entry["status"] == "failed"


def test_a_scheduled_pipeline_runs_the_whole_chain(scaffolded):
    pytest.importorskip("apscheduler")
    (scaffolded / "pipelines" / "nightly.yaml").write_text(
        "name: nightly\ntransform: sample_transform\nmodels: [sample_model]\n"
        "reports: [sample_model/sample_dashboard]\nschedule: '0 6 * * *'\n")
    from tracebi.pipeline_registry import get_runner

    runner = get_runner("nightly")
    runner.start(blocking=False)
    try:
        job = runner._scheduler.get_job("build")
        job.func()                                  # what the cron tick calls
    finally:
        runner.stop()
    assert runner.last_run("transform")["status"] == "success"
    assert runner.last_run("build")["status"] == "success"
    assert (scaffolded / "output" / "sample_model" / "sample_dashboard.html").is_file()


def test_migrating_a_connection_and_a_pipeline(scaffolded, monkeypatch):
    # tracebi connect --python writes the legacy module; its secret is in .env.
    code, out = run_cli("connect", "--python", "pg", "--kind", "postgres", "--no-test",
                        "--url", f"postgresql://u:{SECRET}@h/db")
    assert code == 0, out
    code, out = run_cli("migrate", "connection", "models/_connections/pg.py")
    assert code == 0, out
    assert "url: ${TRACEBI_PG_URL}" in out and SECRET not in out
    code, out = run_cli("migrate", "connection", "models/_connections/pg.py", "--write")
    assert code == 0, out
    assert (scaffolded / "connections" / "pg.yaml").is_file()
    assert (scaffolded / "models" / "_connections" / "pg.py").is_file()      # never deleted
    # The legacy secret still resolves through the new file (from .env, not os.environ).
    monkeypatch.delenv("TRACEBI_PG_URL", raising=False)
    from tracebi.connections import load_connection
    assert load_connection("connections/pg.yaml", scaffolded).describe()["url"] == "${TRACEBI_PG_URL}"

    # A module that is not a declared type is refused, with the reason.
    (scaffolded / "models" / "_connections" / "mem.py").write_text(
        "import pandas as pd\nfrom tracebi import MemoryConnector\n"
        "connector = MemoryConnector('mem', {'t': pd.DataFrame({'a': [1]})})\n")
    code, out = run_cli("migrate", "connection", "models/_connections/mem.py")
    assert code == 1 and "MemoryConnector has no connection type" in out

    # A pipeline: model_pipeline converts; anything else is refused.
    (scaffolded / "pipelines" / "sample_model.yaml").rename(scaffolded / "sample_model.pipeline.old")
    (scaffolded / "pipelines" / "sample_model.py").write_text(
        "from tracebi import model_pipeline\n"
        "runner = model_pipeline('sample_model', transform='sample_transform')\n")
    code, out = run_cli("migrate", "pipeline", "pipelines/sample_model.py")
    assert code == 0, out
    assert "transform: sample_transform" in out and "models: [sample_model]" in out
    (scaffolded / "pipelines" / "layered.py").write_text(
        "from tracebi import PipelineRunner\nrunner = PipelineRunner(db_url='sqlite:///data/l.db')\n")
    code, out = run_cli("migrate", "pipeline", "pipelines/layered.py")
    assert code == 1 and "not built by model_pipeline" in out

"""One ``tracebi_runs`` table, upgraded in place from a database today's code wrote."""

import sqlite3
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect

from tracebi.audit import actor
from tracebi.pipeline.runner import PipelineRunner
from tracebi.state import list_runs, record_run, schedule_records, upgrade

# The table ``_init_db`` left behind before Alembic: actor columns already
# reconciled, none of the shared-store columns yet.
_TODAY = """CREATE TABLE tracebi_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    layer_name TEXT NOT NULL,
    layer_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    rows_in INTEGER,
    rows_out INTEGER,
    upstream_run_id INTEGER,
    error_message TEXT,
    actor TEXT,
    actor_role TEXT
)"""


def test_upgrade_of_todays_sqlite_keeps_history(tmp_path):
    db = tmp_path / "today.db"
    con = sqlite3.connect(db)
    con.execute(_TODAY)
    con.execute(
        "INSERT INTO tracebi_runs"
        " (layer_name, layer_type, started_at, status, actor, actor_role)"
        " VALUES ('orders', 'bronze', '2024-01-01T00:00:00+00:00', 'success',"
        "         'ada', 'admin')"
    )
    con.commit()
    con.close()

    url = f"sqlite:///{db}"
    upgrade(url)
    upgrade(url)
    runner = PipelineRunner(db_url=url)
    cols = [c["name"] for c in inspect(runner._engine_()).get_columns("tracebi_runs")]
    assert cols.count("kind") == 1
    assert cols.count("actor") == 1
    row = runner.run_history("orders", limit=1)[0]
    assert row["layer_name"] == "orders"
    assert row["actor"] == "ada"
    assert row["actor_role"] == "admin"
    assert row["status"] == "success"
    assert row["kind"] is None


def test_a_layer_run_is_tagged_and_a_schedule_row_is_not_its_history(tmp_path):
    url = f"sqlite:///{tmp_path / 'runs.db'}"
    record_run(
        kind="schedule", target="step", status="delivered", url=url,
        detail={"report": "step", "status": "delivered"},
    )
    runner = PipelineRunner(db_url=url)
    runner.register_step("step", lambda: 2)
    runner.run("step")
    history = runner.run_history("step")
    assert len(history) == 1
    assert history[0]["kind"] == "pipeline_layer"
    assert history[0]["target"] == "step"
    assert history[0]["finished"] == history[0]["completed_at"]
    assert history[0]["status"] == "success"
    listed = list_runs(kind="schedule", target="step", url=url)
    assert len(listed) == 1
    assert listed[0]["kind"] == "schedule"


def test_memory_runner_records_a_run():
    runner = PipelineRunner(db_url="sqlite:///:memory:")
    runner.register_step("step", lambda: 1)
    runner.run("step")
    assert runner.last_run("step")["rows_out"] == 1


def test_a_report_build_is_recorded(tmp_path, monkeypatch):
    pkg = tmp_path / "reports" / "weekly"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text("{}", encoding="utf-8")
    (pkg / "template.html").write_text("<p></p>", encoding="utf-8")
    out = tmp_path / "output" / "weekly.html"

    class _Package:
        def __init__(self, _path):
            pass

        def render(self, _models, output, **_k):
            from pathlib import Path
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text("<html></html>", encoding="utf-8")
            return None

    monkeypatch.setattr(
        "tracebi.reports.template_package.TemplatePackage", _Package)
    from tracebi.cli import _build_report_target
    _build_report_target("package", pkg, out, report_name="weekly")
    rows = list_runs(kind="report_build", target="weekly")
    assert len(rows) == 1
    assert rows[0]["status"] == "succeeded"
    assert rows[0]["output_path"] == str(out)
    assert rows[0]["verdict"] is None
    assert rows[0]["detail"]["manifest_path"].endswith(".manifest.json")


def test_get_runs_filters_kind_and_target():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.web.api.routers import runs as runs_router

    record_run(kind="schedule", target="weekly", status="built",
               detail={"report": "weekly"})
    record_run(kind="report_build", target="weekly", status="succeeded")
    record_run(kind="schedule", target="other", status="failed",
               detail={"report": "other"})
    app = FastAPI()
    app.include_router(runs_router.router, prefix="/api")
    client = TestClient(app)
    body = client.get("/api/runs", params={"kind": "schedule", "target": "weekly",
                                           "limit": 10}).json()
    assert [row["target"] for row in body] == ["weekly"]
    assert body[0]["kind"] == "schedule"


def test_schedule_import_is_idempotent_and_scoped(tmp_path):
    url = f"sqlite:///{tmp_path / 'sched.db'}"
    out = tmp_path / "output"
    out.mkdir()
    (out / "schedule_runs.jsonl").write_text(
        '{"report":"weekly","status":"built","started_at":"t0",'
        '"finished_at":"t1","actor":null}\n'
        "not json\n"
        "{}\n",
        encoding="utf-8",
    )
    with actor("alice", role="admin"):
        first = schedule_records(out, url=url)
        again = schedule_records(out, url=url)
    assert first == again
    assert first == [{
        "report": "weekly",
        "status": "built",
        "started_at": "t0",
        "finished_at": "t1",
        "actor": None,
    }]
    other = tmp_path / "other"
    other.mkdir()
    assert schedule_records(other, url=url) == []
    assert schedule_records(tmp_path / "missing", url=url) == []


def _block_store(monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    monkeypatch.setitem(sys.modules, "alembic", None)


class _QuietPackage:
    def __init__(self, _path):
        pass

    def render(self, _models, output, **_k):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text("<html>ok</html>", encoding="utf-8")
        Path(str(output) + ".manifest.json").write_text("{}", encoding="utf-8")

        class _Manifest:
            def to_dict(self):
                return {"figures": [], "embedded_data": [],
                        "transform_contracts": {}}

        return _Manifest()


def _weekly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = tmp_path / "reports" / "weekly"
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text("{}", encoding="utf-8")
    (pkg / "template.html").write_text("<p></p>", encoding="utf-8")
    monkeypatch.setattr(
        "tracebi.reports.template_package.TemplatePackage", _QuietPackage)
    _block_store(monkeypatch)


def test_report_build_succeeds_when_the_store_is_missing(tmp_path, monkeypatch, capsys):
    _weekly(tmp_path, monkeypatch)
    from tracebi.cli import main
    assert main(["report", "build", "weekly"]) == 0
    assert (tmp_path / "output" / "weekly.html").read_text(encoding="utf-8") == "<html>ok</html>"
    err = capsys.readouterr().err
    assert "report build was not recorded:" in err
    assert "tracebi[pipeline]" in err


def test_build_report_succeeds_when_the_store_is_missing(tmp_path, monkeypatch):
    _weekly(tmp_path, monkeypatch)
    from tracebi.mcp_server import gateway_build_report
    result = gateway_build_report("weekly", output_dir="output")
    assert result["ok"] is True
    assert "errors" not in result
    assert "report build was not recorded:" in result["note"]
    assert "tracebi[pipeline]" in result["note"]
    assert (tmp_path / "output" / "weekly.html").is_file()

    monkeypatch.setattr(
        "tracebi.reports.pdf.print_pdf",
        lambda _html, pdf: Path(pdf).write_bytes(b"%PDF-1.4\n"),
    )
    pdf = gateway_build_report("weekly", output_dir="output", format="pdf")
    assert pdf["ok"] is True
    assert "errors" not in pdf
    assert "report build was not recorded:" in pdf["note"]
    assert (tmp_path / "output" / "weekly.pdf").is_file()


def test_a_web_read_names_the_web_extra(monkeypatch):
    _block_store(monkeypatch)
    from tracebi.web.api.routers.runs import get_runs
    with pytest.raises(ImportError, match=r"tracebi\[web\]"):
        get_runs()


def test_a_schedule_read_names_the_pipeline_extra(tmp_path, monkeypatch):
    _block_store(monkeypatch)
    from tracebi.schedule import last_runs
    with pytest.raises(ImportError, match=r"tracebi\[pipeline\]"):
        last_runs(tmp_path)

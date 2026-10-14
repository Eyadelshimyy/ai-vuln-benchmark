"""
Reports in folders (epic E6, step 1).

A report in a folder is named by its path below ``reports/``
(``finance/weekly_summary``), so two folders can each hold a report of the same
name. The done-when from docs/strategy/epics.md: two folders each hold a
``weekly_summary``; both are discovered, build, open and schedule
independently, and nothing a name turns into can climb out of ``reports/`` or
``output/``.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.report_paths import output_html, report_name_error

_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body><h1>{title}</h1>
<span data-tb-figure="value" data-tb-binding="totals" data-tb-cell="revenue"
      id="total"></span>
</body></html>
"""


def _package(root, rel, title, region, schedule=None):
    pkg = root / "reports" / rel
    pkg.mkdir(parents=True)
    decl = {"name": title, "data": {"totals": {"model": "folder_model", "query": {
        "fact": "fact_orders", "measures": ["revenue"],
        "filters": {"dim_region.region": region}}}}}
    if schedule:
        decl["schedule"] = schedule
    (pkg / "report.json").write_text(json.dumps(decl), encoding="utf-8")
    (pkg / "template.html").write_text(_TEMPLATE.format(title=title), encoding="utf-8")


@pytest.fixture
def model():
    m = DataModel("folder_model").add_connector(MemoryConnector("fm", tables={
        "orders": pd.DataFrame({"order_id": [1, 2, 3], "region_id": [1, 1, 2],
                                "revenue": [100.0, 50.0, 7.0]}),
        "regions": pd.DataFrame({"region_id": [1, 2], "region": ["East", "West"]}),
    }))
    m.add_table("orders", connector="fm", source="orders")
    m.add_table("regions", connector="fm", source="regions")
    m.add_dimension("dim_region", table_name="regions", key_col="region_id",
                    attributes=["region"])
    m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
               foreign_keys={"dim_region": "region_id"})
    m.add_measure("revenue", column="revenue", agg="sum")
    m.connect()
    return m


@pytest.fixture
def project(tmp_path, monkeypatch, model):
    """Two folders each holding a weekly_summary, plus one top-level report."""
    from tracebi import cli, model_registry

    _package(tmp_path, "finance/weekly_summary", "Finance weekly", "East",
             schedule={"cron": "0 7 * * MON"})
    _package(tmp_path, "ops/weekly_summary", "Ops weekly", "West")
    _package(tmp_path, "top_level", "Top level", "East")
    (tmp_path / "reports" / "finance" / "notes.txt").write_text("not a report")
    # A JSON spec in a folder, too (specs compile to a package at discovery).
    (tmp_path / "reports" / "finance" / "by_region.json").write_text(json.dumps({
        "name": "Finance by region", "sections": [{
            "type": "table", "title": "Revenue by region",
            "data": {"model": "folder_model", "query": {
                "fact": "fact_orders", "measures": ["revenue"],
                "dimensions": ["dim_region.region"]}}}]}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_load_project_models", lambda: {model.name: model})
    monkeypatch.setattr(model_registry, "list_models", lambda: [model.name])
    monkeypatch.setattr(model_registry, "get_model", lambda _n: model)
    return tmp_path


def test_discovery_names_reports_by_their_path(project):
    from tracebi.registry import registry
    from tracebi.web import discovery

    discovery.clear_discovery_report()
    found = discovery.auto_discover(str(project / "reports"))
    expected = {"finance/weekly_summary", "ops/weekly_summary", "top_level",
                "finance/by_region"}
    failed = [o for o in discovery.discovery_report() if o["status"] == "failed"]
    assert not failed, failed
    assert expected <= set(found)
    assert expected <= {r["name"] for r in registry.list_reports()}


def test_same_named_reports_build_to_separate_files(project):
    from tracebi import cli
    from tracebi.verify import FILE_INTACT, verify_file

    for name in ("finance/weekly_summary", "ops/weekly_summary"):
        assert cli.main(["report", "build", name,
                         "--reports-dir", str(project / "reports")]) == 0
    finance = project / "output" / "finance" / "weekly_summary.html"
    ops = project / "output" / "ops" / "weekly_summary.html"
    # Each file is its own report: its own title and its own region's number.
    assert "Finance weekly" in finance.read_text() and 'id="total">150<' in finance.read_text()
    assert "Ops weekly" in ops.read_text() and 'id="total">7<' in ops.read_text()
    for html in (finance, ops):
        manifest = json.loads(html.with_name(html.name + ".manifest.json").read_text())
        assert verify_file(html.read_text(), manifest)["verdict"] == FILE_INTACT


def test_the_web_api_opens_a_report_in_a_folder(project):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.web import discovery
    from tracebi.web.api.routers import reports as reports_router

    discovery.auto_discover(str(project / "reports"))
    app = FastAPI()
    app.include_router(reports_router.router, prefix="/api")
    client = TestClient(app)

    names = {r["name"] for r in client.get("/api/reports").json()}
    assert {"finance/weekly_summary", "ops/weekly_summary"} <= names
    finance = client.get("/api/reports/finance/weekly_summary/built")
    ops = client.get("/api/reports/ops/weekly_summary/built")
    assert finance.status_code == 200 and ops.status_code == 200, finance.text
    assert "Finance weekly" in finance.json()["html"]
    assert "Ops weekly" in ops.json()["html"]
    source = client.get("/api/reports/ops/weekly_summary/source").json()
    assert any(f["path"].endswith("ops/weekly_summary/report.json")
               for f in source["files"])
    spec = client.get("/api/reports/finance/by_region/built")
    assert spec.status_code == 200 and "Finance by region" in spec.json()["html"]
    assert client.get("/api/reports/finance/nope/built").status_code == 404


def test_schedules_and_the_desk_see_folders(project):
    from tracebi import cli
    from tracebi.desk import review
    from tracebi.schedule import discover_schedules

    schedules, errors = discover_schedules(project / "reports")
    assert errors == []
    assert [s["report"] for s in schedules] == ["finance/weekly_summary"]

    for name in ("finance/weekly_summary", "ops/weekly_summary"):
        cli.main(["report", "build", name, "--reports-dir", str(project / "reports")])
    builds = {b["report"] for b in review(str(project))["builds"]}
    assert {"finance/weekly_summary", "ops/weekly_summary"} <= builds


@pytest.mark.parametrize("name", [
    "", "/etc/passwd", "../x", "finance/../../x", "finance/", "a//b",
    ".hidden/x", "finance/_private", "a\\b",
])
def test_a_name_can_never_leave_reports(name):
    assert report_name_error(name) is not None
    with pytest.raises(ValueError):
        output_html("output", name)


def test_good_names_map_to_folders_under_output():
    assert report_name_error("finance/month_end/close") is None
    assert output_html("out", "finance/month_end/close").replace("\\", "/") == \
        "out/finance/month_end/close.html"
    assert output_html("out", "top").replace("\\", "/") == "out/top.html"


def test_the_cli_refuses_a_climbing_name(project, capsys):
    from tracebi import cli

    assert cli.main(["report", "build", "../escape",
                     "--reports-dir", str(project / "reports")]) == 1
    assert "invalid report name" in capsys.readouterr().err


def test_every_named_report_read_goes_through_open_report(tmp_path, monkeypatch):
    """The permission check will live in ``open_report``. A read that never
    calls it would skip that check, so a patched raise must surface from the
    web Source route, an MCP tool, a schedule run, and a CLI report command.
    """
    import tracebi.report_paths as paths

    class _Denied(Exception):
        def __init__(self, name, purpose):
            super().__init__(f"{name}:{purpose}")
            self.name = name
            self.purpose = purpose

    def _deny(name, *, purpose, reports_dir=None, registry=None):
        raise _Denied(name, purpose)

    monkeypatch.setattr(paths, "open_report", _deny)
    monkeypatch.chdir(tmp_path)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.web.api.routers import reports as reports_router

    app = FastAPI()
    app.include_router(reports_router.router, prefix="/api")
    with pytest.raises(_Denied) as web:
        TestClient(app).get("/api/reports/seam_target/source")
    assert (web.value.name, web.value.purpose) == ("seam_target", "source")

    from tracebi.mcp_server import gateway_workbench_state
    with pytest.raises(_Denied) as mcp:
        gateway_workbench_state("seam_target")
    assert (mcp.value.name, mcp.value.purpose) == ("seam_target", "view")

    from tracebi.schedule import run_schedule
    rec = run_schedule(
        {"report": "seam_target", "cron": "0 9 * * MON", "to": [],
         "timezone": None},
        reports_dir=tmp_path / "reports",
        output_dir=tmp_path / "sched-out",
    )
    assert rec["status"] == "failed"
    assert "seam_target:schedule" in rec["error"]

    from tracebi import cli
    with pytest.raises(_Denied) as command:
        cli.main(["report", "status", "seam_target",
                  "--reports-dir", str(tmp_path / "reports")])
    assert (command.value.name, command.value.purpose) == ("seam_target", "view")


# ── Library mounts (E6 / #193) ──────────────────────────────────────────────


def _package_at(root, rel, title, region, schedule=None):
    """Like ``_package`` but under an arbitrary mount root."""
    pkg = root / rel
    pkg.mkdir(parents=True)
    decl = {"name": title, "data": {"totals": {"model": "folder_model", "query": {
        "fact": "fact_orders", "measures": ["revenue"],
        "filters": {"dim_region.region": region}}}}}
    if schedule:
        decl["schedule"] = schedule
    (pkg / "report.json").write_text(json.dumps(decl), encoding="utf-8")
    (pkg / "template.html").write_text(_TEMPLATE.format(title=title), encoding="utf-8")


@pytest.fixture
def mounts_project(tmp_path, monkeypatch, model):
    """Two absolute mounts that each hold a ``weekly`` package."""
    from tracebi import cli, model_registry
    from tracebi.report_paths import reset_mounts_warning_for_tests

    alpha = tmp_path / "mnt_alpha"
    beta = tmp_path / "mnt_beta"
    _package_at(alpha, "weekly", "Alpha weekly", "East",
                schedule={"cron": "0 7 * * MON"})
    _package_at(beta, "weekly", "Beta weekly", "West")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "TRACEBI_LIBRARY_MOUNTS",
        f"alpha:{alpha.resolve()},beta:{beta.resolve()}")
    monkeypatch.delenv("TRACEBI_REPORTS_DIR", raising=False)
    reset_mounts_warning_for_tests()
    monkeypatch.setattr(cli, "_load_project_models", lambda: {model.name: model})
    monkeypatch.setattr(model_registry, "list_models", lambda: [model.name])
    monkeypatch.setattr(model_registry, "get_model", lambda _n: model)
    return tmp_path, alpha, beta


def test_parse_library_mounts_rejects_bad_labels_and_relative_paths(monkeypatch):
    from tracebi.report_paths import parse_library_mounts

    monkeypatch.delenv("TRACEBI_LIBRARY_MOUNTS", raising=False)
    assert parse_library_mounts() is None

    with pytest.raises(ValueError, match="absolute"):
        parse_library_mounts("finance:relative/path")
    with pytest.raises(ValueError, match="single path segment"):
        parse_library_mounts("a/b:/tmp/x")
    with pytest.raises(ValueError, match="duplicate"):
        parse_library_mounts("a:/tmp/a,a:/tmp/b")
    with pytest.raises(ValueError, match="label:/absolute"):
        parse_library_mounts("nocolon")


def test_two_mounts_each_hold_weekly(mounts_project):
    """Done-when for #193: two mounts each hold weekly; both open and build."""
    from tracebi import cli
    from tracebi.registry import registry
    from tracebi.report_paths import open_report, report_name_for_dir
    from tracebi.schedule import discover_library_schedules
    from tracebi.verify import FILE_INTACT, verify_file
    from tracebi.web import discovery

    _tmp, alpha, beta = mounts_project
    discovery.clear_discovery_report()
    for name in [r["name"] for r in registry.list_reports()]:
        if name.startswith(("alpha/", "beta/")):
            registry.remove_report(name)

    discovery.discover_library()
    names = {r["name"] for r in registry.list_reports()}
    assert {"alpha/weekly", "beta/weekly"} <= names

    assert report_name_for_dir(str(alpha / "weekly")) == "alpha/weekly"
    assert report_name_for_dir(str(beta / "weekly")) == "beta/weekly"

    a = open_report("alpha/weekly", purpose="view")
    b = open_report("beta/weekly", purpose="view")
    assert a.package_dir == alpha / "weekly"
    assert b.package_dir == beta / "weekly"
    assert open_report("weekly", purpose="view").package_dir is None

    assert cli.main(["report", "build", "alpha/weekly"]) == 0
    assert cli.main(["report", "build", "beta/weekly"]) == 0
    a_html = _tmp / "output" / "alpha" / "weekly.html"
    b_html = _tmp / "output" / "beta" / "weekly.html"
    assert "Alpha weekly" in a_html.read_text()
    assert "Beta weekly" in b_html.read_text()
    for html in (a_html, b_html):
        manifest = json.loads(html.with_name(html.name + ".manifest.json").read_text())
        assert verify_file(html.read_text(), manifest)["verdict"] == FILE_INTACT

    schedules, errors = discover_library_schedules()
    assert errors == []
    assert [s["report"] for s in schedules] == ["alpha/weekly"]


def test_mounts_ignore_reports_dir(tmp_path, monkeypatch, capsys):
    from tracebi.report_paths import library_roots, reset_mounts_warning_for_tests

    mount = tmp_path / "only"
    mount.mkdir()
    monkeypatch.setenv("TRACEBI_LIBRARY_MOUNTS", f"only:{mount.resolve()}")
    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(tmp_path / "ignored"))
    reset_mounts_warning_for_tests()
    roots = library_roots()
    assert roots == [("only", mount.resolve())]
    err = capsys.readouterr().err
    assert "TRACEBI_REPORTS_DIR is ignored" in err

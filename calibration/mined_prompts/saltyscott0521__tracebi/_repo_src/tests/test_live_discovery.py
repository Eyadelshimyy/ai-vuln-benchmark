"""Live discovery: a running server picks up report packages, specs and
model files added (or removed) without a restart.

These drive :func:`rescan` and the watcher thread directly on the library
registry. They never import ``tracebi.web.api.main`` (see CLAUDE.md,
invariant 3).
"""

import json
import os
import time

import pytest

from tracebi.registry import registry
from tracebi.web import discovery


# Structure only: discovery never resolves a binding against its model.
_DATA = {"k": {"model": "any_model", "query": {"fact": "f", "measures": ["m"]}}}


def _package(root, name, title="Live"):
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "report.json").write_text(json.dumps({"name": title, "data": _DATA}))
    (pkg / "template.html").write_text("<main><h1>{{ title }}</h1></main>")
    return pkg


def _names():
    return {r["name"] for r in registry.list_reports()}


@pytest.fixture()
def reports(tmp_path):
    root = tmp_path / "reports"
    root.mkdir()
    yield root
    for name in [n for n, src in discovery._live_reports.items()
                 if src.startswith(str(tmp_path))]:
        registry.remove_report(name)
        discovery._live_reports.pop(name, None)
    for name in [n for n, (src, _t) in discovery._failed_sources.items()
                 if src.startswith(str(tmp_path))]:
        discovery._failed_sources.pop(name, None)


def test_a_package_added_after_startup_is_registered(reports):
    _package(reports, "first_live")
    discovery.auto_discover(str(reports))
    assert "first_live" in _names()

    _package(reports / "team", "second_live")
    changes = discovery.rescan(str(reports))
    assert changes["added"] == ["team/second_live"]
    assert "team/second_live" in _names()
    assert registry.report_package_dir("team/second_live").endswith("second_live")

    # Nothing new: a second scan changes nothing.
    assert discovery.rescan(str(reports)) == {
        "added": [], "removed": [], "failed": [], "models": [], "pipelines": [],
        "connections": []}


def test_a_deleted_package_is_forgotten(reports):
    pkg = _package(reports, "goes_away")
    discovery.rescan(str(reports))
    assert "goes_away" in _names()

    for f in pkg.iterdir():
        f.unlink()
    pkg.rmdir()
    changes = discovery.rescan(str(reports))
    assert changes["removed"] == ["goes_away"]
    assert "goes_away" not in _names()


def test_a_broken_package_is_retried_only_when_it_changes(reports, monkeypatch):
    pkg = _package(reports, "broken_live")
    (pkg / "report.json").write_text("{ not json")
    calls = []
    real = discovery._register_template_package

    def counting(path, stem):
        calls.append(stem)
        return real(path, stem)

    monkeypatch.setattr(discovery, "_register_template_package", counting)
    assert discovery.rescan(str(reports))["failed"] == ["broken_live"]
    assert discovery.rescan(str(reports))["failed"] == []
    assert calls == ["broken_live"], "an unchanged broken package is not re-parsed"
    outcome = [o for o in discovery.discovery_report() if o.get("module") == "broken_live"]
    assert len(outcome) == 1 and outcome[0]["status"] == "failed"

    (pkg / "report.json").write_text(json.dumps({"name": "Fixed", "data": _DATA}))
    later = time.time() + 5
    os.utime(pkg / "report.json", (later, later))
    assert discovery.rescan(str(reports))["added"] == ["broken_live"]
    assert "broken_live" in _names()


def test_a_spec_is_shadowed_by_a_same_named_package(reports):
    (reports / "both.json").write_text(json.dumps({
        "title": "Spec", "sections": [{"type": "text", "content": "hi"}]}))
    found = discovery._report_sources(str(reports))
    assert found["both"].endswith("both.json")
    _package(reports, "both")
    found = discovery._report_sources(str(reports))
    assert found["both"].endswith(os.sep + "both")


def test_the_watcher_thread_finds_a_new_package(reports):
    stop = discovery.start_watcher(str(reports), None, interval=0.05)
    try:
        _package(reports, "watched_live")
        deadline = time.time() + 5
        while "watched_live" not in _names() and time.time() < deadline:
            time.sleep(0.05)
        assert "watched_live" in _names()
    finally:
        stop.set()


def test_a_new_model_file_is_registered(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "live_model_x.py").write_text(
        "from tracebi import DataModel\nmodel = DataModel('live_model_x')\n")
    added = discovery.register_models(str(models))
    try:
        assert added == ["live_model_x"]
        assert "live_model_x" in [m["name"] for m in registry.list_models()]
        assert discovery.register_models(str(models)) == []
    finally:
        discovery._live_models.discard("live_model_x")


def test_a_new_pipeline_file_is_registered(tmp_path):
    pipes = tmp_path / "pipelines"
    pipes.mkdir()
    (pipes / "live_pipe_x.py").write_text(
        "from tracebi.pipeline import PipelineRunner\n"
        f"runner = PipelineRunner(db_url='sqlite:///{tmp_path / 'runs.db'}')\n")
    try:
        changes = discovery.rescan(str(tmp_path / "no_reports"), None, str(pipes))
        assert changes["pipelines"] == ["live_pipe_x"]
        assert "live_pipe_x" in registry.list_pipeline_names()
        assert discovery.register_pipelines(str(pipes)) == []
    finally:
        discovery._live_pipelines.discard("live_pipe_x")


def test_a_code_module_that_registers_a_packageless_report_is_flagged(tmp_path, capsys):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "old_style.py").write_text(
        "from tracebi.registry import registry\n"
        "registry.add_report('old_style_live', lambda: None)\n")
    try:
        discovery.auto_discover(str(reports))
        entry = next(o for o in discovery.discovery_report()
                     if o.get("file") == "old_style.py")
        assert entry["status"] == "registered"
        assert "'old_style_live' with no report package" in entry["warning"]
        assert "tracebi new-report" in entry["warning"]
        assert "old_style_live" in capsys.readouterr().err
    finally:
        registry.remove_report("old_style_live")

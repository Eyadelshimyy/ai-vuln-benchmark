"""End-to-end journeys: a project on disk, driven through the entry points a
person or an agent actually uses — the CLI (``tracebi.cli.main``), the web app
(FastAPI ``TestClient``), and the MCP gateway's tools.

They run in-process rather than in subprocesses so coverage sees them: the
suite's coverage gate is how we know a unit test the journeys make redundant
can go. Each journey gets a fresh working directory and fresh model and
pipeline registries, so no state leaks between journeys or into the rest of
the suite.
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

REFERENCE_PROJECT = Path(__file__).resolve().parents[2] / "examples" / "portfolio_project"


@pytest.fixture
def isolated(monkeypatch):
    """Fresh model/pipeline registries for one journey."""
    from tracebi import model_registry, pipeline_registry

    monkeypatch.setattr(model_registry, "_registry", model_registry.ModelRegistry())
    monkeypatch.setattr(model_registry, "_auto_discovered", False)
    monkeypatch.setattr(pipeline_registry, "_registry", pipeline_registry.PipelineRegistry())
    monkeypatch.setattr(pipeline_registry, "_auto_discovered", False)
    # Warn-once-per-process flags: a journey's build must not use up a later
    # test's warning.
    from tracebi.reports import base_renderer
    monkeypatch.setattr(base_renderer, "_GIT_SHA_WARNED", base_renderer._GIT_SHA_WARNED)
    yield
    release_warehouses()


def release_warehouses() -> None:
    """Close every model's warehouse connection, as a finished CLI process
    would. In one process, a model left holding the DuckDB file open blocks
    the next transform from writing it."""
    from tracebi import model_registry
    from tracebi.registry import registry

    models = list(model_registry._registry._models.values())
    models += list(registry._models.values())
    for model in models:
        for connector in model.connectors():
            if hasattr(connector, "disconnect"):
                connector.disconnect()


def run_cli(*args: str) -> tuple[int, str]:
    """``tracebi <args>`` in this process; returns (exit code, stdout+stderr).

    Each call ends like a process would: warehouse connections released."""
    from tracebi import cli

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            # A copied context, as a fresh process would have: a command that
            # sets the audit actor must not leave it set for the next test.
            code = contextvars.copy_context().run(cli.main, list(args))
        except SystemExit as exc:  # argparse errors and explicit exits
            code = exc.code if isinstance(exc.code, int) else 1
        finally:
            release_warehouses()
    return code or 0, out.getvalue()


@pytest.fixture
def cli():
    return run_cli


@pytest.fixture
def scaffolded(tmp_path, monkeypatch, isolated):
    """A fresh ``tracebi init`` project, as the working directory."""
    proj = tmp_path / "proj"
    code, out = run_cli("init", str(proj))
    assert code == 0, out
    monkeypatch.chdir(proj)
    return proj


@pytest.fixture
def reference(tmp_path, monkeypatch, isolated):
    """A copy of the reference project (no data, no output), as the working
    directory."""
    proj = tmp_path / "portfolio_project"
    shutil.copytree(REFERENCE_PROJECT, proj, ignore=shutil.ignore_patterns(
        "data", "output", ".tracebi", "__pycache__", "*.duckdb"))
    monkeypatch.chdir(proj)
    return proj


def manifests(root: Path) -> list[Path]:
    return sorted((root / "output").rglob("*.manifest.json"))


def env_off(monkeypatch, *names: str) -> None:
    for n in names:
        monkeypatch.delenv(n, raising=False)


def serve_app(monkeypatch, *, models: bool = False):
    """The web app over the project's pipelines (and models), as the Refresh page sees it."""
    from fastapi.testclient import TestClient

    from tracebi.registry import registry
    from tracebi.web import discovery
    from tracebi.web.api.main import app

    for attr in ("_connectors", "_models", "_report_factories",
                 "_scheduled_factories", "_pipelines"):
        monkeypatch.setattr(registry, attr, {})
    monkeypatch.setattr(discovery, "_live_pipelines", set())
    monkeypatch.setattr(discovery, "_live_models", set())
    discovery.register_pipelines("pipelines")
    if models:
        discovery.register_models("models")
    return TestClient(app)


def follow_run(client, pipeline, run_id):
    """Read a run's log the way the page does: from where the last read ended,
    until the run is done. Returns the whole text and the final answer."""
    import time

    text, after = "", 0
    deadline = time.monotonic() + 60
    while True:
        got = client.get(f"/api/pipelines/{pipeline}/runs/{run_id}/log", params={"after": after}).json()
        text += got["text"]
        after = got["next"]
        if got["done"]:
            return text, got
        assert time.monotonic() < deadline, f"run never finished:\n{text}"
        time.sleep(0.05)


__all__ = ["run_cli", "manifests", "env_off", "serve_app", "follow_run", "REFERENCE_PROJECT", "os"]

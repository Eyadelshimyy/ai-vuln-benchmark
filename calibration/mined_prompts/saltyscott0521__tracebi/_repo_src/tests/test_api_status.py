"""GET /api/status: each check flips when its condition changes."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from tracebi.model_registry import ModelRegistry
from tracebi.web.api.auth import _required_role
from tracebi.web.api.routers.status import collect_checks


def _by_name(checks: list[dict]) -> dict[str, dict]:
    return {check["name"]: check for check in checks}


def _body(tmp_path: Path, monkeypatch) -> dict:
    """A tiny app, so this file never imports ``tracebi.web.api.main``.

    ``TestPipelineRunEndpoint`` rebinds the registry before the pipelines
    router is first imported. Importing the real app here would bind that
    router first and the later test would 404.
    """
    import tracebi
    from fastapi import FastAPI

    from tracebi.web.api.routers.status import router

    monkeypatch.chdir(tmp_path)
    app = FastAPI(version=tracebi.__version__)
    app.include_router(router, prefix="/api")
    response = TestClient(app).get("/api/status")
    assert response.status_code == 200
    return response.json()


def test_route_returns_version_and_the_six_checks(tmp_path, monkeypatch):
    import tracebi

    body = _body(tmp_path, monkeypatch)
    assert body["version"] == tracebi.__version__
    assert set(_by_name(body["checks"])) == {
        "output_writable", "discovery", "models_loaded",
        "email_configured", "schedules_in_server", "auth",
    }


def test_output_writable_flips(tmp_path, monkeypatch):
    # PermissionError, not chmod: as root a mode of 0o500 still allows the
    # write, so the check would stay ok in a container or a cloud sandbox.
    monkeypatch.chdir(tmp_path)
    assert _by_name(collect_checks())["output_writable"]["ok"] is True

    import tracebi.web.api.routers.status as status

    real_makedirs = status.os.makedirs

    def refuse_output(path, *args, **kwargs):
        if os.path.abspath(path) == os.path.abspath(
                os.path.join(os.getcwd(), "output")):
            raise PermissionError(13, "Permission denied")
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(status.os, "makedirs", refuse_output)
    check = _by_name(collect_checks())["output_writable"]
    assert check["ok"] is False
    assert "cannot write" in check["detail"]

    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    assert _by_name(collect_checks())["output_writable"]["ok"] is True


def test_discovery_flips_when_a_file_failed(monkeypatch):
    monkeypatch.setattr(
        "tracebi.web.discovery.discovery_report",
        lambda: [{"file": "ok.py", "status": "registered", "module": "ok"}],
    )
    assert _by_name(collect_checks())["discovery"] == {
        "name": "discovery", "ok": True, "detail": {"failed": 0, "names": []},
    }

    monkeypatch.setattr(
        "tracebi.web.discovery.discovery_report",
        lambda: [{"file": "broken.py", "status": "failed", "module": "broken",
                  "reason": "SyntaxError: bad"}],
    )
    check = _by_name(collect_checks())["discovery"]
    assert check["ok"] is False
    assert check["detail"] == {"failed": 1, "names": ["broken.py"]}


def test_models_loaded_flips_for_a_broken_file(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "good.py").write_text(
        "from tracebi.model.data_model import DataModel\n"
        "model = DataModel('good')\n",
        encoding="utf-8",
    )
    (models / "bad.py").write_text(
        "raise RuntimeError('broken on purpose')\n",
        encoding="utf-8",
    )
    registry = ModelRegistry()
    registry.auto_discover(str(models))
    monkeypatch.setattr("tracebi.model_registry.list_models", registry.list_models)
    monkeypatch.setattr("tracebi.model_registry.get_model", registry.get)

    check = _by_name(collect_checks())["models_loaded"]
    assert check["ok"] is False
    assert check["detail"]["count"] == 1
    assert check["detail"]["failed"] == ["bad"]

    (models / "bad.py").write_text(
        "from tracebi.model.data_model import DataModel\n"
        "model = DataModel('bad')\n",
        encoding="utf-8",
    )
    # The failed load did not cache a model, so the next get reads the file.
    check = _by_name(collect_checks())["models_loaded"]
    assert check["ok"] is True
    assert check["detail"]["failed"] == []
    assert check["detail"]["count"] == 2


def test_email_check_flips_and_never_echoes_the_url(monkeypatch):
    secret = "smtp://leak-user:leak-pass@mail.example:2525"
    monkeypatch.delenv("TRACEBI_SMTP_URL", raising=False)
    monkeypatch.delenv("TRACEBI_SMTP_FROM", raising=False)
    off = _by_name(collect_checks())["email_configured"]
    assert off["ok"] is False
    assert off["detail"] == {
        "TRACEBI_SMTP_URL": False, "TRACEBI_SMTP_FROM": False,
    }

    monkeypatch.setenv("TRACEBI_SMTP_URL", secret)
    monkeypatch.setenv("TRACEBI_SMTP_FROM", "reports@example.com")
    on = _by_name(collect_checks())["email_configured"]
    assert on["ok"] is True
    assert on["detail"] == {
        "TRACEBI_SMTP_URL": True, "TRACEBI_SMTP_FROM": True,
    }
    dumped = json.dumps(collect_checks())
    assert secret not in dumped
    assert "mail.example" not in dumped
    assert "leak-user" not in dumped
    assert "leak-pass" not in dumped
    assert "reports@example.com" not in dumped


def test_schedules_report_the_switch_and_the_count(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    pkg = reports / "weekly"
    pkg.mkdir(parents=True)
    (pkg / "template.html").write_text("<p></p>", encoding="utf-8")
    (pkg / "report.json").write_text(json.dumps({
        "data": {"totals": {"model": "m", "query": {
            "fact": "fact_totals", "measures": ["n"],
        }}},
        "schedule": {"cron": "0 9 * * MON"},
    }), encoding="utf-8")
    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(reports))
    monkeypatch.delenv("TRACEBI_SCHEDULES_IN_SERVER", raising=False)

    off = _by_name(collect_checks())["schedules_in_server"]["detail"]
    assert off == {"on": False, "count": 1}

    monkeypatch.setenv("TRACEBI_SCHEDULES_IN_SERVER", "1")
    (pkg / "report.json").write_text(json.dumps({
        "data": {"totals": {"model": "m", "query": {
            "fact": "fact_totals", "measures": ["n"],
        }}},
    }), encoding="utf-8")
    on = _by_name(collect_checks())["schedules_in_server"]["detail"]
    assert on == {"on": True, "count": 0}


def test_auth_posture_flips_and_omits_the_password(monkeypatch):
    monkeypatch.delenv("TRACEBI_AUTH_PROXY_HEADER", raising=False)
    monkeypatch.delenv("TRACEBI_AUTH_USER", raising=False)
    monkeypatch.delenv("TRACEBI_AUTH_PASS", raising=False)
    off = _by_name(collect_checks())["auth"]["detail"]
    assert off.startswith("auth posture: off")

    monkeypatch.setenv("TRACEBI_AUTH_USER", "ada")
    monkeypatch.setenv("TRACEBI_AUTH_PASS", "super-secret-auth-pass")
    line = _by_name(collect_checks())["auth"]["detail"]
    assert "auth posture: Basic" in line
    assert "super-secret-auth-pass" not in line
    assert "super-secret-auth-pass" not in json.dumps(collect_checks())


def test_response_never_contains_the_smtp_url(tmp_path, monkeypatch):
    secret = "smtp://leak-user:leak-pass@mail.example:2525"
    monkeypatch.setenv("TRACEBI_SMTP_URL", secret)
    monkeypatch.setenv("TRACEBI_SMTP_FROM", "reports@example.com")
    body = _body(tmp_path, monkeypatch)
    dumped = json.dumps(body)
    assert secret not in dumped
    assert "mail.example" not in dumped
    assert os.environ["TRACEBI_SMTP_URL"] not in dumped

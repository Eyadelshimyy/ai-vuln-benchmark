"""Two processes, one Postgres store.

Skips unless ``TRACEBI_TEST_POSTGRES_URL`` is set. The processes share the
database and nothing else: each is a fresh interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

URL = os.environ.get("TRACEBI_TEST_POSTGRES_URL", "")

pytestmark = pytest.mark.skipif(
    not URL, reason="TRACEBI_TEST_POSTGRES_URL is not set",
)


def _env(tmp: Path) -> dict:
    env = os.environ.copy()
    env["TRACEBI_STATE_URL"] = URL
    env["TB_ROOT"] = str(tmp)
    return env


def _run(mode: str, tmp: Path, **extra: str) -> subprocess.Popen:
    env = _env(tmp)
    env.update(extra)
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), mode],
        cwd=str(tmp),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(proc: subprocess.Popen, timeout: float = 30) -> str:
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise AssertionError(f"worker timed out\n{out}\n{err}")
    if proc.returncode != 0:
        raise AssertionError(f"worker exited {proc.returncode}\n{out}\n{err}")
    return out


def test_a_run_started_on_one_worker_is_polled_on_another(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEBI_STATE_URL", URL)
    from tracebi.state import upgrade
    upgrade(URL)

    page = tmp_path / "page.html"
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "report.json").write_text("{}", encoding="utf-8")
    (pkg / "template.html").write_text("<p></p>", encoding="utf-8")
    a = tmp_path / "worker-a"
    b = tmp_path / "worker-b"
    a.mkdir()
    b.mkdir()

    starter = _run("background-start", a, TB_HTML=str(page), TB_PKG=str(pkg))
    assert starter.stdout is not None
    run_id = starter.stdout.readline().strip()
    assert run_id.isdigit(), run_id
    poller = _run("background-poll", b, TB_HTML=str(page), TB_PKG=str(pkg),
                  TB_RUN_ID=run_id)
    polled = _finish(poller)
    started = _finish(starter)
    assert "from-a" in polled
    assert "succeeded" in started


def test_a_schedule_tick_fires_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEBI_STATE_URL", URL)
    from tracebi.state import schedule_records, upgrade
    upgrade(URL)

    hold = tmp_path / "holding"
    sent = tmp_path / "sent"
    out = tmp_path / "output"
    out.mkdir()
    a = tmp_path / "worker-a"
    b = tmp_path / "worker-b"
    a.mkdir()
    b.mkdir()

    first = _run("schedule", a, TB_HOLD=str(hold), TB_SENT=str(sent),
                 TB_OUTPUT=str(out))
    deadline = time.time() + 20
    while not hold.exists():
        if first.poll() is not None:
            raise AssertionError(
                "first tick exited before taking the lock\n"
                + _finish(first))
        if time.time() > deadline:
            first.kill()
            raise AssertionError("first tick never reached the send")
        time.sleep(0.05)
    second = _run("schedule", b, TB_HOLD=str(hold), TB_SENT=str(sent),
                  TB_OUTPUT=str(out))
    second_out = _finish(second)
    first_out = _finish(first)
    assert "skipped" in second_out
    assert "delivered" in first_out
    assert sent.read_text(encoding="utf-8").count("sent") == 1
    delivered = [r for r in schedule_records(out) if r.get("status") == "delivered"]
    assert len(delivered) == 1


# ── workers (a fresh interpreter; pytest does not collect these) ────────────

def _register_demo() -> None:
    from tracebi.web.api.registry import registry

    def factory():
        return None

    factory._tracebi_package_dir = os.environ["TB_PKG"]
    registry.add_report("demo", factory)


def _background_start() -> None:
    import time as _time

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import tracebi.web.api.routers.reports as reports

    def slow(name: str) -> dict:
        _time.sleep(1.5)
        path = os.environ["TB_HTML"]
        Path(path).write_text("<html>from-a</html>", encoding="utf-8")
        Path(path + ".manifest.json").write_text(
            '{"rendered_at": "t"}', encoding="utf-8")
        return {
            "name": name, "html": "<html>from-a</html>", "manifest": {},
            "retained": True, "html_path": path,
            "manifest_path": path + ".manifest.json",
        }

    reports._render_report_payload = slow
    _register_demo()
    app = FastAPI()
    app.include_router(reports.router, prefix="/api")
    client = TestClient(app)
    body = client.post("/api/reports/demo/runs")
    if body.status_code != 202:
        raise SystemExit(body.text)
    print(body.json()["run_id"], flush=True)
    run_id = body.json()["run_id"]
    deadline = _time.time() + 20
    while _time.time() < deadline:
        got = client.get(f"/api/reports/demo/runs/{run_id}").json()
        if got["status"] != "running":
            print(got["status"], flush=True)
            if got["status"] != "succeeded":
                raise SystemExit(str(got))
            return
        _time.sleep(0.05)
    raise SystemExit("start worker: run did not finish")


def _background_poll() -> None:
    import time as _time

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.web.api.routers import reports

    _register_demo()
    app = FastAPI()
    app.include_router(reports.router, prefix="/api")
    client = TestClient(app)
    run_id = os.environ["TB_RUN_ID"]
    deadline = _time.time() + 20
    body = {}
    while _time.time() < deadline:
        res = client.get(f"/api/reports/demo/runs/{run_id}")
        if res.status_code != 200:
            raise SystemExit(res.text)
        body = res.json()
        if body["status"] != "running":
            break
        _time.sleep(0.05)
    else:
        raise SystemExit("poll worker: still running")
    if body.get("status") != "succeeded":
        raise SystemExit(str(body))
    html = (body.get("result") or {}).get("html") or ""
    built = client.get("/api/reports/demo/built")
    if built.status_code != 200:
        raise SystemExit(built.text)
    print(html)
    print(built.json()["html"])


def _schedule_tick() -> None:
    import time as _time

    from tracebi import schedule as sched

    def resolve(name, _reports_dir):
        return "package", Path(_reports_dir) / name

    def build(_kind, _path, output, **_k):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("<html></html>", encoding="utf-8")
        Path(str(output) + ".manifest.json").write_text(
            '{"figures": [], "sections": []}', encoding="utf-8")
        return output

    def ok_verify(_manifest, _models, **_k):
        return {"verdict": "reproduces", "verdict_detail": "", "exit_code": 0,
                "ok": True}

    def send(*_a, **_k):
        Path(os.environ["TB_HOLD"]).write_text("holding", encoding="utf-8")
        _time.sleep(2)
        with open(os.environ["TB_SENT"], "a", encoding="utf-8") as fh:
            fh.write("sent\n")
        return ["a@example.com"]

    import tracebi.cli as cli
    import tracebi.verify as verify_mod
    cli._build_report_target = build
    cli._resolve_report_target = resolve
    verify_mod.verify_manifest = ok_verify
    verify_mod.load_models = lambda *_a, **_k: {}
    import tracebi._delivery as delivery
    delivery.send_report = send

    rec = sched.run_schedule(
        {"report": "tickonce", "cron": "0 9 * * MON", "to": ["a@example.com"],
         "timezone": None, "retries": 0, "owner": None,
         "refresh": {"transforms": [], "pipelines": []}},
        reports_dir=os.environ["TB_OUTPUT"],
        output_dir=os.environ["TB_OUTPUT"],
        send=True,
    )
    print(rec["status"], flush=True)


if __name__ == "__main__":
    {"background-start": _background_start,
     "background-poll": _background_poll,
     "schedule": _schedule_tick}[sys.argv[1]]()

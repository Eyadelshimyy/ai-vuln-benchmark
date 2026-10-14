"""The scheduled-report journey: a report.json ``schedule`` block, then
``tracebi schedule run`` does build → verify → email → record. Only the mail
server is faked (at smtplib, the network boundary); everything above it runs
for real, including the verify that must pass before anything is sent.
"""

import json
import smtplib

import pytest

from tests.e2e.conftest import run_cli


class _Outbox:
    sent: list = []

    def __init__(self, host, port, timeout=None, **_):
        self.host, self.port = host, port

    def starttls(self, context=None):
        raise smtplib.SMTPNotSupportedError     # a local relay with no TLS

    def send_message(self, msg):
        _Outbox.sent.append(msg)

    def quit(self):
        pass


@pytest.fixture
def scheduled(scaffolded, monkeypatch):
    _Outbox.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _Outbox)
    monkeypatch.setenv("TRACEBI_SMTP_URL", "smtp://localhost:2525")
    monkeypatch.setenv("TRACEBI_SMTP_FROM", "reports@example.com")
    monkeypatch.delenv("TRACEBI_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TRACEBI_SLACK_CHANNEL", raising=False)
    decl = scaffolded / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    spec = json.loads(decl.read_text())
    spec["schedule"] = {"cron": "0 7 * * MON", "to": ["team@example.com"]}
    decl.write_text(json.dumps(spec))
    code, out = run_cli("run-transform", "sample_transform")
    assert code == 0, out
    return scaffolded


def _runs(proj):
    from tracebi.state import schedule_records
    return schedule_records(proj / "output")


def test_a_scheduled_report_is_built_verified_emailed_and_recorded(scheduled):
    code, out = run_cli("schedule", "list")
    assert code == 0 and "sample_model/sample_dashboard" in out and "0 7 * * MON" in out, out

    code, out = run_cli("schedule", "run", "sample_model/sample_dashboard")
    assert code == 0, out
    [msg] = _Outbox.sent
    assert msg["To"] == "team@example.com"
    assert any(part.get_filename() == "sample_dashboard.html"
               for part in msg.iter_attachments())

    [record] = _runs(scheduled)
    assert record["verdict"] == "reproduces"
    assert record["recipients"] == ["team@example.com"]
    assert record["error"] is None
    assert record["attempts"] == 1
    assert "alert" not in record


def test_sending_without_a_mail_server_says_what_to_set(scheduled, monkeypatch):
    monkeypatch.delenv("TRACEBI_SMTP_URL")
    code, out = run_cli("report", "send", "sample_model/sample_dashboard", "--to", "team@example.com")
    assert code != 0
    assert "TRACEBI_SMTP_URL" in out
    assert _Outbox.sent == []


def test_a_refresh_that_fails_once_is_retried_and_succeeds(scheduled, monkeypatch):
    slept = []
    monkeypatch.setattr("tracebi.schedule._sleep", slept.append)
    (scheduled / "transforms" / "flaky_once.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "marker = Path('flaky.once')\n"
        "if not marker.exists():\n"
        "    marker.write_text('1')\n"
        "    print('transient failure', file=sys.stderr)\n"
        "    raise RuntimeError('transient failure')\n",
        encoding="utf-8",
    )
    decl = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    spec = json.loads(decl.read_text())
    spec["schedule"]["refresh"] = {"transforms": ["flaky_once"]}
    decl.write_text(json.dumps(spec))

    code, out = run_cli("schedule", "run", "sample_model/sample_dashboard")
    assert code == 0, out
    assert len(_Outbox.sent) == 1
    [record] = _runs(scheduled)
    assert record["status"] == "delivered"
    assert record["attempts"] == 2
    assert slept == [60]
    assert record["refresh"][0]["ok"] is True
    assert record["error"] is None


def test_a_build_that_always_fails_is_retried_then_recorded(scheduled, monkeypatch):
    slept = []
    monkeypatch.setattr("tracebi.schedule._sleep", slept.append)
    decl = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    spec = json.loads(decl.read_text())
    spec["data"]["kpis"]["query"]["measures"] = ["not_a_measure"]
    decl.write_text(json.dumps(spec))

    code, out = run_cli("schedule", "run", "sample_model/sample_dashboard")
    assert code == 1, out
    assert _Outbox.sent == []
    [record] = _runs(scheduled)
    assert record["status"] == "failed"
    assert record["attempts"] == 3
    assert record["recipients"] == []
    assert slept == [60, 300]
    assert "not_a_measure" in record["error"]


def test_a_receipt_that_does_not_reproduce_is_not_retried(scheduled, monkeypatch):
    slept = []
    monkeypatch.setattr("tracebi.schedule._sleep", slept.append)
    monkeypatch.setattr(
        "tracebi.verify.verify_manifest",
        lambda *a, **k: {
            "verdict": "unexplained",
            "verdict_detail": "stub: did not reproduce",
            "exit_code": 1,
            "ok": False,
        },
    )
    code, out = run_cli("schedule", "run", "sample_model/sample_dashboard")
    assert code == 1, out
    assert _Outbox.sent == []
    [record] = _runs(scheduled)
    assert record["status"] == "refused"
    assert record["attempts"] == 1
    assert slept == []


def test_an_empty_source_alerts_the_owner_and_sends_nothing(
        scheduled, monkeypatch):
    """A figure binding with no rows is recorded empty, not mailed to `to`,
    and the owner gets an alert that names the binding. A binding no figure
    uses does not count. Verify still has to pass first."""
    monkeypatch.setattr("tracebi.schedule._sleep", lambda *_a, **_k: None)
    decl = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    spec = json.loads(decl.read_text())
    spec["schedule"]["owner"] = "owner@example.com"
    nowhere = {"dim_region.region": "no-such-region"}
    spec["data"]["by_region"]["query"]["filters"] = nowhere
    spec["data"]["spare"] = {
        "model": "sample_model",
        "query": {
            "fact": "fact_orders",
            "measures": ["revenue"],
            "dimensions": ["dim_region.region"],
            "filters": dict(nowhere),
        },
    }
    decl.write_text(json.dumps(spec))

    code, out = run_cli("schedule", "run", "sample_model/sample_dashboard")
    assert code == 1, out
    [msg] = _Outbox.sent
    assert msg["To"] == "owner@example.com"
    assert "team@example.com" not in msg["To"]
    assert msg["Subject"] == "TraceBi: sample_model/sample_dashboard scheduled run empty"
    body = msg.get_content()
    assert "by_region" in body
    assert "spare" not in body
    assert "tracebi_runs (kind=schedule)" in body
    assert list(msg.iter_attachments()) == []

    [record] = _runs(scheduled)
    assert record["status"] == "empty"
    assert record["verdict"] == "reproduces"
    assert record["empty_bindings"] == ["by_region"]
    assert record["recipients"] == []
    assert record["alert"] == {
        "to": "owner@example.com", "sent": True, "error": None}


def test_retries_zero_does_not_retry_a_failed_refresh(scheduled, monkeypatch):
    slept = []
    monkeypatch.setattr("tracebi.schedule._sleep", slept.append)
    (scheduled / "transforms" / "always_fails.py").write_text(
        "import sys\n"
        "print('nope', file=sys.stderr)\n"
        "raise RuntimeError('nope')\n",
        encoding="utf-8",
    )
    decl = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    spec = json.loads(decl.read_text())
    spec["schedule"]["retries"] = 0
    spec["schedule"]["refresh"] = {"transforms": ["always_fails"]}
    decl.write_text(json.dumps(spec))

    code, out = run_cli("schedule", "run", "sample_model/sample_dashboard")
    assert code == 1, out
    assert _Outbox.sent == []
    [record] = _runs(scheduled)
    assert record["status"] == "failed"
    assert record["attempts"] == 1
    assert slept == []
    assert not (scheduled / "output" / "sample_model" / "sample_dashboard.html").exists()

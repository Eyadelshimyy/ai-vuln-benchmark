"""
Scheduled reports: a package's report.json "schedule" block, and
`tracebi schedule list | run | serve`.

The end-to-end cases run the scaffolded project in subprocesses (like
test_init_scaffold.py), so they share no registry state with the suite.
Delivery is stubbed inside the subprocess; nothing is emailed.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tracebi import cli
from tracebi import schedule as sched

pytest.importorskip("duckdb")

SCHEDULE = {"cron": "0 9 * * MON", "timezone": "America/New_York",
            "to": ["cfo@example.com"]}


# ── the block ───────────────────────────────────────────────────────────────

class TestParseScheduleBlock:
    def test_normalizes(self):
        got = sched.parse_schedule_block(
            {"cron": "0  9 * *   MON", "to": "a@x.com, b@x.com"}, path="r")
        assert got == {"cron": "0 9 * * MON", "timezone": None,
                       "to": ["a@x.com", "b@x.com"],
                       "refresh": {"transforms": [], "pipelines": []},
                       "retries": 2, "owner": None, "burst": None}

    @pytest.mark.parametrize("raw, fragment", [
        ("0 9 * * MON", "must be an object"),
        ({"cron": "0 9 * *"}, "five-field cron"),
        ({"cron": "0 9 * * MON", "every": "day"}, "unknown schedule field"),
        ({"cron": "0 9 * * MON", "to": ["not-an-address"]}, "email addresses"),
        ({"cron": "0 9 * * MON", "timezone": "Mars/Olympus"}, "unknown schedule timezone"),
        ({"cron": "0 9 * * MON", "refresh": ["t"]}, "'refresh' must be an object"),
        ({"cron": "0 9 * * MON", "refresh": {"models": ["m"]}}, "'refresh' must be an object"),
        ({"cron": "0 9 * * MON", "retries": "x"}, "integer from 0 to 5"),
        ({"cron": "0 9 * * MON", "retries": 9}, "integer from 0 to 5"),
        ({"cron": "0 9 * * MON", "retries": True}, "integer from 0 to 5"),
        ({"cron": "0 9 * * MON", "owner": "not-an-address"}, "one email"),
        ({"cron": "0 9 * * MON", "owner": ["ops@example.com"]}, "one email"),
        ({"cron": "0 9 * * MON", "owner": 3}, "one email"),
    ])
    def test_refuses(self, raw, fragment):
        with pytest.raises(ValueError, match=fragment):
            sched.parse_schedule_block(raw, path="reports/x/report.json")

    def test_retries_accepts_zero_through_five(self):
        for n in (0, 2, 5):
            got = sched.parse_schedule_block(
                {"cron": "0 9 * * MON", "retries": n}, path="r")
            assert got["retries"] == n

    def test_owner_is_one_address(self):
        got = sched.parse_schedule_block(
            {"cron": "0 9 * * MON", "owner": "ops@example.com"}, path="r")
        assert got["owner"] == "ops@example.com"


# ── a scaffolded project ────────────────────────────────────────────────────

# A failed refresh in a child process retries with real sleeps unless this
# runs first. Success paths never call it.
_NO_RETRY_WAIT = """
import tracebi.schedule as _sched
_sched._sleep = lambda *_a, **_k: None
"""


def _run(args, cwd, driver: str = ""):
    """`tracebi <args>` in *cwd*; *driver* is Python run first (stubs)."""
    code = _NO_RETRY_WAIT + textwrap.dedent(driver) + (
        "\nfrom tracebi import cli\n"
        f"raise SystemExit(cli.main({list(args)!r}))\n")
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, cwd=str(cwd))


# Records each send in sent.json instead of emailing.
_STUB_SEND = """
import json, tracebi._delivery as d
def _send(html, manifest, to, subject=None, verify_result=None):
    json.dump({"to": list(to), "html": str(html)}, open("sent.json", "w"))
    return list(to)
d.send_report = _send
"""

_FAIL_VERIFY = """
import tracebi.verify as v
v.verify_manifest = lambda manifest, models, **k: {
    "verdict": "unexplained", "verdict_detail": "stub: did not reproduce",
    "exit_code": 1, "ok": False}
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    proj = tmp_path_factory.mktemp("sched") / "proj"
    assert cli.main(["init", str(proj)]) == 0
    out = subprocess.run([sys.executable, "transforms/sample_transform.py"],
                         capture_output=True, text=True, cwd=str(proj))
    assert out.returncode == 0, out.stderr
    return proj


@pytest.fixture
def scheduled(project, tmp_path):
    """A copy of the project whose sample report carries SCHEDULE."""
    proj = tmp_path / "proj"
    shutil.copytree(project, proj)
    rj = proj / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    decl = json.loads(rj.read_text())
    decl["schedule"] = SCHEDULE
    rj.write_text(json.dumps(decl))
    return proj


def _runs(proj: Path) -> list[dict]:
    from tracebi.state import schedule_records
    return schedule_records(proj / "output")


class TestDiscovery:
    def test_finds_scheduled_packages_only(self, scheduled):
        schedules, errors = sched.discover_schedules(scheduled / "reports")
        assert errors == []
        assert schedules == [{"report": "sample_model/sample_dashboard", **SCHEDULE,
                              "refresh": {"transforms": [], "pipelines": []},
                              "retries": 2, "owner": None, "burst": None}]

    def test_broken_block_is_an_error_entry_not_a_crash(self, scheduled):
        rj = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
        decl = json.loads(rj.read_text())
        decl["schedule"] = {"cron": "weekly"}
        rj.write_text(json.dumps(decl))
        schedules, errors = sched.discover_schedules(scheduled / "reports")
        assert schedules == []
        assert errors[0]["report"] == "sample_model/sample_dashboard"
        assert "five-field cron" in errors[0]["error"]


class TestScheduleRun:
    def test_a_receipt_that_does_not_verify_is_refused_and_not_sent(self, scheduled):
        out = _run(["schedule", "run", "sample_model/sample_dashboard"],
                   scheduled, _STUB_SEND + _FAIL_VERIFY)
        assert out.returncode == 1
        assert not (scheduled / "sent.json").exists()
        [rec] = _runs(scheduled)
        assert rec["status"] == sched.REFUSED
        assert rec["recipients"] == []

    def test_run_needs_a_schedule_block(self, project):
        out = _run(["schedule", "run", "sample_model/sample_dashboard"], project)
        assert out.returncode == 1
        assert "no schedule block" in out.stderr

def _kpi_orders(proj: Path) -> int:
    import csv as _csv
    import io
    import re
    html = (proj / "output" / "sample_model" / "sample_dashboard.html").read_text()
    block = re.search(r'<script[^>]*id="tracebi-data-kpis"[^>]*>(.*?)</script>',
                      html, re.S).group(1)
    row = next(_csv.DictReader(io.StringIO(json.loads(block)["csv"])))
    return int(float(row["orders"]))


class TestRefresh:
    """A schedule's refresh steps run before the build, so the report shows
    fresh data; a failed step stops the run before anything is built."""

    def _with_refresh(self, proj, transforms):
        rj = proj / "reports" / "sample_model" / "sample_dashboard" / "report.json"
        decl = json.loads(rj.read_text())
        decl["schedule"]["refresh"] = {"transforms": transforms}
        rj.write_text(json.dumps(decl))

    def test_refresh_runs_the_transform_before_the_build(self, scheduled):
        # Baseline: a build before any new data.
        assert _run(["report", "build", "sample_model/sample_dashboard"], scheduled).returncode == 0
        before = _kpi_orders(scheduled)
        # One new order lands in the raw input after the warehouse was built.
        csv = scheduled / "inputs" / "orders.csv"
        lines = csv.read_text().splitlines()
        new_row = "9999" + lines[1][lines[1].index(","):]
        csv.write_text("\n".join(lines + [new_row]) + "\n")

        self._with_refresh(scheduled, ["sample_transform"])
        out = _run(["schedule", "run", "sample_model/sample_dashboard", "--no-send"], scheduled)
        assert out.returncode == 0, out.stderr
        [rec] = _runs(scheduled)
        assert rec["status"] == sched.BUILT
        assert [s["step"] for s in rec["refresh"]] == ["transform:sample_transform"]
        assert rec["refresh"][0]["ok"] is True
        # The built report's embedded, fingerprinted KPI data counts it.
        assert _kpi_orders(scheduled) == before + 1

    def test_a_failed_refresh_builds_and_sends_nothing(self, scheduled):
        self._with_refresh(scheduled, ["no_such_transform"])
        out = _run(["schedule", "run", "sample_model/sample_dashboard"], scheduled, _STUB_SEND)
        assert out.returncode == 1
        [rec] = _runs(scheduled)
        assert rec["status"] == sched.FAILED
        assert "refresh transform 'no_such_transform' failed" in rec["error"]
        assert not (scheduled / "output" / "sample_model" / "sample_dashboard.html").exists()
        assert not (scheduled / "sent.json").exists()


@pytest.fixture(autouse=True)
def _no_real_retry_wait(monkeypatch):
    """Default retries must not call time.sleep. A test that asserts the
    delays replaces this with its own recorder. Slack delivery stays off
    unless a test sets it — a developer token must not reach the network."""
    monkeypatch.setattr(sched, "_sleep", lambda *_a, **_k: None)
    monkeypatch.delenv("TRACEBI_SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TRACEBI_SLACK_CHANNEL", raising=False)
    monkeypatch.delenv("TRACEBI_SLACK_WEBHOOK", raising=False)


def _enter(proj: Path, monkeypatch) -> None:
    """The build loads ``models/`` and the warehouse from the working directory."""
    monkeypatch.chdir(proj)


def _scheduled(proj: Path) -> dict:
    schedules, errors = sched.discover_schedules(proj / "reports")
    assert errors == []
    assert len(schedules) == 1
    return schedules[0]


class TestFailurePaths:
    """A failed run is a row in the log with a readable error, and nothing
    goes out that should not. These call ``run_schedule`` in-process: the
    CLI tests above spawn a child, and a child is invisible to coverage."""

    def test_a_broken_build_is_recorded_and_not_sent(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        sent = []
        monkeypatch.setattr(
            "tracebi._delivery.send_report",
            lambda *a, **k: sent.append(a) or ["nobody"],
        )
        rj = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
        decl = json.loads(rj.read_text())
        binding = next(iter(decl["data"]))
        decl["data"][binding]["query"]["measures"] = ["not_a_measure"]
        rj.write_text(json.dumps(decl))

        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == []
        assert rec["status"] == sched.FAILED
        assert rec["recipients"] == []
        assert "not_a_measure" in rec["error"]
        assert _runs(scheduled) == [rec]

    def test_a_failed_refresh_is_recorded_in_process(self, scheduled, monkeypatch):
        sent = []
        monkeypatch.setattr(
            "tracebi._delivery.send_report",
            lambda *a, **k: sent.append(a) or ["nobody"],
        )
        block = _scheduled(scheduled)
        block["refresh"] = {"transforms": ["no_such_transform"], "pipelines": []}
        rec = sched.run_schedule(
            block, reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == []
        assert rec["status"] == sched.FAILED
        assert "refresh transform 'no_such_transform' failed" in rec["error"]
        assert rec["refresh"][0]["ok"] is False
        assert not (scheduled / "output" / "sample_model" / "sample_dashboard.html").exists()

    def test_smtp_failure_is_recorded_and_not_delivered(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)

        sent = {"n": 0}

        def _boom(*args, **kwargs):
            sent["n"] += 1
            raise RuntimeError("smtp refused the message")

        monkeypatch.setattr("tracebi._delivery.send_report", _boom)
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent["n"] == 1
        assert rec["status"] == sched.FAILED
        assert rec["recipients"] == []
        assert rec["error"] == "RuntimeError: smtp refused the message"
        assert rec["attempts"] == 1
        assert _runs(scheduled)[-1]["error"] == rec["error"]

    def test_a_schedule_cannot_force_a_bad_receipt(self, scheduled, monkeypatch):
        """``tracebi report send --force`` can send a failing receipt. A
        schedule block cannot: ``force`` is not a field, so the package is
        skipped, and a receipt that does not verify is refused with nothing
        sent."""
        rj = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
        decl = json.loads(rj.read_text())
        decl["schedule"]["force"] = True
        rj.write_text(json.dumps(decl))
        schedules, errors = sched.discover_schedules(scheduled / "reports")
        assert schedules == []
        assert errors[0]["report"] == "sample_model/sample_dashboard"
        assert "unknown schedule field" in errors[0]["error"]
        assert "force" in errors[0]["error"]

        decl["schedule"].pop("force")
        rj.write_text(json.dumps(decl))
        _enter(scheduled, monkeypatch)
        sent = []
        monkeypatch.setattr(
            "tracebi._delivery.send_report",
            lambda *a, **k: sent.append(a) or ["nobody"],
        )
        monkeypatch.setattr(
            "tracebi.verify.verify_manifest",
            lambda *a, **k: {
                "verdict": "unexplained",
                "verdict_detail": "stub: did not reproduce",
                "exit_code": 1, "ok": False,
            },
        )
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == []
        assert rec["status"] == sched.REFUSED
        assert rec["error"] == "stub: did not reproduce"
        assert rec["recipients"] == []
        assert rec["attempts"] == 1

    def test_make_job_records_the_scheduler_actor(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        job = sched.make_job(scheduled / "reports", scheduled / "output")
        block = _scheduled(scheduled)
        block["to"] = []
        rec = job(block)
        assert rec["status"] == sched.BUILT
        assert rec["actor"] == "scheduler"
        assert rec["recipients"] == []
        assert "alert" not in rec


def _owner(proj: Path, **extra) -> dict:
    """The project's schedule, with *extra* written into the block."""
    rj = proj / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    decl = json.loads(rj.read_text())
    decl["schedule"].update(extra)
    rj.write_text(json.dumps(decl))
    return _scheduled(proj)


def _break_build(proj: Path) -> None:
    rj = proj / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    decl = json.loads(rj.read_text())
    binding = next(iter(decl["data"]))
    decl["data"][binding]["query"]["measures"] = ["not_a_measure"]
    rj.write_text(json.dumps(decl))


def _empty_region(proj: Path, *names: str) -> None:
    """A filter no region matches, so a grouped binding returns zero rows."""
    rj = proj / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    decl = json.loads(rj.read_text())
    nowhere = {"dim_region.region": "no-such-region"}
    for name in names:
        if name not in decl["data"]:
            decl["data"][name] = {
                "model": "sample_model",
                "query": {"fact": "fact_orders", "measures": ["revenue"],
                          "dimensions": ["dim_region.region"]},
            }
        decl["data"][name]["query"]["filters"] = nowhere
    rj.write_text(json.dumps(decl))


class TestOwnerAlerts:
    """An owner is emailed when a run ends failed, refused, or empty.
    The report's ``to`` list is not. No owner means no alert."""

    def _capture(self, monkeypatch):
        alerts = []
        sent = []
        monkeypatch.setattr(
            "tracebi._delivery.send_alert",
            lambda to, subject, body: alerts.append(
                {"to": to, "subject": subject, "body": body}),
        )
        monkeypatch.setattr(
            "tracebi._delivery.send_report",
            lambda *a, **k: sent.append(a) or ["nobody"],
        )
        return alerts, sent

    def test_a_failed_build_after_retries_alerts_the_owner(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, sent = self._capture(monkeypatch)
        _break_build(scheduled)
        block = _owner(scheduled, owner="ops@example.com")
        rec = sched.run_schedule(
            block, reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == []
        assert rec["status"] == sched.FAILED
        assert rec["attempts"] == 3
        assert rec["alert"] == {
            "to": "ops@example.com", "sent": True, "error": None}
        [alert] = alerts
        assert alert["to"] == "ops@example.com"
        assert alert["subject"] == (
            "TraceBi: sample_model/sample_dashboard scheduled run failed")
        assert "not_a_measure" in alert["body"]
        assert "Attempts: 3" in alert["body"]
        assert "tracebi_runs (kind=schedule)" in alert["body"]

    def test_a_refused_receipt_alerts_the_owner(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, sent = self._capture(monkeypatch)
        monkeypatch.setattr(
            "tracebi.verify.verify_manifest",
            lambda *a, **k: {
                "verdict": "unexplained",
                "verdict_detail": "stub: did not reproduce",
                "exit_code": 1, "ok": False,
            },
        )
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == []
        assert rec["status"] == sched.REFUSED
        assert rec["attempts"] == 1
        assert rec["alert"]["sent"] is True
        [alert] = alerts
        assert alert["subject"].endswith("scheduled run refused")
        assert "unexplained" in alert["body"]
        assert "stub: did not reproduce" in alert["body"]

    def test_a_success_sends_the_report_and_no_alert(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, sent = self._capture(monkeypatch)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.DELIVERED
        assert "alert" not in rec
        assert alerts == []
        assert sent and sent[0][2] == ["cfo@example.com"]

    def test_without_an_owner_a_failure_is_unchanged(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, sent = self._capture(monkeypatch)
        _break_build(scheduled)
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == [] and alerts == []
        assert rec["status"] == sched.FAILED
        assert "alert" not in rec
        assert "empty_bindings" not in rec
        assert set(rec) == {
            "report", "started_at", "finished_at", "status", "verdict",
            "recipients", "output", "error", "actor", "actor_role",
            "refresh", "attempts",
        }

    def test_no_send_records_the_alert_and_does_not_email(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, sent = self._capture(monkeypatch)
        _break_build(scheduled)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
            send=False,
        )
        assert sent == [] and alerts == []
        assert rec["status"] == sched.FAILED
        assert rec["alert"] == {
            "to": "ops@example.com", "sent": False, "error": None}

    def test_an_alert_smtp_failure_is_recorded_and_the_status_stays(
            self, scheduled, monkeypatch):
        import smtplib

        _enter(scheduled, monkeypatch)
        _empty_region(scheduled, "by_region", "spare")
        monkeypatch.setenv("TRACEBI_SMTP_URL", "smtp://localhost:2525")
        monkeypatch.setenv("TRACEBI_SMTP_FROM", "reports@example.com")

        class _Down:
            def __init__(self, *args, **kwargs):
                pass

            def starttls(self, context=None):
                raise smtplib.SMTPNotSupportedError

            def send_message(self, msg):
                raise RuntimeError("smtp down")

            def quit(self):
                pass

        monkeypatch.setattr("tracebi._delivery.smtplib.SMTP", _Down)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.EMPTY
        assert rec["empty_bindings"] == ["by_region"]
        assert "spare" not in rec["empty_bindings"]
        assert rec["recipients"] == []
        assert rec["verdict"] == "reproduces"
        assert rec["alert"]["sent"] is False
        assert rec["alert"]["to"] == "ops@example.com"
        assert rec["alert"]["error"] == "RuntimeError: smtp down"

    def _refuse(self, monkeypatch):
        monkeypatch.setattr(
            "tracebi.verify.verify_manifest",
            lambda *a, **k: {
                "verdict": "unexplained",
                "verdict_detail": "stub: did not reproduce",
                "exit_code": 1, "ok": False,
            },
        )

    def test_a_public_url_links_the_alert_to_the_runs_page(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, sent = self._capture(monkeypatch)
        monkeypatch.delenv("TRACEBI_SLACK_WEBHOOK", raising=False)
        monkeypatch.setenv("TRACEBI_PUBLIC_URL", "https://bi.example.com")
        self._refuse(monkeypatch)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == []
        assert rec["status"] == sched.REFUSED
        assert rec["alert"]["sent"] is True
        [alert] = alerts
        assert (
            "See the run: https://bi.example.com/runs?"
            "kind=schedule&target=sample_model%2Fsample_dashboard"
        ) in alert["body"]

    def test_without_a_public_url_the_alert_has_no_runs_link(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, _sent = self._capture(monkeypatch)
        monkeypatch.delenv("TRACEBI_PUBLIC_URL", raising=False)
        monkeypatch.delenv("TRACEBI_SLACK_WEBHOOK", raising=False)
        self._refuse(monkeypatch)
        sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        [alert] = alerts
        assert "See the run" not in alert["body"]
        assert "tracebi_runs (kind=schedule)" in alert["body"]

    def test_a_slack_webhook_is_pinged_with_the_alert_text(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        alerts, _sent = self._capture(monkeypatch)
        pings = []
        monkeypatch.delenv("TRACEBI_PUBLIC_URL", raising=False)
        monkeypatch.setenv(
            "TRACEBI_SLACK_WEBHOOK", "https://hooks.example/abc")
        monkeypatch.setattr(
            "tracebi._delivery.slack_notify",
            lambda url, text: pings.append({"url": url, "text": text}) or 200,
        )
        self._refuse(monkeypatch)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.REFUSED
        assert rec["alert"]["sent"] is True
        assert rec["alert"]["error"] is None
        assert "slack_error" not in rec["alert"]
        [alert] = alerts
        assert pings == [{
            "url": "https://hooks.example/abc", "text": alert["body"]}]
        assert "See the run" not in alert["body"]

    def test_a_failed_slack_ping_stays_on_the_alert(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        self._capture(monkeypatch)
        monkeypatch.setenv(
            "TRACEBI_SLACK_WEBHOOK", "https://hooks.example/abc")

        def boom(_url, _text):
            raise RuntimeError("slack down")

        monkeypatch.setattr("tracebi._delivery.slack_notify", boom)
        self._refuse(monkeypatch)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.REFUSED
        assert rec["alert"]["sent"] is True
        assert rec["alert"]["error"] is None
        assert rec["alert"]["slack_error"] == "RuntimeError: slack down"

    def test_a_failed_email_still_pings_slack(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        pings = []

        def down(*_a, **_k):
            raise RuntimeError("smtp down")

        monkeypatch.setattr("tracebi._delivery.send_alert", down)
        monkeypatch.setenv(
            "TRACEBI_SLACK_WEBHOOK", "https://hooks.example/abc")
        monkeypatch.setattr(
            "tracebi._delivery.slack_notify",
            lambda url, text: pings.append(text) or 200,
        )
        self._refuse(monkeypatch)
        rec = sched.run_schedule(
            _owner(scheduled, owner="ops@example.com"),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.REFUSED
        assert rec["alert"]["sent"] is False
        assert rec["alert"]["error"] == "RuntimeError: smtp down"
        assert "slack_error" not in rec["alert"]
        assert len(pings) == 1
        assert "stub: did not reproduce" in pings[0]

    def test_a_bad_owner_is_rejected_when_the_package_loads(self, scheduled):
        rj = scheduled / "reports" / "sample_model" / "sample_dashboard" / "report.json"
        decl = json.loads(rj.read_text())
        decl["schedule"]["owner"] = "not-an-address"
        rj.write_text(json.dumps(decl))
        schedules, errors = sched.discover_schedules(scheduled / "reports")
        assert schedules == []
        assert errors[0]["report"] == "sample_model/sample_dashboard"
        assert "one email" in errors[0]["error"]


def test_the_runs_link_quotes_the_report_name(monkeypatch):
    monkeypatch.setenv("TRACEBI_PUBLIC_URL", "https://bi.example.com/")
    body = sched._alert_body({
        "report": "finance/weekly board",
        "status": "failed",
        "error": "boom",
        "attempts": 1,
        "started_at": "t0",
        "finished_at": "t1",
    })
    assert (
        "See the run: https://bi.example.com/runs?"
        "kind=schedule&target=finance%2Fweekly%20board"
    ) in body
    monkeypatch.delenv("TRACEBI_PUBLIC_URL")
    assert "See the run" not in sched._alert_body({"report": "x", "error": "e"})


def test_describe_schedule_notes_a_non_default_retry_count():
    base = {"report": "weekly", "cron": "0 9 * * MON", "to": [], "timezone": None}
    assert "retries:" not in sched.describe_schedule({**base, "retries": 2})
    assert "retries: 0" in sched.describe_schedule({**base, "retries": 0})
    assert "owner:" not in sched.describe_schedule(base)
    named = sched.describe_schedule({**base, "owner": "ops@example.com"})
    assert "owner: ops@example.com" in named


def test_a_retry_past_the_second_waits_five_minutes_again(scheduled, monkeypatch):
    """The wait is 60s, then 300s, and 300s again for any further retry."""
    slept = []
    monkeypatch.setattr(sched, "_sleep", slept.append)

    def boom(*_a, **_k):
        raise RuntimeError("build exploded")

    monkeypatch.setattr("tracebi.cli._build_report_target", boom)
    block = _scheduled(scheduled)
    block["retries"] = 3
    rec = sched.run_schedule(
        block, reports_dir=scheduled / "reports",
        output_dir=scheduled / "output", send=False,
    )
    assert rec["status"] == sched.FAILED
    assert rec["attempts"] == 4
    assert slept == [60, 300, 300]
    assert rec["error"] == "RuntimeError: build exploded"
    assert rec["recipients"] == []


def test_a_tick_that_does_not_get_the_lock_records_nothing(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def deny(_engine, name):
        assert name == "schedule:weekly"
        yield False

    monkeypatch.setattr("tracebi.state.advisory_lock", deny)
    rec = sched.run_schedule(
        {"report": "weekly", "cron": "0 9 * * MON", "to": ["a@example.com"],
         "timezone": None, "retries": 0, "owner": "o@example.com",
         "refresh": {"transforms": [], "pipelines": []}},
        reports_dir=tmp_path, output_dir=tmp_path / "output",
    )
    assert rec["status"] == "skipped"
    assert rec["error"] == "already running in another process"
    assert sched.last_runs(tmp_path / "output") == {}


def test_last_runs_skips_a_broken_line(tmp_path):
    log = tmp_path / sched.RUN_LOG
    log.write_text('not json\n{"report": "weekly", "status": "failed"}\n',
                   encoding="utf-8")
    last = sched.last_runs(tmp_path)["weekly"]
    assert last["status"] == "failed"
    assert "last: failed" in sched.describe_schedule(
        {"report": "weekly", "cron": "0 9 * * MON", "to": [], "timezone": None},
        last,
    )
    assert sched.last_runs(tmp_path / "missing") == {}


def test_an_empty_timezone_is_refused():
    with pytest.raises(ValueError, match="IANA"):
        sched.parse_schedule_block(
            {"cron": "0 9 * * MON", "timezone": ""}, path="r")


def test_in_server_logs_a_skipped_package_and_starts_nothing(
        tmp_path, monkeypatch, caplog):
    import logging

    pkg = tmp_path / "reports" / "weekly"
    pkg.mkdir(parents=True)
    (pkg / "template.html").write_text("<p></p>", encoding="utf-8")
    (pkg / "report.json").write_text('{"schedule": {"cron": "nope"}}',
                                     encoding="utf-8")
    monkeypatch.setenv("TRACEBI_SCHEDULES_IN_SERVER", "1")
    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(tmp_path / "reports"))
    caplog.set_level(logging.INFO, logger="tracebi.schedule")
    assert sched.start_server_scheduler() is None
    assert "skipped weekly" in caplog.text
    assert "no schedules" in caplog.text


def test_a_failed_run_is_recorded_not_raised(tmp_path):
    rec = sched.run_schedule({"report": "missing", **SCHEDULE},
                             reports_dir=tmp_path / "reports",
                             output_dir=tmp_path / "output")
    assert rec["status"] == sched.FAILED
    assert rec["attempts"] == 1
    assert "No report 'missing'" in rec["error"]
    assert sched.last_runs(tmp_path / "output")["missing"]["status"] == sched.FAILED


def _weekly_package(root: Path) -> None:
    pkg = root / "reports" / "weekly"
    pkg.mkdir(parents=True)
    (pkg / "template.html").write_text(
        "<!DOCTYPE html><html><body><p>weekly</p></body></html>",
        encoding="utf-8")
    (pkg / "report.json").write_text(json.dumps({
        "name": "weekly",
        "data": {"totals": {"model": "m", "query": {
            "fact": "f", "measures": ["n"]}}},
        "schedule": {"cron": "0 9 * * MON", "to": ["a@example.com"]},
    }), encoding="utf-8")


def _schedule_app():
    """The real schedule lifespan on a tiny app.

    Importing ``tracebi.web.api.main`` binds the routers to the global
    registry and breaks ``test_phase5.py`` when this file runs first.
    """
    from fastapi import FastAPI

    return FastAPI(lifespan=sched.server_lifespan)


def test_in_server_schedules_start_with_the_switch(tmp_path, monkeypatch, caplog):
    """Startup registers the package's job. It does not wait for cron."""
    import inspect
    import logging

    from fastapi.testclient import TestClient

    _weekly_package(tmp_path)
    monkeypatch.setenv("TRACEBI_SCHEDULES_IN_SERVER", "1")
    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(tmp_path / "reports"))
    # The MCP confinement root must not move the schedule log.
    monkeypatch.setenv("TRACEBI_OUTPUT_ROOT", str(tmp_path / "not-the-schedule-log"))
    app = _schedule_app()

    caplog.set_level(logging.INFO, logger="tracebi.schedule")
    with TestClient(app):
        scheduler = app.state.scheduler
        assert scheduler.running
        [job] = scheduler.get_jobs()
        assert job.id == "weekly"
        assert inspect.getclosurevars(job.func).nonlocals["output_dir"] == "output"
    assert scheduler.running is False
    text = caplog.text
    assert "weekly" in text
    assert "advisory lock per report" in text


def test_in_server_schedules_stay_off_by_default(monkeypatch):
    monkeypatch.delenv("TRACEBI_SCHEDULES_IN_SERVER", raising=False)
    assert sched.start_server_scheduler() is None


def test_in_server_schedules_need_apscheduler(tmp_path, monkeypatch):
    import sys

    _weekly_package(tmp_path)
    monkeypatch.setenv("TRACEBI_SCHEDULES_IN_SERVER", "1")
    monkeypatch.setenv("TRACEBI_REPORTS_DIR", str(tmp_path / "reports"))
    for name in ("apscheduler", "apscheduler.schedulers",
                 "apscheduler.schedulers.background",
                 "apscheduler.schedulers.blocking",
                 "apscheduler.triggers", "apscheduler.triggers.cron"):
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ImportError, match=r"tracebi\[pipeline\]"):
        sched.start_server_scheduler()


class TestSlackFileDelivery:
    """A configured bot token and channel upload the built files. The
    webhook ping is a different path. Nothing here calls Slack."""

    def _stub(self, monkeypatch, slack):
        emailed = []
        monkeypatch.setattr(
            "tracebi._delivery.send_report",
            lambda *a, **k: emailed.append(a) or ["cfo@example.com"],
        )
        monkeypatch.setattr("tracebi._delivery.slack_send_report", slack)
        return emailed

    def _env(self, monkeypatch, channel="C0123"):
        monkeypatch.setenv("TRACEBI_SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("TRACEBI_SLACK_CHANNEL", channel)

    def test_uploads_beside_email(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        posted = []

        def _slack(html, manifest, *, channel, token, verify_result=None):
            posted.append({
                "channel": channel, "token": token,
                "html": Path(html).name, "manifest": Path(manifest).name,
                "verdict": verify_result["verdict"],
            })
            return "posted"

        emailed = self._stub(monkeypatch, _slack)
        self._env(monkeypatch)
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.DELIVERED
        assert rec["attempts"] == 1
        assert rec["recipients"] == ["cfo@example.com"]
        assert rec["error"] is None
        assert rec["delivery"] == {
            "slack": {"channel": "C0123", "sent": True, "error": None}}
        assert len(emailed) == 1
        assert posted == [{
            "channel": "C0123", "token": "xoxb-test",
            "html": "sample_dashboard.html",
            "manifest": "sample_dashboard.html.manifest.json",
            "verdict": "reproduces",
        }]

    def test_slack_alone_delivers_when_there_is_no_email_list(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        posted = []
        emailed = self._stub(
            monkeypatch, lambda *a, **k: posted.append(k["channel"]) or "ok")
        self._env(monkeypatch)
        block = _scheduled(scheduled)
        block["to"] = []
        rec = sched.run_schedule(
            block, reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert emailed == []
        assert posted == ["C0123"]
        assert rec["status"] == sched.DELIVERED
        assert rec["recipients"] == []
        assert rec["delivery"]["slack"]["sent"] is True

    def test_missing_config_does_not_upload(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        called = []
        self._stub(monkeypatch, lambda *a, **k: called.append(1))
        monkeypatch.setenv("TRACEBI_SLACK_BOT_TOKEN", "xoxb-test")
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert called == []
        assert "delivery" not in rec
        assert rec["status"] == sched.DELIVERED

    def test_a_failed_upload_is_recorded_and_not_retried(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        calls = {"n": 0}

        def boom(*_a, **_k):
            calls["n"] += 1
            raise RuntimeError("slack upload failed")

        self._stub(monkeypatch, boom)
        self._env(monkeypatch)
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert calls["n"] == 1
        assert rec["attempts"] == 1
        assert rec["status"] == sched.DELIVERED
        assert rec["recipients"] == ["cfo@example.com"]
        assert rec["error"] is None
        assert rec["delivery"]["slack"]["sent"] is False
        assert rec["delivery"]["slack"]["error"] == (
            "RuntimeError: slack upload failed")

    def test_a_failed_slack_only_upload_is_failed_without_retry(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        calls = {"n": 0}

        def boom(*_a, **_k):
            calls["n"] += 1
            raise RuntimeError("slack upload failed")

        self._stub(monkeypatch, boom)
        self._env(monkeypatch)
        block = _scheduled(scheduled)
        block["to"] = []
        rec = sched.run_schedule(
            block, reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert calls["n"] == 1
        assert rec["attempts"] == 1
        assert rec["status"] == sched.FAILED
        assert rec["recipients"] == []
        assert rec["error"] == "RuntimeError: slack upload failed"
        assert rec["delivery"]["slack"]["error"] == rec["error"]

    def test_no_send_records_intent_and_does_not_call_slack(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        called = []
        self._stub(monkeypatch, lambda *a, **k: called.append(1))
        self._env(monkeypatch, channel="#reports")
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
            send=False,
        )
        assert called == []
        assert rec["status"] == sched.BUILT
        assert rec["recipients"] == []
        assert rec["delivery"] == {
            "slack": {"channel": "#reports", "sent": False, "error": None}}

    def test_a_built_page_summary_quotes_figures_the_page_already_shows(
            self, scheduled, monkeypatch):
        """The summary's numbers are text the build wrote, not a new format."""
        import json as _json
        from urllib.parse import parse_qs

        _enter(scheduled, monkeypatch)
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
            send=False,
        )
        assert rec["status"] == sched.BUILT
        html = scheduled / "output" / "sample_model" / "sample_dashboard.html"
        manifest = html.with_name(html.name + ".manifest.json")
        page = html.read_text(encoding="utf-8")
        calls = []

        class _Resp:
            def __init__(self, raw):
                self._raw = raw.encode()

            def read(self):
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def urlopen(req, timeout=None):
            calls.append(req)
            url = req.full_url
            if url.endswith("/files.getUploadURLExternal"):
                name = parse_qs(req.data.decode())["filename"][0]
                fid = "Fhtml" if name.endswith(".html") else "Fman"
                return _Resp(_json.dumps({
                    "ok": True,
                    "upload_url": f"https://files.example/upload/{fid}",
                    "file_id": fid,
                }))
            if url.startswith("https://files.example/"):
                return _Resp("OK")
            if url.endswith("/files.completeUploadExternal"):
                return _Resp(_json.dumps({"ok": True, "files": []}))
            raise AssertionError(url)

        monkeypatch.setattr(
            "tracebi._delivery.urllib.request.urlopen", urlopen)
        from tracebi._delivery import slack_send_report
        text = slack_send_report(
            html, manifest, channel="C0123", token="xoxb-test",
            verify_result={
                "verdict": "reproduces",
                "verdict_detail": "reproduces — matched",
                "exit_code": 0,
            },
        )
        assert "Report:" in text
        assert "Rendered at:" in text
        assert "Verify: reproduces — matched" in text
        headlines = [
            line for line in text.splitlines()
            if ": " in line and not line.startswith(
                ("Report:", "Rendered at:", "Verify:"))
        ]
        assert 1 <= len(headlines) <= 5
        for line in headlines:
            _label, _, value = line.partition(": ")
            assert value in page
        uploaded = [c.full_url.rsplit("/", 1)[-1] for c in calls]
        assert uploaded.count("files.getUploadURLExternal") == 2
        assert "files.completeUploadExternal" in uploaded

    def test_a_refused_receipt_is_not_uploaded(self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        called = []
        self._stub(monkeypatch, lambda *a, **k: called.append(1))
        self._env(monkeypatch)
        monkeypatch.setattr(
            "tracebi.verify.verify_manifest",
            lambda *a, **k: {
                "verdict": "unexplained",
                "verdict_detail": "stub: did not reproduce",
                "exit_code": 1, "ok": False,
            },
        )
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert called == []
        assert rec["status"] == sched.REFUSED
        assert "delivery" not in rec


def test_scheduler_fires_in_the_declared_timezone():
    pytest.importorskip("apscheduler")
    scheduler = sched.build_scheduler(
        [{"report": "weekly", **SCHEDULE}], job=lambda s: None, blocking=False)
    scheduler.start(paused=True)
    try:
        [job] = scheduler.get_jobs()
        assert job.id == "weekly"
        nxt = job.next_run_time
        assert (nxt.weekday(), nxt.hour, nxt.minute) == (0, 9, 0)
        assert str(nxt.tzinfo) == "America/New_York"
    finally:
        scheduler.shutdown(wait=False)


# The sample warehouse's regions, after the transform normalises them.
# Northeast is three orders; West is two. Midwest is in the filter so a
# missing recipient is recorded, and is not built.
_BURST = {
    "cron": "0 9 * * MON",
    "timezone": "America/New_York",
    "to": ["team@example.com"],
    "owner": "ops@example.com",
    "burst": {
        "filter": {"dim_region.region": ["Northeast", "West", "Midwest"]},
        "to": {
            "Northeast": ["ne@example.com"],
            "West": ["west@example.com"],
        },
    },
}


def _write_schedule(proj: Path, block: dict) -> None:
    rj = proj / "reports" / "sample_model" / "sample_dashboard" / "report.json"
    decl = json.loads(rj.read_text())
    decl["schedule"] = block
    rj.write_text(json.dumps(decl))


def _orders_in(html_path: Path) -> int:
    import csv as _csv
    import io
    import re
    html = html_path.read_text()
    block = re.search(
        r'<script[^>]*id="tracebi-data-kpis"[^>]*>(.*?)</script>',
        html, re.S).group(1)
    row = next(_csv.DictReader(io.StringIO(json.loads(block)["csv"])))
    return int(float(row["orders"]))


def _kpis_filters(manifest: dict) -> dict:
    for section in manifest.get("sections") or []:
        if section.get("id") != "kpis":
            continue
        lineage = section.get("dataset_lineage") or []
        return lineage[-1]["metadata"]["query_spec"]["filters"]
    raise AssertionError("manifest has no kpis section")


class TestBurst:
    def test_normalizes_a_list_filter_and_keeps_top_level_to(self):
        got = sched.parse_schedule_block(_BURST, path="r")
        assert got["to"] == ["team@example.com"]
        assert got["owner"] == "ops@example.com"
        assert got["burst"] == {
            "filter": {"dim_region.region": {
                "in": ["Northeast", "West", "Midwest"]}},
            "to": {
                "Northeast": ["ne@example.com"],
                "West": ["west@example.com"],
            },
        }

    @pytest.mark.parametrize("burst, fragment", [
        ({"filter": {"dim_region.region": ["EMEA"]}}, "needs a 'to' object"),
        ({"filter": {
            "dim_region.region": ["EMEA"],
            "dim_product.product": ["A"],
          }, "to": {}}, "one dimension key"),
        ({"filter": {"region": ["EMEA"]}, "to": {}}, "dimension reference"),
        ({"filter": {"dim_region.region": {"gte": 1}}, "to": {}},
         "list of values"),
        ({"filter": {"dim_region.region": []}, "to": {}}, "at least one"),
        ({"filter": {"dim_region.region": ["EMEA", "EMEA"]},
          "to": {"EMEA": ["a@example.com"]}}, "unique"),
        ({"filter": {"dim_region.region": ["EMEA"]},
          "to": {"EMEA": ["not-an-address"]}}, "email addresses"),
        ({"filter": {"dim_region.region": ["EMEA"]},
          "to": {"APAC": ["a@example.com"]}}, "not values of burst.filter"),
        ({"filter": {"dim_region.region": ["EMEA"]},
          "to": {"EMEA": ["a@example.com"]}, "when": "daily"},
         "unknown burst field"),
    ])
    def test_refuses(self, burst, fragment):
        with pytest.raises(ValueError, match=fragment):
            sched.parse_schedule_block(
                {"cron": "0 9 * * MON", "burst": burst},
                path="reports/x/report.json")

    def test_discover_reports_a_burst_without_to(self, scheduled):
        _write_schedule(scheduled, {
            "cron": "0 9 * * MON",
            "burst": {"filter": {"dim_region.region": {"in": ["EMEA"]}}},
        })
        schedules, errors = sched.discover_schedules(scheduled / "reports")
        assert schedules == []
        assert errors[0]["report"] == "sample_model/sample_dashboard"
        assert "needs a 'to' object" in errors[0]["error"]

    def test_one_build_per_value_with_its_own_recipients(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        _write_schedule(scheduled, _BURST)
        sent = []

        def _send(html, _manifest, to, subject=None, verify_result=None):
            sent.append({"to": list(to), "html": Path(html).name})
            return list(to)

        monkeypatch.setattr("tracebi._delivery.send_report", _send)
        block = _scheduled(scheduled)
        rec = sched.run_schedule(
            block,
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert rec["status"] == sched.DELIVERED
        assert rec["verdict"] == "reproduces"
        assert rec["recipients"] == ["ne@example.com", "west@example.com"]
        assert "alert" not in rec
        by = {s["value"]: s for s in rec["slices"]}
        assert set(by) == {"Northeast", "West", "Midwest"}
        assert by["Northeast"]["status"] == sched.DELIVERED
        assert by["West"]["status"] == sched.DELIVERED
        assert by["Midwest"]["status"] == sched.SKIPPED
        assert by["Midwest"]["output"] is None
        assert by["Midwest"]["recipients"] == []
        out = scheduled / "output" / "sample_model"
        ne = out / "sample_dashboard--Northeast.html"
        west = out / "sample_dashboard--West.html"
        assert ne.is_file()
        assert (out / (ne.name + ".manifest.json")).is_file()
        assert west.is_file()
        assert (out / (west.name + ".manifest.json")).is_file()
        assert not (out / "sample_dashboard--Midwest.html").exists()
        assert not (out / "sample_dashboard.html").exists()
        assert sent == [
            {"to": ["ne@example.com"], "html": ne.name},
            {"to": ["west@example.com"], "html": west.name},
        ]
        assert _orders_in(ne) == 3
        assert _orders_in(west) == 2
        manifest = json.loads((out / (ne.name + ".manifest.json")).read_text())
        assert _kpis_filters(manifest) == {"dim_region.region": "Northeast"}
        assert _runs(scheduled) == [rec]
        assert "burst: dim_region.region" in sched.describe_schedule(block)
        assert "× 3" in sched.describe_schedule(block)

    def test_a_refused_slice_is_not_sent_and_one_alert_names_it(
            self, scheduled, monkeypatch):
        _enter(scheduled, monkeypatch)
        _write_schedule(scheduled, _BURST)
        sent = []
        alerts = []

        def _send(_html, _manifest, to, subject=None, verify_result=None):
            sent.append(list(to))
            return list(to)

        monkeypatch.setattr("tracebi._delivery.send_report", _send)
        monkeypatch.setattr(
            "tracebi._delivery.send_alert",
            lambda to, subject, body: alerts.append(
                {"to": to, "subject": subject, "body": body}),
        )
        import tracebi.verify as verify_mod
        real = verify_mod.verify_manifest

        def wrapped(manifest, models, **k):
            if _kpis_filters(manifest).get("dim_region.region") == "West":
                return {
                    "verdict": "unexplained",
                    "verdict_detail": "stub: did not reproduce",
                    "exit_code": 1, "ok": False,
                }
            return real(manifest, models, **k)

        monkeypatch.setattr("tracebi.verify.verify_manifest", wrapped)
        rec = sched.run_schedule(
            _scheduled(scheduled),
            reports_dir=scheduled / "reports",
            output_dir=scheduled / "output",
        )
        assert sent == [["ne@example.com"]]
        by = {s["value"]: s for s in rec["slices"]}
        assert by["Northeast"]["status"] == sched.DELIVERED
        assert by["West"]["status"] == sched.REFUSED
        assert by["West"]["recipients"] == []
        assert by["Midwest"]["status"] == sched.SKIPPED
        assert rec["status"] == sched.REFUSED
        assert rec["verdict"] == "unexplained"
        assert "West:" in rec["error"]
        assert rec["alert"] == {
            "to": "ops@example.com", "sent": True, "error": None}
        [alert] = alerts
        assert alert["subject"] == (
            "TraceBi: sample_model/sample_dashboard scheduled run refused")
        assert "West (refused)" in alert["body"]
        assert "Northeast" not in alert["body"]
        assert len(_runs(scheduled)) == 1

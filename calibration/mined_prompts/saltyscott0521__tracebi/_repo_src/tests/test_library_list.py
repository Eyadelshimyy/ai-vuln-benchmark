"""Library columns on GET /api/reports (issue 187).

The Reports list reads schedule, last schedule run, stored-build count, and
last change from this payload. There is no owner field.
"""

import json
import os
from datetime import datetime, timezone

from tracebi.state import record_run

_TEMPLATE = "<!doctype html><html><body><p>{title}</p></body></html>\n"
_BINDING = {
    "model": "shelf_model",
    "query": {"fact": "fact_orders", "measures": ["revenue"]},
}

# Explicit mtimes so last_change is the newer of the two package files.
_REPORT_MTIME = 1_700_000_000
_TEMPLATE_MTIME = 1_700_086_400
_SPEC_MTIME = 1_699_000_000


def _package(root, rel, title, schedule=None):
    pkg = root / "reports" / rel
    pkg.mkdir(parents=True)
    decl = {"name": title, "data": {"totals": dict(_BINDING)}}
    if schedule is not None:
        decl["schedule"] = schedule
    (pkg / "report.json").write_text(json.dumps(decl), encoding="utf-8")
    (pkg / "template.html").write_text(_TEMPLATE.format(title=title), encoding="utf-8")
    return pkg


def _iso(stamp: int) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def _forget_shelf():
    """Drop this file's reports so a later rescan does not see them as removed."""
    from tracebi.registry import registry
    from tracebi.web import discovery

    for name in (
        "shelf/weekly", "shelf/plain", "shelf/by_region",
        "shelf/broken", "shelf_code",
    ):
        registry.remove_report(name)
        discovery._live_reports.pop(name, None)


def test_the_reports_list_carries_library_columns(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.registry import registry
    from tracebi.web import discovery
    from tracebi.web.api.routers import reports as reports_router

    weekly = _package(
        tmp_path, "shelf/weekly", "Weekly",
        schedule={
            "cron": "0 9 * * MON",
            "timezone": "America/New_York",
            "to": ["cfo@example.com"],
            "owner": "cfo@example.com",
        },
    )
    plain = _package(tmp_path, "shelf/plain", "Plain")
    spec = tmp_path / "reports" / "shelf" / "by_region.json"
    spec.write_text(json.dumps({
        "name": "By region",
        "sections": [{
            "type": "table", "title": "Revenue",
            "data": dict(_BINDING),
        }],
    }), encoding="utf-8")
    os.utime(weekly / "report.json", (_REPORT_MTIME, _REPORT_MTIME))
    os.utime(weekly / "template.html", (_TEMPLATE_MTIME, _TEMPLATE_MTIME))
    os.utime(plain / "report.json", (_REPORT_MTIME, _REPORT_MTIME))
    os.utime(plain / "template.html", (_REPORT_MTIME, _REPORT_MTIME))
    os.utime(spec, (_SPEC_MTIME, _SPEC_MTIME))

    monkeypatch.chdir(tmp_path)
    discovery.clear_discovery_report()
    discovery.auto_discover(str(tmp_path / "reports"))
    failed = [o for o in discovery.discovery_report() if o["status"] == "failed"]
    assert not failed, failed

    record_run(
        kind="schedule", target="shelf/weekly", status="failed",
        started="2026-01-01T00:00:00+00:00", finished="2026-01-01T00:01:00+00:00",
    )
    record_run(
        kind="schedule", target="shelf/weekly", status="built",
        started="2026-02-01T00:00:00+00:00", finished="2026-02-01T00:05:00+00:00",
    )
    record_run(
        kind="schedule", target="shelf/other", status="delivered",
        finished="2026-03-01T00:00:00+00:00",
    )
    record_run(kind="report_build", target="shelf/weekly", status="succeeded")
    record_run(kind="report_build", target="shelf/weekly", status="succeeded")
    record_run(kind="background_run", target="shelf/weekly", status="succeeded")
    record_run(kind="report_build", target="shelf/plain", status="succeeded")

    registry.add_report("shelf_code", lambda: None, "from code")
    try:
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        body = TestClient(app).get("/api/reports").json()
    finally:
        _forget_shelf()

    by_name = {row["name"]: row for row in body}
    weekly_row = by_name["shelf/weekly"]
    assert weekly_row["schedule"] == {
        "cron": "0 9 * * MON", "timezone": "America/New_York",
    }
    assert weekly_row["last_run"] == {
        "status": "built", "time": "2026-02-01T00:05:00+00:00",
    }
    assert weekly_row["past_builds"] == 2
    assert weekly_row["last_change"] == _iso(_TEMPLATE_MTIME)
    assert "owner" not in weekly_row

    plain_row = by_name["shelf/plain"]
    assert plain_row["schedule"] is None
    assert plain_row["last_run"] is None
    assert plain_row["past_builds"] == 1
    assert plain_row["last_change"] == _iso(_REPORT_MTIME)

    spec_row = by_name["shelf/by_region"]
    assert spec_row["schedule"] is None
    assert spec_row["last_run"] is None
    assert spec_row["past_builds"] == 0
    assert spec_row["last_change"] == _iso(_SPEC_MTIME)

    code_row = by_name["shelf_code"]
    assert code_row["schedule"] is None
    assert code_row["last_run"] is None
    assert code_row["past_builds"] == 0
    assert code_row["last_change"] is None
    assert "owner" not in code_row


def test_a_broken_schedule_block_does_not_fail_the_list(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tracebi.web import discovery
    from tracebi.web.api.routers import reports as reports_router

    pkg = _package(
        tmp_path, "shelf/broken", "Broken",
        schedule={"cron": "0 9 * * MON", "timezone": "America/New_York"},
    )
    monkeypatch.chdir(tmp_path)
    discovery.auto_discover(str(tmp_path / "reports"))
    # Discovery already registered the package. A later bad edit must not 500
    # the list; the schedule column is simply empty.
    decl = json.loads((pkg / "report.json").read_text(encoding="utf-8"))
    decl["schedule"] = "every monday"
    (pkg / "report.json").write_text(json.dumps(decl), encoding="utf-8")

    try:
        app = FastAPI()
        app.include_router(reports_router.router, prefix="/api")
        res = TestClient(app).get("/api/reports")
        assert res.status_code == 200
        row = next(r for r in res.json() if r["name"] == "shelf/broken")
        assert row["schedule"] is None
        assert row["last_change"] is not None
        assert "owner" not in row
    finally:
        _forget_shelf()

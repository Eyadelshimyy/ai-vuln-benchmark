"""Browser tests of Build mode: point at a figure, pin a note, watch the agent work.

Skipped unless ``TRACEBI_E2E=1`` (needs Chromium; see test_ui_smoke.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from tests.test_ui_smoke import _PROJECT, _free_port, _tail, _wait_health

pytestmark = pytest.mark.skipif(
    os.environ.get("TRACEBI_E2E") != "1",
    reason="set TRACEBI_E2E=1 to run the browser tests (needs the e2e extra)",
)


def _serve(project: Path, log_path: Path, dev_mode: bool):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRACEBI_")}
    env["TRACEBI_APP"] = ""
    if dev_mode:
        env["TRACEBI_DEV_MODE"] = "1"
    port = _free_port()
    log = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "tracebi.web.run", "--port", str(port), "--no-reload"],
        cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    _wait_health(f"{base}/api/health", server, log_path)
    return base, server, log


def test_pointing_at_a_figure_reaches_the_agent(tmp_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    project = tmp_path / "portfolio_project"
    shutil.copytree(_PROJECT, project, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "data", "output", ".tracebi"))
    done = subprocess.run([sys.executable, "run_workflow.py"], cwd=project,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr or done.stdout

    base, server, log = _serve(project, tmp_path / "server.log", dev_mode=True)
    report = "saas_model/mrr_dashboard"
    api = f"{base}/api/reports/{report}/workbench/pointing"

    def pointing():
        with urllib.request.urlopen(api, timeout=10) as r:
            return json.loads(r.read())["pointing"]

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(60_000)
            # The page of a report nobody has built yet: building is a click, never a side effect of looking.
            page.goto(f"{base}/m/saas_model/reports?r={report}&open=1")
            page.get_by_role("button", name="Build it now").click()
            page.locator("iframe").wait_for()
            frame = page.frame_locator("iframe")
            frame.locator("#kpi-mrr").wait_for()

            page.get_by_role("button", name="Build", exact=True).click()
            # The pane first shows the last build, then swaps to the live render; point
            # once that has happened (a click on the page being replaced is lost).
            page.locator(".wb-version", has_text="Opened").wait_for()
            frame.locator("#kpi-mrr").wait_for()
            assert pointing() is None
            frame.locator("#kpi-mrr").click()

            note = page.locator(".point-note")
            note.get_by_text("Pointing at").wait_for()
            assert "kpis → ending_mrr" in note.inner_text()
            got = pointing()
            assert (got["kind"], got["id"], got["binding"], got["cell"]) == (
                "figure", "kpi-mrr", "kpis", "ending_mrr"), got

            # Not a figure: a heading is described by where it is, not by a query.
            frame.locator("h1").click()
            page.wait_for_function(
                "() => /h1/.test(document.querySelector('.point-note')?.innerText || '')")
            got = pointing()
            assert got["kind"] == "element" and got["tag"] == "h1", got

            # The pane is a live view of the agent's work: it edits the package and
            # the report in the pane changes on its own, with what changed listed.
            template = project / "reports" / "saas_model" / "mrr_dashboard" / "template.html"
            template.write_text(template.read_text()
                .replace("Scale leads ending MRR", "Edited while you watch")
                .replace('data-tb-format="currency0" id="kpi-mrr"',
                         'data-tb-format="currency2" id="kpi-mrr"'))
            frame.get_by_role("heading", name="Edited while you watch").wait_for(timeout=15_000)
            newest = page.locator(".wb-version").first
            newest.get_by_text("kpi-mrr").wait_for(timeout=15_000)
            assert "→" in newest.inner_text()

            # Flip back to the report as it was before the edit, then to the latest.
            page.locator(".wb-version", has_text="Opened").click()
            frame.get_by_role("heading", name="Scale leads ending MRR").wait_for()
            page.get_by_role("button", name="Back to latest").click()
            frame.get_by_role("heading", name="Edited while you watch").wait_for()
            frame.locator("#kpi-mrr").wait_for()

            # Escape stops pointing; so does leaving Build mode.
            frame.locator("#kpi-mrr").click()
            page.keyboard.press("Escape")
            page.wait_for_function(
                "() => /Click a figure/.test(document.querySelector('.point-note')?.innerText || '')")
            # Hold the last "pointing at" request in flight while Build mode
            # is left: the clear must still land after it, or the agent keeps
            # acting on a figure nobody is looking at.
            page.evaluate("""() => {
                const real = window.fetch;
                window.fetch = (url, init) =>
                    String(url).endsWith('/workbench/pointing') && init && init.method === 'POST'
                        ? new Promise(r => setTimeout(r, 600)).then(() => real.call(window, url, init))
                        : real.call(window, url, init);
            }""")
            frame.locator("#kpi-mrr").click()
            page.get_by_role("button", name="Build", exact=True).click()
            page.wait_for_function("() => !document.querySelector('.point-note')")
            page.wait_for_timeout(1500)   # the held request has landed by now
            deadline = 50
            while pointing() is not None and deadline:
                page.wait_for_timeout(100)
                deadline -= 1
            assert pointing() is None, "leaving Point mode must stop pointing"
            browser.close()
    finally:
        log.close()
        server.terminate()
        server.wait(timeout=10)
    assert "Traceback" not in _tail(tmp_path / "server.log"), _tail(tmp_path / "server.log")


def test_build_mode_is_absent_unless_the_server_is_in_dev_mode(tmp_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    project = tmp_path / "portfolio_project"
    shutil.copytree(_PROJECT, project, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "data", "output", ".tracebi"))
    done = subprocess.run([sys.executable, "run_workflow.py"], cwd=project,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr or done.stdout

    base, server, log = _serve(project, tmp_path / "server.log", dev_mode=False)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(60_000)
            page.goto(f"{base}/m/saas_model/reports?r=saas_model/mrr_dashboard&open=1")
            page.get_by_role("button", name="Share").wait_for()   # the page is up, and has asked what the server is
            page.wait_for_load_state("networkidle")
            assert page.get_by_role("button", name="Build", exact=True).count() == 0
            browser.close()
        request = urllib.request.Request(
            f"{base}/api/reports/saas_model/mrr_dashboard/workbench/pointing",
            data=b'{"kind": "figure"}', method="POST",
            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=10)
        assert refused.value.code == 403
    finally:
        log.close()
        server.terminate()
        server.wait(timeout=10)


def test_the_dev_app_url_opens_in_build_mode(tmp_path: Path) -> None:
    """`tracebi dev <name> --app` opens /reports?r=<name>&build=1: the workbench is
    already beside the report, with nothing to click first."""
    from playwright.sync_api import sync_playwright

    project = tmp_path / "portfolio_project"
    shutil.copytree(_PROJECT, project, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "data", "output", ".tracebi"))
    done = subprocess.run([sys.executable, "run_workflow.py"], cwd=project,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr or done.stdout

    base, server, log = _serve(project, tmp_path / "server.log", dev_mode=True)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(60_000)
            page.goto(f"{base}/reports?r=saas_model%2Fmrr_dashboard&build=1")
            page.locator(".wb-version", has_text="Opened").wait_for()
            assert page.get_by_role("button", name="Build", exact=True).get_attribute("aria-pressed") == "true"
            browser.close()
    finally:
        log.close()
        server.terminate()
        server.wait(timeout=10)


def test_build_mode_shows_the_data_and_checks_and_the_project_feed(tmp_path: Path, monkeypatch) -> None:
    """What only the classic workbench page showed is in the app's panel (each binding
    with its size, and the checks), a note left there reaches the agent, and with no
    report open the Reports page carries the agent's exhibits from the project feed."""
    import pandas as pd
    from playwright.sync_api import sync_playwright

    from tracebi.workbench import show

    project = tmp_path / "portfolio_project"
    shutil.copytree(_PROJECT, project, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store", "data", "output", ".tracebi"))
    done = subprocess.run([sys.executable, "run_workflow.py"], cwd=project,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr or done.stdout

    base, server, log = _serve(project, tmp_path / "server.log", dev_mode=True)
    state_url = f"{base}/api/reports/saas_model/mrr_dashboard/workbench/state"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(60_000)
            page.goto(f"{base}/reports?r=saas_model%2Fmrr_dashboard&build=1")

            def section(title: str):
                return page.locator(".wb-section", has=page.locator(
                    ".wb-section__title", has_text=title))

            kpis = section("Data").locator(".wb-item", has_text="kpis")
            kpis.wait_for()
            text = kpis.inner_text()
            assert "from saas_model" in text and "used by" in text, text
            assert re.search(r"\d+ × \d+", text), text
            section("Checks").get_by_text("All clear").wait_for()

            page.get_by_label("Leave your agent a note").fill("make the legend smaller")
            page.get_by_role("button", name="Leave note").click()
            section("Your notes").get_by_text("make the legend smaller").wait_for()
            with urllib.request.urlopen(state_url, timeout=30) as r:
                pins = json.loads(r.read())["pins"]
            assert [(p["kind"], p["note"]) for p in pins] == [("message", "make the legend smaller")]

            # No report open: the project feed. The script posts the way the agent's do,
            # with no setup, because the dev server's heartbeat is live.
            monkeypatch.chdir(project)
            monkeypatch.delenv("TRACEBI_WORKBENCH_DIR", raising=False)
            show(pd.DataFrame({"region": ["East", "West"], "revenue": [10.5, 20.5]}),
                 note="revenue by region", name="region_scan")
            page.goto(f"{base}/reports")
            feed = page.locator(".wb--project")
            feed.get_by_text("region_scan").wait_for()
            assert "East" in feed.inner_text()
            browser.close()
    finally:
        log.close()
        server.terminate()
        server.wait(timeout=10)
    assert "Traceback" not in _tail(tmp_path / "server.log"), _tail(tmp_path / "server.log")

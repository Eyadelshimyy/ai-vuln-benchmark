"""Browser smoke of the served app.

Skipped unless ``TRACEBI_E2E=1``. A normal ``pytest tests/`` does not need
Chromium or the ``e2e`` extra. The CI job ``ui-smoke`` sets the variable.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TRACEBI_E2E") != "1",
    reason="set TRACEBI_E2E=1 to run the browser smoke (needs the e2e extra)",
)

_REPO = Path(__file__).resolve().parents[1]
_PROJECT = _REPO / "examples" / "portfolio_project"
_REPORT = "portfolio_model/portfolio_showcase"   # its path: it lives in a folder
_LABEL = "portfolio_showcase"            # what the list shows under the folder
_TITLE = "Portfolio Showcase"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_health(url: str, proc: subprocess.Popen, log_path: Path, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"server exited {proc.returncode}\n{_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = str(exc)
        time.sleep(0.25)
    raise AssertionError(f"health never came up ({last})\n{_tail(log_path)}")


def _tail(path: Path, n: int = 40) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def test_real_app_smoke(tmp_path: Path) -> None:
    from playwright.sync_api import expect, sync_playwright

    project = tmp_path / "portfolio_project"
    shutil.copytree(
        _PROJECT,
        project,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "data", ".tracebi"),
    )
    # What a fresh checkout has: no pages built on this machine earlier, so
    # opening a report builds it (and records the run the Runs step reads).
    for built in (project / "output").rglob("*.html"):
        built.unlink()
    workflow = subprocess.run(
        [sys.executable, "run_workflow.py"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert workflow.returncode == 0, workflow.stderr or workflow.stdout

    port = _free_port()
    env = os.environ.copy()
    env["TRACEBI_APP"] = ""
    for key in (
        "TRACEBI_REPORTS_DIR",
        "TRACEBI_MODELS_DIR",
        "TRACEBI_PIPELINES_DIR",
        "TRACEBI_DEV_MODE",
        "TRACEBI_OUTPUT_ROOT",
    ):
        env.pop(key, None)
    log_path = tmp_path / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, "-m", "tracebi.web.run", "--port", str(port), "--no-reload"],
        cwd=project,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_health(f"{base}/api/health", server, log_path)
        errors: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(180_000)
            page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
            page.on(
                "console",
                lambda msg: errors.append(f"console {msg.type}: {msg.text}")
                if msg.type == "error"
                else None,
            )

            def fail_on_browser_errors() -> None:
                # pageerror is delivered just after goto returns.
                page.wait_for_timeout(200)
                assert not errors, "\n".join(errors)

            # Two models and no remembered pick: / opens the list of models.
            page.goto(base + "/")
            page.get_by_role("heading", name="Overview", exact=True).wait_for()
            assert page.url.rstrip("/").endswith("/models"), page.url
            fail_on_browser_errors()

            # Picking one opens its Overview: how it is made, as three stations
            # that are links, each with an address of its own. Reports is the
            # one open to start with.
            page.get_by_role("link", name="portfolio_model").first.click()
            page.wait_for_url("**/m/portfolio_model")
            line = page.get_by_role("navigation", name="How portfolio_model is made")

            def station(name: str):
                return line.get_by_role("link", name=re.compile(rf"^{name}\b"))

            expect(station("Reports")).to_have_attribute("aria-current", "page")
            # Its facts are read from what exists: the transform the pipeline
            # runs, the warehouse file it writes, the model's tables and joins.
            expect(station("Transform")).to_contain_text("holdings_transform")
            expect(line).to_contain_text("warehouse.duckdb")
            expect(station("Model")).to_contain_text(re.compile(r"\d+ tables · \d+ joins"))
            station("Model").click()
            page.wait_for_url("**/m/portfolio_model/model")
            expect(station("Model")).to_have_attribute("aria-current", "page")
            station("Transform").click()
            page.wait_for_url("**/m/portfolio_model/transform")
            page.go_back()
            page.wait_for_url("**/m/portfolio_model/model")
            fail_on_browser_errors()

            # Beneath the switcher: Explore narrows to the same model.
            page.get_by_role("link", name="Explore", exact=True).click()
            page.wait_for_url("**/m/portfolio_model/explore")
            page.get_by_role("heading", name="Explore", exact=True).wait_for()
            assert page.get_by_label("Model").input_value() == "portfolio_model"
            fail_on_browser_errors()

            # The pick is remembered: / now opens that model's Overview.
            page.goto(base + "/")
            page.wait_for_url("**/m/portfolio_model")

            # The Map is every model as one picture: opening one zooms in, and
            # picking a measure says in words which reports ask for it.
            page.get_by_role("link", name="Map", exact=True).click()
            page.wait_for_url("**/map")
            page.get_by_role("button", name="Open portfolio_model").click()
            page.wait_for_url("**/map?model=portfolio_model")
            page.get_by_role("button", name=re.compile(r"^measure: fv_share")).click()
            expect(page.get_by_label("Selection")).to_contain_text("portfolio_concentration")
            fail_on_browser_errors()

            # Old addresses land on the page that replaced them.
            page.goto(base + "/models/portfolio_model?tab=refresh")
            page.wait_for_url("**/m/portfolio_model/transform")
            page.goto(base + "/m/portfolio_model/refresh")
            page.wait_for_url("**/m/portfolio_model/transform")
            page.goto(base + "/m/portfolio_model/sources")
            page.wait_for_url("**/m/portfolio_model/transform")
            page.goto(base + "/refresh")
            page.wait_for_url("**/models")

            # The Reports stage is the list with the last build of the picked
            # report beside it. A fresh checkout has built nothing, and looking at a
            # report does not build it: it says so, marked, and offers to build it.
            page.goto(base + "/m/portfolio_model")
            row = page.get_by_role("button", name=re.compile(rf"^{_LABEL}\b"))
            row.click()
            page.wait_for_url(f"**?r={_REPORT.replace('/', '%2F')}")
            expect(row).to_have_attribute("aria-pressed", "true")
            expect(row).to_contain_text("○ Never built")
            expect(row).to_contain_text("Mondays 07:00")   # its schedule, in words
            page.get_by_text("There is no page to show yet").wait_for()
            assert page.locator("iframe").count() == 0
            # The files that define it are one click away, read-only, with the way to a draft.
            page.get_by_role("button", name="Config", exact=True).click()
            page.get_by_role("button", name="report.json").wait_for()
            page.get_by_role("button", name="Edit as a draft").wait_for()
            page.get_by_role("button", name="Preview", exact=True).click()
            page.get_by_role("button", name="Build it now").click()
            try:
                page.locator("iframe").wait_for()
            except Exception as exc:
                fail_on_browser_errors()
                raise AssertionError(page.inner_text("body")[:4000]) from exc
            expect(row).not_to_contain_text("Never built")
            fail_on_browser_errors()

            # Open gives the report the whole page.
            page.get_by_role("link", name="Open").click()
            page.wait_for_url("**open=1")
            page.locator("iframe.rp__frame").wait_for()
            page.wait_for_function(
                """(title) => {
                    const frame = document.querySelector('iframe');
                    if (!frame) return false;
                    return (frame.srcdoc || '').includes(title);
                }""",
                arg=_TITLE,
            )

            # The d3 figure (tracebi.draw) follows the page: drawn in the
            # theme's accent, and redrawn from the server's selection when the
            # sector filter changes, as the declared figures are.
            report = page.frame_locator("iframe")
            report.get_by_role("button", name="Detail", exact=True).click()
            cells = report.locator("#fig-treemap g.cell rect")
            cells.first.wait_for()
            assert cells.count() == 13
            accent = report.locator("html").evaluate(
                "el => getComputedStyle(el).getPropertyValue('--tb-accent').trim()")
            assert cells.first.get_attribute("fill") == accent
            report.locator("select[data-tb-column='dim_issuer.sector']").select_option("Software")
            report.locator("#fig-treemap g.sector").first.wait_for()
            page.wait_for_function(
                """() => document.querySelector('iframe').contentDocument
                    .querySelectorAll('#fig-treemap g.cell').length === 2""")
            fail_on_browser_errors()

            # An open report is a page of its own: the nav stays as a rail, the list
            # does not, and the breadcrumb's Reports link goes back to its model.
            rail = page.get_by_role("navigation", name="Main")
            rail.get_by_role("link", name="Overview").wait_for()
            assert not page.get_by_placeholder("Search reports…").is_visible()
            back = page.get_by_role("link", name="Reports", exact=True)
            back.wait_for()

            # Export is one menu of the three files, and says which carries the receipt.
            page.get_by_role("button", name="Export").click()
            menu = page.get_by_role("menu")
            links = menu.get_by_role("menuitem").all()
            assert [link.inner_text().split("\n")[0] for link in links] == ["PDF", "Excel", "HTML"]
            assert [link.get_attribute("href").rsplit("=", 1)[1] for link in links] == ["pdf", "xlsx", "html"]
            assert "Only the HTML file carries its receipt" in menu.inner_text()
            page.keyboard.press("Escape")
            assert menu.count() == 0

            # Lineage, the check and the code are a panel beside the report.
            page.get_by_role("button", name="Lineage").click()
            page.get_by_text("Where the numbers on this page came from").wait_for()
            page.get_by_text("fact_holdings").first.wait_for()
            page.get_by_role("tab", name="Details").click()
            page.get_by_text("fact_holdings: the sink satisfied its contract").wait_for()
            page.get_by_role("button", name="Check again").wait_for()
            page.get_by_role("tab", name="Code").click()
            page.get_by_role("button", name="report.json").wait_for()
            page.get_by_role("button", name="Close the panel").click()
            assert page.get_by_role("tab", name="Code").count() == 0

            downloaded = page.evaluate(
                """async (url) => {
                    const response = await fetch(url);
                    return {status: response.status, body: await response.text()};
                }""",
                f"/api/reports/{_REPORT}/download?format=html",
            )
            assert downloaded["status"] == 200, _tail(log_path)
            assert _TITLE in downloaded["body"]

            # Rebuild makes more build rows, and the stage re-checks the new one:
            # the sentence under the list says the built report still re-runs to
            # the same numbers, and its row says when it was built.
            page.get_by_role("button", name="Rebuild").click()
            page.get_by_text("Report ran successfully").wait_for()
            back.click()
            page.get_by_text(re.compile(r"re-runs? to the same numbers")).wait_for()
            expect(row).to_contain_text("Built ")

            # A build whose source changed since never reads as fine: its row, the
            # sentence under the list and the bar of its page say so, in words with a
            # glyph, and the sentence does not claim it re-runs.
            def source_changed(route):
                desk = route.fetch().json()
                for built in desk["builds"]:
                    built["verdict"] = "source_drift"
                route.fulfill(json=desk)

            page.route("**/api/desk", source_changed)
            page.reload()
            expect(row).to_contain_text("△ Source changed")
            expect(page.get_by_role("button", name=re.compile(r"^Needs a look \(\d+\)"))).to_be_visible()
            page.get_by_text("0 of 1 built reports re-run to the same numbers").wait_for()
            page.get_by_role("link", name="Open").click()
            page.get_by_text("Source changed since").wait_for()
            bar = page.get_by_role("status").filter(has_text="Source changed.")
            expect(bar).to_contain_text("The data changed since it was built")
            assert "re-run to the same" not in bar.inner_text()
            page.unroute("**/api/desk")

            # The all-models page keeps its list; a row there opens the same page.
            page.goto(base + "/reports")
            fail_on_browser_errors()
            page.get_by_text(_LABEL, exact=True).click()
            page.wait_for_url("**/reports?r=*")
            page.locator("iframe").wait_for()
            assert page.get_by_role("heading", name=_LABEL, exact=True).is_visible()

            # A build is not a check: History shows the list's verdict on the newest
            # build of the report, and a plain "Not checked" (never green) on
            # every older one.
            page.goto(base + "/runs")
            page.get_by_role("heading", name="History", exact=True).wait_for()
            rows = page.locator("tr", has_text=_REPORT)
            rows.first.wait_for()
            page.get_by_text("Checking…").wait_for(state="detached")
            verdicts = [r.locator("td").last.inner_text() for r in rows.all()]
            assert len(verdicts) >= 2, verdicts
            assert verdicts[0] == "Reproduces", verdicts
            assert set(verdicts[1:]) == {"Not checked"}, verdicts
            fail_on_browser_errors()

            # Given room, a report's three KPI cards sit on one row.
            page.goto(f"{base}/reports?r=portfolio_model/portfolio_overview")
            page.get_by_role("button", name="Build it now").click()
            cards = page.frame_locator("iframe").locator(".tb-kpi")
            cards.nth(2).wait_for()
            tops = {round(card.bounding_box()["y"]) for card in cards.all()}
            assert cards.count() == 3 and len(tops) == 1, tops

            # Each stage can show the code behind what it shows.
            page.goto(base + "/m/portfolio_model/model")
            page.get_by_role("button", name="Code", exact=True).click()
            page.get_by_text("models/portfolio_model.yaml").first.wait_for()
            page.goto(base + "/m/portfolio_model/transform")
            page.get_by_role("button", name="Code", exact=True).click()
            page.get_by_role("button", name="transform", exact=True).click()
            page.get_by_text("transforms/holdings_transform.py").first.wait_for()
            # The Transform stage says what the warehouse holds, and that the sink
            # satisfied its contract, before any report is opened.
            table = page.locator("details", has_text="fact_holdings")
            table.wait_for()
            assert re.search(r"[\d,]+ rows", table.locator("summary").inner_text())
            assert table.locator("summary").get_by_text("satisfied", exact=True).count() == 1
            fail_on_browser_errors()
            # Explore leads with the model's own measures (each with what it means),
            # sends their names, and writes the results in their declared formats.
            page.goto(base + "/m/portfolio_model/explore")
            page.get_by_text("Total fair value").wait_for()
            # However long the builder is, Run stays in view.
            expect(page.get_by_text("▶ Run query")).to_be_in_viewport(ratio=1)
            page.get_by_text("fair_value", exact=True).first.click()
            page.get_by_text("positions", exact=True).first.click()
            page.get_by_text("sector", exact=True).first.click()
            page.get_by_text("▶ Run query").click()
            page.get_by_text("The query as code").wait_for()
            assert 'measures=["fair_value", "positions"]' in page.locator(".codeview__code").inner_text()
            assert re.search(r"\$[\d,]+", page.locator("table tbody tr").first.inner_text())
            fail_on_browser_errors()

            # Group by offers only what the picked fact joins.
            page.goto(base + "/m/housing_model/explore")
            builder = page.locator(".explore-grid")
            expect(builder).to_contain_text("dim_year")
            page.get_by_role("button", name="fact_ten_year").click()
            expect(builder).not_to_contain_text("dim_year")
            expect(builder).to_contain_text("dim_cohort")
            fail_on_browser_errors()

            # Transform: Run all opens the Runs tab, which follows the run to its end…
            page.goto(base + "/m/portfolio_model/transform")
            # The page reads the warehouse as it opens; a transform that starts
            # writing it meanwhile fails, so let those reads finish first.
            page.wait_for_load_state("networkidle")
            page.get_by_role("button", name="Run all").click()
            output = page.get_by_role("log", name="Run output")
            output.get_by_text("Run succeeded").wait_for()
            text = output.inner_text()
            assert "[transform] ✓" in text and "[build] ✓" in text, text
            assert page.get_by_role("list", name="Runs").get_by_role("button").count() == 1
            # …the Overview's header now says when it was refreshed…
            page.get_by_text("Last refreshed").wait_for()
            # …and the run links the reports it rebuilt, each to its page in the app.
            page.get_by_role("link", name="portfolio_book", exact=True).click()
            page.wait_for_url("**/m/portfolio_model/reports?r=portfolio_model%2Fportfolio_book")
            fail_on_browser_errors()

            # That refresh is a run of its model: its History (the link in the
            # Overview's header) lists it, and All models names the model beside it.
            page.get_by_role("link", name="History", exact=True).click()
            page.wait_for_url("**/m/portfolio_model/runs")
            page.get_by_role("heading", name="History", exact=True).wait_for()
            page.locator("tr", has_text="Pipeline run").first.wait_for()
            page.goto(base + "/runs")
            pipeline_row = page.locator("tr", has_text="Pipeline run").first
            pipeline_row.wait_for()
            assert "portfolio_model" in pipeline_row.inner_text()
            fail_on_browser_errors()

            # Drafts: a report draft on disk is listed, rendered live, follows
            # edits to its file without a reload, and publishes with a toast.
            draft = project / "drafts" / "anonymous" / "reports" / "portfolio_model" / "draft_demo"
            shutil.copytree(project / "reports" / "portfolio_model" / "portfolio_overview", draft)
            template = draft / "template.html"
            template.write_text(
                template.read_text(encoding="utf-8").replace("Portfolio Overview", "Draft Marker One"),
                encoding="utf-8")
            page.goto(base + "/drafts")
            page.get_by_role("heading", name="Drafts", exact=True).wait_for()
            row = page.get_by_role("list", name="Drafts").get_by_role("link")
            assert "portfolio_model/draft_demo" in row.inner_text()
            assert "changed" in row.inner_text().lower()
            row.click()
            page.wait_for_url("**/drafts/anonymous/reports/portfolio_model/draft_demo")
            page.get_by_role("heading", name="portfolio_model/draft_demo", exact=True).wait_for()
            page.wait_for_function(
                "(t) => (document.querySelector('iframe')?.srcdoc || '').includes(t)",
                arg="Draft Marker One")
            template.write_text(
                template.read_text(encoding="utf-8").replace("Draft Marker One", "Draft Marker Two"),
                encoding="utf-8")
            page.wait_for_function(
                "(t) => (document.querySelector('iframe')?.srcdoc || '').includes(t)",
                arg="Draft Marker Two")
            page.get_by_role("button", name="Files", exact=True).click()
            page.get_by_role("button", name="template.html").wait_for()
            page.get_by_role("button", name="Publish", exact=True).click()
            page.get_by_role("button", name="Publish for everyone").click()
            page.get_by_text("Published report portfolio_model/draft_demo").wait_for()
            page.get_by_role("link", name="Open the published report").wait_for()
            assert (project / "reports" / "portfolio_model" / "draft_demo" / "report.json").is_file()
            fail_on_browser_errors()
            browser.close()
        assert not errors, "\n".join(errors)
    finally:
        log_file.close()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)

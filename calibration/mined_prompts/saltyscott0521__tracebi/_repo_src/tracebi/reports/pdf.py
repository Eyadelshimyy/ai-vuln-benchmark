"""Print a built report package to PDF.

The PDF is the self-contained HTML (``output/<name>.html``) opened in
headless Chromium, so the page runtime draws every figure — charts included —
before ``page.pdf()``. It carries no receipt. ``HTMLRenderer.render_pdf``
(WeasyPrint) is a separate path and is not used here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Both failures name the same two commands: the extra, then the browser.
_INSTALL = (
    "Install with: pip install 'tracebi[pdf]'\n"
    "Then install the browser: python -m playwright install chromium"
)

# Set on <html> at the end of hydration (tracebi.js). A chart that still
# shows only its no-JS SVG fallback has not been drawn.
_DRAWN = """() => {
  const root = document.documentElement;
  if (!root || root.getAttribute("data-tb-ready") !== "1") return false;
  const charts = document.querySelectorAll('[data-tb-figure="chart"]');
  for (const el of charts) {
    if (el.querySelector("canvas")) continue;
    if (el.querySelector("svg:not(.tb-chart-fallback)")) continue;
    if (el.querySelector(".tb-chart-fallback")) return false;
  }
  return true;
}"""


def _launch_args() -> list[str]:
    """Chromium's sandbox cannot start as the image's non-root user."""
    if os.environ.get("TRACEBI_IN_DOCKER") == "1":
        return ["--no-sandbox", "--disable-dev-shm-usage"]
    return []


def print_pdf(html_path, pdf_path) -> None:
    """Print the built HTML at *html_path* to *pdf_path*.

    Opens the file with Playwright Chromium (``file://``; the artifact is
    self-contained, so no network), waits until hydration has set
    ``data-tb-ready`` and every chart figure has been drawn, then prints
    with backgrounds.

    A missing ``playwright`` package raises ``ImportError``. A missing
    Chromium raises ``RuntimeError``. Both name ``pip install 'tracebi[pdf]'``
    and ``python -m playwright install chromium``.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required to print a built report to PDF.\n" + _INSTALL
        ) from exc

    html = Path(html_path)
    if not html.is_file():
        raise FileNotFoundError(f"No built HTML at {html_path}")
    dest = Path(pdf_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=True, args=_launch_args())
            except Exception as exc:
                raise RuntimeError(
                    "Playwright could not launch Chromium to print a report "
                    "to PDF.\n" + _INSTALL + f"\n({type(exc).__name__}: {exc})"
                ) from exc
            try:
                page = browser.new_page()
                page.goto(html.resolve().as_uri(), wait_until="load")
                try:
                    page.wait_for_function(_DRAWN, timeout=30_000)
                except Exception as exc:
                    raise RuntimeError(
                        "The built report did not finish drawing before it "
                        "could be printed (html[data-tb-ready], and a chart "
                        "canvas when the page has charts). "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                page.pdf(path=str(dest), print_background=True)
            finally:
                browser.close()
    except (ImportError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not print {html_path} to PDF. {type(exc).__name__}: {exc}"
        ) from exc

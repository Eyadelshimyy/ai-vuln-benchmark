"""
Report presentation: themes and the page shell.

The HTML renderer used to hold its stylesheet as a module constant and build
the page with f-strings, so restyling or restructuring output meant editing
(or forking) the renderer. Two seams here replace that:

* **:class:`Theme`** — the stylesheet as data. Swap it wholesale, load one
  from a file, or layer overrides on the default. This covers the common
  case: make reports look like your brand.
* **A page-shell template** — the document around the sections. Supply a
  Jinja2 template to add a header, a footer, a nav, or an analytics tag
  without touching section rendering.

Section *internals* stay in Python deliberately. Table formatting and chart
geometry are logic, not layout, and expressing them as templates would make
them harder to read and harder to test. What you override is the shell, the
styling, and — via the renderer's section registry — whole new block types.

Jinja2 is optional: the built-in shell needs no template engine, so reports
work on a base install. It is required only when you supply your own
template, which is what the declared ``jinja2`` dependency is actually for.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

# The default stylesheet. Every rule here is overridable — see
# Theme.with_overrides() to layer on top, or Theme.from_file() to replace.
DEFAULT_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Calibri, Arial, sans-serif;
    font-size: 13px;
    color: #1a1a2e;
    background: #f5f7fa;
    padding: 0;
}
.page {
    max-width: 1100px;
    margin: 0 auto;
    background: #ffffff;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    padding: 48px 56px 64px 56px;
}
/* Cover */
.cover {
    background: linear-gradient(135deg, #1F3864 0%, #2E74B5 100%);
    color: #fff;
    padding: 48px;
    border-radius: 6px;
    margin-bottom: 40px;
}
.cover h1 { font-size: 28px; font-weight: 700; margin-bottom: 10px; }
.cover p  { font-size: 14px; opacity: 0.85; margin-bottom: 6px; }
.cover .meta-table { margin-top: 20px; border-collapse: collapse; }
.cover .meta-table td {
    padding: 5px 18px 5px 0;
    font-size: 12px;
    opacity: 0.9;
}
.cover .meta-table td:first-child { font-weight: 600; opacity: 1; }

/* Sections */
.section { margin-bottom: 36px; }

/* Text styles */
.text-heading1 {
    font-size: 20px; font-weight: 700;
    color: #1F3864;
    border-bottom: 2px solid #2E74B5;
    padding-bottom: 6px;
    margin-bottom: 18px;
}
.text-heading2 {
    font-size: 15px; font-weight: 600;
    color: #2E74B5;
    margin-bottom: 12px;
}
.text-normal { font-size: 13px; line-height: 1.6; color: #333; }
.text-note {
    background: #FFF2CC;
    border-left: 4px solid #F5C518;
    padding: 10px 14px;
    font-size: 12px;
    border-radius: 0 4px 4px 0;
}
.text-callout {
    background: #DEEBF7;
    border-left: 4px solid #2E74B5;
    padding: 10px 14px;
    font-size: 12px;
    border-radius: 0 4px 4px 0;
}
.section-title {
    font-size: 14px; font-weight: 600;
    color: #1F3864;
    margin-bottom: 10px;
}

/* Tables */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 4px;
}
.data-table thead tr {
    background: #1F3864;
    color: #fff;
}
.data-table thead th {
    padding: 9px 12px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
.data-table thead th.num { text-align: right; }
.data-table tbody tr:nth-child(even) { background: #EBF0F7; }
.data-table tbody tr:hover { background: #d6e4f0; }
.data-table tbody td {
    padding: 7px 12px;
    border-bottom: 1px solid #dde4ef;
}
.data-table tbody td.num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tfoot tr {
    background: #D6DCE4;
    font-weight: 700;
    color: #1F3864;
}
.data-table tfoot td {
    padding: 8px 12px;
    border-top: 2px solid #1F3864;
}
.data-table tfoot td.num { text-align: right; }

.data-table tbody td.neg { color: #C62828; }

/* Metric cards */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #f8fafd;
    border: 1px solid #dde4ef;
    border-left: 4px solid #2E74B5;
    border-radius: 6px;
    padding: 14px 18px;
}
.metric-label {
    font-size: 11px; font-weight: 600;
    color: #5a6b8a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-value { font-size: 24px; font-weight: 700; color: #1F3864; margin-top: 4px; }
.metric-delta { font-size: 12px; font-weight: 600; margin-top: 4px; }
.metric-delta.good { color: #2E7D32; }
.metric-delta.bad  { color: #C62828; }

/* Side-by-side layout rows */
.layout-row { display: flex; gap: 24px; align-items: flex-start; }
.layout-row > .layout-col { min-width: 0; }

/* Charts — these classes apply to the print/PDF (WeasyPrint) inline-SVG path
   only; interactive HTML draws via ECharts, which these do not touch. */
.chart-container {
    text-align: center;
    margin: 8px 0;
}
.chart-container img { max-width: 100%; height: auto; }
.tb-chart { width: 100%; height: auto; max-height: 460px; font-family: inherit; }
.tb-grid          { stroke: #E3E7EE; stroke-width: 1; }
.tb-tick          { fill: #6B7280; font-size: 11px; }
.tb-cat           { fill: #374151; font-size: 11px; }
.tb-axis-label    { fill: #374151; font-size: 12px; font-weight: 600; }
.tb-legend        { fill: #374151; font-size: 11.5px; }
.tb-value         { fill: #374151; font-size: 10.5px; }
.tb-chart-empty   { fill: #9CA3AF; font-size: 13px; font-style: italic; }
.tb-bar, .tb-slice { stroke: #FFFFFF; stroke-width: 0.75; }
.tb-point         { stroke: #FFFFFF; stroke-width: 1; }

/* Lineage */
.lineage-toggle {
    margin-top: 48px;
    border-top: 1px solid #dde4ef;
    padding-top: 20px;
}
.lineage-toggle summary {
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    color: #2E74B5;
    padding: 6px 0;
    list-style: none;
}
.lineage-toggle summary::-webkit-details-marker { display: none; }
.lineage-toggle summary::before { content: "▶  "; }
details[open] summary::before { content: "▼  "; }
.lineage-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    margin-top: 12px;
}
.lineage-table th {
    background: #2F5496;
    color: #fff;
    padding: 6px 10px;
    text-align: left;
}
.lineage-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #eee;
    vertical-align: top;
}
.lineage-table tr:nth-child(even) td { background: #f5f7fa; }
.badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-load      { background: #DEEBF7; color: #1F3864; }
.badge-filter    { background: #E2EFDA; color: #375623; }
.badge-transform { background: #FFF2CC; color: #7D6608; }
.badge-join      { background: #FCE4D6; color: #843C0C; }
.badge-sort      { background: #EAD1DC; color: #4A235A; }
.badge-select    { background: #D9EAD3; color: #274E13; }
.badge-rename    { background: #CFE2F3; color: #1C4587; }
.badge-other     { background: #eeeeee; color: #333; }

/* Print */
@media print {
    body { background: #fff; }
    .page { box-shadow: none; padding: 20px; }
    .lineage-toggle { display: none; }
    details { display: none; }
}
"""


@dataclasses.dataclass(frozen=True)
class Theme:
    """
    A report's styling, as data.

    ``Theme.default()`` is the built-in look. ``from_file`` replaces it
    wholesale; ``with_overrides`` appends rules so later declarations win,
    which is usually what you want for a brand tweak.
    """
    name: str = "default"
    css: str = DEFAULT_CSS

    @classmethod
    def default(cls) -> "Theme":
        return cls()

    @classmethod
    def from_file(cls, path: str | Path, name: Optional[str] = None) -> "Theme":
        """Use a stylesheet from disk instead of the built-in one."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Theme stylesheet not found: {p}")
        return cls(name=name or p.stem, css=p.read_text(encoding="utf-8"))

    @classmethod
    def from_css(cls, css: str, name: str = "custom") -> "Theme":
        return cls(name=name, css=css)

    def with_overrides(self, css: str, name: Optional[str] = None) -> "Theme":
        """
        Append rules to this theme.

        Appended rather than prepended so the overrides win at equal
        specificity — the behaviour you want when nudging a few colours.
        """
        return Theme(
            name=name or f"{self.name}+overrides",
            css=f"{self.css}\n\n/* ── overrides ─────────────────────── */\n{css}",
        )


DEFAULT_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{css}
</style>
{head_extra}
</head>
<body>
<div class="page">
  {cover}
  {sections}
  {lineage}
</div>
{body_extra}
</body>
</html>"""


def render_shell(
    *,
    title: str,
    css: str,
    cover: str,
    sections: str,
    lineage: str,
    template: Optional[str] = None,
    head_extra: str = "",
    body_extra: str = "",
    context: Optional[dict] = None,
) -> str:
    """
    Assemble the page.

    With no *template* this uses the built-in shell and needs no template
    engine. Pass a Jinja2 template string to take over the document
    structure; the same names are available, plus anything in *context*.
    """
    values = {
        "title": title, "css": css, "cover": cover,
        "sections": sections, "lineage": lineage,
        "head_extra": head_extra, "body_extra": body_extra,
        **(context or {}),
    }
    if template is None:
        html = DEFAULT_SHELL.format(**values)
        # Empty extras would otherwise leave stray blank lines in the output.
        return html.replace("</style>\n\n</head>", "</style>\n</head>") \
                   .replace("</div>\n\n</body>", "</div>\n</body>")

    try:
        from jinja2 import Environment, StrictUndefined
    except ImportError as exc:
        raise ImportError(
            "A custom report template requires Jinja2. "
            "Install with: pip install 'tracebi[web]' (or `pip install jinja2`). "
            "The built-in shell needs no template engine."
        ) from exc

    # StrictUndefined so a typo'd placeholder fails loudly instead of
    # rendering an empty string into the report.
    env = Environment(autoescape=False, undefined=StrictUndefined)
    return env.from_string(template).render(**values)

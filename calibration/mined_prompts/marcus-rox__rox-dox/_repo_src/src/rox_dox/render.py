from __future__ import annotations

import html
import posixpath
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from urllib.parse import quote

from rox_dox.block_svg import block_svg
from rox_dox.diagrams import (
    emit_diagram_warnings,
    sequence_plantuml,
)
from rox_dox.links import page_href, source_url
from rox_dox.model import (
    Claim,
    Page,
    Related,
    Sequence as SequenceDiagram,
    Source,
    SummaryTable,
    page_sources,
)
from rox_dox.plantuml import DiagramError, render_svg
from rox_dox.repo_tree import RepoEntry
from rox_dox.schema import Table
from rox_dox.schema_svg import schema_svg
from rox_dox.tree import SiteTree


NO_ENTRIES: Mapping[str, Sequence[RepoEntry]] = MappingProxyType({})


PAGE_CSS = """\
:root {
  color-scheme: light;
  --bg: #f4f6f9;
  --surface: #fff;
  --surface-2: #f7f8fa;
  --surface-hover: #f0f3f7;
  --paper: #fff;
  --text: #182230;
  --text-2: #26323f;
  --text-3: #344054;
  --muted: #667085;
  --muted-2: #475467;
  --faint: #7a8594;
  --icon: #8a94a6;
  --boilerplate: #9ca3af;
  --border: #e2e7ee;
  --border-input: #d0d5dd;
  --link: #2458a6;
  --link-hover: #173d78;
  --accent: #145bc4;
  --current-bg: #e3ecfb;
  --current-fg: #0f3f8c;
  --focus: #3b6fd8;
  --pk: #b45309;
  --fk: #2563eb;
  --blob: #d97706;
  color: var(--text);
  background: var(--bg);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-synthesis: none;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0f141b;
  --surface: #171d26;
  --surface-2: #1d2531;
  --surface-hover: #222b38;
  --paper: #171d26;
  --text: #e6ebf2;
  --text-2: #d5dce6;
  --text-3: #c3ccd8;
  --muted: #97a3b4;
  --muted-2: #aab4c3;
  --faint: #7f8b9c;
  --icon: #7f8b9c;
  --boilerplate: #6b7686;
  --border: #2a3443;
  --border-input: #3a4556;
  --link: #7eb0ff;
  --link-hover: #a9c9ff;
  --accent: #7eb0ff;
  --current-bg: #1f3354;
  --current-fg: #cfe0ff;
  --focus: #7eb0ff;
  --pk: #f0a04b;
  --fk: #60a5fa;
  --blob: #f3b25a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0f141b;
    --surface: #171d26;
    --surface-2: #1d2531;
    --surface-hover: #222b38;
    --paper: #171d26;
    --text: #e6ebf2;
    --text-2: #d5dce6;
    --text-3: #c3ccd8;
    --muted: #97a3b4;
    --muted-2: #aab4c3;
    --faint: #7f8b9c;
    --icon: #7f8b9c;
    --boilerplate: #6b7686;
    --border: #2a3443;
    --border-input: #3a4556;
    --link: #7eb0ff;
    --link-hover: #a9c9ff;
    --accent: #7eb0ff;
    --current-bg: #1f3354;
    --current-fg: #cfe0ff;
    --focus: #7eb0ff;
    --pk: #f0a04b;
    --fk: #60a5fa;
    --blob: #f3b25a;
  }
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  line-height: 1.6;
}
.panel-toggle {
  position: fixed;
  top: 0;
  left: 0;
  margin: 0;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
}
.page-shell {
  display: grid;
  grid-template-columns: 280px 1.25rem minmax(0, 1fr);
  grid-template-areas: "nav handle main";
  align-items: start;
  gap: 0.5rem;
  max-width: 1600px;
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}
#toggle-left:checked ~ .page-shell {
  grid-template-columns: 0 1.25rem minmax(0, 1fr);
}
#toggle-left:checked ~ .page-shell #sidebar {
  display: none;
}
.panel-handle {
  grid-area: handle;
  align-self: stretch;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  width: 1.25rem;
  cursor: pointer;
}
.panel-handle::before {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 1px;
  background: var(--border);
  transition: background 0.15s;
}
.panel-handle:hover::before {
  background: var(--focus);
}
.panel-handle-grip {
  position: sticky;
  top: calc(50vh - 1.1rem);
  z-index: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.25rem;
  height: 2.2rem;
  border: 1px solid var(--border-input);
  border-radius: 999px;
  background: var(--surface);
  color: var(--muted);
  transition: color 0.15s, border-color 0.15s;
}
.panel-handle-grip svg {
  width: 12px;
  height: 12px;
  transition: transform 0.2s;
}
.panel-handle:hover .panel-handle-grip {
  color: var(--text);
  border-color: var(--focus);
}
#toggle-left:checked ~ .page-shell .panel-handle-grip svg {
  transform: rotate(180deg);
}
#toggle-left:focus-visible ~ .page-shell .panel-handle-grip {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
}
.panel-controls {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.theme-switch {
  display: inline-flex;
  margin: 0;
  padding: 2px;
  border: 1px solid var(--border-input);
  border-radius: 999px;
  background: var(--surface);
}
.theme-switch button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.85rem;
  height: 1.85rem;
  padding: 0;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--muted);
  font: inherit;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.theme-switch button:hover {
  color: var(--text);
}
.theme-switch button svg {
  width: 16px;
  height: 16px;
}
:root:not([data-theme]) .theme-switch [data-theme-choice="system"],
:root[data-theme="light"] .theme-switch [data-theme-choice="light"],
:root[data-theme="dark"] .theme-switch [data-theme-choice="dark"] {
  background: var(--current-bg);
  color: var(--current-fg);
}
.theme-switch button:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 1px;
}
#sidebar {
  grid-area: nav;
  position: sticky;
  top: 0;
  max-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
#toc, #feature-tree, #site-nav {
  padding: 1rem;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface);
}
#toc {
  flex: 0 0 auto;
}
#feature-tree, #site-nav {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
}
#toc > h2, #feature-tree > h2, #site-nav > h2 {
  margin: 0 0 0.75rem;
  font-size: 1.1rem;
}
.page-main {
  grid-area: main;
  min-width: 0;
}
#site-nav > h2, #explorer > h2 {
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.page-tree, .page-tree ul {
  margin: 0;
  padding: 0;
  list-style: none;
}
.page-tree {
  font: 13px/1.2 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.page-tree ul {
  margin-left: 11px;
  padding-left: 6px;
  border-left: 1px solid var(--border);
}
.page-tree summary {
  list-style: none;
  cursor: pointer;
}
.page-tree summary::-webkit-details-marker {
  display: none;
}
.page-tree .row {
  display: flex;
  align-items: center;
  gap: 4px;
  height: 22px;
  padding: 0 6px 0 2px;
  border-radius: 4px;
  color: var(--text-2);
  text-decoration: none;
  white-space: nowrap;
}
.page-tree .row span {
  overflow: hidden;
  text-overflow: ellipsis;
}
.page-tree .row:hover {
  background: var(--surface-hover);
}
.page-tree .row.current {
  background: var(--current-bg);
  color: var(--current-fg);
  font-weight: 600;
}
.page-tree .row.pending, .page-tree .row.file {
  color: var(--faint);
}
.page-tree .chevron, .page-tree .chevron-spacer {
  flex: 0 0 14px;
  width: 14px;
  height: 14px;
  color: var(--faint);
  transition: transform 0.1s;
}
.page-tree details[open] > summary .chevron {
  transform: rotate(90deg);
}
.page-tree .icon, .repo-tree .icon {
  flex: 0 0 15px;
  width: 15px;
  height: 15px;
  color: var(--icon);
}
.repo-tree .row {
  display: flex;
  align-items: center;
  gap: 4px;
  min-height: 22px;
  color: var(--text-2);
  text-decoration: none;
}
.page-tree .row.page .icon {
  color: var(--accent);
}
.breadcrumbs {
  margin: 0.25rem 0 1rem;
  color: var(--muted);
}
.breadcrumbs a {
  margin-right: 0.4rem;
}
.page-header {
  margin-bottom: 1.5rem;
  padding: 1.5rem 1.75rem;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface);
}
h1, h2, h3 {
  line-height: 1.25;
}
.page-header h1 {
  margin: 0;
  font-size: clamp(1.8rem, 4vw, 2.5rem);
}
.page-meta {
  margin: 0.5rem 0 0;
  color: var(--muted);
  overflow-wrap: anywhere;
}
.page-children {
  margin: 0.5rem 0 0;
}
.repo-tree {
  margin: 0;
  padding: 0;
  list-style: none;
}
section {
  margin: 1rem 0;
  padding: 1.5rem 1.75rem;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--surface);
}
section h2 {
  margin: 0 0 1rem;
  font-size: 1.35rem;
}
h3 {
  margin: 1.25rem 0 0.5rem;
  font-size: 1.05rem;
}
p {
  margin: 0.65rem 0;
}
p, li, th, td, summary, code {
  overflow-wrap: anywhere;
}
a {
  color: var(--link);
  text-decoration-thickness: 1px;
  text-underline-offset: 0.15em;
}
a:hover {
  color: var(--link-hover);
}
.toc, .claim-list, .related-list {
  margin: 0.5rem 0;
  padding-left: 1.35rem;
}
.toc li, .claim-list li, .related-list li {
  margin: 0.35rem 0;
}
.figure-notes {
  font-size: 0.88rem;
}
.figure-notes .claim-list {
  margin-top: 0.3rem;
  margin-bottom: 0.5rem;
}
.citations {
  margin-left: 0.2rem;
  max-width: 100%;
  overflow-wrap: anywhere;
  white-space: normal;
}
details.citations {
  display: inline-block;
  vertical-align: baseline;
}
details.citations summary {
  cursor: pointer;
}
.citation-links {
  overflow-wrap: anywhere;
  white-space: normal;
}
.summary-claim {
  margin: 0.65rem 0;
}
.table-sources {
  max-width: 100%;
  overflow-wrap: anywhere;
}
.membership-table .citations {
  white-space: normal;
}
.citations sup {
  display: inline-block;
  margin-right: 0.2rem;
}
.table-scroll {
  max-width: 100%;
  overflow-x: auto;
}
table {
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
  margin: 0.6rem 0;
}
th, td {
  padding: 0.65rem 0.75rem;
  border: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--surface-2);
  font-weight: 600;
}
.diagram-card {
  margin: 1rem 0 1.5rem;
}
.diagram-card h3 {
  margin-top: 0;
}
.diagram {
  max-width: 100%;
  overflow-x: auto;
  padding: 0.75rem;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--paper);
}
.figure-bar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
}
.figure-bar .collapse, .zoom-figure:target .figure-bar .expand {
  display: none;
}
.zoom-figure:target .figure-bar .collapse {
  display: inline;
}
.zoom-figure:target {
  position: fixed;
  inset: 0;
  z-index: 10;
  margin: 0;
  padding: 1rem 1.5rem;
  overflow: auto;
  background: var(--bg);
}
.zoom-figure:target .diagram {
  border: 0;
}
.block-scroll svg {
  max-width: none !important;
  margin: 0 !important;
}
.block-svg a:hover text {
  text-decoration: underline;
}
.schema-legend {
  margin: 0 0 0.75rem;
  color: var(--muted-2);
  font-size: 0.85rem;
}
.schema-legend-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.4rem;
  margin: 0.4rem 0;
  padding: 0;
  list-style: none;
}
.schema-legend-list li {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.schema-pk {
  color: var(--pk);
  font-weight: 700;
}
.schema-fk {
  color: var(--fk);
  font-weight: 600;
}
.schema-boilerplate {
  color: var(--boilerplate);
}
.schema-line {
  display: inline-block;
  flex: 0 0 34px;
  width: 34px;
  height: 0;
  border-top-width: 2px;
  border-top-style: solid;
}
.schema-line-enforced {
  color: var(--fk);
}
.schema-line-symbolic {
  color: var(--boilerplate);
  border-top-style: dashed;
}
.schema-line-blob {
  color: var(--blob);
  border-top-style: dotted;
}
.schema-table-list {
  font-family: Menlo, Consolas, monospace;
  line-height: 1.8;
}
.diagram svg {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}
.block-scroll svg {
  max-width: none;
}
.sources {
  margin-top: 0.75rem;
  color: var(--muted-2);
}
.sources summary {
  cursor: pointer;
  font-weight: 600;
}
.empty {
  color: var(--muted);
  font-style: italic;
}
blockquote {
  margin: 0.75rem 0;
  padding-left: 1rem;
  border-left: 3px solid var(--border);
  color: var(--muted-2);
}
.related-list li {
  overflow-wrap: anywhere;
}
#related summary {
  cursor: pointer;
  list-style: none;
}
#related summary::-webkit-details-marker {
  display: none;
}
#related summary h2 {
  display: inline-block;
  margin: 0;
}
#related details[open] summary {
  margin-bottom: 0.75rem;
}
#related summary h2::before {
  content: "▸ ";
  color: var(--faint);
}
#related details[open] summary h2::before {
  content: "▾ ";
}
@media (max-width: 1100px) {
  .page-shell {
    grid-template-columns: minmax(0, 1fr);
    grid-template-areas:
      "nav"
      "main";
    padding: 1rem 0.75rem 2rem;
  }
  #toggle-left:checked ~ .page-shell {
    grid-template-columns: minmax(0, 1fr);
  }
  #sidebar {
    position: static;
    max-height: none;
  }
  .panel-handle {
    display: none;
  }
  .page-header, section {
    padding: 1rem;
  }
  th, td {
    padding: 0.5rem;
  }
}"""

SVG_DARK_TOKENS = {
    "ink": "#e6ebf2",
    "muted": "#97a3b4",
    "faint": "#7f8b9c",
    "arrow": "#c3ccd8",
    "divider": "#3a4556",
    "card": "#1d2531",
    "store-fill": "#16263f",
    "store-stroke": "#7eb0ff",
    "queue-fill": "#12301d",
    "queue-stroke": "#4ade80",
    "external-fill": "#2a1f3d",
    "external-stroke": "#b58cf0",
    "schema-header": "#2a3443",
    "schema-row": "#2a3443",
    "schema-store": "#2a2416",
    "pk": "#f0a04b",
    "fk": "#60a5fa",
    "blob": "#f3b25a",
    "shadow": "#000000",
    "lifeline": "#4b5666",
}
BLOCK_SVG_PALETTE = {
    "#1f2937": "ink",
    "#6b7280": "muted",
    "#374151": "arrow",
    "#d1d5db": "divider",
    "#ffffff": "card",
    "#eff6ff": "store-fill",
    "#1e40af": "store-stroke",
    "#f0fdf4": "queue-fill",
    "#166534": "queue-stroke",
    "#9ca3af": "faint",
}
SCHEMA_SVG_PALETTE = {
    "#111827": "ink",
    "#6b7280": "muted",
    "#9ca3af": "faint",
    "#1f2937": "schema-header",
    "#f1f5f9": "schema-row",
    "#cbd5e1": "divider",
    "#ffffff": "card",
    "#fffbeb": "schema-store",
    "#b45309": "pk",
    "#92400e": "pk",
    "#2563eb": "fk",
    "#d97706": "blob",
    "#0f172a": "shadow",
}
PLANTUML_SVG_PALETTE = {
    "#000": "ink",
    "#000000": "ink",
    "#333": "arrow",
    "#333333": "arrow",
    "#00f": "store-stroke",
    "#e8f0fe": "store-fill",
    "#3b6fd8": "store-stroke",
    "#9aa4b2": "lifeline",
    "#fff4e5": "schema-store",
    "#e08a1e": "pk",
    "#f3e8fd": "external-fill",
    "#8b4fd1": "external-stroke",
    "#e7f6ec": "queue-fill",
    "#2f9e5b": "queue-stroke",
}
SVG_COLOR_ATTRIBUTE_PATTERN = re.compile(
    r'(fill|stroke|flood-color)(="|:)(#[0-9a-fA-F]{3,6})\b'
)


def _svg_token_css(indent: str) -> str:
    return "".join(
        f"{indent}--svg-{name}: {value};\n" for name, value in SVG_DARK_TOKENS.items()
    )


PAGE_CSS = re.sub(
    r"^( *)--blob: #f3b25a;\n",
    lambda match: match.group() + _svg_token_css(match.group(1)),
    PAGE_CSS,
    flags=re.MULTILINE,
)

SECTION_LINKS = (
    ("related", "Related"),
    ("tldr", "TLDR"),
    ("block", "Block diagram"),
    ("schema", "Schema"),
    ("sequences", "Sequence diagrams"),
)
EMPTY_MESSAGE = "Nothing to show for this module."
SVG_NAMESPACE_ATTRIBUTES = (
    ' xmlns="http://www.w3.org/2000/svg"',
    ' xmlns:xlink="http://www.w3.org/1999/xlink"',
)
URL_SCHEME_PATTERN = re.compile(r"https?://", re.IGNORECASE)
ANCHOR_TAG_PATTERN = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
HREF_ATTRIBUTE_PATTERN = re.compile(r"""\bhref\s*=\s*(["']).*?\1""", re.IGNORECASE)


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _source_href(source: Source, *, page: Page, repo_url: str) -> str:
    return source_url(source, repo_url=repo_url, commit=page.commit)


def _anchor(url: str, label: str) -> str:
    return f'<a href="{_escape(url)}">{_escape(label)}</a>'


def _citation_links(sources: list[Source], *, page: Page, repo_url: str) -> str:
    references = (
        f"<sup>{_anchor(_source_href(source, page=page, repo_url=repo_url), f'[{number}]')}</sup>"
        for number, source in enumerate(sources, start=1)
    )
    joined = " ".join(references)
    if not joined:
        return ""
    if len(sources) > 3:
        return (
            '<details class="citations">'
            f"<summary>Sources ({len(sources)})</summary>"
            f'<span class="citation-links">{joined}</span></details>'
        )
    return f'<span class="citations">{joined}</span>'


def _claim_text(claim: Claim, *, page: Page, repo_url: str) -> str:
    return (
        f"{_escape(claim.text)}"
        f"{_citation_links(claim.sources, page=page, repo_url=repo_url)}"
    )


def _claim_list(
    claims: list[Claim],
    *,
    page: Page,
    repo_url: str,
) -> str:
    items = "".join(
        f"<li>{_claim_text(claim, page=page, repo_url=repo_url)}</li>"
        for claim in claims
    )
    return f'<ul class="claim-list">{items}</ul>'


def _table_html(page: Page, table: SummaryTable, *, repo_url: str) -> str:
    headings = "".join(
        f'<th scope="col">{_escape(column)}</th>' for column in table.columns
    )
    links = {(link.row, link.column): link.page for link in table.links}
    rows = "".join(
        "<tr>"
        + "".join(
            "<td>"
            + (
                _anchor(page_href(page.id, links[(row_index, column_index)]), cell)
                if (row_index, column_index) in links
                else _escape(cell)
            )
            + "</td>"
            for column_index, cell in enumerate(row)
        )
        + "</tr>"
        for row_index, row in enumerate(table.rows)
    )
    citations = _citation_links(table.sources, page=page, repo_url=repo_url)
    return (
        '<div class="table-scroll"><table>'
        f"<thead><tr>{headings}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        f'<div class="table-sources">Sources: {citations}</div>'
    )


def _tldr_html(page: Page, *, repo_url: str) -> str:
    summary = "".join(
        f'<div class="summary-claim">{_claim_text(claim, page=page, repo_url=repo_url)}</div>'
        for claim in page.tldr.summary
    )
    key_points = (
        _claim_list(page.tldr.key_points, page=page, repo_url=repo_url)
        if page.tldr.key_points
        else f'<p class="empty">{EMPTY_MESSAGE}</p>'
    )
    notes = (
        _claim_list(page.tldr.notes, page=page, repo_url=repo_url)
        if page.tldr.notes
        else f'<p class="empty">{EMPTY_MESSAGE}</p>'
    )
    tables = [
        ("Table", page.tldr.table),
        *(
            (table.title or "Additional table", table)
            for table in page.tldr.additional_tables or []
        ),
    ]
    table_sections = "".join(
        f"<h3>{_escape(title)}</h3>{_table_html(page, table, repo_url=repo_url)}"
        for title, table in tables
    )
    return (
        "<h3>Summary</h3>"
        f"{summary}"
        "<h3>Key Points</h3>"
        f"{key_points}"
        f"{table_sections}"
        "<h3>Interesting Notes</h3>"
        f"{notes}"
    )


def _membership_section(page: Page, *, repo_url: str) -> str:
    if not page.membership:
        return ""

    groups = []
    for group in page.membership:
        if not group.rows:
            continue
        rows = "".join(
            "<tr>"
            f"<td><code>{_escape(row.path)}</code></td>"
            f"<td>{'Primary' if row.primary else 'Shared'}</td>"
            f"<td>{_escape(row.evidence)}"
            f"{_citation_links(row.sources, page=page, repo_url=repo_url)}</td>"
            "</tr>"
            for row in group.rows
        )
        shared_count = sum(not row.primary for row in group.rows)
        table = (
            '<div class="table-scroll membership-table">'
            "<table><thead><tr>"
            '<th scope="col">File</th><th scope="col">Primary/shared</th>'
            '<th scope="col">Evidence</th>'
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
        groups.append(
            "<details>"
            f"<summary>{_escape(group.layer)} — {len(group.rows)} files "
            f"({shared_count} shared)</summary>{table}</details>"
        )
    return _section("membership", "Why these files are one feature", "".join(groups))


def _section(section_id: str, title: str, content: str) -> str:
    return (
        f'<section id="{_escape(section_id)}">'
        f"<h2>{_escape(title)}</h2>{content}</section>"
    )


def _table_of_contents(
    sections: Sequence[tuple[str, str]] = SECTION_LINKS,
) -> str:
    links = "".join(
        f'<li><a href="#{section_id}">{_escape(title)}</a></li>'
        for section_id, title in sections
    )
    return (
        '<nav id="toc" aria-label="On this page">'
        "<h2>On this page</h2>"
        f'<ul class="toc">{links}</ul></nav>'
    )


def _sources_details(
    elements: Iterable[tuple[str, Source]],
    *,
    page: Page,
    repo_url: str,
) -> str:
    rows = "".join(
        "<tr>"
        f'<th scope="row">{_escape(label)}</th>'
        f"<td>{_anchor(_source_href(source, page=page, repo_url=repo_url), 'Source')}</td>"
        "</tr>"
        for label, source in elements
    )
    return (
        '<details class="sources"><summary>Sources</summary>'
        '<div class="table-scroll"><table><thead><tr>'
        '<th scope="col">Element</th><th scope="col">Citation</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div></details>"
    )


def _theme_svg(svg: str, palette: dict[str, str]) -> str:
    def substitute(match: re.Match[str]) -> str:
        attribute, separator, color = match.groups()
        token = palette.get(color.lower())
        if token is None:
            return match.group()
        return f"{attribute}{separator}var(--svg-{token}, {color})"

    return SVG_COLOR_ATTRIBUTE_PATTERN.sub(substitute, svg)


def _inline_svg(svg: str) -> str:
    for namespace_attribute in SVG_NAMESPACE_ATTRIBUTES:
        svg = svg.replace(namespace_attribute, "")
    return svg.replace("xlink:href=", "href=")


def _offline_html(document: str) -> str:
    parts: list[str] = []
    cursor = 0
    for anchor_match in ANCHOR_TAG_PATTERN.finditer(document):
        parts.append(_encode_url_schemes(document[cursor : anchor_match.start()]))
        anchor_tag = anchor_match.group()
        href_match = HREF_ATTRIBUTE_PATTERN.search(anchor_tag)
        if href_match is None:
            parts.append(_encode_url_schemes(anchor_tag))
        else:
            parts.extend(
                (
                    _encode_url_schemes(anchor_tag[: href_match.start()]),
                    href_match.group(),
                    _encode_url_schemes(anchor_tag[href_match.end() :]),
                )
            )
        cursor = anchor_match.end()
    parts.append(_encode_url_schemes(document[cursor:]))
    return "".join(parts)


def _encode_url_schemes(value: str) -> str:
    return URL_SCHEME_PATTERN.sub(
        lambda match: f"{match.group()[:-3]}&#58;//",
        value,
    )


def _diagram_markup(source: str, *, jar: Path) -> str:
    return (
        f'<div class="diagram block-scroll">'
        f"{_theme_svg(_inline_svg(render_svg(source, jar)), PLANTUML_SVG_PALETTE)}</div>"
    )


def _diagram_card(
    title: str,
    source: str,
    elements: Iterable[tuple[str, Source]],
    *,
    diagram_id: str,
    section_id: str,
    page: Page,
    repo_url: str,
    jar: Path,
) -> str:
    try:
        diagram_markup = _diagram_markup(source, jar=jar)
    except DiagramError as error:
        raise DiagramError(f"{title}: {error}") from error
    return (
        f'<article class="diagram-card zoom-figure" id="{_escape(diagram_id)}">'
        '<div class="figure-bar">'
        f"<h3>{_escape(title)}</h3>"
        f'<a class="expand" href="#{_escape(diagram_id)}">Expand full screen</a>'
        f'<a class="collapse" href="#{_escape(section_id)}">Close</a></div>'
        f"{diagram_markup}"
        f"{_sources_details(elements, page=page, repo_url=repo_url)}"
        "</article>"
    )


def _block_section(page: Page, *, repo_url: str) -> str:
    if not page.block.nodes and not page.block_figures:
        return _section(
            "block", "Block diagram", f'<p class="empty">{EMPTY_MESSAGE}</p>'
        )
    cards = []
    if page.block.nodes:
        elements = [
            (label, source)
            for label, source in page_sources(page)
            if label.startswith(
                ("block group ", "block node ", "block edge ", "block note ")
            )
        ]
        notes = (
            f'<div class="figure-notes">'
            f"{_claim_list(page.block.notes, page=page, repo_url=repo_url)}</div>"
            if page.block.notes
            else ""
        )
        unreached_files = ""
        if page.block.unreached_files:
            items = "".join(
                f"<li>{_escape(path)}</li>" for path in page.block.unreached_files
            )
            unreached_files = (
                '<details class="unreached-files">'
                "<summary>"
                f"Files not reached by a component "
                f"({len(page.block.unreached_files)})"
                "</summary>"
                f"<ul>{items}</ul>"
                "</details>"
            )
        cards.append(
            '<article class="diagram-card zoom-figure" id="block-figure">'
            '<div class="figure-bar"><h3>Figure 1. System overview</h3>'
            '<a class="expand" href="#block-figure">Expand full screen</a>'
            '<a class="collapse" href="#block">Close</a></div>'
            f'<div class="diagram block-scroll">'
            f"{_theme_svg(block_svg(page.block, page=page, repo_url=repo_url), BLOCK_SVG_PALETTE)}</div>"
            f"{notes}"
            f"{unreached_files}"
            f"{_sources_details(elements, page=page, repo_url=repo_url)}"
            "</article>"
        )
    for figure_number, figure in enumerate(page.block_figures, start=2):
        figure_prefix = f"block figure {figure.id} "
        elements = [
            (label, source)
            for label, source in page_sources(page)
            if label.startswith(figure_prefix)
        ]
        notes = (
            f'<div class="figure-notes">'
            f"{_claim_list(figure.notes, page=page, repo_url=repo_url)}</div>"
            if figure.notes
            else ""
        )
        figure_id = f"block-figure-{figure.id}"
        cards.append(
            f'<article class="diagram-card zoom-figure" id="{_escape(figure_id)}">'
            '<div class="figure-bar">'
            f"<h3>Figure {figure_number}. {_escape(figure.title)}</h3>"
            f'<a class="expand" href="#{_escape(figure_id)}">Expand full screen</a>'
            '<a class="collapse" href="#block">Close</a></div>'
            f"{notes}"
            f'<div class="diagram block-scroll">'
            f"{_theme_svg(block_svg(figure.block, page=page, repo_url=repo_url), BLOCK_SVG_PALETTE)}</div>"
            f"{_sources_details(elements, page=page, repo_url=repo_url)}"
            "</article>"
        )
    return _section("block", "Block diagram", "".join(cards))


def _schema_sources(
    page: Page,
    tables: Mapping[str, Table],
) -> list[tuple[str, Source]]:
    if page.data.domains:
        elements = [
            (
                f"schema domain {domain.id} key table {table_name}",
                tables[table_name].source,
            )
            for domain in page.data.domains
            for table_name in domain.key_tables
        ]
    else:
        elements = [
            (f"SQL table {table_name}", tables[table_name].source)
            for table_name in page.data.sql_tables
        ]
    elements.extend(
        (
            f"schema domain {domain.id} note {note_number}",
            source,
        )
        for domain in page.data.domains
        for note_number, note in enumerate(domain.notes, start=1)
        for source in note.sources
    )
    elements.extend(
        (f"NoSQL store {store.name}", store.source) for store in page.data.nosql
    )
    elements.extend(
        (f"relation {relation.src} -> {relation.dst}", relation.source)
        for relation in page.data.relations
    )
    return elements


def _schema_legend(page: Page) -> str:
    domain_note = (
        "<p>Each card is a domain; rows are its key tables; the keys the arrows use "
        "are listed on the right.</p>"
        if page.data.domains
        else ""
    )
    return (
        '<div class="schema-legend">'
        "<strong>Legend</strong>"
        '<ul class="schema-legend-list">'
        '<li><span class="schema-pk">PK</span> orange = primary / referenced key</li>'
        '<li><span class="schema-fk">FK</span> blue = referencing (FK-like) column</li>'
        '<li><span class="schema-line schema-line-enforced" aria-hidden="true"></span> '
        "solid blue = DB-enforced FK</li>"
        '<li><span class="schema-line schema-line-symbolic" aria-hidden="true"></span> '
        "dashed grey = symbolic reference (no constraint)</li>"
        '<li><span class="schema-line schema-line-blob" aria-hidden="true"></span> '
        "dotted amber = blob pointer</li>"
        "<li>dot = referenced (one) side; arrow = many side</li>"
        '<li><span class="schema-boilerplate">grey</span> = shared boilerplate columns</li>'
        "</ul>"
        f"{domain_note}</div>"
    )


def _schema_table_guide(
    page: Page,
    *,
    repo_url: str,
    tables: Mapping[str, Table],
) -> str:
    if not page.data.domains:
        return ""
    guides = []
    for domain in page.data.domains:
        links = " ".join(
            f'<a href="{_escape(_source_href(tables[table_name].source, page=page, repo_url=repo_url))}">'
            f"<code>{_escape(table_name)}</code></a>"
            for table_name in domain.tables
        )
        guides.append(
            f'<details id="schema-guide-{_escape(domain.id)}">'
            f"<summary>{_escape(domain.title)} — {len(domain.tables)} tables</summary>"
            f'<p class="schema-table-list">{links}</p></details>'
        )
    return f"<h3>Table guide</h3>{''.join(guides)}"


def _schema_section(
    page: Page,
    *,
    repo_url: str,
    tables: Mapping[str, Table],
) -> str:
    if (
        not page.data.domains
        and not page.data.sql_tables
        and not page.data.nosql
        and not page.data.relations
    ):
        return _section("schema", "Schema", f'<p class="empty">{EMPTY_MESSAGE}</p>')
    try:
        diagram = schema_svg(page, repo_url=repo_url, tables=tables)
    except DiagramError as error:
        raise DiagramError(f"Data stores: {error}") from error
    card = (
        '<article class="diagram-card zoom-figure" id="fig-schema">'
        '<div class="figure-bar"><h3>Data stores</h3>'
        '<a class="expand" href="#fig-schema">Expand full screen</a>'
        '<a class="collapse" href="#schema">Close</a></div>'
        f"{_schema_legend(page)}"
        f'<div class="diagram block-scroll">{_theme_svg(_inline_svg(diagram), SCHEMA_SVG_PALETTE)}</div>'
        f"{_sources_details(_schema_sources(page, tables), page=page, repo_url=repo_url)}"
        "</article>"
    )
    guide = _schema_table_guide(page, repo_url=repo_url, tables=tables)
    return _section("schema", "Schema", f"{card}{guide}")


def _sequence_elements(sequence: SequenceDiagram) -> list[tuple[str, Source]]:
    elements = [
        (
            f"sequence '{sequence.title}' participant {participant.id}",
            participant.source,
        )
        for participant in sequence.participants
    ]
    elements.extend(
        (
            f"sequence '{sequence.title}' step {step_number} {step.src}->{step.dst}",
            step.source,
        )
        for step_number, step in enumerate(sequence.steps, start=1)
    )
    return elements


def _sequence_section(page: Page, *, repo_url: str, jar: Path) -> str:
    if not page.sequences:
        return _section(
            "sequences",
            "Sequence diagrams",
            f'<p class="empty">{EMPTY_MESSAGE}</p>',
        )
    cards = "".join(
        _diagram_card(
            sequence.title,
            sequence_plantuml(page, sequence, repo_url=repo_url),
            _sequence_elements(sequence),
            diagram_id=f"fig-sequences-{number}",
            section_id="sequences",
            page=page,
            repo_url=repo_url,
            jar=jar,
        )
        for number, sequence in enumerate(page.sequences, start=1)
    )
    return _section("sequences", "Sequence diagrams", cards)


def _related_item(
    related: Related,
    *,
    page: Page,
    repo_url: str,
) -> str:
    target = (
        page_href(page.id, related.page)
        if related.page is not None
        else related.url or ""
    )
    label = _anchor(target, related.label)
    citations = _citation_links([related.source], page=page, repo_url=repo_url)
    return f"<li>{label}{citations}</li>"


def _related_section(page: Page, *, repo_url: str) -> str:
    if not page.related:
        content = f'<p class="empty">{EMPTY_MESSAGE}</p>'
    else:
        items = "".join(
            _related_item(related, page=page, repo_url=repo_url)
            for related in page.related
        )
        content = f'<ul class="related-list">{items}</ul>'
    return (
        '<section id="related"><details open>'
        "<summary><h2>Related</h2></summary>"
        f"{content}</details></section>"
    )


def _page_link(tree: SiteTree, from_id: str, to_id: str) -> str:
    return _anchor(page_href(from_id, to_id), tree.pages[to_id].title)


def _current_page_link(tree: SiteTree, from_id: str, page_id: str) -> str:
    return (
        f'<a class="current" aria-current="page" '
        f'href="{_escape(page_href(from_id, page_id))}">'
        f"{_escape(tree.pages[page_id].title)}</a>"
    )


CHEVRON_ICON = (
    '<svg class="chevron" viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>'
)
FOLDER_ICON = (
    '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 3.5h4.5l1.5 '
    '1.5h7v8h-13z" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>'
)
FILE_ICON = (
    '<svg class="icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 1.5h6l3 3v10h-9z'
    'M9.5 1.5v3h3" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>'
)
SPACER_ICON = '<span class="chevron-spacer" aria-hidden="true"></span>'


def _entry_href(entry: RepoEntry, *, page: Page, repo_url: str) -> str:
    kind = "tree" if entry.is_dir else "blob"
    return f"{repo_url.rstrip('/')}/{kind}/{page.commit}/{quote(entry.path, safe='/')}"


def _explorer_row(
    icon: str, label: str, href: str, css_class: str, title: str = ""
) -> str:
    title_attribute = f' title="{_escape(title)}"' if title else ""
    current = ' aria-current="page"' if "current" in css_class else ""
    return (
        f'<a class="row {css_class}" href="{_escape(href)}"{title_attribute}{current}>'
        f"{icon}<span>{_escape(label)}</span></a>"
    )


def _sort_key(item: tuple[bool, str, str]) -> tuple[int, str]:
    is_dir, label, _markup = item
    return (0 if is_dir else 1, label.lower())


def _relative_href(current_path: str, target_path: str) -> str:
    return quote(
        posixpath.relpath(target_path, posixpath.dirname(current_path) or "."),
        safe="/",
    )


def _folder_target(folder_path: str) -> str:
    return "folders/index.html" if folder_path == "." else f"folders/{folder_path}.html"


def _page_tree_item(
    tree: SiteTree,
    page_id: str,
    *,
    current_page_id: str | None,
    current_path: str,
    expanded_ids: set[str],
) -> str:
    page = tree.pages[page_id]
    css_class = "page current" if page_id == current_page_id else "page"
    child_ids = tree.children_of(page_id)
    chevron = CHEVRON_ICON if child_ids else SPACER_ICON
    row = _explorer_row(
        chevron + FOLDER_ICON,
        page.title,
        _relative_href(current_path, f"{page_id}.html"),
        css_class,
    )
    if not child_ids:
        return f"<li>{row}</li>"
    open_attribute = " open" if page_id in expanded_ids else ""
    children = "".join(
        _page_tree_item(
            tree,
            child_id,
            current_page_id=current_page_id,
            current_path=current_path,
            expanded_ids=expanded_ids,
        )
        for child_id in child_ids
    )
    return (
        f"<li><details{open_attribute}><summary>{row}</summary>"
        f"<ul>{children}</ul></details></li>"
    )


def _feature_tree_html(
    tree: SiteTree,
    *,
    current_path: str,
    current_page_id: str | None,
) -> str:
    expanded_ids = {tree.root}
    if current_page_id is not None:
        expanded_ids.update({current_page_id, *tree.ancestors(current_page_id)})
    root_item = _page_tree_item(
        tree,
        tree.root,
        current_page_id=current_page_id,
        current_path=current_path,
        expanded_ids=expanded_ids,
    )
    return (
        '<nav id="feature-tree" aria-label="Feature tree">'
        "<h2>Features</h2>"
        f'<ul class="page-tree">{root_item}</ul></nav>'
    )


def _file_explorer_html(
    tree: SiteTree,
    *,
    entries: Mapping[str, Sequence[RepoEntry]],
    repo_url: str,
    current_path: str,
    folders_only: bool = False,
    folder_path: str | None = None,
    child_folders: Sequence[tuple[str, str]] = (),
) -> str:
    root_page = tree.pages[tree.root]
    explorer = _explorer_html(
        root_page,
        entries=entries,
        current_path=current_path,
        repo_url=repo_url,
        folders_only=folders_only,
        folder_path=folder_path,
        child_folders=child_folders,
    )
    return f'<nav id="site-nav" aria-label="Site navigation">{explorer}</nav>'


def _explorer_html(
    root_page: Page,
    *,
    entries: Mapping[str, Sequence[RepoEntry]],
    current_path: str,
    repo_url: str,
    folders_only: bool = False,
    folder_path: str | None = None,
    child_folders: Sequence[tuple[str, str]] = (),
) -> str:
    items: list[tuple[bool, str, str]] = []
    if folder_path is not None:
        if folder_path != ".":
            parent = posixpath.dirname(folder_path) or "."
            href = _relative_href(current_path, _folder_target(parent))
            markup = _explorer_row(
                SPACER_ICON + FOLDER_ICON,
                "..",
                href,
                "folder",
            )
            items.append(
                (
                    True,
                    "..",
                    f"<li>{markup}</li>",
                )
            )
        folders = (
            child_folders
            if folder_path != "."
            else [
                (entry.name, entry.path)
                for entry in entries.get(root_page.id, [])
                if entry.is_dir
            ]
        )
        for name, path in folders:
            href = _relative_href(current_path, _folder_target(path))
            items.append(
                (
                    True,
                    name,
                    f"<li>{_explorer_row(SPACER_ICON + FOLDER_ICON, name, href, 'folder')}</li>",
                )
            )
        content = "".join(markup for _, _, markup in sorted(items, key=_sort_key))
        if not content:
            content = '<li class="empty">No subfolders.</li>'
        return (
            '<nav id="explorer" aria-label="Repository explorer">'
            "<h2>Explorer</h2>"
            f'<ul class="repo-tree">{content}</ul></nav>'
        )
    for entry in entries.get(root_page.id, []):
        if folders_only and not entry.is_dir:
            continue
        if entry.is_dir:
            href = _relative_href(current_path, _folder_target(entry.path))
            markup = _explorer_row(
                SPACER_ICON + FOLDER_ICON,
                entry.name,
                href,
                "folder",
            )
        else:
            href = _entry_href(entry, page=root_page, repo_url=repo_url)
            markup = _explorer_row(SPACER_ICON + FILE_ICON, entry.name, href, "file")
        items.append((entry.is_dir, entry.name, f"<li>{markup}</li>"))
    content = "".join(markup for _, _, markup in sorted(items, key=_sort_key))
    if not content:
        content = '<li class="empty">No repository entries.</li>'
    return (
        '<nav id="explorer" aria-label="Repository explorer">'
        "<h2>Explorer</h2>"
        f'<ul class="repo-tree">{content}</ul></nav>'
    )


def _breadcrumbs_html(tree: SiteTree, page: Page) -> str:
    ancestors = tree.ancestors(page.id)
    links = [_page_link(tree, page.id, ancestor) for ancestor in ancestors]
    links.append(_escape(page.title))
    trail = ' <span aria-hidden="true">›</span> '.join(links)
    return f'<nav class="breadcrumbs" aria-label="Breadcrumbs">{trail}</nav>'


def _children_html(tree: SiteTree, page: Page) -> str:
    child_ids = tree.children_of(page.id)
    if not child_ids:
        return ""
    links = ", ".join(_page_link(tree, page.id, child_id) for child_id in child_ids)
    label = "Domains" if page.kind == "root" else "Features"
    return f'<p class="page-children"><strong>{label}:</strong> {links}</p>'


def _page_header_html(tree: SiteTree, page: Page) -> str:
    if page.kind == "root":
        covers = ", ".join(f"<code>{_escape(path)}</code>" for path in page.paths)
        covers_html = f'<p class="page-meta">Covers: {covers}</p>'
    else:
        paths = "".join(f"<li><code>{_escape(path)}</code></li>" for path in page.paths)
        covers_html = (
            '<details class="page-covers">'
            f"<summary>Covers {len(page.paths)} files</summary><ul>{paths}</ul>"
            "</details>"
        )
    return (
        '<header class="page-header">'
        f"<h1>{_escape(page.title)}</h1>"
        f'<p class="page-meta">Page ID: <code>{_escape(page.id)}</code>'
        f" · verified at <code>{_escape(page.commit[:10])}</code></p>"
        f"{covers_html}"
        f"{_children_html(tree, page)}"
        "</header>"
    )


PANEL_CHEVRON = (
    '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12.5 5l-5 5 5 5"/></svg>'
)
THEME_CHOICES = (
    (
        "light",
        "Light",
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"'
        ' stroke-linecap="round" aria-hidden="true"><circle cx="10" cy="10" r="3.5"/>'
        '<path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9'
        'l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4"/></svg>',
    ),
    (
        "dark",
        "Dark",
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"'
        ' stroke-linejoin="round" aria-hidden="true">'
        '<path d="M16 12.2A6.5 6.5 0 0 1 7.8 4a6.5 6.5 0 1 0 8.2 8.2z"/></svg>',
    ),
    (
        "system",
        "System",
        '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"'
        ' stroke-linecap="round" aria-hidden="true">'
        '<rect x="2.5" y="3.5" width="15" height="10.5" rx="2"/>'
        '<path d="M7 17h6M10 14v3"/></svg>',
    ),
)
THEME_SCRIPT = (
    "(function(){var k='rox-dox-theme',r=document.documentElement;"
    "var t=localStorage.getItem(k);"
    "if(t==='light'||t==='dark'){r.dataset.theme=t;}"
    "document.addEventListener('click',function(e){"
    "var b=e.target.closest('[data-theme-choice]');if(!b){return;}"
    "var v=b.getAttribute('data-theme-choice');"
    "if(v==='system'){delete r.dataset.theme;localStorage.removeItem(k);}"
    "else{r.dataset.theme=v;localStorage.setItem(k,v);}});})();"
)


def _theme_switch_html() -> str:
    buttons = "".join(
        f'<button type="button" data-theme-choice="{choice}" title="{label} theme"'
        f' aria-label="{label} theme">{icon}</button>'
        for choice, label, icon in THEME_CHOICES
    )
    return f'<div class="theme-switch" role="group" aria-label="Theme">{buttons}</div>'


def _panel_handle_html() -> str:
    return (
        '<label class="panel-handle" for="toggle-left" title="Toggle side panel">'
        f'<span class="panel-handle-grip">{PANEL_CHEVRON}</span></label>'
    )


def _panel_controls_html() -> str:
    return f'<div class="panel-controls">{_theme_switch_html()}</div>'


def _document_shell(
    title: str,
    main_content: str,
    *,
    tree: SiteTree,
    current_path: str,
    current_page_id: str | None,
    entries: Mapping[str, Sequence[RepoEntry]],
    repo_url: str,
    toc_sections: Sequence[tuple[str, str]] = SECTION_LINKS,
    folders_only: bool = False,
    folder_path: str | None = None,
    child_folders: Sequence[tuple[str, str]] = (),
) -> str:
    sidebar = (
        _table_of_contents(toc_sections)
        + _feature_tree_html(
            tree,
            current_path=current_path,
            current_page_id=current_page_id,
        )
        + _file_explorer_html(
            tree,
            entries=entries,
            current_path=current_path,
            repo_url=repo_url,
            folders_only=folders_only,
            folder_path=folder_path,
            child_folders=child_folders,
        )
    )
    document = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title>"
        f"<style>{PAGE_CSS}</style><script>{THEME_SCRIPT}</script></head>"
        f'<body><input class="panel-toggle" id="toggle-left" type="checkbox">'
        f'<div class="page-shell"><div id="sidebar">{sidebar}</div>{_panel_handle_html()}'
        f'<main class="page-main">{_panel_controls_html()}{main_content}</main>'
        "</div></body></html>"
    )
    return _offline_html(document)


def render_page(
    page: Page,
    *,
    tree: SiteTree,
    repo_url: str,
    tables: Mapping[str, Table],
    jar: Path,
    entries: Mapping[str, Sequence[RepoEntry]] = NO_ENTRIES,
) -> str:
    emit_diagram_warnings(page)
    page_header = _page_header_html(tree, page)
    toc_sections = list(SECTION_LINKS[:2])
    if page.membership:
        toc_sections.append(("membership", "Why these files are one feature"))
    toc_sections.extend(SECTION_LINKS[2:])
    sections = "".join(
        [
            _related_section(page, repo_url=repo_url),
            _section("tldr", "TLDR", _tldr_html(page, repo_url=repo_url)),
            _membership_section(page, repo_url=repo_url),
            _block_section(page, repo_url=repo_url),
            _schema_section(page, repo_url=repo_url, tables=tables),
            _sequence_section(page, repo_url=repo_url, jar=jar),
        ]
    )
    main_content = _breadcrumbs_html(tree, page) + page_header + sections
    return _document_shell(
        page.title,
        main_content,
        tree=tree,
        current_path=f"{page.id}.html",
        current_page_id=page.id,
        entries=entries,
        repo_url=repo_url,
        toc_sections=toc_sections,
    )


def render_folder_page(
    *,
    folder_path: str,
    tree: SiteTree,
    entries: Mapping[str, Sequence[RepoEntry]],
    repo_url: str,
    feature_rows: Sequence[tuple[str, str, str, str, int, int, int]],
    uncovered_count: int,
    child_folders: Sequence[tuple[str, str]],
    tests_only: bool,
) -> str:
    current_path = _folder_target(folder_path)
    folder_label = folder_path
    rows = "".join(
        "<tr>"
        f"<td>{_anchor(_relative_href(current_path, f'{feature_id}.html'), feature_title)}</td>"
        f"<td>{_anchor(_relative_href(current_path, f'{domain_id}.html'), domain_title)}</td>"
        f"<td>{files}</td><td>{primary}</td><td>{shared}</td>"
        "</tr>"
        for (
            feature_title,
            feature_id,
            domain_title,
            domain_id,
            files,
            primary,
            shared,
        ) in feature_rows
    )
    if not rows:
        rows = '<tr><td colspan="5" class="empty">No feature pages cover files here.</td></tr>'
    feature_table = (
        '<div class="table-scroll"><table><thead><tr>'
        "<th>Feature</th><th>Domain</th><th>Files</th><th>Primary</th><th>Shared</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )
    child_links = "".join(
        f"<li>{_anchor(_relative_href(current_path, _folder_target(path)), name)}</li>"
        for name, path in child_folders
    )
    child_section = (
        f'<section id="subfolders"><h2>Subfolders</h2><ul>{child_links}</ul></section>'
        if child_links
        else ""
    )
    test_note = (
        '<p class="empty">Tests and migrations are not documented.</p>'
        if tests_only
        else ""
    )
    main_content = (
        '<header class="page-header"><h1>Repository folder</h1>'
        f"<p><code>{_escape(folder_label)}</code></p></header>"
        f'<section><h2>Feature coverage</h2><div id="feature-coverage">{feature_table}</div></section>'
        f'<section id="uncovered-files"><h2>Uncovered files</h2><p>{uncovered_count} in-scope files '
        f"{_anchor(_relative_href(current_path, 'uncovered.html'), 'listed in the uncovered inventory')}.</p>"
        f"{test_note}</section>{child_section}"
    )
    toc_sections = [
        ("feature-coverage", "Feature coverage"),
        ("uncovered-files", "Uncovered files"),
    ]
    if child_links:
        toc_sections.append(("subfolders", "Subfolders"))
    return _document_shell(
        f"Folder: {folder_label}",
        main_content,
        tree=tree,
        current_path=current_path,
        current_page_id=None,
        entries=entries,
        repo_url=repo_url,
        toc_sections=toc_sections,
        folders_only=True,
        folder_path=folder_path,
        child_folders=child_folders,
    )


def render_uncovered_page(
    *,
    tree: SiteTree,
    entries: Mapping[str, Sequence[RepoEntry]],
    repo_url: str,
    uncovered: Mapping[str, Sequence[tuple[str, str]]],
    map_reports: Sequence[tuple[str, Sequence[tuple[str, str]]]],
) -> str:
    current_path = "uncovered.html"
    grouped: dict[str, list[str]] = {}
    for path in sorted(uncovered):
        top = path.split("/", 1)[0]
        grouped.setdefault(top, []).append(path)
    groups = []
    for top, paths in sorted(grouped.items()):
        items = "".join(
            "<li><code>"
            + _escape(path)
            + "</code>"
            + (
                " — "
                + _escape(
                    "; ".join(
                        f"{domain}: {reason}" for domain, reason in uncovered[path]
                    )
                )
                if uncovered[path]
                else ""
            )
            + "</li>"
            for path in paths
        )
        groups.append(
            f"<details><summary>{_escape(top)} — {len(paths)} files</summary>"
            f"<ul>{items}</ul></details>"
        )
    map_details = []
    for domain, files in map_reports:
        items = "".join(
            f"<li><code>{_escape(path)}</code> — {_escape(reason)}</li>"
            for path, reason in files
        )
        if not items:
            items = '<li class="empty">No map-specific uncovered entries.</li>'
        map_details.append(
            f"<details><summary>{_escape(domain)} map — {len(files)} uncovered entries</summary>"
            f"<ul>{items}</ul></details>"
        )
    global_items = "".join(groups) or '<p class="empty">No uncovered files.</p>'
    main_content = (
        '<header class="page-header"><h1>Uncovered files</h1>'
        f"<p>{len(uncovered)} unique in-scope files are not placed on any feature page.</p></header>"
        f'<section id="global-inventory"><h2>Global inventory</h2>{global_items}</section>'
        f'<section id="per-map-reasons"><h2>Per-map reasons</h2>{"".join(map_details)}</section>'
    )
    return _document_shell(
        "Uncovered files",
        main_content,
        tree=tree,
        current_path=current_path,
        current_page_id=None,
        entries=entries,
        repo_url=repo_url,
        toc_sections=[
            ("global-inventory", "Global inventory"),
            ("per-map-reasons", "Per-map reasons"),
        ],
        folders_only=True,
    )

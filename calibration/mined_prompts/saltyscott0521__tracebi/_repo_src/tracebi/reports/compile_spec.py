"""
The spec → artifact compiler — ``tracebi migrate spec``.

A JSON ``ReportSpec`` survives as a *serialization*, not a lane
(architecture v2 §7): the one report form is the artifact package. This
module compiles a spec into that form — each section becomes a
default-component **figure** bound to the same ``DataRef``, so the section
enum ends its life as compile vocabulary rather than renderer control flow.

What the compile promises:

- **Every SectionType compiles.** text, table, chart, metrics, row, spacer —
  a test pins the coverage to the enum, so adding a section type without
  teaching the compiler is loud.
- **Markdown TextSections are honored** (field-notes #4): prose compiles
  through a small, deterministic markdown subset (headings, emphasis, code,
  lists, links, paragraphs) — escaped first, converted second, so content
  can never smuggle markup.
- **Numbers stay claims.** A metric whose value names a query column
  compiles to a live ``value`` figure (``data-tb-cell``); a literal value
  compiles to an honestly ``data-tb-unverified`` card. Nothing is inlined
  as dead text that reads as governed.
- **Nothing silently drops.** Presentation knobs the runtime has no
  equivalent for (heat maps, column widths, metric deltas…)
  are reported as warnings, not swallowed.

The compiler is structural — it needs no models and executes no query. The
artifact it emits then goes through the ordinary ``report build`` gate,
where figure↔binding validation catches anything a wrong spec implied.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from tracebi.reports.figure_markup import (
    TABLE_FORMATS,
    chart_element,
    table_element,
    unverified_value_element,
    value_element,
)
from tracebi.reports.report import SectionType
from tracebi.spec import ReportSpec

# ── a small, deterministic markdown subset ──────────────────────────────────
# Escape first, convert second: content can never introduce live markup.

_MD_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"\*(.+?)\*"), r"<em>\1</em>"),
    (re.compile(r"`(.+?)`"), r"<code>\1</code>"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)"), r'<a href="\2">\1</a>'),
]


def _inline(text: str) -> str:
    # Escape quotes too (html.escape default): the link rule writes the URL
    # into an href="…" attribute, and a URL is captured up to ')' or space —
    # which permits a bare quote. Leaving quotes unescaped let
    # [c](http://x"onmouseover="alert(1)) close the href and inject a live
    # event handler. Escaped, that same quote becomes an inert &quot; inside
    # the attribute value, so content can never break out of the markup.
    out = html.escape(text)
    for pattern, repl in _MD_INLINE:
        out = pattern.sub(repl, out)
    return out


def md_to_html(text: str) -> str:
    """Convert the supported markdown subset to HTML, escaping everything.

    Blocks: ``#``/``##``/``###`` headings, ``-``/``*`` unordered lists,
    ``1.`` ordered lists, blank-line paragraphs. Inline: bold, italic,
    code, links. Unrecognized markdown passes through as escaped prose —
    degraded gracefully, never dropped.
    """
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        m = re.match(r"(#{1,3})\s+(.*)", stripped)
        if m:
            level = len(m.group(1)) + 1          # '#' → h2: h1 is the page's
            blocks.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if re.match(r"[-*]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"[-*]\s+", lines[i].strip()):
                items.append(f"<li>{_inline(lines[i].strip()[2:])}</li>")
                i += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append(f"<li>{_inline(item)}</li>")
                i += 1
            blocks.append("<ol>" + "".join(items) + "</ol>")
            continue
        para = []
        while i < len(lines) and lines[i].strip():
            para.append(lines[i].strip())
            i += 1
        blocks.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(blocks)


# ── the compile ─────────────────────────────────────────────────────────────


@dataclass
class CompiledPackage:
    """What ``compile_spec`` returns: the package files plus honesty notes."""
    name: str
    files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "section"


class _Compiler:
    def __init__(self, spec: ReportSpec) -> None:
        self.spec = spec
        self.bindings: dict[str, dict] = {}
        self.warnings: list[str] = []
        self._fig_ids: set[str] = set()

    # ── helpers ─────────────────────────────────────────────────────────

    def _binding_for(self, raw: dict, where: str) -> str:
        """Register the section's DataRef under a stable binding name."""
        base = _slug(raw.get("id") or raw.get("title") or where)
        name, n = base, 2
        while name in self.bindings and self.bindings[name] != raw["data"]:
            name, n = f"{base}_{n}", n + 1
        self.bindings[name] = raw["data"]
        return name

    def _fig_id(self, base: str) -> str:
        fid, n = f"fig-{base}", 2
        while fid in self._fig_ids:
            fid, n = f"fig-{base}-{n}", n + 1
        self._fig_ids.add(fid)
        return fid

    def _warn_dropped(self, raw: dict, where: str, knobs: tuple[str, ...]) -> None:
        dropped = [k for k in knobs if raw.get(k) not in (None, [], {}, False)]
        if dropped:
            self.warnings.append(
                f"{where}: no runtime equivalent for {', '.join(sorted(dropped))} "
                f"— restyle in style.css / script.js (config can restyle, "
                f"never re-source)"
            )

    @staticmethod
    def _title(raw: dict) -> str:
        t = raw.get("title")
        return f"    <h2>{html.escape(t)}</h2>\n" if t else ""

    @staticmethod
    def _card(raw: dict, element: str) -> str:
        """One card holding the section's title and its figure — a single
        element, so a row lays sections out side by side instead of giving
        each title a cell of its own."""
        t = raw.get("title")
        head = f"    <h3>{html.escape(t)}</h3>\n" if t else ""
        return (f"  <div class=\"tb-card\">\n{head}"
                f"    {element}\n"
                f"  </div>\n")

    # ── per-section compilers (one per SectionType; the test pins this) ──

    def compile_section(self, raw: dict, where: str) -> str:
        stype = raw.get("type")
        fn = {
            SectionType.TEXT.value:    self._text,
            SectionType.TABLE.value:   self._table,
            SectionType.CHART.value:   self._chart,
            SectionType.METRICS.value: self._metrics,
            SectionType.ROW.value:     self._row,
            SectionType.SPACER.value:  self._spacer,
        }.get(stype)
        if fn is None:
            raise ValueError(
                f"{where}: cannot compile unknown section type '{stype}'."
            )
        return fn(raw, where)

    def _text(self, raw: dict, where: str) -> str:
        style = raw.get("style", "normal")
        content = raw.get("content", "")
        title = self._title(raw)
        if style in ("heading1", "heading2"):
            tag = "h2" if style == "heading1" else "h3"
            head = f"  <{tag}>{_inline(raw.get('title') or content)}</{tag}>\n"
            # A heading with both a title and content keeps the content as
            # its lead paragraph instead of dropping it.
            if raw.get("title") and content:
                head += f"  <p>{_inline(content)}</p>\n"
            return head
        body = md_to_html(content)
        if style == "note":
            return f"{title}  <div class=\"tb-note\">{body}</div>\n"
        if style == "callout":
            return f"{title}  <div class=\"tb-callout\">{body}</div>\n"
        return f"{title}  <div>{body}</div>\n"

    def _table(self, raw: dict, where: str) -> str:
        if "data" not in raw:
            self.warnings.append(
                f"{where}: table has no data reference — compiled as an "
                f"empty placeholder"
            )
            return f"{self._title(raw)}  <table></table>\n"
        binding = self._binding_for(raw, where)
        labels = {str(k): str(v) for k, v in (raw.get("column_labels") or {}).items()
                  if not any(ch in str(k) + str(v) for ch in ";=")}
        formats, custom = {}, []
        for col, fmt in (raw.get("number_formats") or {}).items():
            (formats.__setitem__(str(col), fmt) if fmt in TABLE_FORMATS
             else custom.append(str(col)))
        if custom:
            self.warnings.append(
                f"{where}: number_formats for {', '.join(sorted(custom))} are "
                f"not named formats ({', '.join(TABLE_FORMATS)}) — dropped; "
                f"those columns keep the derived format"
            )
        element = table_element(
            binding,
            fig_id=self._fig_id(binding),
            columns=raw.get("columns"),
            style=raw.get("style"),
            labels=labels,
            formats=formats,
            totals=self._totals_binding(raw, binding, where),
        )
        self._warn_dropped(raw, where, (
            "max_rows",
            "highlight_negatives", "color_scale", "column_widths",
        ))
        return self._card(raw, element)

    def _totals_binding(self, raw: dict, binding: str, where: str) -> Optional[str]:
        """A spec's ``totals`` become their own one-row query: the table's
        query with the listed measures and no dimensions, so the model computes
        each total (a ratio's total is a ratio of totals, never a sum of rows).

        Refused, with a warning, when the rows shown are not the whole result
        (``limit`` or ``having``): a grand total under a cut table would not
        add up to what the reader sees.
        """
        wanted = [str(t) for t in (raw.get("totals") or [])]
        if not wanted:
            return None
        data = raw["data"]
        query = dict(data.get("query") or {})
        cut = [k for k in ("limit", "having") if query.get(k)]
        if cut:
            self.warnings.append(
                f"{where}: totals dropped — the query uses {' and '.join(cut)}, "
                f"so the table shows only some rows and a grand total would not "
                f"match them")
            return None
        measures = query.get("measures")
        if isinstance(measures, dict):
            kept = {k: v for k, v in measures.items() if str(k) in wanted}
        else:
            kept = [m for m in (measures or []) if str(m) in wanted]
        missing = sorted(set(wanted) - {str(k) for k in kept})
        if missing:
            self.warnings.append(
                f"{where}: totals for {', '.join(missing)} dropped — only a "
                f"measure the query computes can be totalled")
        if not kept:
            return None
        totals_query = {k: v for k, v in query.items()
                        if k not in ("dimensions", "order_by")}
        totals_query["measures"] = kept
        name, n = f"{binding}_totals", 2
        while name in self.bindings:
            name, n = f"{binding}_totals_{n}", n + 1
        self.bindings[name] = {**data, "query": totals_query}
        return name

    def _chart(self, raw: dict, where: str) -> str:
        if "data" not in raw:
            self.warnings.append(
                f"{where}: chart has no data reference — compiled as an "
                f"empty placeholder"
            )
            return f"{self._title(raw)}  <div></div>\n"
        binding = self._binding_for(raw, where)
        element = chart_element(
            binding,
            fig_id=self._fig_id(binding),
            chart_type=raw.get("chart_type", "bar"),
            x=raw.get("x"),
            y=raw.get("y"),
            color=raw.get("color"),
            palette=raw.get("palette"),
        )
        self._warn_dropped(raw, where, (
            "xlabel", "ylabel", "figsize", "show_values",
        ))
        return self._card(raw, element)

    def _metrics(self, raw: dict, where: str) -> str:
        binding = self._binding_for(raw, where) if raw.get("data") else None
        cards: list[str] = []
        for j, m in enumerate(raw.get("metrics", [])):
            value = m.get("value")
            fmt = m.get("format")
            if m.get("delta") is not None:
                self.warnings.append(
                    f"{where}.metrics[{j}]: delta has no runtime equivalent "
                    f"— dropped; recompute it as a bound cell if it matters"
                )
            live = binding is not None and isinstance(value, str)
            if live:
                fid = self._fig_id(f"{binding}-{_slug(value)}")
                cards.append("    " + value_element(
                    binding, fig_id=fid, cell=value,
                    label=m.get("label", ""), fmt=fmt, indent="    ",
                ) + "\n")
            else:
                # A literal card value is exactly what the unverified mark is
                # for: shown, and honestly not a claim against any binding.
                fid = self._fig_id(f"{_slug(m.get('label') or 'metric')}")
                cards.append("    " + unverified_value_element(
                    fig_id=fid, label=m.get("label", ""), value=value,
                    note="literal spec value", indent="    ",
                ) + "\n")
        return (f"{self._title(raw)}"
                f"  <div class=\"tb-grid\">\n{''.join(cards)}  </div>\n")

    def _row(self, raw: dict, where: str) -> str:
        widths = raw.get("widths") or []
        if len(set(widths)) > 1:
            self.warnings.append(
                f"{where}: unequal widths have no runtime equivalent — "
                f"children compile to equal-width columns; adjust in style.css"
            )
        sections = raw.get("sections", [])
        children = "".join(
            self.compile_section(s, f"{where}.sections[{i}]")
            for i, s in enumerate(sections)
        )
        layout = {2: "tb-cols-2", 3: "tb-cols-3"}.get(len(sections), "tb-grid")
        return f"{self._title(raw)}  <div class=\"{layout}\">\n{children}  </div>\n"

    def _spacer(self, raw: dict, where: str) -> str:
        height = raw.get("height", 1)
        try:
            rem = max(1, int(height))
        except (TypeError, ValueError):
            rem = 1
        return f'  <div style="height:{rem}rem"></div>\n'


def _walk_specs(sections) -> list:
    """Every section dict, rows descended — so nested charts are seen."""
    out: list = []
    for s in sections or []:
        if not isinstance(s, dict):
            continue
        out.append(s)
        out.extend(_walk_specs(s.get("sections") or []))
    return out


def compile_spec(
    spec: ReportSpec,
    theme_css: str = "",
    script_js: str = "",
) -> CompiledPackage:
    """Compile *spec* into artifact package files.

    Returns a :class:`CompiledPackage`: ``report.json`` (the same DataRefs,
    now bindings), ``template.html`` (default-component figures), plus
    ``style.css`` / ``script.js`` when the spec named a theme or script
    (pass their contents in — the compiler resolves nothing from disk).
    """
    c = _Compiler(spec)
    body = "".join(
        c.compile_section(dict(raw), f"sections[{i}]")
        for i, raw in enumerate(spec.sections)
    )

    title = html.escape(spec.name)
    page = (
        "<!doctype html>\n"
        '<html lang="en">\n'
        f"<head><meta charset=\"utf-8\"><title>{title}</title></head>\n"
        "<body>\n"
        '<main class="tb-page">\n'
        f"  <h1>{title}</h1>\n"
        f"{body}"
        "</main>\n"
        "</body></html>\n"
    )

    report_json: dict = {"name": spec.name, "data": c.bindings}
    if spec.author:
        report_json["author"] = spec.author
    if spec.description:
        report_json["description"] = spec.description
    # No "libs": the build inlines D3 for any page with a chart figure.

    files = {
        "report.json": json.dumps(report_json, indent=2) + "\n",
        "template.html": page,
    }
    if theme_css:
        files["style.css"] = theme_css
    if script_js:
        files["script.js"] = script_js
    return CompiledPackage(name=spec.name, files=files, warnings=c.warnings)

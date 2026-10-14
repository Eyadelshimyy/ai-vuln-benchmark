"""
Export a workbench session as an exploration record (architecture v2 §2.5).

The workbench feed — markdown cells, frame excerpts, chart sketches — is
the PRECURSOR record of the pipeline: why the transform drops what it
drops, how the grain was chosen. It dies as dev-state unless saved.
:func:`export_session` renders the FULL feed (``exhibits.jsonl`` read
directly and uncapped — the workbench's ``EXHIBIT_CAP`` is a display cap,
never an archive cap) in chronological order into ONE self-contained HTML
a project can commit as a lab notebook.

Honesty first, in the same spirit as ``TemplatePackage.snapshot``: the
page carries the ``exploration`` stage meta and a persistent banner, and
**no manifest is written, ever** — a lab notebook must never impersonate a
report, so ``tracebi verify`` refuses the file by name. Numbers here were
live when shown and carry no receipts.

Everything user-authored is escaped server-side (``html.escape`` on every
cell, note, and caption); chart data re-embeds through the safe
``embed_json`` block, so a hostile cell value cannot execute in the
exported file. No optional dependencies beyond what the workbench already
uses — the feed is read as JSON, never rebuilt into a DataFrame.
"""

from __future__ import annotations

import html as _html
import json
import os
from typing import Optional

from tracebi.workbench import EXHIBITS_FILE, read_pins, render_note_markdown


def _esc(value) -> str:
    return _html.escape("" if value is None else str(value))


def _pin_seq(pin: dict) -> int:
    """The exhibit a pin belongs to: an ``exhibit-<n>`` id names it exactly;
    ``at_seq`` (the feed position when the pin was placed) is the fallback
    for pins on non-exhibit targets like warehouse tables."""
    pid = str(pin.get("id") or "")
    if pid.startswith("exhibit-"):
        try:
            return int(pid[len("exhibit-"):])
        except ValueError:
            pass
    try:
        return int(pin.get("at_seq", 0))
    except (TypeError, ValueError):
        return 0


def _read_full_feed(wb_dir: str) -> list[dict]:
    """The WHOLE feed, chronological seq order — no cap, torn lines skipped."""
    path = os.path.join(wb_dir, EXHIBITS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no session feed at {path} — nothing to export."
        )
    entries: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
    entries.sort(key=lambda e: e.get("seq", 0))
    return entries


#: The stage banner — locked language, matching what `verify` will say.
_BANNER = (
    '<div style="position:sticky;top:0;z-index:9999;background:#b45309;'
    'color:#fff;font:600 13px/1.4 system-ui,sans-serif;padding:8px 16px;'
    'text-align:center">Exploration record — a lab notebook, not a report. '
    'Numbers here were live when shown and carry no receipts; '
    'tracebi verify refuses this page by name.</div>'
)

_SESSION_CSS = """\
body { font: 15px/1.5 system-ui, sans-serif; background: #fff; color: #1a1a1a;
       margin: 0 0 4rem; }
.tb-page { padding: 0 16px; }
.tb-session-exhibit { margin: 16px auto; }
.tb-session-meta { font: 12px/1.4 ui-monospace, monospace; color: #666;
                   margin-bottom: 6px; }
.tb-session-pin { background: #fef3c7; border-left: 3px solid #b45309;
                  padding: 4px 8px; margin: 6px 0; font-weight: 600; }
.tb-session-auto { color: #666; font-style: italic; }
.tb-session-chart { width: 100%; height: 320px; }
"""

# Rebuilds each chart exhibit from its safe JSON data block using the
# recipe — the same vocabulary the workbench renders live (bar, barh, line,
# area, pie, scatter). Data reaches the page via JSON.parse only, never
# innerHTML (architecture §5).
_CHART_JS = """\
(function () {
  if (!window.echarts) return;
  var blocks = document.querySelectorAll('script[type="application/json"]');
  Array.prototype.forEach.call(blocks, function (el) {
    if (el.id.indexOf("tb-chart-data-") !== 0) return;
    var host = document.getElementById(
      "tb-chart-" + el.id.slice("tb-chart-data-".length));
    if (!host) return;
    var payload;
    try { payload = JSON.parse(el.textContent); } catch (e) { return; }
    var r = payload.recipe || {}, rows = payload.rows || [];
    var cats = rows.map(function (row) { return String(row[r.x]); });
    var series = (r.y || []).map(function (col) {
      if (r.chart === "pie") {
        return { type: "pie", name: col, data: rows.map(function (row) {
          return { name: String(row[r.x]), value: Number(row[col]) };
        }) };
      }
      return {
        type: r.chart === "barh" ? "bar"
            : (r.chart === "area" ? "line" : r.chart),
        name: col,
        areaStyle: r.chart === "area" ? {} : null,
        data: rows.map(function (row) { return Number(row[col]); }),
      };
    });
    var cat = { type: "category", data: cats }, val = { type: "value" };
    var option = { animation: false, tooltip: {}, series: series };
    if ((r.y || []).length > 1) option.legend = {};
    if (r.chart !== "pie") {
      option.xAxis = r.chart === "barh" ? val : cat;
      option.yAxis = r.chart === "barh" ? cat : val;
    }
    var chart = echarts.init(host);
    chart.setOption(option);
    window.addEventListener("resize", function () { chart.resize(); });
  });
})();
"""


def _frame_html(ex: dict) -> str:
    """A frame exhibit as a static tb-table — every cell escaped, capped at
    the excerpt rows already stored."""
    cols = [str(c) for c in ex.get("columns") or []]
    rows = [r for r in ex.get("rows") or [] if isinstance(r, dict)]
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(row.get(c, ''))}</td>" for c in cols)
        + "</tr>"
        for row in rows
    )
    out = ('<table class="tb-table tb-table--compact">'
           f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
    shape = ex.get("shape")
    if isinstance(shape, list) and len(shape) == 2 and shape[0] > len(rows):
        out += (f'<p class="tb-note">excerpt — first {len(rows)} of '
                f"{shape[0]} rows</p>")
    return out


def _chart_html(ex: dict, idx: int) -> str:
    """A chart exhibit: a host div plus its safe JSON data block; the recipe
    is also printed so the record stays honest without scripting."""
    from tracebi.reports.embed import embed_json

    recipe = ex.get("recipe") or {}
    block = embed_json({"recipe": recipe, "rows": ex.get("rows") or []},
                       f"tb-chart-data-{idx}")
    caption = (f"chart={recipe.get('chart')} · x={recipe.get('x')} · "
               f"y={','.join(recipe.get('y') or [])}")
    return (f'<div class="tb-session-chart" id="tb-chart-{idx}"></div>\n'
            f"{block}\n"
            f'<p class="tb-note">{_esc(caption)}</p>')


def export_session(wb_dir: str, out_path: str,
                   title: Optional[str] = None) -> None:
    """Render *wb_dir*'s FULL exhibit feed to *out_path* — one
    self-contained exploration record.

    The whole feed in chronological seq order (``EXHIBIT_CAP`` caps the
    workbench display, never this export), pins highlighted (📌 + note) on
    their exhibits, and a per-exhibit meta line: seq · timestamp · kind ·
    source script when recorded. Writes the record alone: NO manifest,
    ever. An ``out_path`` ending ``.md`` writes the **Markdown twin** —
    the git-review format: notes appear as their raw markdown verbatim,
    frames and chart sketches as pipe tables with the recipe described —
    made for reading in a pull request. Any other extension writes the
    self-contained HTML, whose ``exploration`` stage meta makes ``tracebi
    verify`` refuse it by name. *title* defaults to ``"<session> —
    exploration record"``.
    """
    if str(out_path).lower().endswith(".md"):
        _export_markdown(wb_dir, out_path, title)
        return
    from tracebi.reports.stack import read_asset

    session = os.path.basename(os.path.normpath(wb_dir)) or "session"
    title = title or f"{session} — exploration record"

    exhibits = render_note_markdown(_read_full_feed(wb_dir))
    pins_by_seq: dict[int, list[dict]] = {}
    for pin in read_pins(wb_dir):
        try:
            pins_by_seq.setdefault(_pin_seq(pin), []).append(pin)
        except (TypeError, ValueError):
            continue

    parts: list[str] = []
    n_charts = 0
    for ex in exhibits:
        kind = ex.get("kind") or "note"
        seq = ex.get("seq", 0)
        meta = [f'<span class="seq">#{_esc(seq)}</span>']
        for key in ("at", "kind", "source"):
            if ex.get(key):
                meta.append(_esc(ex[key]))
        pinned = pins_by_seq.get(seq, []) if isinstance(seq, int) else []
        if kind == "chart":
            n_charts += 1
            body = _chart_html(ex, n_charts)
        elif kind == "frame":
            body = _frame_html(ex)
        elif kind == "note":
            body = ex.get("html") or f"<p>{_esc(ex.get('text'))}</p>"
        elif kind == "auto":
            body = f'<p class="tb-session-auto">{_esc(ex.get("text"))}</p>'
        else:   # binding, or a kind this exporter predates — keep the record
            body = f"<p>{_esc(ex.get('name') or ex.get('text') or kind)}</p>"
        pin_html = "".join(
            f'<div class="tb-session-pin">📌 '
            f'{_esc(p.get("note") or "pinned")}</div>'
            for p in pinned
        )
        note = ex.get("note")
        note_html = f'<p class="tb-note">{_esc(note)}</p>' if note else ""
        cls = ("tb-card tb-session-exhibit"
               + (" tb-session-pinned" if pinned else ""))
        parts.append(
            f'<article class="{cls}">'
            f'<div class="tb-session-meta">{" · ".join(meta)}</div>'
            f"{pin_html}{note_html}{body}</article>"
        )

    from tracebi.reports.embed import csp_meta

    # Defense-in-depth: this static file interpolates md_to_html output. The
    # escaping in _inline is the real protection; the strict CSP (connect-src
    # 'none' — an offline file never polls) blocks external script/exfil should
    # that escaping ever regress.
    head = (
        '<meta charset="utf-8">\n'
        + csp_meta()
        + f"<title>{_esc(title)}</title>\n"
        + '<meta name="tracebi-stage" content="exploration">\n'
        + f"<style>\n{read_asset('tracebi.css')}\n</style>\n"
        + f"<style>\n{_SESSION_CSS}\n</style>\n"
    )
    tail = ""
    if n_charts:
        from tracebi.reports.embed import read_lib
        tail = (f"<script>\n{read_lib('echarts')}\n</script>\n"
                f"<script>\n{_CHART_JS}\n</script>\n")

    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n' + head
        + "</head>\n<body>\n" + _BANNER + "\n"
        + f'<main class="tb-page">\n<h1>{_esc(title)}</h1>\n'
        + "\n".join(parts)
        + "\n</main>\n" + tail + "</body>\n</html>\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)


# ── The Markdown twin — the git-review format ───────────────────────────────


def _md_cell(value) -> str:
    """One table cell: pipes escaped, newlines flattened — layout-safe."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _md_table(columns: list, rows: list[dict], shape=None) -> list[str]:
    cols = [str(c) for c in (columns or [])]
    if not cols:
        return ["*(no columns)*"]
    out = ["| " + " | ".join(_md_cell(c) for c in cols) + " |",
           "|" + "---|" * len(cols)]
    for r in rows or []:
        out.append("| " + " | ".join(_md_cell(r.get(c)) for c in cols) + " |")
    if shape and shape[0] > len(rows or []):
        out.append(f"\n*excerpt — first {len(rows or [])} of {shape[0]} rows*")
    return out


def _export_markdown(wb_dir: str, out_path: str,
                     title: Optional[str]) -> None:
    """The reviewable twin: raw-markdown notes verbatim (they ARE
    markdown), tables for frames and sketches, pins as blockquotes. Same
    honesty, stated in the file's own first lines; no manifest, ever."""
    session = os.path.basename(os.path.normpath(wb_dir)) or "session"
    title = title or f"{session} — exploration record"

    exhibits = _read_full_feed(wb_dir)
    pins_by_seq: dict[int, list[dict]] = {}
    for pin in read_pins(wb_dir):
        pins_by_seq.setdefault(_pin_seq(pin), []).append(pin)

    lines: list[str] = [
        f"# {title}",
        "",
        "> **Exploration record — a lab notebook, not a report.** Numbers "
        "here were live when shown and carry no receipts; nothing in this "
        "file is a verified claim.",
        "",
    ]
    for ex in exhibits:
        meta = " · ".join(str(p) for p in (
            f"#{ex.get('seq')}", ex.get("at"), ex.get("kind"),
            ex.get("source")) if p)
        lines += ["---", "", f"`{meta}`", ""]
        for pin in pins_by_seq.get(int(ex.get("seq", -1)), []):
            lines += [f"> 📌 **pinned:** {_md_cell(pin.get('note') or '')}", ""]
        kind = ex.get("kind")
        if kind == "note":
            # Verbatim: the note IS markdown; this format's whole point.
            lines += [str(ex.get("text") or ""), ""]
        elif kind in ("frame", "chart"):
            if kind == "chart":
                r = ex.get("recipe") or {}
                lines += [f"**chart sketch** — {r.get('chart')} · "
                          f"x: {r.get('x')} · y: {', '.join(r.get('y') or [])}",
                          ""]
            if ex.get("note"):
                lines += [f"*{_md_cell(ex['note'])}*", ""]
            lines += _md_table(ex.get("columns"), ex.get("rows"),
                               ex.get("shape")) + [""]
        elif kind == "binding":
            lines += [f"*binding: {_md_cell(ex.get('name'))}*", ""]
        else:
            lines += [f"*{_md_cell(ex.get('text') or '')}*", ""]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

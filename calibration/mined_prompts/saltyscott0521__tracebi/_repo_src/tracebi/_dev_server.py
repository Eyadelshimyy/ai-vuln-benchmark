"""
Live-preview server for ``tracebi dev <name>`` — form-aware (v2 §2.5).

An **artifact package** (``reports/<name>/``) gets the artifact-native loop:
the working state is rendered in memory on every change — exploration blocks
KEPT, badges on, stage ``exploration`` — and served alongside THE WORKBENCH
at ``/__workbench``: figures with provenance and copy-addresses, per-binding
data with quick-charts, the exhibit feed (``tracebi.workbench.show``), pins,
code, and lint. The watcher scans the package directory, ``models/``,
``transforms/``, and ``reports/_theme.css``; the served pages poll
``/__status`` and reload when anything changed. Errors render as a styled
traceback page that also auto-reloads once the code is fixed.

**No target at all** (``tracebi dev`` with no name) is DISCOVERY MODE: no
report is anchored, and the workbench IS the site — the live surface for
phase ① (interrogating source data) and phase ② (designing the model). It
renders the warehouse's tables and sink-contract summaries, every model's
declared star schema, and the report packages that exist; the server
heartbeats ``.tracebi/workbench/_discovery/.active`` each watcher tick so
ANY script run in the project can post to the exhibit feed via
``tracebi.workbench.show`` with zero configuration (see the heartbeat rule
in ``tracebi/workbench.py``).

Everything here is dev-state by construction: nothing this server renders or
records exists in builds or manifests, and no receipts are minted.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import traceback
import webbrowser
from pathlib import Path


def _host_is_local(host_header: str) -> bool:
    """Whether a ``Host`` header names a loopback address — the DNS-rebinding
    guard. A browser pointed at a malicious domain that resolves to 127.0.0.1
    sends that domain as ``Host``; only localhost / 127.0.0.1 / [::1] (and an
    absent header) are legitimate for this local dev server.
    """
    if host_header.startswith("["):              # IPv6 literal, e.g. [::1]:port
        bare = host_header.split("]", 1)[0].lstrip("[")
    else:
        bare = host_header.rsplit(":", 1)[0]
    return bare in ("localhost", "127.0.0.1", "::1", "")


_REFRESH_SNIPPET = """
<script>
(function () {
  var current = __VERSION__;
  setInterval(function () {
    fetch("/__status")
      .then(function (r) { return r.json(); })
      .then(function (s) { if (s.version !== current) location.reload(); })
      .catch(function () {});
  }, 1000);
})();
</script>
"""

_ERROR_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>tracebi dev — error</title>
<style>
body {{ font-family: 'Segoe UI', Calibri, Arial, sans-serif; background: #f5f7fa;
       padding: 40px; color: #1a1a2e; }}
.box {{ max-width: 900px; margin: 0 auto; background: #fff; border-radius: 6px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08); overflow: hidden; }}
.head {{ background: #C62828; color: #fff; padding: 18px 24px; }}
.head h1 {{ font-size: 16px; margin: 0; }}
.head p {{ font-size: 12px; margin: 6px 0 0 0; opacity: 0.9; }}
pre {{ margin: 0; padding: 20px 24px; font-size: 12px; line-height: 1.5;
      overflow-x: auto; white-space: pre-wrap; }}
</style></head>
<body><div class="box">
<div class="head"><h1>{title}</h1>
<p>{file} — fix the code and save; this page reloads automatically.</p></div>
<pre>{trace}</pre>
</div></body></html>"""


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _inject_refresh(html: str, version: int) -> str:
    snippet = _REFRESH_SNIPPET.replace("__VERSION__", str(version))
    if "</body>" in html:
        return html.replace("</body>", snippet + "</body>", 1)
    return html + snippet


# ── The two target forms ────────────────────────────────────────────────────


def _project_models() -> dict:
    """Every model in models/, loaded FRESH — the watcher watches models/,
    so an edited model file must re-exec on the next rebuild, which the
    process-global registry's cache would prevent. A file that fails to load
    is skipped with a stderr note; a binding that needs it reports the miss."""
    from tracebi.model_registry import ModelRegistry

    reg = ModelRegistry()
    models: dict = {}
    for stem in reg.auto_discover(os.environ.get("TRACEBI_MODELS_DIR", "models")):
        try:
            m = reg.get(stem)
        except Exception as exc:  # noqa: BLE001 — surfaced, not fatal to the loop
            print(f"  model '{stem}' failed to load: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        models[m.name] = m
        models.setdefault(stem, m)
    return models


class _PackageTarget:
    """An artifact package — the in-memory exploration render + workbench."""

    workbench = True
    discovery = False

    def __init__(self, directory: Path) -> None:
        from tracebi.workbench import workbench_dir

        from tracebi.report_paths import report_name_for_dir

        self.directory = Path(directory)
        self.name = report_name_for_dir(str(self.directory))
        self.label = self.name
        self.wb_dir = workbench_dir(os.getcwd(), self.name)
        # Per-binding fingerprints from the previous build: an auto-entry is
        # appended only when a fingerprint CHANGED, so the feed reads as a
        # log of actual data movement, not of every keystroke.
        self._fingerprints: dict[str, str] = {}
        self._primed = False

    def render(self) -> str:
        from tracebi.reports.template_package import TemplatePackage

        # show() calls in report.py land in this session's feed — the env
        # var is set only around the build, so nothing outside dev sees it.
        previous = os.environ.get("TRACEBI_WORKBENCH_DIR")
        os.environ["TRACEBI_WORKBENCH_DIR"] = self.wb_dir
        try:
            models = _project_models()
            pkg = TemplatePackage(str(self.directory))
            page, inputs, outputs = pkg.render_exploration(models)
        except (Exception, SystemExit):
            return _ERROR_PAGE.format(
                title="Report package failed",
                file=_esc(self.name),
                trace=_esc(traceback.format_exc()),
            )
        finally:
            if previous is None:
                os.environ.pop("TRACEBI_WORKBENCH_DIR", None)
            else:
                os.environ["TRACEBI_WORKBENCH_DIR"] = previous
        self._note_fingerprints(inputs + outputs)
        # Dev-only CSP relaxation: the shipped page's connect-src 'none'
        # would block the /__status reload poll, silently killing the live
        # loop. Served pages may talk to THIS server and nothing else; the
        # built artifact keeps the strict policy untouched.
        return page.replace("connect-src 'none'", "connect-src 'self'", 1)

    def _note_fingerprints(self, stamped) -> None:
        from tracebi.workbench import auto_entry

        for sd in stamped:
            if self._primed and self._fingerprints.get(sd.name) != sd.fingerprint:
                auto_entry(
                    self.wb_dir,
                    f"binding {sd.name} updated · "
                    f"{len(sd.dataset.to_pandas())} rows · "
                    f"{sd.fingerprint[:12]}",
                )
            self._fingerprints[sd.name] = sd.fingerprint
        self._primed = True

    def watch_paths(self) -> list[Path]:
        reports_dir = Path(os.environ.get("TRACEBI_REPORTS_DIR", "reports"))
        return [
            self.directory,
            Path(os.environ.get("TRACEBI_MODELS_DIR", "models")),
            Path(os.environ.get("TRACEBI_TRANSFORMS_DIR", "transforms")),
            reports_dir / "_theme.css",
        ]

    def state(self) -> dict:
        """The workbench state — collect_state, degraded to an error state
        rather than a 500 when even loading the package fails."""
        from tracebi.workbench import collect_state

        try:
            return collect_state(str(self.directory), _project_models())
        except Exception as exc:  # noqa: BLE001 — the panel must show the break
            return {
                "name": self.name,
                "error": f"{type(exc).__name__}: {exc}",
                "figures": [],
                "coverage": {"total": 0, "verified": 0, "derived": 0,
                             "unverified": 0, "unbound_errors": 0},
                "bindings": [], "unused_bindings": [],
                "lint": {"numeric_literals_outside_figures": 0},
                "exhibits": [], "pins": [], "resolved": [],
                "resolved_count": 0, "code": {},
            }


class _DiscoveryTarget:
    """Discovery mode — no report anchored; the workbench IS the site."""

    workbench = True
    discovery = True

    def __init__(self) -> None:
        from tracebi.workbench import DISCOVERY_NAME, discovery_dir

        self.name = DISCOVERY_NAME
        self.label = "discovery"
        self.wb_dir = discovery_dir(os.getcwd())

    def render(self) -> str:
        # In discovery there is no report to preview — the root just points
        # at the workbench.
        return ('<!DOCTYPE html>\n<html><head><meta charset="UTF-8">'
                '<meta http-equiv="refresh" content="0; url=/__workbench">'
                '<title>tracebi dev — discovery</title></head>'
                '<body><p><a href="/__workbench">workbench</a></p>'
                '</body></html>')

    def watch_paths(self) -> list[Path]:
        # data/ is watched so a sink landing bumps the version and the
        # workbench page refreshes with the new tables.
        return [
            Path(os.environ.get("TRACEBI_TRANSFORMS_DIR", "transforms")),
            Path(os.environ.get("TRACEBI_MODELS_DIR", "models")),
            Path(os.environ.get("TRACEBI_REPORTS_DIR", "reports")),
            Path("data"),
        ]

    def state(self) -> dict:
        """collect_discovery_state, degraded to an error state rather than
        a 500 when even collecting fails (mirrors _PackageTarget.state)."""
        from tracebi.workbench import collect_discovery_state

        try:
            return collect_discovery_state(os.getcwd(), _project_models())
        except Exception as exc:  # noqa: BLE001 — the panel must show the break
            return {
                "mode": "discovery",
                "name": self.name,
                "error": f"{type(exc).__name__}: {exc}",
                "warehouse": {"path": "", "exists": False, "tables": [],
                              "contracts": None},
                "models": [], "packages": [], "exhibits": [], "pins": [],
                "resolved": [], "resolved_count": 0,
            }


def _scan_signature(paths) -> tuple:
    """(max mtime, file count) across files and directory trees — a change
    in either means something to rebuild. Files mid-atomic-save are skipped
    for this tick and picked up on the next."""
    latest = 0.0
    count = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    try:
                        latest = max(latest,
                                     os.path.getmtime(os.path.join(root, fn)))
                        count += 1
                    except OSError:
                        continue
        elif p.is_file():
            try:
                latest = max(latest, p.stat().st_mtime)
                count += 1
            except OSError:
                continue
    return (latest, count)


def _file_sig(path: str):
    try:
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


# ── The workbench page (dev-only; never injected into build output) ─────────

_WORKBENCH_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
__CSP__
<title>workbench</title>
<style>
__TRACEBI_CSS__
</style>
<style>
/* Workbench page layer — stacked after tracebi.css, so later wins. */
body { max-width: 1100px; }
.wb-badge { position: static; }        /* .tb-badge is corner-absolute */
.wb-badge--error { color: #b91c1c; background: #fdeaea;
                   border-color: #f2b8b8; }
.wb-bar { height: 8px; background: var(--tb-rule); border-radius: 999px;
          overflow: hidden; margin: 4px 0 16px; }
.wb-bar > div { height: 100%; background: #1e6b34; }
.wb-row { display: flex; align-items: center; flex-wrap: wrap;
          gap: var(--tb-space-2); padding: 6px 0;
          border-bottom: 1px solid var(--tb-rule); }
.wb-card { border: 1px solid var(--tb-rule); border-radius: var(--tb-radius);
           padding: var(--tb-space-3); margin: var(--tb-space-3) 0; }
.wb-card .wb-row { border-bottom: 0; }
.wb-meta { color: var(--tb-muted); font-size: 0.85em; }
.wb-error { color: #C62828; white-space: pre-wrap; }
.wb-warn { color: #8a5a00; }
.wb-btn { font: inherit; font-size: 0.8em; padding: 2px 8px; cursor: pointer;
          border: 1px solid var(--tb-rule); border-radius: var(--tb-radius);
          background: #f6f7f9; }
.wb-btn--pinned { background: #fdf3d7; border-color: #ecd393; }
.wb-chart { height: 260px; margin-top: var(--tb-space-2); }
.wb-table-wrap { max-height: 320px; overflow: auto; }
select { font: inherit; font-size: 0.85em; }
/* Side by side: the report preview beside the workbench (package mode). */
body.wb-split-on { max-width: none; margin: 0; padding: 0; }
#wb-split { display: block; }
body.wb-split-on #wb-split { display: grid; height: 100vh;
  grid-template-columns: minmax(0, 1.25fr) minmax(360px, 1fr); }
#wb-preview { display: none; width: 100%; height: 100%; border: 0;
              border-right: 1px solid var(--tb-rule); background: #fff; }
body.wb-split-on #wb-preview { display: block; }
body.wb-split-on #wb-panel { overflow: auto; padding: 16px 20px; }
@media (max-width: 1000px) {
  body.wb-split-on #wb-split { display: block; height: auto; }
  body.wb-split-on #wb-preview { height: 70vh; border-right: 0;
                                 border-bottom: 1px solid var(--tb-rule); }
}
.wb-selected { border-color: #2a78d6; box-shadow: 0 0 0 2px rgba(42,120,214,.15); }
.wb-chip { font-size: 0.75em; padding: 1px 7px; border-radius: 999px;
           border: 1px solid var(--tb-rule); background: #f6f7f9; }
.wb-chip--changed { background: #fdf3d7; border-color: #ecd393; }
.wb-chip--new { background: #e6f1fb; border-color: #b9d6f2; }
.wb-request { border-left: 3px solid #2a78d6; }
.wb-code { font-size: 0.8em; background: #f6f7f9; padding: 8px;
           border-radius: var(--tb-radius); overflow: auto; margin: 4px 0 0; }
/* Tabs: a section shows only on its tab, and only in its mode. */
.wb-off { display: none !important; }
.wb-tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--tb-rule);
           margin: 12px 0 8px; position: sticky; top: 0; z-index: 2;
           background: var(--tb-page, #fff); }
.wb-tabs button { font: inherit; font-size: 0.9em; font-weight: 600;
                  border: 0; background: none; padding: 8px 12px;
                  cursor: pointer; color: var(--tb-muted);
                  border-bottom: 2px solid transparent; margin-bottom: -1px; }
.wb-tabs button.wb-on { color: var(--tb-ink); border-bottom-color: #2a78d6; }
/* The timeline reads like a notebook: In (the code) then Out (what it
   showed), oldest at the top, newest at the bottom — and your messages
   sit in the same stream, like a chat. */
.wb-cell { margin: 14px 0; }
.wb-cell-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.wb-in { font-family: ui-monospace, Menlo, monospace; font-size: 0.75em;
         color: #2a78d6; min-width: 3.2em; }
.wb-out { border-left: 3px solid var(--tb-rule); padding: 2px 0 2px 12px;
          margin-top: 6px; }
.wb-divider { display: flex; align-items: center; gap: 10px; margin: 22px 0 6px;
              font-size: 0.72em; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: var(--tb-muted); }
.wb-divider::after { content: ""; flex: 1; height: 1px; background: var(--tb-rule); }
.wb-marker { text-align: center; color: var(--tb-muted); font-size: 0.78em;
             margin: 8px 0; }
.wb-me { margin: 12px 0 12px auto; max-width: 80%; background: #e6f1fb;
         border-radius: 12px 12px 2px 12px; padding: 8px 12px; }
.wb-me .wb-meta { display: block; font-size: 0.75em; }
.wb-composer { position: sticky; bottom: 0; display: flex; gap: 8px;
               padding: 10px 0; background: var(--tb-page, #fff);
               border-top: 1px solid var(--tb-rule); }
.wb-composer textarea { flex: 1; font: inherit; font-size: 0.9em; resize: vertical;
                        border: 1px solid var(--tb-rule);
                        border-radius: var(--tb-radius); padding: 6px 8px; }
</style>
</head>
<body>
<div id="wb-split">
<iframe id="wb-preview" title="report preview"></iframe>
<div id="wb-panel">
<h1 id="wb-title">workbench</h1>
<p class="wb-meta">Dev-state only — nothing on this page exists in builds or
manifests, and no receipts are minted here.
<a id="wb-preview-link" href="/">← report preview</a>
<button id="wb-layout" class="wb-btn" type="button" hidden>full width</button></p>
<div id="wb-error"></div>
<nav id="wb-tabs" class="wb-tabs">
<button type="button" data-tab="timeline">Timeline</button>
<button type="button" data-tab="figures" data-mode="package">Figures &amp; data</button>
<button type="button" data-tab="project" data-mode="discovery">Project</button>
<button type="button" data-tab="code" data-mode="package">Code</button>
</nav>
<section id="wb-sec-requests" data-tab="timeline" hidden>
<h2>For your agent</h2>
<p class="wb-meta">What you asked for here. Your agent sees these in
<code>tracebi report status</code> and the <code>workbench_state</code> tool,
or copy one into your chat.</p>
<div id="wb-requests"></div>
</section>
<section id="wb-sec-feed" data-tab="timeline">
<div id="wb-feed"></div>
<form id="wb-composer" class="wb-composer">
<textarea id="wb-message" rows="2"
  placeholder="Leave a note for your agent — it reads notes on its next pass, not live. e.g. 'split this by fund'"></textarea>
<button type="submit" class="wb-btn">Leave note</button>
</form>
</section>
<section id="wb-sec-selected" data-tab="figures" data-mode="package" hidden>
<h2>Selected figure</h2>
<div id="wb-selected"></div>
</section>
<section id="wb-sec-figures" data-tab="figures" data-mode="package">
<h2>Figures</h2>
<div id="wb-coverage"></div>
<div id="wb-figures"></div>
</section>
<section id="wb-sec-data" data-tab="figures" data-mode="package">
<h2>Data</h2>
<div id="wb-data"></div>
</section>
<section id="wb-sec-warehouse" data-tab="project" data-mode="discovery">
<h2>Warehouse</h2>
<div id="wb-warehouse"></div>
</section>
<section id="wb-sec-models" data-tab="project" data-mode="discovery">
<h2>Models</h2>
<div id="wb-models"></div>
</section>
<section id="wb-sec-packages" data-tab="project" data-mode="discovery">
<h2>Packages</h2>
<div id="wb-packages"></div>
</section>
<section id="wb-sec-code" data-tab="code" data-mode="package">
<h2>Code</h2>
<div id="wb-code"></div>
</section>
<section id="wb-sec-lint" data-tab="code" data-mode="package">
<h2>Lint</h2>
<div id="wb-lint"></div>
</section>
</div>
</div>
<script>
__ECHARTS__
</script>
<script>
(function () {
  "use strict";
  var REPORT = __NAME__;
  var last = null;

  /* All data reaches the DOM through textContent/createElement — never
     innerHTML of a data value. */
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  function button(label, cls, onclick) {
    var b = el("button", "wb-btn" + (cls ? " " + cls : ""), label);
    b.addEventListener("click", onclick);
    return b;
  }

  function copyBtn(label, text) {
    return button(label, null, function () {
      var b = this;
      var done = function () {
        b.textContent = "copied";
        setTimeout(function () { b.textContent = label; }, 1200);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else { done(); }
    });
  }

  function post(path, body) {
    return fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    }).then(function () { return refresh(); }, function () {});
  }

  function pinBtn(id, pinned) {
    return button(pinned ? "unpin" : "pin",
                  pinned ? "wb-btn--pinned" : null, function () {
      if (pinned) { post("/__workbench/unpin", {id: id}); return; }
      var note = window.prompt("Pin note (optional):", "") || "";
      post("/__workbench/pin", {id: id, note: note});
    });
  }

  function badge(provenance) {
    var labels = {verified: "verified", derived: "python-derived",
                  unverified: "unverified", error: "unbound"};
    var cls = provenance === "error" ? "wb-badge--error"
                                     : "tb-badge--" + provenance;
    return el("span", "tb-badge wb-badge " + cls,
              labels[provenance] || provenance);
  }

  function smallTable(columns, rows, cap, sortable, display) {
    /* display (optional) holds each row as the report would write it; the
       raw values stay for sorting. */
    if (display && display.length) {
      rows = rows.map(function (r, i) {
        return display[i] ? Object.assign({}, r, {__shown: display[i]}) : r;
      });
    }
    var wrap = el("div", "wb-table-wrap");
    var t = el("table", "tb-table tb-table--compact");
    var thead = el("thead");
    var tr = el("tr");
    var tbody = el("tbody");
    var shown = rows.slice(0, cap || rows.length);
    var sortCol = null;
    var sortAsc = true;

    function fill(list) {
      tbody.textContent = "";
      list.forEach(function (r) {
        var tr2 = el("tr");
        columns.forEach(function (c) {
          var v = r[c];
          var shown = r.__shown && r.__shown[c] !== undefined ? r.__shown[c] : v;
          tr2.appendChild(el("td", typeof v === "number" ? "tb-num" : null,
                             shown === null || shown === undefined ? "" : shown));
        });
        tbody.appendChild(tr2);
      });
    }

    function isNum(v) {
      return v !== null && v !== undefined && v !== "" && !isNaN(Number(v));
    }

    columns.forEach(function (c) {
      var th = el("th", null, c);
      if (sortable) {
        /* Sorting is presentation of the excerpt only — it reorders the
           rows already on screen, never the data they came from. */
        th.style.cursor = "pointer";
        th.addEventListener("click", function () {
          sortAsc = sortCol === c ? !sortAsc : true;
          sortCol = c;
          var sorted = shown.slice().sort(function (ra, rb) {
            var a = ra[c], b = rb[c];
            var d;
            if (isNum(a) && isNum(b)) d = Number(a) - Number(b);
            else d = String(a) < String(b) ? -1
                                           : (String(a) > String(b) ? 1 : 0);
            return sortAsc ? d : -d;
          });
          fill(sorted);
        });
      }
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    t.appendChild(thead);
    fill(shown);
    t.appendChild(tbody);
    wrap.appendChild(t);
    return wrap;
  }

  function profileToggle(profile, note) {
    /* The per-column stats card, collapsed behind a button. A null profile
       (capped or failed upstream) shows only its note. */
    if (!profile) {
      return note ? el("p", "wb-meta", note) : null;
    }
    var wrap = el("div");
    var cols = ["column", "dtype", "nulls", "distinct",
                "min", "max", "mean", "top"];
    var rows = Object.keys(profile).map(function (c) {
      var p = profile[c] || {};
      return {column: c, dtype: p.dtype, nulls: p.nulls,
              distinct: p.distinct, min: p.min, max: p.max, mean: p.mean,
              top: (p.top || []).join(", ")};
    });
    var body = smallTable(cols, rows);
    body.style.display = "none";
    var btn = button("profile", null, function () {
      var open = body.style.display !== "none";
      body.style.display = open ? "none" : "block";
      btn.textContent = open ? "profile" : "hide profile";
    });
    wrap.appendChild(btn);
    wrap.appendChild(body);
    return wrap;
  }

  function renderFigures(state) {
    var cov = document.getElementById("wb-coverage");
    cov.textContent = "";
    var c = state.coverage;
    cov.appendChild(el("p", null,
        c.verified + " of " + c.total + " figures model-backed"));
    var bar = el("div", "wb-bar");
    var fill = el("div");
    fill.style.width = (c.total ? Math.round(100 * c.verified / c.total) : 0) + "%";
    bar.appendChild(fill);
    cov.appendChild(bar);

    var host = document.getElementById("wb-figures");
    host.textContent = "";
    state.figures.forEach(function (f) {
      var row = el("div", "wb-row");
      row.appendChild(badge(f.unbound ? "error" : f.provenance));
      row.appendChild(el("code", null, f.id));
      row.appendChild(el("span", "wb-meta",
          f.kind + (f.binding ? " · " + f.binding : "")
                 + (f.note ? " · " + f.note : "")));
      row.appendChild(copyBtn("copy address", REPORT + "#fig:" + f.id));
      row.appendChild(pinBtn(f.id, f.pinned));
      host.appendChild(row);
    });
    if (!state.figures.length) {
      host.appendChild(el("p", "wb-meta", "no figures yet — mark elements "
          + "with data-tb-figure in template.html"));
    }
  }

  /* One ECharts option builder for every workbench chart surface — the
     quick-chart picker and chart exhibits share it, so a sketch's recipe
     has exactly one rendering. kind is the runtime's vocabulary
     (bar/barh/line/area/pie/scatter); ys is a list of y columns. */
  function chartOption(kind, x, ys, rows) {
    if (kind === "pie") {
      return {series: [{type: "pie", radius: "70%",
                        data: rows.map(function (r) {
                          return {name: String(r[x]), value: r[ys[0]]};
                        })}]};
    }
    var grid = {top: 16, right: 16, bottom: 48, left: 72};
    if (kind === "scatter") {
      return {grid: grid,
              xAxis: {type: "value"}, yAxis: {type: "value"},
              series: ys.map(function (y) {
                return {type: "scatter",
                        data: rows.map(function (r) { return [r[x], r[y]]; })};
              })};
    }
    var catAxis = {type: "category",
                   data: rows.map(function (r) { return String(r[x]); })};
    var valAxis = {type: "value"};
    return {grid: grid,
            xAxis: kind === "barh" ? valAxis : catAxis,
            yAxis: kind === "barh" ? catAxis : valAxis,
            series: ys.map(function (y) {
              var s = {type: kind === "line" || kind === "area" ? "line" : "bar",
                       data: rows.map(function (r) { return r[y]; })};
              if (kind === "area") s.areaStyle = {};
              return s;
            })};
  }

  function quickChart(binding) {
    var wrap = el("div");
    var xSel = el("select");
    var ySel = el("select");
    binding.columns.forEach(function (c) {
      var ox = el("option", null, c); ox.value = c; xSel.appendChild(ox);
      var oy = el("option", null, c); oy.value = c; ySel.appendChild(oy);
    });
    if (binding.columns.length > 1) ySel.selectedIndex = 1;
    var chartDiv = el("div", "wb-chart");
    chartDiv.style.display = "none";
    var markupPre = el("pre");
    markupPre.style.display = "none";
    var markupCode = el("code");
    markupPre.appendChild(markupCode);
    var actions = el("div");

    var row = el("div", "wb-row");
    row.appendChild(el("span", "wb-meta", "x"));
    row.appendChild(xSel);
    row.appendChild(el("span", "wb-meta", "y"));
    row.appendChild(ySel);
    row.appendChild(button("chart it", null, function () {
      var x = xSel.value, y = ySel.value;
      chartDiv.style.display = "block";
      var chart = window.echarts.init(chartDiv);
      chart.setOption(chartOption("bar", x, [y], binding.preview));
      /* The copy-paste IS the adoption gesture: the markup below is the
         real figure grammar, ready for template.html. */
      var markup = '<div data-tb-figure="chart" data-tb-binding="'
          + binding.name + '" data-tb-type="bar" data-tb-x="' + x
          + '" data-tb-y="' + y + '" id="fig-' + binding.name
          + '" style="height:320px"></div>';
      markupCode.textContent = markup;
      markupPre.style.display = "block";
      actions.textContent = "";
      actions.appendChild(copyBtn("copy markup", markup));
    }));
    wrap.appendChild(row);
    wrap.appendChild(chartDiv);
    wrap.appendChild(markupPre);
    wrap.appendChild(actions);
    return wrap;
  }

  function renderData(state) {
    var host = document.getElementById("wb-data");
    host.textContent = "";
    state.bindings.forEach(function (b) {
      var card = el("div", "wb-card");
      var head = el("div", "wb-row");
      head.appendChild(el("span", "tb-badge wb-badge tb-badge--"
          + (b.source === "python" ? "derived" : "verified"),
          b.source === "python" ? "python" : "query"));
      head.appendChild(el("strong", null, b.name));
      if (b.error === undefined) {
        head.appendChild(el("span", "wb-meta",
            b.rows + " × " + b.columns.length + " · " + b.fingerprint));
      }
      card.appendChild(head);
      if (b.error !== undefined) {
        card.appendChild(el("pre", "wb-error", b.error));
        host.appendChild(card);
        return;
      }
      if (!b.used_by.length) {
        card.appendChild(el("p", "wb-meta wb-warn",
            "unused — no figure references this binding"));
      }
      card.appendChild(el("p", "wb-meta", b.columns.map(function (c) {
        return c + ": " + b.dtypes[c];
      }).join("  ·  ")));
      card.appendChild(smallTable(b.columns, b.preview, 25, false, b.display));
      card.appendChild(quickChart(b));
      host.appendChild(card);
    });
  }

  function exhibitFrame(card, ex) {
    if (ex.name) card.appendChild(el("strong", null, ex.name));
    if (ex.note) card.appendChild(el("p", null, ex.note));
    if (ex.shape) {
      card.appendChild(el("p", "wb-meta",
          ex.shape[0] + " × " + ex.shape[1]
          + (ex.shape[0] > (ex.rows || []).length
             ? " (first " + (ex.rows || []).length + " shown)" : "")));
    }
    card.appendChild(smallTable(ex.columns || [], ex.rows || [], 10, true,
                                ex.display));
    var prof = profileToggle(ex.profile);
    if (prof) card.appendChild(prof);
  }

  function exhibitChart(card, ex) {
    var recipe = ex.recipe || {};
    var ys = recipe.y || [];
    if (!window.echarts || !window.echarts.init || !recipe.chart) {
      /* No ECharts (or a torn entry) — degrade to the frame-table
         rendering, never a blank card. */
      exhibitFrame(card, ex);
      return;
    }
    if (ex.name) card.appendChild(el("strong", null, ex.name));
    var chartDiv = el("div", "wb-chart");
    card.appendChild(chartDiv);
    var option = chartOption(recipe.chart, recipe.x, ys, ex.rows || []);
    if (ex.note) option.title = {text: ex.note, textStyle: {fontSize: 13}};
    /* init after the card lands in the DOM, or ECharts sees width 0. */
    setTimeout(function () {
      window.echarts.init(chartDiv).setOption(option);
    }, 0);
    card.appendChild(el("p", "wb-meta",
        recipe.chart + " · x: " + recipe.x + " · y: " + ys.join(", ")
        + " — the sketch's recipe; promotion = re-expressing it as a "
        + "binding + figure"));
    var prof = profileToggle(ex.profile);
    if (prof) card.appendChild(prof);
  }

  var feedStep = "all";

  function keepBtn(state, ex) {
    /* "Keep this": a request for the agent to promote this exhibit into a
       figure. The workbench never edits the report itself. */
    var pid = "keep-" + ex.seq;
    var kept = state.pins.some(function (p) { return p.id === pid; });
    if (kept) {
      return button("kept · undo", "wb-btn--pinned", function () {
        post("/__workbench/unpin", {id: pid});
      });
    }
    return button("keep this", null, function () {
      var note = window.prompt(
        "What should this become? (optional — e.g. 'bar chart under the KPIs')",
        "");
      if (note === null) return;
      post("/__workbench/pin",
           {id: pid, kind: "promote", exhibit: ex.seq, note: note});
    });
  }

  function scroller() {
    return document.body.classList.contains("wb-split-on")
      ? document.getElementById("wb-panel") : document.scrollingElement;
  }

  var STEP_LABELS = {transform: "Transform", model: "Model",
                     pipeline: "Pipeline", report: "Report", script: "Script"};

  function exhibitCell(state, ex) {
    /* A notebook cell: In [n] is the code that ran show(), Out [n] is what
       it showed. */
    var cell = el("div", "wb-cell");
    var head = el("div", "wb-cell-head");
    head.appendChild(el("span", "wb-in", "In [" + ex.seq + "]"));
    var o = ex.origin || {};
    if (o.file) head.appendChild(el("span", "wb-meta", o.file + ":" + o.line));
    if (ex.change) {
      head.appendChild(el("span", "wb-chip wb-chip--" + ex.change,
          {new: "new", changed: "changed since last run",
           same: "unchanged"}[ex.change] || ex.change));
    }
    var pid = "exhibit-" + ex.seq;
    head.appendChild(pinBtn(pid, state.pins.some(function (p) {
      return p.id === pid;
    })));
    if (ex.kind === "frame" || ex.kind === "chart") {
      head.appendChild(keepBtn(state, ex));
    }
    cell.appendChild(head);
    if (o.code) cell.appendChild(el("pre", "wb-code", o.code));
    var out = el("div", "wb-out");
    out.appendChild(el("span", "wb-in", "Out [" + ex.seq + "]"));
    if (ex.kind === "chart") exhibitChart(out, ex);
    else exhibitFrame(out, ex);
    cell.appendChild(out);
    return cell;
  }

  function noteCell(ex) {
    var cell = el("div", "wb-cell");
    if (ex.html) {
      /* The ONE innerHTML exception on this page: ex.html is produced
         server-side by the escaped-first markdown subset
         (render_note_markdown), so a note reads like a notebook cell
         while content can never smuggle live markup. */
      var md = el("div", "wb-md");
      md.innerHTML = ex.html;
      cell.appendChild(md);
    } else {
      var text = ex.text || "";
      cell.appendChild(el(text.indexOf("\\n") !== -1 ? "pre" : "p", null, text));
    }
    if (ex.note) cell.appendChild(el("p", "wb-meta", ex.note));
    return cell;
  }

  function myMessage(p) {
    var bubble = el("div", "wb-me");
    bubble.appendChild(el("span", "wb-meta", "You"
        + (p.at ? " · " + p.at.replace("T", " ") : "")
        + (p.kind === "promote" ? " · keep #" + p.exhibit : "")));
    bubble.appendChild(el("span", null, p.kind === "promote"
        ? (p.note || "Keep this as a figure.") : p.note));
    return bubble;
  }

  function renderFeed(state) {
    var host = document.getElementById("wb-feed");
    var sc = scroller();
    var atBottom = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 80;
    host.textContent = "";
    var resolved = state.resolved || [];
    if (!state.exhibits.length && !state.pins.length && !resolved.length) {
      host.appendChild(el("p", "wb-meta", "Nothing yet. Code that calls "
          + "tracebi.workbench.show(df, note=...) — a transform, report.py, "
          + "or a script your agent runs — lands here as a notebook cell, "
          + "newest at the bottom. Write below to talk to your agent."));
      return;
    }
    /* One chip per workflow step present, so a long session reads by stage. */
    var steps = ["all"];
    state.exhibits.forEach(function (ex) {
      var st = ex.step || "script";
      if (ex.kind !== "auto" && steps.indexOf(st) === -1) steps.push(st);
    });
    if (steps.length > 2) {
      var chips = el("div", "wb-row");
      steps.forEach(function (st) {
        chips.appendChild(button(STEP_LABELS[st] || st,
          st === feedStep ? "wb-btn--pinned" : null,
          function () { feedStep = st; renderFeed(state); }));
      });
      host.appendChild(chips);
    }
    /* Oldest first. Your messages and keeps sit after the exhibit that was
       newest when you wrote them (at_seq). */
    var items = state.exhibits.map(function (ex) {
      return {order: ex.seq, ex: ex};
    });
    state.pins.forEach(function (p) {
      if (p.kind === "message" || p.kind === "promote") {
        items.push({order: (p.at_seq || 0) + 0.5, pin: p});
      }
    });
    resolved.forEach(function (p) {
      items.push({order: (p.at_seq || 0) + 0.5, done: p});
    });
    items.sort(function (a, b) { return a.order - b.order; });
    var lastStep = null;
    items.forEach(function (it) {
      if (it.pin) {
        if (feedStep === "all") host.appendChild(myMessage(it.pin));
        return;
      }
      if (it.done) {
        if (feedStep === "all") host.appendChild(el("div", "wb-marker",
            "done · " + (it.done.resolved_note || "")));
        return;
      }
      var ex = it.ex;
      var st = ex.step || "script";
      if (feedStep !== "all" && st !== feedStep) return;
      if (ex.kind === "auto") {
        host.appendChild(el("div", "wb-marker", ex.text));
        return;
      }
      var where = (ex.origin && ex.origin.file) || ex.source || "";
      var key = st + "|" + where;
      if (key !== lastStep) {
        host.appendChild(el("div", "wb-divider",
            (STEP_LABELS[st] || st) + (where ? " · " + where : "")));
        lastStep = key;
      }
      if (ex.kind === "frame" || ex.kind === "chart") {
        host.appendChild(exhibitCell(state, ex));
      } else if (ex.kind === "binding") {
        host.appendChild(el("div", "wb-marker", "binding: " + ex.name
            + (ex.note ? " — " + ex.note : "")));
      } else {
        host.appendChild(noteCell(ex));
      }
    });
    if (atBottom && activeTab === "timeline") sc.scrollTop = sc.scrollHeight;
  }

  function renderRequests(state) {
    var sec = document.getElementById("wb-sec-requests");
    var host = document.getElementById("wb-requests");
    host.textContent = "";
    var reqs = state.pins.filter(function (p) {
      return p.kind === "promote" || p.kind === "message";
    });
    sec.hidden = !reqs.length;
    reqs.forEach(function (p) {
      var card = el("div", "wb-card wb-request");
      card.appendChild(el("p", null, p.request));
      var row = el("div", "wb-row");
      row.appendChild(copyBtn("copy request", p.request));
      row.appendChild(button("remove", null, function () {
        post("/__workbench/unpin", {id: p.id});
      }));
      card.appendChild(row);
      host.appendChild(card);
    });
  }

  /* ── Side by side: click a figure in the preview to inspect it. ── */
  var selectedId = null;
  var lastState = null;

  function renderSelected(state) {
    var sec = document.getElementById("wb-sec-selected");
    var host = document.getElementById("wb-selected");
    host.textContent = "";
    var f = selectedId && state.figures.filter(function (x) {
      return x.id === selectedId;
    })[0];
    sec.hidden = !f;
    if (!f) return;
    var card = el("div", "wb-card wb-selected");
    var head = el("div", "wb-row");
    head.appendChild(badge(f.unbound ? "error" : f.provenance));
    head.appendChild(el("code", null, f.id));
    head.appendChild(el("span", "wb-meta", f.kind
        + (f.binding ? " · " + f.binding : "")));
    head.appendChild(copyBtn("copy address", REPORT + "#fig:" + f.id));
    head.appendChild(button("close", null, function () {
      selectedId = null; highlight(); renderSelected(state);
    }));
    card.appendChild(head);
    var b = state.bindings.filter(function (x) { return x.name === f.binding; })[0];
    if (b && b.error === undefined) {
      card.appendChild(el("p", "wb-meta", b.rows + " rows · fingerprint "
          + b.fingerprint));
      card.appendChild(smallTable(b.columns, b.preview, 25, true, b.display));
    } else if (b && b.error !== undefined) {
      card.appendChild(el("pre", "wb-error", b.error));
    } else if (!f.binding) {
      card.appendChild(el("p", "wb-meta",
          "no binding — this figure is marked unverified"));
    }
    var related = state.exhibits.filter(function (ex) {
      return ex.name && (ex.name === f.binding || ex.name === f.id);
    });
    if (related.length) {
      card.appendChild(el("p", "wb-meta", "exhibits named " + (f.binding || f.id)
          + ": " + related.map(function (ex) { return "#" + ex.seq; }).join(", ")));
    }
    host.appendChild(card);
  }

  function highlight() {
    var frame = document.getElementById("wb-preview");
    var doc = frame && frame.contentDocument;
    if (!doc) return;
    Array.prototype.forEach.call(doc.querySelectorAll("[data-tb-figure]"),
      function (node) {
        node.style.outline = node.id && node.id === selectedId
          ? "2px solid #2a78d6" : "";
        node.style.outlineOffset = "3px";
      });
  }

  function wirePreview() {
    var frame = document.getElementById("wb-preview");
    var doc = frame.contentDocument;
    if (!doc) return;
    Array.prototype.forEach.call(doc.querySelectorAll("[data-tb-figure]"),
      function (node) {
        if (!node.id) return;
        node.style.cursor = "pointer";
        node.title = "Inspect " + node.id + " in the workbench";
        node.addEventListener("click", function () {
          selectedId = node.id;
          highlight();
          if (lastState) renderSelected(lastState);
          showTab("figures");
          document.getElementById("wb-panel").scrollTop = 0;
        });
      });
    highlight();
  }

  function renderCode(state) {
    var host = document.getElementById("wb-code");
    host.textContent = "";
    ["report.json", "report.py", "script.js"].forEach(function (label) {
      var text = (state.code || {})[label];
      if (!text) return;
      host.appendChild(el("h3", null, label));
      host.appendChild(el("pre", null, text));
    });
  }

  function renderLint(state) {
    var host = document.getElementById("wb-lint");
    host.textContent = "";
    var ul = el("ul");
    ul.appendChild(el("li", null,
        state.lint.numeric_literals_outside_figures
        + " numeric literal(s) in prose outside figures"));
    ul.appendChild(el("li", null, state.unused_bindings.length
        + " unused binding(s)"
        + (state.unused_bindings.length
           ? ": " + state.unused_bindings.join(", ") : "")));
    var unbound = state.figures.filter(function (f) { return f.unbound; });
    ul.appendChild(el("li", null, unbound.length + " unbound figure(s)"
        + (unbound.length
           ? ": " + unbound.map(function (f) { return f.id; }).join(", ")
           : "")));
    host.appendChild(ul);
    host.appendChild(el("p", "wb-meta",
        "non-blocking — the final build enforces; the workbench points."));
  }

  /* ── Discovery mode: no report anchored — phases ① and ② live here. ── */

  function renderWarehouse(state) {
    var host = document.getElementById("wb-warehouse");
    host.textContent = "";
    var wh = state.warehouse || {};
    if (wh.error) host.appendChild(el("pre", "wb-error", wh.error));
    if (!wh.exists) {
      host.appendChild(el("p", "wb-meta",
          "no warehouse yet — run a transform"));
      return;
    }
    host.appendChild(el("p", "wb-meta", wh.path));
    var tables = wh.tables || [];
    tables.forEach(function (tb) {
      var card = el("div", "wb-card");
      var head = el("div", "wb-row");
      head.appendChild(el("strong", null, tb.name));
      head.appendChild(el("span", "wb-meta", tb.rows + " rows"));
      /* Same pin mechanism as figures — a warehouse table pins by the
         stable id "table-<name>". */
      var pid = "table-" + tb.name;
      head.appendChild(pinBtn(pid, state.pins.some(function (p) {
        return p.id === pid;
      })));
      card.appendChild(head);
      if (tb.error) card.appendChild(el("pre", "wb-error", tb.error));
      var cols = tb.columns || {};
      card.appendChild(el("p", "wb-meta",
          Object.keys(cols).map(function (c) {
            return c + ": " + cols[c];
          }).join("  ·  ")));
      var prof = profileToggle(tb.profile, tb.note);
      if (prof) card.appendChild(prof);
      host.appendChild(card);
    });
    if (!tables.length) {
      host.appendChild(el("p", "wb-meta", "the warehouse has no tables"));
    }
    var contracts = wh.contracts || {};
    Object.keys(contracts).forEach(function (t) {
      var c = contracts[t];
      /* Locked language: the SINK satisfied its contract — this line
         certifies the landed tables, never the transform's pandas. */
      host.appendChild(el("p", "wb-meta",
          "sink contract " + t + " · " + c.checks + " checks passed"
          + (c.checked_at ? " · " + c.checked_at : "")
          + " · tables: " + (c.tables || []).join(", ")));
    });
  }

  function renderModels(state) {
    var host = document.getElementById("wb-models");
    host.textContent = "";
    var models = state.models || [];
    if (!models.length) {
      host.appendChild(el("p", "wb-meta",
          "no models yet — tracebi new-model"));
      return;
    }
    function names(list) {
      return (list || []).map(function (x) { return x.name; }).join(", ");
    }
    models.forEach(function (m) {
      var card = el("div", "wb-card");
      var head = el("div", "wb-row");
      head.appendChild(el("strong", null, m.name));
      card.appendChild(head);
      if (m.error !== undefined) {
        card.appendChild(el("pre", "wb-error", m.error));
        host.appendChild(card);
        return;
      }
      card.appendChild(el("p", "wb-meta", "facts: " + (names(m.facts) || "—")));
      card.appendChild(el("p", "wb-meta",
          "dimensions: " + (names(m.dimensions) || "—")));
      card.appendChild(el("p", "wb-meta",
          "measures: " + (names(m.measures) || "—")));
      host.appendChild(card);
    });
  }

  function renderPackages(state) {
    var host = document.getElementById("wb-packages");
    host.textContent = "";
    var packages = state.packages || [];
    if (!packages.length) {
      host.appendChild(el("p", "wb-meta",
          "no packages yet — tracebi new-report \\"My Report\\""));
      return;
    }
    packages.forEach(function (name) {
      var row = el("div", "wb-row");
      row.appendChild(el("strong", null, name));
      row.appendChild(el("span", "wb-meta", "tracebi dev " + name));
      host.appendChild(row);
    });
  }

  var activeTab = "timeline";
  var mode = "package";

  function applyTabs() {
    Array.prototype.forEach.call(
      document.querySelectorAll("#wb-tabs button"), function (b) {
        b.classList.toggle("wb-off", !!b.dataset.mode && b.dataset.mode !== mode);
        b.classList.toggle("wb-on", b.dataset.tab === activeTab);
      });
    Array.prototype.forEach.call(
      document.querySelectorAll("section[data-tab]"), function (sec) {
        sec.classList.toggle("wb-off", sec.dataset.tab !== activeTab
            || (!!sec.dataset.mode && sec.dataset.mode !== mode));
      });
  }

  function showTab(tab) {
    activeTab = tab;
    applyTabs();
    if (tab === "timeline") {
      var sc = scroller();
      sc.scrollTop = sc.scrollHeight;
    }
  }

  Array.prototype.forEach.call(
    document.querySelectorAll("#wb-tabs button"), function (b) {
      b.addEventListener("click", function () { showTab(b.dataset.tab); });
    });

  document.getElementById("wb-composer").addEventListener("submit", function (e) {
    e.preventDefault();
    var box = document.getElementById("wb-message");
    var text = box.value.trim();
    if (!text) return;
    box.value = "";
    post("/__workbench/pin", {id: "msg-" + Date.now(), kind: "message",
                              note: text});
  });
  document.getElementById("wb-message").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      document.getElementById("wb-composer").requestSubmit();
    }
  });

  function setMode(discovery) {
    mode = discovery ? "discovery" : "package";
    applyTabs();
    document.getElementById("wb-preview-link").hidden = discovery;
    var frame = document.getElementById("wb-preview");
    var layoutBtn = document.getElementById("wb-layout");
    layoutBtn.hidden = discovery;
    if (!discovery && !frame.getAttribute("src")) {
      frame.addEventListener("load", wirePreview);
      frame.setAttribute("src", "/");
      document.body.classList.add("wb-split-on");
      layoutBtn.addEventListener("click", function () {
        var on = document.body.classList.toggle("wb-split-on");
        layoutBtn.textContent = on ? "full width" : "side by side";
      });
    }
  }

  function render(state) {
    var discovery = state.mode === "discovery";
    setMode(discovery);
    var title = discovery ? "discovery workbench"
                          : state.name + " — workbench";
    document.getElementById("wb-title").textContent = title;
    document.title = title;
    var err = document.getElementById("wb-error");
    err.textContent = "";
    var problem = state.error || state.render_error;
    if (problem) err.appendChild(el("pre", "wb-error", problem));
    if (discovery) {
      renderWarehouse(state);
      renderModels(state);
      renderPackages(state);
      renderRequests(state);
      renderFeed(state);
      return;
    }
    lastState = state;
    renderRequests(state);
    renderSelected(state);
    renderFigures(state);
    renderData(state);
    renderFeed(state);
    renderCode(state);
    renderLint(state);
  }

  function refresh() {
    return fetch("/__workbench/state.json")
      .then(function (r) { return r.text(); })
      .then(function (text) {
        if (text === last) return;
        last = text;
        render(JSON.parse(text));
      })
      .catch(function () {});
  }

  refresh();
  setInterval(refresh, 2000);
})();
</script>
</body></html>"""


def _workbench_page(name: str) -> str:
    """The workbench shell, styled by the shipped design system. Static —
    all data arrives via /__workbench/state.json polling."""
    from tracebi.reports.embed import CSP, read_lib
    from tracebi.reports.stack import read_asset

    # Defense-in-depth for the ONE innerHTML on this page (md_to_html output):
    # the escaping in _inline is the real protection, but unlike the built
    # artifact this page carried no CSP. Add one, relaxing connect-src to 'self'
    # for the /__workbench/state.json poll (as the served exploration page does).
    csp = CSP.replace("connect-src 'none'", "connect-src 'self'")
    # The side-by-side view frames the live preview, served by this server.
    csp = csp.replace("default-src 'none';", "default-src 'none'; frame-src 'self';")
    csp_meta = f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
    return (_WORKBENCH_PAGE
            .replace("__CSP__", csp_meta)
            .replace("__NAME__", json.dumps(name))
            .replace("__TRACEBI_CSS__", read_asset("tracebi.css"))
            .replace("__ECHARTS__", read_lib("echarts")))


# ── The server ──────────────────────────────────────────────────────────────


def serve_dev(
    target,
    port: int = 8001,
    open_browser: bool = True,
    poll_interval: float = 0.5,
) -> int:
    """Serve *target* with live reload until Ctrl+C. Returns an exit code.

    *target* is an artifact package directory (``reports/<name>/``) or
    ``None`` — discovery mode: no report anchored, the project-level workbench
    served for phases ① and ②.
    """
    if target is None:
        return _serve(_DiscoveryTarget(), port=port,
                      open_browser=open_browser, poll_interval=poll_interval)
    target_path = Path(target)
    if not target_path.is_dir():
        print(f"  error: {target_path} is not an artifact package. "
              f"`tracebi dev <name>` serves a package under reports/<name>/.",
              file=sys.stderr)
        return 1
    return _serve(_PackageTarget(target_path), port=port,
                  open_browser=open_browser, poll_interval=poll_interval)


def _serve(t, port: int, open_browser: bool, poll_interval: float) -> int:
    from tracebi import workbench as _wb

    # No bytecode for hot-reloaded project files: a .pyc written by this
    # process would be judged "still valid" against a same-second, same-size
    # edit of a model file (mtime compares in whole seconds), and the rebuild
    # would silently exec stale code — the one failure a dev loop must not
    # have.
    sys.dont_write_bytecode = True

    state = {"html": t.render(), "version": 0}
    lock = threading.Lock()
    wb_cache: dict = {}
    wb_page = _workbench_page(t.label) if t.workbench else ""
    stop = threading.Event()   # set when the server stops, so the watcher ends with it

    def watch():
        # Discovery liveness: the heartbeat is what lets a script's show()
        # post with no env var — touched every tick, stale within seconds
        # of Ctrl+C (the rule in tracebi/workbench.py).
        if t.discovery:
            _wb.heartbeat(t.wb_dir)
        last_sig = _scan_signature(t.watch_paths())
        while not stop.wait(poll_interval):
            if t.discovery:
                _wb.heartbeat(t.wb_dir)
            sig = _scan_signature(t.watch_paths())
            if sig != last_sig:
                last_sig = sig
                html = t.render()
                with lock:
                    state["html"] = html
                    state["version"] += 1
                print(f"  Reloaded {t.label} (v{state['version']})")

    def workbench_state() -> dict:
        # Recomputed only when the report version or the feed/pin files
        # moved — a 2s poll must not re-run the bindings' queries.
        with lock:
            version = state["version"]
        key = (version,
               _file_sig(os.path.join(t.wb_dir, _wb.EXHIBITS_FILE)),
               _file_sig(os.path.join(t.wb_dir, _wb.PINS_FILE)))
        if wb_cache.get("key") != key:
            wb_cache["key"] = key
            wb_cache["state"] = t.state()
        return wb_cache["state"]

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self) -> bool:
            # Block DNS rebinding: a malicious domain resolving to 127.0.0.1
            # could otherwise read the rich /__workbench/state.json.
            return _host_is_local(self.headers.get("Host", ""))

        def do_GET(self):
            if not self._host_ok():
                self._send(b'{"error": "host not allowed"}',
                           "application/json", 403)
                return
            if self.path == "/__status":
                with lock:
                    body = json.dumps({"version": state["version"]}).encode()
                self._send(body, "application/json")
                return
            if t.workbench and self.path == "/__workbench":
                self._send(wb_page.encode(), "text/html; charset=utf-8")
                return
            if t.workbench and self.path == "/__workbench/state.json":
                body = json.dumps(workbench_state(), default=str).encode()
                self._send(body, "application/json")
                return
            with lock:
                body = _inject_refresh(state["html"], state["version"]).encode()
            self._send(body, "text/html; charset=utf-8")

        def do_POST(self):
            if not self._host_ok():
                self._send(b'{"error": "host not allowed"}',
                           "application/json", 403)
                return
            if not (t.workbench
                    and self.path in ("/__workbench/pin", "/__workbench/unpin")):
                self._send(b'{"error": "not found"}', "application/json", 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                payload = {}
            pin_id = payload.get("id")
            if not pin_id:
                self._send(b'{"error": "missing id"}', "application/json", 400)
                return
            with lock:
                if self.path == "/__workbench/pin":
                    # kind "promote": "Keep this" on an exhibit, a request for
                    # the agent to promote it into a figure. kind "message": the
                    # author typing to the agent in the timeline.
                    pins = _wb.add_pin(
                        t.wb_dir, pin_id, note=payload.get("note") or "",
                        kind=payload.get("kind"), exhibit=payload.get("exhibit"))
                else:
                    pins = _wb.remove_pin(t.wb_dir, pin_id)
            self._send(json.dumps({"ok": True, "pins": pins}).encode(),
                       "application/json")

        def log_message(self, fmt, *args):  # silence request logs
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    url = f"http://127.0.0.1:{port}"
    if t.discovery:
        print("\n  TraceBi dev — discovery mode (no report anchored)")
        print(f"  Workbench at {url}/__workbench")
        print("  Exhibits: any script you run can call "
              "tracebi.workbench.show(df, note=...) while this server is "
              "up — no env var needed (or set TRACEBI_WORKBENCH_DIR=<dir> "
              "explicitly).")
    else:
        print(f"\n  TraceBi dev — watching {t.label}")
        print(f"  Preview at {url} (reloads on save)")
        if t.workbench:
            print(f"  Workbench at {url}/__workbench")
    print("  Press Ctrl+C to stop.\n")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dev server stopped.")
    finally:
        stop.set()
        server.server_close()
    return 0

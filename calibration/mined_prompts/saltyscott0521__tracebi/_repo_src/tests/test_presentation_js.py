"""
The presentation runtime asset — tracebi/reports/assets/tracebi.js (M2).

The runtime ships inside self-contained HTML under the strict CSP, so the
asset must be dependency-free, eval-free, and must never build markup from
data. Its number formatter is a JS port of ``ChartSpec._fmt`` — the one
implementation of "550.7B" — so when node is available the two are compared
byte-for-byte on the values where the port could plausibly drift (unit
boundaries, mantissa rollover, negatives, trailing-zero trimming).
"""

import json
import os
import shutil
import subprocess

import pytest

import tracebi.reports
from tracebi.reports.chart import ChartSpec

ASSET = os.path.join(
    os.path.dirname(tracebi.reports.__file__), "assets", "tracebi.js"
)

#: (value, compact) pairs covering the honesty rules: rounded-value unit
#: boundaries (999999.99 → "1M", never "1000K"), K→M→B→T mantissa rollover,
#: integer thousands separators, and trailing-zero trimming.
COMPACT_VALUES = [
    0,
    123,
    999.999,
    1234.5,
    999999,
    999999.99,
    2400240.38,
    550696024575,
    -1500000,
    1e12 * 1.5,
]


class TestAssetHygiene:
    def test_no_eval(self):
        with open(ASSET, encoding="utf-8") as f:
            src = f.read()
        assert "eval(" not in src

    def test_no_innerhtml(self):
        # Data reaches the DOM only via textContent / ECharts — never markup.
        with open(ASSET, encoding="utf-8") as f:
            src = f.read()
        assert src.count("innerHTML") == 0

@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestFmtParity:
    """tracebi.fmt must agree byte-for-byte with ChartSpec._fmt."""

    def _js_fmt(self, cases):
        """Run tracebi.fmt in node for [(value, mode), ...] → list of strings."""
        script = (
            "require(process.argv[1]);"
            "var cases = JSON.parse(process.argv[2]);"
            "var out = cases.map(function (c) {"
            "  return globalThis.tracebi.fmt(c[0], c[1] || undefined);"
            "});"
            "process.stdout.write(JSON.stringify(out));"
        )
        result = subprocess.run(
            ["node", "-e", script, ASSET, json.dumps(cases)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def test_compact_parity(self):
        expected = [ChartSpec._fmt(v, compact=True) for v in COMPACT_VALUES]
        actual = self._js_fmt([[v, "compact"] for v in COMPACT_VALUES])
        assert actual == expected

    def test_non_compact_parity(self):
        assert self._js_fmt([[1234.5, None]]) == ["1,234.5"]
        assert ChartSpec._fmt(1234.5) == "1,234.5"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestSsrValueFormatParity:
    """The SSR fill formats value figures in Python; the runtime hydrates the
    same figure with applyNamedFormat. They MUST be byte-identical or a no-JS
    number and the hydrated one disagree — a visible flicker / a changed number.
    This pins _ssr_format against applyNamedFormat over a corpus that stresses
    half-to-even ties, negative zero, and large/small magnitudes.
    """

    NAMES = ["currency", "currency0", "comma", "decimal", "percent"]
    EDGES = [0, 1, -1, 0.5, -0.5, 1.5, 2.5, 3.5, -2.5, 1234.5, -1234.5,
             0.005, -0.005, 0.125, -0.125, 0.069, -0.3, -0.001, -0.00005,
             999.995, 1000000, 12345.678, -0.0]

    def _js(self, cases):
        script = (
            "var fs=require('fs');var src=fs.readFileSync(process.argv[1],'utf8');"
            "var p=src.replace('root.tracebi = {',"
            " 'root.tracebi = { applyNamedFormat: applyNamedFormat, toNum: toNum,');"
            "if(p===src) throw new Error('export line not found');"
            "new Function(p)();"
            "var cs=JSON.parse(process.argv[2]);"
            "process.stdout.write(JSON.stringify(cs.map(function(c){"
            "  return globalThis.tracebi.applyNamedFormat("
            "    globalThis.tracebi.toNum(c[0]), c[1]); })));"
        )
        r = subprocess.run(["node", "-e", script, ASSET, json.dumps(cases)],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    def test_ssr_format_matches_the_runtime(self):
        from tracebi.reports.template_package import _ssr_format
        cases = [[v, n] for v in self.EDGES for n in self.NAMES]
        js = self._js(cases)
        for (v, n), expected in zip(cases, js):
            got = _ssr_format(v, n)
            assert got == expected, f"{v} {n}: python {got!r} != js {expected!r}"

    def test_non_finite_renders_the_raw_token_and_never_crashes(self):
        # A null in a formatted numeric column used to crash `report build` in
        # _is_tie (NaN slips past `a >= 1e15`). The runtime's toNum returns null
        # for a non-finite value, so renderBody skips the format and shows the
        # raw token: NaN → blank, inf → "inf". SSR must match, byte for byte.
        from tracebi.reports.template_package import _ssr_format, _is_tie
        assert _is_tie(float("nan"), 0) is False        # the crash site, hardened
        assert _is_tie(float("inf"), 3) is False
        for name in self.NAMES:
            assert _ssr_format(float("nan"), name) == ""
            assert _ssr_format(float("inf"), name) == "inf"
            assert _ssr_format(float("-inf"), name) == "-inf"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestChartValueFormat:
    """optionFor honors data-tb-value-format on every chart type and mode."""

    PIE_ROWS = [
        {"region": "US", "mv": "550696024575"},
        {"region": "EU", "mv": "1234.5"},
    ]

    def _probe(self, plan, rows, expr):
        """Build optionFor(plan, rows) under node; JSON-print `expr` over opt.

        optionFor is internal to the runtime's IIFE (the public global stays
        data/fmt/configureChart), so the harness exposes it by rewriting the
        export line of the source before executing it — the asset on disk is
        untouched, and a failed rewrite fails loudly below.
        """
        script = (
            "var fs = require('fs');"
            "var src = fs.readFileSync(process.argv[1], 'utf8');"
            "var patched = src.replace('root.tracebi = {',"
            " 'root.tracebi = { optionFor: optionFor,');"
            "if (patched === src) throw new Error('export line not found');"
            "new Function(patched)();"
            "var opt = globalThis.tracebi.optionFor("
            "JSON.parse(process.argv[2]), JSON.parse(process.argv[3]));"
            "process.stdout.write(JSON.stringify(" + expr + "));"
        )
        result = subprocess.run(
            ["node", "-e", script, ASSET, json.dumps(plan), json.dumps(rows)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def _pie_plan(self, value_format=None):
        plan = {"type": "pie", "x": "region", "y": ["mv"], "palette": []}
        if value_format is not None:
            plan["valueFormat"] = value_format
        return plan

    def test_pie_without_value_format_has_no_formatter(self):
        out = self._probe(
            self._pie_plan(), self.PIE_ROWS,
            "[typeof opt.series[0].label.formatter,"
            " typeof opt.tooltip.valueFormatter,"
            " opt.series[0].label.show]",
        )
        assert out == ["undefined", "undefined", True]

    def test_pie_label_and_tooltip_use_fmt(self):
        expected = ChartSpec._fmt(550696024575, compact=True)
        out = self._probe(
            self._pie_plan("compact"), self.PIE_ROWS,
            "[opt.series[0].label.formatter(opt.series[0].data[0]),"
            " opt.tooltip.valueFormatter(opt.series[0].data[0].value),"
            " opt.series[0].label.show]",
        )
        assert out == [f"US: {expected}", expected, True]

    def test_categorical_honors_non_compact_mode(self):
        plan = {"type": "bar", "x": "region", "y": ["mv"],
                "valueFormat": "currency", "palette": []}
        out = self._probe(
            plan, self.PIE_ROWS,
            "[opt.yAxis.axisLabel.formatter(1234.5),"
            " opt.series[0].label.formatter({value: 1234.5})]",
        )
        assert out == ["$1,234.50", "$1,234.50"]

    def test_unknown_mode_returns_raw_value(self):
        # The guard: an unknown mode never blanks a number.
        plan = {"type": "bar", "x": "region", "y": ["mv"],
                "valueFormat": "bogus", "palette": []}
        out = self._probe(
            plan, self.PIE_ROWS,
            "[opt.yAxis.axisLabel.formatter(1234.5),"
            " opt.series[0].label.formatter({value: 1234.5})]",
        )
        assert out == [1234.5, 1234.5]

    def test_cartesian_tooltip_uses_value_format(self):
        plan = {"type": "bar", "x": "region", "y": ["mv"],
                "valueFormat": "currency", "palette": []}
        out = self._probe(
            plan, self.PIE_ROWS, "[opt.tooltip.valueFormatter(1234.5)]")
        assert out == ["$1,234.50"]

    def test_scatter_axes_and_tooltip_use_value_format(self):
        plan = {"type": "scatter", "x": "mv", "y": ["mv"],
                "valueFormat": "compact", "palette": []}
        expected = ChartSpec._fmt(550696024575, compact=True)
        out = self._probe(
            plan, self.PIE_ROWS,
            "[opt.xAxis.axisLabel.formatter(550696024575),"
            " opt.yAxis.axisLabel.formatter(550696024575),"
            " opt.tooltip.valueFormatter(550696024575)]",
        )
        assert out == [expected, expected, expected]

# ── The control-grammar harness ──────────────────────────────────────────
# A DOM stub covering exactly the element surface the interactive runtime
# touches: compound + descendant selectors, attributes, events, Blob/URL,
# and an echarts stub that records every setOption/resize. The fixture is
# built BEFORE the runtime loads, so the load-time hydration pass exercises
# the real wiring; internals (filteredRows) are exposed by the same
# export-line rewrite _probe uses — the asset on disk is untouched.

_DOM_STUB = """
var fs = require('fs');

function El(tag) {
  this.tagName = String(tag).toUpperCase();
  this.attributes = {};
  this.children = [];
  this.parentNode = null;
  this.className = '';
  this.id = '';
  this.title = '';
  this.value = '';
  this.textContent = '';
  this.style = {};
  this._h = {};
}
El.prototype.getAttribute = function (n) {
  if (n === 'class') return this.className || null;
  if (n === 'id') return this.id || null;
  return Object.prototype.hasOwnProperty.call(this.attributes, n)
    ? this.attributes[n] : null;
};
El.prototype.setAttribute = function (n, v) {
  if (n === 'class') this.className = String(v);
  else if (n === 'id') this.id = String(v);
  else this.attributes[n] = String(v);
};
El.prototype.removeChild = function (c) {
  var i = this.children.indexOf(c);
  if (i !== -1) this.children.splice(i, 1);
  c.parentNode = null;
  return c;
};
El.prototype.appendChild = function (c) {
  if (c.parentNode) c.parentNode.removeChild(c);
  c.parentNode = this;
  this.children.push(c);
  return c;
};
El.prototype.insertBefore = function (c, ref) {
  if (c.parentNode) c.parentNode.removeChild(c);
  var i = ref ? this.children.indexOf(ref) : -1;
  if (i === -1) this.children.push(c); else this.children.splice(i, 0, c);
  c.parentNode = this;
  return c;
};
Object.defineProperty(El.prototype, 'firstChild', {
  get: function () { return this.children[0] || null; }
});
El.prototype.addEventListener = function (type, fn) {
  (this._h[type] = this._h[type] || []).push(fn);
};
El.prototype._fire = function (type) {
  var self = this;
  (this._h[type] || []).slice().forEach(function (fn) { fn({ target: self }); });
};
El.prototype.click = function () {
  (globalThis.__clicks = globalThis.__clicks || []).push(this);
  this._fire('click');
};

function parseCompound(src) {
  var out = { tag: null, classes: [], attrs: [] };
  var re = /([a-zA-Z][\\w-]*)|\\.([\\w-]+)|\\[([\\w-]+)(?:="([^"]*)")?\\]/g, m;
  while ((m = re.exec(src)) !== null) {
    if (m[1]) out.tag = m[1].toUpperCase();
    else if (m[2]) out.classes.push(m[2]);
    else out.attrs.push([m[3], m[4] === undefined ? null : m[4]]);
  }
  return out;
}
function matchesCompound(el, c) {
  if (c.tag && el.tagName !== c.tag) return false;
  var i, v;
  for (i = 0; i < c.classes.length; i++) {
    if ((' ' + el.className + ' ').indexOf(' ' + c.classes[i] + ' ') === -1) {
      return false;
    }
  }
  for (i = 0; i < c.attrs.length; i++) {
    v = el.getAttribute(c.attrs[i][0]);
    if (v === null) return false;
    if (c.attrs[i][1] !== null && v !== c.attrs[i][1]) return false;
  }
  return true;
}
function matchesSelector(el, parts) {
  if (!matchesCompound(el, parts[parts.length - 1])) return false;
  var i = parts.length - 2, node = el.parentNode;
  while (i >= 0 && node) {
    if (matchesCompound(node, parts[i])) i--;
    node = node.parentNode;
  }
  return i < 0;
}
function queryAll(rootEl, sel) {
  var parts = sel.split(/\\s+/).filter(Boolean).map(parseCompound);
  var out = [];
  (function walk(el) {
    el.children.forEach(function (c) {
      if (matchesSelector(c, parts)) out.push(c);
      walk(c);
    });
  })(rootEl);
  return out;
}
El.prototype.querySelectorAll = function (sel) { return queryAll(this, sel); };
El.prototype.querySelector = function (sel) {
  return queryAll(this, sel)[0] || null;
};

var ROOT = new El('html');
var BODY = new El('body');
ROOT.appendChild(BODY);
globalThis.document = {
  readyState: 'complete',
  body: BODY,
  createElement: function (t) { return new El(t); },
  querySelectorAll: function (sel) { return queryAll(ROOT, sel); },
  querySelector: function (sel) { return queryAll(ROOT, sel)[0] || null; },
  getElementById: function (id) {
    var found = null;
    (function walk(el) {
      if (found) return;
      if (el.id === id) { found = el; return; }
      for (var i = 0; i < el.children.length && !found; i++) walk(el.children[i]);
    })(ROOT);
    return found;
  },
  addEventListener: function () {}
};
globalThis.addEventListener = function () {};

globalThis.Blob = function (parts, opts) {
  this.parts = parts;
  this.opts = opts || {};
};
globalThis.URL = {
  createObjectURL: function (b) { globalThis.__lastBlob = b; return 'blob:tb'; },
  revokeObjectURL: function () {}
};
globalThis.__charts = [];
globalThis.echarts = {
  init: function (target) {
    var c = { el: target, options: [], resized: 0 };
    c.setOption = function (o) { c.options.push(o); };
    c.resize = function () { c.resized++; };
    globalThis.__charts.push(c);
    return c;
  }
};

function el(tag, attrs, parent) {
  var e = new El(tag);
  for (var k in (attrs || {})) e.setAttribute(k, attrs[k]);
  (parent || BODY).appendChild(e);
  return e;
}
function dataBlock(name, csv) {
  var s = el('script', { id: 'tracebi-data-' + name, type: 'application/json' });
  s.textContent = JSON.stringify({ name: name, csv: csv });
  return s;
}
function loadRuntime() {
  var src = fs.readFileSync(process.argv[1], 'utf8');
  var patched = src.replace('root.tracebi = {',
    'root.tracebi = { _filteredRows: filteredRows,');
  if (patched === src) throw new Error('export line not found');
  new Function(patched)();
}
function hasClass(e, cls) {
  return (' ' + e.className + ' ').indexOf(' ' + cls + ' ') !== -1;
}
function allText(e) {
  var t = e.textContent || '';
  e.children.forEach(function (c) { t += ' ' + allText(c); });
  return t;
}
"""


def _run_dom(fixture: str):
    """Execute _DOM_STUB + *fixture* under node; the fixture prints JSON."""
    result = subprocess.run(
        ["node", "-e", _DOM_STUB + fixture, ASSET],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestViewLayer:
    """Controls subset WHICH STAMPED ROWS figures display — never compute.

    Filters AND-combine per binding; search ANDs in on top; table and chart
    figures re-render through filteredRows; value figures never react (a
    filtered KPI would be client-side aggregation — forbidden).
    """

    _SCRIPT = """
var CSV = 'region,category,mv\\n' +
  'US,Bond,100\\nUS,Equity,200\\nEU,Equity,300\\nEU,Bond,150\\nAPAC,Equity,50\\n';
dataBlock('b', CSV);
var kpi = el('div', { 'data-tb-figure': 'value', 'data-tb-binding': 'b',
                      'data-tb-cell': 'mv' });
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b' });
var cht = el('div', { 'data-tb-figure': 'chart', 'data-tb-binding': 'b',
                      'data-tb-x': 'region', 'data-tb-y': 'mv' });
var fRegion = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'b',
                             'data-tb-column': 'region' });
var fCat = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'b',
                          'data-tb-column': 'category' });
var search = el('input', { 'data-tb-search': '', 'data-tb-binding': 'b' });
var badSel = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'nope',
                            'data-tb-column': 'x' });
loadRuntime();
var tb = globalThis.tracebi;
var tbody = tbl.querySelector('tbody');
var out = {
  kpi_initial: kpi.textContent,
  initial_rows: tbody.children.length,
  region_options: fRegion.children.map(function (o) { return o.textContent; }),
  chart_opts_initial: __charts[0].options.length,
  bad_title: badSel.title,
  bad_options: badSel.children.length,
  no_receipt_btn: BODY.querySelector('.tb-receipt-btn') === null
};

fRegion.value = 'US';
fRegion._fire('change');
out.rows_after_region = tbody.children.length;
out.chart_cats_after = __charts[0].options[1].xAxis.data;
out.kpi_after_filter = kpi.textContent;

fCat.value = 'Equity';
fCat._fire('change');
out.and_mv = tb._filteredRows('b').map(function (r) { return r.mv; });

search.value = 'BOND';
search._fire('input');
out.and_search_count = tb._filteredRows('b').length;
out.tbody_after_search = tbody.children.length;
out.tbody_after_search_text = tbody.children.length ? tbody.children[0].children[0].textContent : null;

fCat.value = 'All';
fCat._fire('change');
out.search_and_region_mv = tb._filteredRows('b')
  .map(function (r) { return r.mv; });
out.kpi_final = kpi.textContent;
process.stdout.write(JSON.stringify(out));
"""

    def test_filters_and_combine_with_search_and_value_figures_exempt(self):
        out = _run_dom(self._SCRIPT)
        # Load-time hydration: full stamped rows everywhere.
        assert out["kpi_initial"] == "100"
        assert out["initial_rows"] == 5
        # "All" plus the sorted distinct stamped values.
        assert out["region_options"] == ["All", "APAC", "EU", "US"]
        assert out["chart_opts_initial"] == 1
        # One filter: the table tbody and the chart re-render, subset only.
        assert out["rows_after_region"] == 2
        assert out["chart_cats_after"] == ["US"]
        # Two filters on one binding AND-combine: US ∧ Equity → the 200 row.
        assert out["and_mv"] == ["200"]
        # Search ANDs in (case-insensitive): US ∧ Equity ∧ "bond" → nothing.
        assert out["and_search_count"] == 0
        # A filter/search matching nothing shows a "no data" row, not a void.
        assert out["tbody_after_search"] == 1
        assert out["tbody_after_search_text"] == "no data"
        # Releasing one filter re-widens: US ∧ "bond" → the 100 row.
        assert out["search_and_region_mv"] == ["100"]
        # Value figures never react to controls — a filtered KPI needs its
        # own binding. The KPI text survives every control change above.
        assert out["kpi_after_filter"] == "100"
        assert out["kpi_final"] == "100"
        # Unknown binding: loudly-silent — a title note, nothing populated.
        assert "unknown binding" in out["bad_title"]
        assert out["bad_options"] == 0
        # No receipt block on this page → no drawer chrome.
        assert out["no_receipt_btn"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestScrollWrap:
    """Post-hydration scroll wrap: >N rows wraps, <=N and "all" do not."""

    _SCRIPT = """
function csvRows(n) {
  var s = 'k,v\\n';
  for (var i = 0; i < n; i++) s += 'k' + i + ',' + i + '\\n';
  return s;
}
dataBlock('big', csvRows(12));
dataBlock('small', csvRows(10));
dataBlock('optout', csvRows(12));
dataBlock('five', csvRows(6));
var big = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'big' });
var small = el('table', { 'data-tb-figure': 'table',
                          'data-tb-binding': 'small' });
var optout = el('table', { 'data-tb-figure': 'table',
                           'data-tb-binding': 'optout',
                           'data-tb-rows': 'all' });
var five = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'five',
                         'data-tb-rows': '5' });
loadRuntime();
process.stdout.write(JSON.stringify({
  big_parent_class: big.parentNode.className,
  big_max_height: big.parentNode.style.maxHeight || null,
  small_unwrapped: small.parentNode === BODY,
  optout_unwrapped: optout.parentNode === BODY,
  five_parent_class: five.parentNode.className,
  big_rows: big.querySelector('tbody').children.length
}));
"""

    def test_wrap_thresholds(self):
        out = _run_dom(self._SCRIPT)
        # 12 rows > the default 10 → wrapped, sized in em from the count.
        assert out["big_parent_class"] == "tb-scroll"
        assert out["big_max_height"] and out["big_max_height"].endswith("em")
        # Presentation only: every stamped row is still in the table.
        assert out["big_rows"] == 12
        # 10 rows <= the default 10 → untouched.
        assert out["small_unwrapped"] is True
        # data-tb-rows="all" opts out even at 12 rows.
        assert out["optout_unwrapped"] is True
        # An explicit data-tb-rows="5" wraps at 6 rows.
        assert out["five_parent_class"] == "tb-scroll"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestDownload:
    """The download button ships the embedded csv string byte-verbatim."""

    # Quoted commas and a trailing newline: any re-serialisation would
    # change these bytes and break the receipt.
    CSV = 'a,b\n"x,1",2\n'

    _SCRIPT = """
var CSV = %s;
dataBlock('dl', CSV);
var btn = el('button', { 'data-tb-download': '', 'data-tb-binding': 'dl' });
var lbl = el('button', { 'data-tb-download': '', 'data-tb-binding': 'dl',
                         'data-tb-label': 'Export' });
var bad = el('button', { 'data-tb-download': '', 'data-tb-binding': 'zzz' });
var fSel = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'dl',
                          'data-tb-column': 'a' });
loadRuntime();
fSel.value = 'x,1';  /* activate a filter first: the download must ignore it */
fSel._fire('change');
btn._fire('click');
process.stdout.write(JSON.stringify({
  text_default: btn.textContent,
  text_label: lbl.textContent,
  blob_parts: globalThis.__lastBlob.parts,
  blob_type: globalThis.__lastBlob.opts.type,
  anchor_download: globalThis.__clicks[0].download,
  anchor_href: globalThis.__clicks[0].href,
  bad_title: bad.title,
  bad_handlers: (bad._h.click || []).length
}));
"""

    def test_verbatim_csv_blob(self):
        out = _run_dom(self._SCRIPT % json.dumps(self.CSV))
        assert out["text_default"] == "CSV"
        assert out["text_label"] == "Export"
        # Byte-verbatim: the embedded triple's csv string, even while a
        # filter is active — the receipt-preserving export, never the view.
        assert out["blob_parts"] == [self.CSV]
        assert out["blob_type"] == "text/csv"
        assert out["anchor_download"] == "dl.csv"
        assert out["anchor_href"] == "blob:tb"
        # Unknown binding: a title note, and no click handler wired.
        assert "unknown binding" in out["bad_title"]
        assert out["bad_handlers"] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestTabs:
    """The runtime builds the bar from labels; one section shows at a time."""

    _SCRIPT = """
var group = el('div', { 'class': 'tb-tabs' });
var s1 = el('section', { 'data-tb-tab': 'One' }, group);
var s2 = el('section', { 'data-tb-tab': 'Two' }, group);
dataBlock('t', 'x,y\\nA,1\\n');
el('div', { 'data-tb-figure': 'chart', 'data-tb-binding': 't',
            'data-tb-x': 'x', 'data-tb-y': 'y' }, s2);
loadRuntime();
var bar = group.children[0];
var out = {
  bar_class: bar.className,
  labels: bar.children.map(function (b) { return b.textContent; }),
  btn0_active: hasClass(bar.children[0], 'tb-tab-active'),
  s1_active: hasClass(s1, 'tb-tab-active'),
  s2_active: hasClass(s2, 'tb-tab-active'),
  resized_before: __charts[0].resized
};
bar.children[1]._fire('click');
out.btn1_active_after = hasClass(bar.children[1], 'tb-tab-active');
out.btn0_active_after = hasClass(bar.children[0], 'tb-tab-active');
out.s2_active_after = hasClass(s2, 'tb-tab-active');
out.s1_active_after = hasClass(s1, 'tb-tab-active');
out.resized_after = __charts[0].resized;
process.stdout.write(JSON.stringify(out));
"""

    def test_tab_bar_built_and_switches(self):
        out = _run_dom(self._SCRIPT)
        # The bar is inserted first, one button per labelled section.
        assert out["bar_class"] == "tb-tab-bar"
        assert out["labels"] == ["One", "Two"]
        # First tab active on load: button and section both marked.
        assert out["btn0_active"] is True
        assert out["s1_active"] is True
        assert out["s2_active"] is False
        # Switching moves the class pair and remeasures charts (a chart
        # hidden at init has no size until its section shows).
        assert out["btn1_active_after"] is True
        assert out["btn0_active_after"] is False
        assert out["s2_active_after"] is True
        assert out["s1_active_after"] is False
        assert out["resized_after"] > out["resized_before"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestReceiptDrawer:
    """The drawer renders ONLY what #tracebi-receipt records — display,
    never re-judged, in the locked language."""

    _SCRIPT = """
var FP = new Array(65).join('f');
var rc = el('script', { id: 'tracebi-receipt', type: 'application/json' });
rc.textContent = JSON.stringify({
  report: 'portfolio_book',
  rendered_at: '2026-08-17 09:00',
  git_sha: 'abc123d',
  figures: [
    { id: 'fig-a', kind: 'table', binding: 'holdings', fingerprint: FP },
    { id: 'fig-b', kind: 'chart', unverified: true }
  ],
  transform_contracts: {
    fact_holdings: { status: 'satisfied' },
    dim_sector: { status: 'stale' },
    dim_misc: { status: 'no_contract' }
  },
  semantic_contract_models: ['portfolio_model'],
  methodology: true
});
loadRuntime();
var btn = BODY.querySelector('.tb-receipt-btn');
var drawer = BODY.querySelector('.tb-receipt-drawer');
var rows = drawer.querySelectorAll('.tb-receipt-row');
var out = {
  btn_text: btn.textContent,
  closed_on_load: !hasClass(drawer, 'tb-receipt-open'),
  head_text: drawer.querySelector('.tb-receipt-head').textContent,
  a_text: allText(rows[0]),
  a_fp_text: rows[0].querySelector('.tb-receipt-fp').textContent,
  a_fp_title: rows[0].querySelector('.tb-receipt-fp').title,
  b_badge_class: rows[1].querySelector('.tb-badge')
    ? rows[1].querySelector('.tb-badge').className : null,
  b_badge_text: rows[1].querySelector('.tb-badge')
    ? rows[1].querySelector('.tb-badge').textContent : null,
  drawer_text: allText(drawer)
};
btn._fire('click');
out.open_after_click = hasClass(drawer, 'tb-receipt-open');
btn._fire('click');
out.closed_after_second = !hasClass(drawer, 'tb-receipt-open');
process.stdout.write(JSON.stringify(out));
"""

    def test_drawer_rows_from_sample_block(self):
        out = _run_dom(self._SCRIPT)
        assert out["btn_text"] == "Receipt"
        assert out["closed_on_load"] is True
        assert out["head_text"] == "portfolio_book"
        # First line names the binding. The id, kind and full fingerprint
        # sit in the details disclosure.
        assert "Holdings · table" in out["a_text"]
        for piece in ("fig-a", "table", "holdings"):
            assert piece in out["a_text"]
        assert "Details" in out["a_text"]
        assert out["a_fp_text"] == "f" * 64
        assert out["a_fp_title"] == "f" * 64
        # The unverified row carries the existing badge tone, untouched.
        assert out["b_badge_class"] == "tb-badge tb-badge--unverified"
        assert out["b_badge_text"] == "unverified"
        text = out["drawer_text"]
        assert "rendered 2026-08-17 09:00" in text
        assert "git abc123d" in text
        # The locked language, per recorded status — never re-colored, and
        # never "the transform was verified".
        assert "fact_holdings: the sink satisfied its contract" in text
        assert "dim_sector: contract stale" in text
        assert "dim_misc: no contract" in text
        assert "the transform was verified" not in text
        assert "semantic contract: portfolio_model" in text
        assert "stated methodology aboard" in text
        # The button toggles the drawer open and closed again.
        assert out["open_after_click"] is True
        assert out["closed_after_second"] is True
        _assert_rows_avoid_honesty_words(out["drawer_text"])

    def test_value_row_reads_the_rendered_figure(self):
        out = _run_dom(_VALUE_ROW_SCRIPT)
        assert out["title"] == "Fair value · $285.9M"
        assert out["from"] == "Fair value, by Sector, Sector is Software"
        assert out["table_title"] == "Fair value by sector · table, 6 rows"
        assert out["table_from"] == "Fair value, by Sector"
        assert out["fp_text"] == "a" * 64
        assert out["fp_in_details"] is True
        assert "kpi-fv" in out["details_text"]
        assert "kpis" in out["details_text"]
        assert "kpi-fv" not in out["title"]
        _assert_rows_avoid_honesty_words(out["rows_text"])


_VALUE_ROW_SCRIPT = """
var FP = new Array(65).join('a');
var kpi = el('div', {
  id: 'kpi-fv', 'data-tb-figure': 'value', 'data-tb-binding': 'kpis',
  'data-tb-cell': 'fair_value'
});
var lab = el('span', { 'class': 'tb-kpi-label' }, kpi);
lab.textContent = 'Fair value';
var val = el('span', { 'class': 'tb-kpi-value' }, kpi);
val.textContent = '$285.9M';
var table = el('table', {
  id: 'tbl-sector', 'data-tb-figure': 'table',
  'data-tb-binding': 'fair_value_by_sector'
});
var tbody = el('tbody', {}, table);
for (var i = 0; i < 6; i++) el('tr', {}, tbody);
var rc = el('script', { id: 'tracebi-receipt', type: 'application/json' });
rc.textContent = JSON.stringify({
  report: 'book',
  figures: [
    { id: 'kpi-fv', kind: 'value', binding: 'kpis', cell: 'fair_value',
      fingerprint: FP,
      query: { measures: ['fair_value'], dimensions: ['dim_issuer.sector'],
               filters: { 'dim_issuer.sector': 'Software' } } },
    { id: 'tbl-sector', kind: 'table', binding: 'fair_value_by_sector',
      fingerprint: FP }
  ],
  bindings: {
    fair_value_by_sector: {
      query: { measures: ['fair_value'], dimensions: ['dim_issuer.sector'] }
    }
  }
});
loadRuntime();
var drawer = BODY.querySelector('.tb-receipt-drawer');
var rows = drawer.querySelectorAll('.tb-receipt-row');
var valueRow = null, tableRow = null, i, text;
for (i = 0; i < rows.length; i++) {
  text = allText(rows[i]);
  if (text.indexOf('kpi-fv') !== -1) valueRow = rows[i];
  if (text.indexOf('tbl-sector') !== -1) tableRow = rows[i];
}
var details = valueRow.querySelector('.tb-receipt-details');
var out = {
  title: valueRow.querySelector('.tb-receipt-title').textContent,
  from: valueRow.querySelector('.tb-receipt-from').textContent,
  table_title: tableRow.querySelector('.tb-receipt-title').textContent,
  table_from: tableRow.querySelector('.tb-receipt-from').textContent,
  fp_text: details.querySelector('.tb-receipt-fp').textContent,
  fp_in_details: details.querySelector('.tb-receipt-fp') !== null,
  details_text: allText(details),
  rows_text: allText(drawer)
};
process.stdout.write(JSON.stringify(out));
"""


def _assert_rows_avoid_honesty_words(text: str) -> None:
    """Provenance display. Only ``tracebi verify`` may say a number reproduces."""
    import re
    assert not re.search(r"\b(verified|reproduces|correct)\b", text, re.I)


def assert_built_receipt_rows(html: str) -> None:
    """A built page's value row reads ``<Label> · <formatted value>``.

    The formatted value is the one the build already wrote into the figure.
    The fingerprint stays in the details disclosure.
    """
    import re
    block = re.search(
        r'<script[^>]*id="tracebi-receipt"[^>]*>(.*?)</script>', html, re.S)
    assert block, "built page has no receipt"
    receipt = json.loads(block.group(1))
    fig = next(f for f in receipt["figures"] if f["id"] == "kpi-fv")
    tag = re.search(r'<[^>]*id="kpi-fv"[^>]*>', html)
    assert tag, "built page has no kpi-fv figure"
    cell = re.search(r'data-tb-cell="([^"]+)"', tag.group(0))
    assert cell, "kpi-fv names no column"
    value = re.search(
        r'id="kpi-fv"[^>]*>.*?<span class="tb-kpi-value">([^<]*)</span>',
        html, re.S)
    assert value and value.group(1).strip(), "kpi-fv has no rendered value"
    shown = value.group(1).strip()
    label = cell.group(1).replace("_", " ")
    label = label[0].upper() + label[1:]
    out = _run_dom(
        "var kpi = el('div', { id: 'kpi-fv', 'data-tb-figure': 'value',"
        " 'data-tb-binding': 'kpis', 'data-tb-cell': "
        + json.dumps(cell.group(1)) + " });\n"
        "var val = el('span', { 'class': 'tb-kpi-value' }, kpi);\n"
        "val.textContent = " + json.dumps(shown) + ";\n"
        "var rc = el('script', { id: 'tracebi-receipt',"
        " type: 'application/json' });\n"
        "rc.textContent = " + json.dumps(json.dumps(
            {"report": receipt.get("report"), "figures": [fig]})) + ";\n"
        "loadRuntime();\n"
        "var row = BODY.querySelector('.tb-receipt-row');\n"
        "var details = row.querySelector('.tb-receipt-details');\n"
        "process.stdout.write(JSON.stringify({\n"
        "  title: row.querySelector('.tb-receipt-title').textContent,\n"
        "  fp: details.querySelector('.tb-receipt-fp').textContent,\n"
        "  details_text: allText(details),\n"
        "  row_text: allText(row)\n"
        "}));\n"
    )
    assert out["title"] == f"{label} · {shown}"
    assert out["fp"] == fig["fingerprint"]
    assert fig["id"] in out["details_text"]
    assert fig["binding"] in out["details_text"]
    assert fig["fingerprint"] not in out["title"]
    _assert_rows_avoid_honesty_words(out["row_text"])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestBadgeAnchoring:
    """hydrateBadges pins a table's badge to the table, not the page.

    A <table> is not a reliable containing block for the absolute .tb-badge
    (tracebi.css positions it top/right against the nearest positioned
    ancestor — with a static table that is the page corner), so the runtime
    wraps the table in a positioned .tb-badge-anchor div and badges that.
    There was no DOM stub in this harness before; the stub below covers
    exactly the element surface hydrateBadges touches, nothing more.
    """

    # hydrateBadges is internal to the IIFE; exposed by the same
    # export-line rewrite _probe uses. The stub document must exist before
    # the runtime loads (it schedules hydration at load); the fixture is
    # registered after, so the load-time pass sees an empty page.
    _SCRIPT = """
var fs = require('fs');

function El(tag) {
  this.tagName = String(tag).toUpperCase();
  this.children = [];
  this.parentNode = null;
  this.className = '';
  this.textContent = '';
}
El.prototype.removeChild = function (c) {
  var i = this.children.indexOf(c);
  if (i !== -1) this.children.splice(i, 1);
  c.parentNode = null;
  return c;
};
El.prototype.appendChild = function (c) {
  if (c.parentNode) c.parentNode.removeChild(c);
  c.parentNode = this;
  this.children.push(c);
  return c;
};
El.prototype.insertBefore = function (c, ref) {
  if (c.parentNode) c.parentNode.removeChild(c);
  var i = ref ? this.children.indexOf(ref) : -1;
  if (i === -1) this.children.push(c); else this.children.splice(i, 0, c);
  c.parentNode = this;
  return c;
};
El.prototype.querySelector = function (sel) {  // class selectors only
  var cls = sel.slice(1);
  var stack = this.children.slice();
  while (stack.length) {
    var el = stack.shift();
    if ((' ' + el.className + ' ').indexOf(' ' + cls + ' ') !== -1) return el;
    stack = stack.concat(el.children);
  }
  return null;
};
Object.defineProperty(El.prototype, 'firstChild', {
  get: function () { return this.children[0] || null; }
});

var byId = {};
globalThis.document = {
  readyState: 'complete',
  querySelectorAll: function () { return []; },
  getElementById: function (id) { return byId[id] || null; },
  createElement: function (t) { return new El(t); },
  addEventListener: function () {}
};

var src = fs.readFileSync(process.argv[1], 'utf8');
var patched = src.replace('root.tracebi = {',
  'root.tracebi = { _badges: hydrateBadges,');
if (patched === src) throw new Error('export line not found');
new Function(patched)();

var page = new El('div');
var kpi = new El('div');
kpi.className = 'tb-kpi';
page.appendChild(kpi);
var section = new El('div');
page.appendChild(section);
var table = new El('table');
section.appendChild(table);
var cfg = new El('script');
cfg.textContent = JSON.stringify({
  badges: true,
  figures: [{ id: 'k1', provenance: 'verified' },
            { id: 't1', provenance: 'unverified', note: 'stale' }]
});
byId['tracebi-figures'] = cfg;
byId['k1'] = kpi;
byId['t1'] = table;

globalThis.tracebi._badges();
globalThis.tracebi._badges();  // second pass must not double-badge or re-wrap

var wrap = table.parentNode;
process.stdout.write(JSON.stringify({
  kpi_children: kpi.children.map(function (c) { return c.className; }),
  kpi_not_wrapped: kpi.parentNode === page,
  wrap_class: wrap ? wrap.className : null,
  wrap_in_section: wrap !== null && wrap.parentNode === section,
  section_children: section.children.length,
  wrap_children: wrap ? wrap.children.map(function (c) { return c.tagName; }) : null,
  table_badge_class: wrap && wrap.children[0] ? wrap.children[0].className : null,
  table_badge_title: wrap && wrap.children[0] ? wrap.children[0].title : null
}));
"""

    def test_table_badge_anchors_to_wrapper_kpi_badges_in_place(self):
        result = subprocess.run(
            ["node", "-e", self._SCRIPT, ASSET],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        # The KPI is its own anchor (position:relative in css): badged in
        # place, never wrapped — and exactly once across two passes.
        assert out["kpi_children"] == ["tb-badge tb-badge--verified"]
        assert out["kpi_not_wrapped"] is True
        # The table is wrapped in the positioned anchor div, which holds
        # the badge and then the table — badge pinned to the table corner.
        assert out["wrap_class"] == "tb-badge-anchor"
        assert out["wrap_in_section"] is True
        assert out["section_children"] == 1        # wrapper replaced the table
        assert out["wrap_children"] == ["SPAN", "TABLE"]
        assert out["table_badge_class"] == "tb-badge tb-badge--unverified"
        assert out["table_badge_title"] == "stale"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestSelectionControls:
    """An opted-in page does not subset-and-sum when the model is unreachable,
    and paints the query result — value figures included — when it answers."""

    _OFFLINE = """
dataBlock('b', 'region,mv\\nEast,10\\nWest,20\\n');
var cfg = el('script', { id: 'tracebi-selection', type: 'application/json' });
cfg.textContent = JSON.stringify({ report: 'demo', model: 'm', filters: {} });
var kpi = el('div', { 'data-tb-figure': 'value', 'data-tb-binding': 'b',
                      'data-tb-cell': 'mv', id: 'fig-kpi' });
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b' });
var f = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'b',
                       'data-tb-column': 'region' });
function Rejected() {}
Rejected.prototype.then = function () { return this; };
Rejected.prototype.catch = function (fn) { fn(); return this; };
globalThis.fetch = function () { return new Rejected(); };
loadRuntime();
var tbody = tbl.querySelector('tbody');
var before = tbody.children.length;
var kpiBefore = kpi.textContent;
f.value = 'East';
f._fire('change');
process.stdout.write(JSON.stringify({
  before: before,
  after: tbody.children.length,
  kpi_before: kpiBefore,
  kpi_after: kpi.textContent,
  title: f.title,
  value: f.value
}));
"""

    _ONLINE = """
dataBlock('b', 'region,mv\\nEast,10\\nWest,20\\n');
var cfg = el('script', { id: 'tracebi-selection', type: 'application/json' });
cfg.textContent = JSON.stringify({ report: 'demo', model: 'm', filters: {} });
var kpi = el('div', { 'data-tb-figure': 'value', 'data-tb-binding': 'b',
                      'data-tb-cell': 'mv', id: 'fig-kpi' });
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b' });
var f = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'b',
                       'data-tb-column': 'region' });
var rec = el('script', { id: 'tracebi-receipt', type: 'application/json' });
rec.textContent = JSON.stringify({ report: 'demo', figures: [] });
function Thenable(value) { this.value = value; }
Thenable.prototype.then = function (ok) {
  var ret = ok(this.value);
  return (ret && ret.then) ? ret : new Thenable(ret);
};
Thenable.prototype.catch = function () { return this; };
var PAYLOAD = {
  authored: false,
  filters: { region: 'East' },
  figures: [{
    id: 'fig-kpi', binding: 'b', kind: 'value', cell: 'mv',
    value: 99, formatted: '99', fingerprint: 'abc123',
    rows: [{ region: 'East', mv: 99 }]
  }],
  controls: [{ binding: 'b', column: 'region', included: ['East'], excluded: ['West'] }]
};
globalThis.fetch = function () {
  return new Thenable({
    ok: true,
    json: function () { return new Thenable(PAYLOAD); }
  });
};
loadRuntime();
f.value = 'East';
f._fire('change');
var tbody = tbl.querySelector('tbody');
var west = null;
f.children.forEach(function (o) { if (o.value === 'West') west = o; });
process.stdout.write(JSON.stringify({
  kpi: kpi.textContent,
  rows: tbody.children.length,
  west_disabled: !!(west && west.disabled),
  receipt: (document.getElementById('tb-selection-status') || {}).textContent || ''
}));
"""

    def test_offline_keeps_the_authored_view(self):
        out = _run_dom(self._OFFLINE)
        assert out["before"] == out["after"] == 2
        assert out["kpi_before"] == out["kpi_after"]
        assert out["title"] == "Further slices need the model"
        assert out["value"] == "All"  # the failed change is reverted

    def test_a_reachable_model_paints_value_figures(self):
        out = _run_dom(self._ONLINE)
        assert out["kpi"] == "99"
        assert out["rows"] == 1
        assert out["west_disabled"] is True
        assert "live" in out["receipt"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestTableOverridesAndChartPolish:
    """data-tb-labels / data-tb-formats override a table's derived header and
    number format; the house chart style restyles without touching data."""

    _SCRIPT = """
dataBlock('b', 'region,revenue\\nUS,1234.5\\nEU,-50\\n');
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b',
                        'data-tb-labels': 'region=Market; revenue=Net revenue, USD',
                        'data-tb-formats': 'revenue=currency' });
var cht = el('div', { 'data-tb-figure': 'chart', 'data-tb-binding': 'b',
                      'data-tb-x': 'region', 'data-tb-y': 'revenue' });
loadRuntime();
var opt = __charts[0].options[0];
process.stdout.write(JSON.stringify({
  head: tbl.querySelector('thead').children[0].children.map(function (th) {
    return th.textContent; }),
  rows: tbl.querySelector('tbody').children.map(function (tr) {
    return tr.children.map(function (td) { return td.textContent; }); }),
  data: opt.series[0].data
}));
"""

    def test_labels_and_formats_override_the_derived_defaults(self):
        out = _run_dom(self._SCRIPT)
        assert out["head"] == ["Market", "Net revenue, USD"]
        # the sign leads the symbol
        assert out["rows"] == [["US", "$1,234.50"], ["EU", "-$50.00"]]

    def test_chart_polish_restyles_but_never_changes_the_data(self):
        out = _run_dom(self._SCRIPT)
        assert out["data"] == [1234.5, -50]

    def test_a_declared_measure_format_beats_the_shape_guess(self):
        # The build writes the model's declared formats to tracebi-formats;
        # the runtime must use them, as the server render does.
        out = _run_dom("""
dataBlock('b', 'region,cost\\nUS,1000.4\\nEU,2000.75\\n');
var f = el('script', { id: 'tracebi-formats', type: 'application/json' });
f.textContent = JSON.stringify({ b: { cost: 'currency0' } });
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b' });
loadRuntime();
process.stdout.write(JSON.stringify({
  rows: tbl.querySelector('tbody').children.map(function (tr) {
    return tr.children.map(function (td) { return td.textContent; }); })
}));
""")
        assert out["rows"] == [["US", "$1,000"], ["EU", "$2,001"]]

    def test_an_unformatted_value_figure_takes_the_table_precedence(self):
        # Same order as a table column (and the server render): the author's
        # format, then the declared measure format, then the shape guess.
        out = _run_dom("""
dataBlock('k', 'gain,cost,label\\n1234.5,3001.15,West\\n');
var f = el('script', { id: 'tracebi-formats', type: 'application/json' });
f.textContent = JSON.stringify({ k: { cost: 'currency0' } });
var g = el('span', { 'data-tb-figure': 'value', 'data-tb-binding': 'k', 'data-tb-cell': 'gain' });
var c = el('span', { 'data-tb-figure': 'value', 'data-tb-binding': 'k', 'data-tb-cell': 'cost' });
var d = el('span', { 'data-tb-figure': 'value', 'data-tb-binding': 'k', 'data-tb-cell': 'cost',
                     'data-tb-format': 'decimal' });
var t = el('span', { 'data-tb-figure': 'value', 'data-tb-binding': 'k', 'data-tb-cell': 'label' });
loadRuntime();
process.stdout.write(JSON.stringify([g.textContent, c.textContent,
                                     d.textContent, t.textContent]));
""")
        assert out == ["1,234.50", "$3,001", "3,001.15", "West"]

    def test_a_narrow_pie_names_slices_in_a_legend_and_keeps_its_data(self):
        out = _run_dom("""
dataBlock('b', 'asset_class,aum\\nfixed income,2726364\\nmoney market,1688904\\n');
var pie = el('div', { 'data-tb-figure': 'chart', 'data-tb-binding': 'b',
                      'data-tb-type': 'pie', 'data-tb-x': 'asset_class',
                      'data-tb-y': 'aum' });
pie.clientWidth = 420;
loadRuntime();
var opt = __charts[0].options[0];
process.stdout.write(JSON.stringify({
  label: opt.series[0].label.show, legend: !!(opt.legend && opt.legend.show),
  data: opt.series[0].data.map(function (d) { return d.value; })
}));
""")
        assert out == {"label": False, "legend": True, "data": [2726364, 1688904]}

    _TOTALS = """
dataBlock('b', 'region,revenue,margin_pct\\nUS,1000,0.2\\nEU,3000,0.4\\n');
dataBlock('b_totals', 'revenue,margin_pct\\n4000,0.35\\n');
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b',
                        'data-tb-totals': 'b_totals' });
var sel = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'b',
                         'data-tb-column': 'region' });
loadRuntime();
var foot = tbl.querySelector('tfoot');
var cells = foot.children[0].children.map(function (td) { return td.textContent; });
var before = foot.hidden;
sel.value = 'US';
sel._fire('change');
process.stdout.write(JSON.stringify({ cells: cells, hidden: before,
                                      filtered: tbl.querySelector('tfoot').hidden }));
"""

    def test_totals_row_shows_the_models_one_row_result(self):
        out = _run_dom(self._TOTALS)
        # 0.35 is the model's ratio of totals, not the 0.30 mean of the rows.
        assert out["cells"] == ["Total", "4,000", "35.0%"]
        assert out["hidden"] is False

    def test_totals_row_steps_aside_while_a_filter_is_on(self):
        # A grand total under a filtered view would not match the rows shown.
        assert _run_dom(self._TOTALS)["filtered"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestSortBarsDirection:
    """Table and KPI polish borrowed from data-grid libraries — each one
    reorders or decorates stamped values and never computes a number."""

    _SORT = """
dataBlock('b', 'region,revenue\\nUS,100\\nEU,300\\nAPAC,\\nLATAM,200\\n');
dataBlock('t', 'revenue\\n600\\n');
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b',
                        'data-tb-sort': '', 'data-tb-totals': 't' });
loadRuntime();
function order() {
  return tbl.querySelector('tbody').children.map(function (tr) {
    return tr.children[0].textContent; });
}
var th = tbl.querySelectorAll('thead th')[1], btn = th.querySelector('.tb-sort');
var out = { query: order() };
btn._fire('click'); out.asc = order(); out.ariaAsc = th.getAttribute('aria-sort');
btn._fire('click'); out.desc = order();
btn._fire('click'); out.back = order(); out.ariaBack = th.getAttribute('aria-sort');
out.totalsHidden = tbl.querySelector('tfoot').hidden;
process.stdout.write(JSON.stringify(out));
"""

    def test_header_click_cycles_asc_desc_and_back_to_the_query_order(self):
        out = _run_dom(self._SORT)
        assert out["query"] == ["US", "EU", "APAC", "LATAM"]
        assert out["asc"] == ["US", "LATAM", "EU", "APAC"]   # blanks last
        assert out["desc"] == ["EU", "LATAM", "US", "APAC"]  # blanks last both ways
        assert out["back"] == out["query"]
        assert out["ariaAsc"] == "ascending" and out["ariaBack"] == "none"

    def test_sorting_keeps_the_totals_row(self):
        # All rows are still shown, so the model's total still describes them.
        assert _run_dom(self._SORT)["totalsHidden"] is False

    def test_bars_scale_from_zero_and_center_zero_when_negative(self):
        out = _run_dom("""
dataBlock('b', 'name,pos,gain\\nA,50,-25\\nB,100,50\\n');
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b',
                        'data-tb-bars': 'pos, gain, name' });
loadRuntime();
var rows = tbl.querySelector('tbody').children;
function bar(r, c) {
  var b = rows[r].children[c].querySelector('.tb-bar');
  return b ? [b.style.left, b.style.width, b.className] : null;
}
process.stdout.write(JSON.stringify({
  pos: [bar(0, 1), bar(1, 1)], gain: [bar(0, 2), bar(1, 2)], name: bar(0, 0),
  text: rows[0].children[2].textContent }));
""")
        assert out["pos"] == [["0%", "50%", "tb-bar"], ["0%", "100%", "tb-bar"]]
        # a column with negatives puts zero in the middle; each side is honest
        assert out["gain"] == [["25%", "25%", "tb-bar tb-bar--neg"],
                               ["50%", "50%", "tb-bar"]]
        assert out["name"] is None             # text columns never get a bar
        assert out["text"] == "-25"            # the number itself is untouched

    def test_direction_marks_sign_and_meaning_without_changing_text(self):
        out = _run_dom("""
dataBlock('k', 'gain,cost_change,flat\\n-6.3,-2,0\\n');
function v(cell, dir) {
  return el('span', { 'data-tb-figure': 'value', 'data-tb-binding': 'k',
                      'data-tb-cell': cell, 'data-tb-direction': dir });
}
var a = v('gain', 'up-good'), b = v('cost_change', 'down-good'), c = v('flat', 'up-good');
loadRuntime();
process.stdout.write(JSON.stringify([[a.className, a.textContent],
                                     [b.className, b.textContent],
                                     [c.className, c.textContent]]));
""")
        assert out == [["tb-down tb-bad", "-6.30"],
                       ["tb-down tb-good", "-2"],
                       ["tb-flat", "0"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestTableWindow:
    """Past 500 rows the runtime paints a window; 500 stays the full tbody.

    The DOM stub has no layout, so the window uses the fallback row height.
    A real browser covers scroll, print, and bar pixels in test_table_window.
    """

    _SCRIPT = """
function csvRows(n) {
  var s = 'k,v,g\\n';
  for (var i = 0; i < n; i++) s += 'k' + i + ',' + (i + 1) + ',' + (i === 7 ? 'z' : 'a') + '\\n';
  return s;
}
dataBlock('wide', csvRows(501));
dataBlock('cap', csvRows(500));
var wide = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'wide',
                         'data-tb-sort': '', 'data-tb-bars': 'v' });
var cap = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'cap',
                        'data-tb-sort': '' });
var search = el('input', { 'data-tb-search': '', 'data-tb-binding': 'wide' });
var filt = el('select', { 'data-tb-filter': '', 'data-tb-binding': 'wide',
                          'data-tb-column': 'g' });
loadRuntime();
function dataRows(tbl) {
  return tbl.querySelector('tbody').children.filter(function (tr) {
    return tr.className.indexOf('tb-window-pad') === -1;
  });
}
function barAt(tbl) {
  var row = dataRows(tbl)[0];
  var bar = row.children[1].querySelector('.tb-bar');
  return bar ? bar.style.width : null;
}
var box = wide.parentNode;
var out = {
  wide: dataRows(wide).length,
  pad: wide.querySelector('tbody').children.length - dataRows(wide).length,
  windowClass: wide.className.indexOf('tb-window') !== -1,
  cap: dataRows(cap).length,
  capWindow: cap.className.indexOf('tb-window') !== -1,
  bar: barAt(wide),
  barGlobal: barAt(wide) === (100 / 501) + '%'
};
box.scrollTop = 999999;
box._fire('scroll');
var tailRows = dataRows(wide);
out.tail = tailRows[tailRows.length - 1].children[0].textContent;
out.tailCount = tailRows.length;
box.scrollTop = 0;
box._fire('scroll');
out.barAfter = barAt(wide);
var th = wide.querySelectorAll('thead th')[1];
th.querySelector('.tb-sort')._fire('click');
th.querySelector('.tb-sort')._fire('click');
out.sorted = dataRows(wide)[0].children[0].textContent;
search.value = 'k500';
search._fire('input');
out.search = dataRows(wide).map(function (tr) { return tr.children[0].textContent; });
search.value = '';
search._fire('input');
filt.value = 'z';
filt._fire('change');
out.filtered = dataRows(wide).map(function (tr) { return tr.children[0].textContent; });
filt.value = 'All';
filt._fire('change');
out.restored = dataRows(wide).length;
process.stdout.write(JSON.stringify(out));
"""

    def test_above_500_windows_and_500_renders_every_row(self):
        out = _run_dom(self._SCRIPT)
        assert out["wide"] < 80
        assert out["pad"] >= 1
        assert out["windowClass"] is True
        assert out["cap"] == 500
        assert out["capWindow"] is False
        # Scale is the column max (501), not the max inside the window.
        assert out["barGlobal"] is True
        assert out["barAfter"] == out["bar"]
        assert out["tail"] == "k500"
        assert out["tailCount"] < 80
        assert out["sorted"] == "k500"          # descending: the true max
        assert out["search"] == ["k500"]
        assert out["filtered"] == ["k7"]
        assert out["restored"] < 80

    _SELECTION = """
dataBlock('b', 'region,mv\\nEast,1\\n');
var cfg = el('script', { id: 'tracebi-selection', type: 'application/json' });
cfg.textContent = JSON.stringify({ report: 'demo', model: 'm', filters: {} });
var tbl = el('table', { 'data-tb-figure': 'table', 'data-tb-binding': 'b' });
var rows = [];
for (var i = 0; i < 600; i++) rows.push({ region: 'R' + i, mv: String(i + 1) });
rows[599].region = 'TAIL';
var payload = { authored: false, filters: {}, figures: [
  { binding: 'b', kind: 'table', rows: rows }
], controls: [] };
function Thenable(value) { this.value = value; }
Thenable.prototype.then = function (ok) {
  var ret = ok(this.value);
  return (ret && ret.then) ? ret : new Thenable(ret);
};
Thenable.prototype.catch = function () { return this; };
globalThis.fetch = function () {
  return new Thenable({ ok: true, json: function () { return new Thenable(payload); } });
};
loadRuntime();
function dataRows() {
  return tbl.querySelector('tbody').children.filter(function (tr) {
    return tr.className.indexOf('tb-window-pad') === -1;
  });
}
var before = dataRows().length;
var box = tbl.parentNode;
box.scrollTop = 999999;
box._fire('scroll');
var tail = dataRows();
process.stdout.write(JSON.stringify({
  before: before,
  parent: box.className,
  tail: tail[tail.length - 1].children[0].textContent,
  count: tail.length
}));
"""

    def test_a_selection_of_more_than_500_rows_is_windowed(self):
        out = _run_dom(self._SELECTION)
        assert out["before"] < 80
        assert out["parent"] == "tb-scroll"
        assert out["tail"] == "TAIL"
        assert out["count"] < 80

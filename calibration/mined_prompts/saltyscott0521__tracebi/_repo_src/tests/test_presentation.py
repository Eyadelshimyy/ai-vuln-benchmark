"""
The presentation stack (architecture v2 §2.4).

The injection order IS the override chain — tracebi.css, then the
project's _theme.css, then the report's style.css, with the author's
script running after the runtime. Badges carry provenance decided at
build time from what was embedded; a config flag can omit them but the
manifest never changes. Presentation must never change a number.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.reports.template_package import TemplatePackage


@pytest.fixture()
def stack_model():
    df = pd.DataFrame({"region": ["NE", "SE"], "revenue": [100.0, 250.0]})
    m = DataModel("stack_model")
    m.add_connector(MemoryConnector("st_mem", tables={"t": df}))
    m.add_table("t", connector="st_mem", source="t")
    m.add_dimension("dim_r", table_name="t", key_col="region",
                    attributes=["region"])
    m.add_fact("f", table_name="t", measures=["revenue"], foreign_keys={})
    m.add_measure("total", column="revenue", agg="sum")
    m.connect()
    return m


def _pkg(tmp_path, with_report_py=False):
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "pkg",
        "data": {
            "kpi": {"model": "stack_model",
                    "query": {"fact": "f", "measures": ["total"]}},
            "by_region": {"model": "stack_model",
                          "query": {"fact": "f",
                                    "measures": {"revenue": "sum"},
                                    "dimensions": ["dim_r.region"]}},
        },
    }))
    (pkg / "template.html").write_text(
        "<html><head><title>x</title></head><body>"
        '<div data-tb-figure="value" data-tb-binding="kpi" '
        'data-tb-cell="total" id="fig-kpi"></div>'
        '<table data-tb-figure="table" data-tb-binding="by_region" '
        'id="fig-tbl"></table>'
        + ('<table data-tb-figure="table" data-tb-binding="ranked" '
           'id="fig-rank"></table>' if with_report_py else "")
        + "</body></html>"
    )
    (pkg / "style.css").write_text(":root { --tb-accent: #ff0000; }")
    if with_report_py:
        (pkg / "report.py").write_text(
            "def build(frames):\n"
            "    return {'ranked': frames['by_region'].head(1)}\n")
    return pkg


class TestStackOrder:
    def test_later_wins_layer_order(self, stack_model, tmp_path, monkeypatch):
        # A project _theme.css between the shipped sheet and the report's own.
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "_theme.css").write_text(
            ":root { --tb-accent: #00ff00; }")
        monkeypatch.chdir(tmp_path)

        out = tmp_path / "out.html"
        TemplatePackage(str(_pkg(tmp_path))).render(
            {"stack_model": stack_model}, str(out))
        html = out.read_text(encoding="utf-8")

        base = html.index("--tb-font")               # shipped tracebi.css
        project = html.index("--tb-accent: #00ff00")  # project layer
        report = html.index("--tb-accent: #ff0000")   # report layer
        assert base < project < report, (
            "the stack must inject shipped → project → report, so later wins"
        )
        # The runtime precedes the data blocks; the author script would come
        # after both (order is the contract that lets authors patch charts).
        assert html.index("window.tracebi") < html.index("tracebi-data-kpi")

    def test_badges_config_carries_provenance(self, stack_model, tmp_path):
        out = tmp_path / "out.html"
        TemplatePackage(str(_pkg(tmp_path, with_report_py=True))).render(
            {"stack_model": stack_model}, str(out))
        html = out.read_text(encoding="utf-8")
        start = html.index('id="tracebi-figures"')
        payload = html[html.index(">", start) + 1:html.index("</script>", start)]
        cfg = json.loads(payload)
        # On-page badges are OFF by default (a mark on every figure is noise),
        # but the provenance is still recorded in the config — the receipt
        # drawer and any opt-in badge read it.
        assert cfg["badges"] is False
        prov = {f["id"]: f["provenance"] for f in cfg["figures"]}
        assert prov["fig-kpi"] == "verified"
        assert prov["fig-tbl"] == "verified"
        assert prov["fig-rank"] == "derived"     # report.py output — never green

    def test_badges_opt_in_renders_them(self, stack_model, tmp_path):
        out = tmp_path / "out.html"
        TemplatePackage(str(_pkg(tmp_path))).render(
            {"stack_model": stack_model}, str(out), badges=True)
        html = out.read_text(encoding="utf-8")
        start = html.index('id="tracebi-figures"')
        payload = html[html.index(">", start) + 1:html.index("</script>", start)]
        assert json.loads(payload)["badges"] is True

    def test_no_badges_omits_rendering_but_not_the_manifest(
            self, stack_model, tmp_path):
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(_pkg(tmp_path))).render(
            {"stack_model": stack_model}, str(out), badges=False).to_dict()
        html = out.read_text(encoding="utf-8")
        start = html.index('id="tracebi-figures"')
        payload = html[html.index(">", start) + 1:html.index("</script>", start)]
        assert json.loads(payload)["badges"] is False
        # The receipt is unaffected: the figure claims are all still there.
        assert {f["id"] for f in manifest["figures"]} == {"fig-kpi", "fig-tbl"}

    def test_self_containment_and_csp_hold(self, stack_model, tmp_path):
        out = tmp_path / "out.html"
        TemplatePackage(str(_pkg(tmp_path))).render(
            {"stack_model": stack_model}, str(out))
        html = out.read_text(encoding="utf-8")
        # Exactly one CSP meta: the renderer and the stack both know how to
        # inject it, and the stack yields when one is already present
        # (round-2 review, finding 11 — a duplicated policy is noise).
        assert html.count("Content-Security-Policy") == 1
        assert "http://" not in html.replace("http://www.w3.org", "")
        assert 'src="https://' not in html and "@import" not in html
        # The typeface travels inside the file: inlined (the CSP allows font-src
        # data: only), never fetched, and the first face the stylesheet names.
        assert 'font-family:"Hanken Grotesk"' in html
        assert 'url("data:font/woff2;base64,' in html
        assert not any(f"url({q}http" in html for q in ("", "'", '"'))
        assert '--tb-font: "Hanken Grotesk"' in html


# ── the receipt block ───────────────────────────────────────────────────────
# The build emits one tracebi-receipt JSON block: the drawer's provenance
# feed. It duplicates facts the manifest records — the MANIFEST remains the
# receipt of record — so every entry is checked against the manifest it
# mirrors.

#: The block's full opening tag. The runtime script also names the bare id
#: string (it reads the block), so tests anchor on the element itself.
RECEIPT_OPEN = '<script id="tracebi-receipt" type="application/json">'


def _receipt(html):
    start = html.index(RECEIPT_OPEN)
    payload = html[start + len(RECEIPT_OPEN):html.index("</script>", start)]
    return json.loads(payload)


class TestReceiptBlock:
    def test_records_manifest_facts_and_figure_fingerprints(
            self, stack_model, tmp_path):
        out = tmp_path / "out.html"
        manifest = TemplatePackage(
            str(_pkg(tmp_path, with_report_py=True))).render(
            {"stack_model": stack_model}, str(out)).to_dict()
        receipt = _receipt(out.read_text(encoding="utf-8"))
        # The header duplicates the manifest, not a second computation.
        assert receipt["report"] == manifest["report_name"]
        assert receipt["rendered_at"] == manifest["rendered_at"]
        assert receipt["git_sha"] == manifest["git_sha"]
        assert receipt["semantic_contract_models"] == ["stack_model"]
        assert receipt["methodology"] is False
        # Every figure's fingerprint equals its binding's embedded_sha256.
        sha = {r["name"]: r["embedded_sha256"]
               for r in manifest["embedded_data"]}
        assert [f["id"] for f in receipt["figures"]] == [
            "fig-kpi", "fig-tbl", "fig-rank"]
        by_id = {f["id"]: f for f in receipt["figures"]}
        assert by_id["fig-kpi"]["fingerprint"] == sha["kpi"]
        assert by_id["fig-tbl"]["fingerprint"] == sha["by_region"]
        assert by_id["fig-rank"]["fingerprint"] == sha["ranked"]
        # Model names come from the bindings' DataRefs; a report.py output
        # has no DataRef, so its figure records no model.
        assert by_id["fig-kpi"]["model"] == "stack_model"
        assert "model" not in by_id["fig-rank"]

    def test_byte_stable_across_renders_modulo_timestamp(
            self, stack_model, tmp_path):
        pkg = TemplatePackage(str(_pkg(tmp_path)))
        receipts = []
        for name in ("a.html", "b.html"):
            out = tmp_path / name
            pkg.render({"stack_model": stack_model}, str(out))
            receipts.append(_receipt(out.read_text(encoding="utf-8")))
        r1, r2 = receipts
        assert r1.pop("rendered_at") and r2.pop("rendered_at")
        # dicts preserve insertion order, so equal dumps means the same
        # bytes in the same order — byte-stable but for the render stamp.
        assert json.dumps(r1) == json.dumps(r2)

    def test_snapshot_carries_no_receipt_block(self, stack_model, tmp_path):
        # A draft page carries no receipt affordance — nothing for the
        # drawer to make look final (it also carries no manifest at all).
        pkg = TemplatePackage(str(_pkg(tmp_path)))
        snap = tmp_path / "snap.html"
        pkg.snapshot({"stack_model": stack_model}, str(snap))
        assert RECEIPT_OPEN not in snap.read_text(encoding="utf-8")


class TestServerSideRender:
    """Value figures are filled at build (SSR): a reader with JavaScript off
    still sees the numbers, and the runtime hydrates the identical bytes."""

    @staticmethod
    def _model():
        m = DataModel("ssr_model").add_connector(MemoryConnector("m", tables={
            "orders": pd.DataFrame({"order_id": [1, 2, 3],
                                    "region": ["NE", "SE", "MW"],
                                    "revenue": [100.0, 200.0, 300.0]})}))
        m.add_table("orders", connector="m", source="orders")
        m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                   foreign_keys={})
        m.add_measure("revenue", column="revenue", agg="sum")
        m.connect()
        return m

    def _render(self, tmp_path):
        from tracebi.reports.compile_spec import compile_spec
        from tracebi.spec import ReportSpec
        spec = ReportSpec.from_dict({"name": "SSR", "sections": [
            {"type": "metrics", "title": "K",
             "data": {"model": "ssr_model",
                      "query": {"fact": "fact_orders", "measures": ["revenue"]}},
             "metrics": [{"label": "Revenue", "value": "revenue",
                          "format": "currency0"}]}]})
        compiled = compile_spec(spec)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        for fn, content in compiled.files.items():
            (pkg / fn).write_text(content, encoding="utf-8")
        out = tmp_path / "o.html"
        manifest = TemplatePackage(str(pkg)).render(
            {"ssr_model": self._model()}, str(out))
        return out.read_text(encoding="utf-8"), manifest

    def test_value_is_in_the_raw_html_without_js(self, tmp_path):
        import re
        html, _ = self._render(tmp_path)
        assert "$600" in html                          # the number, no JS needed
        m = re.search(r'<span class="tb-kpi-value">([^<]*)</span>', html)
        assert m and m.group(1) == "$600"              # the KPI span is filled

    def test_ssr_leaves_the_embedded_data_intact(self, tmp_path):
        from tracebi.verify import verify_file, FILE_INTACT
        html, manifest = self._render(tmp_path)
        # SSR fills presentation, never the <script> data blocks; verify --file
        # rehashes those, so an INTACT verdict proves no fingerprint moved.
        assert verify_file(html, manifest.to_dict())["verdict"] == FILE_INTACT

    def _render_table(self, tmp_path):
        from tracebi.reports.compile_spec import compile_spec
        from tracebi.spec import ReportSpec
        model = DataModel("ssr_t").add_connector(MemoryConnector("m", tables={
            "orders": pd.DataFrame({"order_id": [1, 2, 3, 4],
                                    "customer_id": [1, 1, 2, 2],
                                    "revenue": [100.0, 200.5, 50.0, 150.0]}),
            "customers": pd.DataFrame({"customer_id": [1, 2],
                                       "region": ["NE", "SE"]})}))
        model.add_table("orders", connector="m", source="orders")
        model.add_table("customers", connector="m", source="customers")
        model.add_dimension("dim_customer", table_name="customers",
                            key_col="customer_id", attributes=["region"])
        model.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                       foreign_keys={"dim_customer": "customer_id"})
        model.add_measure("revenue", column="revenue", agg="sum")
        model.connect()
        spec = ReportSpec.from_dict({"name": "T", "sections": [
            {"type": "table", "title": "By region",
             "data": {"model": "ssr_t", "query": {"fact": "fact_orders",
                      "measures": ["revenue"], "dimensions": ["dim_customer.region"]}}}]})
        compiled = compile_spec(spec)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        for fn, content in compiled.files.items():
            (pkg / fn).write_text(content, encoding="utf-8")
        out = tmp_path / "o.html"
        manifest = TemplatePackage(str(pkg)).render({"ssr_t": model}, str(out))
        return out.read_text(encoding="utf-8"), manifest

    def test_table_rows_are_in_the_raw_html_without_js(self, tmp_path):
        import re
        html, _ = self._render_table(tmp_path)
        m = re.search(r"<tbody data-tb-hydrate>(.*?)</tbody>", html, re.S)
        assert m, "the table figure must carry a server-rendered tbody"
        body = m.group(1)
        # both dimension groups + their column-derived decimal totals, no JS
        assert "<td>NE</td>" in body and '<td class="tb-num">300.50</td>' in body
        assert "<td>SE</td>" in body and '<td class="tb-num">200.00</td>' in body
        assert body.count("<tr>") == 2                 # one row per group, no dup
        assert '<th class="tb-num">Revenue</th>' in html   # humanised numeric header

    def test_table_ssr_leaves_the_embedded_data_intact(self, tmp_path):
        from tracebi.verify import verify_file, FILE_INTACT
        html, manifest = self._render_table(tmp_path)
        assert verify_file(html, manifest.to_dict())["verdict"] == FILE_INTACT

    def _render_chart(self, tmp_path):
        from tracebi.reports.compile_spec import compile_spec
        from tracebi.spec import ReportSpec
        model = DataModel("ssr_c").add_connector(MemoryConnector("m", tables={
            "orders": pd.DataFrame({"order_id": [1, 2, 3, 4],
                                    "customer_id": [1, 1, 2, 2],
                                    "revenue": [100.0, 200.0, 50.0, 150.0]}),
            "customers": pd.DataFrame({"customer_id": [1, 2],
                                       "region": ["NE", "SE"]})}))
        model.add_table("orders", connector="m", source="orders")
        model.add_table("customers", connector="m", source="customers")
        model.add_dimension("dim_customer", table_name="customers",
                            key_col="customer_id", attributes=["region"])
        model.add_fact("fact_orders", table_name="orders", measures=["revenue"],
                       foreign_keys={"dim_customer": "customer_id"})
        model.add_measure("revenue", column="revenue", agg="sum")
        model.connect()
        spec = ReportSpec.from_dict({"name": "C", "sections": [
            {"type": "chart", "title": "By region", "chart_type": "bar",
             "x": "dim_customer.region", "y": "revenue",
             "data": {"model": "ssr_c", "query": {"fact": "fact_orders",
                      "measures": ["revenue"], "dimensions": ["dim_customer.region"]}}}]})
        compiled = compile_spec(spec)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        for fn, content in compiled.files.items():
            (pkg / fn).write_text(content, encoding="utf-8")
        out = tmp_path / "o.html"
        manifest = TemplatePackage(str(pkg)).render({"ssr_c": model}, str(out))
        return out.read_text(encoding="utf-8"), manifest

    def test_chart_has_a_static_svg_fallback_without_js(self, tmp_path):
        import re
        html, _ = self._render_chart(tmp_path)
        m = re.search(r'<div[^>]*data-tb-figure="chart"[^>]*>(.*?)</div>', html, re.S)
        assert m, "the chart figure must carry a server-rendered fallback"
        inner = m.group(1)
        # a real picture of the chart (tagged so the runtime removes it), no JS
        assert "tb-chart-fallback" in inner and "<svg" in inner
        assert "<rect" in inner                        # bars are drawn
        # it reads like the live chart: no raw column name as an axis title,
        # and each mark names its palette slot so the page theme colours it
        assert "tb-axis-label" not in inner
        assert 'data-tb-slot="1"' in inner

    def test_percent_bar_fallback_formats_axis_and_tooltip(self, tmp_path):
        import re
        model = DataModel("pct").add_connector(MemoryConnector("m", tables={
            "orders": pd.DataFrame({"order_id": [1, 2], "customer_id": [1, 2],
                                    "share": [0.26, 0.10]}),
            "customers": pd.DataFrame({"customer_id": [1, 2],
                                       "region": ["NE", "SE"]}),
        }))
        model.add_table("orders", connector="m", source="orders")
        model.add_table("customers", connector="m", source="customers")
        model.add_dimension("dim_customer", table_name="customers",
                            key_col="customer_id", attributes=["region"])
        model.add_fact("fact_orders", table_name="orders", measures=["share"],
                       foreign_keys={"dim_customer": "customer_id"})
        model.add_measure("share", column="share", agg="sum", format="percent")
        model.connect()
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(json.dumps({
            "name": "P",
            "data": {"shares": {"model": "pct", "query": {
                "fact": "fact_orders", "measures": ["share"],
                "dimensions": ["dim_customer.region"],
            }}},
        }), encoding="utf-8")
        (pkg / "template.html").write_text(
            "<!doctype html><html><head><title>t</title></head><body>"
            '<div data-tb-figure="chart" data-tb-binding="shares" '
            'data-tb-type="bar" data-tb-x="dim_customer.region" '
            'data-tb-y="share" data-tb-value-format="percent" '
            'id="fig-share"></div></body></html>',
            encoding="utf-8",
        )
        out = tmp_path / "p.html"
        TemplatePackage(str(pkg)).render({"pct": model}, str(out))
        html = out.read_text(encoding="utf-8")
        svg = re.search(r'class="tb-chart-fallback.*?</svg>', html, re.S)
        assert svg, "the no-JS fallback must be in the page"
        picture = svg.group(0)
        assert "26.0%" in picture
        assert "0.26" not in picture
        ticks = re.findall(r'class="tb-tick"[^>]*>([^<]*)', picture)
        assert ticks and all(t.endswith("%") for t in ticks)

    def test_chart_ssr_leaves_the_embedded_data_intact(self, tmp_path):
        from tracebi.verify import verify_file, FILE_INTACT
        html, manifest = self._render_chart(tmp_path)
        assert verify_file(html, manifest.to_dict())["verdict"] == FILE_INTACT

    def test_empty_binding_table_says_no_data(self, tmp_path):
        import re
        from tracebi.reports.compile_spec import compile_spec
        from tracebi.spec import ReportSpec
        model = DataModel("ssr_e").add_connector(MemoryConnector("m", tables={
            "t": pd.DataFrame({"region": ["NE", "SE"], "status": ["open", "open"],
                               "rev": [100.0, 200.0]})}))
        model.add_table("t", connector="m", source="t")
        model.add_dimension("dim_r", table_name="t", key_col="region",
                            attributes=["region"])
        model.add_fact("f", table_name="t", measures=["rev"], foreign_keys={})
        model.add_measure("rev", column="rev", agg="sum")
        model.connect()
        spec = ReportSpec.from_dict({"name": "E", "sections": [
            {"type": "table", "title": "none", "data": {"model": "ssr_e", "query": {
                "fact": "f", "measures": ["rev"], "dimensions": ["dim_r.region"],
                "filters": {"status": "shipped"}}}}]})   # no 'shipped' rows
        compiled = compile_spec(spec)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        for fn, content in compiled.files.items():
            (pkg / fn).write_text(content, encoding="utf-8")
        out = tmp_path / "o.html"
        TemplatePackage(str(pkg)).render({"ssr_e": model}, str(out))
        html = out.read_text(encoding="utf-8")
        m = re.search(r"<tbody data-tb-hydrate>(.*?)</tbody>", html, re.S)
        assert m and 'class="tb-empty"' in m.group(1) and "no data" in m.group(1)


class TestPackageAssets:
    """Fonts and images under a package's assets/ ride inside the one file."""

    def test_css_and_img_references_are_inlined_as_data_uris(
            self, stack_model, tmp_path):
        pkg = _pkg(tmp_path)
        (pkg / "assets").mkdir()
        (pkg / "assets" / "mark.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        (pkg / "assets" / "face.woff2").write_bytes(b"wOF2fake")
        (pkg / "style.css").write_text(
            "@font-face { font-family: X; src: url(assets/face.woff2); }\n"
            ".hero { background: url('assets/mark.svg'); }")
        tpl = (pkg / "template.html").read_text().replace(
            "<body>", '<body><img src="assets/mark.svg" alt="">')
        (pkg / "template.html").write_text(tpl)
        out = tmp_path / "o.html"
        TemplatePackage(str(pkg)).render({"stack_model": stack_model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert 'url("data:font/woff2;base64,' in html
        assert 'url("data:image/svg+xml;base64,' in html
        assert '<img src="data:image/svg+xml;base64,' in html
        assert "url(assets/" not in html and "url('assets/" not in html
        assert 'src="assets/' not in html

    @pytest.mark.parametrize("ref, err, fragment", [
        ("assets/missing.png", FileNotFoundError, "does not exist"),
        ("assets/../report.json", ValueError, "leaves the package"),
        ("assets/notes.txt", ValueError, "has type '.txt'"),
    ])
    def test_a_bad_reference_fails_the_load(self, tmp_path, ref, err, fragment):
        pkg = _pkg(tmp_path)
        (pkg / "assets").mkdir()
        (pkg / "assets" / "notes.txt").write_text("x")
        (pkg / "style.css").write_text(f".a {{ background: url({ref}); }}")
        with pytest.raises(err, match=fragment):
            TemplatePackage(str(pkg))


def test_every_report_names_what_it_inlines(stack_model, tmp_path):
    out = tmp_path / "o.html"
    TemplatePackage(str(_pkg(tmp_path))).render(
        {"stack_model": stack_model}, str(out))
    head = out.read_text(encoding="utf-8").split("</head>")[0]
    assert "Built with TraceBi" in head
    assert "tracebi.css  TraceBi design system (MIT)" in head
    assert "<!-- style.css: this report's own styles -->" in head


class TestDeclaredFormats:
    """A format the model declares on a measure reaches the table, including
    a ratio's inputs, which the query returns as their own columns."""

    def test_ratio_input_column_uses_its_declared_format(self, tmp_path):
        import re

        df = pd.DataFrame({"region": ["NE", "SE"], "gain": [10.5, 20.25],
                           "cost": [1000.4, 2000.75]})
        m = DataModel("fmt_model")
        m.add_connector(MemoryConnector("fm", tables={"t": df}))
        m.add_table("t", connector="fm", source="t")
        m.add_dimension("dim_r", table_name="t", key_col="region", attributes=["region"])
        m.add_fact("f", table_name="t", measures=["gain", "cost"], foreign_keys={})
        m.add_measure("gain", column="gain", agg="sum", format="currency0")
        m.add_measure("cost", column="cost", agg="sum", format="currency0")
        m.add_measure("gain_pct", ratio=("gain", "cost"), format="percent")
        m.connect()

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(json.dumps({"name": "p", "data": {"t": {
            "model": "fmt_model",
            "query": {"fact": "f", "measures": ["gain_pct"],
                      "dimensions": ["dim_r.region"]}}}}))
        (pkg / "template.html").write_text(
            '<html><head><title>p</title></head><body><table data-tb-figure="table" data-tb-binding="t" '
            'id="tbl"></table></body></html>')
        out = tmp_path / "o.html"
        TemplatePackage(str(pkg)).render({"fmt_model": m}, str(out))
        html = out.read_text(encoding="utf-8")

        # The server-rendered rows use the declared format, not the shape guess.
        assert "$1,000" in html and "1,000.40" not in html
        # The browser gets the same map, so hydration writes the same text.
        block = re.search(
            r'<script id="tracebi-formats" type="application/json">(.*?)</script>', html)
        assert block and json.loads(block.group(1))["t"]["cost"] == "currency0"

    def test_an_unformatted_value_figure_takes_the_table_precedence(self, tmp_path):
        """A KPI with no data-tb-format reads like a table cell would: the
        model's declared format, else the shape guess — never raw float noise."""
        df = pd.DataFrame({"region": ["NE", "SE"], "gain": [1000.25, 234.25],
                           "cost": [1000.4, 2000.75]})
        m = DataModel("val_fmt")
        m.add_connector(MemoryConnector("vf", tables={"t": df}))
        m.add_table("t", connector="vf", source="t")
        m.add_fact("f", table_name="t", measures=["gain", "cost"], foreign_keys={})
        m.add_measure("gain", column="gain", agg="sum")
        m.add_measure("cost", column="cost", agg="sum", format="currency0")
        m.connect()

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(json.dumps({"name": "p", "data": {"k": {
            "model": "val_fmt",
            "query": {"fact": "f", "measures": ["gain", "cost"]}}}}))
        (pkg / "template.html").write_text(
            '<html><head><title>p</title></head><body>'
            '<span data-tb-figure="value" data-tb-binding="k" data-tb-cell="gain" id="g"></span>'
            '<span data-tb-figure="value" data-tb-binding="k" data-tb-cell="cost" id="c"></span>'
            '<span data-tb-figure="value" data-tb-binding="k" data-tb-cell="cost" '
            'data-tb-format="decimal" id="d"></span>'
            '</body></html>')
        out = tmp_path / "o.html"
        TemplatePackage(str(pkg)).render({"val_fmt": m}, str(out))
        html = out.read_text(encoding="utf-8")

        assert 'id="g">1,234.50</span>' in html        # shape guess: two decimals
        assert 'id="c">$3,001</span>' in html       # the measure's declared format
        assert 'id="d">3,001.15</span>' in html     # the author's format wins


class TestReaderAidsFailLoudly:
    """data-tb-bars and data-tb-direction are checked at build: a typo'd
    column or direction fails with the fix, never renders as nothing."""

    def _build(self, tmp_path, body):
        df = pd.DataFrame({"region": ["NE", "SE"], "gain": [10.0, -5.0]})
        m = DataModel("aids")
        m.add_connector(MemoryConnector("a", tables={"t": df}))
        m.add_table("t", connector="a", source="t")
        m.add_dimension("dim_r", table_name="t", key_col="region",
                        attributes=["region"])
        m.add_fact("f", table_name="t", measures=["gain"], foreign_keys={})
        m.add_measure("gain", column="gain", agg="sum")
        m.connect()
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(json.dumps({"name": "p", "data": {
            "by": {"model": "aids", "query": {"fact": "f", "measures": ["gain"],
                                              "dimensions": ["dim_r.region"]}},
            "k": {"model": "aids", "query": {"fact": "f", "measures": ["gain"]}}}}))
        (pkg / "template.html").write_text(
            f"<html><head><title>p</title></head><body>{body}</body></html>")
        out = tmp_path / "o.html"
        TemplatePackage(str(pkg)).render({"aids": m}, str(out))
        return out.read_text(encoding="utf-8")

    def test_valid_aids_build(self, tmp_path):
        html = self._build(tmp_path,
            '<table data-tb-figure="table" data-tb-binding="by" id="t" '
            'data-tb-sort data-tb-bars="gain"></table>'
            '<span data-tb-figure="value" data-tb-binding="k" data-tb-cell="gain" '
            'data-tb-direction="up-good" id="v"></span>')
        assert 'data-tb-bars="gain"' in html

    @pytest.mark.parametrize("body, match", [
        ('<table data-tb-figure="table" data-tb-binding="by" id="t" '
         'data-tb-bars="gian"></table>', "Did you mean 'gain'"),
        ('<table data-tb-figure="table" data-tb-binding="by" id="t" '
         'data-tb-bars="dim_r.region"></table>', "not a numeric column"),
        ('<span data-tb-figure="value" data-tb-binding="k" data-tb-cell="gain" '
         'data-tb-direction="up" id="v"></span>', "up-good"),
        ('<table data-tb-figure="table" data-tb-binding="by" id="t" '
         'data-tb-direction="up-good"></table>', "on a table figure"),
    ])
    def test_bad_aids_fail_the_build(self, tmp_path, body, match):
        from tracebi.reports.figures import FigureError
        with pytest.raises(FigureError, match=match):
            self._build(tmp_path, body)

"""
Report generator — M0 trust kernel.

Covers the keystone claim: a reviewer with the shipped ``.html`` and its
manifest can confirm offline that the numbers it ships are exactly what the
recorded queries produced, while a number edited in the file is caught.

* the safe embedder neutralises a hostile cell value and round-trips exact bytes
* ``stamp`` fingerprints via the single ``frame_fingerprint`` algorithm
* ``verify_file`` passes an untampered file, fails a tampered one naming the
  binding, and is independent of ``verify_manifest`` (query → model), which is
  unchanged.

Self-contained: a tiny in-memory star model, no warehouse, no project files.
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector, QuerySpec
from tracebi.reports.embed import (
    canonical_triple,
    embed_block,
    embed_json,
    embedded_record,
    fingerprint_triple,
    stamp,
    stamp_frame,
)
from tracebi.reports.report import Report, TableSection
from tracebi.reports.template_package import TemplatePackage
from tracebi.verify import (
    FILE_ALTERED,
    FILE_INTACT,
    FILE_MATCHES,
    FILE_MISSING,
    FILE_TAMPERED,
    FILE_UNBACKED,
    FILE_UNRECORDED,
    REPRODUCES,
    UNVERIFIABLE,
    verify_file,
    verify_manifest,
)


# ── Fixtures: a tiny star model, one hostile issuer name in the data ──────────

@pytest.fixture
def orders_df():
    return pd.DataFrame({
        "order_id":    [1, 2, 3, 4],
        "customer_id": [10, 10, 20, 20],
        "revenue":     [100.5, 200.25, 300.0, 400.75],
    })


@pytest.fixture
def customers_df():
    # An attacker-influencable label (issuer names come from prose) that would
    # break out of a <script> if embedded naively.
    return pd.DataFrame({
        "customer_id": [10, 20],
        "region":      ["</script><img src=x onerror=alert(1)>", "North"],
    })


@pytest.fixture
def model(orders_df, customers_df):
    m = DataModel("kernel_model")
    m.add_connector(MemoryConnector("mem", tables={
        "orders": orders_df, "customers": customers_df,
    }))
    m.add_table("orders", connector="mem", source="orders")
    m.add_table("customers", connector="mem", source="customers")
    m.add_dimension("dim_customer", table_name="customers",
                    key_col="customer_id", attributes=["region"])
    m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
               foreign_keys={"dim_customer": "customer_id"})
    m.add_measure("revenue", column="revenue", agg="sum")
    return m


@pytest.fixture
def spec():
    return QuerySpec(fact="fact_orders", measures=["revenue"],
                     dimensions=["dim_customer.region"])


@pytest.fixture
def artifacts(model, spec):
    """Hand-wire a manifest + matching self-contained .html, as M0 does by hand.

    Returns (html_str, manifest_dict, models_map).
    """
    sd = stamp(model, spec, name="by_region")
    report = Report("kernel").author("t")
    report.add(TableSection(title="By region", dataset=sd.dataset, id="by_region"))
    manifest = report.build_manifest("html", "report.html")
    manifest.embedded_data = [embedded_record(sd)]
    manifest_dict = manifest.to_dict()
    html = (
        "<!doctype html><html><head></head><body>\n"
        f"{embed_block(sd)}\n</body></html>\n"
    )
    return html, manifest_dict, {model.name: model}


# ── The safe embedder (architecture §5) ───────────────────────────────────────

class TestEmbedJson:
    def test_neutralises_injection(self):
        payload = {"cell": "</script><img src=x onerror=alert(1)>"}
        block = embed_json(payload, "x")
        inner = re.match(r'<script id="x" type="application/json">(.*)</script>',
                         block, re.DOTALL).group(1)
        # No raw breakout characters survive into the script body.
        assert "</script>" not in inner
        assert "<" not in inner
        assert ">" not in inner
        assert "&" not in inner
        assert " " not in inner and " " not in inner

    def test_roundtrips_exact_bytes(self):
        payload = {"cell": "</script> &<>", "n": 12345.6789, "list": [1, "a"]}
        block = embed_json(payload, "x")
        inner = re.match(r'<script[^>]*>(.*)</script>', block, re.DOTALL).group(1)
        # JSON.parse's Python equivalent recovers the object exactly.
        assert json.loads(inner) == payload

    def test_block_is_non_executable_json(self):
        block = embed_json({"a": 1}, "elem")
        assert 'type="application/json"' in block
        assert 'id="elem"' in block


# ── The canonical triple is the single fingerprint algorithm ──────────────────

class TestStamp:
    def test_fingerprint_equals_triple_hash(self, model, spec):
        sd = stamp(model, spec, name="by_region")
        assert sd.fingerprint == fingerprint_triple(sd.triple)

    def test_triple_matches_frame_fingerprint(self, model, spec):
        sd = stamp(model, spec, name="by_region")
        # The triple is exactly what frame_fingerprint hashed on the frame.
        assert sd.triple == canonical_triple(sd.dataset.to_pandas())
        assert sd.fingerprint == sd.dataset.fingerprint()

    def test_recovers_query_and_model_from_lineage(self, model, spec):
        sd = stamp(model, spec, name="by_region")
        assert sd.model == "kernel_model"
        assert sd.query_spec == spec.to_dict()

    def test_block_carries_no_unverified_display_copy(self, model, spec):
        # The embedded block is only the fingerprinted triple — no separate
        # `records` copy the page could draw from unverified. The page parses
        # the same `csv` the receipt covers, so a displayed number cannot
        # silently diverge from a verified one.
        sd = stamp(model, spec, name="by_region")
        block = embed_block(sd)
        body = json.loads(re.search(r">(.*?)</script>", block, re.DOTALL).group(1))
        assert set(body) == {"name", "columns", "dtypes", "csv"}
        assert "records" not in body

    def test_embedded_record_shape(self, model, spec):
        sd = stamp(model, spec, name="by_region")
        rec = embedded_record(sd)
        assert rec == {
            "name": "by_region",
            "embedded_sha256": sd.fingerprint,
            "query_spec": spec.to_dict(),
            "model": "kernel_model",
        }

    def test_embedded_bytes_recover_to_fingerprint(self, model, spec):
        """The bytes shipped in the block rehash to the fingerprint — no frame."""
        sd = stamp(model, spec, name="by_region")
        block = embed_block(sd)
        parsed = json.loads(
            re.match(r'<script[^>]*>(.*)</script>', block, re.DOTALL).group(1)
        )
        triple = {k: parsed[k] for k in ("columns", "dtypes", "csv")}
        assert fingerprint_triple(triple) == sd.fingerprint


# ── verify_file: embedded bytes → manifest (architecture §3.2) ────────────────

class TestVerifyFile:
    def test_untampered_passes(self, artifacts):
        html, manifest, _ = artifacts
        result = verify_file(html, manifest)
        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert result["verdict"] == FILE_INTACT
        assert [b["status"] for b in result["bindings"]] == [FILE_MATCHES]
        assert result["bindings"][0]["binding"] == "by_region"

    def test_tampered_number_fails_naming_binding(self, artifacts):
        html, manifest, _ = artifacts
        # Edit a real figure inside the embedded CSV — the $99,999,999 case.
        # 700.75 is an aggregated revenue in the embedded CSV; the csv copy
        # precedes the display 'records' copy, so the first replace hits the
        # fingerprinted bytes.
        assert "700.75" in html
        tampered = html.replace("700.75", "99999999.75", 1)
        assert tampered != html
        result = verify_file(tampered, manifest)
        assert result["ok"] is False
        assert result["exit_code"] == 1
        assert result["verdict"] == FILE_ALTERED
        offenders = [b for b in result["bindings"] if b["status"] == FILE_TAMPERED]
        assert [b["binding"] for b in offenders] == ["by_region"]

    def test_missing_block_in_file_fails(self, artifacts):
        html, manifest, _ = artifacts
        # A file that dropped the data block entirely still must not pass.
        stripped = re.sub(r'<script id="tracebi-data[^>]*>.*?</script>', "",
                          html, flags=re.DOTALL)
        result = verify_file(stripped, manifest)
        assert result["ok"] is False
        assert result["bindings"][0]["status"] == FILE_MISSING

    def test_block_with_no_manifest_binding_fails(self, artifacts):
        html, manifest, _ = artifacts
        manifest_no_rec = dict(manifest)
        manifest_no_rec.pop("embedded_data", None)
        result = verify_file(html, manifest_no_rec)
        assert result["ok"] is False
        assert result["bindings"][0]["status"] == FILE_UNRECORDED

    def test_unbacked_when_no_section_carries_the_fingerprint(self, artifacts):
        # A manifest whose sections were stripped still records the embedded
        # binding — the bytes "match", but nothing in the receipt vouches for
        # them. That must fail, not read as INTACT (it did, before the fix).
        html, manifest, _ = artifacts
        broken = dict(manifest)
        broken["sections"] = [
            {k: v for k, v in s.items() if k != "dataset_fingerprint"}
            for s in manifest["sections"]
        ]
        result = verify_file(html, broken)
        assert result["ok"] is False
        assert result["verdict"] == FILE_ALTERED
        assert result["bindings"][0]["status"] == FILE_UNBACKED


# ── verify_manifest (query → model) is a separate, unaffected check ────────────

class TestChecksAreIndependent:
    def test_tampering_the_file_does_not_affect_manifest_verify(self, artifacts):
        """The whole point: a number edited in the .html leaves the model-
        reproduction check green, and only ``verify_file`` catches it."""
        html, manifest, models = artifacts
        tampered = html.replace("700.75", "99999999.75", 1)
        # File check fails …
        assert verify_file(tampered, manifest)["ok"] is False
        # … while manifest (query → model) reproduces, untouched.
        assert verify_manifest(manifest, models)["ok"] is True


# ── CLI wiring: `tracebi verify --file` dispatches to the file checker ─────────

class TestVerifyFileCLI:
    def _write(self, tmp_path, artifacts):
        html, manifest, _ = artifacts
        html_path = tmp_path / "report.html"
        html_path.write_text(html, encoding="utf-8")
        (tmp_path / "report.html.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return html_path

    def test_cli_missing_sibling_manifest_errors(self, tmp_path, artifacts, capsys):
        from tracebi.cli import main
        html, _, _ = artifacts
        html_path = tmp_path / "lonely.html"
        html_path.write_text(html, encoding="utf-8")
        assert main(["verify", "--file", str(html_path)]) == 1
        assert "no manifest found" in capsys.readouterr().err


# ── M0: metric receipts in the legacy spec lane (plan §2.2) ───────────────────

class TestMetricReceipts:
    """A metrics section with a ``data`` query embeds its one-row canonical
    triple in the rendered HTML and records it in ``embedded_data``, so
    ``verify --file`` covers the page's biggest numbers (report architecture
    v2 §2.2 — the metric-receipt hole). The KPI cards themselves stay
    server-rendered; the embedded triple is the receipt, not the display.
    """

    def _render(self, tmp_path, model):
        from tracebi.reports.html_renderer import HTMLRenderer
        from tracebi.spec import ReportSpec

        spec = ReportSpec.from_dict({
            "name": "KPIs",
            "sections": [{
                "type": "metrics", "title": "Totals", "id": "kpis",
                "data": {"model": "kernel_model", "query": {
                    "fact": "fact_orders", "measures": ["revenue"]}},
                "metrics": [
                    {"label": "Revenue", "value": "revenue",
                     "format": "currency0"},
                ],
            }],
        })
        report = spec.build({model.name: model})
        out = tmp_path / "kpis.html"
        manifest = HTMLRenderer().render(report, str(out))
        return out.read_text(encoding="utf-8"), manifest

    def test_kpi_triple_embedded_and_recorded(self, tmp_path, model):
        html, manifest = self._render(tmp_path, model)
        # The card resolves from the query (total revenue = 1001.5) …
        assert "$1,002" in html
        # … and the same one-row triple ships as a fingerprinted block.
        assert 'id="tracebi-data-kpis"' in html
        rec = manifest.to_dict()["embedded_data"][0]
        assert rec["name"] == "kpis"
        assert rec["model"] == "kernel_model"
        assert rec["query_spec"]["fact"] == "fact_orders"
        # The embedded fingerprint is the metrics section's fingerprint —
        # one algorithm (M0 flip ledger: the section gains these fields).
        assert rec["embedded_sha256"] == (
            manifest.sections[0]["dataset_fingerprint"])
        # No chart on the page: the KPI block rides without the chart
        # machinery (no bundle, no config, no init) — but the CSP still ships.
        assert "tracebi-charts" not in html
        assert "!function(" not in html
        assert "Content-Security-Policy" in html

    def test_file_check_passes_then_catches_kpi_tamper(self, tmp_path, model):
        html, manifest = self._render(tmp_path, model)
        man = manifest.to_dict()
        assert verify_file(html, man)["verdict"] == FILE_INTACT

        # Edit the KPI total inside the embedded triple — the $99,999,999 case.
        assert "1001.5" in html
        tampered = html.replace("1001.5", "99999999.5", 1)
        result = verify_file(tampered, man)
        assert result["ok"] is False
        assert result["verdict"] == FILE_ALTERED
        offenders = [b["binding"] for b in result["bindings"]
                     if b["status"] == FILE_TAMPERED]
        assert offenders == ["kpis"]

    def test_metrics_and_chart_coexist_in_one_receipt(self, tmp_path, model):
        """A page with both renders the chart as inline SVG and embeds both
        triples, recorded in the manifest."""
        from tracebi.reports.html_renderer import HTMLRenderer
        from tracebi.spec import ReportSpec

        spec = ReportSpec.from_dict({
            "name": "Mixed",
            "sections": [
                {"type": "metrics", "id": "kpis",
                 "data": {"model": "kernel_model", "query": {
                     "fact": "fact_orders", "measures": ["revenue"]}},
                 "metrics": [{"label": "Revenue", "value": "revenue"}]},
                {"type": "chart", "id": "by_region", "chart_type": "bar",
                 "x": "dim_customer.region", "y": "revenue",
                 "data": {"model": "kernel_model", "query": {
                     "fact": "fact_orders", "measures": ["revenue"],
                     "dimensions": ["dim_customer.region"]}}},
            ],
        })
        report = spec.build({model.name: model})
        out = tmp_path / "mixed.html"
        manifest = HTMLRenderer().render(report, str(out))
        html = out.read_text(encoding="utf-8")

        assert 'id="tracebi-data-kpis"' in html
        assert 'id="tracebi-data-by_region"' in html
        assert '<svg class="tb-chart' in html   # the chart renders as inline SVG
        names = [r["name"] for r in manifest.to_dict()["embedded_data"]]
        assert names == ["kpis", "by_region"]
        assert verify_file(html, manifest.to_dict())["verdict"] == FILE_INTACT


# ── M1: the template-package renderer (architecture §2 lane B, §8-M1) ─────────

# A template that OMITS every framework placeholder — no {{ head_extra }},
# {{ body_extra }}, no data mount. The renderer must still inject the data by
# string insertion, or the page ships with no numbers (the §8-M1 hole).
_BARE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title></head>
<body><main><h1>Bare</h1><table id="mount"><tbody></tbody></table></main></body>
</html>
"""

_STYLE = "body { color: rebeccapurple; } table { border-collapse: collapse; }"

_SCRIPT = (
    'var el = document.getElementById("tracebi-data-by_region");'
    "var data = JSON.parse(el.textContent); void data;"
)


def _write_package(tmp_path, *, template=_BARE_TEMPLATE, style=_STYLE,
                   script=_SCRIPT, data=None, name="Regions",
                   dirname="regions", report_py=None, libs=None,
                   figures=None, title=None):
    """Write a package dir and return its path."""
    if data is None:
        data = {
            "by_region": {
                "model": "kernel_model",
                "query": {
                    "fact": "fact_orders",
                    "measures": ["revenue"],
                    "dimensions": ["dim_customer.region"],
                },
            }
        }
    pkg = tmp_path / dirname
    pkg.mkdir()
    decl = {"name": name, "author": "t", "data": data}
    if title is not None:
        decl["title"] = title
    if libs is not None:
        decl["libs"] = libs
    if figures is not None:
        decl["figures"] = figures
    (pkg / "report.json").write_text(json.dumps(decl), encoding="utf-8")
    (pkg / "template.html").write_text(template, encoding="utf-8")
    if style is not None:
        (pkg / "style.css").write_text(style, encoding="utf-8")
    if script is not None:
        (pkg / "script.js").write_text(script, encoding="utf-8")
    if report_py is not None:
        (pkg / "report.py").write_text(report_py, encoding="utf-8")
    return pkg


class TestProseNumericGate:
    def test_a_hand_typed_number_fails_the_build(self, tmp_path, model):
        template = _BARE_TEMPLATE.replace(
            "<h1>Bare</h1>",
            "<h1>Bare</h1><p>The book is worth 12.</p>",
        )
        pkg = _write_package(tmp_path, template=template, dirname="prose")
        with pytest.raises(Exception, match="numeric literal"):
            TemplatePackage(str(pkg)).render({model.name: model}, str(tmp_path / "out.html"))

    def test_a_number_inside_exploration_is_stripped_before_the_gate(self, tmp_path, model):
        template = _BARE_TEMPLATE.replace(
            "<h1>Bare</h1>",
            '<h1>Bare</h1><div data-tb-stage="exploration"><p>Worth 12.</p></div>',
        )
        pkg = _write_package(tmp_path, template=template, dirname="explore_num")
        TemplatePackage(str(pkg)).render(
            {model.name: model}, str(tmp_path / "out.html"))


class TestTemplatePackageRender:
    def test_self_contained_no_external_refs(self, tmp_path, model):
        pkg = _write_package(tmp_path)
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        # No external CSS/JS/img: nothing that would phone home when opened.
        assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', html)

    def test_strict_csp_on_every_page(self, tmp_path, model):
        # The CSP (architecture §5) ships even on a data-only report: no
        # network, and — since ECharts needs none — no unsafe-eval.
        pkg = _write_package(tmp_path)
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert 'http-equiv="Content-Security-Policy"' in html
        assert "connect-src 'none'" in html
        # JS eval is never granted. 'wasm-unsafe-eval' IS (WebAssembly
        # compilation only, for the artifact's worker engine) — and since
        # it CONTAINS "'unsafe-eval'", drop it before checking.
        assert "'unsafe-eval'" not in html.replace("'wasm-unsafe-eval'", "")

    def test_author_text_cannot_suppress_the_csp(self):
        """The strict CSP must not be dropped just because a template mentions
        the phrase 'Content-Security-Policy' (a comment, prose, or a weaker
        policy an injection tried to smuggle) — dedup is on the exact meta."""
        from tracebi.reports.embed import csp_meta
        from tracebi.reports.stack import apply_stack

        page = ("<html><head></head><body>"
                "<!-- notes on Content-Security-Policy -->"
                "<p>our Content-Security-Policy is strict</p></body></html>")
        out = apply_stack(page, libs=(), data_blocks_html="")
        assert csp_meta().strip() in out, "author text must not disable the CSP"

        # But the EXACT framework meta (as html_renderer inserts) is not doubled.
        page2 = f"<html><head>{csp_meta()}</head><body></body></html>"
        out2 = apply_stack(page2, libs=(), data_blocks_html="")
        assert out2.count('http-equiv="Content-Security-Policy"') == 1


#: A template that owns its own layout and asks the framework only for the
#: figure elements — the "custom layout, framework-built figures" middle
#: ground between a rigid spec and hand-written data-tb-* markup.
_FIGURE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title></head>
<body><main class="my-own-grid">
  <section class="my-card">{{ figure("total") }}</section>
  <section class="my-wide">{{ figure("detail") }}</section>
</main></body></html>
"""


def _doc(body: str) -> str:
    """A complete document — the data/style/script injection needs one."""
    return (f"<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            f"<title>t</title></head><body>{body}</body></html>\n")


class TestFigureHelper:
    """``{{ figure("name") }}`` — the framework builds the figure, the
    analyst keeps the layout."""

    _DATA = {
        "by_region": {
            "model": "kernel_model",
            "query": {"fact": "fact_orders", "measures": ["revenue"],
                      "dimensions": ["dim_customer.region"]},
        },
        "totals": {
            "model": "kernel_model",
            "query": {"fact": "fact_orders", "measures": ["revenue"]},
        },
    }

    _FIGURES = {
        "total": {"kind": "value", "binding": "totals", "cell": "revenue",
                  "label": "Revenue", "format": "currency"},
        "detail": {"kind": "table", "binding": "by_region"},
    }

    def _pkg(self, tmp_path, **kw):
        kw.setdefault("template", _FIGURE_TEMPLATE)
        kw.setdefault("figures", self._FIGURES)
        kw.setdefault("data", self._DATA)
        kw.setdefault("script", None)
        return _write_package(tmp_path, **kw)

    def test_helper_emits_the_grammar_and_keeps_author_layout(
            self, tmp_path, model):
        out = tmp_path / "out.html"
        TemplatePackage(str(self._pkg(tmp_path))).render(
            {model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        # The framework's grammar, which the author never hand-wrote…
        assert 'data-tb-figure="value"' in html
        assert 'data-tb-binding="by_region"' in html
        assert 'data-tb-cell="revenue"' in html
        assert 'data-tb-figure="table"' in html
        # …placed inside the author's own markup, untouched.
        assert '<section class="my-card">' in html
        assert 'class="my-own-grid"' in html

    def test_helper_figures_are_receipted_like_hand_written_ones(
            self, tmp_path, model):
        """The receipt cannot tell who typed the markup — that is the point."""
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(self._pkg(tmp_path))).render(
            {model.name: model}, str(out))
        ids = {f["id"] for f in manifest.to_dict()["figures"]}
        assert {"fig-total", "fig-detail"} <= ids
        html = out.read_text(encoding="utf-8")
        assert verify_file(html, manifest.to_dict())["verdict"] == FILE_INTACT

    def test_byte_identical_to_the_spec_compiler(self, tmp_path, model):
        """One emitter, two lanes: the same declaration must produce the same
        bytes whether the spec compiler or the template helper built it — or
        the lanes have quietly forked."""
        from tracebi.reports.compile_spec import compile_spec
        from tracebi.spec import ReportSpec

        ref = {"model": model.name,
               "query": {"fact": "fact_orders", "measures": ["revenue"],
                         "dimensions": ["dim_customer.region"]}}
        spec = ReportSpec.from_dict({
            "name": "Cmp",
            "sections": [{"type": "chart", "id": "by_region", "data": ref,
                          "chart_type": "line", "x": "region",
                          "y": ["revenue"]}],
        })
        compiled = compile_spec(spec).files["template.html"]

        pkg = TemplatePackage(str(self._pkg(
            tmp_path,
            figures={"by_region": {"kind": "chart", "binding": "by_region",
                                   "chart_type": "line", "x": "region",
                                   "y": ["revenue"]}},
            template=_doc('{{ figure("by_region") }}'),
        )))
        element = pkg._build_figure("by_region", pkg.figures["by_region"])
        assert element in compiled, (
            f"the helper emitted:\n  {element}\nwhich the spec compiler's "
            f"output does not contain:\n{compiled}"
        )

    def test_declared_but_never_placed_refuses_to_build(self, tmp_path, model):
        """The failure this package format exists to prevent: a declared
        number that silently does not appear."""
        pkg = self._pkg(
            tmp_path,
            template=_doc('{{ figure("total") }}'),
        )
        with pytest.raises(ValueError, match=r"never places"):
            TemplatePackage(str(pkg)).render(
                {model.name: model}, str(tmp_path / "out.html"))

    def test_placing_one_figure_twice_refuses(self, tmp_path, model):
        """A figure id is the receipt's address for ONE number."""
        pkg = self._pkg(
            tmp_path,
            figures={"total": self._FIGURES["total"]},
            template=_doc('{{ figure("total") }}{{ figure("total") }}'),
        )
        with pytest.raises(ValueError, match=r"more than once"):
            TemplatePackage(str(pkg)).render(
                {model.name: model}, str(tmp_path / "out.html"))

    def test_undeclared_figure_name_refuses(self, tmp_path, model):
        pkg = self._pkg(
            tmp_path,
            figures={"total": self._FIGURES["total"]},
            template=_doc('{{ figure("nope") }}'),
        )
        with pytest.raises(ValueError, match=r"not declared"):
            TemplatePackage(str(pkg)).render(
                {model.name: model}, str(tmp_path / "out.html"))

    def test_figure_naming_an_undeclared_binding_fails_at_load(self, tmp_path):
        """Caught when the package loads, not as a blank panel at render."""
        pkg = self._pkg(
            tmp_path,
            figures={"total": {"kind": "value", "binding": "ghost",
                               "cell": "revenue"}},
        )
        with pytest.raises(ValueError, match=r"not declared in 'data'"):
            TemplatePackage(str(pkg))

    def test_value_figure_without_a_cell_fails_at_load(self, tmp_path):
        pkg = self._pkg(
            tmp_path,
            figures={"total": {"kind": "value", "binding": "by_region"}},
        )
        with pytest.raises(ValueError, match=r"needs 'cell'"):
            TemplatePackage(str(pkg))

    def test_unknown_kind_fails_at_load(self, tmp_path):
        pkg = self._pkg(
            tmp_path,
            figures={"total": {"kind": "sparkline", "binding": "by_region"}},
        )
        with pytest.raises(ValueError, match=r"expected one of"):
            TemplatePackage(str(pkg))

    def test_packages_without_figures_are_unaffected(self, tmp_path, model):
        """Hand-written data-tb-* packages keep working: the helper is sugar."""
        out = tmp_path / "out.html"
        TemplatePackage(str(_write_package(tmp_path))).render(
            {model.name: model}, str(out))
        assert TemplatePackage(str(_write_package(
            tmp_path, dirname="r2"))).figures == {}


class TestChartLibraries:
    def test_echarts_bundle_inlined_offline(self, tmp_path, model):
        pkg = _write_package(tmp_path, libs=["echarts"])
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        # The library is inlined verbatim (its global is defined), and still no
        # loaded external resource — the bundle carries its own license URL as
        # an inert comment, which is not a src/href.
        assert "window.echarts" in html or "echarts" in html
        assert len(html) > 400_000  # the bundle is actually present
        assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', html)
        # The inlined bundle is a redistribution of Apache ECharts (+ zrender,
        # tslib); its required attribution must travel with every report.
        assert "Apache ECharts" in html
        assert "NOTICE in the TraceBi distribution" in html

    def test_unknown_lib_is_rejected(self, tmp_path):
        pkg = _write_package(tmp_path, libs=["nope"])
        with pytest.raises(ValueError, match="libs"):
            TemplatePackage(str(pkg))

    _CHART_DATA = {
        "by_region": {
            "model": "kernel_model",
            "query": {
                "fact": "fact_orders",
                "measures": ["revenue"],
                "dimensions": ["dim_customer.region"],
            },
        },
    }
    _CHART_FIGURES = {
        "by_region": {
            "kind": "chart", "binding": "by_region", "chart_type": "bar",
            "x": "dim_customer.region", "y": "revenue",
        },
    }

    def _render_chart(self, tmp_path, model, *, libs=None, template=None,
                      figures=None, dirname="charted"):
        from tracebi.reports.embed import read_lib

        pkg = _write_package(
            tmp_path, dirname=dirname, script=None, libs=libs,
            data=self._CHART_DATA,
            figures=self._CHART_FIGURES if figures is None else figures,
            template=template or _doc('{{ figure("by_region") }}'),
        )
        out = tmp_path / f"{dirname}.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out))
        return out.read_text(encoding="utf-8"), manifest, read_lib("d3")

    def test_chart_figure_inlines_d3_without_libs(self, tmp_path, model):
        """A chart on the page the reader gets pulls D3 in, and not ECharts.
        ``libs`` is optional, including a chart emitted by ``{{ figure() }}``."""
        from tracebi.reports.embed import read_lib

        html, manifest, bundle = self._render_chart(tmp_path, model)
        assert 'data-tb-figure="chart"' in html
        assert html.count(bundle) == 1
        assert read_lib("echarts") not in html
        assert verify_file(html, manifest.to_dict())["verdict"] == FILE_INTACT

    def test_hand_written_chart_inlines_d3_without_libs(
            self, tmp_path, model):
        html, _manifest, bundle = self._render_chart(
            tmp_path, model, dirname="hand", figures={},
            template=_doc(
                '<div data-tb-figure="chart" data-tb-binding="by_region" '
                'data-tb-type="bar" data-tb-x="dim_customer.region" '
                'data-tb-y="revenue" id="by-region"></div>'),
        )
        assert html.count(bundle) == 1

    def test_page_without_a_chart_omits_echarts(self, tmp_path, model):
        from tracebi.reports.embed import read_lib

        pkg = _write_package(tmp_path, dirname="plain")
        out = tmp_path / "plain.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert read_lib("echarts") not in html
        assert "Apache ECharts" not in html

    def test_explicit_echarts_keeps_the_echarts_engine(self, tmp_path, model):
        """A report that lists echarts keeps that engine (its configureChart
        patches apply only there): ECharts once, and no D3 added."""
        from tracebi.reports.embed import read_lib

        html, _manifest, d3_bundle = self._render_chart(
            tmp_path, model, libs=["echarts"], dirname="listed")
        assert html.count(read_lib("echarts")) == 1
        assert d3_bundle not in html

    def test_exploration_chart_draws_in_dev_and_not_in_the_final_build(
            self, tmp_path, model):
        """A final build decides after exploration is stripped. Dev keeps
        the block, so a chart that lives only there still draws."""
        from tracebi.reports.embed import read_lib

        bundle = read_lib("d3")
        pkg = _write_package(
            tmp_path, dirname="explore_chart", script=None,
            data=self._CHART_DATA, figures=self._CHART_FIGURES,
            template=_doc(
                '<div data-tb-stage="exploration">'
                '{{ figure("by_region") }}</div>'),
        )
        package = TemplatePackage(str(pkg))
        out = tmp_path / "explore_chart.html"
        package.render({model.name: model}, str(out))
        final = out.read_text(encoding="utf-8")
        assert bundle not in final
        dev, _inputs, _outputs = package.render_exploration(
            {model.name: model})
        assert dev.count(bundle) == 1

    def test_data_embedded_as_safe_json_with_triple(self, tmp_path, model):
        pkg = _write_package(tmp_path)
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        block = re.search(
            r'<script id="tracebi-data-by_region" '
            r'type="application/json">(.*?)</script>',
            html, re.DOTALL,
        )
        assert block is not None
        parsed = json.loads(block.group(1))
        assert all(k in parsed for k in ("columns", "dtypes", "csv"))
        # The hostile region label is neutralised — no raw breakout survives.
        inner = block.group(1)
        assert "</script>" not in inner
        assert "<" not in inner and ">" not in inner

    def test_manifest_records_embedded_fingerprints(self, tmp_path, model):
        pkg = _write_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        man = json.loads((tmp_path / "out.html.manifest.json").read_text())
        assert man["embedded_data"] == manifest.to_dict()["embedded_data"]
        rec = man["embedded_data"][0]
        assert rec["name"] == "by_region"
        assert rec["model"] == "kernel_model"
        # The embedded fingerprint is the section's fingerprint — one algorithm.
        sec_fps = [s.get("dataset_fingerprint") for s in man["sections"]]
        assert rec["embedded_sha256"] in sec_fps

    def test_verify_file_passes_then_fails_on_tamper(self, tmp_path, model):
        pkg = _write_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        man = manifest.to_dict()
        html = out.read_text(encoding="utf-8")

        ok = verify_file(html, man)
        assert ok["ok"] is True
        assert ok["verdict"] == FILE_INTACT
        assert [b["status"] for b in ok["bindings"]] == [FILE_MATCHES]

        # Tamper a real aggregated figure inside the canonical csv bytes.
        assert "700.75" in html
        bad = verify_file(html.replace("700.75", "99999999.75", 1), man)
        assert bad["ok"] is False
        assert bad["verdict"] == FILE_ALTERED
        offenders = [b["binding"] for b in bad["bindings"]
                     if b["status"] == FILE_TAMPERED]
        assert offenders == ["by_region"]

    def test_bare_template_still_gets_data_and_assets_injected(self, tmp_path, model):
        """The §8-M1 hole: a template with no placeholders must not ship empty."""
        template_src = _BARE_TEMPLATE
        # The template references none of the injection slots.
        assert "head_extra" not in template_src
        assert "body_extra" not in template_src
        assert "tracebi-data" not in template_src

        pkg = _write_package(tmp_path, template=template_src)
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        # Data block injected …
        assert 'id="tracebi-data-by_region"' in html
        # … style injected before </head> …
        assert "rebeccapurple" in html
        assert html.index("rebeccapurple") < html.lower().index("</head>")
        # … app script injected before </body>.
        assert 'getElementById("tracebi-data-by_region")' in html
        assert (html.index('getElementById("tracebi-data-by_region")')
                < html.lower().index("</body>"))

    def test_multiple_bindings_each_embedded_and_recorded(self, tmp_path, model):
        data = {
            "by_region": {
                "model": "kernel_model",
                "query": {"fact": "fact_orders", "measures": ["revenue"],
                          "dimensions": ["dim_customer.region"]},
            },
            "grand_total": {
                "model": "kernel_model",
                "query": {"fact": "fact_orders", "measures": ["revenue"]},
            },
        }
        pkg = _write_package(tmp_path, data=data, script=None)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert 'id="tracebi-data-by_region"' in html
        assert 'id="tracebi-data-grand_total"' in html
        names = {r["name"] for r in manifest.to_dict()["embedded_data"]}
        assert names == {"by_region", "grand_total"}
        assert verify_file(html, manifest.to_dict())["ok"] is True


class TestTemplatePackageLoading:
    def test_missing_report_json_raises(self, tmp_path):
        pkg = tmp_path / "p"
        pkg.mkdir()
        (pkg / "template.html").write_text("<html></html>", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            TemplatePackage(str(pkg))

    def test_missing_template_raises(self, tmp_path):
        pkg = tmp_path / "p"
        pkg.mkdir()
        (pkg / "report.json").write_text('{"data": {}}', encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            TemplatePackage(str(pkg))

    def test_empty_data_object_raises(self, tmp_path):
        pkg = tmp_path / "p"
        pkg.mkdir()
        (pkg / "report.json").write_text('{"data": {}}', encoding="utf-8")
        (pkg / "template.html").write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty 'data'"):
            TemplatePackage(str(pkg))

    def test_binding_missing_query_raises_structurally(self, tmp_path):
        pkg = tmp_path / "p"
        pkg.mkdir()
        (pkg / "report.json").write_text(
            '{"data": {"x": {"model": "m"}}}', encoding="utf-8")
        (pkg / "template.html").write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="query"):
            TemplatePackage(str(pkg))

    def test_unknown_model_raises_at_render(self, tmp_path, model):
        pkg = _write_package(tmp_path)
        out = tmp_path / "out.html"
        with pytest.raises(ValueError, match="was not supplied"):
            TemplatePackage(str(pkg)).render({}, str(out))
        # Manifest-first ordering means no page was written on the failed render.
        assert not out.exists()

    def test_fragment_template_builds_a_complete_document(self, tmp_path, model):
        # <header> must not be read as <head>. A pure fragment is wrapped,
        # then the runtime is injected into that document.
        pkg = _write_package(
            tmp_path,
            template="<header><section>Regions</section></header>",
            style=None, script=None, title="Book Value",
        )
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert html.lower().lstrip().startswith("<!doctype html>")
        assert "<html>" in html.lower()
        assert "</head>" in html.lower()
        assert '<meta charset="utf-8">' in html
        assert "<title>Book Value</title>" in html
        assert "<header><section>Regions</section></header>" in html
        assert "<!-- tracebi.js: the TraceBi runtime -->" in html
        assert "Content-Security-Policy" in html

    def test_fragment_without_title_uses_the_report_name(self, tmp_path, model):
        pkg = _write_package(
            tmp_path, template="<section>Regions</section>",
            style=None, script=None,
        )
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert "<title>Regions</title>" in html

    def test_malformed_document_without_head_close_still_fails(self, tmp_path, model):
        pkg = _write_package(
            tmp_path,
            template="<html><body>no head</body></html>",
            style=None, script=None,
        )
        out = tmp_path / "out.html"
        with pytest.raises(ValueError, match=r"no </head>"):
            TemplatePackage(str(pkg)).render({model.name: model}, str(out))

# ── M2: discovery branch (architecture §7) ────────────────────────────────────

class TestPackageDiscovery:
    def _reports_dir(self, tmp_path):
        """A reports/ dir: one freeform package, one bare (non-package) subdir."""
        reports = tmp_path / "reports"
        pkg = reports / "regions"
        pkg.mkdir(parents=True)
        (pkg / "report.json").write_text(json.dumps({
            "name": "Regions", "author": "t",
            "data": {"by_region": {"model": "kernel_model", "query": {
                "fact": "fact_orders", "measures": ["revenue"],
                "dimensions": ["dim_customer.region"]}}},
        }), encoding="utf-8")
        (pkg / "template.html").write_text(_BARE_TEMPLATE, encoding="utf-8")
        (reports / "not_a_package").mkdir()
        return reports

    def test_package_dir_is_registered_bare_subdir_skipped(self, tmp_path):
        from tracebi.registry import registry
        from tracebi.web import discovery

        discovery.clear_discovery_report()
        discovered = discovery.auto_discover(str(self._reports_dir(tmp_path)))
        assert "regions" in discovered
        assert "regions" in {r["name"] for r in registry.list_reports()}

        outcomes = {o["file"]: o for o in discovery.discovery_report()
                    if o.get("file")}
        assert outcomes["regions"]["status"] == "registered"
        # A subdirectory that isn't a package is a folder: scanned, and it
        # registers nothing itself.
        assert outcomes["not_a_package"]["status"] == "skipped"
        assert "a folder" in outcomes["not_a_package"]["reason"]

    def test_factory_resolves_model_at_call_time(self, tmp_path, model, monkeypatch):
        """Discovery registers the package before the model exists; the factory
        binds it only when called — so a Report comes back with a section per
        binding, resolved lazily (architecture §7)."""
        from tracebi import model_registry
        from tracebi.registry import registry
        from tracebi.web import discovery

        discovery.clear_discovery_report()
        discovery.auto_discover(str(self._reports_dir(tmp_path)))

        # The package's factory resolves models from the model registry at call
        # time; stand the in-memory fixture in for that lookup.
        monkeypatch.setattr(model_registry, "list_models", lambda: ["kernel_model"])
        monkeypatch.setattr(model_registry, "get_model", lambda _n: model)
        report = registry.run_report("regions")
        assert type(report).__name__ == "Report"
        assert len(report.sections) == 1

    def test_broken_package_records_failure_not_crash(self, tmp_path):
        from tracebi.web import discovery

        reports = tmp_path / "reports"
        pkg = reports / "broken"
        pkg.mkdir(parents=True)
        (pkg / "report.json").write_text("{ not json", encoding="utf-8")
        (pkg / "template.html").write_text(_BARE_TEMPLATE, encoding="utf-8")

        discovery.clear_discovery_report()
        assert "broken" not in discovery.auto_discover(str(reports))
        outcome = next(o for o in discovery.discovery_report()
                       if o.get("file") == "broken")
        assert outcome["status"] == "failed"


# ── M2: the CLI (new-report → report build → verify --file) ───────────────────

class TestReportCLI:
    def test_new_report_scaffolds_a_teaching_package_and_binds_first_model(
        self, tmp_path, model, monkeypatch
    ):
        from tracebi import cli

        monkeypatch.setattr(cli, "_load_project_models",
                            lambda: {model.name: model})
        reports = tmp_path / "reports"
        assert cli.main(["new-report", "Regions Demo",
                         "--reports-dir", str(reports)]) == 0

        pkg = reports / "regions_demo"
        # Two files, and NO hand-rolled script.js/style.css — the runtime draws
        # every figure from the stamped bytes (the L0 trap round 3 flagged).
        for f in ("report.json", "template.html"):
            assert (pkg / f).is_file()
        assert not (pkg / "script.js").exists()
        assert not (pkg / "style.css").exists()
        decl = json.loads((pkg / "report.json").read_text())
        totals = decl["data"]["totals"]
        assert totals["model"] == model.name
        assert totals["query"]["fact"] == "fact_orders"
        assert totals["query"]["measures"] == ["revenue"]
        # The template TEACHES the grammar, not a bare table.
        template = (pkg / "template.html").read_text()
        for marker in ("data-tb-figure", "data-tb-binding", "data-tb-filter",
                       "data-tb-search", "data-tb-download", "data-tb-stage",
                       "data-tb-methodology", "tb-lede", "tb-grid", "tb-cols-2",
                       "tb-kpi-context"):
            assert marker in template, f"scaffold no longer teaches {marker}"
        assert "parseCsv" not in template   # no reimplemented runtime

    def test_scaffold_placeholder_when_no_model(self, tmp_path, monkeypatch):
        from tracebi import cli

        monkeypatch.setattr(cli, "_load_project_models", dict)
        name, query, note = cli._scaffold_binding()
        assert name == "your_model"
        assert "measures" in query and note  # a note tells the author to edit

    def test_build_unknown_report_errors(self, tmp_path, monkeypatch):
        from tracebi import cli

        monkeypatch.setattr(cli, "_load_project_models", dict)
        assert cli.main(["report", "build", "nope",
                         "--reports-dir", str(tmp_path / "reports"),
                         "--output", str(tmp_path / "out.html")]) == 1

    def test_spec_validate_names_a_package_report_json_by_lane(self, tmp_path):
        # A package report.json (a 'data' block) is not a JSON ReportSpec —
        # spec validate must say so by name, not reject it as "unknown fields"
        # (which reads as malformed).
        from tracebi import cli
        pkg = tmp_path / "reports" / "my_pkg"
        pkg.mkdir(parents=True)
        (pkg / "report.json").write_text(json.dumps({
            "name": "P", "libs": ["echarts"],
            "data": {"rows": {"model": "m", "query": {"fact": "f",
                                                      "measures": ["x"]}}}}))
        rc = cli.main(["spec", "validate", str(pkg / "report.json")])
        assert rc == 1

    def test_package_build_warns_on_unknown_git_sha(
        self, tmp_path, model, monkeypatch, capsys
    ):
        # The primary (package) lane used to write git_sha:"unknown" silently;
        # the build is the moment to say the receipt can't pin its code state.
        import tracebi.reports.base_renderer as br
        import tracebi.reports.report as report_mod
        from tracebi import cli

        monkeypatch.setattr(cli, "_load_project_models",
                            lambda: {model.name: model})
        monkeypatch.setattr(report_mod, "_GIT_SHA", "unknown")
        monkeypatch.setattr(br, "_GIT_SHA_WARNED", False)
        reports = tmp_path / "reports"
        out = tmp_path / "data" / "gitless.html"
        assert cli.main(["new-report", "Gitless",
                         "--reports-dir", str(reports)]) == 0
        assert cli.main(["report", "build", "gitless",
                         "--reports-dir", str(reports),
                         "--output", str(out)]) == 0
        assert "git_sha is 'unknown'" in capsys.readouterr().err


# ── M3: the report.py escape hatch + honesty (architecture §4, §8-M3) ─────────

# A template that mounts the report.py OUTPUT block (not an input binding).
_OUTPUT_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title></head>
<body><main><h1>Derived</h1><table id="mount"><tbody></tbody></table></main></body>
</html>
"""

# report.py: combines the stamped input into a python-derived output carrying a
# running cumulative — a window the declarative query surface cannot express.
_REPORT_PY = """
import pandas as pd


def build(inputs):
    df = inputs["by_region"].copy()
    df = df.sort_values("revenue", ascending=False).reset_index(drop=True)
    df["cumulative"] = df["revenue"].cumsum()
    return {"ranked": df}
"""


def _escape_hatch_package(tmp_path):
    return _write_package(
        tmp_path, template=_OUTPUT_TEMPLATE,
        script='var el=document.getElementById("tracebi-data-ranked");void el;',
        report_py=_REPORT_PY, name="Ranked", dirname="ranked",
    )


class TestEscapeHatchRender:
    def test_input_and_output_both_embedded(self, tmp_path, model):
        """M0 flip (report-architecture-v2 §2.1, §5 ledger): the page embeds
        report.py's OUTPUT (`ranked`) AND the stamped input (`by_region`) —
        the declared binding is no longer a receipt-only carrier, so the
        offline file check covers it too."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        assert 'id="tracebi-data-ranked"' in html
        assert 'id="tracebi-data-by_region"' in html

    def test_output_is_python_derived_and_windowed(self, tmp_path, model):
        """The embedded output carries the cumulative column report.py computed
        — data the input query never produced."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render({model.name: model}, str(out))
        html = out.read_text(encoding="utf-8")
        block = re.search(
            r'id="tracebi-data-ranked"[^>]*>(.*?)</script>', html, re.DOTALL)
        csv = json.loads(block.group(1))["csv"]
        assert "cumulative" in csv.splitlines()[0]

    def test_manifest_marks_output_unverifiable_input_reproducible(self, tmp_path, model):
        """M0 flip (report-architecture-v2 §2.1, §5 ledger): verifiability is
        per binding, not per package. The manifest records the query binding
        with no ``verifiable: false`` flag (green-eligible) beside the
        report.py output recorded verifiable=false — the escape hatch no
        longer flattens the declared binding to false."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out)).to_dict()
        by_name = {r["name"]: r for r in manifest["embedded_data"]}
        assert set(by_name) == {"by_region", "ranked"}
        assert by_name["by_region"].get("verifiable") is None     # query binding
        assert by_name["by_region"]["query_spec"] is not None     # replay-checkable
        assert by_name["by_region"]["model"] == model.name
        assert by_name["ranked"]["verifiable"] is False           # python-derived
        assert by_name["ranked"]["query_spec"] is None
        by_id = {s.get("id"): s for s in manifest["sections"]}
        assert by_id["by_region"].get("verifiable") is None       # reproducible carrier
        assert by_id["by_region"]["dataset_fingerprint"]          # query-stamped
        assert by_id["ranked"]["verifiable"] is False             # python-derived carrier

    def test_verify_file_passes_then_fails_on_output_tamper(self, tmp_path, model):
        """File integrity holds for python-derived data too: the untouched file
        is intact; editing the embedded output makes it TAMPERED."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out)).to_dict()
        html = out.read_text(encoding="utf-8")
        assert verify_file(html, manifest)["verdict"] == FILE_INTACT

        block = re.search(
            r'(id="tracebi-data-ranked"[^>]*>)(.*?)(</script>)', html, re.DOTALL)
        tampered = html[:block.start(2)] + block.group(2).replace(
            "700.75", "999999.0", 1) + html[block.end(2):]
        res = verify_file(tampered, manifest)
        assert res["verdict"] == FILE_ALTERED
        # M0 flip (report-architecture-v2 §5 ledger): the input binding is now
        # embedded too, so look the tampered output up by name.
        by_binding = {b["binding"]: b for b in res["bindings"]}
        assert by_binding["ranked"]["status"] == FILE_TAMPERED
        assert by_binding["by_region"]["status"] == FILE_MATCHES


class TestEscapeHatchHonesty:
    def test_output_classified_unverifiable_input_reproduces(self, tmp_path, model):
        """verify (query -> model): the input query reproduces; the
        python-derived output is UNVERIFIABLE, named as report.py output."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out)).to_dict()
        result = verify_manifest(manifest, {model.name: model})
        by_section = {s["section"]: s for s in result["sections"]}
        assert by_section["by_region"]["status"] == REPRODUCES
        assert by_section["ranked"]["status"] == UNVERIFIABLE
        assert by_section["ranked"]["python_derived"] is True
        assert "report.py" in by_section["ranked"]["detail"]

    def test_green_never_reads_as_verified(self, tmp_path, model):
        """Exit 0 (the input reproduced, nothing failed), but the verdict line
        explicitly says the python-derived output was not query-reproducible —
        so a clean exit never reads as 'these numbers were verified'."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out)).to_dict()
        result = verify_manifest(manifest, {model.name: model})
        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert result["python_derived"] == 1
        assert "python-derived" in result["verdict_detail"]
        assert "not query-reproducible" in result["verdict_detail"]

    def test_verifiable_false_forces_unverifiable_even_with_a_query_node(
            self, tmp_path, model, spec):
        """Defence in depth: a section flagged verifiable=false is UNVERIFIABLE
        by construction — even if its lineage happens to carry a query node, it
        must not read as reproduced."""
        sd = stamp(model, spec, name="ranked")   # a fully query-stamped dataset
        # Manifest section with a real query in lineage BUT flagged python-derived.
        section = {
            "id": "ranked", "section_type": "table",
            "dataset_fingerprint": sd.fingerprint,
            "dataset_lineage": sd.dataset.lineage_to_dict(),
            "verifiable": False,
        }
        manifest = {"report_name": "x", "schema_version": 1, "sections": [section]}
        result = verify_manifest(manifest, {model.name: model})
        assert result["sections"][0]["status"] == UNVERIFIABLE
        assert result["sections"][0]["python_derived"] is True


class TestPerBindingVerifiability:
    """M0 seam (report-architecture-v2 §2.1 "The per-binding verifiability
    seam", §5 ledger): a report.py beside a report.json no longer poisons the
    declarative bindings — a receipt STRENGTHENING. What was grey (or absent
    from the offline check) becomes green; python-derived outputs stay grey,
    hardcoded, never green."""

    def test_mixed_package_query_binding_reproduces_python_output_stays_grey(
            self, tmp_path, model):
        """The pre-pattern mixed package (one query binding + one report.py
        output): the query binding classifies REPRODUCES under replay, the
        python output stays UNVERIFIABLE, and the offline file check passes
        with BOTH bindings covered."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out)).to_dict()

        # Replay check (query -> model): per-binding truth.
        result = verify_manifest(manifest, {model.name: model})
        by_section = {s["section"]: s for s in result["sections"]}
        assert by_section["by_region"]["status"] == REPRODUCES
        assert by_section["ranked"]["status"] == UNVERIFIABLE
        assert by_section["ranked"]["python_derived"] is True

        # Offline check: the query binding's bytes are now embedded and
        # vouched for too — it was not even file-checkable before the seam.
        file_result = verify_file(out.read_text(encoding="utf-8"), manifest)
        assert file_result["verdict"] == FILE_INTACT
        assert file_result["summary"][FILE_MATCHES] == 2
        assert {b["binding"] for b in file_result["bindings"]} == {
            "by_region", "ranked"}

    def test_mixed_verdict_names_both_kinds(self, tmp_path, model):
        """The verdict for a mixed receipt names both kinds honestly: the
        checked-and-matching count AND the python-derived remainder
        (_verdict_fields already supports mixed — this pins it)."""
        pkg = _escape_hatch_package(tmp_path)
        out = tmp_path / "out.html"
        manifest = TemplatePackage(str(pkg)).render(
            {model.name: model}, str(out)).to_dict()
        result = verify_manifest(manifest, {model.name: model})
        assert result["verdict"] == REPRODUCES
        assert result["summary"][REPRODUCES] == 1
        assert result["summary"][UNVERIFIABLE] == 1
        assert "1 of 2 section(s) checked and matching" in result["verdict_detail"]
        assert "python-derived" in result["verdict_detail"]
        assert "not query-reproducible" in result["verdict_detail"]

class TestEscapeHatchContract:
    def test_missing_build_function_raises(self, tmp_path, model):
        pkg = _write_package(
            tmp_path, template=_OUTPUT_TEMPLATE, report_py="x = 1\n",
            dirname="nobuild")
        with pytest.raises(ValueError, match="must define a module-level build"):
            TemplatePackage(str(pkg)).render({model.name: model},
                                             str(tmp_path / "o.html"))

    def test_build_returning_non_dataframe_raises(self, tmp_path, model):
        pkg = _write_package(
            tmp_path, template=_OUTPUT_TEMPLATE,
            report_py="def build(inputs):\n    return {'ranked': 42}\n",
            dirname="badout")
        with pytest.raises(ValueError, match="not a DataFrame"):
            TemplatePackage(str(pkg)).render({model.name: model},
                                             str(tmp_path / "o.html"))

    def test_output_name_colliding_with_input_raises(self, tmp_path, model):
        pkg = _write_package(
            tmp_path, template=_OUTPUT_TEMPLATE,
            report_py="def build(inputs):\n    return {'by_region': inputs['by_region']}\n",
            dirname="collide")
        with pytest.raises(ValueError, match="collides with a stamped input"):
            TemplatePackage(str(pkg)).render({model.name: model},
                                             str(tmp_path / "o.html"))

    def test_stamp_frame_is_python_derived(self):
        """stamp_frame carries the canonical triple but no query/model, and its
        lineage node is marked python_derived."""
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        sd = stamp_frame(df, name="out")
        assert sd.query_spec is None and sd.model is None
        assert sd.fingerprint == fingerprint_triple(sd.triple)
        assert sd.dataset.lineage_to_dict()[-1]["metadata"]["python_derived"] is True


def test_preview_server_binds_loopback_not_all_interfaces(tmp_path, monkeypatch):
    """The unauthenticated preview server must bind 127.0.0.1, not '' (all
    interfaces) — otherwise a directory of rendered reports is published to
    anyone on the network, while the printed URL says localhost."""
    import http.server

    from tracebi.cli import _serve_file

    captured = {}

    class FakeServer:
        def __init__(self, address, handler):
            captured["address"] = address

        def serve_forever(self):
            raise KeyboardInterrupt   # exit the blocking loop at once

        def server_close(self):
            pass

    monkeypatch.setattr(http.server, "HTTPServer", FakeServer)
    html = tmp_path / "r.html"
    html.write_text("<p>x</p>")
    _serve_file(html, "r", 8099, open_browser=False)
    assert captured["address"][0] == "127.0.0.1"


def test_preview_srcdoc_does_not_double_decode(monkeypatch, tmp_path):
    """A cell escaped to &lt;script&gt; during render must not be decoded back
    into live markup when embedded in an iframe srcdoc: srcdoc is decoded once
    by the browser, so the content's '&' must be escaped too (double-decode
    XSS otherwise)."""
    import sys
    import types

    from tracebi.reports.html_renderer import HTMLRenderer

    captured = {}
    fake = types.ModuleType("IPython.display")
    fake.display = lambda x: captured.__setitem__("shown", x)
    fake.HTML = lambda s: captured.__setitem__("srcdoc", s) or s
    fake.IFrame = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", fake)

    r = HTMLRenderer()

    def fake_render(report, path, save_manifest=False):
        with open(path, "w", encoding="utf-8") as f:
            f.write("<body>&lt;script&gt;alert(1)&lt;/script&gt;</body>")

    monkeypatch.setattr(r, "render", fake_render)
    r.preview(None)
    s = captured["srcdoc"]
    # The already-escaped entity is DOUBLE-escaped, so one srcdoc-decode leaves
    # it as text (&lt;script&gt;), never a live <script>.
    assert "&amp;lt;script&amp;gt;" in s
    assert 'srcdoc="<body>&lt;script' not in s

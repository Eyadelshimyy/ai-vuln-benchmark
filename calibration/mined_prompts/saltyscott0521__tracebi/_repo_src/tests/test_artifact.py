"""
The artifact layer — figure extraction, id assignment, exploration strip.

figures.py is the ONE parser for data-tb-* markup (architecture v2 §2.1,
flaw 1): build validation, the exploration strip, and verify --file all
share it, so these tests are the trust tests for all three. Hostile and
malformed markup must fail LOUDLY — a figure silently missed here would
vanish from the build check and the offline check at once.
"""

import json

import pandas as pd
import pytest

from tracebi.reports.figures import (
    Figure,
    FigureError,
    assign_figure_ids,
    extract_figures,
    methodology_insertion,
    read_stage_meta,
    strip_stage,
)


PAGE = """<!doctype html>
<html><head><title>p</title>
<meta name="tracebi-stage" content="final">
</head><body>
<div class="tb-kpi" data-tb-figure="value" data-tb-binding="kpi"
     data-tb-cell="book_fv" id="fig-book"><span>x</span></div>
<div data-tb-figure="chart" data-tb-binding="bands" id="fig-bands"></div>
<table data-tb-figure="table" data-tb-binding="top10" id="fig-top10"></table>
<div data-tb-figure="custom" data-tb-binding="bands" id="fig-heat"></div>
<div data-tb-figure="value" data-tb-unverified
     data-tb-note="analyst estimate" id="fig-est">42</div>
<section data-tb-stage="exploration">
  <p>working note</p>
  <table data-tb-figure="table" data-tb-binding="nulls" id="fig-nulls"></table>
</section>
</body></html>"""


class TestExtract:
    def test_extracts_every_figure_in_document_order(self):
        figs = extract_figures(PAGE)
        assert [f.id for f in figs] == [
            "fig-book", "fig-bands", "fig-top10", "fig-heat", "fig-est",
            "fig-nulls",
        ]
        book = figs[0]
        assert book.kind == "value" and book.binding == "kpi"
        assert book.cell == "book_fv" and not book.unverified
        est = figs[4]
        assert est.unverified and est.binding is None
        assert est.note == "analyst estimate"

    def test_unknown_kind_is_refused(self):
        with pytest.raises(FigureError, match="unknown figure kind"):
            extract_figures('<div data-tb-figure="kpi" id="x"></div>')

    def test_binding_and_unverified_together_are_refused(self):
        with pytest.raises(FigureError, match="never both"):
            extract_figures(
                '<div data-tb-figure="value" data-tb-binding="b" '
                'data-tb-unverified id="x"></div>'
            )

    def test_duplicate_ids_are_refused(self):
        with pytest.raises(FigureError, match="duplicate figure id"):
            extract_figures(
                '<div data-tb-figure="chart" data-tb-binding="a" id="f"></div>'
                '<div data-tb-figure="chart" data-tb-binding="b" id="f"></div>'
            )

    def test_misnested_markup_fails_loudly(self):
        with pytest.raises(FigureError, match="mis-nested|closes nothing"):
            extract_figures("<div><section></div></section>")

    def test_stray_close_fails_loudly(self):
        with pytest.raises(FigureError, match="closes nothing"):
            extract_figures("</div><div data-tb-figure='chart' id='x'></div>")

    def test_uppercase_and_whitespace_attrs_still_parse(self):
        figs = extract_figures(
            '<DIV DATA-TB-FIGURE="chart"\n  data-tb-binding = "b"\n'
            '  ID="f1"></DIV>'
        )
        assert figs == [Figure(kind="chart", id="f1", binding="b", tag="div",
                               attrs=figs[0].attrs)]

    def test_figure_attr_hidden_in_text_or_comment_is_not_a_figure(self):
        # The classic regex traps: markup-looking bytes in places that are
        # not markup. A real tokenizer must not see figures here.
        figs = extract_figures(
            "<pre>&lt;div data-tb-figure=\"chart\" id=\"fake\"&gt;</pre>"
            "<!-- <div data-tb-figure='table' id='ghost'></div> -->"
            "<div data-tb-figure=\"chart\" data-tb-binding=\"b\" id=\"real\">"
            "</div>"
        )
        assert [f.id for f in figs] == ["real"]


class TestAssignIds:
    def test_missing_ids_are_assigned_with_warnings(self):
        html = ('<div data-tb-figure="chart" data-tb-binding="a"></div>'
                '<div data-tb-figure="table" data-tb-binding="b" id="mine">'
                '</div>')
        out, warnings = assign_figure_ids(html)
        figs = extract_figures(out)
        assert [f.id for f in figs] == ["fig-1", "mine"]
        assert len(warnings) == 1 and "fig-1" in warnings[0]

    def test_assignment_skips_taken_ids(self):
        html = ('<div data-tb-figure="chart" data-tb-binding="a" id="fig-1">'
                '</div><div data-tb-figure="chart" data-tb-binding="b"></div>')
        out, _ = assign_figure_ids(html)
        assert [f.id for f in extract_figures(out)] == ["fig-1", "fig-2"]

    def test_page_with_all_ids_is_returned_unchanged(self):
        out, warnings = assign_figure_ids(PAGE)
        assert out == PAGE and warnings == []


class TestStripStage:
    def test_exploration_blocks_are_removed_whole(self):
        out = strip_stage(PAGE)
        assert "working note" not in out
        assert "fig-nulls" not in out
        # Everything else survives byte-for-byte around the cut.
        assert "fig-book" in out and "fig-est" in out

    def test_nested_exploration_blocks_remove_as_one_outermost_range(self):
        html = ('<div>keep</div>'
                '<section data-tb-stage="exploration">outer'
                '<div data-tb-stage="exploration">inner</div>'
                '</section><p>after</p>')
        out = strip_stage(html)
        assert out == "<div>keep</div><p>after</p>"

    def test_unclosed_stage_block_fails_loudly(self):
        with pytest.raises(FigureError, match="never closed"):
            strip_stage('<section data-tb-stage="exploration"><p>x</p>')

    def test_unknown_stage_is_refused(self):
        with pytest.raises(FigureError, match="unknown stage"):
            strip_stage('<div data-tb-stage="draft"></div>')

    def test_void_element_cannot_be_staged(self):
        with pytest.raises(FigureError, match="void element"):
            strip_stage('<img data-tb-stage="exploration">')


class TestStageMeta:
    def test_reads_the_stage_meta(self):
        assert read_stage_meta(PAGE) == "final"

# ── the stated-methodology appendix ─────────────────────────────────────────
# About this report is on by default (auto footer, or the author's ONE
# data-tb-methodology container). The build appends plain-language About
# copy and the pipeline's STATED methodology after the author's own
# children. Prose, never a verified claim — no badge, no status.

# Deliberately markup-shaped prose: the appendix must render it inert.
TRANSFORM_NOTE = "dropped the 9 unkeyed rows & <b>counted</b> them"
CHECK_NOTE = "an empty fact means the pull failed"
MEASURE_NOTE = "Book fair value: sum of revenue over kept rows"

CONTAINER = ('<section data-tb-methodology><h2>Method</h2>'
             '<p id="authors-own">The author\'s prose stays first.</p>'
             '</section>')


def _noted_warehouse(tmp_path):
    from tracebi.connectors.duckdb_connector import DuckDBConnector
    from tracebi.contracts import contract

    wh_path = str(tmp_path / "warehouse.duckdb")
    wh = DuckDBConnector("wh", database=wh_path)
    wh.write(pd.DataFrame({"region_id": [1, 2], "region": ["NE", "SE"]}),
             "dim_region")
    wh.write(pd.DataFrame({"order_id": [1, 2, 3], "region_id": [1, 2, 1],
                           "revenue": [100.0, 250.0, 75.0]}), "fact_orders")
    wh.disconnect()
    with contract("orders", warehouse=wh_path, note=TRANSFORM_NOTE) as c:
        c.rows("fact_orders", at_least=1, note=CHECK_NOTE)
        c.unique("fact_orders", ["order_id"])
    return wh_path


def _noted_model(wh_path):
    from tracebi import DataModel
    from tracebi.connectors.duckdb_connector import DuckDBConnector

    m = DataModel("wh_model")
    m.add_connector(DuckDBConnector("wh", database=wh_path))
    m.add_table("fact_orders", connector="wh", source="fact_orders")
    m.add_table("dim_region", connector="wh", source="dim_region")
    m.add_relationship("o2r", left_table="fact_orders",
                       right_table="dim_region", left_key="region_id")
    m.add_dimension("dim_r", table_name="dim_region",
                    key_col="region_id", attributes=["region"])
    m.add_fact("f", table_name="fact_orders", measures=["revenue"],
               foreign_keys={"dim_r": "region_id"})
    m.add_measure("total", column="revenue", agg="sum",
                  description=MEASURE_NOTE)
    m.connect()
    return m


def _render_package(tmp_path, extra_body=""):
    from tracebi.reports.template_package import TemplatePackage

    wh_path = _noted_warehouse(tmp_path)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "report.json").write_text(json.dumps({
        "name": "pkg",
        "data": {"kpi": {"model": "wh_model",
                         "query": {"fact": "f", "measures": ["total"]}}},
    }))
    (pkg / "template.html").write_text(
        "<html><head><title>x</title></head><body>"
        '<div data-tb-figure="value" data-tb-binding="kpi" '
        'data-tb-cell="total" id="fig-kpi"></div>'
        + extra_body +
        "</body></html>"
    )
    out = tmp_path / "out.html"
    manifest = TemplatePackage(str(pkg)).render(
        {"wh_model": _noted_model(wh_path)}, str(out))
    return out.read_text(encoding="utf-8"), manifest.to_dict()


#: The appendix div's own marker — tracebi.css (inlined into every page's
#: head) also contains the bare string "tb-methodology", so tests anchor on
#: the class attribute.
APPENDIX = 'class="tb-methodology"'


class TestMethodologyAppendix:
    def test_appendix_is_appended_after_the_authors_own_children(
            self, tmp_path):
        page, _ = _render_package(tmp_path, CONTAINER)
        assert APPENDIX in page
        # The author's prose stays theirs, and stays FIRST.
        assert 'id="authors-own"' in page
        assert page.index('id="authors-own"') < page.index(APPENDIX)
        # The appendix still sits inside the container.
        assert page.index(APPENDIX) < page.index("</section>")

    def test_appendix_states_transform_check_and_measure_notes(
            self, tmp_path):
        page, _ = _render_package(tmp_path, CONTAINER)
        # Locked language: the transform STATES — never "verified".
        assert "the transform 'orders' states:" in page
        assert f"rows(fact_orders): {CHECK_NOTE}" in page
        assert f"measure 'total': {MEASURE_NOTE}" in page
        # No badge anywhere near it: stated methodology is never green.
        appendix = page[page.index(APPENDIX):page.index("</section>")]
        assert "tb-badge" not in appendix
        assert "verified" not in appendix.lower()

    def test_every_note_is_escaped(self, tmp_path):
        page, _ = _render_package(tmp_path, CONTAINER)
        assert "dropped the 9 unkeyed rows &amp; &lt;b&gt;counted&lt;/b&gt; them" in page
        assert TRANSFORM_NOTE not in page       # the raw markup never ships

    def test_manifest_records_what_stated_methodology_shipped(self, tmp_path):
        _, manifest = _render_package(tmp_path, CONTAINER)
        assert manifest["methodology"] == {
            "transform_notes": {"fact_orders": TRANSFORM_NOTE},
            "check_notes": [{"transform": "orders", "check": "rows",
                             "table": "fact_orders", "note": CHECK_NOTE}],
            "measure_notes": {"total": MEASURE_NOTE},
        }
        # A separate claim: it colors nothing, and the contracts block is
        # unchanged beside it.
        assert manifest["transform_contracts"]["fact_orders"]["status"] \
            == "satisfied"

    def test_without_the_container_about_footer_is_default_on(self, tmp_path):
        page, manifest = _render_package(tmp_path, extra_body="")
        assert "About this report" in page
        assert 'class="tb-about"' in page
        assert "This file carries a receipt" in page
        assert "claim the numbers are right" in page
        assert "Definitions from the model wh_model" in page
        assert "Built by unknown" not in page
        assert "Built on " in page
        receipt = page[page.index('class="tb-about-receipt"'):
                       page.index('class="tb-about-receipt"') + 400]
        for word in ("verified", "reproduces", "correct"):
            assert word not in receipt.lower()
        # Stated methodology notes ship through the default About home.
        assert APPENDIX in page
        assert manifest["methodology"]["transform_notes"]["fact_orders"] \
            == TRANSFORM_NOTE

    def test_two_containers_error_loudly(self, tmp_path):
        with pytest.raises(FigureError, match="ONE home"):
            _render_package(
                tmp_path,
                '<div data-tb-methodology></div><p data-tb-methodology></p>',
            )


class TestAboutFooter:
    def test_author_container_keeps_prose_first_and_still_gets_about(
            self, tmp_path):
        page, _ = _render_package(tmp_path, CONTAINER)
        assert 'id="authors-own"' in page
        assert page.index('id="authors-own"') < page.index(
            "This file carries a receipt")
        assert page.index("This file carries a receipt") < page.index(APPENDIX)
        # Author supplied the home — no second auto footer.
        assert page.count('class="tb-about"') == 0

    def test_report_author_is_named_when_set(self, tmp_path):
        from tracebi.reports.template_package import TemplatePackage

        wh_path = _noted_warehouse(tmp_path)
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(json.dumps({
            "name": "pkg",
            "author": "Ada Analyst",
            "data": {"kpi": {"model": "wh_model",
                             "query": {"fact": "f", "measures": ["total"]}}},
        }))
        (pkg / "template.html").write_text(
            "<html><head><title>x</title></head><body>"
            '<div data-tb-figure="value" data-tb-binding="kpi" '
            'data-tb-cell="total" id="fig-kpi"></div>'
            "</body></html>"
        )
        out = tmp_path / "out.html"
        TemplatePackage(str(pkg)).render(
            {"wh_model": _noted_model(wh_path)}, str(out))
        page = out.read_text(encoding="utf-8")
        assert "Built by Ada Analyst on " in page


# ── the embedded semantic contract ──────────────────────────────────────────
# The built page carries the model contract AS EXERCISED — the slice of the
# vocabulary the bindings' queries actually used, snapshotted at render and
# fingerprinted in the manifest. The slice, not the whole model: a report
# must not leak vocabulary it never used.

def _sc_model():
    from tracebi import DataModel, MemoryConnector

    t = pd.DataFrame({"region": ["NE", "SE", "MW"],
                      "flag": ["y", "n", "y"],
                      "revenue": [100.0, 250.0, 75.0],
                      "cost": [40.0, 100.0, 30.0]})
    t2 = pd.DataFrame({"region": ["NE", "SE"]})
    m = DataModel("sc_model")
    m.add_connector(MemoryConnector("sc_mem", tables={"t": t, "t2": t2}))
    m.add_table("t", connector="sc_mem", source="t")
    m.add_table("t2", connector="sc_mem", source="t2")
    m.add_dimension("dim_r", table_name="t", key_col="region",
                    attributes=["region"])
    m.add_dimension("dim_f", table_name="t", key_col="region",
                    attributes=["flag"])
    m.add_dimension("dim_ghost", table_name="t2", key_col="region",
                    attributes=["region"])         # never referenced
    m.add_fact("f", table_name="t", measures=["revenue", "cost"],
               foreign_keys={})
    m.add_measure("total", column="revenue", agg="sum")
    m.add_measure("cost_total", column="cost", agg="sum")
    m.add_measure("mark", ratio=("total", "cost_total"))
    m.add_measure("ghost", column="cost", agg="mean")   # never referenced
    m.connect()
    return m


def _sc_render(tmp_path):
    from tracebi.reports.template_package import TemplatePackage

    pkg = tmp_path / "scpkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "scpkg",
        "data": {
            "kpi": {"model": "sc_model",
                    "query": {"fact": "f", "measures": ["total", "mark"]}},
            "by_region": {"model": "sc_model",
                          "query": {"fact": "f",
                                    "measures": {"revenue": "sum"},
                                    "dimensions": ["dim_r.region"],
                                    "filters": {"dim_f.flag": "y"}}},
        },
    }))
    (pkg / "template.html").write_text(
        "<html><head><title>x</title></head><body>"
        '<div data-tb-figure="value" data-tb-binding="kpi" '
        'data-tb-cell="total" id="fig-kpi"></div>'
        "</body></html>")
    out = tmp_path / "sc_out.html"
    manifest = TemplatePackage(str(pkg)).render(
        {"sc_model": _sc_model()}, str(out))
    return out.read_text(encoding="utf-8"), manifest.to_dict()


class TestSemanticContract:
    def test_the_built_page_embeds_one_block_per_referenced_model(
            self, tmp_path):
        page, _ = _sc_render(tmp_path)
        assert 'id="tb-semantic-contract-sc_model"' in page

    def test_the_slice_holds_only_exercised_vocabulary(self, tmp_path):
        _, manifest = _sc_render(tmp_path)
        s = manifest["semantic_contract"]["sc_model"]["slice"]
        assert [f["name"] for f in s["facts"]] == ["f"]
        # Grouped AND filter-referenced dimensions are exercised…
        assert [d["name"] for d in s["dimensions"]] == ["dim_f", "dim_r"]
        # …and only measures the queries name. The never-referenced
        # measure, dimension, and its backing table must not leak.
        assert [m["name"] for m in s["measures"]] == ["mark", "total"]
        assert [t["name"] for t in s["tables"]] == ["t"]

    def test_ratio_measures_carry_their_formula(self, tmp_path):
        _, manifest = _sc_render(tmp_path)
        s = manifest["semantic_contract"]["sc_model"]["slice"]
        mark = next(m for m in s["measures"] if m["name"] == "mark")
        assert mark["kind"] == "ratio"
        assert mark["ratio"] == ["total", "cost_total"]

    def test_byte_stable_across_renders(self, tmp_path):
        _, m1 = _sc_render(tmp_path)
        _, m2 = _sc_render(tmp_path)
        assert (m1["semantic_contract"]["sc_model"]["sha256"]
                == m2["semantic_contract"]["sc_model"]["sha256"])

    def test_manifest_field_shape_and_presence(self, tmp_path):
        # Always present for a package with bindings — and a package
        # without bindings cannot exist (report.json requires them).
        _, manifest = _sc_render(tmp_path)
        assert set(manifest["semantic_contract"]) == {"sc_model"}
        rec = manifest["semantic_contract"]["sc_model"]
        assert set(rec) == {"slice", "sha256"}
        assert set(rec["slice"]) == {"model", "facts", "dimensions",
                                     "measures", "tables"}
        assert rec["slice"]["model"] == "sc_model"


class TestMethodologyContainerParsing:
    def test_insertion_point_is_the_containers_closing_tag(self):
        html = '<div>x</div><section data-tb-methodology><p>mine</p></section>'
        pos = methodology_insertion(html)
        assert html[pos:].startswith("</section>")

    def test_void_element_cannot_be_the_container(self):
        with pytest.raises(FigureError, match="void element"):
            methodology_insertion('<img data-tb-methodology>')

    def test_self_closing_container_is_refused(self):
        with pytest.raises(FigureError, match="closing tag"):
            methodology_insertion('<div data-tb-methodology/>')

    def test_unclosed_container_fails_loudly(self):
        with pytest.raises(FigureError, match="never closed"):
            methodology_insertion('<div data-tb-methodology><p>x</p>')


# ── the receipt block: phase-① and methodology entries ──────────────────────
# The tracebi-receipt block is the drawer's provenance feed. It duplicates
# manifest facts — the MANIFEST remains the receipt of record — so each
# entry is checked against the manifest field it mirrors.

class TestReceiptBlockJoins:
    RECEIPT_OPEN = '<script id="tracebi-receipt" type="application/json">'

    def _receipt(self, page):
        start = page.index(self.RECEIPT_OPEN)
        payload = page[start + len(self.RECEIPT_OPEN):
                       page.index("</script>", start)]
        return json.loads(payload)

    def test_contract_statuses_are_compact_and_mirror_the_manifest(
            self, tmp_path):
        page, manifest = _render_package(tmp_path, CONTAINER)
        receipt = self._receipt(page)
        # Status per table ONLY — the drawer names the status; the full
        # record (transform, checked_at, fingerprint, note) stays in the
        # manifest, and the block never carries what it did not record.
        assert receipt["transform_contracts"] == {
            table: rec["status"]
            for table, rec in manifest["transform_contracts"].items()
        }
        assert receipt["transform_contracts"]["fact_orders"] == "satisfied"
        assert receipt["methodology"] is True
        assert receipt["semantic_contract_models"] == ["wh_model"]

    def test_methodology_flag_is_true_when_notes_ship_via_default_about(
            self, tmp_path):
        # No author container — the default About home still carries stated
        # methodology notes, and the receipt drawer names that.
        page, manifest = _render_package(tmp_path, extra_body="")
        receipt = self._receipt(page)
        assert receipt["methodology"] is True
        assert "methodology" in manifest

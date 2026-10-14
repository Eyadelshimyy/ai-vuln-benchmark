"""
Verify v2 — the figure claims layer (manifest schema 2, architecture v2 §2.3).

Figures inherit their binding's replay status; an author-marked figure is
UNVERIFIED (distinct from python-derived UNVERIFIABLE); a figure naming a
binding the receipt doesn't carry is ERROR and fails; the offline check is
symmetric (missing-in-file AND unrecorded-in-manifest both fail); a stage
mismatch is FILE ALTERED; a snapshot is refused by name; --strict holds a
receipt to 100% green figures. A v1 manifest behaves exactly as before.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.reports.template_package import TemplatePackage
from tracebi.verify import (
    ERROR,
    REPRODUCES,
    UNEXPLAINED,
    UNVERIFIABLE,
    UNVERIFIED,
    verify_file,
    verify_manifest,
)


@pytest.fixture()
def v2_model():
    df = pd.DataFrame({"region": ["NE", "SE", "MW"],
                       "revenue": [100.0, 250.0, 75.0]})
    m = DataModel("v2_model")
    m.add_connector(MemoryConnector("v2_mem", tables={"t": df}))
    m.add_table("t", connector="v2_mem", source="t")
    m.add_dimension("dim_r", table_name="t", key_col="region",
                    attributes=["region"])
    m.add_fact("f", table_name="t", measures=["revenue"], foreign_keys={})
    m.add_measure("total", column="revenue", agg="sum")
    m.connect()
    return m


def _package(tmp_path, template_body: str, with_report_py: bool = False):
    pkg = tmp_path / "pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "report.json").write_text(json.dumps({
        "name": "pkg",
        "data": {
            "by_region": {"model": "v2_model",
                          "query": {"fact": "f", "measures": {"revenue": "sum"},
                                    "dimensions": ["dim_r.region"]}},
            "kpi": {"model": "v2_model",
                    "query": {"fact": "f", "measures": ["total"]}},
        },
    }))
    (pkg / "template.html").write_text(
        f"<html><head><title>x</title></head><body>{template_body}</body></html>"
    )
    if with_report_py:
        (pkg / "report.py").write_text(
            "def build(frames):\n"
            "    return {'ranked': frames['by_region'].head(2)}\n"
        )
    return pkg


_FULL_BODY = (
    '<div data-tb-figure="value" data-tb-binding="kpi" data-tb-cell="total" '
    'id="fig-kpi"></div>'
    '<table data-tb-figure="table" data-tb-binding="by_region" id="fig-tbl">'
    '</table>'
    '<div data-tb-figure="value" data-tb-unverified '
    'data-tb-note="analyst estimate" id="fig-est">42</div>'
)


def _render(tmp_path, model, body=_FULL_BODY, with_report_py=False):
    pkg = TemplatePackage(str(_package(tmp_path, body, with_report_py)))
    out = tmp_path / "out.html"
    manifest = pkg.render({"v2_model": model}, str(out)).to_dict()
    return out.read_text(encoding="utf-8"), manifest


class TestManifestV2Shape:
    def test_artifact_manifest_carries_figures_and_stage(self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        assert manifest["schema_version"] == 2
        assert manifest["stage"] == "final"
        by_id = {f["id"]: f for f in manifest["figures"]}
        assert by_id["fig-kpi"]["binding"] == "kpi"
        assert by_id["fig-kpi"]["cell"] == "total"
        assert by_id["fig-est"]["unverified"] is True

    def test_section_type_follows_a_single_figure_kind(self, v2_model, tmp_path):
        body = (
            '<div data-tb-figure="value" data-tb-binding="kpi" '
            'data-tb-cell="total" id="fig-kpi"></div>'
            '<div data-tb-figure="chart" data-tb-binding="by_region" '
            'data-tb-type="bar" data-tb-x="dim_r.region" data-tb-y="revenue" '
            'id="fig-chart"></div>'
        )
        _html, manifest = _render(tmp_path, v2_model, body=body)
        stored = {s["id"]: s["section_type"] for s in manifest["sections"]}
        assert stored == {"kpi": "table", "by_region": "table"}
        result = verify_manifest(manifest, {"v2_model": v2_model})
        got = {s["section"]: s["section_type"] for s in result["sections"]}
        assert got["kpi"] == "value"
        assert got["by_region"] == "chart"

        mixed = (
            '<div data-tb-figure="value" data-tb-binding="kpi" '
            'data-tb-cell="total" id="fig-kpi"></div>'
            '<table data-tb-figure="table" data-tb-binding="by_region" '
            'id="fig-tbl"></table>'
            '<div data-tb-figure="chart" data-tb-binding="by_region" '
            'data-tb-type="bar" data-tb-x="dim_r.region" data-tb-y="revenue" '
            'id="fig-chart"></div>'
        )
        _html, manifest = _render(tmp_path, v2_model, body=mixed)
        result = verify_manifest(manifest, {"v2_model": v2_model})
        got = {s["section"]: s["section_type"] for s in result["sections"]}
        assert got["kpi"] == "value"
        assert got["by_region"] == "table"

    def test_build_refuses_a_figure_with_no_binding_and_no_mark(self, v2_model, tmp_path):
        from tracebi.reports.figures import FigureError
        with pytest.raises(FigureError, match="no third\nstate|third .?state|no binding"):
            _render(tmp_path, v2_model,
                    body='<div data-tb-figure="chart" id="f"></div>')

    def test_build_refuses_an_unknown_binding_with_a_hint(self, v2_model, tmp_path):
        from tracebi.reports.figures import FigureError
        with pytest.raises(FigureError, match="by_regio"):
            _render(tmp_path, v2_model,
                    body='<div data-tb-figure="chart" '
                         'data-tb-binding="by_regio" id="f"></div>')

    def test_build_refuses_a_value_figure_over_a_multirow_binding(self, v2_model, tmp_path):
        from tracebi.reports.figures import FigureError
        with pytest.raises(FigureError, match="3 rows"):
            _render(tmp_path, v2_model,
                    body='<div data-tb-figure="value" '
                         'data-tb-binding="by_region" data-tb-cell="revenue" '
                         'id="f"></div>')

    def test_exploration_blocks_are_stripped_from_the_final_build(self, v2_model, tmp_path):
        html, manifest = _render(
            tmp_path, v2_model,
            body=_FULL_BODY + '<section data-tb-stage="exploration">'
                              '<p>scratch note</p></section>')
        assert "scratch note" not in html
        assert {f["id"] for f in manifest["figures"]} == {
            "fig-kpi", "fig-tbl", "fig-est"}


class TestFigureRollup:
    def test_figures_inherit_binding_status_and_marked_is_unverified(
            self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        result = verify_manifest(manifest, {"v2_model": v2_model})
        by_id = {f["figure"]: f["status"] for f in result["figures"]}
        assert by_id["fig-kpi"] == REPRODUCES
        assert by_id["fig-tbl"] == REPRODUCES
        assert by_id["fig-est"] == UNVERIFIED
        assert result["ok"]                        # unverified never fails…
        assert "unverified" in result["verdict_detail"].lower()  # …or hides

    def test_python_derived_figure_is_unverifiable_not_unverified(
            self, v2_model, tmp_path):
        body = _FULL_BODY + ('<table data-tb-figure="table" '
                             'data-tb-binding="ranked" id="fig-rank"></table>')
        _html, manifest = _render(tmp_path, v2_model, body=body,
                                  with_report_py=True)
        result = verify_manifest(manifest, {"v2_model": v2_model})
        by_id = {f["figure"]: f["status"] for f in result["figures"]}
        assert by_id["fig-rank"] == UNVERIFIABLE   # ran python, can't replay
        assert by_id["fig-est"] == UNVERIFIED      # nobody claimed anything

    def test_figure_naming_an_absent_binding_is_error_and_fails(
            self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        manifest["figures"].append(
            {"id": "fig-ghost", "kind": "table", "binding": "nope"})
        result = verify_manifest(manifest, {"v2_model": v2_model})
        by_id = {f["figure"]: f["status"] for f in result["figures"]}
        assert by_id["fig-ghost"] == ERROR
        assert not result["ok"] and result["exit_code"] == 1

    def test_strict_fails_a_receipt_with_any_non_green_figure(
            self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        loose = verify_manifest(manifest, {"v2_model": v2_model})
        strict = verify_manifest(manifest, {"v2_model": v2_model}, strict=True)
        assert loose["ok"]
        assert not strict["ok"] and "STRICT" in strict["verdict_detail"]

    def test_v1_manifest_result_shape_is_untouched(self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        manifest.pop("figures"); manifest.pop("stage")
        manifest["schema_version"] = 1
        result = verify_manifest(manifest, {"v2_model": v2_model})
        assert "figures" not in result and "figure_summary" not in result


class TestFileCheckV2:
    def test_intact_file_names_the_honest_limit(self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        result = verify_file(html, manifest)
        assert result["ok"]
        assert "scripting is not provable" in result["verdict_detail"]
        by_id = {f["figure"]: f["status"] for f in result["figures"]}
        assert by_id["fig-kpi"] == "figure_matches"

    def test_a_figure_added_after_render_fails_symmetrically(
            self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        html = html.replace(
            "</body>",
            '<div data-tb-figure="table" data-tb-binding="by_region" '
            'id="fig-added"></div></body>')
        result = verify_file(html, manifest)
        assert not result["ok"]
        assert any(f["status"] == "figure_unrecorded"
                   for f in result["figures"])

    def test_a_manifest_figure_missing_from_the_file_fails(
            self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        manifest["figures"].append(
            {"id": "fig-gone", "kind": "table", "binding": "by_region"})
        result = verify_file(html, manifest)
        assert not result["ok"]
        assert any(f["status"] == "figure_missing_in_file"
                   for f in result["figures"])

    def test_stage_mismatch_is_file_altered(self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        manifest["stage"] = "exploration"     # receipt claims a draft
        result = verify_file(html, manifest)
        assert not result["ok"]
        assert "stage meta" in result["verdict_detail"]

    def test_a_snapshot_is_refused_by_name(self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        html = html.replace('content="final"', 'content="exploration"')
        result = verify_file(html, manifest)
        assert not result["ok"]
        assert result["verdict"] == "refused_snapshot"
        assert "snapshot" in result["verdict_detail"].lower()


def _strip_semantic_blocks(html: str) -> str:
    import re
    return re.sub(
        r'<script id="tb-semantic-contract-[^"]*"[^>]*>.*?</script>\n?',
        "", html, flags=re.DOTALL)


class TestSemanticContractFileCheck:
    def test_intact_page_contract_matches(self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        result = verify_file(html, manifest)
        assert result["ok"]
        assert [r["status"] for r in result["semantic_contract"]] == ["matches"]

    def test_tampering_the_embedded_contract_is_file_altered(
            self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        tampered = html.replace('"name": "total"', '"name": "grand_total"')
        assert tampered != html            # the edit landed inside the block
        result = verify_file(tampered, manifest)
        assert not result["ok"] and result["verdict"] == "file_altered"
        assert [r["status"] for r in result["semantic_contract"]] == ["tampered"]
        # The data blocks themselves still check out — the detail says so.
        assert all(b["status"] == "matches" for b in result["bindings"])
        assert "semantic contract" in result["verdict_detail"]

    def test_deleting_the_block_the_manifest_records_fails(
            self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        result = verify_file(_strip_semantic_blocks(html), manifest)
        assert not result["ok"]
        assert [r["status"] for r in result["semantic_contract"]] \
            == ["missing_in_file"]

    def test_a_block_the_manifest_does_not_record_fails_symmetrically(
            self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        manifest.pop("semantic_contract")
        result = verify_file(html, manifest)
        assert not result["ok"]
        assert [r["status"] for r in result["semantic_contract"]] \
            == ["unrecorded_in_manifest"]

    def test_older_artifacts_without_the_field_verify_exactly_as_before(
            self, v2_model, tmp_path):
        html, manifest = _render(tmp_path, v2_model)
        manifest.pop("semantic_contract")      # an older render carries
        html = _strip_semantic_blocks(html)    # neither side
        result = verify_file(html, manifest)
        assert result["ok"] and result["verdict"] == "file_intact"
        assert "semantic_contract" not in result


class TestSemanticContractDiagnosis:
    def test_model_shaped_failure_names_the_changed_measure(
            self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        # Re-declare the measure between render and verify.
        v2_model.add_measure("total", column="revenue", agg="mean")
        result = verify_manifest(manifest, {"v2_model": v2_model})
        kpi = next(s for s in result["sections"] if s["section"] == "kpi")
        assert kpi["status"] == UNEXPLAINED     # status decided as today
        assert ("measure 'total' was sum('revenue'), now mean('revenue')"
                in kpi["detail"])

    def test_diagnosis_sharpens_detail_but_never_changes_a_status(
            self, v2_model, tmp_path):
        _html, manifest = _render(tmp_path, v2_model)
        v2_model.add_measure("total", column="revenue", agg="mean")
        with_sc = verify_manifest(manifest, {"v2_model": v2_model})
        bare = dict(manifest)
        bare.pop("semantic_contract")
        without_sc = verify_manifest(bare, {"v2_model": v2_model})
        assert ([s["status"] for s in with_sc["sections"]]
                == [s["status"] for s in without_sc["sections"]])
        assert with_sc["verdict"] == without_sc["verdict"]
        assert with_sc["exit_code"] == without_sc["exit_code"]


class TestSnapshot:
    def test_snapshot_keeps_exploration_banners_appends_code_and_writes_no_manifest(
            self, v2_model, tmp_path):
        body = _FULL_BODY + ('<section data-tb-stage="exploration">'
                             '<p>working note kept</p></section>')
        pkg = TemplatePackage(str(_package(tmp_path, body,
                                           with_report_py=True)))
        out = tmp_path / "snap.html"
        pkg.snapshot({"v2_model": v2_model}, str(out))
        html = out.read_text(encoding="utf-8")

        assert "working note kept" in html            # exploration KEPT
        assert "EXPLORATION SNAPSHOT" in html         # persistent banner
        assert "Code appendix" in html                # reviewer reads the code
        assert "def build(frames):" in html
        assert not (tmp_path / "snap.html.manifest.json").exists(), (
            "a snapshot must carry NO manifest — no draft receipt may exist"
        )
        # And the file refuses verification by name, against any manifest.
        _final_html, manifest = _render(tmp_path, v2_model)
        result = verify_file(html, manifest)
        assert result["verdict"] == "refused_snapshot"

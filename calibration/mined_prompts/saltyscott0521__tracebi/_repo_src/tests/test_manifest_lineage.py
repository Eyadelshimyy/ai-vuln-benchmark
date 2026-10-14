"""
Tests for manifest lineage capture on custom section types.

The audit manifest must record dataset lineage for EVERY section that
carries a dataset — including custom/user-defined section types with a
plain-string section_type — and the built-in sections' manifest output
must remain exactly as it was (regression pins below).
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pytest

from tracebi.model.dataset import DataSet, LineageNode
from tracebi.reports.report import (
    Report, ReportSection, RowSection, TableSection, ChartSection,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_ds(name: str = "test") -> DataSet:
    df = pd.DataFrame({
        "region":  ["North", "South", "East"],
        "orders":  [100, 85, 120],
        "revenue": [5000.0, 4250.0, 6100.0],
    })
    node = LineageNode(
        operation="load", description=f"Load {name}",
        connector={"connector_name": "test", "connector_type": "CSV"},
        source=f"{name}.csv",
    )
    return DataSet(df=df, name=name, lineage=[node])


@dataclass
class MapSection(ReportSection):
    """A custom section type: string section_type, carries a dataset."""
    dataset: Optional[DataSet] = None

    def __post_init__(self):
        self.section_type = "map"


@dataclass
class BadgeSection(ReportSection):
    """A custom section type with no dataset field."""
    label: str = ""

    def __post_init__(self):
        self.section_type = "badge"


# ─────────────────────────────────────────────
# Custom sections in the manifest
# ─────────────────────────────────────────────

class TestCustomSectionManifest:

    def test_custom_section_with_dataset_records_lineage(self):
        ds = make_ds("points").filter("orders > 90", description="High volume")
        section = MapSection(title="Map", dataset=ds)
        d = section.to_manifest_dict()
        assert d["section_type"] == "map"
        assert d["dataset_name"] == "points"
        assert d["dataset_shape"] == list(ds.shape)
        assert d["dataset_fingerprint"] == ds.fingerprint()
        assert d["dataset_lineage"] == ds.lineage_to_dict()
        assert len(d["dataset_lineage"]) > 0

    def test_custom_section_with_dataset_via_report_manifest(self):
        ds = make_ds("points")
        manifest = (
            Report("Custom Report")
            .add(MapSection(title="Map", dataset=ds))
            .build_manifest("html", "/tmp/custom.html")
        )
        entry = manifest.sections[0]
        for key in ("dataset_name", "dataset_shape",
                    "dataset_lineage", "dataset_fingerprint"):
            assert key in entry

    def test_custom_section_without_dataset(self):
        section = BadgeSection(title="Badge", label="OK")
        d = section.to_manifest_dict()
        assert d["section_type"] == "badge"
        assert "dataset_name" not in d
        assert "dataset_shape" not in d
        assert "dataset_lineage" not in d
        assert "dataset_fingerprint" not in d

# ─────────────────────────────────────────────
# Built-in sections: regression pins
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# describe() with custom string section types
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Review-pass fixes (adversarial findings)
# ─────────────────────────────────────────────

@dataclass
class ForeignPayloadSection(ReportSection):
    """A custom section whose 'dataset' field holds a non-DataSet."""
    dataset: object = None

    def __post_init__(self):
        self.section_type = "foreign"


class TestForeignDatasetPayloads:
    """A field *named* dataset is not necessarily a DataSet. Foreign types
    must not crash the manifest — they are simply not lineage-bearing."""

    def test_raw_dataframe_payload_omits_keys_no_crash(self):
        s = ForeignPayloadSection(title="df", dataset=pd.DataFrame({"a": [1]}))
        d = s.to_manifest_dict()
        assert "dataset_fingerprint" not in d
        assert "dataset_lineage" not in d

    def test_string_payload_omits_keys_no_crash(self):
        d = ForeignPayloadSection(title="s", dataset="not-a-dataset").to_manifest_dict()
        assert "dataset_fingerprint" not in d

    def test_render_with_foreign_payload_still_builds_manifest(self):
        report = Report(name="foreign")
        report.add(ForeignPayloadSection(title="odd",
                                         dataset=pd.DataFrame({"a": [1]})))
        report.add(TableSection(title="real", dataset=make_ds()))
        m = report.build_manifest("html", "/dev/null").to_dict()
        assert len(m["sections"]) == 2
        assert m["sections"][1]["dataset_fingerprint"]


class TestEmptyDatasetRecordsLineage:
    """A present-but-0-row DataSet is data with provenance, not absence of
    data. The old truthiness guard silently skipped it (DataSet defines
    __len__); the is-not-None semantics are pinned here."""

    def test_empty_dataset_emits_all_four_keys(self):
        empty = DataSet(df=pd.DataFrame({"region": [], "revenue": []}),
                        name="empty")
        d = TableSection(title="t", dataset=empty).to_manifest_dict()
        assert d["dataset_name"] == "empty"
        assert d["dataset_shape"] == [0, 2]
        assert d["dataset_fingerprint"] == empty.fingerprint()

    def test_empty_dataset_on_custom_section(self):
        empty = DataSet(df=pd.DataFrame({"a": []}), name="empty")
        d = MapSection(title="m", dataset=empty).to_manifest_dict()
        assert d["dataset_shape"] == [0, 1]


@dataclass
class TabsSection(ReportSection):
    """A custom container: children live in `panes`, wrapped in tuples."""
    panes: list = field(default_factory=list)   # [(label, ReportSection)]

    def __post_init__(self):
        self.section_type = "tabs"


@dataclass
class PanelSection(ReportSection):
    """A custom container keyed by name rather than ordered."""
    panels: dict = field(default_factory=dict)

    def __post_init__(self):
        self.section_type = "panels"


@dataclass(eq=False)
class HashablePanel(ReportSection):
    """A section written with identity equality, so it can be a set member or
    a dict key — which a plain @dataclass section (``__hash__ = None``) cannot."""
    dataset: Optional[DataSet] = None
    __hash__ = object.__hash__

    def __post_init__(self):
        self.section_type = "hashable_panel"


def render_tabs(section) -> str:
    """A project's own renderer for its own container — public API only."""
    return "".join(f"<div>{label}{t.dataset.to_pandas().to_html(index=False)}</div>"
                   for label, t in section.panes)


def all_fingerprints(sections) -> set:
    """Every dataset_fingerprint anywhere in a manifest's section tree."""
    found = set()
    for s in sections or []:
        if s.get("dataset_fingerprint"):
            found.add(s["dataset_fingerprint"])
        found |= all_fingerprints(s.get("sections") or [])
    return found


def distinct_ds(i: int) -> DataSet:
    """A dataset whose *content* is unique to *i*.

    fingerprint() is derived from content, so datasets that merely differ by
    name collapse to one fingerprint — and a coverage assertion over them
    would then hold even if every nested section were dropped.
    """
    node = LineageNode(
        operation="load", description=f"Load d{i}",
        connector={"connector_name": "test", "connector_type": "CSV"},
        source=f"d{i}.csv",
    )
    return DataSet(df=pd.DataFrame({"region": ["North"], "orders": [100 + i]}),
                   name=f"d{i}", lineage=[node])


class TestContainerSectionsCannotForfeitLineage:
    """A page of real figures must never produce a manifest with nothing in
    it. Custom containers are the case that used to: only RowSection knew how
    to descend, so any other container dropped every dataset it held and
    `tracebi verify` then passed the artifact as having no data at all."""

    def test_rendered_figure_is_covered_by_a_manifest_fingerprint(self, tmp_path):
        from tracebi.reports.html_renderer import HTMLRenderer

        df = pd.DataFrame({"cut": ["North"], "market_value": [523044.32]})
        ds = DataSet(df=df, name="holdings", lineage=[LineageNode(
            operation="load", description="Load holdings",
            connector={"connector_name": "t", "connector_type": "CSV"},
            source="holdings.csv")])
        report = Report("Holdings Review").add(TabsSection(
            title="Holdings by cut",
            panes=[("By region", TableSection(dataset=ds))],
        ))

        out = tmp_path / "holdings.html"
        HTMLRenderer(section_renderers={"tabs": render_tabs}).render(report, str(out))
        html = out.read_text()
        manifest = json.loads((tmp_path / "holdings.html.manifest.json").read_text())

        assert "523044.32" in html, "the figure under test must reach the page"
        fps = all_fingerprints(manifest["sections"])
        assert fps, "a report whose only data sits in a custom container " \
                    "produced a manifest with no fingerprint at all"
        assert ds.fingerprint() in fps, (
            "the number rendered in the HTML is not covered by any fingerprint "
            "in the manifest"
        )
        # …and the same descent must feed the on-page lineage appendix, so the
        # manifest and what a reader sees cannot disagree about one report.
        assert "holdings.csv" in html, (
            "the nested dataset is missing from the HTML lineage appendix"
        )

    def test_every_dataset_in_the_report_reaches_the_manifest(self):
        datasets = [distinct_ds(i) for i in range(5)]
        assert len({d.fingerprint() for d in datasets}) == 5, \
            "the datasets must be distinguishable or this test proves nothing"
        report = (
            Report("Mixed")
            .add(TableSection(title="flat", dataset=datasets[0]))
            .add(RowSection(sections=[
                TableSection(dataset=datasets[1]),
                RowSection(sections=[TableSection(dataset=datasets[2])]),
            ]))
            .add(TabsSection(panes=[("a", TableSection(dataset=datasets[3]))]))
            .add(PanelSection(panels={"left": MapSection(dataset=datasets[4])}))
        )
        m = report.build_manifest("html", "/dev/null").to_dict()
        assert all_fingerprints(m["sections"]) == {d.fingerprint() for d in datasets}

    def test_a_child_pointing_back_at_its_parent_still_renders(self, tmp_path):
        """A back-pointer is not containment. A built-in RowSection whose
        children know their parent renders on main; it must keep rendering,
        and it must still get a manifest."""
        from tracebi.reports.html_renderer import HTMLRenderer

        ds = distinct_ds(0)
        row = RowSection(title="side by side")
        kid = TableSection(dataset=ds)
        kid.parent = row
        row.sections = [kid]

        out = tmp_path / "back.html"
        HTMLRenderer().render(Report("R").add(row), str(out))

        assert out.exists()
        manifest = json.loads((tmp_path / "back.html.manifest.json").read_text())
        assert all_fingerprints(manifest["sections"]) == {ds.fingerprint()}

    def test_self_containing_section_terminates_and_keeps_its_other_children(self):
        ds = distinct_ds(1)
        loop = TabsSection(title="loop")
        loop.panes = [("self", loop), ("real", TableSection(dataset=ds))]
        d = loop.to_manifest_dict()
        assert all_fingerprints([d]) == {ds.fingerprint()}
        assert len(d["sections"]) == 1, "the self-edge must not be serialised"

    def test_self_referential_container_does_not_recurse_forever(self):
        ds = distinct_ds(2)
        panes: list = []
        panes.append(panes)
        panes.append(TableSection(dataset=ds))
        d = TabsSection(panes=panes).to_manifest_dict()
        assert all_fingerprints([d]) == {ds.fingerprint()}

    def test_same_section_twice_is_a_dag_not_a_cycle(self):
        shared = TableSection(dataset=make_ds("shared"))
        d = TabsSection(panes=[("a", shared), ("b", shared)]).to_manifest_dict()
        assert len(d["sections"]) == 2

    def test_slots_container_does_not_lose_its_children(self):
        """`@dataclass(slots=True)` keeps field values out of __dict__ — a
        container written that way must still yield its lineage."""
        @dataclass(slots=True)
        class SlottedTabs(ReportSection):
            panes: list = field(default_factory=list)

            def __post_init__(self):
                self.section_type = "slotted"

        ds = make_ds("slotted")
        d = SlottedTabs(panes=[TableSection(dataset=ds)]).to_manifest_dict()
        assert all_fingerprints([d]) == {ds.fingerprint()}

    def test_children_found_in_a_deque_a_frozenset_and_a_dict_key(self):
        """list/tuple/dict-value are not the only ways a container holds its
        children, and a shape this code does not recognise loses lineage
        silently — the exact failure being fixed."""
        from collections import deque

        a, b, c = distinct_ds(3), distinct_ds(4), distinct_ds(5)
        assert all_fingerprints([TabsSection(
            panes=deque([TableSection(dataset=a)])).to_manifest_dict()]) == {a.fingerprint()}
        assert all_fingerprints([TabsSection(
            panes=frozenset({HashablePanel(dataset=b)})).to_manifest_dict()]) == {b.fingerprint()}
        assert all_fingerprints([PanelSection(
            panels={HashablePanel(dataset=c): "label"}).to_manifest_dict()]) == {c.fingerprint()}

    def test_a_generator_of_sections_is_left_unconsumed(self):
        """Building a receipt must never empty the thing it describes, so an
        iterator is not treated as a container. Anyone broadening the search
        to plain Iterable would silently blank the section's own render."""
        panes = iter([TableSection(dataset=distinct_ds(6))])
        TabsSection(panes=panes).to_manifest_dict()
        assert len(list(panes)) == 1, "manifest building consumed the section's children"

    def test_non_section_payload_in_a_sections_attribute_is_ignored(self):
        d = TabsSection(title="odd", panes=["a label", 3, None]).to_manifest_dict()
        assert "sections" not in d

    def test_a_manifest_failure_leaves_no_unauditable_artifact(self, tmp_path):
        """The receipt is built before the artifact: an artifact on disk that
        nothing can audit is the state this whole mechanism exists to
        prevent, so it must not be reachable by failing halfway."""
        from tracebi.reports.excel_renderer import ExcelRenderer
        from tracebi.reports.html_renderer import HTMLRenderer

        @dataclass
        class UnserialisableSection(TableSection):
            def to_manifest_dict(self):
                raise RuntimeError("manifest boom")

        report = Report("R").add(UnserialisableSection(dataset=make_ds("x")))
        for renderer, name in ((HTMLRenderer(), "r.html"), (ExcelRenderer(), "r.xlsx")):
            out = tmp_path / name
            with pytest.raises(RuntimeError, match="manifest boom"):
                renderer.render(report, str(out))
            assert not out.exists(), f"{name} was written despite having no receipt"


class TestSpecExportToleratesCustomSections:
    """from_report() is best-effort by design; a custom string section_type
    must export (as its string), not crash with AttributeError."""

    def test_from_report_with_custom_section_does_not_raise(self):
        from tracebi.spec import ReportSpec

        report = Report(name="mixed")
        report.add(MapSection(title="map", dataset=make_ds()))
        report.add(TableSection(title="tbl", dataset=make_ds()))
        rspec = ReportSpec.from_report(report)
        types = [s.get("type") for s in rspec.to_dict()["sections"]]
        assert "map" in types


# ─────────────────────────────────────────────
# The one query-node recovery rule (last_query_node)
# ─────────────────────────────────────────────

def query_node(model: str) -> LineageNode:
    """A lineage node shaped like the one ``DataModel.execute`` stamps."""
    return LineageNode(
        operation="query",
        description=f"Query {model}",
        metadata={"model": model,
                  "query_spec": {"fact": f"fact_{model}",
                                 "measures": ["revenue"]}},
    )


class TestLastQueryNode:
    """The shared rule itself: last node whose metadata carries a query_spec."""

    def test_last_node_wins_with_multiple_query_nodes(self):
        from tracebi.model.dataset import last_query_node

        lineage = [query_node("first").to_dict(), query_node("second").to_dict()]
        node = last_query_node(lineage)
        assert node is lineage[-1]          # identity, not a copy
        assert node["metadata"]["model"] == "second"

    def test_no_query_node_returns_none(self):
        from tracebi.model.dataset import last_query_node

        assert last_query_node(make_ds().lineage_to_dict()) is None
        assert last_query_node([]) is None
        assert last_query_node(None) is None

    def test_non_dict_nodes_and_metadata_are_skipped(self):
        from tracebi.model.dataset import last_query_node

        target = query_node("m").to_dict()
        lineage = [target, "junk", {"metadata": None},
                   {"metadata": {"query_spec": None}}]
        assert last_query_node(lineage) is target


class TestRewiredSitesUnchanged:
    """One regression per call site rewired onto last_query_node."""

    def test_stamp_dataset_recovers_last_query_metadata(self):
        """embed._query_metadata: stamping reads the LAST query node."""
        from tracebi.reports.embed import stamp_dataset

        base = make_ds("orders")
        ds = DataSet(df=base.to_pandas(), name="orders",
                     lineage=base.lineage + [query_node("first"),
                                             query_node("second")])
        stamped = stamp_dataset(ds)
        assert stamped.model == "second"
        assert stamped.query_spec["fact"] == "fact_second"

    def test_stamp_dataset_without_query_node_stamps_none(self):
        from tracebi.reports.embed import stamp_dataset

        stamped = stamp_dataset(make_ds("adhoc"))
        assert stamped.model is None
        assert stamped.query_spec is None

    def test_verify_section_classifies_from_last_query_node(self):
        """verify._query_node: model/query_spec come from the LAST query node,
        and the node-identity check (transformed after the query) still holds."""
        from tracebi.verify import ERROR, UNVERIFIABLE, _verify_section

        lineage = [query_node("first").to_dict(), query_node("ghost").to_dict()]
        section = {"section_type": "table", "dataset_fingerprint": "f" * 64,
                   "dataset_lineage": lineage}
        out = _verify_section(section, models={}, label="s1")
        assert out["status"] == ERROR                # model 'ghost' not found
        assert out["model"] == "ghost"
        assert out["query_spec"]["fact"] == "fact_ghost"

        # No query node at all → unverifiable, same wording as before.
        section["dataset_lineage"] = make_ds().lineage_to_dict()
        out = _verify_section(section, models={}, label="s1")
        assert out["status"] == UNVERIFIABLE
        assert "no recorded query" in out["detail"]

        # Transformed after the query → the query node is not last.
        section["dataset_lineage"] = lineage + [
            LineageNode(operation="filter", description="post-query").to_dict()]
        out = _verify_section(section, models={}, label="s1")
        assert out["status"] == UNVERIFIABLE
        assert "transformed after" in out["detail"]

    def test_spec_data_ref_exports_last_query(self):
        """spec._data_ref_of: the exported data ref is the LAST query node's."""
        from tracebi.spec import section_to_dict

        base = make_ds("orders")
        ds = DataSet(df=base.to_pandas(), name="orders",
                     lineage=base.lineage + [query_node("first"),
                                             query_node("second")])
        d = section_to_dict(TableSection(title="t", dataset=ds))
        assert d["data"]["model"] == "second"
        assert d["data"]["query"]["fact"] == "fact_second"

        # Ad-hoc data (no query node) exports no data ref, as before.
        assert "data" not in section_to_dict(TableSection(dataset=make_ds()))

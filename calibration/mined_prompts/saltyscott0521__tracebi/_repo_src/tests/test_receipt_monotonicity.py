"""
The receipt-monotonicity harness (architecture v2 §5).

The reshape's contract applied to itself: a change may only ever GROW the set
of fingerprinted, verifiable claims a rendered report's receipt carries —
never shrink it, and never demote a green-eligible claim to grey. The
baselines below are the PRE-M0 claim sets; every milestone from M0 on must
produce a superset. If an edit makes one of these assertions fail, that edit
silently weakened a receipt — which is exactly what this file exists to make
loud.

Baselines are hard-coded on purpose. They may be edited only to GROW (a new
milestone adds claims); an edit that removes or demotes a baseline claim is
the defect this harness exists to catch, not a test to update.
"""

import json

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.spec import ReportSpec


@pytest.fixture()
def mono_model():
    orders = pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 5, 6],
        "customer_id": [1, 2, 1, 3, 2, 1],
        "revenue":     [100.0, 250.0, 75.0, 300.0, 125.0, 50.0],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "region":      ["NE", "SE", "MW"],
    })
    m = DataModel("mono_model")
    m.add_connector(MemoryConnector("mono_mem",
                                    tables={"orders": orders,
                                            "customers": customers}))
    m.add_table("orders", connector="mono_mem", source="orders")
    m.add_table("customers", connector="mono_mem", source="customers")
    m.add_relationship("o2c", left_table="orders", right_table="customers",
                       left_key="customer_id")
    m.add_dimension("dim_customer", table_name="customers",
                    key_col="customer_id", attributes=["region"])
    m.add_fact("fact_orders", table_name="orders", measures=["revenue"],
               foreign_keys={"dim_customer": "customer_id"})
    m.add_measure("total_revenue", column="revenue", agg="sum")
    m.connect()
    return m


def _spec() -> dict:
    """A spec exercising every data-bearing section kind."""
    data = {"model": "mono_model",
            "query": {"fact": "fact_orders", "measures": ["total_revenue"],
                      "dimensions": ["dim_customer.region"]}}
    kpi_data = {"model": "mono_model",
                "query": {"fact": "fact_orders",
                          "measures": ["total_revenue"]}}
    return {
        "name": "Mono Probe",
        "sections": [
            {"type": "metrics", "title": "KPIs",
             "data": kpi_data,
             "metrics": [{"label": "Revenue", "value": "total_revenue"}]},
            {"type": "chart", "title": "By region", "chart_type": "bar",
             "x": "dim_customer.region", "y": "total_revenue", "data": data},
            {"type": "table", "title": "Regions", "data": data},
        ],
    }


def _fingerprinted_sections(manifest: dict) -> set[str]:
    """Titles of manifest sections carrying a dataset fingerprint."""
    return {
        s.get("title") or s.get("section_type")
        for s in manifest.get("sections", [])
        if s.get("dataset_fingerprint")
    }


class TestSpecLaneMonotonicity:
    # PRE-M0 baseline: chart and table sections were fingerprinted; the
    # metrics section was NOT (field-notes finding #1 — its receipt was the
    # M0 growth). Grow-only.
    _BASELINE_FINGERPRINTED = {"By region", "Regions"}

    def test_fingerprinted_claims_are_a_superset_of_the_baseline(self, mono_model):
        spec = ReportSpec.from_dict(_spec())
        report = spec.build({"mono_model": mono_model})
        manifest = report.build_manifest("html", "(harness)").to_dict()
        claims = _fingerprinted_sections(manifest)
        missing = self._BASELINE_FINGERPRINTED - claims
        assert not missing, (
            f"receipt WEAKENED: baseline claims lost their fingerprint: "
            f"{sorted(missing)}"
        )
        # The M0 growth itself, pinned: the KPI strip is now a claim too.
        assert "KPIs" in claims, (
            "the metrics section lost the receipt M0 gave it (finding #1 "
            "regression)"
        )


class TestPackageLaneMonotonicity:
    # PRE-M0 baseline for a mixed package (query binding + report.py output):
    # ONLY the python output was embedded, and everything readable was
    # verifiable:false — zero green-eligible claims. M0's per-binding seam
    # made the query binding an embedded, green-eligible claim. Grow-only.
    _BASELINE_GREEN = set()             # pre-M0: nothing green in a mixed pkg
    _BASELINE_EMBEDDED = {"ranked"}     # pre-M0: only the python output

    def _mixed_package(self, tmp_path):
        pkg = tmp_path / "mono_pkg"
        pkg.mkdir()
        (pkg / "report.json").write_text(json.dumps({
            "name": "mono_pkg",
            "data": {"by_region": {
                "model": "mono_model",
                "query": {"fact": "fact_orders",
                          "measures": ["total_revenue"],
                          "dimensions": ["dim_customer.region"]}}},
        }))
        (pkg / "template.html").write_text(
            "<!doctype html><html><head><title>p</title></head>"
            "<body><div id='x'></div></body></html>"
        )
        (pkg / "report.py").write_text(
            "inputs = ['by_region']\n"
            "def build(frames):\n"
            "    df = frames['by_region'].sort_values('total_revenue',\n"
            "                                         ascending=False)\n"
            "    return {'ranked': df.head(2)}\n"
        )
        return pkg

    def test_green_claims_only_grow(self, mono_model, tmp_path):
        from tracebi.reports.template_package import TemplatePackage

        pkg = TemplatePackage(str(self._mixed_package(tmp_path)))
        out = tmp_path / "out.html"
        manifest = pkg.render({"mono_model": mono_model}, str(out)).to_dict()
        records = {r["name"]: r for r in manifest["embedded_data"]}

        # Nothing embedded before may vanish.
        missing = self._BASELINE_EMBEDDED - set(records)
        assert not missing, f"receipt WEAKENED: embedded claims lost: {missing}"

        # Green-eligible = embedded without a verifiable:false brand.
        green = {n for n, r in records.items()
                 if r.get("verifiable") is not False}
        demoted = self._BASELINE_GREEN - green
        assert not demoted, f"receipt WEAKENED: green claims demoted: {demoted}"

        # The M0 growth itself, pinned: the query binding is green-eligible,
        # and the escape hatch stays branded — never green (manifesto rule).
        assert "by_region" in green, (
            "per-binding verifiability regressed: a declarative query "
            "binding reads grey because a report.py exists beside it"
        )
        assert records["ranked"].get("verifiable") is False, (
            "the escape hatch's never-green brand disappeared"
        )

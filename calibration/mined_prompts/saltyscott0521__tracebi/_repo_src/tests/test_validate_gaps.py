"""
Validation-before-execution gaps (ROADMAP "Now" item 2).

"Validation before execution" is the keystone claim, and it used to hold
only for section structure, fact names, named measures, and dimension
names. The typo classes an agent actually produces — a bad filter column,
a bad aggregation name, a bad dimension attribute, a chart axis naming a
column the query never produces, ``dataset`` instead of ``data`` — passed
``validate()`` with ``ok: true`` and detonated at render.

Every test here asserts two things: the mistake is caught with a *pathed*
message, and it is caught **before any data loads** — the connector
records every ``load()`` call precisely so that claim is checkable.

References only the physical table can confirm (an ad-hoc dict-measure
column, a bare filter column beyond the declared ones) surface as pathed
*warnings* — erroring on those would refuse specs that execute fine, and
would break the documented from_report round-trip. Execution still fails
loudly on them, and the gateway's render channel now returns that failure
as structured ``{ok: false, errors: [...]}`` rather than a raw traceback.
"""

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.spec import ReportSpec


class RecordingConnector(MemoryConnector):
    """A MemoryConnector that records every load — proof nothing executed."""

    def __init__(self, name, tables):
        super().__init__(name, tables)
        self.loads: list[str] = []

    def load(self, source, filter=None, columns=None):
        self.loads.append(source)
        return super().load(source, filter=filter, columns=columns)


def _model(connector_cls=RecordingConnector, name="VG Sales"):
    orders = pd.DataFrame({
        "order_id":    [1, 2, 3, 4],
        "customer_id": [1, 2, 1, 2],
        "revenue":     [100.0, 200.0, 300.0, 400.0],
        "cost":        [60.0, 150.0, 180.0, 320.0],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2], "region": ["West", "NE"],
    })
    connector = connector_cls(
        "vg_mem", {"orders": orders, "customers": customers}
    )
    m = DataModel(name).add_connector(connector)
    m.add_table("orders", connector="vg_mem", source="orders")
    m.add_table("customers", connector="vg_mem", source="customers")
    m.add_dimension("dim_customer", table_name="customers",
                    key_col="customer_id", attributes=["region"])
    m.add_fact("fact_orders", table_name="orders",
               measures=["revenue", "cost"],
               foreign_keys={"dim_customer": "customer_id"})
    m.add_measure("total_revenue", column="revenue", agg="sum")
    m.add_measure("gross_margin", expr="revenue - cost", agg="sum")
    m.add_measure("margin_pct", ratio=("gross_margin", "total_revenue"),
                  format="percent")
    m.connect()
    return m, connector


def _spec(section):
    return {"name": "VG", "sections": [section]}


def _table(query):
    return _spec({
        "type": "table", "title": "T",
        "data": {"model": "VG Sales", "query": query},
    })


def _validate(spec_dict, model):
    return ReportSpec.from_dict(spec_dict).validate({"VG Sales": model})


# ── Filter columns ─────────────────────────────────────────────────────────

def test_bare_filter_naming_a_dimension_attribute_is_a_pathed_warning():
    # A warning, not an error: the bare name may also be a physical column
    # on a denormalised fact table, and execution accepts that spelling with
    # different semantics than the dimension filter — rejecting it outright
    # was a false rejection (review finding on the validate-gaps branch).
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "filters": {"region": "West"},
    }), model)
    assert result["ok"]
    warn = next(w for w in result["warnings"]
                if w.startswith("sections[0].data.query.filters.region:"))
    assert "dim_customer.region" in warn
    assert conn.loads == [], "validation must not load data"


def test_dotted_filter_with_unknown_dimension_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "filters": {"dim_customre.region": "West"},
    }), model)
    assert not result["ok"]
    err = next(e for e in result["errors"]
               if "filters.dim_customre.region" in e)
    assert "Did you mean 'dim_customer'?" in err
    assert conn.loads == []


def test_dotted_filter_with_undeclared_attribute_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "filters": {"dim_customer.regoin": "West"},
    }), model)
    assert not result["ok"]
    err = next(e for e in result["errors"]
               if "filters.dim_customer.regoin" in e)
    assert "Did you mean 'region'?" in err
    assert conn.loads == []


def test_unverifiable_bare_filter_column_warns_with_a_path():
    """A bare column the model does not declare may still exist on the
    table, so refusing it would reject specs that execute fine. It is
    surfaced as a pathed warning instead, and execution checks it."""
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "filters": {"statas": "shipped"},
    }), model)
    assert result["ok"]
    assert any(w.startswith("sections[0].data.query.filters.statas:")
               for w in result["warnings"])
    assert conn.loads == []


# ── Aggregation names and dict measures ────────────────────────────────────

def test_unknown_aggregation_function_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": {"revenue": "summ"},
    }), model)
    assert not result["ok"]
    err = next(e for e in result["errors"]
               if e.startswith("sections[0].data.query.measures.revenue:"))
    assert "summ" in err and "Did you mean 'sum'?" in err
    assert conn.loads == []


def test_undeclared_dict_measure_column_warns_with_a_hint():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": {"revnue": "sum"},
    }), model)
    assert result["ok"]  # only the table itself could confirm the column
    warn = next(w for w in result["warnings"]
                if w.startswith("sections[0].data.query.measures.revnue:"))
    assert "Did you mean 'revenue'?" in warn
    assert conn.loads == []


# ── Dimension attributes ───────────────────────────────────────────────────

def test_undeclared_dimension_attribute_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "dimensions": ["dim_customer.regionn"],
    }), model)
    assert not result["ok"]
    err = next(e for e in result["errors"]
               if e.startswith("sections[0].data.query.dimensions:"))
    assert "regionn" in err and "Did you mean 'region'?" in err
    assert conn.loads == []


def test_dimension_reference_without_a_dot_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "dimensions": ["dim_customer"],
    }), model)
    assert not result["ok"]
    assert any("dot notation" in e for e in result["errors"])
    assert conn.loads == []


def test_dimension_key_column_is_a_valid_attribute():
    """The key column joins like any attribute; execution allows it."""
    model, _ = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "dimensions": ["dim_customer.customer_id"],
    }), model)
    assert result["ok"], result["errors"]


# ── Chart x/y references ───────────────────────────────────────────────────

def _chart(query, **axes):
    return _spec({
        "type": "chart", "chart_type": "bar",
        "data": {"model": "VG Sales", "query": query},
        **axes,
    })


def test_chart_x_must_be_a_column_the_query_produces():
    model, conn = _model()
    result = _validate(_chart({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "dimensions": ["dim_customer.region"],
    }, x="region", y="total_revenue"), model)
    assert not result["ok"]
    err = next(e for e in result["errors"] if e.startswith("sections[0].x:"))
    assert "dim_customer.region" in err  # names the columns it will have
    assert conn.loads == []


def test_chart_y_typo_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_chart({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "dimensions": ["dim_customer.region"],
    }, x="dim_customer.region", y="total_revenu"), model)
    assert not result["ok"]
    assert any(e.startswith("sections[0].y:") for e in result["errors"])
    assert conn.loads == []


def test_chart_may_reference_a_ratio_components_column():
    """measures=["margin_pct"] also produces its component columns."""
    model, _ = _model()
    result = _validate(_chart({
        "fact": "fact_orders", "measures": ["margin_pct"],
        "dimensions": ["dim_customer.region"],
    }, x="dim_customer.region", y=["gross_margin", "total_revenue"]), model)
    assert result["ok"], result["errors"]


# ── The dataset-vs-data trap ───────────────────────────────────────────────

def test_dataset_key_is_caught_with_a_pointed_message():
    model, conn = _model()
    result = _validate(_spec({
        "type": "table", "title": "T",
        "dataset": {"model": "VG Sales", "query": {
            "fact": "fact_orders", "measures": ["total_revenue"]}},
    }), model)
    assert not result["ok"]
    assert any(
        e.startswith("sections[0]:") and "'dataset' is the Python field" in e
        and "rename to 'data'" in e
        for e in result["errors"]
    )
    assert conn.loads == []


def test_dataset_key_is_refused_at_build_too():
    model, _ = _model()
    spec = ReportSpec.from_dict(_spec({"type": "table", "dataset": {}}))
    with pytest.raises(ValueError, match="rename to 'data'"):
        spec.build({"VG Sales": model})


# ── A good spec still validates ────────────────────────────────────────────

def test_good_spec_validates_ok_without_loading():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders",
        "measures": ["total_revenue", "margin_pct"],
        "dimensions": ["dim_customer.region"],
        "filters": {"dim_customer.region": "West", "revenue": {"gte": 0}},
    }), model)
    assert result["ok"], result["errors"]
    assert result["errors"] == []
    assert conn.loads == [], "validation must not load data"


# ── The gateway render error channel ───────────────────────────────────────

def test_render_returns_ok_false_for_an_execution_time_failure(tmp_path):
    """A spec that validates (with a warning) but fails at execution must
    come back as {ok: false, errors: [...]} — typed, pathless is fine,
    but never a raw traceback and never an unhandled exception."""
    from tracebi import model_registry
    from tracebi.mcp_server import gateway_render_spec

    model, _ = _model(name="vg_render_demo")
    model_registry.register(model)

    spec = {
        "name": "VG Render",
        "sections": [{
            "type": "table", "title": "T",
            "data": {"model": "vg_render_demo", "query": {
                "fact": "fact_orders",
                # Validates (warning only — the column might exist), then
                # fails loudly inside execute() when the table disagrees.
                "measures": {"revnue": "sum"},
            }},
        }],
    }
    out = gateway_render_spec(spec, output_dir=str(tmp_path))
    assert out["ok"] is False
    assert any("ValueError" in e and "revnue" in e for e in out["errors"])
    assert all("Traceback" not in e for e in out["errors"])
    assert not any(p.suffix == ".html" for p in tmp_path.iterdir()), \
        "no artifact may exist for a failed render"


# ─────────────────────────────────────────────
# Review-pass fixes (adversarial findings)
# ─────────────────────────────────────────────

def test_mixed_type_measures_list_is_a_pathed_error_not_a_crash():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders",
        "measures": ["total_revenue", ["revenue"], 7],
    }), model)
    assert not result["ok"]
    msgs = [e for e in result["errors"] if "list-form measures" in e]
    assert len(msgs) == 2   # one per non-string entry
    assert conn.loads == []


def test_unknown_filter_operator_is_a_pathed_error():
    model, conn = _model()
    result = _validate(_table({
        "fact": "fact_orders", "measures": ["total_revenue"],
        "filters": {"revenue": {"gte!": 1000}},
    }), model)
    assert not result["ok"]
    err = next(e for e in result["errors"] if "gte!" in e)
    assert "filters.revenue" in err
    assert "gte" in err   # did-you-mean or supported list
    assert conn.loads == []


def test_chart_color_typo_is_a_pathed_error():
    model, conn = _model()
    section = {
        "type": "chart", "chart_type": "bar",
        "x": "dim_customer.region", "y": "total_revenue",
        "color": "regoin",
        "data": {"model": model.name, "query": {
            "fact": "fact_orders", "measures": ["total_revenue"],
            "dimensions": ["dim_customer.region"],
        }},
    }
    result = _validate({"name": "s", "sections": [section]}, model)
    assert not result["ok"]
    assert any("color" in e and "regoin" in e for e in result["errors"])
    assert conn.loads == []


def test_tuple_y_is_treated_as_a_list_of_axes():
    model, conn = _model()
    section = {
        "type": "chart", "chart_type": "bar",
        "x": "dim_customer.region", "y": ("total_revenue",),
        "data": {"model": model.name, "query": {
            "fact": "fact_orders", "measures": ["total_revenue"],
            "dimensions": ["dim_customer.region"],
        }},
    }
    result = _validate({"name": "s", "sections": [section]}, model)
    assert result["ok"], result["errors"]

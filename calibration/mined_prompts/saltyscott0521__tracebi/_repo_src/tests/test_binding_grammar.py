"""
The binding grammar — QuerySpec order_by/limit and the determinism guard.

Two protections, per docs/architecture/report-architecture-v2.md §2.2:

* The fingerprint corpus: queries WITHOUT order_by/limit must produce
  byte-identical results (and therefore identical fingerprints) before and
  after the grammar gained ordering. The hard-coded fingerprints below were
  computed on the pre-change engine; if one moves, an existing manifest
  somewhere stopped re-verifying.
* The trap refusal: limit without order_by is an error everywhere — "first N
  rows" in engine order must never masquerade as "top N".
"""

import pandas as pd
import pytest

from tracebi import DataModel, MemoryConnector
from tracebi.model.data_model import QuerySpec, _normalize_order_by


@pytest.fixture()
def corpus_model():
    orders = pd.DataFrame({
        "order_id":    [1, 2, 3, 4, 5, 6, 7, 8],
        "customer_id": [1, 2, 1, 3, 2, 1, 3, 2],
        "revenue":     [100.0, 250.0, 75.0, 300.0, 125.0, 50.0, 220.0, 90.0],
        "qty":         [1, 2, 1, 3, 1, 1, 2, 1],
    })
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "region":      ["NE", "SE", "MW"],
        "segment":     ["ent", "smb", "ent"],
    })
    m = DataModel("corpus_model")
    m.add_connector(MemoryConnector("corpus_mem",
                                    tables={"orders": orders, "customers": customers}))
    m.add_table("orders", connector="corpus_mem", source="orders")
    m.add_table("customers", connector="corpus_mem", source="customers")
    m.add_relationship("o2c", left_table="orders", right_table="customers",
                       left_key="customer_id")
    m.add_dimension("dim_customer", table_name="customers", key_col="customer_id",
                    attributes=["region", "segment"])
    m.add_fact("fact_orders", table_name="orders", measures=["revenue", "qty"],
               foreign_keys={"dim_customer": "customer_id"})
    m.add_measure("total_revenue", column="revenue", agg="sum")
    m.add_measure("units", column="qty", agg="sum")
    m.add_measure("rev_per_unit", ratio=("total_revenue", "units"))
    m.connect()
    return m


# Computed on the PRE-order_by engine. A moved fingerprint here means a
# recorded query in an existing manifest no longer reproduces.
_PRE_CHANGE_CORPUS = [
    ({"fact": "fact_orders", "measures": ["total_revenue"]},
     "48496eed8ae3aeac5994b1c127277ec378f60b443d8b1ba22f41a58afba094dc", 1),
    ({"fact": "fact_orders", "measures": ["total_revenue", "units"],
      "dimensions": ["dim_customer.region"]},
     "ea474a3072bf42fc31ebee90549117006f8c115cd2218f3b42e1341cd177192e", 3),
    ({"fact": "fact_orders", "measures": {"revenue": "sum", "qty": "mean"},
      "dimensions": ["dim_customer.segment"]},
     "990a9832661cf5337542b438dab0410a7d2f4221beef9d9359fbeb6d1ca61a3b", 2),
    ({"fact": "fact_orders", "measures": ["total_revenue", "rev_per_unit"],
      "dimensions": ["dim_customer.region", "dim_customer.segment"]},
     "e392b481ba3182c17d03333168bb6f6007bb4c27b7b2417b27c88d8123c1244b", 3),
    ({"fact": "fact_orders", "measures": ["total_revenue"],
      "dimensions": ["dim_customer.region"],
      "filters": {"dim_customer.segment": "ent"}},
     "3e641c2c4d4edbedbbe9bc7a5b23b77a460aef1824ff49878e8b953f6cee20f7", 2),
]


class TestFingerprintCorpus:
    def test_queries_without_ordering_are_byte_stable(self, corpus_model):
        for query, expected_fp, expected_rows in _PRE_CHANGE_CORPUS:
            ds = corpus_model.execute(QuerySpec.from_dict(query))
            assert ds.fingerprint() == expected_fp, (
                f"fingerprint moved for {query} — an existing manifest "
                f"somewhere no longer re-verifies"
            )
            assert len(ds.to_pandas()) == expected_rows

    def test_unordered_spec_serializes_without_ordering_keys(self):
        spec = QuerySpec.from_dict({"fact": "f", "measures": ["m"]})
        d = spec.to_dict()
        assert "order_by" not in d and "limit" not in d, (
            "an unordered spec must serialize byte-for-byte as before the "
            "grammar gained ordering"
        )


class TestDtypeCanonicalization:
    """pandas 3.0 renamed the default string dtype (object → str). The
    fingerprint canonicalizes toward the LEGACY name so a receipt rendered
    under pandas 2 still re-verifies under pandas 3 — same bytes, same
    receipt, whichever pandas produced it. This is the cross-version half
    of the corpus above: the corpus pins the bytes on THIS pandas; this
    pins that every string-dtype spelling hashes as the same dtype."""

    def test_string_dtype_spellings_fingerprint_identically(self):
        from tracebi.model.dataset import frame_fingerprint

        base = pd.DataFrame({"label": ["a", "b", None],
                             "revenue": [1.5, 2.0, 3.25]})
        variants = [base]
        for spelling in ("string", "str"):
            try:
                v = base.astype({"label": spelling})
            except (TypeError, ValueError):
                continue    # this pandas doesn't know this spelling — fine
            # Only a RENAMED dtype counts: pandas 2's astype('str') is the
            # old stringify-everything cast (None → "None"), which changes
            # the values themselves — a genuinely different frame, not a
            # spelling of this one.
            if v.to_csv(index=False) == base.to_csv(index=False):
                variants.append(v)
        assert len(variants) >= 2, "no comparable string-dtype spelling found"
        fps = {frame_fingerprint(v) for v in variants}
        assert len(fps) == 1, (
            "a string column must fingerprint identically under every "
            "dtype spelling — otherwise a pandas upgrade orphans every "
            "existing receipt"
        )

    def test_canonical_dtypes_use_the_legacy_name(self):
        from tracebi.model.dataset import canonical_dtypes_repr

        df = pd.DataFrame({"label": ["a"], "n": [1]})
        assert canonical_dtypes_repr(df) == "['object', 'int64']", (
            "the canonical name is the legacy one: existing manifests and "
            "the corpus above were hashed with 'object'"
        )


class TestOrderByLimit:
    def _top(self, model, **kw):
        return model.query(
            fact="fact_orders", measures=["total_revenue"],
            dimensions=["dim_customer.region"], **kw,
        ).to_pandas()

    def test_order_by_desc_sorts_the_result(self, corpus_model):
        df = self._top(corpus_model,
                       order_by=[{"column": "total_revenue", "desc": True}])
        vals = list(df["total_revenue"])
        assert vals == sorted(vals, reverse=True)

    def test_limit_takes_the_top_n_not_the_first_n(self, corpus_model):
        df = self._top(corpus_model, order_by=["-total_revenue"], limit=1)
        assert len(df) == 1
        full = self._top(corpus_model, order_by=["-total_revenue"])
        assert df["total_revenue"][0] == full["total_revenue"].max()

    def test_limit_without_order_by_is_refused(self, corpus_model):
        with pytest.raises(ValueError, match="masquerades"):
            self._top(corpus_model, limit=5)

    def test_ratio_measures_are_sortable(self, corpus_model):
        df = corpus_model.query(
            fact="fact_orders", measures=["total_revenue", "rev_per_unit"],
            dimensions=["dim_customer.region"],
            order_by=["-rev_per_unit"], limit=2,
        ).to_pandas()
        assert len(df) == 2
        vals = list(df["rev_per_unit"])
        assert vals == sorted(vals, reverse=True)

    def test_unknown_order_column_fails_loudly_with_hint(self, corpus_model):
        with pytest.raises(ValueError, match="total_revenu"):
            self._top(corpus_model, order_by=["-total_revenu"])

    def test_stamped_spec_carries_the_fully_resolved_ordering(self, corpus_model):
        ds = corpus_model.query(
            fact="fact_orders", measures=["total_revenue"],
            dimensions=["dim_customer.region", "dim_customer.segment"],
            order_by=["-total_revenue"], limit=2,
        )
        stamped = None
        for node in ds.lineage:
            qs = (node.metadata or {}).get("query_spec")
            if qs:
                stamped = qs
        assert stamped is not None
        cols = [o["column"] for o in stamped["order_by"]]
        # The stated ordering first, then the remaining dims as tie-break.
        assert cols[0] == "total_revenue"
        assert set(cols[1:]) == {"dim_customer.region", "dim_customer.segment"}
        assert stamped["limit"] == 2

    def test_replaying_the_stamped_spec_reproduces_the_fingerprint(self, corpus_model):
        ds = corpus_model.query(
            fact="fact_orders", measures=["total_revenue"],
            dimensions=["dim_customer.region"],
            order_by=["-total_revenue"], limit=2,
        )
        stamped = None
        for node in ds.lineage:
            qs = (node.metadata or {}).get("query_spec")
            if qs:
                stamped = qs
        replay = corpus_model.execute(QuerySpec.from_dict(stamped))
        assert replay.fingerprint() == ds.fingerprint()

    def test_tie_break_makes_ties_deterministic(self, corpus_model):
        # Two runs of an order_by over a column with ties must agree exactly.
        kw = dict(order_by=[{"column": "units", "desc": True}])
        df1 = corpus_model.query(fact="fact_orders",
                                 measures=["total_revenue", "units"],
                                 dimensions=["dim_customer.region"], **kw).to_pandas()
        df2 = corpus_model.query(fact="fact_orders",
                                 measures=["total_revenue", "units"],
                                 dimensions=["dim_customer.region"], **kw).to_pandas()
        assert df1.equals(df2)


class TestCheckQuerySpecOrdering:
    def test_limit_without_order_by_is_a_pathed_error(self, corpus_model):
        spec = QuerySpec.from_dict({
            "fact": "fact_orders", "measures": ["total_revenue"], "limit": 5,
        })
        errors, _ = corpus_model.check_query_spec(spec)
        assert any(path == "limit" for path, _ in errors)

    def test_unknown_order_column_is_a_pathed_error(self, corpus_model):
        spec = QuerySpec.from_dict({
            "fact": "fact_orders", "measures": ["total_revenue"],
            "dimensions": ["dim_customer.region"],
            "order_by": ["-not_a_column"],
        })
        errors, _ = corpus_model.check_query_spec(spec)
        assert any(path.startswith("order_by") for path, _ in errors)

class TestNormalizeOrderBy:
    def test_shorthand_and_dict_forms_normalize_identically(self):
        assert _normalize_order_by(["-x"]) == ({"column": "x", "desc": True},)
        assert _normalize_order_by(["x"]) == ({"column": "x", "desc": False},)
        assert _normalize_order_by([{"column": "x", "desc": True}]) == (
            {"column": "x", "desc": True},)

    def test_unknown_keys_and_bad_types_are_refused(self):
        with pytest.raises(ValueError, match="unknown key"):
            _normalize_order_by([{"column": "x", "direction": "desc"}])
        with pytest.raises(ValueError, match="expected a dict or string"):
            _normalize_order_by([42])

    def test_from_dict_rejects_nonpositive_limit(self):
        with pytest.raises(ValueError, match="positive"):
            QuerySpec.from_dict({"fact": "f", "measures": ["m"],
                                 "order_by": ["-m"], "limit": 0})

"""
DataModel — governed semantic model: tables, relationships, declared
facts/dimensions/measures, and a star-schema query surface.

A DataModel registers connectors and tables, then exposes two layered query
surfaces over them:

* **Ad-hoc navigation** — declare named relationships with
  ``add_relationship()`` and resolve them via ``resolve()`` /
  ``resolve_chain()`` to get a flat, lineage-tracked DataSet.

* **Analytic queries** — tag tables as dimensions and facts with
  ``add_dimension()`` / ``add_fact()`` and call ``query()`` for filtered,
  aggregated OLAP-style results. Joins are auto-resolved from the fact's
  foreign keys; execution uses DuckDB (a required extra) as the one engine.

Both surfaces share the connector/table registry and produce DataSets with
full lineage chains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from tracebi.connectors.base import BaseConnector
from tracebi.model.dataset import (
    CURRENT_FINGERPRINT_ALGO,
    DataSet,
    LineageNode,
    frame_fingerprint,
)


# Threshold (rows) at which an unfiltered, full-table load triggers a
# lineage warning. Visible in the lineage chain, non-blocking.
LARGE_LOAD_WARN_ROWS = 100_000

#: Date grains a declared time-grain attribute can roll up to (DuckDB
#: ``date_trunc`` units). All deterministic — no fingerprint-algo concern.
_TIME_GRAINS = {"year", "quarter", "month", "week", "day"}


def _pop_offset(unit: str, n: int):
    """A period offset as a ``pandas.DateOffset`` — the shift a period-over-
    period measure applies to the time axis to find the prior period. Aligns
    with the same units a time grain buckets to, so ``unit`` shifting a
    grain-start lands on the earlier grain-start exactly (deterministic match)."""
    if unit == "year":    return pd.DateOffset(years=n)
    if unit == "quarter": return pd.DateOffset(months=3 * n)
    if unit == "month":   return pd.DateOffset(months=n)
    if unit == "week":    return pd.DateOffset(weeks=n)
    if unit == "day":     return pd.DateOffset(days=n)
    raise ValueError(f"Unsupported period unit '{unit}'.")


#: Fine-to-coarse ordering of the time grains, so a to-date measure can refuse a
#: reset period that is not COARSER than the query's grain (YTD over monthly is
#: fine; MTD over monthly is degenerate — one row per bucket).
_GRAIN_RANK = {"day": 1, "week": 2, "month": 3, "quarter": 4, "year": 5}

#: The pandas period frequency each to-date reset partition truncates to.
_PERIOD_FREQ = {"year": "Y", "quarter": "Q", "month": "M", "week": "W"}

_AGG_FUNCS = {"sum", "count", "mean", "avg", "min", "max", "nunique",
              # Distribution aggregations — a mean hides skew and tails, so
              # dispersion/robust-centre reporting used to force report.py.
              # DuckDB-native and deterministic; map to MEDIAN()/STDDEV() by the
              # default func.upper() path, no special-casing.
              "median", "stddev"}

#: Percentile aggregations, spelled ``p<N>`` (p0–p100): p50 is the median, p90/
#: p95/p99 are the tail a mean and even a median hide — the fund-ops tail-risk
#: view (worst marks, largest drawdowns). A parameterised family rather than a
#: fixed member of _AGG_FUNCS; maps to DuckDB ``quantile_cont`` (interpolated,
#: deterministic). See `tracebi knowledge summarize-a-distribution`.
_PERCENTILE_RE = re.compile(r"^p(\d{1,3})$")


def _percentile_fraction(func: str) -> "float | None":
    """``"p90"`` → 0.90, for any p0–p100; ``None`` if not a percentile agg."""
    m = _PERCENTILE_RE.match(func.lower())
    if not m:
        return None
    n = int(m.group(1))
    return n / 100 if 0 <= n <= 100 else None


def _valid_agg(func: str) -> bool:
    """True for a named aggregation or a ``p<N>`` percentile."""
    return func.lower() in _AGG_FUNCS or _percentile_fraction(func) is not None


def _agg_choices() -> str:
    """Human list of aggregations for an error/hint — the named set plus the
    percentile family, described once so every site reads the same."""
    return ", ".join(sorted(_AGG_FUNCS)) + ", p0–p100 (percentiles, e.g. p90)"

#: Name tokens that mark a measure as already a rate/ratio — summing or
#: averaging one of these per-row is a silent-wrong number (see the rate-
#: aggregation guard in add_measure and `tracebi knowledge ratio-of-totals`).
_RATE_TOKEN = re.compile(r"(?:^|_)(pct|percent|ratio|rate|bps|yield|apr|apy)(?:_|$)")

#: Value-based rate guard. The name guard above is blind to a per-row ratio
#: whose column name matches no token (``mark_cost`` = fair_value/cost, ~1.0):
#: summing/averaging it is the same silent-wrong number. So at execution — where
#: the values ARE available — refuse an additive aggregation of a column that
#: LOOKS like per-row ratios: floating, mostly non-integer, robust centre within
#: a 2× band of 1.0. Additive quantities (money, counts, sizes) are essentially
#: never centred at ~1, so this fires narrowly; gated on row count so small
#: fixtures never trip. See `tracebi knowledge ratio-of-totals`.
_RATE_MIN_ROWS = 20
_RATE_BAND_LO, _RATE_BAND_HI = 0.5, 2.0
_RATE_GUARD_AGGS = ("sum", "mean", "avg")

#: Name tokens that mark a measure as a point-in-time STOCK (a balance, not a
#: flow): summing one across snapshots double-counts it (Jan AUM + Feb AUM is
#: not "AUM"). Guarded in add_measure; the fix is a period_end (semi-additive)
#: measure. See `tracebi knowledge semi-additive`.
_STOCK_TOKEN = re.compile(
    r"(?:^|_)(aum|nav|balance|headcount|inventory|outstanding)(?:_|$)")

# Filter operators. A closed set rather than free SQL: free SQL cannot be
# validated and is an injection surface. Every operator below is parameterised
# in the DuckDB path.
#   {"region": "NE"}                        → eq  (scalar shorthand)
#   {"region": ["NE", "SE"]}                → in  (list shorthand)
#   {"revenue": {"gte": 1000}}              → explicit operator
#   {"dim_customer.region": "NE"}           → filter on a dimension attribute
FILTER_OPS = (
    "eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte",
    "between", "is_null", "not_null", "contains",
)

# Operators whose value is not a plain scalar.
_LIST_OPS = ("in", "not_in")
_NULL_OPS = ("is_null", "not_null")


# ─────────────────────────────────────────────────────────────
# Internal config dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Predicate:
    """
    One parsed filter condition.

    ``target`` is either a fact column name or a ``"dim_name.attribute"``
    reference. Dimension predicates require the dimension to be joined, so
    they are applied after the join; fact predicates can be pushed down.
    """
    target: str
    op: str
    value: Any
    dim_name: Optional[str] = None      # set for dimension predicates
    attribute: Optional[str] = None

    @property
    def is_dim(self) -> bool:
        return self.dim_name is not None

    def describe(self) -> str:
        if self.op in _NULL_OPS:
            return f"{self.target} {self.op.replace('_', ' ')}"
        return f"{self.target} {self.op} {self.value!r}"


@dataclass(frozen=True)
class _FilterNode:
    """A node in the filter's boolean tree. A leaf holds one ``_Predicate``; an
    ``and``/``or`` node holds children. The common all-AND filter is a single
    ``and`` node over leaves and renders exactly as before; an ``or`` node is
    what a ``{"or": [...]}`` group builds."""
    op: str                                     # "and" | "or" | "leaf"
    children: tuple = ()                        # tuple[_FilterNode] for and/or
    predicate: Optional[_Predicate] = None      # for a leaf


def _filter_leaves(node: "Optional[_FilterNode]") -> "list[_Predicate]":
    """Every leaf predicate in the tree, left to right — the flat set the
    consumers that only care WHICH columns are filtered use (dim loading,
    per-condition lineage, dim-value validation), independent of the AND/OR
    structure the WHERE renderer needs."""
    if node is None:
        return []
    if node.op == "leaf":
        return [node.predicate] if node.predicate is not None else []
    out: list[_Predicate] = []
    for child in node.children:
        out.extend(_filter_leaves(child))
    return out


@dataclass(frozen=True)
class MeasureDef:
    """
    A named, governed measure declared on the model.

    Defined once in ``models/<name>.py``, reviewed in a pull request,
    versioned in git, and referenced by name from every report — this is
    what makes the model a shared vocabulary rather than a pile of ad-hoc
    groupbys.

    Exactly six kinds, deliberately closed:

    * **simple** — an aggregation of one column::

          add_measure("revenue", column="revenue", agg="sum")

    * **expression** — an aggregation of a row-level arithmetic expression::

          add_measure("gross_margin", expr="revenue - cost", agg="sum")

    * **ratio** — one measure divided by another, computed after
      aggregation so the result is a true ratio of totals, not a mean of
      per-row ratios::

          add_measure("margin_pct", ratio=("gross_margin", "revenue"))

    * **share** — a value over the total of the measure it names, computed
      after aggregation — a governed "% of total"::

          add_measure("revenue_share", share="revenue", format="percent")

    * **rank** / **running** — window measures over a base measure, ordered by
      it descending (rank 1 = largest; running is the cumulative sum,
      largest-first) — the concentration/Pareto direction::

          add_measure("rev_rank", rank="revenue")
          add_measure("cum_share", running="revenue_share", format="percent")

    * **period_end** — a semi-additive (point-in-time) measure: a balance
      summed across non-time dimensions but taken at the LATEST snapshot over
      time, never summed across snapshots (the AUM/NAV double-count)::

          add_measure("aum", period_end=("balance", "dim_date.as_of_date"))

    These cover the large majority of real usage and need no expression
    parser. A richer grammar can arrive later as sugar that compiles down to
    these same structures.

    Measures are **data, never callables.** A lambda cannot be serialized,
    diffed, reviewed as data, validated before execution, or sent over the
    wire — accepting one would forfeit reproducibility at the root.
    """
    name: str
    kind: str                        # simple|expression|ratio|share|rank|running|period_end
    agg: Optional[str] = None
    column: Optional[str] = None
    expr: Optional[str] = None
    ratio: Optional[tuple[str, str]] = None    # (numerator, denominator)
    share: Optional[str] = None                # measure name: value / total(value)
    rank: Optional[str] = None                 # measure name: 1..N by it, desc
    running: Optional[str] = None              # measure name: cumsum by it, desc
    partition_by: tuple[str, ...] = ()         # rank/running restart per group
    period_end: Optional[tuple[str, str]] = None  # (value_column, date_dim_attr)
    offset: Optional[tuple[str, str, int]] = None  # (base, unit, n): prior value
    growth: Optional[tuple[str, str, int]] = None  # (base, unit, n): (cur-prior)/prior
    to_date: Optional[tuple[str, str]] = None      # (base, period): YTD/QTD/MTD
    description: str = ""
    format: Optional[str] = None               # presentation hint, e.g. "percent"
    allow_rate_agg: bool = False               # author vouched: skip the value guard

    def to_dict(self) -> dict:
        d = {"name": self.name, "kind": self.kind}
        for key in ("agg", "column", "expr", "description", "format"):
            val = getattr(self, key)
            if val:
                d[key] = val
        if self.ratio:
            d["ratio"] = list(self.ratio)
        if self.share:
            d["share"] = self.share
        if self.period_end:
            d["period_end"] = list(self.period_end)
        if self.partition_by:
            d["partition_by"] = list(self.partition_by)
        if self.offset:
            d["offset"] = list(self.offset)
        if self.growth:
            d["growth"] = list(self.growth)
        if self.to_date:
            d["to_date"] = list(self.to_date)
        return d


# Characters permitted in a row-level measure expression. Restricted to
# arithmetic over bare column names and numeric literals: no function calls,
# no quotes, no subqueries. The expression is validated against the fact's
# actual columns before it ever reaches an engine.
_EXPR_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPR_ALLOWED = re.compile(r"^[A-Za-z0-9_+\-*/(). ]+$")
# An identifier immediately followed by "(" is a function call. Row
# expressions are arithmetic only — aggregation is declared via agg=.
_EXPR_CALL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")


def _normalize_order_by(raw: Any) -> tuple[dict, ...]:
    """
    Normalize ``order_by`` to the canonical tuple-of-dicts form.

    Accepts ``{"column": str, "desc": bool}`` dicts and the ``"col"`` /
    ``"-col"`` string shorthand, either as a list of them or as a lone element.
    The stamped resolved spec always carries the dict form, so replay compares
    like with like.
    """
    if not raw:
        return ()
    # A lone element form, not a list of them: order_by documents both
    # {"column":..,"desc":..} and "-col" as PER-KEY forms, and an agent (or the
    # docs read literally) naturally passes one bare. Left as-is a bare string
    # would iterate as characters ("-col" → column "") and a bare dict as its
    # keys — a cryptic error pointing at a character. Wrap the singular in a list.
    if isinstance(raw, (str, dict)):
        raw = [raw]
    out: list[dict] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            desc = entry.startswith("-")
            out.append({"column": entry.lstrip("-"), "desc": desc})
            continue
        if isinstance(entry, dict):
            unknown = set(entry) - {"column", "desc"}
            if unknown:
                raise ValueError(
                    f"order_by[{i}]: unknown key(s) {sorted(unknown)}. "
                    f"Each entry is {{'column': str, 'desc': bool}} or the "
                    f"'col' / '-col' string shorthand."
                )
            col = entry.get("column")
            if not col or not isinstance(col, str):
                raise ValueError(f"order_by[{i}]: each entry needs a 'column' (string).")
            out.append({"column": col, "desc": bool(entry.get("desc", False))})
            continue
        raise ValueError(
            f"order_by[{i}]: expected a dict or string, got {type(entry).__name__}."
        )
    return tuple(out)


@dataclass(frozen=True)
class QuerySpec:
    """
    A star-schema query as data.

    Everything needed to run a query, in a form that can be serialized,
    diffed, code-reviewed, committed, and replayed. ``DataModel.execute()``
    takes one of these; ``DataModel.query()`` is sugar that builds one.

    The resolved spec is stamped into the query's lineage, so an audit trail
    records not merely "a query ran" but exactly which query — and combined
    with ``DataSet.fingerprint()`` that is end-to-end reproducibility.

    Deliberately contains no callables and no free-form SQL: a spec that
    cannot be validated before execution is not a spec.
    """
    fact: str
    measures: Any                                   # {col: agg} or [names]
    dimensions: tuple[str, ...] = ()
    filters: Any = None                             # WHERE — before aggregation
    having: Any = None                              # HAVING — after aggregation
    aggregate: bool = True
    allow_fanout: bool = False
    allow_rate_agg: bool = False                    # skip the value-based rate guard
    order_by: tuple = ()                            # ({"column": str, "desc": bool}, ...)
    limit: Optional[int] = None

    def __post_init__(self):
        # Normalize order_by at construction so EVERY path is forgiving — not
        # only query()/from_dict, which already normalize, but a QuerySpec built
        # directly with a bare "-col" or a lone {"column":..} dict. Without this
        # a singular form iterates as characters/keys and fails cryptically in
        # to_dict/execute. Idempotent, so double-normalization is harmless.
        object.__setattr__(self, "order_by", _normalize_order_by(self.order_by))

    def to_dict(self) -> dict:
        d = {
            "fact": self.fact,
            "measures": (list(self.measures) if isinstance(self.measures, (list, tuple))
                         else dict(self.measures)),
            "dimensions": list(self.dimensions),
            "filters": dict(self.filters) if self.filters else {},
            "aggregate": self.aggregate,
            "allow_fanout": self.allow_fanout,
        }
        # Only when present: a spec without ordering serializes byte-for-byte
        # as it did before order_by/limit existed, so recorded specs in
        # existing manifests and lineage stay stable. allow_rate_agg is the
        # same — a query that never sets it stamps identically to before the
        # guard existed, so no issued receipt moves.
        if self.allow_rate_agg:
            d["allow_rate_agg"] = True
        if self.having:
            d["having"] = dict(self.having)
        if self.order_by:
            d["order_by"] = [dict(o) for o in self.order_by]
        if self.limit is not None:
            d["limit"] = int(self.limit)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "QuerySpec":
        if "fact" not in d:
            raise ValueError("QuerySpec requires a 'fact'.")
        if "measures" not in d:
            raise ValueError("QuerySpec requires 'measures'.")
        unknown = set(d) - {
            "fact", "measures", "dimensions", "filters", "having", "aggregate",
            "allow_fanout", "allow_rate_agg", "order_by", "limit",
        }
        if unknown:
            raise ValueError(
                f"Unknown QuerySpec field(s): {sorted(unknown)}. "
                f"Allowed: fact, measures, dimensions, filters, having, "
                f"aggregate, allow_fanout, allow_rate_agg, order_by, limit."
            )
        measures = d["measures"]
        limit = d.get("limit")
        if limit is not None:
            limit = int(limit)
            if limit < 1:
                raise ValueError(f"limit must be a positive integer, got {limit}.")
        return cls(
            fact=d["fact"],
            measures=list(measures) if isinstance(measures, list) else dict(measures),
            dimensions=tuple(d.get("dimensions") or ()),
            filters=dict(d.get("filters") or {}) or None,
            having=dict(d.get("having") or {}) or None,
            aggregate=bool(d.get("aggregate", True)),
            allow_fanout=bool(d.get("allow_fanout", False)),
            allow_rate_agg=bool(d.get("allow_rate_agg", False)),
            order_by=_normalize_order_by(d.get("order_by")),
            limit=limit,
        )


@dataclass(frozen=True)
class _TableDef:
    name: str
    connector_name: str
    source: str


@dataclass(frozen=True)
class _RelationshipDef:
    name: str
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    how: str  # left | inner | right | outer


@dataclass
class _DimensionDef:
    name: str
    table_name: str
    key_col: str
    attributes: list[str] = field(default_factory=list)
    # Derived attributes computed at query time, e.g. a date grain:
    #   {"order_month": {"kind": "date_trunc", "grain": "month",
    #                    "source": "order_date"}}
    # Referenced like any attribute (dim.name); the SQL builder emits the
    # transform instead of a raw column.
    derived: dict = field(default_factory=dict)


@dataclass
class _FactDef:
    name: str
    table_name: str
    measures: list[str]
    foreign_keys: dict[str, str] = field(default_factory=dict)  # dim_name -> fk_col


@dataclass
class _Resolved:
    """What ``_resolve_measures`` produces — each measure kind in its own bucket,
    applied at the right stage of ``execute()``. ``agg_map`` runs in the engine
    (GROUP BY); the rest run post-aggregation in this order: ``derived`` (row
    expressions, materialised pre-agg), ``period_ends`` (semi-additive rollups,
    which run their own internal aggregation so later kinds can reference them),
    ``ratios``, ``shares``, then the window measures ``ranks``/``runnings``. A
    named result rather than a growing tuple, so a new measure kind adds a
    field, never re-arities every call site."""

    agg_map: "dict[str, str]"                    # {output: agg_func}
    derived: "dict[str, str]"                    # {output: row_expression}
    ratios: "dict[str, tuple[str, str]]"         # {output: (numerator, denom)}
    shares: "dict[str, str]"                     # {output: base measure}
    ranks: "dict[str, str]"                      # {output: base measure}
    runnings: "dict[str, str]"                   # {output: base measure}
    period_ends: "dict[str, tuple[str, str]]"    # {output: (value_col, date_attr)}
    partitions: "dict[str, tuple[str, ...]]"     # {rank/running output: part cols}
    pops: "dict[str, tuple[str, str, int, str]]"  # {output: (base, unit, n, mode)}
    to_dates: "dict[str, tuple[str, str]]"        # {output: (base, reset period)}


# ─────────────────────────────────────────────────────────────
# DataModel
# ─────────────────────────────────────────────────────────────

class DataModel:
    """
    A code-defined relational data model with optional star-schema semantics.

    Ad-hoc navigation::

        from tracebi import DataModel, CSVConnector, SQLConnector

        model = DataModel("SalesModel")
        model.add_connector(SQLConnector("sales_db", url="sqlite:///sales.db"))
        model.add_connector(CSVConnector("lookups", directory="data/"))

        model.add_table("orders",    connector="sales_db", source="orders")
        model.add_table("customers", connector="sales_db", source="customers")
        model.add_table("regions",   connector="lookups",  source="regions.csv")

        model.add_relationship(
            name="orders_customers",
            left_table="orders",
            right_table="customers",
            left_key="customer_id",
            how="left",
        )

        model.connect()
        orders_ds = model.load("orders")
        joined_ds = model.resolve("orders_customers")
        full_ds   = model.resolve_chain(["orders_customers", "customers_regions"])

    Analytic queries::

        model.add_dimension(
            name="dim_customer",
            table_name="customers",
            key_col="customer_id",
            attributes=["region", "segment"],
        )
        model.add_fact(
            name="fact_orders",
            table_name="orders",
            measures=["revenue", "qty"],
            foreign_keys={"dim_customer": "customer_id"},
        )

        ds = model.query(
            fact="fact_orders",
            measures={"revenue": "sum", "qty": "sum"},
            dimensions=["dim_customer.region"],
            filters={"status": "shipped"},
            aggregate=True,
        )
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._connectors: dict[str, BaseConnector] = {}
        self._tables: dict[str, _TableDef] = {}
        self._relationships: dict[str, _RelationshipDef] = {}
        self._dimensions: dict[str, _DimensionDef] = {}
        self._facts: dict[str, _FactDef] = {}
        self._measures: dict[str, MeasureDef] = {}

    # ── Fluent builder ─────────────────────────────────────────

    def add_connector(self, connector: BaseConnector) -> "DataModel":
        """Register a connector by its ``name`` attribute."""
        self._connectors[connector.name] = connector
        return self

    def connectors(self) -> list[BaseConnector]:
        """Return the connector objects registered on this model."""
        return list(self._connectors.values())

    def add_table(
        self,
        name: str,
        connector: str,
        source: str,
    ) -> "DataModel":
        """
        Register a table.

        Args:
            name:      Logical table name used in relationships and ``load()``.
            connector: Name of a previously registered connector.
            source:    Source identifier passed to the connector's ``load()``
                       (e.g. table name, file name, SQL query).
        """
        if connector not in self._connectors:
            raise ValueError(
                f"Connector '{connector}' is not registered in model '{self.name}'. "
                f"Call add_connector() first."
            )
        self._tables[name] = _TableDef(name=name, connector_name=connector, source=source)
        return self

    def add_relationship(
        self,
        name: str,
        left_table: str,
        right_table: str,
        left_key: str,
        right_key: Optional[str] = None,
        how: str = "left",
    ) -> "DataModel":
        """
        Declare a named join relationship between two tables.

        Args:
            name:        Unique relationship name; used in ``resolve()`` and
                         ``resolve_chain()``.
            left_table:  Name of the left-hand table.
            right_table: Name of the right-hand table.
            left_key:    Column in the left table to join on.
            right_key:   Column in the right table to join on.
                         Defaults to ``left_key`` when omitted.
            how:         Pandas merge type: ``'left'``, ``'inner'``,
                         ``'right'``, ``'outer'``. Default ``'left'``.
        """
        for tbl in (left_table, right_table):
            if tbl not in self._tables:
                raise ValueError(
                    f"Table '{tbl}' is not registered in model '{self.name}'. "
                    f"Call add_table() first."
                )
        self._relationships[name] = _RelationshipDef(
            name=name,
            left_table=left_table,
            right_table=right_table,
            left_key=left_key,
            right_key=right_key or left_key,
            how=how,
        )
        return self

    def add_dimension(
        self,
        name: str,
        table_name: str,
        key_col: str,
        attributes: Optional[list[str]] = None,
    ) -> "DataModel":
        """Tag a registered table as a dimension with a key column and exposed attributes."""
        if table_name not in self._tables:
            raise ValueError(
                f"Table '{table_name}' is not registered in model '{self.name}'. "
                f"Call add_table() first."
            )
        self._dimensions[name] = _DimensionDef(
            name=name,
            table_name=table_name,
            key_col=key_col,
            attributes=list(attributes) if attributes else [],
        )
        return self

    def add_time_grain(
        self,
        dim: str,
        name: str,
        source: str,
        grain: str,
    ) -> "DataModel":
        """Declare a derived date-grain attribute on a dimension.

        ``name`` becomes a groupable attribute that rolls ``source`` (a date/
        timestamp column on the dimension) up to ``grain`` — a governed
        "group by month" (``date_trunc``) instead of pre-baking every bucket in
        the transform or dropping to report.py::

            model.add_dimension("dim_date", table_name="dates", key_col="d",
                                attributes=["order_date"])
            model.add_time_grain("dim_date", "order_month",
                                 source="order_date", grain="month")
            model.query(fact="orders", measures=["revenue"],
                        dimensions=["dim_date.order_month"])   # governed monthly

        Reference it like any attribute (``dim.name``); it is deterministic, so
        the result stays reproducible.
        """
        if dim not in self._dimensions:
            raise ValueError(
                f"Dimension '{dim}' is not registered in model '{self.name}'. "
                f"Available: {list(self._dimensions)}"
            )
        g = grain.lower()
        if g not in _TIME_GRAINS:
            raise ValueError(
                f"Time grain '{grain}' is not supported."
                f"{self._hint(g, sorted(_TIME_GRAINS))} "
                f"One of: {', '.join(sorted(_TIME_GRAINS))}."
            )
        self._dimensions[dim].derived[name] = {
            "kind": "date_trunc", "grain": g, "source": str(source),
        }
        return self

    @staticmethod
    def _default_bin_labels(edges: "list[float]") -> "list[str]":
        """Readable band labels for ``edges`` — ``"< 600"``, ``"600–700"``,
        ``"≥ 800"``. Whole numbers render without a trailing ``.0``."""
        def n(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else str(v)
        labels = [f"< {n(edges[0])}"]
        for i in range(1, len(edges)):
            labels.append(f"{n(edges[i - 1])}–{n(edges[i])}")
        labels.append(f"≥ {n(edges[-1])}")
        return labels

    def add_value_bins(
        self,
        dim: str,
        name: str,
        source: str,
        edges: "list[float]",
        labels: "list[str] | None" = None,
    ) -> "DataModel":
        """Declare a derived banding attribute on a dimension: bucket a numeric
        ``source`` column into ranges, groupable as ``dim.name``.

        ``edges`` are the strictly increasing breakpoints; ``N`` edges make
        ``N+1`` bands (below the first, between each pair, at/above the last).
        A governed "group by band" — a CASE expression built once — instead of
        pre-baking the bucket in the transform or dropping to report.py::

            model.add_dimension("dim_customer", table_name="customers",
                                key_col="id", attributes=["credit_score"])
            model.add_value_bins("dim_customer", "score_band",
                                  source="credit_score", edges=[600, 700, 800])
            model.query(fact="loans", measures=["balance"],
                        dimensions=["dim_customer.score_band"])   # by band

        Reference it like any attribute (``dim.name``); it is deterministic
        (a CASE), so the result stays reproducible. ``labels`` (one per band,
        ``len(edges)+1`` of them) overrides the defaults; band values sort
        lexicographically by label, so choose labels that sort if magnitude
        order matters. A NULL ``source`` groups as NULL, never the top band.
        Bins are groupable, not filterable — filter on the ``source`` column.
        """
        if dim not in self._dimensions:
            raise ValueError(
                f"Dimension '{dim}' is not registered in model '{self.name}'. "
                f"Available: {list(self._dimensions)}"
            )
        try:
            edge_vals = [float(e) for e in edges]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"value bins '{name}': edges must be numbers, got {edges!r}."
            ) from e
        if not edge_vals:
            raise ValueError(
                f"value bins '{name}': need at least one edge (N edges make "
                f"N+1 bands)."
            )
        if any(edge_vals[i] >= edge_vals[i + 1] for i in range(len(edge_vals) - 1)):
            raise ValueError(
                f"value bins '{name}': edges must be strictly increasing, got "
                f"{edge_vals}."
            )
        if labels is not None:
            labels = [str(v) for v in labels]
            if len(labels) != len(edge_vals) + 1:
                raise ValueError(
                    f"value bins '{name}': need one label per band — "
                    f"len(edges)+1 = {len(edge_vals) + 1}, got {len(labels)}."
                )
        else:
            labels = self._default_bin_labels(edge_vals)
        self._dimensions[dim].derived[name] = {
            "kind": "bin", "source": str(source),
            "edges": edge_vals, "labels": labels,
        }
        return self

    def add_fact(
        self,
        name: str,
        table_name: str,
        measures: list[str],
        foreign_keys: Optional[dict[str, str]] = None,
    ) -> "DataModel":
        """Tag a registered table as a fact with measure columns and FK mapping to dimensions."""
        if table_name not in self._tables:
            raise ValueError(
                f"Table '{table_name}' is not registered in model '{self.name}'. "
                f"Call add_table() first."
            )
        self._facts[name] = _FactDef(
            name=name,
            table_name=table_name,
            measures=list(measures),
            foreign_keys=dict(foreign_keys) if foreign_keys else {},
        )
        return self

    def add_measure(
        self,
        name: str,
        *,
        column: Optional[str] = None,
        agg: Optional[str] = None,
        expr: Optional[str] = None,
        ratio: Optional[tuple[str, str]] = None,
        share: Optional[str] = None,
        rank: Optional[str] = None,
        running: Optional[str] = None,
        partition_by: "str | list[str] | None" = None,
        period_end: Optional[tuple[str, str]] = None,
        offset: Optional[tuple[str, str, int]] = None,
        growth: Optional[tuple[str, str, int]] = None,
        to_date: Optional[tuple[str, str]] = None,
        description: str = "",
        format: Optional[str] = None,
        allow_rate_agg: bool = False,
        allow_additive: bool = False,
    ) -> "DataModel":
        """
        Declare a named measure on the model.

        Define the calculation once, review it in a pull request, and
        reference it by name from every report and query::

            model.add_measure("revenue", column="revenue", agg="sum")
            model.add_measure("orders", column="order_id", agg="nunique")
            model.add_measure("gross_margin", expr="revenue - cost", agg="sum")
            model.add_measure("margin_pct", ratio=("gross_margin", "revenue"),
                              format="percent")

            model.query(fact="fact_orders",
                        measures=["revenue", "margin_pct"],
                        dimensions=["dim_customer.region"])

        Exactly one of ``column``, ``expr``, ``ratio``, ``share``, ``rank``,
        ``running``, or ``period_end`` must be given. ``agg`` is required for
        the first two. Ratios and shares are
        computed after aggregation — a ratio is a ratio of totals (not a mean of
        per-row ratios), a share is a value over the total of the measure it
        names (a governed ``%`` of total).

        ``period_end=(value_column, "dim_date.attr")`` declares a semi-additive
        (point-in-time) measure — a balance summed across non-time dimensions
        but taken at the LATEST snapshot date within each query group, so it is
        never summed across snapshots (Jan AUM + Feb AUM is not "AUM"). Assumes
        entities snapshot on common dates (month-end); see
        ``tracebi knowledge semi-additive``.

        Aggregating a rate-named measure (``*_pct``, ``*_bps``, ``*_yield``,
        anything "weighted") with ``sum``/``mean``/``avg`` is REFUSED — you do
        not additively combine per-row rates (summing is meaningless, averaging
        overweights small rows); declare a ratio measure instead. ``min``/
        ``max`` stay fine. Pass ``allow_rate_agg=True`` in the rare case an
        additive aggregation of a rate is genuinely correct.

        Summing a stock-named measure (``aum``, ``nav``, ``balance``,
        ``headcount``, ``inventory``) is REFUSED for the same reason — a
        point-in-time balance double-counts when summed across snapshots.
        Declare a ``period_end`` measure instead, or pass ``allow_additive=True``
        if the column is genuinely a flow (a delta, not a level).

        Everything here is declarative data — callables are rejected, since
        a lambda cannot be serialized, diffed, reviewed, or validated
        before it runs.
        """
        for label, value in (("column", column), ("expr", expr), ("agg", agg)):
            if callable(value):
                raise TypeError(
                    f"Measure '{name}': {label} must be declarative data, not a "
                    f"callable. A function cannot be serialized, diffed, or "
                    f"validated before execution. Use expr='a - b' instead of "
                    f"a lambda."
                )

        given = [k for k, v in (("column", column), ("expr", expr),
                                ("ratio", ratio), ("share", share),
                                ("rank", rank), ("running", running),
                                ("period_end", period_end),
                                ("offset", offset), ("growth", growth),
                                ("to_date", to_date))
                 if v is not None]
        if len(given) != 1:
            raise ValueError(
                f"Measure '{name}' must specify exactly one of column=, expr=, "
                f"ratio=, share=, rank=, running=, period_end=, offset=, "
                f"growth=, or to_date=, got {given or 'none'}."
            )

        kind = {"column": "simple", "expr": "expression", "ratio": "ratio",
                "share": "share", "rank": "rank", "running": "running",
                "period_end": "period_end", "offset": "offset",
                "growth": "growth", "to_date": "to_date"}[given[0]]

        # share/rank/running are all post-aggregation window measures over a
        # named base measure — none takes an agg.
        if kind in ("share", "rank", "running"):
            if agg is not None:
                raise ValueError(
                    f"Measure '{name}': {kind} measures take no agg — they are "
                    f"computed after aggregation from the measure they name."
                )
            share = str(share) if share is not None else None
            rank = str(rank) if rank is not None else None
            running = str(running) if running is not None else None

        # partition_by makes rank/running restart per group (top-N-per-group).
        part: tuple[str, ...] = ()
        if partition_by is not None:
            if kind not in ("rank", "running"):
                raise ValueError(
                    f"Measure '{name}': partition_by only applies to rank and "
                    f"running measures (it restarts the window per group); "
                    f"{kind} measures do not take it."
                )
            cols = ([partition_by] if isinstance(partition_by, str)
                    else list(partition_by))
            for c in cols:
                if "." not in str(c):
                    raise ValueError(
                        f"Measure '{name}': partition_by expects dimension "
                        f"references in 'dim_name.attribute' form, got {c!r}."
                    )
            part = tuple(str(c) for c in cols)

        if kind == "ratio":
            if not (isinstance(ratio, (tuple, list)) and len(ratio) == 2):
                raise ValueError(
                    f"Measure '{name}': ratio must be a (numerator, denominator) "
                    f"pair of measure names, got {ratio!r}."
                )
            if agg is not None:
                raise ValueError(
                    f"Measure '{name}': ratio measures take no agg — the ratio "
                    f"is computed from the aggregated numerator and denominator."
                )
            ratio = (str(ratio[0]), str(ratio[1]))
        elif kind == "period_end":
            if agg is not None:
                raise ValueError(
                    f"Measure '{name}': period_end measures take no agg — the "
                    f"balance is summed across non-time dimensions and taken at "
                    f"the latest snapshot over time."
                )
            if not (isinstance(period_end, (tuple, list)) and len(period_end) == 2):
                raise ValueError(
                    f"Measure '{name}': period_end must be a (value_column, "
                    f"'dim_name.attribute') pair, got {period_end!r}."
                )
            value_col, date_ref = str(period_end[0]), str(period_end[1])
            if "." not in date_ref:
                raise ValueError(
                    f"Measure '{name}': period_end's date must be a dimension "
                    f"attribute in 'dim_name.attribute' form (the snapshot date "
                    f"axis), got {date_ref!r}."
                )
            period_end = (value_col, date_ref)
        elif kind in ("offset", "growth"):
            # Period-over-period: the base measure's value one (unit, n) earlier
            # (offset), or its growth vs then ((cur - prior) / prior). No agg —
            # the base is already an aggregated measure; the query must group by
            # a declared time grain (checked at execution).
            if agg is not None:
                raise ValueError(
                    f"Measure '{name}': {kind} measures take no agg — they read "
                    f"the base measure's value in an earlier period."
                )
            spec_val = offset if kind == "offset" else growth
            if not (isinstance(spec_val, (tuple, list)) and len(spec_val) == 3):
                raise ValueError(
                    f"Measure '{name}': {kind} must be a (base_measure, unit, n) "
                    f"triple — e.g. ('revenue', 'year', 1), got {spec_val!r}."
                )
            base_m, unit, n = str(spec_val[0]), str(spec_val[1]).lower(), spec_val[2]
            if unit not in _TIME_GRAINS:
                raise ValueError(
                    f"Measure '{name}': {kind} unit '{spec_val[1]}' is not "
                    f"supported.{self._hint(unit, sorted(_TIME_GRAINS))} "
                    f"One of: {', '.join(sorted(_TIME_GRAINS))}."
                )
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise ValueError(
                    f"Measure '{name}': {kind} n must be a positive integer "
                    f"(periods back), got {n!r}."
                )
            triple = (base_m, unit, n)
            if kind == "offset":
                offset = triple
            else:
                growth = triple
        elif kind == "to_date":
            # Cumulative from the start of a reset period (year/quarter/month/
            # week) up to the current period — YTD/QTD/MTD. No agg; the query
            # must group by a time grain FINER than the reset period (checked at
            # execution against the grain).
            if agg is not None:
                raise ValueError(
                    f"Measure '{name}': to_date measures take no agg — they "
                    f"accumulate the base measure over the query's time grain."
                )
            if not (isinstance(to_date, (tuple, list)) and len(to_date) == 2):
                raise ValueError(
                    f"Measure '{name}': to_date must be a (base_measure, period) "
                    f"pair — e.g. ('revenue', 'year') for YTD, got {to_date!r}."
                )
            base_m, period = str(to_date[0]), str(to_date[1]).lower()
            if period not in _PERIOD_FREQ:
                raise ValueError(
                    f"Measure '{name}': to_date period '{to_date[1]}' is not "
                    f"supported.{self._hint(period, sorted(_PERIOD_FREQ))} "
                    f"One of: {', '.join(sorted(_PERIOD_FREQ))}."
                )
            to_date = (base_m, period)
        elif kind in ("simple", "expression"):    # these require an agg
            if agg is None:
                raise ValueError(
                    f"Measure '{name}' needs an agg (one of {_agg_choices()})."
                )
            if not _valid_agg(agg):
                raise ValueError(
                    f"Measure '{name}': unsupported aggregation '{agg}'."
                    f"{self._hint(agg, sorted(_AGG_FUNCS))} "
                    f"Supported: {_agg_choices()}"
                )
            agg = agg.lower()
            # The rate-aggregation guard — the fanout raise's sibling. You do not
            # additively aggregate a per-row rate: summing rates is meaningless
            # and averaging them overweights small rows. Two independent signals,
            # because they scope differently:
            #   • a RATE-TOKEN name (*_pct, *_bps, *_yield) is a rate value — bad
            #     to sum OR mean;
            #   • a "weighted" LABEL is the weighted-mean contradiction, but only
            #     for a MEAN — a SUM of a weighted numerator, Σ(value × weight),
            #     is exactly the correct building block, so sum is left alone.
            # min/max of a rate stay fine (widest spread, lowest yield). Refuse
            # at definition, cite the lesson, and offer the escape (allow_fanout's
            # sibling).
            name_is_rate = bool(_RATE_TOKEN.search(name.lower()))
            says_weighted = "weighted" in f"{name} {description}".lower()
            bad_rate = name_is_rate and agg in ("sum", "mean", "avg")
            bad_weighted = says_weighted and agg in ("mean", "avg")
            if not allow_rate_agg and (bad_rate or bad_weighted):
                wrong = ("summing per-row rates is meaningless" if agg == "sum"
                         else "the mean of per-row rates overweights small rows")
                raise ValueError(
                    f"Measure '{name}': agg='{agg}' aggregates a rate additively "
                    f"— {wrong}. The blended rate is sum(numerator) / "
                    f"sum(denominator), never the {agg} of the per-row rates. "
                    f"Declare a ratio measure instead (ratio=(numerator, "
                    f"denominator)); a weighted average is a ratio whose "
                    f"numerator carries the weight (sum it, then divide). See "
                    f"`tracebi knowledge ratio-of-totals` and "
                    f"`weighted-vs-plain-mean`. If this aggregation genuinely is "
                    f"correct here, pass allow_rate_agg=True."
                )
            # The stock (semi-additive) guard — the rate guard's twin for a
            # different silent-wrong number. A stock-named measure (aum, nav,
            # balance, headcount, inventory) is a point-in-time LEVEL; summing
            # it across snapshots double-counts (Jan AUM + Feb AUM is not
            # "AUM"). Summing across NON-time dimensions is fine — but that is
            # exactly what period_end does, correctly, so a true stock never
            # wants a plain sum. mean/min/max of a level are legitimate (average
            # / peak balance), so guard only sum. Escape for a genuine flow.
            if (not allow_additive and agg == "sum"
                    and _STOCK_TOKEN.search(name.lower())):
                raise ValueError(
                    f"Measure '{name}': agg='sum' sums a point-in-time stock "
                    f"across snapshots — a double-count (Jan AUM + Feb AUM is "
                    f"not 'AUM'). Declare a semi-additive measure instead: "
                    f"period_end=('{column or name}', 'dim_date.<date>'), which "
                    f"sums across non-time dimensions but takes the latest "
                    f"snapshot over time. See `tracebi knowledge semi-additive`. "
                    f"If this column is genuinely a flow (a delta, not a level), "
                    f"pass allow_additive=True."
                )

        if kind == "expression":
            if not _EXPR_ALLOWED.match(expr or ""):
                raise ValueError(
                    f"Measure '{name}': expression {expr!r} may only contain "
                    f"column names, numbers, and + - * / ( ). Function calls, "
                    f"quotes, and SQL fragments are not allowed."
                )
            if _EXPR_CALL.search(expr or ""):
                raise ValueError(
                    f"Measure '{name}': expression {expr!r} looks like a "
                    f"function call. Row expressions are arithmetic only — "
                    f"declare the aggregation separately, e.g. "
                    f"expr='revenue - cost', agg='sum'."
                )

        self._measures[name] = MeasureDef(
            name=name, kind=kind, agg=agg, column=column, expr=expr,
            ratio=tuple(ratio) if ratio else None, share=share,
            rank=rank, running=running, partition_by=part,
            period_end=tuple(period_end) if period_end else None,
            offset=tuple(offset) if offset else None,
            growth=tuple(growth) if growth else None,
            to_date=tuple(to_date) if to_date else None,
            description=description, format=format,
            allow_rate_agg=bool(allow_rate_agg),
        )
        return self

    def measures(self) -> dict[str, MeasureDef]:
        """Declared measures, by name."""
        return dict(self._measures)

    # ── Connection ─────────────────────────────────────────────

    def connect(self) -> None:
        """Call ``connect()`` on every registered connector."""
        for connector in self._connectors.values():
            connector.connect()

    def disconnect(self) -> None:
        """Release every connector's open handle (a no-op for those without one).

        A DuckDB warehouse held open read-only by a model cannot also be
        opened read-write by the transform that rebuilds it, in the same
        process. Call this before rewriting the data; the next ``load``
        reconnects by itself.
        """
        for connector in self._connectors.values():
            if hasattr(connector, "disconnect"):
                connector.disconnect()

    # ── Data loading ───────────────────────────────────────────

    def load(
        self,
        table_name: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> DataSet:
        """
        Load a registered table and return a lineage-tracked DataSet.

        Every call to ``load()`` re-reads from the source (no caching),
        so lineage is always fresh.

        Args:
            table_name: Registered table name.
            filter:     Optional ``{column: value}`` equality filters pushed
                        down to the connector where possible (SQL WHERE,
                        DuckDB predicate), applied in pandas otherwise.
            columns:    Optional list of columns to project at source.

        Lineage: emits one ``operation="load"`` node, plus a non-blocking
        ``operation="warning"`` node when the load is unfiltered, unprojected,
        and returns more than ``LARGE_LOAD_WARN_ROWS`` rows.
        """
        if table_name not in self._tables:
            raise ValueError(
                f"Table '{table_name}' is not registered in model '{self.name}'. "
                f"Available tables: {list(self._tables.keys())}"
            )
        tdef = self._tables[table_name]
        connector = self._connectors[tdef.connector_name]
        df = connector.load(tdef.source, filter=filter, columns=columns)
        pushdown = connector.supports_pushdown() and (filter or columns)
        load_node = LineageNode(
            operation="load",
            description=f"Loaded '{table_name}' from connector '{tdef.connector_name}'",
            connector={
                "connector_name": tdef.connector_name,
                "connector_type": type(connector).__name__,
            },
            source=tdef.source,
            metadata={
                "rows_loaded": len(df),
                "filter":      filter,
                "columns":     columns,
                "pushdown":    bool(pushdown),
                # The frame is already in memory here, so fingerprint it now.
                # Every stamped query — and every manifest section built from
                # one — then records exactly which inputs produced it, which
                # is what lets `tracebi verify` tell source drift apart from
                # an unexplained mismatch.
                "input": {
                    "table":       table_name,
                    "fingerprint": frame_fingerprint(df),
                    "rows":        len(df),
                },
            },
        )
        nodes = [load_node]
        if not filter and not columns and len(df) > LARGE_LOAD_WARN_ROWS:
            nodes.append(LineageNode(
                operation="warning",
                description=(
                    f"Large unfiltered load: {len(df):,} rows from "
                    f"'{table_name}'. Consider passing filter= or columns= "
                    "to push the predicate to the source."
                ),
                metadata={
                    "rows_loaded": len(df),
                    "threshold":   LARGE_LOAD_WARN_ROWS,
                    "table":       table_name,
                },
            ))
        return DataSet(df=df, name=table_name, lineage=nodes)

    def resolve(self, relationship_name: str) -> DataSet:
        """
        Load and join two tables according to a named relationship.

        Returns a DataSet whose lineage includes load steps for both tables
        and a join step.
        """
        if relationship_name not in self._relationships:
            raise ValueError(
                f"Relationship '{relationship_name}' is not registered in model '{self.name}'. "
                f"Available: {list(self._relationships.keys())}"
            )
        rel = self._relationships[relationship_name]
        left_ds = self.load(rel.left_table)
        right_ds = self.load(rel.right_table)

        merged = left_ds.to_pandas().merge(
            right_ds.to_pandas(),
            left_on=rel.left_key,
            right_on=rel.right_key,
            how=rel.how,
            suffixes=("", f"_{rel.right_table}"),
        )
        join_node = LineageNode(
            operation="join",
            description=(
                f"Joined '{rel.left_table}' → '{rel.right_table}' "
                f"on {rel.left_key}={rel.right_key} ({rel.how})"
            ),
            metadata={
                "relationship": relationship_name,
                "left_key":     rel.left_key,
                "right_key":    rel.right_key,
                "how":          rel.how,
                "rows_left":    len(left_ds),
                "rows_right":   len(right_ds),
                "rows_after":   len(merged),
                "right_chain_len": len(right_ds.lineage),
            },
        )
        combined_lineage = left_ds.lineage + right_ds.lineage + [join_node]
        return DataSet(
            df=merged,
            name=f"{rel.left_table}_{rel.right_table}",
            lineage=combined_lineage,
        )

    def resolve_chain(self, relationship_names: list[str]) -> DataSet:
        """
        Resolve a chain of relationships left-to-right.

        Equivalent to calling ``resolve()`` iteratively, accumulating all
        lineage steps.

        Example:
            full = model.resolve_chain(["orders_customers", "customers_regions"])
        """
        if not relationship_names:
            raise ValueError("resolve_chain() requires at least one relationship name.")

        ds = self.resolve(relationship_names[0])

        for rel_name in relationship_names[1:]:
            if rel_name not in self._relationships:
                raise ValueError(
                    f"Relationship '{rel_name}' is not registered in model '{self.name}'."
                )
            rel = self._relationships[rel_name]
            right_ds = self.load(rel.right_table)
            merged = ds.to_pandas().merge(
                right_ds.to_pandas(),
                left_on=rel.left_key,
                right_on=rel.right_key,
                how=rel.how,
                suffixes=("", f"_{rel.right_table}"),
            )
            join_node = LineageNode(
                operation="join",
                description=(
                    f"Joined → '{rel.right_table}' "
                    f"on {rel.left_key}={rel.right_key} ({rel.how})"
                ),
                metadata={
                    "relationship": rel_name,
                    "left_key":     rel.left_key,
                    "right_key":    rel.right_key,
                    "how":          rel.how,
                    "rows_left":    len(ds),
                    "rows_right":   len(right_ds),
                    "rows_after":   len(merged),
                    "right_chain_len": len(right_ds.lineage),
                },
            )
            ds = DataSet(
                df=merged,
                name=f"{ds.name}_{rel.right_table}",
                lineage=ds.lineage + right_ds.lineage + [join_node],
            )

        return ds

    # ── Analytic query (star-schema) ───────────────────────────

    def query(
        self,
        fact: str,
        measures: "dict[str, str] | list[str]",
        dimensions: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        having: Optional[dict[str, Any]] = None,
        aggregate: bool = True,
        allow_fanout: bool = False,
        allow_rate_agg: bool = False,
        order_by: Optional[list] = None,
        limit: Optional[int] = None,
    ) -> DataSet:
        """
        Run a star-schema analytic query and return a lineage-tracked DataSet.

        Args:
            fact:       Fact name registered via ``add_fact()``.
            measures:   ``{column: agg_func}`` — e.g.
                        ``{"revenue": "sum", "order_id": "count"}``.
                        Supported agg funcs: ``sum``, ``count``, ``mean``,
                        ``min``, ``max``, ``nunique``.
            dimensions: List of ``"dim_name.attribute"`` references for
                        grouping and selection. May be ``None`` or ``[]``
                        to return aggregate totals only.
            filters:    Predicates on fact columns **or dimension
                        attributes**. Three spellings per entry::

                            {"status": "shipped"}              # equality
                            {"region": ["NE", "SE"]}           # IN
                            {"revenue": {"gte": 1000}}         # operator
                            {"dim_customer.region": "West"}    # dimension

                        Operators: eq, ne, in, not_in, gt, gte, lt, lte,
                        between, is_null, not_null, contains. A dimension
                        referenced only by a filter is still joined.
                        Equality filters on fact columns are pushed down to
                        the connector where it supports it.
                        ``filters`` are WHERE — they apply **before**
                        aggregation, so a filter on a measure column changes
                        the group totals. To filter on an aggregated value,
                        use ``having`` instead.
            having:     Post-aggregation predicates (HAVING) on **result**
                        columns — measure names and ratios — applied *after*
                        grouping. ``{"revenue": {"gte": 250}}`` keeps groups
                        whose *total* revenue is ≥ 250, with their totals
                        intact; the same key in ``filters`` would instead drop
                        raw rows and change the totals. Same operators and
                        spellings as ``filters``.
            aggregate:  When ``True`` (default), group by dimension attributes
                        and aggregate measures. When ``False``, return the
                        flat joined rows with measures selected.
            allow_fanout: Permit joining a dimension whose key is not unique.
                        Off by default: a repeated key multiplies fact rows
                        and silently inflates additive measures. Set ``True``
                        only when the multiplication is intended — the opt-in
                        is then recorded as a warning node in the lineage.
            order_by:   Result ordering: ``[{"column": ..., "desc": bool}]``
                        dicts or ``"col"`` / ``"-col"`` shorthand, over result
                        columns (dimension refs, measure names — ratios
                        included). Remaining dimension columns are appended
                        as an ascending tie-break so the order is total, and
                        the fully resolved ordering is stamped in the spec.
            limit:      Keep the first N rows after sorting. Refused without
                        ``order_by`` — "first N" must never masquerade as
                        "top N".

        Returns:
            A DataSet with full lineage covering load → join → filter →
            aggregate steps.

        Raises:
            ValueError: if a referenced column does not exist, or if a joined
                dimension has a non-unique key and ``allow_fanout`` is False.
        """
        return self.execute(QuerySpec(
            fact=fact,
            measures=measures,
            dimensions=tuple(dimensions or ()),
            filters=filters,
            having=having,
            aggregate=aggregate,
            allow_fanout=allow_fanout,
            allow_rate_agg=allow_rate_agg,
            order_by=_normalize_order_by(order_by),
            limit=limit,
        ))

    def execute(self, spec: QuerySpec) -> DataSet:
        """
        Run a :class:`QuerySpec` and return a lineage-tracked DataSet.

        This is the primitive; :meth:`query` is keyword sugar over it. Taking
        a spec means a query can be built as data — validated, serialized,
        committed, reviewed, and replayed — rather than only ever existing as
        a function call.
        """
        # DuckDB is the one query engine. Import it up front so an absent
        # engine fails with a clear install error before any source I/O,
        # rather than after loading the fact and its dimensions.
        try:
            import duckdb
        except ImportError as e:
            raise ImportError(
                "TraceBi's query engine requires DuckDB. Install it with:  "
                "pip install 'tracebi[analyst]'  (or the minimal "
                "'tracebi[duckdb]')."
            ) from e

        fact = spec.fact
        measures = spec.measures
        dimensions = list(spec.dimensions or [])
        filters = spec.filters
        aggregate = spec.aggregate
        allow_fanout = spec.allow_fanout

        if fact not in self._facts:
            raise ValueError(
                f"Fact '{fact}' is not registered in model '{self.name}'. "
                f"Available facts: {list(self._facts.keys())}"
            )
        fact_def = self._facts[fact]
        dimensions = list(dimensions or [])
        filters = dict(filters or {})
        resolved = self._resolve_measures(measures, fact_def)
        measures, derived, ratios = (resolved.agg_map, resolved.derived,
                                     resolved.ratios)
        shares, ranks, runnings = (resolved.shares, resolved.ranks,
                                   resolved.runnings)
        period_ends = resolved.period_ends
        partitions = resolved.partitions
        pops = resolved.pops
        to_dates = resolved.to_dates

        # A semi-additive rollup is an aggregation (latest-snapshot-then-sum);
        # there is no per-row "period end", so refuse it in the raw-row mode.
        if period_ends and not aggregate:
            raise ValueError(
                f"period_end measure(s) {sorted(period_ends)} aggregate to a "
                f"latest-snapshot total, but this query has aggregate=False. "
                f"Query them with aggregate=True (the default)."
            )

        # ── Parse dimension references ─────────────────────────
        parsed_dims: list[tuple[str, str]] = []
        for ref in dimensions:
            if "." not in ref:
                raise ValueError(
                    f"Dimension reference '{ref}' must use dot notation: "
                    f"'dim_name.attribute'"
                )
            dim_name, attribute = ref.split(".", 1)
            if dim_name not in self._dimensions:
                raise ValueError(
                    f"Dimension '{dim_name}' is not registered in model "
                    f"'{self.name}'. Available: {list(self._dimensions.keys())}"
                )
            parsed_dims.append((dim_name, attribute))

        # ── Load fact ─────────────────────────────────────────
        lineage: list[LineageNode] = []

        # Parse filters before loading so equality predicates on fact columns
        # can still be pushed down to the connector. A first load without a
        # filter is needed to know the fact's columns; use the declared
        # measures/foreign keys to avoid it where possible.
        probe_cols = set(fact_def.measures) | set(fact_def.foreign_keys.values())
        pushdown: dict[str, Any] = {}
        deferred: dict[str, Any] = {}
        for key, raw in (filters or {}).items():
            simple = (
                "." not in key
                and not isinstance(raw, (dict, list, tuple, set))
            )
            (pushdown if simple else deferred)[key] = raw

        fact_ds = self.load(fact_def.table_name, filter=pushdown or None)
        lineage.extend(fact_ds.lineage)
        fact_df = fact_ds.to_pandas()
        # Materialise row-level measure expressions before validation, so
        # derived columns are visible to everything downstream.
        if derived:
            fact_df = self._apply_derived(fact_df, derived, fact_def)
            lineage.append(LineageNode(
                operation="assign",
                description=(
                    "Derived measure column(s): "
                    + ", ".join(f"{k} = {v}" for k, v in derived.items())
                ),
                metadata={"derived": dict(derived)},
            ))
        fact_cols = set(fact_df.columns) | probe_cols

        # Value-based rate guard. The name guard at add_measure is blind to a
        # per-row ratio whose column name matches no token (mark_cost =
        # fair_value / cost, ~1.0); at execution the values are here, so refuse
        # an additive aggregation of a column that LOOKS like per-row ratios —
        # the same silent-wrong number (a mean of ratios), caught where the name
        # could not see it. The author (a declared measure's allow_rate_agg) or
        # the query (spec.allow_rate_agg) may vouch that the column is additive.
        if not spec.allow_rate_agg:
            for out_name, func in measures.items():
                if func not in _RATE_GUARD_AGGS:
                    continue
                md = self._measures.get(out_name)
                if md is not None and md.allow_rate_agg:
                    continue
                src = derived.get(out_name, out_name)
                if not (isinstance(src, str)
                        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", src)
                        and src in fact_df.columns):
                    continue
                if self._looks_like_per_row_ratio(fact_df[src]):
                    raise ValueError(
                        f"Measure '{out_name}': agg='{func}' aggregates column "
                        f"'{src}', whose values look like per-row ratios (a "
                        f"floating column centred near 1.0) — summing per-row "
                        f"ratios is meaningless and their mean overweights small "
                        f"rows. The blended rate is sum(numerator) / "
                        f"sum(denominator): declare a ratio measure "
                        f"(ratio=(numerator, denominator)). See `tracebi "
                        f"knowledge ratio-of-totals`. If this column really is an "
                        f"additive quantity, pass allow_rate_agg=True (on the "
                        f"measure or the query)."
                    )

        filter_tree = self._parse_filters(filters, fact_def, fact_cols)
        # The flat leaf set drives dim loading, per-condition lineage, and
        # dim-value validation — none of which cares about the AND/OR structure;
        # only the WHERE renderer (via filter_tree) does.
        predicates = _filter_leaves(filter_tree)
        # A predicate is `pushed_down` only when the connector ACTUALLY filtered
        # at source — which its load node records as pushdown=True (it requires
        # supports_pushdown() and a filter). The old guard `fact_ds.lineage` is
        # always truthy (a DataSet always has a load node), so it marked every
        # simple equality filter pushed_down even against a connector that
        # ignored the hint and returned every row. The engine re-applies all
        # predicates regardless (idempotent), so this only corrects the lineage
        # flag — but a receipt that claims a source-side filter that never
        # happened is exactly the kind of untruth this project refuses.
        load_pushed_down = any(
            (node.metadata or {}).get("pushdown") for node in fact_ds.lineage
        )
        pushed = set(pushdown) if load_pushed_down else set()

        # ── Load dimensions: those grouped by, those filtered on, AND the
        # snapshot-date dimension each period_end measure names ───
        needed_dims = {dim_name for dim_name, _ in parsed_dims}
        needed_dims |= {p.dim_name for p in predicates if p.is_dim}
        period_end_dims: dict[str, tuple[str, str]] = {}   # {name: (dim, attr)}
        for pname, (value_col, date_ref) in period_ends.items():
            date_dim, date_attr = date_ref.split(".", 1)
            if date_dim not in self._dimensions:
                raise ValueError(
                    f"period_end measure '{pname}': dimension '{date_dim}' is "
                    f"not registered in model '{self.name}'. Available: "
                    f"{list(self._dimensions.keys())}"
                )
            period_end_dims[pname] = (date_dim, date_attr)
            needed_dims.add(date_dim)
        dim_dfs: dict[str, pd.DataFrame] = {}
        for dim_name in needed_dims:
            dim_def = self._dimensions[dim_name]
            dim_ds = self.load(dim_def.table_name)
            lineage.extend(dim_ds.lineage)
            dim_dfs[dim_name] = dim_ds.to_pandas()

        # A typo'd column must never produce a silently wrong result.
        self._validate_query_columns(
            fact_def, fact_df, dim_dfs, parsed_dims, measures, {}
        )
        # period_end measures name a fact value column and a snapshot-date
        # dimension attribute — both outside `measures`/`parsed_dims`, so
        # validate them here rather than letting a typo surface as a raw engine
        # error deep in the internal aggregation.
        for pname, (value_col, date_ref) in period_ends.items():
            if value_col not in fact_df.columns:
                raise ValueError(
                    f"period_end measure '{pname}': value column '{value_col}' "
                    f"is not a column of fact '{fact_def.table_name}'. "
                    f"Columns: {sorted(fact_df.columns)}."
                )
            date_dim, date_attr = period_end_dims[pname]
            if date_attr not in dim_dfs[date_dim].columns:
                raise ValueError(
                    f"period_end measure '{pname}': '{date_attr}' is not a "
                    f"column of dimension '{date_dim}'. Columns: "
                    f"{sorted(dim_dfs[date_dim].columns)}."
                )
        self._validate_dim_predicates(predicates, dim_dfs)
        # Neither must a non-unique dimension key. Every joined dimension is
        # checked, including ones joined only to satisfy a filter.
        join_dims = [(d, "") for d in needed_dims]
        self._check_dimension_keys(
            dim_dfs, join_dims, fact_df, fact_def, allow_fanout, lineage
        )

        for p in predicates:
            lineage.append(LineageNode(
                operation="filter",
                description=f"Filter: {p.describe()}",
                metadata={
                    "target": p.target,
                    "operator": p.op,
                    "value": p.value,
                    "on": "dimension" if p.is_dim else "fact",
                    "pushed_down": p.target in pushed,
                },
            ))

        # ── Single canonical query engine: DuckDB ─────────────
        # One engine means one semantics and one set of bytes, so a receipt
        # reproduces no matter which environment re-runs it. DuckDB is a
        # required extra (guarded at the top of execute); its absence is a
        # clear install error rather than a silent pandas fallback that
        # returned different numbers for NULL groups, integer sums,
        # `contains`, and non-aggregate row order.
        result_df = self._execute_duckdb(
            fact_df=fact_df,
            fact_def=fact_def,
            dim_dfs=dim_dfs,
            parsed_dims=parsed_dims,
            measures=measures,
            filter_tree=filter_tree,
            aggregate=aggregate,
            lineage=lineage,
        )
        engine = "duckdb"
        engine_version = duckdb.__version__

        # Semi-additive (period_end) measures run their own internal
        # aggregation over (query dims + snapshot date), then reduce to each
        # group's latest snapshot — computed first so a ratio/share/rank can
        # reference a period_end measure like any other aggregated column.
        if period_ends:
            result_df = self._apply_period_ends(
                result_df, period_ends, period_end_dims,
                fact_df=fact_df, fact_def=fact_def, dim_dfs=dim_dfs,
                parsed_dims=parsed_dims, filter_tree=filter_tree,
            )
            lineage.append(LineageNode(
                operation="assign",
                description=("Semi-additive measure(s): " + ", ".join(
                    f"{k} = period_end({v[0]} over {v[1]})"
                    for k, v in period_ends.items())),
                metadata={"period_ends": {k: list(v)
                                          for k, v in period_ends.items()}},
            ))

        # Ratios divide the aggregated totals, so they must run after the
        # engine — sum(margin)/sum(revenue), not the mean of row ratios.
        if ratios:
            result_df = self._apply_ratios(result_df, ratios)
            lineage.append(LineageNode(
                operation="assign",
                description=(
                    "Ratio measure(s): "
                    + ", ".join(f"{k} = {n} / {d}" for k, (n, d) in ratios.items())
                ),
                metadata={"ratios": {k: list(v) for k, v in ratios.items()}},
            ))

        # Shares (% of total): each value over the total of its base measure,
        # computed on the FULL aggregated result (before HAVING/limit) so it is
        # a true share of the whole. Order-independent, so it needs none of the
        # ordering machinery rank/running would — the reason it ships first.
        if shares:
            result_df = self._apply_shares(result_df, shares)
            lineage.append(LineageNode(
                operation="assign",
                description=("Share measure(s): "
                             + ", ".join(f"{k} = {b} / total({b})"
                                         for k, b in shares.items())),
                metadata={"shares": dict(shares)},
            ))

        # Window measures — rank and running (cumulative), each over its own base
        # measure descending, with a TOTAL tie-break so the value is
        # reproducible. Applied on the full result, before HAVING/limit, so a
        # rank/cumulative is of the whole (top-10 shows its rank among all). A
        # partition_by restarts the window per group — rank 1 per group — which
        # a HAVING on the rank turns into top-N-per-group.
        if ranks or runnings:
            # A partition column must be one of the query's dimensions (a result
            # column) — else "rank within sector" silently ranks globally.
            for wname, part in partitions.items():
                for col in part:
                    if col not in result_df.columns:
                        raise ValueError(
                            f"Measure '{wname}': partition_by references "
                            f"'{col}', which is not in the result — add it to "
                            f"the query's dimensions so the window can group by "
                            f"it. Dimensions present: "
                            f"{[c for c in result_df.columns if '.' in str(c)]}."
                        )
            result_df = self._apply_windows(result_df, ranks, runnings, partitions)
            lineage.append(LineageNode(
                operation="assign",
                description=("Window measure(s): " + ", ".join(
                    [f"{k} = rank({b} desc"
                     + (f" within {'/'.join(partitions[k])}" if partitions.get(k) else "")
                     + ")" for k, b in ranks.items()]
                    + [f"{k} = running({b} desc"
                       + (f" within {'/'.join(partitions[k])}" if partitions.get(k) else "")
                       + ")" for k, b in runnings.items()])),
                metadata={"ranks": dict(ranks), "runnings": dict(runnings),
                          "partitions": {k: list(v) for k, v in partitions.items()}},
            ))

        # Period-over-period (offset/growth): read each base measure's value one
        # (unit, n) earlier by shifting the query's time-grain column and matching
        # on the other dimensions — YoY / QoQ / MoM. Runs after windows so a base
        # can be any earlier-computed measure; before HAVING/order so a PoP is
        # filterable and sortable.
        if pops:
            result_df = self._apply_pop(result_df, pops, parsed_dims)
            lineage.append(LineageNode(
                operation="assign",
                description=("Period-over-period measure(s): " + ", ".join(
                    f"{k} = {mode}({b}, {n} {unit} back)"
                    for k, (b, unit, n, mode) in pops.items())),
                metadata={"pops": {k: list(v) for k, v in pops.items()}},
            ))

        # To-date cumulatives (YTD/QTD/MTD): a running total from the start of
        # the reset period, over the query's finer time grain — a time-ordered
        # cumulative, distinct from the concentration `running` (base-ordered).
        if to_dates:
            result_df = self._apply_to_date(result_df, to_dates, parsed_dims)
            lineage.append(LineageNode(
                operation="assign",
                description=("To-date measure(s): " + ", ".join(
                    f"{k} = {b} accumulated within each {period}"
                    for k, (b, period) in to_dates.items())),
                metadata={"to_dates": {k: list(v) for k, v in to_dates.items()}},
            ))

        # Post-aggregation filters (HAVING): applied after the engine AND
        # ratios, so they filter the aggregated measures/ratios — the
        # difference between "groups whose total ≥ 250" and `filters`'
        # "raw rows whose value ≥ 250", which silently changes the totals.
        if spec.having:
            if not aggregate:
                raise ValueError(
                    "having applies after aggregation, but this query has "
                    "aggregate=False — there are no group totals to filter. "
                    "Filter the raw rows with `filters` instead."
                )
            result_df = self._apply_having(result_df, spec.having, lineage)

        # ── Ordering (after ratios, so a ratio measure is sortable) ────
        # limit without order_by is refused: "first N rows" in engine order
        # silently masquerades as "top N" — the grammar will not express it.
        if spec.limit is not None and not spec.order_by:
            raise ValueError(
                "limit without order_by is refused: without a stated ordering, "
                "'first N rows' silently masquerades as 'top N'. Add order_by "
                "to state the ranking."
            )
        stamped_spec = spec.to_dict()
        if spec.order_by:
            resolved = [dict(o) for o in spec.order_by]
            result_cols = list(result_df.columns)
            for o in resolved:
                if o["column"] not in result_df.columns:
                    raise ValueError(
                        f"order_by column '{o['column']}' is not a result "
                        f"column of this query."
                        f"{self._hint(o['column'], result_cols)} "
                        f"Result columns: {result_cols}"
                    )
            # Deterministic on ties: append the remaining grouped dimension
            # columns as an implicit ascending tie-break, and stamp the FULLY
            # resolved ordering so replay reproduces it exactly. An aggregate
            # spec without order_by keeps its byte-for-byte output — the engine
            # already ORDER BYs the group columns — so no aggregate fingerprint
            # moves; a non-aggregate spec without order_by is total-ordered in
            # the branch just below.
            # Total tie-break: after the stated order_by keys, break remaining
            # ties by EVERY other column — the grouped dimensions first (the
            # natural secondary sort), then every measure — so the sort key
            # uniquely orders every row. Dimension columns alone do NOT identify
            # a row, so a tie on them used to fall back to the engine's
            # nondeterministic scan order — which made "top 10" (order_by +
            # limit) return a different set/order across identical runs whenever
            # values tied at the boundary. Ordering by all columns removes that.
            named = {o["column"] for o in resolved}
            tie_cols = ([f"{d}.{a}" for d, a in parsed_dims]
                        + list(result_df.columns))
            for col in tie_cols:
                if col in result_df.columns and col not in named:
                    resolved.append({"column": col, "desc": False})
                    named.add(col)
            by = [o["column"] for o in resolved]
            asc = [not o["desc"] for o in resolved]
            try:
                result_df = result_df.sort_values(
                    by=by, ascending=asc, kind="mergesort", na_position="last",
                ).reset_index(drop=True)
            except TypeError:
                # A column holds values pandas can't order natively (a decoded
                # BLOB, a list cell) — order by a stable string projection so the
                # result stays deterministic rather than crashing.
                result_df = result_df.sort_values(
                    by=by, ascending=asc, kind="mergesort", na_position="last",
                    key=lambda s: s.astype(str),
                ).reset_index(drop=True)
            stamped_spec["order_by"] = [dict(o) for o in resolved]
            if spec.limit is not None:
                result_df = result_df.head(int(spec.limit)).reset_index(drop=True)
            lineage.append(LineageNode(
                operation="sort",
                description=(
                    "Ordered by "
                    + ", ".join(
                        f"{o['column']}{' desc' if o['desc'] else ''}"
                        for o in resolved
                    )
                    + (f", limit {spec.limit}" if spec.limit is not None else "")
                ),
                metadata={"order_by": [dict(o) for o in resolved],
                          "limit": spec.limit},
            ))
        elif not aggregate and len(result_df.columns) and len(result_df):
            # Fingerprint algo 2: a non-aggregate query returns raw rows in
            # engine scan/hash order, which varies between identical runs — so
            # a fingerprint over them was not reproducible. Impose a canonical
            # total order over every column so the same query always stamps the
            # same bytes. order_by states an intentional order; without one
            # there is no natural order to preserve, only a nondeterministic
            # one to replace.
            cols = list(result_df.columns)
            try:
                result_df = result_df.sort_values(
                    by=cols, kind="mergesort"
                ).reset_index(drop=True)
            except TypeError:
                # A column holds values that can't be ordered natively — a SQL
                # BLOB decoded to bytes, or a LIST/array cell. Order by a stable
                # string projection so the result is still deterministic rather
                # than crashing a query that used to succeed.
                result_df = result_df.sort_values(
                    by=cols, key=lambda s: s.astype(str), kind="mergesort"
                ).reset_index(drop=True)

        lineage.append(LineageNode(
            operation="transform",
            description=f"Star-schema query executed via {engine}",
            metadata={
                "engine": engine,
                "engine_version": engine_version,
                "fingerprint_algo": CURRENT_FINGERPRINT_ALGO,
                "rows_out": len(result_df),
                "measures": dict(measures),
                # The model and the exact spec that produced this frame —
                # with the fully resolved ordering, so replay compares like
                # with like. With DataSet.fingerprint() that makes the run
                # reproducible, and it lets a report recover a declarative
                # data reference for its own sections (see tracebi.spec).
                "model": self.name,
                "query_spec": stamped_spec,
            },
        ))

        return DataSet(df=result_df, name=f"{fact}_result", lineage=lineage)

    @staticmethod
    def _hint(name: str, options) -> str:
        import difflib
        close = difflib.get_close_matches(name, [str(o) for o in options], n=1)
        return f" Did you mean '{close[0]}'?" if close else ""

    def _resolve_measures(
        self,
        measures: "dict[str, str] | list[str]",
        fact_def: "_FactDef",
    ) -> "_Resolved":
        """
        Expand the ``measures`` argument into what the engine understands.

        Accepts either the legacy ``{column: agg}`` mapping or a list of
        declared measure names, and returns:

        * ``agg_map``  — ``{output_column: agg_func}`` for the engine
        * ``derived``  — ``{output_column: row_expression}`` to materialise
                          on the fact before aggregating
        * ``ratios``   — ``{output_column: (numerator, denominator)}`` to
                          compute after aggregation

        Ad-hoc ``{column: agg}`` still works unchanged, so existing callers
        and the Explore UI are unaffected.
        """
        agg_map: dict[str, str] = {}
        derived: dict[str, str] = {}
        ratios: dict[str, tuple[str, str]] = {}
        shares: dict[str, str] = {}            # {output: base measure name}
        ranks: dict[str, str] = {}
        runnings: dict[str, str] = {}
        period_ends: dict[str, tuple[str, str]] = {}
        partitions: dict[str, tuple[str, ...]] = {}
        pops: dict[str, tuple[str, str, int, str]] = {}
        to_dates: dict[str, tuple[str, str]] = {}

        if isinstance(measures, dict):
            if not measures:
                raise ValueError("query() requires at least one measure.")
            # Ad-hoc form — the same three measure kinds a declared measure has,
            # so a chat-only agent is not walled the moment it needs a derived
            # number the model has not (yet) declared. A value may be:
            #   "sum"                              — aggregate the column
            #   ("revenue", "sum")                 — aggregate a renamed column
            #   {"expr": "market_value - cost", "agg": "sum"}
            #                                      — aggregate a row expression
            #   {"ratio": ["margin", "revenue"]}   — ratio of two other measures,
            #                                        divided AFTER aggregation
            for out_col, spec in measures.items():
                if isinstance(spec, dict):
                    if "ratio" in spec:
                        r = spec["ratio"]
                        if not (isinstance(r, (list, tuple)) and len(r) == 2):
                            raise ValueError(
                                f"measure {out_col!r}: 'ratio' must be "
                                f"[numerator, denominator], naming two other "
                                f"measures in this query."
                            )
                        ratios[out_col] = (str(r[0]), str(r[1]))
                    elif "expr" in spec:
                        derived[out_col] = str(spec["expr"])
                        agg_map[out_col] = str(spec.get("agg", "sum")).lower()
                    else:
                        raise ValueError(
                            f"measure {out_col!r}: a dict measure needs 'expr' "
                            f"(with optional 'agg', default sum) or 'ratio'; "
                            f"got keys {sorted(spec)}."
                        )
                elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                    src, func = spec
                    derived[out_col] = str(src)   # rename, not arithmetic
                    agg_map[out_col] = str(func).lower()
                else:
                    agg_map[out_col] = str(spec).lower()
            # A declared ratio's numerator/denominator are validated by the name
            # expansion below; an ad-hoc ratio references other measures in THIS
            # query by output name, so validate it here — else a typo silently
            # vanishes in _apply_ratios (which skips a ratio whose columns are
            # absent) and returns a result missing the measure, no error.
            produced = set(agg_map) | set(ratios)
            for out_col, (num, den) in ratios.items():
                for ref in (num, den):
                    if ref == out_col or ref not in produced:
                        raise ValueError(
                            f"measure {out_col!r}: ratio references {ref!r}, "
                            f"which is not another measure in this query — add "
                            f"it (e.g. {{{ref!r}: 'sum'}}). Measures here: "
                            f"{sorted(produced - {out_col})}."
                        )
            return _Resolved(agg_map, derived, ratios, shares, ranks,
                             runnings, period_ends, partitions, pops, to_dates)

        names = list(measures or [])
        if not names:
            raise ValueError("query() requires at least one measure.")

        def _expand(mname: str, seen: tuple) -> None:
            if (mname in agg_map or mname in ratios or mname in shares
                    or mname in ranks or mname in runnings
                    or mname in period_ends or mname in pops
                    or mname in to_dates):
                return
            if mname not in self._measures:
                raise ValueError(
                    f"Measure '{mname}' is not declared on model '{self.name}'."
                    f"{self._hint(mname, self._measures)} "
                    f"Declared measures: {sorted(self._measures)}. "
                    f"Pass {{'{mname}': 'sum'}} for an ad-hoc measure, or "
                    f"declare it with add_measure()."
                )
            if mname in seen:
                raise ValueError(
                    f"Measure '{mname}' is defined in terms of itself "
                    f"({' → '.join(seen + (mname,))})."
                )
            m = self._measures[mname]
            if m.kind == "simple":
                agg_map[m.name] = m.agg
                if m.column != m.name:
                    derived[m.name] = m.column
            elif m.kind == "expression":
                agg_map[m.name] = m.agg
                derived[m.name] = m.expr
            elif m.kind == "share":
                _expand(m.share, seen + (mname,))    # the measure it's a share of
                shares[m.name] = m.share
            elif m.kind == "rank":
                _expand(m.rank, seen + (mname,))
                ranks[m.name] = m.rank
                if m.partition_by:
                    partitions[m.name] = m.partition_by
            elif m.kind == "running":
                _expand(m.running, seen + (mname,))
                runnings[m.name] = m.running
                if m.partition_by:
                    partitions[m.name] = m.partition_by
            elif m.kind == "period_end":
                # value_col is a raw fact column, date is a dim attribute —
                # neither is another measure, so nothing to recurse into.
                period_ends[m.name] = m.period_end
            elif m.kind in ("offset", "growth"):
                base_m, unit, n = m.offset if m.kind == "offset" else m.growth
                _expand(base_m, seen + (mname,))     # the measure shifted in time
                pops[m.name] = (base_m, unit, n,
                                "value" if m.kind == "offset" else "growth")
            elif m.kind == "to_date":
                base_m, period = m.to_date
                _expand(base_m, seen + (mname,))     # the measure accumulated
                to_dates[m.name] = (base_m, period)
            else:  # ratio
                num, den = m.ratio
                _expand(num, seen + (mname,))
                _expand(den, seen + (mname,))
                ratios[m.name] = (num, den)

        for n in names:
            _expand(n, ())
        return _Resolved(agg_map, derived, ratios, shares, ranks,
                         runnings, period_ends, partitions, pops, to_dates)

    def _apply_derived(
        self,
        df: pd.DataFrame,
        derived: dict[str, str],
        fact_def: "_FactDef",
    ) -> pd.DataFrame:
        """
        Materialise row-level measure expressions onto the fact frame.

        Validates every referenced name against the frame's actual columns
        first — an unknown column must fail loudly, never evaluate to NaN
        and quietly poison a total.
        """
        if not derived:
            return df
        cols = set(df.columns)
        out = df.copy()
        for out_col, expression in derived.items():
            if expression in cols:            # plain rename/alias
                out[out_col] = out[expression]
                continue
            referenced = [
                t for t in _EXPR_TOKEN.findall(expression)
                if not t.isdigit()
            ]
            for token in referenced:
                if token not in cols:
                    raise ValueError(
                        f"Measure '{out_col}' references column '{token}', "
                        f"which is not on fact table "
                        f"'{fact_def.table_name}'.{self._hint(token, cols)} "
                        f"Available columns: {sorted(cols)}"
                    )
            try:
                out[out_col] = out.eval(expression)
            except Exception as exc:  # noqa: BLE001 — re-raised with context
                raise ValueError(
                    f"Measure '{out_col}' expression {expression!r} could not "
                    f"be evaluated: {type(exc).__name__}: {exc}"
                ) from exc
        return out

    def _apply_ratios(
        self,
        df: pd.DataFrame,
        ratios: dict[str, tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Compute ratio measures from already-aggregated columns.

        Division happens on the totals, so ``margin_pct`` is
        ``sum(margin) / sum(revenue)`` — not the mean of per-row ratios,
        which is a different and usually wrong number.
        """
        if not ratios:
            return df
        out = df.copy()
        for name, (num, den) in ratios.items():
            if num not in out.columns or den not in out.columns:
                continue
            denominator = out[den].replace(0, pd.NA)
            out[name] = out[num] / denominator
        return out

    def _apply_shares(
        self,
        df: pd.DataFrame,
        shares: dict[str, str],
    ) -> pd.DataFrame:
        """Compute share (% of total) measures: each row's value over the total
        of its base measure across the whole result. Order-independent and
        deterministic — no ordering needed, unlike rank or running totals."""
        if not shares:
            return df
        out = df.copy()
        for name, base in shares.items():
            if base not in out.columns:
                continue
            total = out[base].sum()
            out[name] = (out[base] / total) if total else pd.NA
        return out

    def _apply_windows(
        self,
        df: pd.DataFrame,
        ranks: dict[str, str],
        runnings: dict[str, str],
        partitions: "dict[str, tuple[str, ...]] | None" = None,
    ) -> pd.DataFrame:
        """Compute rank (1..N) and running (cumulative sum) window measures.

        Each is ordered by its OWN base measure descending, with a total
        tie-break on every other column — so ties never fall back to input
        order and the values are reproducible. Deterministic; needs no
        ``order_by`` on the query. rank 1 is the largest; running accumulates
        largest-first (the concentration / Pareto direction).

        With a ``partition_by`` the window restarts per partition: rank 1 is the
        largest WITHIN each group (top-N-per-group, paired with a ``having`` on
        the rank), and running accumulates within the group. Partition columns
        sort first so groups are contiguous; the global path (no partition) is
        byte-identical to before, so existing receipts are unmoved."""
        out = df.copy()
        partitions = partitions or {}

        def _ordered_index(base: str, part: tuple):
            # partition columns first (contiguous groups), then base descending,
            # then every remaining column ascending — a total order per partition.
            part = list(part)
            rest = [c for c in out.columns if c != base and c not in part]
            by = part + [base] + rest
            asc = [True] * len(part) + [False] + [True] * len(rest)
            try:
                return out.sort_values(by=by, ascending=asc, kind="mergesort",
                                       na_position="last").index
            except TypeError:
                return out.sort_values(by=by, ascending=asc, kind="mergesort",
                                       na_position="last",
                                       key=lambda s: s.astype(str)).index

        for name, base in ranks.items():
            if base not in out.columns:
                continue
            part = partitions.get(name) or ()
            order = _ordered_index(base, part)
            if part:
                pos = out.loc[order].groupby(list(part), sort=False).cumcount() + 1
            else:
                pos = pd.Series(range(1, len(order) + 1), index=order)
            out[name] = pos.reindex(out.index)
        for name, base in runnings.items():
            if base not in out.columns:
                continue
            part = partitions.get(name) or ()
            order = _ordered_index(base, part)
            col = out.loc[order, base]
            if part:
                keys = [out.loc[order, c] for c in part]
                col = col.groupby(keys, sort=False).cumsum()
            else:
                col = col.cumsum()
            out[name] = col.reindex(out.index)
        return out

    def _apply_pop(
        self,
        result_df: pd.DataFrame,
        pops: "dict[str, tuple[str, str, int, str]]",
        parsed_dims: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """Compute period-over-period (offset/growth) measures.

        For each row, read the base measure's value one ``(unit, n)`` earlier —
        the prior period — by shifting the query's time-grain column back and
        matching on the OTHER dimensions. ``offset`` returns that prior value;
        ``growth`` returns ``(current - prior) / prior``. YoY is
        ``('revenue', 'year', 1)``; QoQ/MoM change the unit.

        The query must group by exactly ONE declared time grain (the period
        axis) — a ``date_trunc`` attribute, whose values are period-aligned, so
        shifting a period-start lands on the earlier period-start exactly. The
        match is exact-date equality, so the result is deterministic; a period
        with no prior in the data is NULL."""
        out = result_df.copy()
        time_aliases = [
            f"{d}.{a}" for d, a in parsed_dims
            if (self._dimensions[d].derived.get(a) or {}).get("kind") == "date_trunc"
        ]
        if len(time_aliases) != 1:
            raise ValueError(
                f"Period-over-period measure(s) {sorted(pops)} need exactly one "
                f"declared time grain among the query's dimensions (the period "
                f"axis), found {time_aliases or 'none'}. Group by a time-grain "
                f"dimension (add_time_grain) — e.g. dimensions=['dim_date.month']."
            )
        time_alias = time_aliases[0]
        others = [f"{d}.{a}" for d, a in parsed_dims
                  if f"{d}.{a}" != time_alias]
        tcol = pd.to_datetime(out[time_alias])
        for name, (base, unit, n, mode) in pops.items():
            if base not in out.columns:
                continue
            # Match each row to the row at (same other dims, this time − n·unit).
            left = out[others].copy()
            left["__shifted__"] = tcol - _pop_offset(unit, n)
            prior = out[[*others, base]].copy()
            prior["__shifted__"] = pd.to_datetime(out[time_alias])
            prior = prior.rename(columns={base: "__prior__"})
            merged = left.merge(prior, on=[*others, "__shifted__"], how="left")
            prior_s = pd.Series(merged["__prior__"].to_numpy(), index=out.index)
            if mode == "value":
                out[name] = prior_s
            else:  # growth: (current − prior) / prior, null-safe
                denom = prior_s.where(prior_s != 0)
                out[name] = (out[base] - prior_s) / denom
        return out

    def _apply_to_date(
        self,
        result_df: pd.DataFrame,
        to_dates: "dict[str, tuple[str, str]]",
        parsed_dims: list[tuple[str, str]],
    ) -> pd.DataFrame:
        """Compute to-date cumulative (YTD/QTD/MTD) measures.

        A running total from the start of each reset period (year/quarter/month/
        week) up to the current period — accumulating the base measure over the
        query's finer time grain, within each other-dimension group. Ordered by
        time ascending, so YTD revenue in March is Jan+Feb+Mar.

        The query must group by exactly one declared time grain, and that grain
        must be FINER than the reset period (YTD/QTD over monthly; MTD needs a
        daily/weekly grain). ``(other dims, time)`` is unique in the result, so
        the ascending-time order within each reset partition is total — the
        cumulative is deterministic."""
        out = result_df.copy()
        time_grains = [
            (f"{d}.{a}", self._dimensions[d].derived.get(a) or {})
            for d, a in parsed_dims
            if (self._dimensions[d].derived.get(a) or {}).get("kind") == "date_trunc"
        ]
        if len(time_grains) != 1:
            raise ValueError(
                f"To-date measure(s) {sorted(to_dates)} need exactly one declared "
                f"time grain among the query's dimensions (the period axis), "
                f"found {[a for a, _ in time_grains] or 'none'}. Group by a "
                f"time-grain dimension (add_time_grain)."
            )
        time_alias, grain_def = time_grains[0]
        query_grain = grain_def.get("grain")
        others = [f"{d}.{a}" for d, a in parsed_dims if f"{d}.{a}" != time_alias]
        tcol = pd.to_datetime(out[time_alias])
        for name, (base, period) in to_dates.items():
            if base not in out.columns:
                continue
            if _GRAIN_RANK.get(period, 0) <= _GRAIN_RANK.get(query_grain, 0):
                raise ValueError(
                    f"To-date measure '{name}': reset period '{period}' must be "
                    f"coarser than the query's time grain '{query_grain}' — a "
                    f"{period}-to-date over a {query_grain} grain has one row per "
                    f"bucket. Group by a finer grain, or pick a coarser period."
                )
            part = tcol.dt.to_period(_PERIOD_FREQ[period])
            keyframe = out[others].copy()
            keyframe["__part__"] = part.astype(str)
            keyframe["__t__"] = tcol
            order = keyframe.sort_values(
                by=[*others, "__part__", "__t__"], kind="mergesort").index
            keys = ([keyframe.loc[order, c] for c in others]
                    + [keyframe.loc[order, "__part__"]])
            cum = out.loc[order, base].groupby(keys, sort=False).cumsum()
            out[name] = cum.reindex(out.index)
        return out

    def _apply_period_ends(
        self,
        result_df: pd.DataFrame,
        period_ends: dict[str, tuple[str, str]],
        period_end_dims: dict[str, tuple[str, str]],
        *,
        fact_df: pd.DataFrame,
        fact_def: "_FactDef",
        dim_dfs: dict[str, pd.DataFrame],
        parsed_dims: list[tuple[str, str]],
        filter_tree: "_FilterNode",
    ) -> pd.DataFrame:
        """Compute semi-additive (period_end) measures.

        A stock — AUM, NAV, headcount — is additive across ordinary dimensions
        but NOT across time: Jan AUM + Feb AUM is not "AUM". For each such
        measure this runs an internal aggregation grouped by the query's
        dimensions PLUS the snapshot date (reusing the engine, so the same
        filters and joins apply), then in pandas keeps each group's LATEST
        snapshot rows and sums the balance across them. So the balance is summed
        across non-time dimensions and taken at period-end over time.

        With no dimensions the group is the whole result — one global latest
        snapshot. With a time grain among the dimensions (``dim_date.month``),
        each period takes its own latest snapshot. Deterministic: max-date
        selection and a sum are both order-independent, so no ordering
        machinery and no fingerprint-algorithm concern.

        Assumes entities snapshot on common dates (the month-end convention).
        Where entities carry different latest dates, a per-entity variant is
        future work; documented on :meth:`add_measure`.
        """
        out = result_df.copy()
        group_aliases = [f"{d}.{a}" for d, a in parsed_dims]
        for name, (value_col, _date_ref) in period_ends.items():
            date_dim, date_attr = period_end_dims[name]
            date_alias = f"{date_dim}.{date_attr}"
            # Group by (query dims + the raw snapshot date), summing the
            # balance. The date is added only if it is not already grouped on.
            inner_dims = list(parsed_dims)
            if (date_dim, date_attr) not in inner_dims:
                inner_dims.append((date_dim, date_attr))
            snap = self._execute_duckdb(
                fact_df=fact_df, fact_def=fact_def, dim_dfs=dim_dfs,
                parsed_dims=inner_dims, measures={value_col: "sum"},
                filter_tree=filter_tree, aggregate=True, lineage=[],
            )
            if group_aliases:
                grp = snap.groupby(group_aliases, dropna=False)
                latest = grp[date_alias].transform("max")
                period = snap[snap[date_alias] == latest]
                reduced = (period.groupby(group_aliases, dropna=False)[value_col]
                           .sum().reset_index()
                           .rename(columns={value_col: name}))
                out = out.merge(reduced, on=group_aliases, how="left")
            else:
                latest = snap[date_alias].max()
                out[name] = snap.loc[snap[date_alias] == latest, value_col].sum()
        return out

    @staticmethod
    def _looks_like_per_row_ratio(series: "pd.Series") -> bool:
        """Value-based signal that a column holds per-row ratios (marks,
        multiples) rather than an additive quantity: floating, mostly
        non-integer, and a robust centre within a 2× band of 1.0. Additive
        business quantities (money, counts, sizes) are essentially never centred
        at ~1, so this fires narrowly. Uses the median, so a heavy tail (one
        316,000× warrant) does not hide the ~1.0 centre; gated on row count so
        small fixtures never trip it."""
        if not pd.api.types.is_float_dtype(series):
            return False
        a = series[series.notna()].abs()
        a = a[a != float("inf")]
        if len(a) < _RATE_MIN_ROWS:
            return False
        if float((a.mod(1.0) != 0.0).mean()) < 0.5:   # mostly non-integer
            return False
        med = float(a.median())
        return _RATE_BAND_LO <= med <= _RATE_BAND_HI

    def _parse_filters(
        self,
        filters: dict[str, Any],
        fact_def: "_FactDef",
        fact_cols: set,
    ) -> "_FilterNode":
        """
        Turn the user-facing filter dict into a validated boolean tree.

        Accepts three spellings per column condition::

            {"status": "shipped"}                 # eq
            {"region": ["NE", "SE"]}              # in
            {"revenue": {"gte": 1000}}            # explicit operator

        A key containing a dot is resolved as ``dim_name.attribute`` against
        the registered dimensions.

        The reserved keys ``"or"`` and ``"and"`` take a list of condition
        groups (each itself a filter dict) and build a boolean node — every
        other key in the same dict AND-s with it::

            {"or": [{"sector": "Tech"}, {"rating": "AAA"}]}
            {"status": "active",
             "or": [{"region": "West"}, {"revenue": {"gte": 1000}}]}

        The all-AND case (no ``or``/``and``) is a single AND node over leaves —
        byte-equivalent row selection to before boolean grouping existed.
        """
        return self._parse_filter_dict(filters, fact_def, fact_cols)

    def _parse_filter_dict(
        self, node: Any, fact_def: "_FactDef", fact_cols: set,
    ) -> "_FilterNode":
        if not isinstance(node, dict):
            raise ValueError(
                f"A filter group must be a dict of conditions, got "
                f"{type(node).__name__}: {node!r}."
            )
        children: list[_FilterNode] = []
        for key, raw in node.items():
            kl = str(key).lower()
            if kl in ("or", "and"):
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise ValueError(
                        f"filter '{kl}' takes a non-empty list of condition "
                        f"groups (each a filter dict), got {raw!r}."
                    )
                subs = tuple(
                    self._parse_filter_dict(sub, fact_def, fact_cols)
                    for sub in raw
                )
                children.append(_FilterNode(op=kl, children=subs))
            else:
                children.append(_FilterNode(
                    op="leaf",
                    predicate=self._parse_one_predicate(
                        key, raw, fact_def, fact_cols),
                ))
        if len(children) == 1:
            return children[0]
        return _FilterNode(op="and", children=tuple(children))

    def _parse_one_predicate(
        self, key: str, raw: Any, fact_def: "_FactDef", fact_cols: set,
    ) -> "_Predicate":
        """Validate and normalise one column condition into a ``_Predicate``."""
        dim_name = attribute = None
        if "." in key:
            cand_dim, cand_attr = key.split(".", 1)
            if cand_dim not in self._dimensions:
                raise ValueError(
                    f"Filter '{key}' refers to dimension '{cand_dim}', "
                    f"which is not registered in model '{self.name}'."
                    f"{self._hint(cand_dim, self._dimensions)} "
                    f"Available dimensions: {sorted(self._dimensions)}"
                )
            dim_name, attribute = cand_dim, cand_attr
        elif key not in fact_cols:
            # Not a fact column — maybe they meant a dimension attribute.
            matches = [
                f"{dn}.{key}" for dn, dd in self._dimensions.items()
                if key in (dd.attributes or []) or key == dd.key_col
            ]
            extra = (
                f" It exists on {matches[0].split('.')[0]}; "
                f"reference it as '{matches[0]}'."
                if matches else self._hint(key, fact_cols)
            )
            raise ValueError(
                f"Filter column '{key}' not found on fact table "
                f"'{fact_def.table_name}'.{extra} "
                f"Available columns: {sorted(fact_cols)}"
            )

        # Normalise the value into (op, value)
        if isinstance(raw, dict):
            if len(raw) != 1:
                raise ValueError(
                    f"Filter '{key}' must specify exactly one operator, "
                    f"got {sorted(raw)}. Combine conditions by passing "
                    f"separate filter entries."
                )
            op, value = next(iter(raw.items()))
            op = str(op).lower()
            if op not in FILTER_OPS:
                raise ValueError(
                    f"Unknown filter operator '{op}' for '{key}'."
                    f"{self._hint(op, FILTER_OPS)} "
                    f"Supported: {', '.join(FILTER_OPS)}"
                )
        elif isinstance(raw, (list, tuple, set)):
            op, value = "in", list(raw)
        else:
            op, value = "eq", raw

        if op in _LIST_OPS and not isinstance(value, (list, tuple, set)):
            raise ValueError(
                f"Filter '{key}' with operator '{op}' needs a list of "
                f"values, got {type(value).__name__}."
            )
        if op == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"Filter '{key}' with operator 'between' needs a "
                    f"two-element [low, high], got {value!r}."
                )
        if op in _LIST_OPS:
            value = list(value)

        return _Predicate(
            target=key, op=op, value=value,
            dim_name=dim_name, attribute=attribute,
        )

    def _filter_where(self, node: "Optional[_FilterNode]") -> "tuple[str, list]":
        """Render a filter tree to a parameterised SQL boolean expression.

        Leaves parameterise through ``_predicate_sql``; ``and``/``or`` nodes
        join their children with the matching keyword and parenthesise. Params
        come out left-to-right, matching the positional ``?`` order DuckDB
        expects. Empty groups contribute nothing."""
        if node is None:
            return "", []
        if node.op == "leaf":
            p = node.predicate
            col = (f'dim_{p.dim_name}."{p.attribute}"' if p.is_dim
                   else f'fact."{p.target}"')
            return self._predicate_sql(p, col)
        parts: list[str] = []
        params: list[Any] = []
        for child in node.children:
            sql, ps = self._filter_where(child)
            if sql:
                parts.append(sql)
                params.extend(ps)
        if not parts:
            return "", []
        if len(parts) == 1:
            return parts[0], params
        joiner = " AND " if node.op == "and" else " OR "
        return "(" + joiner.join(parts) + ")", params

    def _apply_having(
        self,
        df: "pd.DataFrame",
        having: dict[str, Any],
        lineage: list[LineageNode],
    ) -> "pd.DataFrame":
        """Apply post-aggregation (HAVING) filters on the result frame.

        ``filters`` are WHERE (before aggregation); ``having`` is HAVING
        (after), so it filters the aggregated measures and ratios. Targets
        must be result columns. It runs in pandas on the already-aggregated
        frame — not as SQL HAVING — so it can also filter ratio measures,
        which are computed after the engine.
        """
        cols = list(df.columns)
        for key, raw in having.items():
            if key not in df.columns:
                raise ValueError(
                    f"having column '{key}' is not a result column of this "
                    f"query.{self._hint(key, cols)} Result columns: {cols}. "
                    f"`having` filters aggregated measures/ratios; to filter "
                    f"raw rows before aggregation use `filters`."
                )
            # Normalise the value into (op, value) — same spellings as filters.
            if isinstance(raw, dict):
                if len(raw) != 1:
                    raise ValueError(
                        f"having '{key}' must specify exactly one operator, "
                        f"got {sorted(raw)}."
                    )
                op, value = next(iter(raw.items()))
                op = str(op).lower()
            elif isinstance(raw, (list, tuple, set)):
                op, value = "in", list(raw)
            else:
                op, value = "eq", raw
            if op not in FILTER_OPS:
                raise ValueError(
                    f"Unknown having operator '{op}' for '{key}'."
                    f"{self._hint(op, FILTER_OPS)} "
                    f"Supported: {', '.join(FILTER_OPS)}"
                )

            s = df[key]
            if op == "eq":         mask = s == value
            elif op == "ne":       mask = s != value
            elif op == "gt":       mask = s > value
            elif op == "gte":      mask = s >= value
            elif op == "lt":       mask = s < value
            elif op == "lte":      mask = s <= value
            elif op == "between":  mask = s.between(value[0], value[1])
            elif op == "in":       mask = s.isin(list(value))
            elif op == "not_in":   mask = ~s.isin(list(value))
            elif op == "is_null":  mask = s.isna()
            elif op == "not_null": mask = s.notna()
            elif op == "contains": mask = s.astype(str).str.contains(str(value), na=False)
            else:
                raise ValueError(f"Unhandled having operator '{op}'")

            df = df[mask]
            lineage.append(LineageNode(
                operation="filter",
                description=f"Having: {key} {op} {value!r}",
                metadata={"target": key, "operator": op,
                          "value": value, "on": "aggregate"},
            ))

        return df.reset_index(drop=True)

    def _validate_dim_predicates(
        self,
        preds: list[_Predicate],
        dim_dfs: dict[str, pd.DataFrame],
    ) -> None:
        """Check that every dimension predicate's attribute actually exists."""
        for p in preds:
            if not p.is_dim:
                continue
            dim_def = self._dimensions[p.dim_name]
            declared = dim_def.attributes
            if declared and p.attribute not in declared and p.attribute != dim_def.key_col:
                raise ValueError(
                    f"Filter attribute '{p.attribute}' is not declared on "
                    f"dimension '{p.dim_name}'.{self._hint(p.attribute, declared)} "
                    f"Declared attributes: {declared}"
                )
            cols = set(dim_dfs[p.dim_name].columns)
            if p.attribute not in cols:
                raise ValueError(
                    f"Filter attribute '{p.attribute}' not found on dimension "
                    f"table '{dim_def.table_name}'."
                    f"{self._hint(p.attribute, cols)} "
                    f"Available columns: {sorted(cols)}"
                )

    def _check_dimension_keys(
        self,
        dim_dfs: dict[str, pd.DataFrame],
        parsed_dims: list[tuple[str, str]],
        fact_df: pd.DataFrame,
        fact_def: "_FactDef",
        allow_fanout: bool,
        lineage: list[LineageNode],
    ) -> None:
        """
        Reject dimensions whose key is not unique.

        A star-schema join assumes one dimension row per key. When the key
        repeats, the LEFT JOIN multiplies fact rows and every additive
        measure is silently inflated — the query returns a confident,
        fully-lineaged, wrong number. Same principle as
        ``_validate_query_columns``: never return a silently wrong result.

        With ``allow_fanout=True`` the join proceeds (legitimate for
        many-to-many) but the opt-in is recorded as a warning lineage node
        so the audit trail shows the multiplication was intentional.
        """
        for dim_name in {d for d, _ in parsed_dims}:
            dim_def = self._dimensions[dim_name]
            dim_df = dim_dfs[dim_name]
            key = dim_def.key_col

            n_null = int(dim_df[key].isna().sum())
            dup_mask = dim_df[key].duplicated(keep=False)
            n_dup_rows = int(dup_mask.sum())

            if n_dup_rows:
                dup_values = dim_df.loc[dup_mask, key].drop_duplicates()
                sample = [repr(v) for v in dup_values.head(3).tolist()]
                if len(dup_values) > 3:
                    sample.append("…")
                # Rows the fact side would gain, for the keys it actually uses.
                fk_col = fact_def.foreign_keys.get(dim_name, key)
                if fk_col in fact_df.columns:
                    counts = dim_df[key].value_counts()
                    rows_after = int(fact_df[fk_col].map(counts).fillna(1).sum())
                else:
                    rows_after = len(fact_df)
                factor = (rows_after / len(fact_df)) if len(fact_df) else 1.0

                if not allow_fanout:
                    raise ValueError(
                        f"Dimension '{dim_name}' has a non-unique key: "
                        f"'{dim_def.table_name}.{key}' repeats across "
                        f"{n_dup_rows} rows ({len(dup_values)} duplicated "
                        f"value(s): {', '.join(sample)}). Joining would fan "
                        f"out the fact table {len(fact_df)} → {rows_after} "
                        f"rows (x{factor:.2f}) and inflate every additive "
                        f"measure. Deduplicate the dimension, pick a key "
                        f"column that is unique, or pass allow_fanout=True "
                        f"if the multiplication is intended."
                    )

                lineage.append(LineageNode(
                    operation="warning",
                    description=(
                        f"Fan-out allowed on dimension '{dim_name}': "
                        f"key '{key}' is not unique; measures are multiplied "
                        f"x{factor:.2f}"
                    ),
                    metadata={
                        "dimension": dim_name,
                        "key_column": key,
                        "duplicate_rows": n_dup_rows,
                        "duplicate_values": len(dup_values),
                        "rows_before": len(fact_df),
                        "rows_after": rows_after,
                        "fanout_factor": round(factor, 4),
                        "allow_fanout": True,
                    },
                ))

            if n_null:
                lineage.append(LineageNode(
                    operation="warning",
                    description=(
                        f"Dimension '{dim_name}' key '{key}' contains "
                        f"{n_null} null value(s); those rows cannot be "
                        f"matched by the join"
                    ),
                    metadata={
                        "dimension": dim_name,
                        "key_column": key,
                        "null_keys": n_null,
                    },
                ))

    def validate(self) -> dict[str, Any]:
        """
        Check the model's declared structure against the actual data.

        Loads only each dimension's key column and reports uniqueness and
        null status. Returns a structured result rather than printing, so
        the CLI, the web API, and agent tooling can all consume it::

            {"ok": bool, "dimensions": [{...}], "errors": [str], "warnings": [str]}

        Call this before running queries — a non-unique dimension key is the
        difference between a correct number and a silently inflated one.
        """
        dims: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        for dim_name, dim_def in self._dimensions.items():
            entry: dict[str, Any] = {
                "dimension": dim_name,
                "table": dim_def.table_name,
                "key_column": dim_def.key_col,
            }
            try:
                df = self.load(dim_def.table_name, columns=[dim_def.key_col]).to_pandas()
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                entry.update(ok=False, error=str(exc))
                errors.append(f"Dimension '{dim_name}': could not load key column — {exc}")
                dims.append(entry)
                continue

            n_dup = int(df[dim_def.key_col].duplicated(keep=False).sum())
            n_null = int(df[dim_def.key_col].isna().sum())
            entry.update(
                rows=len(df),
                duplicate_rows=n_dup,
                null_keys=n_null,
                key_unique=(n_dup == 0),
                ok=(n_dup == 0),
            )
            if n_dup:
                errors.append(
                    f"Dimension '{dim_name}': key "
                    f"'{dim_def.table_name}.{dim_def.key_col}' is not unique "
                    f"({n_dup} duplicated rows) — joins will inflate measures."
                )
            if n_null:
                warnings.append(
                    f"Dimension '{dim_name}': key "
                    f"'{dim_def.table_name}.{dim_def.key_col}' has "
                    f"{n_null} null value(s)."
                )
            dims.append(entry)

        return {
            "ok": not errors,
            "model": self.name,
            "dimensions": dims,
            "errors": errors,
            "warnings": warnings,
        }

    def _validate_query_columns(
        self,
        fact_def,
        fact_df: pd.DataFrame,
        dim_dfs: dict[str, pd.DataFrame],
        parsed_dims: list[tuple[str, str]],
        measures: dict[str, str],
        filters: dict[str, Any],
    ) -> None:
        """Raise ValueError for any measure, filter, or dimension-attribute
        reference that doesn't exist. The query engine relies on this —
        a typo must fail loudly, never return a silently wrong result."""
        fact_cols = set(fact_df.columns)
        for col in measures:
            if col not in fact_cols:
                raise ValueError(
                    f"Measure column '{col}' not found on fact table "
                    f"'{fact_def.table_name}'.{self._hint(col, fact_cols)} "
                    f"Available columns: {sorted(fact_cols)}"
                )
        for col in filters:
            if col not in fact_cols:
                raise ValueError(
                    f"Filter column '{col}' not found on fact table "
                    f"'{fact_def.table_name}'.{self._hint(col, fact_cols)} "
                    f"Available columns: {sorted(fact_cols)}"
                )
        for dim_name, attribute in parsed_dims:
            dim_def = self._dimensions[dim_name]
            dim_cols = set(dim_dfs[dim_name].columns)
            derived = dim_def.derived.get(attribute)
            if derived is not None:
                # A declared derived attribute (e.g. a date grain): valid as a
                # group key; validate its SOURCE column exists instead.
                src = derived["source"]
                if src not in dim_cols:
                    raise ValueError(
                        f"Derived attribute '{attribute}' on dimension "
                        f"'{dim_name}' reads source column '{src}', which is not "
                        f"on table '{dim_def.table_name}'."
                        f"{self._hint(src, dim_cols)} "
                        f"Available columns: {sorted(dim_cols)}"
                    )
                continue
            declared = dim_def.attributes
            if declared and attribute not in declared and attribute != dim_def.key_col:
                raise ValueError(
                    f"Attribute '{attribute}' is not declared on dimension "
                    f"'{dim_name}'.{self._hint(attribute, declared + list(dim_def.derived))} "
                    f"Declared attributes: {declared}"
                    + (f"; derived: {sorted(dim_def.derived)}"
                       if dim_def.derived else "")
                )
            if attribute not in dim_cols:
                raise ValueError(
                    f"Attribute '{attribute}' not found on dimension table "
                    f"'{dim_def.table_name}'.{self._hint(attribute, dim_cols)} "
                    f"Available columns: {sorted(dim_cols)}"
                )

    def _declared_fact_columns(self, fact_def: "_FactDef") -> set[str]:
        """
        Every fact-table column the model *declares*: the fact's measure
        columns and foreign keys, plus any column a named measure reads
        (directly or inside a row expression). A subset of the table's
        actual columns — knowable without loading a row.
        """
        cols = set(fact_def.measures) | set(fact_def.foreign_keys.values())
        for m in self._measures.values():
            if m.column:
                cols.add(m.column)
            if m.expr:
                cols |= {
                    t for t in _EXPR_TOKEN.findall(m.expr) if not t.isdigit()
                }
        return cols

    def check_query_spec(
        self, spec: QuerySpec
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """
        Validate a :class:`QuerySpec` against the model's declared structure,
        without loading any data.

        The declaration-time twin of the execution-time checks
        (``_validate_query_columns``, ``_parse_filters``,
        ``_validate_dim_predicates``), kept beside them so the rules cannot
        drift. Whatever the model itself declares — facts, named measures,
        dimensions and their attributes, the closed aggregation-function
        set — is authoritative, and a mismatch is an **error**. A reference
        only the physical table can confirm (an ad-hoc measure column or a
        bare filter column beyond the declared ones) is a **warning**:
        execution re-checks it against the actual columns and fails loudly
        there.

        Returns ``(errors, warnings)``, each a list of ``(subpath, message)``
        pairs relative to the query — e.g. ``("filters.region", "…")`` — so
        a caller can prefix its own location (``sections[0].data.query``).
        """
        errors: list[tuple[str, str]] = []
        warnings: list[tuple[str, str]] = []

        fact_def = self._facts.get(spec.fact)
        if fact_def is None:
            errors.append((
                "fact",
                f"'{spec.fact}' is not a fact on model '{self.name}'."
                f"{self._hint(spec.fact, self._facts)} "
                f"Available: {sorted(self._facts)}",
            ))
            return errors, warnings

        declared_cols = self._declared_fact_columns(fact_def)

        # ── Measures ──────────────────────────────────────────
        if isinstance(spec.measures, dict):
            for out_col, mspec in spec.measures.items():
                if isinstance(mspec, (tuple, list)) and len(mspec) == 2:
                    src, func = str(mspec[0]), str(mspec[1])
                else:
                    src, func = str(out_col), str(mspec)
                if not _valid_agg(func):
                    errors.append((
                        f"measures.{out_col}",
                        f"unsupported aggregation '{func}'."
                        f"{self._hint(func, sorted(_AGG_FUNCS))} "
                        f"Supported: {_agg_choices()}",
                    ))
                if src not in declared_cols:
                    warnings.append((
                        f"measures.{out_col}",
                        f"column '{src}' is not among the declared columns "
                        f"of fact '{spec.fact}'."
                        f"{self._hint(src, declared_cols)} "
                        f"Declared: {sorted(declared_cols)}. It cannot be "
                        f"verified before execution, which checks it "
                        f"against the actual table.",
                    ))
        else:
            for m in spec.measures or ():
                if not isinstance(m, str):
                    errors.append((
                        "measures",
                        f"list-form measures must be declared measure names "
                        f"(strings); got {type(m).__name__}: {m!r}. Use the "
                        f"dict form {{column: agg}} for ad-hoc measures.",
                    ))
                    continue
                if m not in self._measures:
                    errors.append((
                        "measures",
                        f"'{m}' is not a declared measure on model "
                        f"'{self.name}'.{self._hint(m, self._measures)} "
                        f"Declared: {sorted(self._measures)}",
                    ))

        # ── Dimensions ────────────────────────────────────────
        for ref in spec.dimensions or ():
            ref_s = str(ref)
            if "." not in ref_s:
                errors.append((
                    "dimensions",
                    f"'{ref_s}' must use dot notation: 'dim_name.attribute'",
                ))
                continue
            dim_name, attribute = ref_s.split(".", 1)
            err = self._check_declared_dim_ref(dim_name, attribute)
            if err:
                errors.append(("dimensions", err))

        # ── Filters ───────────────────────────────────────────
        # A filter group AND-s its keys; the reserved "or"/"and" keys hold a
        # list of sub-groups, so validation recurses into them exactly as the
        # execution-time parser does.
        def _check_filter_group(group: Any, path: str) -> None:
            if not isinstance(group, dict):
                errors.append((path, "a filter group must be a dict of "
                                     "conditions."))
                return
            for key, val in group.items():
                key_s = str(key)
                sub = f"{path}.{key_s}"
                if key_s.lower() in ("or", "and"):
                    if not isinstance(val, (list, tuple)) or not val:
                        errors.append((sub, f"'{key_s.lower()}' takes a "
                                            f"non-empty list of condition "
                                            f"groups."))
                        continue
                    for i, item in enumerate(val):
                        _check_filter_group(item, f"{sub}[{i}]")
                    continue
                if isinstance(val, dict):
                    for op in val:
                        if str(op) not in FILTER_OPS:
                            errors.append((
                                f"{sub}.{op}",
                                f"unknown filter operator '{op}'."
                                f"{self._hint(str(op), sorted(FILTER_OPS))} "
                                f"Supported: {', '.join(sorted(FILTER_OPS))}",
                            ))
                if "." in key_s:
                    dim_name, attribute = key_s.split(".", 1)
                    err = self._check_declared_dim_ref(dim_name, attribute)
                    if err:
                        errors.append((sub, err))
                    continue
                if key_s in declared_cols:
                    continue
                matches = [
                    f"{dn}.{key_s}" for dn, dd in self._dimensions.items()
                    if key_s in (dd.attributes or []) or key_s == dd.key_col
                ]
                if matches:
                    # A warning, not an error: the name may also be a physical
                    # column on the fact table (denormalised facts do this), and
                    # a bare fact-column filter and a dimension-attribute filter
                    # have different semantics — execution accepts the former.
                    warnings.append((
                        sub,
                        f"'{key_s}' is not a declared column on fact "
                        f"'{spec.fact}' but matches dimension attribute "
                        f"'{matches[0]}'. If you mean the dimension, reference "
                        f"it as '{matches[0]}'; a bare name filters the fact "
                        f"table's own column, which is checked at execution.",
                    ))
                else:
                    warnings.append((
                        sub,
                        f"filter column '{key_s}' is not among the declared "
                        f"columns of fact '{spec.fact}'."
                        f"{self._hint(key_s, declared_cols)} "
                        f"Declared: {sorted(declared_cols)}. It cannot be "
                        f"verified before execution, which checks it against "
                        f"the actual table.",
                    ))

        if spec.filters:
            _check_filter_group(spec.filters, "filters")

        # ── Ordering ──────────────────────────────────────────
        if spec.limit is not None and not (spec.order_by or ()):
            errors.append((
                "limit",
                "limit without order_by is refused: without a stated "
                "ordering, 'first N rows' silently masquerades as 'top N'. "
                "Add order_by to state the ranking.",
            ))
        if spec.order_by:
            try:
                normalized = _normalize_order_by(spec.order_by)
            except ValueError as exc:
                errors.append(("order_by", str(exc)))
                normalized = ()
            result_cols: Optional[list[str]] = None
            if not errors:
                try:
                    result_cols = self.spec_result_columns(spec)
                except ValueError:
                    result_cols = None   # unresolvable measures already reported
            for i, o in enumerate(normalized):
                col = o["column"]
                if result_cols is not None and col not in result_cols:
                    errors.append((
                        f"order_by[{i}]",
                        f"'{col}' is not a result column of this query."
                        f"{self._hint(col, result_cols)} "
                        f"Result columns: {sorted(result_cols)}",
                    ))

        return errors, warnings

    def _check_declared_dim_ref(
        self, dim_name: str, attribute: str
    ) -> Optional[str]:
        """One ``dim_name.attribute`` reference against the declarations."""
        dim_def = self._dimensions.get(dim_name)
        if dim_def is None:
            return (
                f"'{dim_name}' is not a dimension on model '{self.name}'."
                f"{self._hint(dim_name, self._dimensions)} "
                f"Available: {sorted(self._dimensions)}"
            )
        declared = dim_def.attributes
        if declared and attribute not in declared and attribute != dim_def.key_col:
            return (
                f"attribute '{attribute}' is not declared on dimension "
                f"'{dim_name}'.{self._hint(attribute, declared)} "
                f"Declared attributes: {declared}"
            )
        return None

    def spec_result_columns(self, spec: QuerySpec) -> list[str]:
        """
        The column names a :class:`QuerySpec`'s result frame will carry,
        computed from declarations alone: each dimension reference as
        written (``dim_name.attribute``) plus every measure output —
        ratio components included — exactly as ``execute()`` produces
        them via ``_resolve_measures``. Raises ``ValueError`` when the
        fact or measures cannot be resolved; check the spec first.
        """
        fact_def = self._facts.get(spec.fact)
        if fact_def is None:
            raise ValueError(
                f"Fact '{spec.fact}' is not registered in model '{self.name}'."
            )
        r = self._resolve_measures(spec.measures, fact_def)
        return (
            [str(d) for d in (spec.dimensions or ())]
            + list(r.agg_map)
            + list(r.ratios)
            + list(r.shares)
            + list(r.ranks)
            + list(r.runnings)
            + list(r.period_ends)
            + list(r.pops)
            + list(r.to_dates)
        )

    @staticmethod
    def _predicate_sql(pred: _Predicate, column_sql: str) -> tuple[str, list]:
        """Render one predicate as parameterised SQL."""
        op, v = pred.op, pred.value
        if op == "eq":       return f"{column_sql} = ?", [v]
        if op == "ne":       return f"{column_sql} <> ?", [v]
        if op == "gt":       return f"{column_sql} > ?", [v]
        if op == "gte":      return f"{column_sql} >= ?", [v]
        if op == "lt":       return f"{column_sql} < ?", [v]
        if op == "lte":      return f"{column_sql} <= ?", [v]
        if op == "between":  return f"{column_sql} BETWEEN ? AND ?", [v[0], v[1]]
        if op == "is_null":  return f"{column_sql} IS NULL", []
        if op == "not_null": return f"{column_sql} IS NOT NULL", []
        if op == "contains": return f"CAST({column_sql} AS VARCHAR) LIKE ?", [f"%{v}%"]
        if op in _LIST_OPS:
            if not v:
                # An empty IN matches nothing; NOT IN matches everything.
                return ("1 = 0", []) if op == "in" else ("1 = 1", [])
            marks = ", ".join("?" for _ in v)
            neg = "NOT " if op == "not_in" else ""
            return f"{column_sql} {neg}IN ({marks})", list(v)
        raise ValueError(f"Unhandled filter operator '{op}'")

    # ── DuckDB engine ──────────────────────────────────────────

    def _execute_duckdb(
        self,
        *,
        fact_df: pd.DataFrame,
        fact_def: _FactDef,
        dim_dfs: dict[str, pd.DataFrame],
        parsed_dims: list[tuple[str, str]],
        measures: dict[str, str],
        filter_tree: "_FilterNode",
        aggregate: bool,
        lineage: list[LineageNode],
    ) -> pd.DataFrame:
        import duckdb

        con = duckdb.connect(":memory:")
        try:
            con.register("fact", fact_df)
            for dim_name, dim_df in dim_dfs.items():
                con.register(f"dim_{dim_name}", dim_df)

            group_cols_sql: list[str] = []
            select_cols_sql: list[str] = []
            for dim_name, attribute in parsed_dims:
                alias = f'"{dim_name}.{attribute}"'
                derived = self._dimensions[dim_name].derived.get(attribute)
                if derived and derived["kind"] == "date_trunc":
                    # A declared date grain: group by date_trunc(grain, source).
                    # grain is validated against a closed set at declaration, so
                    # it is never attacker/free text here.
                    col_sql = (f"date_trunc('{derived['grain']}', "
                               f'dim_{dim_name}."{derived["source"]}")')
                elif derived and derived["kind"] == "bin":
                    # A declared value band: CASE the source into ranges. edges
                    # are validated floats (never free text); labels are quoted
                    # with '' escaping. NULL groups as NULL, not the top band.
                    src_sql = f'dim_{dim_name}."{derived["source"]}"'
                    edges, labels = derived["edges"], derived["labels"]
                    whens = [f"WHEN {src_sql} IS NULL THEN NULL"]
                    for i, edge in enumerate(edges):
                        lit = labels[i].replace("'", "''")
                        whens.append(f"WHEN {src_sql} < {edge} THEN '{lit}'")
                    else_lit = labels[-1].replace("'", "''")
                    col_sql = ("CASE " + " ".join(whens)
                               + f" ELSE '{else_lit}' END")
                else:
                    col_sql = f'dim_{dim_name}."{attribute}"'
                select_cols_sql.append(f"{col_sql} AS {alias}")
                group_cols_sql.append(alias)

            if aggregate:
                for col, func in measures.items():
                    func_l = func.lower()
                    frac = _percentile_fraction(func_l)
                    if not (frac is not None or func_l in _AGG_FUNCS):
                        raise ValueError(
                            f"Unsupported aggregation '{func}' for measure '{col}'"
                        )
                    if frac is not None:
                        # Interpolated percentile (p90 → quantile_cont(col, 0.9)).
                        # Deterministic; the fraction comes from a validated
                        # p0–p100 integer, never free text.
                        select_cols_sql.append(
                            f'quantile_cont(fact."{col}", {frac}) AS "{col}"'
                        )
                    elif func_l == "nunique":
                        select_cols_sql.append(
                            f'COUNT(DISTINCT fact."{col}") AS "{col}"'
                        )
                    else:
                        sql_func = "AVG" if func_l == "mean" else func_l.upper()
                        select_cols_sql.append(
                            f'{sql_func}(fact."{col}") AS "{col}"'
                        )
            else:
                for col in measures.keys():
                    select_cols_sql.append(f'fact."{col}" AS "{col}"')

            # Join every loaded dimension — those grouped by and those only
            # referenced in a filter.
            from_clause = "fact"
            for dim_name in dim_dfs:
                dim_def = self._dimensions[dim_name]
                fk_col = fact_def.foreign_keys.get(dim_name, dim_def.key_col)
                from_clause += (
                    f' LEFT JOIN dim_{dim_name} '
                    f'ON fact."{fk_col}" = dim_{dim_name}."{dim_def.key_col}"'
                )
                lineage.append(LineageNode(
                    operation="join",
                    description=(
                        f"Joined fact '{fact_def.table_name}' → dim "
                        f"'{dim_def.table_name}' on {fk_col} = "
                        f"{dim_def.key_col} (left)"
                    ),
                    metadata={
                        "fact":      fact_def.table_name,
                        "dimension": dim_def.table_name,
                        "fact_key":  fk_col,
                        "dim_key":   dim_def.key_col,
                        "engine":    "duckdb",
                    },
                ))

            # The filter tree renders to one parameterised boolean expression —
            # a flat AND for the common case, parenthesised AND/OR when the
            # query used an `or`/`and` group.
            where_sql, params = self._filter_where(filter_tree)

            # A dimensionless aggregate with no engine measures (its only
            # measures are semi-additive period_ends, filled in afterwards)
            # would emit an empty SELECT list. Emit a single-row skeleton the
            # post-passes attach to, then drop the placeholder column.
            skeleton = aggregate and not select_cols_sql
            if skeleton:
                select_cols_sql = ["1 AS __skeleton__"]

            sql = "SELECT " + ", ".join(select_cols_sql) + f" FROM {from_clause}"
            if where_sql:
                sql += " WHERE " + where_sql
            if skeleton:
                sql += " LIMIT 1"
            if aggregate and group_cols_sql:
                sql += " GROUP BY " + ", ".join(group_cols_sql)
                # Deterministic row order. Without it DuckDB returns groups in
                # hash-table order, which varies between identical runs — so
                # DataSet.fingerprint() differed across runs of the same query
                # and manifest fingerprints were not re-verifiable.
                sql += " ORDER BY " + ", ".join(group_cols_sql)

            out = con.execute(sql, params).df()
            if skeleton:
                out = out.drop(columns=["__skeleton__"])
            return out
        finally:
            con.close()

    # ── Inspection ─────────────────────────────────────────────

    def _table_info(self, tdef) -> dict:
        """One table's public description, plus columns when metadata can say.

        A connector that cannot describe the source, or a warehouse that is
        not there yet, leaves the key off. ``info()`` must not fail a model
        listing because one table could not be described, and it must not
        scan the table to learn the names.
        """
        entry = {
            "name": tdef.name,
            "connector": tdef.connector_name,
            "source": tdef.source,
        }
        conn = self._connectors.get(tdef.connector_name)
        if conn is None:
            return entry
        try:
            cols = conn.column_schema(tdef.source)
        except Exception:  # noqa: BLE001 — schema is optional metadata
            return entry
        if cols is not None:
            entry["columns"] = cols
        return entry

    def _measure_fact_columns(self, name: str, fact_def: "_FactDef") -> set[str]:
        """The fact-table columns a declared measure reads, through every
        measure it is built on — what ``execute`` needs on the fact's table
        for the measure to resolve (``_apply_derived``, the column check, and
        the period_end value column)."""
        r = self._resolve_measures([name], fact_def)
        cols: set[str] = set()
        for out in r.agg_map:
            cols |= {t for t in _EXPR_TOKEN.findall(r.derived.get(out, out))
                     if not t.isdigit()}
        cols |= {value_col for value_col, _ in r.period_ends.values()}
        return cols

    def _runnable_measures(
        self, fact_def: "_FactDef", columns: "Optional[set[str]]"
    ) -> list[str]:
        """The declared measures whose columns all exist on this fact's table.

        A measure is declared once on the model, not per fact, so whether it
        runs on a fact is decided by the columns it reads. *columns* is the
        fact table's column set from its connector's metadata; when the
        connector cannot say (or the warehouse is not built yet) nothing can be
        ruled out, so every declared measure is listed and a run reports the
        real error."""
        if columns is None:
            return list(self._measures)
        out = []
        for name in self._measures:
            try:
                if self._measure_fact_columns(name, fact_def) <= columns:
                    out.append(name)
            except ValueError:
                continue    # a definition that cannot resolve runs on no fact
        return out

    def info(self) -> dict:
        """
        The model's structure as a plain dict (tables, relationships,
        facts, dimensions, measures). This is the stable surface the web
        layer reads — prefer it over touching private attributes.

        It is also the machine-readable description of the model's
        vocabulary: what can be measured, sliced, and filtered. Declared
        measures carry their definition, description, and format so a
        consumer can tell what a number *means*, not just what it's called.
        """
        tables = [self._table_info(t) for t in self._tables.values()]
        table_columns = {
            t["name"]: {c["name"] for c in t["columns"]}
            for t in tables if "columns" in t
        }
        return {
            "name": self.name,
            "connectors": list(self._connectors.keys()),
            # Where each connector's data lives, so a model file (code) can be
            # traced to its stored data (a file, folder, database or cloud).
            "connector_details": [
                {**c.describe(), "storage": c.storage()}
                for c in self._connectors.values()
            ],
            "tables": tables,
            "relationships": [
                {
                    "name": r.name,
                    "left_table": r.left_table,
                    "right_table": r.right_table,
                    "left_key": r.left_key,
                    "right_key": r.right_key,
                    "how": r.how,
                }
                for r in self._relationships.values()
            ],
            "facts": [
                {
                    "name": f.name,
                    "table": f.table_name,
                    "measures": list(f.measures),
                    "foreign_keys": dict(f.foreign_keys),
                    # The declared measures (top-level "measures") that can run
                    # on this fact, so a UI offers only what will execute.
                    "runnable_measures": self._runnable_measures(
                        f, table_columns.get(f.table_name)),
                }
                for f in self._facts.values()
            ],
            "dimensions": [
                {
                    "name": d.name,
                    "table": d.table_name,
                    "key": d.key_col,
                    "attributes": list(d.attributes),
                    # Derived groupable attributes (date grains, value bins):
                    # reference as dim.name like any attribute. Present only
                    # when declared; each carries its kind and source column.
                    **({"derived": {
                        n: {"kind": v.get("kind"), "of": v.get("source"),
                            **({"grain": v["grain"]} if v.get("grain") else {}),
                            **({"bands": v["labels"]} if v.get("labels") else {}),
                            **({"edges": v["edges"]} if v.get("edges") else {})}
                        for n, v in d.derived.items()}}
                       if d.derived else {}),
                }
                for d in self._dimensions.values()
            ],
            "measures": [m.to_dict() for m in self._measures.values()],
            "filter_operators": list(FILTER_OPS),
        }

    def describe(self) -> None:
        """Print a summary of the model's connectors, tables, relationships, facts, and dimensions."""
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  DataModel: '{self.name}'")
        print(f"  Connectors ({len(self._connectors)}): {list(self._connectors.keys())}")
        print(f"  Tables ({len(self._tables)}): {list(self._tables.keys())}")
        print(f"  Relationships ({len(self._relationships)}): {list(self._relationships.keys())}")
        if self._facts:
            print(f"  Facts ({len(self._facts)}):")
            for f in self._facts.values():
                print(f"    {f.name} → {f.table_name}  measures={f.measures}")
        if self._dimensions:
            print(f"  Dimensions ({len(self._dimensions)}):")
            for d in self._dimensions.values():
                print(f"    {d.name} → {d.table_name}  key={d.key_col}  attrs={d.attributes}")
        print(f"{sep}\n")

    def help_text(self) -> str:
        """Return a cheat sheet of the DataModel API as a string."""
        return (
            "\nDataModel — governed semantic model over your tables.\n"
            "\n"
            "Building:\n"
            "  .add_connector(connector)                Register a data source\n"
            '  .add_table(name, connector=, source=)    Declare a table\n'
            "  .add_relationship(name, left_table=, right_table=, left_key=, right_key=)\n"
            "  .add_dimension(name, table_name=, key_col=, attributes=[...])\n"
            "  .add_fact(name, table_name=, measures=[...], foreign_keys={dim: fk_col})\n"
            "  .add_measure(name, column=, agg=)        Or: expr=, ratio=, share=,\n"
            "        rank=, running=, period_end=, offset=, growth=, to_date=\n"
            "  .add_time_grain(dim, name, source=, grain=)   Group by month/quarter/…\n"
            "  .add_value_bins(dim, name, source=, edges=[...])  Group by a band\n"
            "\n"
            "Loading & querying (all return a DataSet with lineage):\n"
            "  .load(table_name)                        Load one table\n"
            "  .resolve(relationship_name)              Join two tables\n"
            "  .resolve_chain([rel1, rel2, ...])        Chain multiple joins\n"
            "  .query(fact=, dimensions=[...], measures={...}, filters={...})\n"
            "\n"
            "Inspection:\n"
            "  .info()        Structure as a dict\n"
            "  .describe()    Print a text summary\n"
        )

    def help(self) -> None:
        """Print a cheat sheet of the DataModel API."""
        print(self.help_text())

    def _repr_html_(self) -> str:
        """Rich notebook display: tables, relationships, facts, dimensions."""
        import html as _h

        info = self.info()
        h_css = "font-weight:700;color:#1F3864;font-size:12px;margin:8px 0 4px 0"
        item_css = "font-size:11px;color:#333;padding:1px 0"

        def block(title: str, lines: list[str]) -> str:
            if not lines:
                return ""
            items = "".join(f'<div style="{item_css}">{line}</div>' for line in lines)
            return f'<div style="{h_css}">{title}</div>{items}'

        tables = [
            f'<b>{_h.escape(t["name"])}</b> '
            f'<span style="color:#999">← {_h.escape(t["connector"])} / {_h.escape(t["source"])}</span>'
            for t in info["tables"]
        ]
        rels = [
            f'<b>{_h.escape(r["name"])}</b> '
            f'<span style="color:#999">{_h.escape(r["left_table"])}.{_h.escape(r["left_key"])} '
            f'{_h.escape(r["how"])}-join {_h.escape(r["right_table"])}.{_h.escape(r["right_key"])}</span>'
            for r in info["relationships"]
        ]
        facts = [
            f'<b>{_h.escape(f["name"])}</b> '
            f'<span style="color:#999">→ {_h.escape(f["table"])} measures={f["measures"]}</span>'
            for f in info["facts"]
        ]
        dims = [
            f'<b>{_h.escape(d["name"])}</b> '
            f'<span style="color:#999">→ {_h.escape(d["table"])} key={_h.escape(d["key"])} '
            f'attrs={d["attributes"]}</span>'
            for d in info["dimensions"]
        ]

        return f"""
<div style="font-family:'Segoe UI',Calibri,Arial,sans-serif;border:1px solid #dde4ef;border-radius:6px;padding:12px 16px;display:inline-block;max-width:100%">
  <div>
    <span style="font-weight:700;color:#1F3864;font-size:14px">DataModel: {_h.escape(self.name)}</span>
    <span style="color:#666;font-size:11px;margin-left:8px">connectors: {", ".join(_h.escape(c) for c in info["connectors"]) or "—"}</span>
  </div>
  {block("Tables", tables)}
  {block("Relationships", rels)}
  {block("Facts", facts)}
  {block("Dimensions", dims)}
</div>"""

    def __repr__(self) -> str:
        return (
            f"<DataModel name={self.name!r} "
            f"tables={list(self._tables.keys())} "
            f"relationships={list(self._relationships.keys())} "
            f"facts={list(self._facts.keys())} "
            f"dimensions={list(self._dimensions.keys())}>"
        )

"""Offline selection for the measure kinds that match ``DataModel``.

Python is the source of truth. :func:`aggregate_frame` recomputes ``simple``
aggregations and ``ratio`` from a sealed fact-grain frame. ``share``, ``rank``,
``running``, ``period_end``, and the time-intelligence kinds stay server-only:
:func:`plan_binding` marks them ``offline: false`` and names the kind, and the
worker port refuses them the same way. A one-row change in the grain changes
the fingerprint — the corpus test is that comparison.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import pandas as pd

from tracebi.model.data_model import QuerySpec
from tracebi.reports.selection import _json_cell, conjoin_filters, query_under_selection

#: Aggregations the port implements. Anything else (``p90``, …) stays on the server.
_OFFLINE_AGGS = {"sum", "count", "mean", "avg", "min", "max", "nunique"}

_SERVER_ONLY = (
    ("period_ends", "period_end"),
    ("shares", "share"),
    ("ranks", "rank"),
    ("runnings", "running"),
    ("pops", "offset"),
    ("to_dates", "to_date"),
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def plan_binding(model, query: QuerySpec) -> dict:
    """How to recompute *query* from a grain, or why it stays on the server."""
    fact = query.fact
    try:
        fact_def = model._facts[fact]
        resolved = model._resolve_measures(query.measures, fact_def)
    except Exception as exc:  # noqa: BLE001 — the plan records the refusal
        return {"offline": False, "reason": "unresolved", "fact": fact,
                "detail": str(exc)}
    for attr, reason in _SERVER_ONLY:
        if getattr(resolved, attr):
            return {"offline": False, "reason": reason, "fact": fact}
    if query.having:
        return {"offline": False, "reason": "having", "fact": fact}
    aggs = []
    for name, func in resolved.agg_map.items():
        func_l = str(func).lower()
        if func_l not in _OFFLINE_AGGS:
            return {"offline": False, "reason": func_l, "fact": fact}
        source = resolved.derived.get(name, name)
        if not (isinstance(source, str) and _IDENT.fullmatch(source)):
            return {"offline": False, "reason": "expression", "fact": fact}
        aggs.append({
            "name": name,
            "source": source,
            "func": "mean" if func_l == "avg" else func_l,
        })
    ratios = [
        {"name": name, "num": num, "den": den}
        for name, (num, den) in resolved.ratios.items()
    ]
    order = [dict(item) for item in (query.order_by or ())]
    return {
        "offline": True,
        "fact": fact,
        "dimensions": list(query.dimensions),
        "aggs": aggs,
        "ratios": ratios,
        "order_by": order,
        "limit": query.limit,
        "binding_filters": dict(query.filters or {}),
    }


def build_grain_payload(model, bindings: dict, selection: dict,
                         controls: list[tuple[str, str]]) -> Optional[dict]:
    """The sealed grain plus one plan per binding, or None when nothing ports.

    The grain is the fact rows joined to the dimension attributes the controls
    cut, with no selection applied. The worker filters and aggregates. Bindings
    that are not ``simple``/``ratio`` are listed ``offline: false`` so the page
    leaves them on the authored view.
    """
    selection_model = selection["model"]
    plans: dict[str, dict] = {}
    by_fact: dict[str, dict] = {}
    for name, ref in bindings.items():
        if ref.model != selection_model and ref.model != getattr(model, "name", None):
            continue
        plan = plan_binding(model, ref.query)
        plans[name] = plan
        if not plan["offline"]:
            continue
        bucket = by_fact.setdefault(plan["fact"], {"dims": [], "sources": []})
        for dim in plan["dimensions"]:
            if dim not in bucket["dims"]:
                bucket["dims"].append(dim)
        for agg in plan["aggs"]:
            if agg["source"] not in bucket["sources"]:
                bucket["sources"].append(agg["source"])
    if not by_fact:
        return None

    extra_dims = []
    for key in (selection.get("filters") or {}):
        if "." in str(key):
            extra_dims.append(str(key))
    for _binding, column in controls:
        if "." in column:
            extra_dims.append(column)
    for ref in bindings.values():
        if ref.model != selection_model and ref.model != getattr(model, "name", None):
            continue
        for key in (ref.query.filters or {}):
            if "." in str(key):
                extra_dims.append(str(key))

    facts = {}
    for fact, bucket in by_fact.items():
        dims = list(bucket["dims"])
        for dim in extra_dims:
            if dim not in dims:
                dims.append(dim)
        sources = bucket["sources"] or ["_row"]
        # aggregate=False selects the measure keys as raw fact columns.
        measures = {src: "sum" for src in sources if src != "_row"}
        if not measures:
            continue
        spec = QuerySpec(
            fact=fact, measures=measures, dimensions=tuple(dims), aggregate=False,
        )
        frame = model.execute(spec).to_pandas()
        facts[fact] = {"rows": [_row(frame, i) for i in range(len(frame))]}
    if not facts:
        return None
    return {"facts": facts, "bindings": plans}


def _row(frame: pd.DataFrame, index: int) -> dict:
    return {str(col): _json_cell(frame.iloc[index][col]) for col in frame.columns}


def aggregate_frame(grain: pd.DataFrame, plan: dict, selection_filters=None) -> pd.DataFrame:
    """Recompute one offline plan. Refuses a plan that is not offline."""
    if not plan.get("offline"):
        raise OfflineRefusal(plan.get("reason") or "server")
    filters = conjoin_filters(plan.get("binding_filters"), selection_filters)
    filtered = _apply_filters(grain, filters)
    dims = list(plan.get("dimensions") or [])
    aggs = list(plan.get("aggs") or [])
    ratios = list(plan.get("ratios") or [])
    if dims:
        if filtered.empty:
            return _empty_result(grain, dims, aggs, ratios)
        grouped = filtered.groupby(dims, dropna=False, sort=True)
        data: dict[str, Any] = {}
        for agg in aggs:
            data[agg["name"]] = _reduce_grouped(grouped[agg["source"]], agg["func"])
        out = pd.DataFrame(data)
        out = out.reset_index()
    else:
        row = {
            agg["name"]: _reduce_series(filtered[agg["source"]], agg["func"])
            for agg in aggs
        }
        out = pd.DataFrame([row])
    for agg in aggs:
        out[agg["name"]] = _cast_agg(out[agg["name"]], agg["func"], grain[agg["source"]])
    for dim in dims:
        if dim in grain.columns and dim in out.columns and len(out):
            out[dim] = out[dim].astype(grain[dim].dtype)
    for ratio in ratios:
        den = out[ratio["den"]].replace(0, pd.NA)
        out[ratio["name"]] = (out[ratio["num"]] / den).astype("float64")
    columns = dims + [a["name"] for a in aggs] + [r["name"] for r in ratios]
    out = out[columns]
    out = _sort_like_model(out, plan.get("order_by") or [], dims)
    if plan.get("limit") is not None:
        out = out.head(int(plan["limit"])).reset_index(drop=True)
    return out.reset_index(drop=True)


class OfflineRefusal(ValueError):
    """A measure kind the worker must not recompute."""


def _empty_result(grain, dims, aggs, ratios) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for dim in dims:
        # An empty grouped result has no values to infer a string dtype from;
        # DuckDB returns object for that column. Sums stay float64.
        dtype = "object" if not pd.api.types.is_numeric_dtype(grain[dim]) else grain[dim].dtype
        columns[dim] = pd.Series(dtype=dtype)
    for agg in aggs:
        dtype = "float64" if agg["func"] in ("sum", "mean", "min", "max") else "int64"
        columns[agg["name"]] = pd.Series(dtype=dtype)
    for ratio in ratios:
        columns[ratio["name"]] = pd.Series(dtype="float64")
    order = dims + [a["name"] for a in aggs] + [r["name"] for r in ratios]
    return pd.DataFrame(columns).reindex(columns=order)


def _reduce_grouped(series_group, func: str):
    if func == "sum":
        return series_group.sum(min_count=1)
    if func == "count":
        return series_group.count()
    if func == "mean":
        return series_group.mean()
    if func == "min":
        return series_group.min()
    if func == "max":
        return series_group.max()
    if func == "nunique":
        return series_group.nunique(dropna=True)
    raise OfflineRefusal(func)


def _reduce_series(series: pd.Series, func: str):
    if func == "sum":
        return float(series.sum(min_count=1)) if series.notna().any() else float("nan")
    if func == "count":
        return int(series.count())
    if func == "mean":
        return float(series.mean()) if series.notna().any() else float("nan")
    if func == "min":
        return series.min() if series.notna().any() else float("nan")
    if func == "max":
        return series.max() if series.notna().any() else float("nan")
    if func == "nunique":
        return int(series.nunique(dropna=True))
    raise OfflineRefusal(func)


def _cast_agg(series: pd.Series, func: str, source: pd.Series) -> pd.Series:
    if func in ("sum", "mean"):
        return series.astype("float64")
    if func in ("count", "nunique"):
        return series.astype("int64")
    # min/max keep an integer source's dtype when every group has a value.
    if pd.api.types.is_integer_dtype(source) and series.notna().all():
        return series.astype("int64")
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for key, raw in (filters or {}).items():
        if key not in out.columns:
            raise OfflineRefusal(f"filter target {key}")
        out = out[_mask(out[key], raw)]
    return out


def _mask(series: pd.Series, raw) -> pd.Series:
    if isinstance(raw, (list, tuple)):
        return series.map(lambda cell: any(_same(cell, item) for item in raw))
    if isinstance(raw, dict):
        if len(raw) != 1:
            raise OfflineRefusal("filter operator")
        op, value = next(iter(raw.items()))
        return _op_mask(series, str(op), value)
    return series.map(lambda cell: _same(cell, raw))


def _op_mask(series: pd.Series, op: str, value) -> pd.Series:
    if op in ("eq",):
        return series.map(lambda cell: _same(cell, value))
    if op == "ne":
        return series.map(lambda cell: not _null(cell) and not _same(cell, value))
    if op == "in":
        return series.map(lambda cell: any(_same(cell, item) for item in (value or [])))
    if op == "not_in":
        items = list(value or [])
        return series.map(lambda cell: not _null(cell) and not any(_same(cell, item) for item in items))
    if op == "is_null":
        return series.map(_null)
    if op == "not_null":
        return series.map(lambda cell: not _null(cell))
    if op == "contains":
        needle = "" if value is None else str(value)
        return series.map(lambda cell: not _null(cell) and needle in str(cell))
    if op == "between":
        lo, hi = value[0], value[1]
        return series.map(lambda cell: _cmp(cell, lo) >= 0 and _cmp(cell, hi) <= 0)
    if op in ("gt", "gte", "lt", "lte"):
        return series.map(lambda cell: _ordered(cell, value, op))
    raise OfflineRefusal(op)


def _null(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _same(left, right) -> bool:
    if _null(left) or _null(right):
        return False
    if left == right:
        return True
    try:
        return float(left) == float(right) and not isinstance(left, bool)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _cmp(left, right) -> int:
    """-1/0/1, numeric when both sides are numbers, else string order."""
    if _null(left):
        return 1  # nulls sort last; comparisons with null are not a match
    try:
        lf, rf = float(left), float(right)
        if math.isfinite(lf) and math.isfinite(rf):
            return (lf > rf) - (lf < rf)
    except (TypeError, ValueError):
        pass
    ls, rs = str(left), str(right)
    return (ls > rs) - (ls < rs)


def _ordered(cell, value, op: str) -> bool:
    if _null(cell) or _null(value):
        return False
    cmp = _cmp(cell, value)
    if op == "gt":
        return cmp > 0
    if op == "gte":
        return cmp >= 0
    if op == "lt":
        return cmp < 0
    return cmp <= 0


def _sort_like_model(df: pd.DataFrame, order_by: list, dimensions: list) -> pd.DataFrame:
    if not len(df):
        return df
    resolved = [dict(item) for item in order_by]
    named = {item["column"] for item in resolved}
    for col in list(dimensions) + list(df.columns):
        if col in df.columns and col not in named:
            resolved.append({"column": col, "desc": False})
            named.add(col)
    if not resolved:
        return df
    by = [item["column"] for item in resolved]
    asc = [not item["desc"] for item in resolved]
    return df.sort_values(
        by=by, ascending=asc, kind="mergesort", na_position="last",
    ).reset_index(drop=True)


def frame_under_selection(model, query: QuerySpec, selection_filters=None) -> pd.DataFrame:
    """The engine's own result for the conjoined query. The corpus compares to this."""
    spec = query_under_selection(query, selection_filters)
    return model.execute(spec).to_pandas()

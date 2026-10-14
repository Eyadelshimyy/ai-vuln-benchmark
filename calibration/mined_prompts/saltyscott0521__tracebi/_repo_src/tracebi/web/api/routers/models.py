import io
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tracebi.web.api.errors import error_detail
from tracebi.web.api.lineage_graph import lineage_to_graph
from tracebi.web.api.registry import registry
from tracebi.web.api.source_view import source_payload

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models():
    """List all registered data models."""
    return registry.list_models()


@router.get("/{name}")
def get_model(name: str):
    """Get full detail for a data model (tables, relationships)."""
    detail = registry.describe_model(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    detail = dict(detail)
    from tracebi.model_registry import model_path

    path = model_path(name) or model_path(detail.get("name") or name)
    if path:
        try:
            detail["source_file"] = os.path.relpath(path, os.getcwd())
        except ValueError:
            detail["source_file"] = path
    return detail


@router.get("/{name}/source")
def model_source(name: str):
    """The model's file, read-only: the contract the Data model page describes."""
    detail = registry.describe_model(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    from tracebi.model_registry import model_path

    path = model_path(name) or model_path(detail.get("name") or name)
    return source_payload(
        [("model", path)],
        missing_hint="This model is registered in Python code (an app module), not from a models/ file.",
        hint="The model: its tables, joins, facts, dimensions and measures.",
    )


def _safe_filename(name: str) -> str:
    """Keep header-safe characters only (Content-Disposition filename)."""
    return "".join(c for c in name if c.isalnum() or c in "._- ") or "export"


def _load_table(name: str, table_name: str):
    model = registry.get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    try:
        return model.load(table_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _records(df) -> list[dict]:
    """Rows for a JSON response. JSON has no NaN or infinity, and data can hold
    one (an empty cell, an empty total, a division by zero): those go out as
    null rather than failing the whole response."""
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@router.get("/{name}/tables/{table_name}/preview")
def preview_table(name: str, table_name: str, rows: int = 50):
    """Load a table from a model and return the first N rows."""
    ds = _load_table(name, table_name)
    full_df = ds.to_pandas()
    df = full_df.head(rows)
    return {
        "model": name,
        "table": table_name,
        "rows": len(df),
        "total_rows": len(full_df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in full_df.dtypes.items()},
        "data": _records(df),
        "lineage": ds.lineage_to_dict(),
    }


class QueryRequest(BaseModel):
    # One grammar on every surface (architecture v2 §2.2): the body is the
    # same QuerySpec dict Python, report specs, and the MCP gateway speak —
    # validated by QuerySpec.from_dict, not by a parallel pydantic shape.
    # measures accepts BOTH forms: {column: agg} ad-hoc aggregations AND a
    # list of declared measure names (ratio measures included) — the
    # semantic layer's vocabulary is reachable over HTTP.
    fact: str
    measures: Any                             # {column: agg} or [measure names]
    dimensions: Optional[list[str]] = None    # ["dim_customer.region", ...]
    filters: Optional[dict[str, Any]] = None  # WHERE — before aggregation
    having: Optional[dict[str, Any]] = None   # HAVING — on aggregated measures
    aggregate: bool = True
    allow_fanout: bool = False
    order_by: Optional[list] = None           # [{"column":..,"desc":..}] or "-col"
    limit: Optional[int] = None               # requires order_by


@router.post("/{name}/query")
def run_query(name: str, body: QueryRequest):
    """
    Run a star-schema query against a model's facts/dimensions and return
    the result rows plus the full lineage of the query that was executed.
    """
    from tracebi.model.data_model import QuerySpec

    model = registry.get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    started = time.perf_counter()
    try:
        spec = QuerySpec.from_dict({
            k: v for k, v in {
                "fact": body.fact,
                "measures": body.measures,
                "dimensions": body.dimensions,
                "filters": body.filters,
                "having": body.having,
                "aggregate": body.aggregate,
                "allow_fanout": body.allow_fanout,
                "order_by": body.order_by,
                "limit": body.limit,
            }.items() if v is not None
        })
        ds = model.execute(spec)
    except ValueError as exc:
        # Unknown fact/dimension/agg or malformed reference — caller error.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=error_detail("Query failed", exc))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    # JSON has no NaN or infinity, and a result can hold one (an empty total, a
    # division by zero): they go out as missing, null in data and blank in
    # display, rather than failing the whole response.
    df = ds.to_pandas().replace([float("inf"), float("-inf")], float("nan"))
    lineage = ds.lineage_to_dict()
    engine = next(
        (n["metadata"].get("engine") for n in reversed(lineage)
         if n.get("metadata", {}).get("engine")),
        None,
    )
    # The cells as a built report's table writes them (the model's declared
    # formats), beside the raw values: "data" is what a download uses.
    from tracebi.reports.template_package import declared_column_formats, display_rows

    return {
        "model": name,
        "fact": body.fact,
        "rows": len(df),
        "columns": list(df.columns),
        "data": _records(df),
        "display": display_rows(df, declared_column_formats(model, df.columns)),
        "engine": engine,
        "elapsed_ms": elapsed_ms,
        "lineage": lineage,
        "lineage_graph": lineage_to_graph(lineage),
    }


@router.get("/{name}/tables/{table_name}/export.csv")
def export_table_csv(name: str, table_name: str):
    """Download the full table as CSV."""
    ds = _load_table(name, table_name)
    buf = io.StringIO()
    ds.to_pandas().to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="{_safe_filename(table_name)}.csv"',
        },
    )

"""FastAPI app entry point.

Phase 1 wires the dataset / dataset-group / exposures / ert-jobs / exports
routers under ``/api`` and converts all unhandled errors to the canonical
:class:`~app.models.warnings.ErrorEnvelope` shape so the wire is consistent
(ERROR_HANDLING.md).

Domain outcomes (missing IED denominator, failed ERT job, county fallback) are
NOT errors — they ride in 200 responses as warnings (CLAUDE.md rule 11).
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .brand import BRAND_NAME, SERVICE_ID
from .api import (
    admin,
    calc,
    cedents,
    counties,
    dataset_groups,
    ert_jobs,
    exports,
    exposures,
    flood,
    fund_analysis,
    hazards,
    hurricanes,
    industry_loss,
    live,
    wildfire,
)
from .config import get_settings
from .models.enums import ErrorCode
from .models.warnings import ErrorEnvelope, ErrorEnvelopeBody

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Release provider-owned connection pools when the process exits."""
    yield
    from .providers import get_provider

    if not get_provider.cache_info().currsize:
        return
    provider = get_provider()
    registry = getattr(provider, "connection_registry", None)
    if registry is not None:
        registry.close_all()


app = FastAPI(
    title=f"{BRAND_NAME} API",
    version="0.1.0",
    description="Property Cat exposure management workbench — backend API.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api = APIRouter(prefix="/api")


@api.get("/health", tags=["meta"])
def health() -> dict[str, object]:
    """Liveness probe — see docs/API_SPEC.md table at line 318."""
    from .providers import get_provider

    body: dict[str, object] = {
        "status": "ok",
        "service": SERVICE_ID,
        "version": app.version,
        "dataProvider": settings.data_provider,
        "factCacheMaxDatasets": settings.fact_cache_max_datasets,
        "factLoadMaxWorkers": settings.fact_load_max_workers,
    }
    try:
        provider = get_provider()
        cache = getattr(provider, "fact_cache", None)
        if cache is not None:
            stats = cache.stats()
            body["factCache"] = {
                "datasetsCached": stats.datasets_cached,
                "rowsCached": stats.rows_cached,
                "hits": stats.hits,
                "misses": stats.misses,
            }
        registry = getattr(provider, "connection_registry", None)
        if registry is not None:
            body["sqlServersRegistered"] = len(registry)
    except Exception:
        pass
    return body


app.include_router(api)
app.include_router(admin.router, prefix="/api")
app.include_router(calc.router, prefix="/api")
app.include_router(cedents.router, prefix="/api")
app.include_router(counties.router, prefix="/api")
app.include_router(dataset_groups.router, prefix="/api")
app.include_router(exposures.router, prefix="/api")
app.include_router(ert_jobs.router, prefix="/api")
app.include_router(exports.router, prefix="/api")
app.include_router(hurricanes.router, prefix="/api")
app.include_router(industry_loss.router, prefix="/api")
app.include_router(hazards.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(wildfire.router, prefix="/api")
app.include_router(flood.router, prefix="/api")
app.include_router(fund_analysis.router, prefix="/api")


# ───────────────────────── Exception handlers ─────────────────────────


def _envelope(
    *,
    code: ErrorCode | str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Wrap any error into the canonical :class:`ErrorEnvelope` JSON body."""
    code_str = code.value if isinstance(code, ErrorCode) else str(code)
    try:
        envelope_code = ErrorCode(code_str)
    except ValueError:
        envelope_code = ErrorCode.INTERNAL_ERROR
    body = ErrorEnvelope(
        error=ErrorEnvelopeBody(
            code=envelope_code,
            message=message,
            details=details,
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(by_alias=True, mode="json"))


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert Pydantic/FastAPI validation errors → ``VALIDATION_ERROR`` 422 envelope."""
    errors = jsonable_encoder(exc.errors())
    hint = ""
    if errors:
        first = errors[0]
        loc = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        msg = first.get("msg") or ""
        hint = f" {loc}: {msg}".rstrip() if loc else f" {msg}"
    return _envelope(
        code=ErrorCode.VALIDATION_ERROR,
        message=f"Request validation failed.{hint}",
        status_code=422,
        details={"errors": errors},
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Reshape ``HTTPException`` payloads into the canonical envelope."""
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", ErrorCode.INTERNAL_ERROR.value)
        message = detail.get("message", str(exc.detail))
        details = detail.get("details")
    else:
        code = ErrorCode.INTERNAL_ERROR.value
        message = str(detail) if detail else "Unexpected error."
        details = None
    return _envelope(
        code=code,
        message=message,
        status_code=exc.status_code,
        details=details,
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:  # pragma: no cover — last-resort guard
    logger.exception("Unhandled exception on %s", request.url.path)
    return _envelope(
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred.",
        status_code=500,
        details={"type": exc.__class__.__name__},
    )

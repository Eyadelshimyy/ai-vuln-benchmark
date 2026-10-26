"""OtterWorks Document Service - FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import telemetry
from app.api import comments, documents, health, templates
from app.config import settings
from app.db.migrate import upgrade_to_head
from app.db.session import async_session, engine
from app.jobs import stats_rollup
from app.middleware import request_log

logger = structlog.get_logger()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("document_service_starting")
    await upgrade_to_head(engine)
    rollup_task = stats_rollup.start(async_session)

    logger.info("document_service_started")
    yield
    logger.info("document_service_shutting_down")
    if rollup_task is not None:
        rollup_task.cancel()
    await engine.dispose()


app = FastAPI(
    title="OtterWorks Document Service",
    description="Document CRUD, versioning, and content management",
    version=settings.app_version,
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

telemetry.instrument_sql()
telemetry.instrument_app(app)
telemetry.setup_tracing(app, engine)
request_log.install(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(comments.router, prefix="/api/v1/documents", tags=["comments"])
app.include_router(templates.router, prefix="/api/v1/templates", tags=["templates"])

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(log_level=settings.log_level, use_json=settings.environment == "production")
    logger.info("BugForge API starting up (environment=%s)", settings.environment)
    from app.plugins import get_plugin_catalog

    catalog = get_plugin_catalog()
    logger.info(
        "Plugin catalog ready: languages=%s ai_providers=%s",
        ",".join(catalog.languages.available_ids()),
        ",".join(catalog.ai_providers.available_ids()),
    )
    yield
    logger.info("BugForge API shutting down")


app = FastAPI(
    title="BugForge API",
    description=(
        "Evidence-first debugging and authorized security research. "
        "The AI is advisory. ScopeGuard, SafetyController, and human approval "
        "remain authoritative. Version 1.8 includes Phase 8 research intelligence."
    ),
    version="1.8.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.v1.router import api_router  # noqa: E402 (after app creation to avoid circular)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": settings.version, "environment": settings.environment}

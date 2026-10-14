from __future__ import annotations

import os

# Bounty campaigns persist through a real database (Phase 51). Tests use a shared
# in-memory SQLite so the Phase 49 SqlStore and the campaign record store run end
# to end. This must be set before any module reads the cached settings.
os.environ.setdefault(
    "BOUNTY_CAMPAIGN_DATABASE_URL",
    "sqlite:///file:bugforge_campaign_test?mode=memory&cache=shared&uri=true",
)
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.models import Analysis, Project  # noqa: F401 — ensure models are registered
from app.models.base import Base
from app.models.debugging import AIModelCall, DebuggingHypothesis, DebuggingSession  # noqa: F401
from app.models.finding import DBFinding  # noqa: F401
from app.models.github import GitHubDelivery, GitHubRepository  # noqa: F401
from app.models.hackerone import (  # noqa: F401
    DBHackerOneApprovalEvent,
    DBHackerOneAttachment,
    DBHackerOneAuditEvent,
    DBHackerOneProgram,
    DBHackerOneReportDraft,
    DBHackerOneReportIntent,
    DBHackerOneScopeExclusion,
    DBHackerOneStructuredScope,
    DBHackerOneSubmission,
    DBHackerOneSync,
    DBHackerOneWeakness,
)
from app.models.repair import PatchCandidate, RepairSession  # noqa: F401
from app.models.reproduction import BugReproductionAttempt, BugReproductionSession  # noqa: F401
from app.models.security_agent import (  # noqa: F401
    DBReproductionPlan,
    DBResearchCheckpoint,
    DBResearchEvidenceEdge,
    DBResearchEvidenceLink,
    DBResearchEvidenceNode,
    DBResearchExploratoryAttempt,
    DBResearchFinding,
    DBResearchHypothesis,
    DBResearchIdentity,
    DBResearchLead,
    DBResearchMemory,
    DBResearchProject,
    DBResearchSession,
    DBResearchTimelineEvent,
    DBResearchToolCall,
)
from app.models.security_audit import DBSecurityAuditEvent  # noqa: F401
from app.models.security_finding import DBSecurityFinding  # noqa: F401
from app.models.test_generation import GeneratedTest, TestGenerationSession  # noqa: F401
from app.models.test_run import TestResult, TestRun  # noqa: F401
from app.models.verification import PatchVerification  # noqa: F401

os.environ.setdefault("BUGFORGE_OPERATOR_TOKEN", "test-operator-token")
os.environ.setdefault("BUGFORGE_OPERATOR_IDENTITY", "alice")

OPERATOR_HEADERS = {"X-BugForge-Operator-Token": "test-operator-token"}

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def engine():
    eng = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    # Override both the FastAPI dependency AND the module-level factory used by
    # background tasks, so analysis background work also targets the test DB.
    import app.database as db_module

    original_factory = db_module.async_session_factory
    db_module.async_session_factory = session_factory

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    db_module.async_session_factory = original_factory

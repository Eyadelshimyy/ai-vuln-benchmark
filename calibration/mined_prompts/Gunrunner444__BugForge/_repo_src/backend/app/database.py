from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def create_tables() -> None:
    """Create all tables defined in the metadata. Used only for testing / dev convenience."""
    import app.models.analysis  # noqa: F401
    import app.models.debugging  # noqa: F401
    import app.models.discovery  # noqa: F401
    import app.models.finding  # noqa: F401
    import app.models.github  # noqa: F401
    import app.models.hackerone  # noqa: F401
    import app.models.project  # noqa: F401
    import app.models.repair  # noqa: F401
    import app.models.reproduction  # noqa: F401
    import app.models.security_agent  # noqa: F401
    import app.models.security_audit  # noqa: F401
    import app.models.security_finding  # noqa: F401
    import app.models.test_generation  # noqa: F401
    import app.models.test_run  # noqa: F401
    import app.models.verification  # noqa: F401
    from app.models.base import Base  # noqa: PLC0415  (avoid circular at module level)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

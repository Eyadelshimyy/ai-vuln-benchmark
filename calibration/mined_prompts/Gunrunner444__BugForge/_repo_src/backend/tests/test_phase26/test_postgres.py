"""PostgreSQL lifecycle writers cannot reuse a stale semantic target."""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.domain.findings import FindingStatus
from app.domain.lifecycle_policy import positive_reproduction
from app.domain.trusted_evidence import is_trusted_observation
from app.models.base import Base
from app.models.project import Project
from app.models.security_finding import DBSecurityFinding
from app.repositories.security_finding_repo import SecurityFindingRepository, to_domain
from app.services.finding_lifecycle import FindingLifecycleService
from tests.test_phase26.test_target_binding import _finding, _http, _repro


def _issued(finding, item, execution):
    from app.domain.trusted_evidence import issue_for_finding

    return issue_for_finding(finding, item, execution)


@pytest.mark.asyncio
async def test_postgres_rejects_stale_target_and_keeps_stronger_state() -> None:
    url = os.environ.get(
        "BUGFORGE_POSTGRES_TEST_URL",
        "postgresql+asyncpg://bugforge:bugforge_dev@localhost:5432/bugforge_phase23",
    )
    if not url.startswith("postgresql"):
        pytest.skip("PostgreSQL URL is required")
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL is not available: {exc}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    finding_id = None
    project_id = None
    try:
        async with factory() as session:
            project = Project(name="pg26", repository_path="/tmp/bugforge-phase26")
            session.add(project)
            await session.flush()
            project_id = project.id
            service = FindingLifecycleService(SecurityFindingRepository(session))
            created = await service.persist_static_scan(
                [_finding(finding_key="pg26", project_id=str(project.id))],
                project_id=project.id,
                analysis_id=None,
            )
            finding_id = created[0].id
            await session.commit()

        async with factory() as session:
            current = to_domain(await SecurityFindingRepository(session).get(finding_id))
        verified = current.verify([_issued(current, _http(details="live"), "exec-live")])
        stale_target = current.reproduce([_repro()])
        # A writer that still holds the pre-verification snapshot must not erase it.
        started = asyncio.Event()
        release = asyncio.Event()

        async def save_verified() -> None:
            async with factory() as session:
                repo = SecurityFindingRepository(session)
                row = await repo.get_for_update(finding_id)
                assert row is not None
                started.set()
                await release.wait()
                await repo.save_lifecycle(verified, project_id=project_id)
                await session.commit()

        async def save_reproduction() -> None:
            await started.wait()
            async with factory() as session:
                repo = SecurityFindingRepository(session)
                await repo.save_lifecycle(stale_target, project_id=project_id)
                await session.commit()

        first = asyncio.create_task(save_verified())
        second = asyncio.create_task(save_reproduction())
        await started.wait()
        done, _pending = await asyncio.wait({second}, timeout=0.4)
        assert second not in done
        release.set()
        await first
        await second

        async with factory() as session:
            final = to_domain(await SecurityFindingRepository(session).get(finding_id))
        assert final.status is FindingStatus.VERIFIED
        assert any(is_trusted_observation(item) for item in final.evidence.items)
        assert any(positive_reproduction(item) for item in final.evidence.items)
        assert final.flow_sink == "eval"

        rejected = final.reject()
        async with factory() as session:
            repo = SecurityFindingRepository(session)
            await repo.save_lifecycle(rejected, project_id=project_id)
            await session.commit()
        async with factory() as session:
            repo = SecurityFindingRepository(session)
            await repo.save_lifecycle(final, project_id=project_id)
            await session.commit()
            stored = to_domain(await repo.get(finding_id))
        assert stored.status is FindingStatus.REJECTED
    finally:
        if finding_id is not None and project_id is not None:
            async with factory() as session:
                row = await session.get(DBSecurityFinding, finding_id)
                if row is not None:
                    await session.delete(row)
                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
                await session.commit()
        await engine.dispose()

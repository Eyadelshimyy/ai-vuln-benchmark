"""Discovery run API endpoints (v1.1.0).

POST /api/v1/discovery/run          — trigger one discovery sweep
GET  /api/v1/discovery/runs         — list recent discovery runs
GET  /api/v1/discovery/runs/{id}    — get one run
GET  /api/v1/discovery/settings     — read current discovery config
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database import get_db
from app.repositories.discovery_repo import (
    DiscoveryRunRepository,
    RepositoryCandidateRepository,
)
from app.schemas.discovery import (
    DiscoveryRunResponse,
    DiscoveryRunsListResponse,
    DiscoverySettingsResponse,
    TriggerDiscoveryRequest,
)

router = APIRouter(prefix="/discovery", tags=["Discovery"])
logger = logging.getLogger(__name__)


async def _run_discovery_background(
    run_id: UUID,
    max_pages: int,
) -> None:
    """Background task: run one discovery sweep."""

    from app.database import async_session_factory
    from app.repositories.discovery_repo import (
        DiscoveryRunRepository,
    )
    from app.services.discovery_service import (
        GitHubDiscoveryService,
    )
    from app.services.eligibility_service import EligibilityService

    logger.info("Discovery run %s starting (max_pages=%d)", run_id, max_pages)
    disc_svc = GitHubDiscoveryService(settings)
    elig_svc = EligibilityService(settings)

    discovered = 0
    eligible = 0
    rejected = 0
    error_msg: str | None = None

    try:
        result = await disc_svc.discover(max_pages=max_pages)

        if result.error:
            error_msg = result.error
            logger.warning("Discovery run %s error: %s", run_id, result.error)

        async with async_session_factory() as session:
            cand_repo = RepositoryCandidateRepository(session)

            for repo in result.repos:
                discovered += 1
                candidate, _ = await cand_repo.upsert_from_github(
                    github_repo_id=repo.github_repo_id,
                    owner=repo.owner,
                    name=repo.name,
                    full_name=repo.full_name,
                    html_url=repo.html_url,
                    clone_url=repo.clone_url,
                    stars=repo.stars,
                    is_fork=repo.is_fork,
                    is_archived=repo.is_archived,
                    default_branch=repo.default_branch,
                    primary_language=repo.primary_language,
                    license_key=repo.license_key,
                    size_kb=repo.size_kb,
                    open_issues=repo.open_issues,
                    topics=repo.topics,
                    description=repo.description,
                    last_updated_at=repo.last_updated_at,
                    last_pushed_at=repo.last_pushed_at,
                )

                # Run eligibility screening
                assessment = elig_svc.assess(
                    full_name=candidate.full_name,
                    owner=candidate.owner,
                    stars=candidate.stars,
                    is_fork=candidate.is_fork,
                    is_archived=candidate.is_archived,
                    primary_language=candidate.primary_language,
                    license_key=candidate.license_key,
                    size_kb=candidate.size_kb,
                    topics=repo.topics,
                    last_pushed_at=candidate.last_pushed_at,
                    description=candidate.description,
                )
                await cand_repo.update_eligibility(
                    candidate.id,
                    eligibility_status=assessment.eligibility_status,
                    eligibility_score=assessment.eligibility_score,
                    rejection_reason=assessment.rejection_reason,
                    safety_classification=assessment.safety_classification,
                    safety_detail=assessment.to_detail_json(),
                )
                if assessment.is_eligible:
                    eligible += 1
                else:
                    rejected += 1

            await session.commit()

    except Exception as exc:
        error_msg = str(exc)
        logger.error("Discovery run %s unhandled error: %s", run_id, exc, exc_info=True)

    finally:
        async with async_session_factory() as session:
            run_repo = DiscoveryRunRepository(session)
            await run_repo.complete(
                run_id,
                discovered=discovered,
                eligible=eligible,
                rejected=rejected,
                api_requests=getattr(result if "result" in dir() else object(), "api_requests", 0),
                error_message=error_msg,
            )
            await session.commit()
        logger.info(
            "Discovery run %s done: discovered=%d eligible=%d rejected=%d error=%s",
            run_id,
            discovered,
            eligible,
            rejected,
            error_msg,
        )


@router.post(
    "/run",
    response_model=DiscoveryRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_discovery(
    request: TriggerDiscoveryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> DiscoveryRunResponse:
    """Trigger an immediate repository discovery sweep.

    The sweep runs as a background task; the response returns immediately
    with the newly created run record.
    """
    from app.services.discovery_service import build_criteria_snapshot

    criteria = build_criteria_snapshot(settings)
    run_repo = DiscoveryRunRepository(db)
    run = await run_repo.create(search_criteria=criteria)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_run_discovery_background, run.id, request.max_pages)
    logger.info("Discovery run %s queued.", run.id)
    return DiscoveryRunResponse.model_validate(run)


@router.get("/runs", response_model=DiscoveryRunsListResponse)
async def list_discovery_runs(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> DiscoveryRunsListResponse:
    run_repo = DiscoveryRunRepository(db)
    runs, total = await run_repo.list_recent(limit=limit, offset=offset)
    return DiscoveryRunsListResponse(
        items=[DiscoveryRunResponse.model_validate(r) for r in runs],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/runs/{run_id}", response_model=DiscoveryRunResponse)
async def get_discovery_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DiscoveryRunResponse:
    run_repo = DiscoveryRunRepository(db)
    run = await run_repo.get_by_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Discovery run not found")
    return DiscoveryRunResponse.model_validate(run)


@router.get("/settings", response_model=DiscoverySettingsResponse)
async def get_discovery_settings() -> DiscoverySettingsResponse:
    """Return the current discovery and safety policy configuration.

    Credentials are never included.
    """
    return DiscoverySettingsResponse(
        discovery_mode=settings.discovery_mode,
        discovery_interval_hours=settings.discovery_interval_hours,
        discovery_min_stars=settings.discovery_min_stars,
        discovery_max_stars=settings.discovery_max_stars,
        discovery_languages=settings.discovery_languages,
        discovery_require_license=settings.discovery_require_license,
        discovery_skip_forks=settings.discovery_skip_forks,
        discovery_skip_archived=settings.discovery_skip_archived,
        discovery_max_staleness_days=settings.discovery_max_staleness_days,
        discovery_daily_repo_limit=settings.discovery_daily_repo_limit,
        discovery_max_concurrent=settings.discovery_max_concurrent,
        discovery_max_size_kb=settings.discovery_max_size_kb,
        discovery_excluded_topics=settings.discovery_excluded_topics,
        discovery_excluded_owners=settings.discovery_excluded_owners,
        safety_max_repo_size_kb=settings.safety_max_repo_size_kb,
        safety_max_file_count=settings.safety_max_file_count,
        safety_allow_docker_exec=settings.safety_allow_docker_exec,
        safety_allow_sandbox_network=settings.safety_allow_sandbox_network,
        safety_allow_dep_install=settings.safety_allow_dep_install,
    )

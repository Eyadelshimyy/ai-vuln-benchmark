from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.project import Project
from app.schemas.analysis import AnalysisResponse, AnalysisSummarySchema
from app.schemas.debugging import (
    DebuggingSessionResponse,
    PaginatedDebuggingSessionsResponse,
    StartDebuggingRequest,
)
from app.schemas.project import ProjectCreate, ProjectListResponse, ProjectResponse
from app.schemas.repair import (
    PaginatedRepairSessionsResponse,
    RepairSessionResponse,
    StartRepairRequest,
)
from app.schemas.reproduction import (
    BugReproductionSessionResponse,
    PaginatedReproductionSessionsResponse,
    StartReproductionRequest,
)
from app.schemas.test_generation import (
    PaginatedGeneratedTestsResponse,
    PaginatedTestGenSessionsResponse,
    StartTestGenerationRequest,
    TestGenerationSessionResponse,
)
from app.schemas.test_run import PaginatedTestRunsResponse, TestRunResponse
from app.services.analysis_service import AnalysisService
from app.services.project_service import ProjectNotFoundError, ProjectService

router = APIRouter(prefix="/projects", tags=["Projects"])
logger = logging.getLogger(__name__)


def _project_to_response(project: Project) -> ProjectResponse:
    latest = project.analyses[0] if project.analyses else None
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        repository_path=project.repository_path,
        created_at=project.created_at,
        updated_at=project.updated_at,
        latest_analysis_id=latest.id if latest else None,
        latest_analysis_status=latest.status if latest else None,
    )


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    service = ProjectService(db)
    try:
        project = await service.create_project(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return _project_to_response(project)


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> ProjectListResponse:
    service = ProjectService(db)
    return await service.list_projects(offset=offset, limit=limit)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    service = ProjectService(db)
    try:
        project = await service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return _project_to_response(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    service = ProjectService(db)
    try:
        await service.delete_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")


@router.post(
    "/{project_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_analysis(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> AnalysisResponse:
    # Verify the project exists (uses the request-scoped session)
    project_service = ProjectService(db)
    try:
        project = await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # Create the analysis record with the request session so tests can override it
    from app.repositories.analysis_repo import AnalysisRepository as _Repo

    analysis_repo = _Repo(db)
    analysis = await analysis_repo.create(
        project_id=project.id, repository_path=project.repository_path
    )
    await db.commit()
    await db.refresh(analysis)

    # Schedule the heavy analysis to run after the response is sent
    analysis_service = AnalysisService()
    background_tasks.add_task(
        analysis_service.run_analysis,
        analysis_id=analysis.id,
        repository_path=project.repository_path,
    )

    return AnalysisResponse(
        id=analysis.id,
        project_id=analysis.project_id,
        status=analysis.status,
        repository_path=analysis.repository_path,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        error_message=analysis.error_message,
        summary=None,
        created_at=analysis.created_at,
    )


@router.get("/{project_id}/analyses")
async def list_project_analyses(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    from app.repositories.analysis_repo import AnalysisRepository

    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    analysis_repo = AnalysisRepository(db)
    analyses, total = await analysis_repo.list_for_project(project_id, offset=offset, limit=limit)

    items = []
    for a in analyses:
        summary = AnalysisSummarySchema(**a.summary) if a.summary else None
        items.append(
            AnalysisResponse(
                id=a.id,
                project_id=a.project_id,
                status=a.status,
                repository_path=a.repository_path,
                started_at=a.started_at,
                completed_at=a.completed_at,
                error_message=a.error_message,
                summary=summary,
                created_at=a.created_at,
            )
        )

    return {
        "items": [i.model_dump() for i in items],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.post(
    "/{project_id}/tests/run",
    response_model=TestRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_test_run(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> TestRunResponse:
    project_service = ProjectService(db)
    try:
        project = await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.execution import ExecutorFactory
    from app.repositories.test_run_repo import TestRunRepository
    from app.services.test_runner_service import TestRunnerService
    from app.workers.job_runner import FastAPIBackgroundRunner

    run_repo = TestRunRepository(db)
    run = await run_repo.create(project_id=project.id, repository_path=project.repository_path)
    await db.commit()
    await db.refresh(run)

    svc = TestRunnerService(executor=ExecutorFactory.create())
    runner = FastAPIBackgroundRunner(background_tasks)
    runner.submit(
        svc.execute_test_run,
        test_run_id=run.id,
        repository_path=project.repository_path,
    )

    return TestRunResponse(
        id=run.id,
        project_id=run.project_id,
        status=run.status,
        framework=run.framework,
        repository_path=run.repository_path,
        command=run.command,
        exit_code=run.exit_code,
        duration_seconds=run.duration_seconds,
        error_message=run.error_message,
        total_tests=run.total_tests,
        passed=run.passed,
        failed=run.failed,
        skipped=run.skipped,
        errors=run.errors,
        started_at=run.started_at,
        completed_at=run.completed_at,
        created_at=run.created_at,
    )


@router.get("/{project_id}/test-runs", response_model=PaginatedTestRunsResponse)
async def list_project_test_runs(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedTestRunsResponse:
    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.test_run_repo import TestRunRepository

    run_repo = TestRunRepository(db)
    runs, total = await run_repo.list_for_project(project_id, offset=offset, limit=limit)

    return PaginatedTestRunsResponse(
        items=runs,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post(
    "/{project_id}/debug",
    response_model=DebuggingSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_debugging_session(
    project_id: UUID,
    request: StartDebuggingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> DebuggingSessionResponse:
    project_service = ProjectService(db)
    try:
        project = await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.debugging_repo import DebuggingRepository
    from app.services.debugging_service import DebuggingService
    from app.workers.job_runner import FastAPIBackgroundRunner

    # Create the session record in the request session (test-overrideable)
    debug_repo = DebuggingRepository(db)
    ds = await debug_repo.create_session(
        project_id=project.id,
        analysis_id=request.analysis_id,
        test_run_id=request.test_run_id,
    )
    await db.commit()
    await db.refresh(ds)

    svc = DebuggingService()
    FastAPIBackgroundRunner(background_tasks).submit(
        svc.run_session,
        session_id=ds.id,
        repository_path=project.repository_path,
        project_name=project.name,
    )

    return DebuggingSessionResponse(
        id=ds.id,
        project_id=ds.project_id,
        analysis_id=ds.analysis_id,
        test_run_id=ds.test_run_id,
        status=ds.status,
        error_message=ds.error_message,
        created_at=ds.created_at,
        started_at=ds.started_at,
        completed_at=ds.completed_at,
        hypothesis_count=0,
    )


@router.get("/{project_id}/debugging", response_model=PaginatedDebuggingSessionsResponse)
async def list_project_debugging_sessions(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedDebuggingSessionsResponse:
    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.debugging_repo import DebuggingRepository

    repo = DebuggingRepository(db)
    sessions, total = await repo.list_for_project(project_id, offset=offset, limit=limit)

    return PaginatedDebuggingSessionsResponse(
        items=[
            DebuggingSessionResponse(
                id=s.id,
                project_id=s.project_id,
                analysis_id=s.analysis_id,
                test_run_id=s.test_run_id,
                status=s.status,
                error_message=s.error_message,
                created_at=s.created_at,
                started_at=s.started_at,
                completed_at=s.completed_at,
                hypothesis_count=len(s.hypotheses),
            )
            for s in sessions
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post(
    "/{project_id}/test-generation",
    response_model=TestGenerationSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_test_generation(
    project_id: UUID,
    request: StartTestGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> TestGenerationSessionResponse:
    project_service = ProjectService(db)
    try:
        project = await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.test_generation_repo import TestGenerationRepository
    from app.services.test_generation_service import TestGenerationService
    from app.workers.job_runner import FastAPIBackgroundRunner as _TGRunner

    gen_repo = TestGenerationRepository(db)
    s = await gen_repo.create_session(
        project_id=project.id,
        analysis_id=request.analysis_id,
        test_run_id=request.test_run_id,
        debugging_session_id=request.debugging_session_id,
    )
    await db.commit()
    await db.refresh(s)

    svc = TestGenerationService()
    _TGRunner(background_tasks).submit(
        svc.run_session,
        session_id=s.id,
        project_id=project.id,
        repository_path=project.repository_path,
        project_name=project.name,
        analysis_id=request.analysis_id,
        test_run_id=request.test_run_id,
        debugging_session_id=request.debugging_session_id,
    )
    return TestGenerationSessionResponse.model_validate(s)


@router.get("/{project_id}/test-generation", response_model=PaginatedTestGenSessionsResponse)
async def list_test_generation_sessions(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedTestGenSessionsResponse:
    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.test_generation_repo import TestGenerationRepository

    repo = TestGenerationRepository(db)
    sessions, total = await repo.list_sessions(project_id, offset=offset, limit=limit)
    return PaginatedTestGenSessionsResponse(
        items=sessions,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{project_id}/generated-tests", response_model=PaginatedGeneratedTestsResponse)
async def list_project_generated_tests(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> PaginatedGeneratedTestsResponse:
    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.test_generation_repo import TestGenerationRepository

    repo = TestGenerationRepository(db)
    tests, total = await repo.list_for_project(project_id, offset=offset, limit=limit)
    return PaginatedGeneratedTestsResponse(
        items=tests,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post(
    "/{project_id}/reproduction",
    response_model=BugReproductionSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_reproduction(
    project_id: UUID,
    request: StartReproductionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> BugReproductionSessionResponse:
    project_service = ProjectService(db)
    try:
        project = await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.reproduction_repo import ReproductionRepository
    from app.services.reproduction_service import BugReproductionService
    from app.workers.job_runner import FastAPIBackgroundRunner as _BRRunner

    repo = ReproductionRepository(db)
    s = await repo.create(
        project_id=project.id,
        hypothesis_id=request.hypothesis_id,
        generated_test_id=request.generated_test_id,
        debugging_session_id=request.debugging_session_id,
        total_attempts=request.total_attempts,
    )
    await db.commit()
    await db.refresh(s)

    svc = BugReproductionService()
    _BRRunner(background_tasks).submit(
        svc.run_session,
        session_id=s.id,
        project_id=project.id,
        repository_path=project.repository_path,
        hypothesis_id=request.hypothesis_id,
        generated_test_id=request.generated_test_id,
        total_attempts=request.total_attempts,
    )
    return BugReproductionSessionResponse.model_validate(s)


@router.get("/{project_id}/reproduction", response_model=PaginatedReproductionSessionsResponse)
async def list_reproduction_sessions(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedReproductionSessionsResponse:
    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.reproduction_repo import ReproductionRepository

    repo = ReproductionRepository(db)
    sessions, total = await repo.list_for_project(project_id, offset=offset, limit=limit)
    return PaginatedReproductionSessionsResponse(
        items=sessions,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )


# ── Repair endpoints ──────────────────────────────────────────────────────────


@router.post(
    "/{project_id}/repair",
    response_model=RepairSessionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_repair(
    project_id: UUID,
    request: StartRepairRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> RepairSessionResponse:
    project_service = ProjectService(db)
    try:
        project = await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.repair_repo import RepairRepository
    from app.services.repair_service import RepairService
    from app.workers.job_runner import FastAPIBackgroundRunner as _RRRunner

    repo = RepairRepository(db)
    s = await repo.create_session(
        project_id=project.id,
        hypothesis_id=request.hypothesis_id,
        reproduction_session_id=request.reproduction_session_id,
        debugging_session_id=request.debugging_session_id,
    )
    await db.commit()
    await db.refresh(s)

    svc = RepairService()
    _RRRunner(background_tasks).submit(
        svc.run_session,
        session_id=s.id,
        project_id=project.id,
        repository_path=project.repository_path,
        hypothesis_id=request.hypothesis_id,
        reproduction_session_id=request.reproduction_session_id,
        debugging_session_id=request.debugging_session_id,
        max_candidates=request.max_candidates,
    )
    return RepairSessionResponse.model_validate(s)


@router.get("/{project_id}/repair", response_model=PaginatedRepairSessionsResponse)
async def list_repair_sessions(
    project_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> PaginatedRepairSessionsResponse:
    project_service = ProjectService(db)
    try:
        await project_service.get_project(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.repositories.repair_repo import RepairRepository

    repo = RepairRepository(db)
    sessions, total = await repo.list_for_project(project_id, offset=offset, limit=limit)
    return PaginatedRepairSessionsResponse(
        items=sessions,  # type: ignore[arg-type]
        total=total,
        offset=offset,
        limit=limit,
    )

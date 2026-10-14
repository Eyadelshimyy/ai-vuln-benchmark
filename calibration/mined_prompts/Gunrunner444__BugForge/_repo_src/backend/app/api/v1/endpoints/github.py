"""GitHub Integration API endpoints — Phase 9."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.repositories.github_repo import GitHubDeliveryRepo, GitHubRepositoryRepo
from app.repositories.project_repo import ProjectRepository
from app.schemas.github import (
    GitHubConnectRequest,
    GitHubDeliveryResponse,
    GitHubRepositoryResponse,
)

router = APIRouter(tags=["GitHub Integration"])
logger = logging.getLogger(__name__)


def _repo_response(gr: object) -> GitHubRepositoryResponse:
    from app.models.github import GitHubRepository

    assert isinstance(gr, GitHubRepository)
    return GitHubRepositoryResponse(
        id=gr.id,
        project_id=gr.project_id,
        owner=gr.owner,
        repo=gr.repo,
        github_id=gr.github_id,
        default_branch=gr.default_branch,
        html_url=gr.html_url,
        connected=gr.connected,
        created_at=gr.created_at,
        updated_at=gr.updated_at,
    )


def _delivery_response(d: object) -> GitHubDeliveryResponse:
    from app.models.github import GitHubDelivery

    assert isinstance(d, GitHubDelivery)
    return GitHubDeliveryResponse(
        id=d.id,
        project_id=d.project_id,
        candidate_id=d.candidate_id,
        verification_id=d.verification_id,
        owner=d.owner,
        repo=d.repo,
        base_branch=d.base_branch,
        delivery_branch=d.delivery_branch,
        commit_sha=d.commit_sha,
        pull_request_number=d.pull_request_number,
        pull_request_url=d.pull_request_url,
        verified_patch_hash=d.verified_patch_hash,
        delivered_patch_hash=d.delivered_patch_hash,
        status=d.status,
        error_message=d.error_message,
        created_at=d.created_at,
        started_at=d.started_at,
        completed_at=d.completed_at,
    )


@router.post(
    "/projects/{project_id}/github/connect",
    response_model=GitHubRepositoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Connect a GitHub repository to a project",
)
async def connect_github_repo(
    project_id: UUID,
    body: GitHubConnectRequest,
    db: AsyncSession = Depends(get_db),
) -> GitHubRepositoryResponse:
    """Verify GitHub access and associate a repository with the project."""
    project_repo = ProjectRepository(db)
    project = await project_repo.get_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.services.github_service import GitHubService

    try:
        svc = GitHubService()
        gr_id = await svc.connect_repository(
            project_id=project_id,
            owner=body.owner,
            repo=body.repo,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    github_repo_repo = GitHubRepositoryRepo(db)
    gr = await github_repo_repo.get_by_id(gr_id)
    if gr is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Repository not found after creation",
        )
    return _repo_response(gr)


@router.get(
    "/projects/{project_id}/github",
    response_model=GitHubRepositoryResponse,
    summary="Get the connected GitHub repository for a project",
)
async def get_github_repo(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> GitHubRepositoryResponse:
    project_repo = ProjectRepository(db)
    project = await project_repo.get_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    github_repo_repo = GitHubRepositoryRepo(db)
    gr = await github_repo_repo.get_by_project(project_id)
    if gr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No GitHub repository connected to this project",
        )
    return _repo_response(gr)


@router.delete(
    "/projects/{project_id}/github",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Disconnect the GitHub repository from a project",
)
async def disconnect_github_repo(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    project_repo = ProjectRepository(db)
    project = await project_repo.get_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.services.github_service import GitHubService

    svc = GitHubService()
    await svc.disconnect_repository(project_id)


@router.post(
    "/projects/{project_id}/github/deliver/{candidate_id}",
    response_model=GitHubDeliveryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Deliver a verified patch candidate to GitHub",
)
async def deliver_candidate(
    project_id: UUID,
    candidate_id: UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> GitHubDeliveryResponse:
    """Create a delivery pipeline for a VERIFIED patch candidate.

    Only candidates with verification_decision='verified' can be delivered.
    Unverified, rejected, or inconclusive candidates are rejected with 400.
    """
    project_repo = ProjectRepository(db)
    project = await project_repo.get_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    from app.services.github_service import GitHubService
    from app.workers.job_runner import FastAPIBackgroundRunner

    try:
        svc = GitHubService()
        delivery_id = await svc.start_delivery(
            project_id=project_id,
            candidate_id=candidate_id,
        )
    except ValueError as exc:
        # Check if it's a "not found" vs a security rejection
        msg = str(exc)
        if "not found" in msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg) from exc

    # Queue background delivery
    FastAPIBackgroundRunner(background_tasks).submit(
        svc.run_delivery_pipeline,
        delivery_id=delivery_id,
    )

    delivery_repo = GitHubDeliveryRepo(db)
    delivery = await delivery_repo.get_by_id(delivery_id)
    if delivery is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Delivery record not found"
        )
    return _delivery_response(delivery)


@router.get(
    "/projects/{project_id}/github/deliveries",
    response_model=list[GitHubDeliveryResponse],
    summary="List all GitHub deliveries for a project",
)
async def list_project_deliveries(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[GitHubDeliveryResponse]:
    project_repo = ProjectRepository(db)
    project = await project_repo.get_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    delivery_repo = GitHubDeliveryRepo(db)
    deliveries = await delivery_repo.list_for_project(project_id)
    return [_delivery_response(d) for d in deliveries]


@router.get(
    "/github/deliveries/{delivery_id}",
    response_model=GitHubDeliveryResponse,
    summary="Get a GitHub delivery record by ID",
)
async def get_delivery(
    delivery_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> GitHubDeliveryResponse:
    delivery_repo = GitHubDeliveryRepo(db)
    delivery = await delivery_repo.get_by_id(delivery_id)
    if delivery is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Delivery not found")
    return _delivery_response(delivery)

"""Comment API endpoints."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.document import CommentCreate, CommentResponse
from app.services.document_service import DocumentService

logger = structlog.get_logger()
router = APIRouter()


@router.post(
    "/{document_id}/comments",
    response_model=CommentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_comment(
    document_id: UUID,
    body: CommentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a comment to a document."""
    service = DocumentService(db)
    comment = await service.add_comment(document_id, body)
    if not comment:
        raise HTTPException(status_code=404, detail="Document not found")
    logger.info("comment_added", document_id=str(document_id), comment_id=str(comment.id))
    return comment


@router.get("/{document_id}/comments", response_model=list[CommentResponse])
async def list_comments(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """List comments for a document."""
    service = DocumentService(db)
    return await service.list_comments(document_id)


@router.delete(
    "/{document_id}/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_comment(
    document_id: UUID,
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a comment."""
    service = DocumentService(db)
    deleted = await service.delete_comment(document_id, comment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Comment not found")
    logger.info(
        "comment_deleted", document_id=str(document_id), comment_id=str(comment_id)
    )

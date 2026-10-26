"""Pydantic schemas for request/response validation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ---- Document schemas ----


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(default="")
    content_type: str = Field(default="text/markdown")
    owner_id: UUID | None = None
    folder_id: UUID | None = None


class DocumentUpdate(BaseModel):
    """Full replace (PUT)."""

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(default="")
    content_type: str = Field(default="text/markdown")
    folder_id: UUID | None = None


class DocumentPatch(BaseModel):
    """Partial update (PATCH)."""

    title: str | None = Field(None, min_length=1, max_length=500)
    content: str | None = None
    content_type: str | None = None
    folder_id: UUID | None = None

    @field_validator("title", "content", "content_type", mode="before")
    @classmethod
    def reject_null_for_not_null_fields(cls, v: object, info) -> object:  # noqa: N805
        if info.field_name in ("title", "content", "content_type") and v is None:
            raise ValueError(f"{info.field_name} cannot be null")
        return v


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    content: str
    content_type: str
    owner_id: UUID
    folder_id: UUID | None
    is_deleted: bool
    is_template: bool
    word_count: int
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    size: int
    pages: int


# ---- Version schemas ----


class DocumentVersionResponse(BaseModel):
    id: UUID
    document_id: UUID
    version_number: int
    title: str
    content: str
    created_by: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


# ---- Comment schemas ----


class CommentCreate(BaseModel):
    author_id: UUID
    content: str = Field(..., min_length=1)


class CommentResponse(BaseModel):
    id: UUID
    document_id: UUID
    author_id: UUID
    content: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---- Template schemas ----


class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="")
    content: str = Field(default="")
    content_type: str = Field(default="text/markdown")
    created_by: UUID


class TemplateResponse(BaseModel):
    id: UUID
    name: str
    description: str
    content: str
    content_type: str
    created_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentFromTemplate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    owner_id: UUID
    folder_id: UUID | None = None

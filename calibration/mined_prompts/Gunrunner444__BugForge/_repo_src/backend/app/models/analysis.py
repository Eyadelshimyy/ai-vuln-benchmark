from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.finding import DBFinding
    from app.models.project import Project


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # pending | running | completed | failed
    repository_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Aggregated summary stored as JSON to avoid expensive joins for list views
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    project: Mapped[Project] = relationship("Project", back_populates="analyses")
    files: Mapped[list[RepositoryFile]] = relationship(
        "RepositoryFile",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )
    findings: Mapped[list[DBFinding]] = relationship(
        "DBFinding",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )


class RepositoryFile(Base):
    __tablename__ = "repository_files"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relative_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    file_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # source | test | config | other
    language: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    line_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    has_errors: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parser_backend: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parser_tier: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    analysis: Mapped[Analysis] = relationship("Analysis", back_populates="files")
    entities: Mapped[list[CodeEntity]] = relationship(
        "CodeEntity", back_populates="file", cascade="all, delete-orphan"
    )
    imports: Mapped[list[ImportRecord]] = relationship(
        "ImportRecord", back_populates="file", cascade="all, delete-orphan"
    )


class CodeEntity(Base):
    __tablename__ = "code_entities"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    file_id: Mapped[UUID] = mapped_column(
        ForeignKey("repository_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # function | async_function | class | method | async_method
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    qualified_name: Mapped[str] = mapped_column(String(1000), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_async: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decorators: Mapped[list[str] | None] = mapped_column(JSON, nullable=True, default=list)
    parameters: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True, default=list
    )
    return_annotation: Mapped[str | None] = mapped_column(String(500), nullable=True)
    parent_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    file: Mapped[RepositoryFile] = relationship("RepositoryFile", back_populates="entities")


class ImportRecord(Base):
    __tablename__ = "import_records"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    file_id: Mapped[UUID] = mapped_column(
        ForeignKey("repository_files.id", ondelete="CASCADE"), nullable=False, index=True
    )
    module_name: Mapped[str] = mapped_column(String(500), nullable=False)
    imported_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    alias: Mapped[str | None] = mapped_column(String(500), nullable=True)
    import_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # stdlib | third_party | relative | local
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    is_from_import: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    file: Mapped[RepositoryFile] = relationship("RepositoryFile", back_populates="imports")

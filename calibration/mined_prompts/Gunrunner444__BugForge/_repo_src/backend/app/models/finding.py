from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class DBFinding(Base):
    """Persisted static-analysis finding."""

    __tablename__ = "findings"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    analysis_id: Mapped[UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    column: Mapped[int | None] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    analyzer: Mapped[str] = mapped_column(String(100), nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, default="")
    suggested_fix: Mapped[str] = mapped_column(Text, nullable=False, default="")
    catalog: Mapped[str] = mapped_column(
        String(50), nullable=False, default="code_quality", index=True
    )
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parser_backend: Mapped[str | None] = mapped_column(String(50), nullable=True)
    node_id: Mapped[str | None] = mapped_column(String(500), nullable=True)
    start_byte: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_byte: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_column: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_column: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    analysis: Mapped[Analysis] = relationship("Analysis", back_populates="findings")

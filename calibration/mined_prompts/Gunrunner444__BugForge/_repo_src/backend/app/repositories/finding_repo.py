from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.finding import Finding as FindingDC
from app.models.finding import DBFinding


class FindingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def bulk_create(self, analysis_id: UUID, findings: list[FindingDC]) -> int:
        """Insert all findings for an analysis and return the count."""
        for f in findings:
            self.session.add(
                DBFinding(
                    analysis_id=analysis_id,
                    category=f.category,
                    severity=f.severity,
                    confidence=f.confidence,
                    file_path=f.file_path,
                    line=f.line,
                    end_line=f.end_line,
                    column=f.column,
                    message=f.message,
                    explanation=f.explanation,
                    analyzer=f.analyzer,
                    evidence=f.evidence,
                    suggested_fix=f.suggested_fix,
                    catalog=getattr(f, "catalog", "code_quality"),
                    language=getattr(f, "language", None),
                    parser_backend=getattr(f, "parser_backend", None),
                    node_id=getattr(f, "node_id", None) or None,
                    start_byte=getattr(f, "start_byte", None),
                    end_byte=getattr(f, "end_byte", None),
                    start_column=getattr(f, "start_column", None) or f.column,
                    end_column=getattr(f, "end_column", None),
                )
            )
        await self.session.flush()
        return len(findings)

    async def list_for_analysis(
        self,
        analysis_id: UUID,
        severity: str | None = None,
        category: str | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> tuple[list[DBFinding], int]:
        filters = [DBFinding.analysis_id == analysis_id]
        if severity:
            filters.append(DBFinding.severity == severity)
        if category:
            filters.append(DBFinding.category == category)

        count_result = await self.session.execute(
            select(func.count()).select_from(DBFinding).where(*filters)
        )
        total: int = count_result.scalar_one()

        result = await self.session.execute(
            select(DBFinding)
            .where(*filters)
            .order_by(DBFinding.severity.desc(), DBFinding.line)
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from app.analysis.engine import StaticAnalysisEngine
from app.analyzers.repo_analyzer import AnalysisResult, RepoAnalyzer
from app.models.analysis import Analysis, CodeEntity, ImportRecord, RepositoryFile
from app.parsing.engine import installed_parser_report

logger = logging.getLogger(__name__)


class AnalysisNotFoundError(Exception):
    pass


class AnalysisService:
    """Orchestrates repository analysis and persists results."""

    def __init__(self) -> None:
        self._analyzer = RepoAnalyzer()
        self._static_engine = StaticAnalysisEngine()

    async def start_analysis(self, project_id: UUID, repository_path: str) -> Analysis:
        """Creates the analysis record and returns it. The caller schedules run_analysis()."""
        from app.database import async_session_factory
        from app.repositories.analysis_repo import AnalysisRepository

        async with async_session_factory() as session:
            repo = AnalysisRepository(session)
            analysis = await repo.create(project_id=project_id, repository_path=repository_path)
            await session.commit()
            await session.refresh(analysis)
            return analysis

    async def run_analysis(self, analysis_id: UUID, repository_path: str) -> None:
        """Full analysis pipeline. Designed to run as a background task."""
        from app.database import async_session_factory
        from app.repositories.analysis_repo import AnalysisRepository

        logger.info("Starting analysis %s for %s", analysis_id, repository_path)

        async with async_session_factory() as session:
            repo = AnalysisRepository(session)
            await repo.update_status(analysis_id, "running")
            await session.commit()

        try:
            start_time = time.monotonic()
            analysis_result = await asyncio.to_thread(self._analyzer.analyze, Path(repository_path))
            duration = time.monotonic() - start_time

            # Run static analysis on the files collected during repo analysis
            all_file_paths = [fr.absolute_path for fr in analysis_result.file_results]
            static_findings = await asyncio.to_thread(
                self._static_engine.analyze_repository,
                Path(repository_path),
                all_file_paths,
            )

            from app.security.engine import SecurityAnalysisEngine

            security_result = await asyncio.to_thread(
                SecurityAnalysisEngine().analyze_repository,
                Path(repository_path),
                all_file_paths,
            )

            async with async_session_factory() as session:
                await self._persist_results(
                    session,
                    analysis_id,
                    analysis_result,
                    duration,
                    static_findings,
                    security_findings=security_result.findings,
                )
                await session.commit()

            logger.info(
                "Analysis %s completed in %.2fs — %d files, %d findings",
                analysis_id,
                duration,
                analysis_result.total_files,
                len(static_findings),
            )
        except Exception as exc:
            logger.exception("Analysis %s failed: %s", analysis_id, exc)
            async with async_session_factory() as session:
                repo = AnalysisRepository(session)
                await repo.update_status(analysis_id, "failed", error_message=str(exc))
                await session.commit()

    async def _persist_results(
        self,
        session: Any,
        analysis_id: UUID,
        result: AnalysisResult,
        duration: float,
        static_findings: list[Any] | None = None,
        security_findings: list[Any] | None = None,
    ) -> None:
        from app.repositories.analysis_repo import AnalysisRepository
        from app.repositories.finding_repo import FindingRepository

        total_entities = 0
        total_imports = 0

        for fr in result.file_results:
            file_record = RepositoryFile(
                analysis_id=analysis_id,
                relative_path=fr.relative_path,
                file_type=fr.file_type,
                language=fr.language,
                size_bytes=fr.size_bytes,
                line_count=fr.line_count,
                has_errors=fr.has_parse_errors,
                parser_backend=fr.parse_result.parser_backend if fr.parse_result else None,
                parser_tier=fr.parse_result.parser_tier if fr.parse_result else None,
                error_count=fr.parse_result.error_count if fr.parse_result else 0,
            )
            session.add(file_record)
            await session.flush()  # obtain file_record.id

            if fr.parse_result:
                for entity in fr.parse_result.entities:
                    session.add(
                        CodeEntity(
                            file_id=file_record.id,
                            entity_type=entity.entity_type,
                            name=entity.name,
                            qualified_name=entity.qualified_name,
                            start_line=entity.start_line,
                            end_line=entity.end_line,
                            docstring=entity.docstring,
                            is_async="async" in entity.entity_type,
                            decorators=entity.decorators,
                            parameters=[
                                {
                                    "name": p.name,
                                    "annotation": p.annotation,
                                    "default": p.default,
                                    "kind": p.kind,
                                }
                                for p in entity.parameters
                            ],
                            return_annotation=entity.return_annotation,
                            parent_name=entity.parent,
                        )
                    )
                total_entities += len(fr.parse_result.entities)

                for imp in fr.parse_result.imports:
                    session.add(
                        ImportRecord(
                            file_id=file_record.id,
                            module_name=imp.module,
                            imported_name=imp.name,
                            alias=imp.alias,
                            import_type=imp.import_type,
                            line_number=imp.line_number,
                            is_from_import=imp.is_from_import,
                        )
                    )
                total_imports += len(fr.parse_result.imports)

        summary = {
            "total_files": result.total_files,
            "source_files": result.source_file_count,
            "test_files": result.test_file_count,
            "ignored_files": result.ignored_file_count,
            "total_entities": total_entities,
            "total_imports": total_imports,
            "total_findings": len(static_findings) if static_findings else 0,
            "security_findings": len(security_findings) if security_findings else 0,
            "languages": [
                {
                    "language": ls.language,
                    "file_count": ls.file_count,
                    "percentage": ls.percentage,
                }
                for ls in result.language_stats
            ],
            "frameworks": [
                {
                    "name": fw.name,
                    "language": fw.language,
                    "confidence": fw.confidence,
                    "evidence": fw.evidence,
                }
                for fw in result.framework_detections
            ],
            "analysis_duration_seconds": round(duration, 3),
            "language_capabilities": _language_capability_summary(),
        }

        repo = AnalysisRepository(session)
        await repo.complete_analysis(analysis_id, summary)

        if static_findings:
            finding_repo = FindingRepository(session)
            await finding_repo.bulk_create(analysis_id, static_findings)

        if security_findings:
            from app.models.analysis import Analysis
            from app.repositories.security_finding_repo import SecurityFindingRepository
            from app.services.finding_lifecycle import FindingLifecycleService

            analysis = await session.get(Analysis, analysis_id)
            project_id = analysis.project_id if analysis is not None else None
            service = FindingLifecycleService(SecurityFindingRepository(session))
            await service.persist_static_scan(
                security_findings, project_id=project_id, analysis_id=analysis_id
            )


def _language_capability_summary() -> list[dict[str, Any]]:
    from app.plugins import get_plugin_catalog

    catalog = get_plugin_catalog()
    rows: list[dict[str, Any]] = []
    for adapter in catalog.languages.all_adapters():
        rows.append(
            {
                "language": adapter.language_id,
                "parser_tier": str(adapter.parser_tier()),
                "parser_backend": adapter.parser_backend(),
                "capabilities": sorted(cap.value for cap in adapter.capabilities),
                "native_available": bool(
                    installed_parser_report(adapter.language_id)["native_available"]
                ),
                "parser_status": str(installed_parser_report(adapter.language_id)["status"]),
            }
        )
    return rows

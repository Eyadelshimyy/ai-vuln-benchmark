"""Debugging session service — orchestrates AI analysis pipeline."""

from __future__ import annotations

import json
import logging
from uuid import UUID

logger = logging.getLogger(__name__)


class DebuggingNotFoundError(Exception):
    pass


class DebuggingService:
    async def start_session(
        self,
        project_id: UUID,
        analysis_id: UUID | None,
        test_run_id: UUID | None,
    ) -> object:
        """Create a pending debugging session and return it."""
        from app.database import async_session_factory
        from app.repositories.debugging_repo import DebuggingRepository

        async with async_session_factory() as session:
            repo = DebuggingRepository(session)
            ds = await repo.create_session(
                project_id=project_id,
                analysis_id=analysis_id,
                test_run_id=test_run_id,
            )
            await session.commit()
            await session.refresh(ds)
            return ds

    async def run_session(self, session_id: UUID, repository_path: str, project_name: str) -> None:
        """Full AI analysis pipeline. Runs as a background task."""
        from app.ai import ContextBuilder, get_provider
        from app.database import async_session_factory
        from app.repositories.debugging_repo import DebuggingRepository

        logger.info("Starting debugging session %s", session_id)

        async with async_session_factory() as db_session:
            repo = DebuggingRepository(db_session)
            await repo.update_status(session_id, "running")
            await db_session.commit()

        try:
            async with async_session_factory() as db_session:
                ds = await DebuggingRepository(db_session).get_by_id(session_id)
                if ds is None:
                    raise ValueError(f"Session {session_id} not found")

                context_builder = ContextBuilder(db_session)
                request = await context_builder.build(
                    project_name=project_name,
                    repository_path=repository_path,
                    test_run_id=ds.test_run_id,
                    analysis_id=ds.analysis_id,
                )

            context_summary = json.dumps(
                {
                    "failing_tests": len(request.failing_tests),
                    "static_findings": len(request.static_findings),
                    "source_files": len(request.source_files),
                }
            )

            provider = get_provider()
            response = await provider.analyze(request)

            async with async_session_factory() as db_session:
                repo = DebuggingRepository(db_session)

                for hypothesis in response.hypotheses:
                    await repo.add_hypothesis(
                        session_id=session_id,
                        hypothesis=hypothesis,
                        provider=response.provider,
                        model=response.model,
                    )

                await repo.record_ai_call(
                    session_id=session_id,
                    provider=response.provider,
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    duration_seconds=response.duration_seconds,
                    success=response.error is None,
                    error_message=response.error,
                )

                final_status = "failed" if response.error else "completed"
                await repo.update_status(
                    session_id,
                    final_status,
                    error_message=response.error,
                    context_summary=context_summary,
                )
                await db_session.commit()

            logger.info(
                "Debugging session %s %s — %d hypotheses from %s/%s",
                session_id,
                final_status,
                len(response.hypotheses),
                response.provider,
                response.model,
            )

        except Exception as exc:
            logger.exception("Debugging session %s failed: %s", session_id, exc)
            async with async_session_factory() as db_session:
                repo = DebuggingRepository(db_session)
                await repo.update_status(session_id, "failed", error_message=str(exc))
                await db_session.commit()

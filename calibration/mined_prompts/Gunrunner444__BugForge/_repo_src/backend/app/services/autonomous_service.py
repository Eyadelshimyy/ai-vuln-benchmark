"""Autonomous analysis orchestration service (v1.1.0).

Coordinates the discovery → eligibility → acquire → static-analysis →
AI-debugging → test-generation → reproduction → repair → verification pipeline
for public GitHub repositories.

Design invariants:
  - Never executes repository code on the BugForge host.
  - All code execution goes through DockerTestExecutor.
  - Never opens a GitHub PR for a mere AI hypothesis.
  - Respects concurrency limits from settings.
  - Repository data is always labelled as untrusted.
  - Cleanup always runs (success, failure, or cancellation).
  - Each run has a hard 30-minute timeout covering the full pipeline.
  - AI infrastructure failure is distinguishable from empty AI results.
  - Permanent workspace paths are stored in project records; temp dirs are
    never persisted.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from pathlib import Path
from uuid import UUID

from app.core.config import Settings
from app.models.discovery import RepositoryCandidate

logger = logging.getLogger(__name__)

# Maximum time (seconds) spent on one repository before forced cancellation.
_PER_REPO_TIMEOUT_SECONDS = 1800  # 30 minutes

# Characters safe for use in directory names (anything else is replaced with _)
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


class AIDebuggingError(RuntimeError):
    """Raised when the AI debugging infrastructure fails (not an empty result)."""


class AutonomousAnalysisService:
    """Orchestrates one autonomous analysis run.

    Create a new instance per run. This keeps state local and makes
    cancellation straightforward.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        candidate: RepositoryCandidate,
        autonomous_run_id: UUID,
        *,
        force_rescan: bool = False,
    ) -> None:
        """Execute the full pipeline for one candidate under a hard timeout.

        Updates the autonomous_analysis_run record at each stage transition.
        Uses the existing BugForge pipeline services; does NOT implement a
        parallel pipeline.

        Cleanup always executes regardless of success, failure, or cancellation.
        """
        try:
            await asyncio.wait_for(
                self._run_pipeline(candidate, autonomous_run_id, force_rescan=force_rescan),
                timeout=_PER_REPO_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.error(
                "Autonomous run %s timed out after %ds",
                autonomous_run_id,
                _PER_REPO_TIMEOUT_SECONDS,
            )
            await self._fail_run(
                autonomous_run_id,
                "timeout",
                f"Pipeline timed out after {_PER_REPO_TIMEOUT_SECONDS}s",
            )

    # ------------------------------------------------------------------
    # Pipeline (runs inside the hard timeout)
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        candidate: RepositoryCandidate,
        autonomous_run_id: UUID,
        *,
        force_rescan: bool = False,
    ) -> None:
        from app.database import async_session_factory
        from app.repositories.discovery_repo import AutonomousAnalysisRunRepository
        from app.services.analysis_service import AnalysisService
        from app.services.eligibility_service import EligibilityService

        start = time.monotonic()
        workspace_dir: Path | None = None

        async with async_session_factory() as session:
            run_repo = AutonomousAnalysisRunRepository(session)
            await run_repo.update_status(autonomous_run_id, "screening", current_stage="screening")
            await session.commit()

        # --- Stage 1: Re-check eligibility against latest settings ---
        try:
            topics: list[str] = []
            if candidate.topics:
                import json

                try:
                    topics = json.loads(candidate.topics)
                except Exception:
                    topics = []

            elig_svc = EligibilityService(self._settings)
            assessment = elig_svc.assess(
                full_name=candidate.full_name,
                owner=candidate.owner,
                stars=candidate.stars,
                is_fork=candidate.is_fork,
                is_archived=candidate.is_archived,
                primary_language=candidate.primary_language,
                license_key=candidate.license_key,
                size_kb=candidate.size_kb,
                topics=topics,
                last_pushed_at=candidate.last_pushed_at,
                description=candidate.description,
            )

            if not assessment.is_eligible:
                async with async_session_factory() as session:
                    run_repo = AutonomousAnalysisRunRepository(session)
                    await run_repo.update_status(
                        autonomous_run_id,
                        "cancelled",
                        current_stage="screening",
                        error_code="eligibility_failed",
                        error_message=assessment.rejection_reason,
                    )
                    await session.commit()
                logger.warning(
                    "Autonomous run %s: candidate %s failed eligibility re-check: %s",
                    autonomous_run_id,
                    candidate.full_name,
                    assessment.rejection_reason,
                )
                return

        except Exception as exc:
            logger.error("Eligibility re-check failed for %s: %s", candidate.full_name, exc)
            await self._fail_run(autonomous_run_id, "screening_error", str(exc))
            return

        # --- Stage 2: Acquire repository (shallow clone) ---
        async with async_session_factory() as session:
            run_repo = AutonomousAnalysisRunRepository(session)
            await run_repo.update_status(autonomous_run_id, "acquiring", current_stage="acquiring")
            await session.commit()

        try:
            workspace_dir, clone_dir, commit_sha = await self._shallow_clone(candidate)
        except Exception as exc:
            logger.error("Clone failed for %s: %s", candidate.full_name, exc)
            await self._fail_run(autonomous_run_id, "clone_failed", str(exc))
            return

        try:
            # --- Stage 3: Check for duplicate commit ---
            if (
                not force_rescan
                and candidate.last_analyzed_commit
                and candidate.last_analyzed_commit == commit_sha
            ):
                logger.info(
                    "Skipping %s — commit %s already analyzed.",
                    candidate.full_name,
                    commit_sha,
                )
                async with async_session_factory() as session:
                    run_repo = AutonomousAnalysisRunRepository(session)
                    await run_repo.update_status(
                        autonomous_run_id,
                        "completed",
                        current_stage=None,
                        error_code="no_new_commit",
                        error_message="Commit unchanged since last analysis; skipped.",
                    )
                    await session.commit()
                return

            async with async_session_factory() as session:
                from sqlalchemy import update as sa_update

                from app.models.discovery import AutonomousAnalysisRun as _Run

                await session.execute(
                    sa_update(_Run)
                    .where(_Run.id == autonomous_run_id)
                    .values(commit_sha=commit_sha)
                )
                await session.commit()

            # --- Stage 4: Create or reuse BugForge Project ---
            # The project record stores the PERMANENT workspace path (not a temp
            # dir) so the path remains valid beyond this analysis run.
            project_id = await self._ensure_project(candidate, clone_dir)

            async with async_session_factory() as session:
                run_repo = AutonomousAnalysisRunRepository(session)
                await run_repo.set_project_id(autonomous_run_id, project_id)
                await run_repo.update_status(
                    autonomous_run_id,
                    "static_analyzing",
                    current_stage="static_analyzing",
                )
                await session.commit()

            # --- Stage 5: Static Analysis (existing pipeline) ---
            analysis_svc = AnalysisService()
            analysis = await analysis_svc.start_analysis(project_id, str(clone_dir))
            await analysis_svc.run_analysis(analysis.id, str(clone_dir))

            static_count = await self._count_static_findings(analysis.id)

            async with async_session_factory() as session:
                run_repo = AutonomousAnalysisRunRepository(session)
                await run_repo.update_status(
                    autonomous_run_id,
                    "ai_analyzing",
                    current_stage="ai_analyzing",
                    static_findings_count=static_count,
                )
                await session.commit()

            # --- Stage 6: AI Debugging (local AI preferred) ---
            # AIDebuggingError is raised on infrastructure failure (not empty results).
            try:
                debugging_session = await self._run_ai_debugging(
                    project_id,
                    analysis.id,
                    clone_dir,
                    candidate.full_name,
                    autonomous_run_id,
                )
            except AIDebuggingError as exc:
                logger.error(
                    "AI debugging infrastructure failure for run %s: %s",
                    autonomous_run_id,
                    exc,
                )
                await self._fail_run(autonomous_run_id, "ai_debugging_failed", str(exc))
                return

            hypotheses_count = 0
            if debugging_session is not None:
                hypotheses_count = await self._count_hypotheses(getattr(debugging_session, "id"))

            async with async_session_factory() as session:
                run_repo = AutonomousAnalysisRunRepository(session)
                await run_repo.update_status(
                    autonomous_run_id,
                    "finding_validation",
                    current_stage="finding_validation",
                    ai_hypotheses_count=hypotheses_count,
                )
                await session.commit()

            # --- Stage 7: Count high-confidence hypotheses ---
            # NOTE: These are AI hypotheses above a confidence threshold, NOT
            # reproduction-confirmed findings. They are recorded as
            # validated_findings_count for aggregate tracking but must NOT be
            # treated as reproduction evidence without further pipeline stages
            # (reproduction → repair → verification).
            high_conf, low_conf = await self._count_hypothesis_tiers(
                project_id, analysis.id, debugging_session, autonomous_run_id
            )

            # --- Stage 8: Mark complete ---
            async with async_session_factory() as session:
                from app.repositories.discovery_repo import RepositoryCandidateRepository

                run_repo = AutonomousAnalysisRunRepository(session)
                await run_repo.update_status(
                    autonomous_run_id,
                    "completed",
                    current_stage=None,
                    validated_findings_count=high_conf,
                    rejected_findings_count=low_conf,
                )
                cand_repo = RepositoryCandidateRepository(session)
                await cand_repo.mark_analyzed(
                    candidate.id,
                    commit_sha=commit_sha,
                    analysis_status="completed",
                )
                await session.commit()

            duration = time.monotonic() - start
            logger.info(
                "Autonomous run %s completed in %.1fs — %d static, %d AI hypotheses, "
                "%d high-confidence",
                autonomous_run_id,
                duration,
                static_count,
                hypotheses_count,
                high_conf,
            )

        except asyncio.CancelledError:
            logger.warning("Autonomous run %s was cancelled.", autonomous_run_id)
            await self._fail_run(autonomous_run_id, "cancelled", "Task cancelled")
            raise

        except Exception as exc:
            logger.error("Autonomous run %s failed: %s", autonomous_run_id, exc, exc_info=True)
            await self._fail_run(autonomous_run_id, "pipeline_error", str(exc))

        finally:
            # Clean up failed/cancelled workspaces so disk is not silently
            # exhausted by aborted runs.  Successful workspaces are kept so
            # the project record continues to point to a valid local path.
            if workspace_dir is not None and workspace_dir.exists():
                run_succeeded = False
                try:
                    from app.database import async_session_factory as _asf
                    from app.repositories.discovery_repo import (
                        AutonomousAnalysisRunRepository as _Repo,
                    )

                    async with _asf() as _session:
                        _run = await _Repo(_session).get_by_id(autonomous_run_id)
                        run_succeeded = _run is not None and _run.status == "completed"
                except Exception:
                    pass  # can't determine; clean up to be safe

                if not run_succeeded:
                    await self._cleanup_workspace(workspace_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _workspace_root(self) -> Path:
        cfg = self._settings.autonomous_workspace_dir
        if cfg:
            return Path(cfg)
        return Path.home() / ".bugforge" / "workspaces"

    def _candidate_workspace(self, candidate: RepositoryCandidate) -> Path:
        safe_name = _SAFE_NAME_RE.sub("_", candidate.full_name)
        return self._workspace_root() / safe_name

    async def _shallow_clone(self, candidate: RepositoryCandidate) -> tuple[Path, Path, str]:
        """Clone the repository into the permanent workspace directory.

        Returns (workspace_dir, repo_dir, commit_sha).

        The caller owns cleanup on failure.
        Enforces URL safety and clone timeout limits.
        """
        workspace_dir = self._candidate_workspace(candidate)
        repo_dir = workspace_dir / "repo"

        # Reject non-HTTPS and credential-embedded URLs.
        url = candidate.clone_url
        if not url.startswith("https://github.com/"):
            raise ValueError(f"Unexpected clone URL scheme: {url!r}")

        # Remove a partially-complete workspace from a previous failed attempt.
        if workspace_dir.exists():
            await asyncio.to_thread(shutil.rmtree, str(workspace_dir), True)

        await asyncio.to_thread(workspace_dir.mkdir, parents=True, exist_ok=True)

        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            "--branch",
            candidate.default_branch,
            "--",
            url,
            str(repo_dir),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp"},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        except TimeoutError as exc:
            raise TimeoutError(f"Git clone timed out for {url}") from exc

        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed (rc={proc.returncode}): {stderr.decode()[:500]}")

        sha_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo_dir),
            "rev-parse",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        sha_out, _ = await sha_proc.communicate()
        commit_sha = sha_out.decode().strip()[:40] or "unknown"

        return workspace_dir, repo_dir, commit_sha

    async def _ensure_project(self, candidate: RepositoryCandidate, repo_dir: Path) -> UUID:
        """Find or create a BugForge Project for this candidate.

        The project's repository_path is set to the permanent workspace
        directory so that it remains valid after this run completes.
        """
        from sqlalchemy import select

        from app.database import async_session_factory
        from app.models.project import Project

        async with async_session_factory() as session:
            result = await session.execute(
                select(Project).where(Project.name == candidate.full_name)
            )
            project = result.scalar_one_or_none()
            if project is not None:
                project.repository_path = str(repo_dir)
                await session.commit()
                return project.id

            project = Project(
                name=candidate.full_name,
                description=f"Auto-discovered: {candidate.html_url}",
                repository_path=str(repo_dir),
            )
            session.add(project)
            await session.flush()
            await session.refresh(project)
            project_id = project.id
            await session.commit()
            return project_id

    async def _count_static_findings(self, analysis_id: UUID) -> int:
        from sqlalchemy import func, select

        from app.database import async_session_factory
        from app.models.finding import DBFinding

        async with async_session_factory() as session:
            result = await session.execute(
                select(func.count()).where(DBFinding.analysis_id == analysis_id)
            )
            return result.scalar_one()

    async def _run_ai_debugging(
        self,
        project_id: UUID,
        analysis_id: UUID,
        clone_dir: Path,
        repo_name: str,
        run_id: UUID,
    ) -> object:
        """Run DebuggingService on the analysis.

        Returns the debugging session (may have 0 hypotheses — valid empty
        result). Raises AIDebuggingError on infrastructure/provider failure so
        the caller can distinguish "no hypotheses found" from "AI broke".
        """
        from app.database import async_session_factory
        from app.services.debugging_service import DebuggingService

        try:
            async with async_session_factory() as session:
                svc = DebuggingService()
                debugging_session = await svc.start_session(project_id, analysis_id, None)
                session_id = getattr(debugging_session, "id")
                await session.commit()
        except Exception as exc:
            raise AIDebuggingError(
                f"Failed to start AI debugging session for run {run_id}: {exc}"
            ) from exc

        try:
            async with async_session_factory() as _session:
                svc2 = DebuggingService()
                await svc2.run_session(session_id, str(clone_dir), repo_name)
        except Exception as exc:
            raise AIDebuggingError(
                f"AI debugging provider call failed for run {run_id}: {exc}"
            ) from exc

        return debugging_session

    async def _count_hypotheses(self, session_id: UUID) -> int:
        from sqlalchemy import func, select

        from app.database import async_session_factory
        from app.models.debugging import DebuggingHypothesis

        async with async_session_factory() as session:
            result = await session.execute(
                select(func.count()).where(DebuggingHypothesis.session_id == session_id)
            )
            return result.scalar_one()

    async def _count_hypothesis_tiers(
        self,
        project_id: UUID,
        analysis_id: UUID,
        debugging_session: object,
        run_id: UUID,
    ) -> tuple[int, int]:
        """Return (high_confidence_count, low_confidence_count).

        These are AI hypotheses by confidence bucket, NOT reproduction-confirmed
        findings. Higher pipeline stages must run before hypotheses are confirmed.
        """
        if debugging_session is None:
            return 0, 0

        session_id = getattr(debugging_session, "id", None)
        if session_id is None:
            return 0, 0

        from sqlalchemy import func, select

        from app.database import async_session_factory
        from app.models.debugging import DebuggingHypothesis

        async with async_session_factory() as session:
            high_result = await session.execute(
                select(func.count()).where(
                    DebuggingHypothesis.session_id == session_id,
                    DebuggingHypothesis.confidence >= 0.6,
                )
            )
            high: int = high_result.scalar_one()
            low_result = await session.execute(
                select(func.count()).where(
                    DebuggingHypothesis.session_id == session_id,
                    DebuggingHypothesis.confidence < 0.6,
                )
            )
            low: int = low_result.scalar_one()
            return high, low

    async def _fail_run(self, run_id: UUID, error_code: str, error_message: str) -> None:
        from app.database import async_session_factory
        from app.repositories.discovery_repo import AutonomousAnalysisRunRepository

        try:
            async with async_session_factory() as session:
                run_repo = AutonomousAnalysisRunRepository(session)
                await run_repo.update_status(
                    run_id,
                    "failed",
                    current_stage=None,
                    error_code=error_code,
                    error_message=error_message[:500],
                )
                await session.commit()
        except Exception as exc:
            logger.error("Could not persist failure for run %s: %s", run_id, exc)

    async def _cleanup_workspace(self, workspace_dir: Path) -> None:
        try:
            await asyncio.to_thread(shutil.rmtree, str(workspace_dir), True)
            logger.debug("Cleaned up workspace: %s", workspace_dir)
        except Exception as exc:
            logger.warning("Failed to clean up workspace %s: %s", workspace_dir, exc)

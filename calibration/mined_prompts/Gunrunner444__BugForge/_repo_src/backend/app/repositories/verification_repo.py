"""Repository for PatchVerification records — Phase 8."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.verification import PatchVerification


class VerificationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        candidate_id: UUID,
        session_id: UUID,
        project_id: UUID,
    ) -> PatchVerification:
        v = PatchVerification(
            candidate_id=candidate_id,
            session_id=session_id,
            project_id=project_id,
            status="pending",
        )
        self.session.add(v)
        await self.session.flush()
        await self.session.refresh(v)
        return v

    async def get_by_id(self, verification_id: UUID) -> PatchVerification | None:
        result = await self.session.execute(
            select(PatchVerification).where(PatchVerification.id == verification_id)
        )
        return result.scalar_one_or_none()

    async def get_by_candidate(self, candidate_id: UUID) -> PatchVerification | None:
        result = await self.session.execute(
            select(PatchVerification).where(PatchVerification.candidate_id == candidate_id)
        )
        return result.scalar_one_or_none()

    async def list_for_session(self, session_id: UUID) -> list[PatchVerification]:
        result = await self.session.execute(
            select(PatchVerification)
            .where(PatchVerification.session_id == session_id)
            .order_by(PatchVerification.created_at.asc())
        )
        return list(result.scalars().all())

    async def update(
        self,
        verification_id: UUID,
        **kwargs: object,
    ) -> None:
        v = await self.get_by_id(verification_id)
        if v is None:
            return
        for key, value in kwargs.items():
            setattr(v, key, value)
        await self.session.flush()

    async def set_baseline(
        self,
        verification_id: UUID,
        *,
        reproduced: bool | None,
        reproduction_evidence: str,
        tests: dict[str, Any],
        static_findings: int,
        test_execution_status: str = "not_run",
        static_analysis_status: str = "not_run",
        finding_ids: list[str] | None = None,
        duration_seconds: float | None = None,
        executor_type: str | None = None,
    ) -> None:
        await self.update(
            verification_id,
            baseline_reproduced=reproduced,
            baseline_reproduction_evidence=reproduction_evidence[:4096]
            if reproduction_evidence
            else None,
            baseline_tests_total=tests.get("total", 0),
            baseline_tests_passed=tests.get("passed", 0),
            baseline_tests_failed=tests.get("failed", 0),
            baseline_tests_error=tests.get("error", 0),
            baseline_tests_skipped=tests.get("skipped", 0),
            baseline_passing_ids_json=json.dumps(tests.get("passing_ids", [])),
            baseline_failing_ids_json=json.dumps(tests.get("failing_ids", [])),
            baseline_static_findings=static_findings,
            baseline_test_execution_status=test_execution_status,
            baseline_static_analysis_status=static_analysis_status,
            baseline_finding_ids_json=json.dumps(finding_ids) if finding_ids is not None else None,
            baseline_duration_seconds=duration_seconds,
            executor_type=executor_type,
        )

    async def set_post_patch(
        self,
        verification_id: UUID,
        *,
        reproduced: bool | None,
        reproduction_evidence: str,
        tests: dict[str, Any],
        static_findings: int,
        test_execution_status: str = "not_run",
        static_analysis_status: str = "not_run",
        finding_ids: list[str] | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        await self.update(
            verification_id,
            post_patch_reproduced=reproduced,
            post_patch_reproduction_evidence=reproduction_evidence[:4096]
            if reproduction_evidence
            else None,
            post_tests_total=tests.get("total", 0),
            post_tests_passed=tests.get("passed", 0),
            post_tests_failed=tests.get("failed", 0),
            post_tests_error=tests.get("error", 0),
            post_tests_skipped=tests.get("skipped", 0),
            post_passing_ids_json=json.dumps(tests.get("passing_ids", [])),
            post_failing_ids_json=json.dumps(tests.get("failing_ids", [])),
            post_static_findings=static_findings,
            post_test_execution_status=test_execution_status,
            post_static_analysis_status=static_analysis_status,
            post_finding_ids_json=json.dumps(finding_ids) if finding_ids is not None else None,
            post_duration_seconds=duration_seconds,
        )

    async def set_comparison(
        self,
        verification_id: UUID,
        *,
        target_bug_fixed: bool | None,
        newly_failing: list[str],
        recovered: list[str],
        regression_count: int,
        new_static_introduced: int,
        static_resolved: int,
        new_finding_ids: list[str] | None = None,
        resolved_finding_ids: list[str] | None = None,
    ) -> None:
        await self.update(
            verification_id,
            target_bug_fixed=target_bug_fixed,
            newly_failing_ids_json=json.dumps(newly_failing),
            recovered_ids_json=json.dumps(recovered),
            regression_count=regression_count,
            new_static_introduced=new_static_introduced,
            static_resolved=static_resolved,
            new_finding_ids_json=json.dumps(new_finding_ids)
            if new_finding_ids is not None
            else None,
            resolved_finding_ids_json=json.dumps(resolved_finding_ids)
            if resolved_finding_ids is not None
            else None,
        )

    async def set_security(
        self,
        verification_id: UUID,
        *,
        passed: bool,
        issues: list[str],
    ) -> None:
        await self.update(
            verification_id,
            security_passed=passed,
            security_issues_json=json.dumps(issues),
        )

    async def set_decision(
        self,
        verification_id: UUID,
        *,
        score: float,
        decision: str,
        reasons: list[str],
        evidence_summary: str,
        completed_at: datetime,
    ) -> None:
        await self.update(
            verification_id,
            verification_score=round(score, 3),
            verification_decision=decision,
            decision_reasons_json=json.dumps(reasons),
            evidence_summary=evidence_summary[:8192] if evidence_summary else None,
            status=decision,
            completed_at=completed_at,
        )

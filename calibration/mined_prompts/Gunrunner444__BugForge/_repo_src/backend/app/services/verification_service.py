"""Patch Verification service — Phase 8.

Evidence-first patch verification pipeline:
  1. Load context (candidate, session, evidence)
  2. Build clean baseline (pre-patch workspace)
  3. Record baseline reproduction + test suite + static analysis
  4. Apply patch to workspace
  5. Record post-patch reproduction + test suite + static analysis
  6. Compare before / after (by identity, not only count)
  7. Run security checks
  8. Score and decide — ALL required stages must succeed
  9. Persist full evidence record
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

_OUTPUT_LIMIT = 4_096
_WORKSPACE_CONTAINER_PATH = "/bugforge-workspace"


# ── Structured result types ───────────────────────────────────────────────────


@dataclass
class _TestRunResult:
    """Result of running the project test suite in an isolated workspace."""

    # success | timeout | environment_error | report_error | not_run
    execution_status: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0
    passing_ids: list[str] = field(default_factory=list)
    failing_ids: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    executor_type: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "error": self.error,
            "skipped": self.skipped,
            "passing_ids": self.passing_ids,
            "failing_ids": self.failing_ids,
        }

    @property
    def succeeded(self) -> bool:
        return self.execution_status == "success"


@dataclass
class _StaticAnalysisResult:
    """Result of running static analysis on a workspace."""

    # success | error | not_run
    status: str
    count: int = 0
    finding_ids: list[str] = field(default_factory=list)
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


# ── Service ───────────────────────────────────────────────────────────────────


class VerificationService:
    """Runs the complete v0.8 patch verification pipeline."""

    async def verify_candidate(
        self,
        candidate_id: UUID,
    ) -> UUID:
        """Start verification for a patch candidate; return the verification record ID."""
        from app.database import async_session_factory
        from app.repositories.repair_repo import RepairRepository
        from app.repositories.verification_repo import VerificationRepository

        async with async_session_factory() as db:
            repair_repo = RepairRepository(db)
            candidate = await repair_repo.get_candidate(candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} not found")

            if candidate.validation_status not in (None, "valid"):
                raise ValueError(
                    f"Candidate {candidate_id} has validation_status={candidate.validation_status!r}; "
                    "only valid candidates can be verified"
                )

            session_id = candidate.session_id
            repair_session = await repair_repo.get_session_by_id(session_id)
            if repair_session is None:
                raise ValueError(f"Repair session {session_id} not found")

            project_id = repair_session.project_id

            ver_repo = VerificationRepository(db)
            existing = await ver_repo.get_by_candidate(candidate_id)
            if existing is not None:
                return existing.id

            v = await ver_repo.create(
                candidate_id=candidate_id,
                session_id=session_id,
                project_id=project_id,
            )
            verification_id = v.id
            await db.commit()

        await self._run_pipeline(verification_id, candidate_id)
        return verification_id

    async def _run_pipeline(self, verification_id: UUID, candidate_id: UUID) -> None:
        from app.database import async_session_factory
        from app.repositories.verification_repo import VerificationRepository

        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.update(verification_id, status="running", started_at=datetime.now(UTC))
            await db.commit()

        try:
            context = await self._load_context(candidate_id)
            repository_path = context["repository_path"]
            patch_diff = context["patch_diff"]
            changed_files = context["changed_files"]
            reproducer_code = context["reproducer_code"]
            expected_failure_pattern = context["expected_failure_pattern"]

            from app.testing.repair_workspace import RepairWorkspace

            workspace = RepairWorkspace.create(repository_path)
            try:
                await self._run_in_workspace(
                    verification_id=verification_id,
                    workspace=workspace,
                    patch_diff=patch_diff,
                    changed_files=changed_files,
                    reproducer_code=reproducer_code,
                    expected_failure_pattern=expected_failure_pattern,
                )
            finally:
                workspace.destroy()

        except Exception as exc:
            logger.exception("Verification %s failed: %s", verification_id, exc)
            async with async_session_factory() as db:
                ver_repo = VerificationRepository(db)
                await ver_repo.update(
                    verification_id,
                    status="environment_failed",
                    error_message=str(exc),
                    completed_at=datetime.now(UTC),
                )
                await db.commit()

    async def _run_in_workspace(
        self,
        verification_id: UUID,
        workspace: Any,
        patch_diff: str,
        changed_files: list[str],
        reproducer_code: str | None,
        expected_failure_pattern: str | None,
    ) -> None:
        from app.database import async_session_factory
        from app.repositories.verification_repo import VerificationRepository
        from app.testing.patch_validator import validate_patch

        # ── Security check ────────────────────────────────────────────
        validation = validate_patch(patch_diff, changed_files)
        security_passed = validation.valid
        security_issues: list[str] = (
            [] if security_passed else [validation.error or "Validation failed"]
        )
        extra_issues = _security_check(patch_diff, changed_files)
        if extra_issues:
            security_passed = False
            security_issues.extend(extra_issues)

        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.set_security(
                verification_id, passed=security_passed, issues=security_issues
            )
            if not security_passed:
                await ver_repo.update(
                    verification_id,
                    status="rejected",
                    error_message="Security validation failed: " + "; ".join(security_issues),
                    completed_at=datetime.now(UTC),
                )
                await db.commit()
                return
            await ver_repo.update(verification_id, status="running")
            await db.commit()

        # ── Baseline: pre-patch state ─────────────────────────────────
        baseline_repro, baseline_evidence = await self._run_reproducer(
            reproducer_code, workspace.repo_root, expected_failure_pattern
        )
        baseline_tests = await self._run_tests(workspace.repo_root)
        baseline_static = _run_static_analysis(workspace.repo_root, changed_files)

        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.set_baseline(
                verification_id,
                reproduced=baseline_repro,
                reproduction_evidence=baseline_evidence,
                tests=baseline_tests.as_dict(),
                static_findings=baseline_static.count,
                test_execution_status=baseline_tests.execution_status,
                static_analysis_status=baseline_static.status,
                finding_ids=baseline_static.finding_ids,
                duration_seconds=baseline_tests.duration_seconds,
                executor_type=baseline_tests.executor_type,
            )
            await ver_repo.update(verification_id, status="applying")
            await db.commit()

        # ── Apply patch ───────────────────────────────────────────────
        ok, apply_err = workspace.apply_patch(patch_diff)
        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.update(
                verification_id,
                patch_applied=ok,
                patch_apply_error=apply_err if not ok else None,
            )
            if not ok:
                await ver_repo.update(
                    verification_id,
                    status="rejected",
                    error_message=f"Patch application failed: {apply_err}",
                    completed_at=datetime.now(UTC),
                )
                await db.commit()
                return
            await ver_repo.update(verification_id, status="testing")
            await db.commit()

        # ── Post-patch state ──────────────────────────────────────────
        post_repro, post_evidence = await self._run_reproducer(
            reproducer_code, workspace.repo_root, expected_failure_pattern
        )
        post_tests = await self._run_tests(workspace.repo_root)
        post_static = _run_static_analysis(workspace.repo_root, changed_files)

        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.set_post_patch(
                verification_id,
                reproduced=post_repro,
                reproduction_evidence=post_evidence,
                tests=post_tests.as_dict(),
                static_findings=post_static.count,
                test_execution_status=post_tests.execution_status,
                static_analysis_status=post_static.status,
                finding_ids=post_static.finding_ids,
                duration_seconds=post_tests.duration_seconds,
            )
            await ver_repo.update(verification_id, status="comparing")
            await db.commit()

        # ── Compare ───────────────────────────────────────────────────
        target_bug_fixed = _determine_bug_fixed(baseline_repro, post_repro)
        newly_failing, recovered, regression_count = _compare_tests(
            baseline_tests.as_dict(), post_tests.as_dict()
        )
        new_static = max(0, post_static.count - baseline_static.count)
        resolved_static = max(0, baseline_static.count - post_static.count)
        new_finding_ids, resolved_finding_ids = _compare_findings(
            baseline_static.finding_ids, post_static.finding_ids
        )

        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.set_comparison(
                verification_id,
                target_bug_fixed=target_bug_fixed,
                newly_failing=newly_failing,
                recovered=recovered,
                regression_count=regression_count,
                new_static_introduced=new_static,
                static_resolved=resolved_static,
                new_finding_ids=new_finding_ids,
                resolved_finding_ids=resolved_finding_ids,
            )
            await db.commit()

        # ── Score and decide ──────────────────────────────────────────
        score, decision, reasons = _compute_verification(
            baseline_repro=baseline_repro,
            target_bug_fixed=target_bug_fixed,
            regression_count=regression_count,
            new_static_introduced=new_static,
            security_passed=security_passed,
            post_repro=post_repro,
            baseline_test_exec_ok=baseline_tests.succeeded,
            post_test_exec_ok=post_tests.succeeded,
            baseline_static_ok=baseline_static.succeeded,
            post_static_ok=post_static.succeeded,
        )
        evidence_summary = _build_evidence_summary(
            baseline_repro=baseline_repro,
            post_repro=post_repro,
            target_bug_fixed=target_bug_fixed,
            baseline_tests=baseline_tests.as_dict(),
            post_tests=post_tests.as_dict(),
            newly_failing=newly_failing,
            regression_count=regression_count,
            new_static_introduced=new_static,
            static_resolved=resolved_static,
            security_passed=security_passed,
            security_issues=security_issues,
            decision=decision,
            reasons=reasons,
            baseline_test_exec_status=baseline_tests.execution_status,
            post_test_exec_status=post_tests.execution_status,
            baseline_static_status=baseline_static.status,
            post_static_status=post_static.status,
        )

        async with async_session_factory() as db:
            ver_repo = VerificationRepository(db)
            await ver_repo.set_decision(
                verification_id,
                score=score,
                decision=decision,
                reasons=reasons,
                evidence_summary=evidence_summary,
                completed_at=datetime.now(UTC),
            )
            await db.commit()

    async def _load_context(self, candidate_id: UUID) -> dict[str, Any]:
        from app.database import async_session_factory
        from app.models.project import Project
        from app.models.repair import PatchCandidate, RepairSession
        from app.models.reproduction import BugReproductionSession

        async with async_session_factory() as db:
            candidate = await db.get(PatchCandidate, candidate_id)
            if candidate is None:
                raise ValueError(f"Candidate {candidate_id} not found")

            repair_session = await db.get(RepairSession, candidate.session_id)
            project = await db.get(Project, repair_session.project_id) if repair_session else None

            patch_diff = candidate.patch_diff or ""
            changed_files: list[str] = []
            if candidate.changed_files_json:
                try:
                    changed_files = json.loads(candidate.changed_files_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            reproducer_code: str | None = None
            expected_failure_pattern: str | None = None

            if repair_session and repair_session.reproduction_session_id:
                repro = await db.get(BugReproductionSession, repair_session.reproduction_session_id)
                if repro:
                    expected_failure_pattern = repro.expected_failure_pattern
                    if repro.attempts:
                        for attempt in repro.attempts:
                            if attempt.reproduced and attempt.reproducer_code:
                                reproducer_code = attempt.reproducer_code
                                break

            return {
                "repository_path": project.repository_path if project else "",
                "patch_diff": patch_diff,
                "changed_files": changed_files,
                "reproducer_code": reproducer_code,
                "expected_failure_pattern": expected_failure_pattern,
            }

    async def _run_reproducer(
        self,
        code: str | None,
        workspace_repo: str,
        expected_failure_pattern: str | None,
    ) -> tuple[bool | None, str]:
        if not code:
            return None, "No reproducer available"

        from app.execution import ExecutorFactory
        from app.execution.base import ExecutionConfig
        from app.testing.test_validator import validate_test_code

        try:
            with tempfile.TemporaryDirectory(prefix="bugforge_vfy_") as tmpdir:
                test_file = Path(tmpdir) / "test_reproducer.py"
                test_file.write_text(code)

                validation = validate_test_code(code)
                if not validation.valid:
                    return None, f"Reproducer failed validation: {validation.error}"

                executor = ExecutorFactory.create()
                config = ExecutionConfig(
                    command=[
                        "python3",
                        "-m",
                        "pytest",
                        "test_reproducer.py",
                        "-v",
                        "--tb=short",
                        "-q",
                    ],
                    working_directory=tmpdir,
                    timeout_seconds=60,
                    output_dir=tmpdir,
                    environment={"PYTHONPATH": _WORKSPACE_CONTAINER_PATH},
                    read_only_volumes={workspace_repo: _WORKSPACE_CONTAINER_PATH},
                )
                result = await executor.execute(config)
                evidence = (result.stdout + result.stderr)[:_OUTPUT_LIMIT]
                reproduced = _classify_reproduction(
                    result.exit_code, evidence, expected_failure_pattern
                )
                return reproduced, evidence
        except Exception as exc:
            return None, f"Reproducer execution error: {exc}"

    async def _run_tests(self, workspace_repo: str) -> _TestRunResult:
        """Run the project test suite; always returns an explicit execution status."""
        from app.execution import ExecutorFactory
        from app.execution.base import ExecutionConfig

        try:
            executor = ExecutorFactory.create()
            executor_type = type(executor).__name__.replace("TestExecutor", "").lower()

            with tempfile.TemporaryDirectory(prefix="bugforge_vts_") as tmpdir:
                config = ExecutionConfig(
                    command=[
                        "python3",
                        "-m",
                        "pytest",
                        "/bugforge-workspace",
                        "--tb=no",
                        "-q",
                        "--json-report",
                        "--json-report-file=/bugforge-output/report.json",
                    ],
                    working_directory=tmpdir,
                    timeout_seconds=120,
                    output_dir=tmpdir,
                    environment={"PYTHONPATH": _WORKSPACE_CONTAINER_PATH},
                    read_only_volumes={workspace_repo: _WORKSPACE_CONTAINER_PATH},
                )
                exec_result = await executor.execute(config)

                if exec_result.timed_out:
                    return _TestRunResult(
                        execution_status="timeout",
                        executor_type=executor_type,
                        duration_seconds=exec_result.duration_seconds,
                    )

                report_json = exec_result.artifact_contents.get("report.json", "")
                if not report_json:
                    return _TestRunResult(
                        execution_status="report_error",
                        executor_type=executor_type,
                        duration_seconds=exec_result.duration_seconds,
                    )

                try:
                    report = json.loads(report_json)
                    summary = report.get("summary", {})
                    passing_ids: list[str] = []
                    failing_ids: list[str] = []
                    for t in report.get("tests", []):
                        nid = t.get("nodeid", "")
                        outcome = t.get("outcome", "")
                        if outcome == "passed":
                            passing_ids.append(nid)
                        elif outcome in ("failed", "error"):
                            failing_ids.append(nid)
                    return _TestRunResult(
                        execution_status="success",
                        total=summary.get("total", 0),
                        passed=summary.get("passed", 0),
                        failed=summary.get("failed", 0),
                        error=summary.get("error", 0),
                        skipped=summary.get("skipped", 0),
                        passing_ids=passing_ids,
                        failing_ids=failing_ids,
                        duration_seconds=exec_result.duration_seconds,
                        executor_type=executor_type,
                    )
                except (json.JSONDecodeError, KeyError):
                    return _TestRunResult(
                        execution_status="report_error",
                        executor_type=executor_type,
                        duration_seconds=exec_result.duration_seconds,
                    )
        except Exception as exc:
            logger.warning("Test run failed: %s", exc)
            return _TestRunResult(execution_status="environment_error")


# ── Module-level pure helpers ─────────────────────────────────────────────────


def _classify_reproduction(
    exit_code: int, output: str, expected_failure_pattern: str | None
) -> bool | None:
    """Return True=bug present, False=bug absent, None=inconclusive."""
    if exit_code < 0:
        return None
    if exit_code == 0:
        return False
    if exit_code != 1:
        return None
    if expected_failure_pattern:
        if expected_failure_pattern.lower() not in output.lower():
            return None
    return True


def _run_static_analysis(workspace_repo: str, changed_files: list[str]) -> _StaticAnalysisResult:
    """Run static analysis; always returns explicit status, never silently returns 0 on failure."""
    from app.analysis.engine import StaticAnalysisEngine

    try:
        repo_path = Path(workspace_repo)
        file_paths = [
            repo_path / f for f in changed_files if (repo_path / f).is_file() and f.endswith(".py")
        ]
        if not file_paths:
            return _StaticAnalysisResult(status="success", count=0, finding_ids=[])

        engine = StaticAnalysisEngine()
        findings = engine.analyze_repository(repo_path, file_paths)
        # Normalized identity stable across line-number shifts
        finding_ids = [f"{f.analyzer}:{f.category}:{f.file_path}" for f in findings]
        return _StaticAnalysisResult(
            status="success",
            count=len(findings),
            finding_ids=finding_ids,
        )
    except Exception as exc:
        logger.warning("Static analysis failed: %s", exc)
        return _StaticAnalysisResult(
            status="error",
            count=0,
            finding_ids=[],
            error_message=str(exc),
        )


def _compare_findings(baseline_ids: list[str], post_ids: list[str]) -> tuple[list[str], list[str]]:
    """Return (new_finding_ids, resolved_finding_ids) using set difference."""
    baseline_set = set(baseline_ids)
    post_set = set(post_ids)
    new_ids = sorted(post_set - baseline_set)
    resolved_ids = sorted(baseline_set - post_set)
    return new_ids, resolved_ids


def _compare_tests(
    baseline: dict[str, Any], post: dict[str, Any]
) -> tuple[list[str], list[str], int]:
    """Return (newly_failing, recovered, regression_count)."""
    baseline_passing = set(baseline.get("passing_ids", []))
    baseline_failing = set(baseline.get("failing_ids", []))
    post_failing = set(post.get("failing_ids", []))
    post_passing = set(post.get("passing_ids", []))

    newly_failing = sorted(baseline_passing & post_failing)
    recovered = sorted(baseline_failing & post_passing)
    return newly_failing, recovered, len(newly_failing)


def _determine_bug_fixed(baseline_repro: bool | None, post_repro: bool | None) -> bool | None:
    if baseline_repro is True and post_repro is False:
        return True
    if baseline_repro is True and post_repro is True:
        return False
    return None


_DANGEROUS_PATTERNS = re.compile(
    r"\b(os\.system|subprocess\.|eval\s*\(|exec\s*\(|__import__\s*\()",
    re.MULTILINE,
)

_CI_CONFIG_PATTERN = re.compile(
    r"^\+\+\+\s+b/(\.github/|\.circleci/|\.travis\.yml|Jenkinsfile|Makefile)",
    re.MULTILINE,
)


def _security_check(patch_diff: str, changed_files: list[str]) -> list[str]:
    """Additional security checks beyond PatchValidator."""
    issues: list[str] = []
    if _CI_CONFIG_PATTERN.search(patch_diff):
        issues.append("Patch modifies CI/security configuration files")
    added_lines = "\n".join(
        line
        for line in patch_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    if _DANGEROUS_PATTERNS.search(added_lines):
        issues.append("Patch adds potentially dangerous code patterns")
    return issues


def _compute_verification(
    baseline_repro: bool | None,
    target_bug_fixed: bool | None,
    regression_count: int,
    new_static_introduced: int,
    security_passed: bool,
    post_repro: bool | None,
    *,
    baseline_test_exec_ok: bool = True,
    post_test_exec_ok: bool = True,
    baseline_static_ok: bool = True,
    post_static_ok: bool = True,
) -> tuple[float, str, list[str]]:
    """Return (score, decision, reasons).

    A patch CANNOT become 'verified' unless all required verification stages
    completed successfully.  Infrastructure failures return 'inconclusive'.
    """
    reasons: list[str] = []
    score = 0.0

    if not security_passed:
        return 0.0, "rejected", ["Security validation failed"]

    if not baseline_test_exec_ok:
        reasons.append("Baseline test execution failed — cannot verify regressions")
        return 0.0, "inconclusive", reasons

    if not post_test_exec_ok:
        reasons.append("Post-patch test execution failed — cannot verify regressions")
        return 0.0, "inconclusive", reasons

    if not baseline_static_ok:
        reasons.append("Baseline static analysis failed — cannot compare findings")
        return 0.0, "inconclusive", reasons

    if not post_static_ok:
        reasons.append("Post-patch static analysis failed — cannot compare findings")
        return 0.0, "inconclusive", reasons

    if baseline_repro is False:
        reasons.append("Baseline did not reproduce the target bug — baseline invalid")
        return 0.0, "baseline_failed", reasons

    if target_bug_fixed is True:
        score += 0.7
        reasons.append("Target bug no longer reproduces after patch")
    elif target_bug_fixed is False:
        reasons.append("Target bug still reproduces after patch")
    else:
        reasons.append("Bug fix status inconclusive (no reproducer available)")

    if regression_count == 0:
        score += 0.3
        reasons.append("No regression in existing tests")
    else:
        penalty = min(regression_count * 0.1, 0.3)
        score -= penalty
        reasons.append(f"{regression_count} regression(s): newly failing test(s) introduced")

    if new_static_introduced > 0:
        penalty = min(new_static_introduced * 0.05, 0.2)
        score -= penalty
        reasons.append(f"{new_static_introduced} new static finding(s) introduced")

    score = round(max(0.0, min(1.0, score)), 3)

    if target_bug_fixed is True and regression_count == 0 and security_passed:
        decision = "verified"
    elif target_bug_fixed is None:
        decision = "inconclusive"
    else:
        decision = "rejected"

    return score, decision, reasons


def _build_evidence_summary(
    *,
    baseline_repro: bool | None,
    post_repro: bool | None,
    target_bug_fixed: bool | None,
    baseline_tests: dict[str, Any],
    post_tests: dict[str, Any],
    newly_failing: list[str],
    regression_count: int,
    new_static_introduced: int,
    static_resolved: int,
    security_passed: bool,
    security_issues: list[str],
    decision: str,
    reasons: list[str],
    baseline_test_exec_status: str = "success",
    post_test_exec_status: str = "success",
    baseline_static_status: str = "success",
    post_static_status: str = "success",
) -> str:
    lines = [
        "=== PATCH VERIFICATION EVIDENCE ===",
        f"Decision: {decision.upper()}",
        "",
        "--- Baseline (pre-patch) ---",
        f"  Bug reproduced: {baseline_repro}",
        f"  Test execution status: {baseline_test_exec_status}",
        f"  Tests: {baseline_tests.get('total', 0)} total, "
        f"{baseline_tests.get('passed', 0)} passed, "
        f"{baseline_tests.get('failed', 0) + baseline_tests.get('error', 0)} failed",
        f"  Static analysis status: {baseline_static_status}",
        "",
        "--- Post-patch ---",
        f"  Bug reproduced: {post_repro}",
        f"  Test execution status: {post_test_exec_status}",
        f"  Tests: {post_tests.get('total', 0)} total, "
        f"{post_tests.get('passed', 0)} passed, "
        f"{post_tests.get('failed', 0) + post_tests.get('error', 0)} failed",
        f"  Static analysis status: {post_static_status}",
        "",
        "--- Comparison ---",
        f"  Target bug fixed: {target_bug_fixed}",
        f"  Regressions: {regression_count}",
    ]
    if newly_failing:
        lines.append(f"  Newly failing tests: {', '.join(newly_failing[:5])}")
    lines += [
        f"  New static findings: {new_static_introduced}",
        f"  Resolved static findings: {static_resolved}",
        "",
        "--- Security ---",
        f"  Security passed: {security_passed}",
    ]
    if security_issues:
        for issue in security_issues:
            lines.append(f"  - {issue}")
    lines += ["", "--- Reasons ---"]
    for r in reasons:
        lines.append(f"  - {r}")
    return "\n".join(lines)

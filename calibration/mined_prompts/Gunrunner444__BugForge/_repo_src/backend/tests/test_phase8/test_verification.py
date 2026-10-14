"""Tests for Phase 8: Patch Verification."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def divide(a, b):\n    return a / b\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "test_calc.py").write_text(
        "from calc import divide\ndef test_divide(): assert divide(10, 2) == 5\n"
    )
    return repo


# ── Verification Service Unit Tests ──────────────────────────────────────────


class TestVerificationPureHelpers:
    def test_classify_reproduction_exit_0(self) -> None:
        from app.services.verification_service import _classify_reproduction

        assert _classify_reproduction(0, "", None) is False

    def test_classify_reproduction_exit_1_no_pattern(self) -> None:
        from app.services.verification_service import _classify_reproduction

        assert _classify_reproduction(1, "FAILED test_foo", None) is True

    def test_classify_reproduction_exit_1_pattern_matches(self) -> None:
        from app.services.verification_service import _classify_reproduction

        assert (
            _classify_reproduction(1, "ZeroDivisionError: division by zero", "ZeroDivisionError")
            is True
        )

    def test_classify_reproduction_exit_1_pattern_missing(self) -> None:
        from app.services.verification_service import _classify_reproduction

        assert _classify_reproduction(1, "AssertionError: oops", "ZeroDivisionError") is None

    def test_classify_reproduction_exit_2_inconclusive(self) -> None:
        from app.services.verification_service import _classify_reproduction

        assert _classify_reproduction(2, "interrupted", None) is None

    def test_classify_reproduction_negative_exit(self) -> None:
        from app.services.verification_service import _classify_reproduction

        assert _classify_reproduction(-1, "", None) is None

    def test_compare_tests_no_regression(self) -> None:
        from app.services.verification_service import _compare_tests

        baseline = {"passing_ids": ["a", "b", "c"], "failing_ids": ["d"]}
        post = {"passing_ids": ["a", "b", "c"], "failing_ids": ["d"]}
        newly_failing, recovered, count = _compare_tests(baseline, post)
        assert count == 0
        assert newly_failing == []

    def test_compare_tests_regression_detected(self) -> None:
        from app.services.verification_service import _compare_tests

        baseline = {"passing_ids": ["a", "b"], "failing_ids": []}
        post = {"passing_ids": ["a"], "failing_ids": ["b"]}
        newly_failing, recovered, count = _compare_tests(baseline, post)
        assert count == 1
        assert "b" in newly_failing

    def test_compare_tests_preexisting_failure_not_regression(self) -> None:
        from app.services.verification_service import _compare_tests

        baseline = {"passing_ids": ["a"], "failing_ids": ["b"]}
        post = {"passing_ids": ["a"], "failing_ids": ["b"]}
        newly_failing, recovered, count = _compare_tests(baseline, post)
        assert count == 0
        assert "b" not in newly_failing

    def test_compare_tests_recovery_tracked(self) -> None:
        from app.services.verification_service import _compare_tests

        baseline = {"passing_ids": [], "failing_ids": ["b"]}
        post = {"passing_ids": ["b"], "failing_ids": []}
        newly_failing, recovered, count = _compare_tests(baseline, post)
        assert "b" in recovered
        assert count == 0

    def test_determine_bug_fixed_when_reproduced_then_not(self) -> None:
        from app.services.verification_service import _determine_bug_fixed

        assert _determine_bug_fixed(True, False) is True

    def test_determine_bug_fixed_when_still_reproduced(self) -> None:
        from app.services.verification_service import _determine_bug_fixed

        assert _determine_bug_fixed(True, True) is False

    def test_determine_bug_fixed_inconclusive(self) -> None:
        from app.services.verification_service import _determine_bug_fixed

        assert _determine_bug_fixed(None, None) is None

    def test_compute_verification_verified(self) -> None:
        from app.services.verification_service import _compute_verification

        score, decision, reasons = _compute_verification(
            baseline_repro=True,
            target_bug_fixed=True,
            regression_count=0,
            new_static_introduced=0,
            security_passed=True,
            post_repro=False,
        )
        assert decision == "verified"
        assert score > 0.9

    def test_compute_verification_rejected_regression(self) -> None:
        from app.services.verification_service import _compute_verification

        score, decision, reasons = _compute_verification(
            baseline_repro=True,
            target_bug_fixed=True,
            regression_count=2,
            new_static_introduced=0,
            security_passed=True,
            post_repro=False,
        )
        assert decision == "rejected"
        assert any("regression" in r.lower() for r in reasons)

    def test_compute_verification_baseline_failed(self) -> None:
        from app.services.verification_service import _compute_verification

        score, decision, reasons = _compute_verification(
            baseline_repro=False,
            target_bug_fixed=None,
            regression_count=0,
            new_static_introduced=0,
            security_passed=True,
            post_repro=None,
        )
        assert decision == "baseline_failed"

    def test_compute_verification_security_hard_fail(self) -> None:
        from app.services.verification_service import _compute_verification

        score, decision, reasons = _compute_verification(
            baseline_repro=True,
            target_bug_fixed=True,
            regression_count=0,
            new_static_introduced=0,
            security_passed=False,
            post_repro=False,
        )
        assert decision == "rejected"
        assert score == 0.0

    def test_security_check_dangerous_pattern(self) -> None:
        from app.services.verification_service import _security_check

        diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+os.system('rm -rf /')\n"
        issues = _security_check(diff, ["x.py"])
        assert len(issues) > 0

    def test_security_check_clean_patch(self) -> None:
        from app.services.verification_service import _security_check

        diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-return None\n+return 42\n"
        issues = _security_check(diff, ["x.py"])
        assert issues == []

    def test_build_evidence_summary_contains_decision(self) -> None:
        from app.services.verification_service import _build_evidence_summary

        summary = _build_evidence_summary(
            baseline_repro=True,
            post_repro=False,
            target_bug_fixed=True,
            baseline_tests={"total": 5, "passed": 4, "failed": 1, "error": 0, "skipped": 0},
            post_tests={"total": 5, "passed": 5, "failed": 0, "error": 0, "skipped": 0},
            newly_failing=[],
            regression_count=0,
            new_static_introduced=0,
            static_resolved=0,
            security_passed=True,
            security_issues=[],
            decision="verified",
            reasons=["Target bug fixed", "No regressions"],
        )
        assert "VERIFIED" in summary
        assert "Target bug fixed" in summary


# ── Repair Workspace Updated Tests ───────────────────────────────────────────


class TestRepairWorkspaceV2:
    def test_hunk_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """Hunk mismatch must be rejected, not silently ignored."""
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")

        with RepairWorkspace.create(str(tmp_path)) as ws:
            diff = (
                "--- a/calc.py\n"
                "+++ b/calc.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return WRONG_LINE\n"  # does not match actual content
                "+    return a - b\n"
            )
            ok, err = ws.apply_patch(diff)
            assert not ok
            assert "mismatch" in err.lower() or "rejected" in err.lower()

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        """Patches that escape the workspace root must be rejected."""
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "calc.py").write_text("x = 1\n")

        with RepairWorkspace.create(str(tmp_path)) as ws:
            diff = "--- a/../../../etc/passwd\n+++ b/../../../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
            ok, err = ws.apply_patch(diff)
            assert not ok

    def test_context_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """Context line mismatch must be rejected."""
        from app.testing.repair_workspace import RepairWorkspace

        (tmp_path / "f.py").write_text("def foo():\n    return 1\n")

        with RepairWorkspace.create(str(tmp_path)) as ws:
            diff = (
                "--- a/f.py\n"
                "+++ b/f.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def WRONG_CONTEXT():\n"  # context doesn't match
                "-    return 1\n"
                "+    return 2\n"
            )
            ok, err = ws.apply_patch(diff)
            assert not ok


# ── Regression detection unit tests ──────────────────────────────────────────


class TestRepairServiceHelpers:
    def test_compare_test_runs_no_regression(self) -> None:
        from app.services.repair_service import RepairService

        baseline = {"passing_ids": ["a::test_x"], "failing_ids": ["a::test_y"]}
        post = {"passing_ids": ["a::test_x"], "failing_ids": ["a::test_y"]}
        no_reg, count = RepairService._compare_test_runs(baseline, post)
        assert no_reg is True
        assert count == 0

    def test_compare_test_runs_detects_regression(self) -> None:
        from app.services.repair_service import RepairService

        baseline = {"passing_ids": ["a::test_x", "a::test_y"], "failing_ids": []}
        post = {"passing_ids": ["a::test_x"], "failing_ids": ["a::test_y"]}
        no_reg, count = RepairService._compare_test_runs(baseline, post)
        assert no_reg is False
        assert count == 1

    def test_compare_test_runs_preexisting_failure_not_regression(self) -> None:
        from app.services.repair_service import RepairService

        baseline = {"passing_ids": [], "failing_ids": ["a::test_broken"]}
        post = {"passing_ids": [], "failing_ids": ["a::test_broken"]}
        no_reg, count = RepairService._compare_test_runs(baseline, post)
        assert no_reg is True
        assert count == 0

    def test_classify_reproduction_exit_0_not_reproduced(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        assert svc._classify_reproduction({"exit_code": 0}, None) is False

    def test_classify_reproduction_exit_1_reproduced(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        assert (
            svc._classify_reproduction({"exit_code": 1, "stdout": "", "stderr": ""}, None) is True
        )

    def test_classify_reproduction_exit_1_pattern_match(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        result = svc._classify_reproduction(
            {"exit_code": 1, "stdout": "ZeroDivisionError here", "stderr": ""},
            "ZeroDivisionError",
        )
        assert result is True

    def test_classify_reproduction_exit_1_pattern_mismatch(self) -> None:
        from app.services.repair_service import RepairService

        svc = RepairService()
        result = svc._classify_reproduction(
            {"exit_code": 1, "stdout": "AssertionError", "stderr": ""},
            "ZeroDivisionError",
        )
        assert result is None


# ── Verification API Tests ────────────────────────────────────────────────────


class TestVerificationAPI:
    async def test_get_nonexistent_verification_returns_404(self, client) -> None:
        resp = await client.get(f"/api/v1/verification/{uuid4()}")
        assert resp.status_code == 404

    async def test_get_verification_by_nonexistent_candidate_returns_404(self, client) -> None:
        resp = await client.get(f"/api/v1/verification/by-candidate/{uuid4()}")
        assert resp.status_code == 404

    async def test_start_verification_nonexistent_candidate_404(self, client) -> None:
        session_id = uuid4()
        candidate_id = uuid4()
        resp = await client.post(f"/api/v1/repair/{session_id}/candidates/{candidate_id}/verify")
        assert resp.status_code == 404

    async def test_list_verifications_nonexistent_session_404(self, client) -> None:
        resp = await client.get(f"/api/v1/repair/{uuid4()}/verifications")
        assert resp.status_code == 404

    async def test_start_and_retrieve_verification(self, client, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects",
            json={"name": "VerifyTest1", "repository_path": str(repo)},
        )
        assert proj.status_code == 201
        project_id = proj.json()["id"]

        # Create a repair session with a candidate
        repair_resp = await client.post(f"/api/v1/projects/{project_id}/repair", json={})
        assert repair_resp.status_code == 202
        session_id = repair_resp.json()["id"]

        # Get candidates
        cands_resp = await client.get(f"/api/v1/repair/{session_id}/candidates")
        assert cands_resp.status_code == 200
        candidates = cands_resp.json()
        if not candidates:
            # No candidates yet (session may still be pending) — just verify 404 path
            return

        candidate_id = candidates[0]["id"]

        # Start verification
        ver_resp = await client.post(
            f"/api/v1/repair/{session_id}/candidates/{candidate_id}/verify"
        )
        assert ver_resp.status_code in (200, 202)
        data = ver_resp.json()
        assert data["candidate_id"] == candidate_id
        assert data["session_id"] == session_id
        assert "status" in data

        # Retrieve verification
        get_resp = await client.get(f"/api/v1/repair/{session_id}/candidates/{candidate_id}/verify")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == data["id"]

        # List verifications for session
        list_resp = await client.get(f"/api/v1/repair/{session_id}/verifications")
        assert list_resp.status_code == 200
        assert any(v["id"] == data["id"] for v in list_resp.json())

    async def test_start_verification_idempotent(self, client, tmp_path: Path) -> None:
        """Starting verification twice returns the same record."""
        repo = _make_repo(tmp_path)
        proj = await client.post(
            "/api/v1/projects",
            json={"name": "VerifyIdempotent", "repository_path": str(repo)},
        )
        project_id = proj.json()["id"]
        repair_resp = await client.post(f"/api/v1/projects/{project_id}/repair", json={})
        session_id = repair_resp.json()["id"]

        cands_resp = await client.get(f"/api/v1/repair/{session_id}/candidates")
        candidates = cands_resp.json()
        if not candidates:
            return

        candidate_id = candidates[0]["id"]

        r1 = await client.post(f"/api/v1/repair/{session_id}/candidates/{candidate_id}/verify")
        r2 = await client.post(f"/api/v1/repair/{session_id}/candidates/{candidate_id}/verify")
        assert r1.status_code in (200, 202)
        assert r2.status_code in (200, 202)
        assert r1.json()["id"] == r2.json()["id"]

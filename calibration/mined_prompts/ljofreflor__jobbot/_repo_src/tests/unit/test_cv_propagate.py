"""Planning for `jobbot cv propagate` (read-only, browser-free)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.adapters.diff_engine import save_snapshot
from jobbot.browser.sessions import SessionState, SessionStatus
from jobbot.config import JobbotConfig
from jobbot.cv.propagate import (
    DEFAULT_TARGETS,
    PropagationTarget,
    UnknownTargetError,
    parse_targets,
    plan_cv,
    plan_getonboard,
    plan_indeed,
    plan_linkedin,
    plan_propagation,
    summarize_plans,
    with_session,
)
from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from tests.fixtures.profile import sample_profile_dict


def _candidate() -> Candidate:
    return Candidate.model_validate(sample_profile_dict())


def _config(tmp_path: Path, project_root: Path) -> JobbotConfig:
    from jobbot.config import PathsConfig

    return JobbotConfig(
        root=tmp_path,
        paths=PathsConfig(templates=project_root / "templates", output=Path("output")),
    )


def test_parse_targets_defaults_to_all_in_canonical_order() -> None:
    assert parse_targets("all") == list(DEFAULT_TARGETS)
    assert parse_targets("") == list(DEFAULT_TARGETS)
    assert parse_targets("linkedin, cv") == [PropagationTarget.CV, PropagationTarget.LINKEDIN]


def test_permanent_is_the_same_as_all() -> None:
    """`--targets all` means permanent profiles, not every company ATS in the registry."""
    assert parse_targets("permanent") == list(DEFAULT_TARGETS)
    assert parse_targets("permanent") == parse_targets("all")


def test_parse_targets_rejects_unknown_destination() -> None:
    with pytest.raises(UnknownTargetError):
        parse_targets("cv,twitter")


def test_company_portals_are_not_propagation_targets() -> None:
    """Greenhouse/Workday/etc. belong to cv sync / signup, not propagate --targets."""
    for name in ("companies", "greenhouse", "workday", "all-companies", "portals"):
        with pytest.raises(UnknownTargetError, match="cv sync") as exc:
            parse_targets(name)
        text = str(exc.value).casefold()
        assert "cv sync" in text
        assert "permanent" in text


def test_cv_plan_lists_local_artifacts(tmp_path: Path, project_root: Path) -> None:
    plan = plan_cv(_config(tmp_path, project_root), _candidate())
    assert plan.ready and plan.actionable
    assert any("cv.pdf" in op for op in plan.operations)
    assert any("cv_ats.txt" in op for op in plan.operations)
    assert plan.note is not None and "first build" in plan.note


def test_cv_plan_blocks_without_templates(tmp_path: Path) -> None:
    from jobbot.config import PathsConfig

    config = JobbotConfig(root=tmp_path, paths=PathsConfig(templates=Path("missing-templates")))
    plan = plan_cv(config, _candidate())
    assert not plan.ready
    assert plan.blocked_reason is not None and "templates" in plan.blocked_reason


def test_indeed_plan_needs_a_snapshot_instead_of_opening_a_browser(
    tmp_path: Path, project_root: Path
) -> None:
    plan = plan_indeed(_config(tmp_path, project_root), _candidate())
    assert not plan.ready
    assert plan.hint == "jobbot indeed pull"


def test_indeed_plan_diffs_against_stored_snapshot(tmp_path: Path, project_root: Path) -> None:
    config = _config(tmp_path, project_root)
    save_snapshot(
        ExternalProfile(source="indeed", headline="Old headline", summary="Old summary"),
        config.output_dir,
    )
    plan = plan_indeed(config, _candidate(), section="headline")
    assert plan.ready and plan.actionable
    assert all("headline" in op for op in plan.operations)


def test_linkedin_plan_covers_publications_only(tmp_path: Path, project_root: Path) -> None:
    plan = plan_linkedin(_config(tmp_path, project_root), _candidate())
    assert plan.operations == ["add publication — Marketplace interference"]
    assert plan.note is not None and "remote" in plan.note


def test_getonboard_plan_is_cumulative_after_first_run(
    tmp_path: Path, project_root: Path
) -> None:
    from jobbot.adapters.getonboard.draft import save_permanent_profile

    config = _config(tmp_path, project_root)
    first = plan_getonboard(config, _candidate())
    assert any("create" in op for op in first.operations)

    from jobbot.nlp.refine import refine_permanent_profile

    result = refine_permanent_profile(_candidate(), None)
    save_permanent_profile(result.fields, config.output_dir)

    again = plan_getonboard(config, _candidate())
    assert any("cumulative" in op for op in again.operations)


def _session(status: SessionStatus, **kwargs: object) -> SessionState:
    defaults: dict[str, object] = {
        "site": "linkedin",
        "status": status,
        "evidence": "signed-in page open: www.linkedin.com/feed/",
    }
    defaults.update(kwargs)
    return SessionState(**defaults)  # type: ignore[arg-type]


def test_busy_profile_blocks_the_plan_before_a_browser_is_launched(
    tmp_path: Path, project_root: Path
) -> None:
    """Regression F0002: launching over a profile another Chrome holds always fails."""
    busy = _session(
        SessionStatus.PROFILE_BUSY,
        evidence="browser-data/linkedin held by Chrome pid 30237",
        hint="close that Chrome window, or point --cdp at it",
        holders=(30237,),
    )
    plan = with_session(plan_linkedin(_config(tmp_path, project_root), _candidate()), busy)
    assert not plan.ready
    assert plan.blocked_reason is not None and "30237" in plan.blocked_reason
    assert plan.hint is not None and "--cdp" in plan.hint


def test_ready_session_is_suggested_never_used_on_its_own(
    tmp_path: Path, project_root: Path
) -> None:
    ready = _session(
        SessionStatus.READY,
        cdp_url="http://127.0.0.1:9223",
        hint="--cdp http://127.0.0.1:9223",
    )
    plan = with_session(plan_linkedin(_config(tmp_path, project_root), _candidate()), ready)
    assert plan.ready and plan.actionable
    assert plan.hint == "--cdp http://127.0.0.1:9223"
    assert plan.note is not None and "signed-in page open" in plan.note


def test_explicit_cdp_wins_over_the_preflight(tmp_path: Path, project_root: Path) -> None:
    busy = _session(SessionStatus.PROFILE_BUSY, holders=(30237,))
    plan = with_session(
        plan_linkedin(_config(tmp_path, project_root), _candidate()),
        busy,
        cdp_url="http://127.0.0.1:9223",
    )
    assert plan.ready
    assert plan.session is busy


def test_propagation_only_preflights_the_browser_targets(
    tmp_path: Path, project_root: Path
) -> None:
    sessions = [_session(SessionStatus.PROFILE_BUSY, site="indeed", holders=(1,))]
    plans = plan_propagation(_config(tmp_path, project_root), _candidate(), sessions=sessions)
    by_target = {plan.target: plan for plan in plans}
    assert by_target[PropagationTarget.CV].session is None
    assert by_target[PropagationTarget.GETONBOARD].session is None
    assert by_target[PropagationTarget.LINKEDIN].session is None  # no state for linkedin
    assert by_target[PropagationTarget.INDEED].session is not None


def test_summary_reports_counts_and_blocked_targets(tmp_path: Path, project_root: Path) -> None:
    plans = plan_propagation(_config(tmp_path, project_root), _candidate())
    summary = summarize_plans(plans)
    assert "cv: 2" in summary
    assert "blocked: indeed" in summary

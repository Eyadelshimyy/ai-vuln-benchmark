"""Single-writer regression for repository-local review-repair automation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RETIRED_CALLER = WORKFLOWS / "hourly-review-repair.yml"
CENTRAL_REPAIR_CALL = "pr-review-fix-scheduler.yml@"


def test_competing_hourly_review_repair_caller_is_absent() -> None:
    """The dedicated repository writer must not race a second scheduled writer."""
    assert not RETIRED_CALLER.exists()


def test_no_repository_workflow_delegates_to_the_retired_repair_scheduler() -> None:
    """Renaming the retired caller must not silently restore a competing writer."""
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        if CENTRAL_REPAIR_CALL in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []

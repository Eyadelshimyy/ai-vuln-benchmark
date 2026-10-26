"""Diff / SyncPlan unit tests."""

from jobbot.adapters.diff_engine import build_diff_operations, build_sync_plan
from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from jobbot.models.sync import SyncOpType
from tests.fixtures.profile import sample_profile_dict


def test_headline_change_in_sync_plan() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    external = ExternalProfile(source="indeed", headline="Data Scientist")
    plan = build_sync_plan("indeed", candidate, external, section="headline")
    assert plan.actionable
    assert plan.actionable[0].op == SyncOpType.CHANGE
    assert plan.actionable[0].after == candidate.personal.headline


def test_idempotent_when_same_headline() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    external = ExternalProfile(source="indeed", headline=candidate.personal.headline)
    plan = build_sync_plan("indeed", candidate, external, section="headline")
    assert plan.actionable == []


def test_removals_ignored_by_default() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    external = ExternalProfile(
        source="indeed",
        headline=candidate.personal.headline,
        skills=["Python", "COBOL"],
    )
    ops = build_diff_operations(candidate, external, section="skills")
    assert any(o.op == SyncOpType.REMOVE and o.field == "COBOL" for o in ops)
    plan = build_sync_plan("indeed", candidate, external, section="skills")
    assert all(o.op != SyncOpType.REMOVE for o in plan.operations)

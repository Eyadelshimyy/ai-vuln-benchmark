"""Indeed sync package + marketing headline detection tests."""

from __future__ import annotations

from jobbot.adapters.diff_engine import build_sync_plan
from jobbot.adapters.indeed.client import _looks_like_marketing_headline
from jobbot.adapters.indeed.package import build_indeed_sync_package, truncate
from jobbot.models.candidate import Candidate
from jobbot.models.external_profile import ExternalProfile
from jobbot.models.sync import SyncOpType
from tests.fixtures.profile import sample_profile_dict


def test_truncate_respects_max() -> None:
    assert truncate("abcdef", 4) == "abc…"
    assert truncate("ab", 10) == "ab"


def test_build_indeed_sync_package_from_profile() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    package = build_indeed_sync_package(candidate)
    assert package.headline
    assert "Python" in package.skills or package.skills
    assert package.experience_blocks


def test_marketing_headline_rejected() -> None:
    assert _looks_like_marketing_headline("¿Tienes todo listo para dar el siguiente paso?")
    assert not _looks_like_marketing_headline("Senior Data Scientist | MSc Estadística")


def test_experience_add_when_remote_empty() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    external = ExternalProfile(source="indeed", headline=candidate.personal.headline)
    plan = build_sync_plan("indeed", candidate, external, section="experience")
    assert plan.actionable
    assert all(o.op == SyncOpType.ADD for o in plan.actionable)

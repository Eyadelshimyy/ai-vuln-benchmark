"""Tests for profile loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jobbot.models.candidate import Candidate
from jobbot.profile.loader import ProfileLoadError, load_profile
from jobbot.profile.validator import validate_candidate
from tests.fixtures.profile import sample_profile_dict


def test_load_example_profile(project_root: Path) -> None:
    path = project_root / "data" / "profile.example.yaml"
    candidate = load_profile(path)
    assert candidate.personal.name == "Ana Ejemplo"
    assert len(candidate.experience) >= 1
    assert candidate.achievement_count() >= 1
    assert candidate.projects


def test_projects_are_loaded_not_silently_dropped() -> None:
    """A `projects:` key used to be ignored by the model, so it never reached the CV."""
    data = sample_profile_dict()
    data["projects"] = [
        {
            "id": "huerto",
            "name": "Huerto Comunitario",
            "description": "Coordinación de un huerto vecinal.",
            "url": "https://example.org/huerto",
            "tags": ["Compostaje"],
        }
    ]
    candidate = Candidate.model_validate(data)
    assert candidate.projects[0].name == "Huerto Comunitario"
    assert candidate.projects[0].url == "https://example.org/huerto"
    assert candidate.projects[0].tags == ["Compostaje"]


def test_project_url_must_be_http() -> None:
    data = sample_profile_dict()
    data["projects"] = [
        {"id": "p", "name": "P", "description": "D.", "url": "example.org/p"},
    ]
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


def test_duplicate_project_id() -> None:
    data = sample_profile_dict()
    project = {"id": "p", "name": "P", "description": "D."}
    data["projects"] = [project, {**project, "name": "Q"}]
    result = validate_candidate(Candidate.model_validate(data))
    assert not result.ok
    assert any("duplicate id" in i.message for i in result.issues)


def test_validate_ok() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    result = validate_candidate(candidate)
    assert result.ok


def test_duplicate_experience_id() -> None:
    data = sample_profile_dict()
    data["experience"].append({**data["experience"][0], "id": "meli-ds"})
    candidate = Candidate.model_validate(data)
    result = validate_candidate(candidate)
    assert not result.ok
    assert any("duplicate id" in i.message for i in result.issues)


def test_end_before_start_rejected() -> None:
    data = sample_profile_dict()
    data["experience"][0]["start_date"] = "2024-01"
    data["experience"][0]["end_date"] = "2020-01"
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


def test_invalid_email_rejected() -> None:
    data = sample_profile_dict()
    data["personal"]["email"] = "not-an-email"
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


def test_orcid_is_optional() -> None:
    candidate = Candidate.model_validate(sample_profile_dict())
    assert candidate.personal.orcid is None


def test_orcid_with_valid_checksum_is_accepted() -> None:
    data = sample_profile_dict()
    data["personal"]["orcid"] = "0000-0002-1825-0097"
    candidate = Candidate.model_validate(data)
    assert candidate.personal.orcid == "0000-0002-1825-0097"
    assert candidate.personal.orcid_url == "https://orcid.org/0000-0002-1825-0097"


def test_orcid_accepts_x_check_digit() -> None:
    data = sample_profile_dict()
    data["personal"]["orcid"] = "0000-0002-9079-593X"
    assert Candidate.model_validate(data).personal.orcid == "0000-0002-9079-593X"


def test_orcid_empty_string_means_absent() -> None:
    data = sample_profile_dict()
    data["personal"]["orcid"] = ""
    assert Candidate.model_validate(data).personal.orcid is None


@pytest.mark.parametrize(
    "value",
    [
        "0000-0002-1825-0098",  # wrong check digit
        "0000000218250097",  # no hyphens
        "https://orcid.org/0000-0002-1825-0097",  # URL, not the iD
        "0000-0002-1825-009",  # too short
    ],
)
def test_orcid_rejects_bad_format_or_checksum(value: str) -> None:
    data = sample_profile_dict()
    data["personal"]["orcid"] = value
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


def test_duplicate_achievement_text() -> None:
    data = sample_profile_dict()
    data["experience"][0]["achievements"].append(
        {
            "id": "other-id",
            "text": "Share of Wallet for 6M sellers.",
            "tags": [],
            "metrics": {},
        }
    )
    candidate = Candidate.model_validate(data)
    result = validate_candidate(candidate)
    assert not result.ok
    assert any("duplicate achievement text" in i.message for i in result.issues)


def test_missing_profile_file(tmp_path: Path) -> None:
    with pytest.raises(ProfileLoadError, match="not found"):
        load_profile(tmp_path / "missing.yaml")


def test_load_from_tmp(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(sample_profile_dict()), encoding="utf-8")
    candidate = load_profile(path)
    assert candidate.personal.email == "ana@example.com"

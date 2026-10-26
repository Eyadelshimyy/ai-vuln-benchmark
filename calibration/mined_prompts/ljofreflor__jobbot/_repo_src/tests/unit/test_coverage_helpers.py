"""Coverage for thin helpers that the suite used to skip entirely."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobbot.matching.scoring import format_match_report
from jobbot.models.match import JobMatch, MatchItem, MatchStrength
from jobbot.profile.diff import compare_summaries, summarize_profile
from jobbot.profile.loader import ProfileLoadError, load_profile, load_profile_raw


def test_format_match_report_lists_each_bucket() -> None:
    match = JobMatch(
        job_id="J0001",
        score=72.0,
        items=[
            MatchItem(label="Python", strength=MatchStrength.STRONG),
            MatchItem(label="dbt", strength=MatchStrength.PARTIAL),
            MatchItem(label="Scala", strength=MatchStrength.MISSING),
            MatchItem(label="visa", strength=MatchStrength.UNKNOWN),
        ],
    )
    report = format_match_report(match)
    assert "MATCH: 72%" in report
    assert "✓ Python" in report
    assert "~ dbt" in report
    assert "✗ Scala" in report
    assert "? visa" in report


def test_summarize_and_compare_profiles() -> None:
    local = {
        "personal": {"name": "Ada", "headline": "Engineer"},
        "experience": [{"achievements": [{"text": "one"}, {"text": "two"}]}],
        "education": [{}],
        "skills": {"other": ["Python"]},
        "publications": [],
    }
    generated = {
        "personal": {"name": "Ada", "headline": "Senior Engineer"},
        "experience": [{"achievements": [{"text": "one"}]}],
        "education": [{}],
        "skills": {"other": ["Python", "SQL"]},
        "publications": [{}],
    }
    left = summarize_profile(local)
    right = summarize_profile(generated)
    assert left["achievements"] == 2
    assert right["skills"] == 2
    lines = compare_summaries(left, right)
    assert any(line.startswith("= name") for line in lines)
    assert any(line.startswith("! headline") for line in lines)


def test_load_profile_happy_path(tmp_path: Path, project_root: Path) -> None:
    src = project_root / "data" / "profile.example.yaml"
    path = tmp_path / "profile.yaml"
    path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    candidate = load_profile(path)
    assert candidate.personal.name
    raw = load_profile_raw(path)
    assert isinstance(raw, dict)


def test_load_profile_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ProfileLoadError, match="not found"):
        load_profile(missing)
    bad = tmp_path / "bad.yaml"
    bad.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="mapping"):
        load_profile(bad)
    with pytest.raises(ProfileLoadError, match="mapping"):
        load_profile_raw(bad)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="empty"):
        load_profile(empty)
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump({"personal": {"email": "not-an-email"}}), encoding="utf-8")
    with pytest.raises(ProfileLoadError):
        load_profile(invalid)

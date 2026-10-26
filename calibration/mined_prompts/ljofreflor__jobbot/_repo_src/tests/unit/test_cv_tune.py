"""Bounded tune-for helpers (#54): resolve ref, HITL decision, timestamped backup."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.cv.tune import (
    backup_profile,
    char_delta_label,
    prompt_advice_decision,
    resolve_tune_ref,
)


def test_resolve_tune_ref_accepts_job_id() -> None:
    assert resolve_tune_ref("j0047") == ("id", "J0047")
    assert resolve_tune_ref("J0114") == ("id", "J0114")


def test_resolve_tune_ref_accepts_hard_link() -> None:
    url = "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    assert resolve_tune_ref(url) == ("url", url)


def test_resolve_tune_ref_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not a job id"):
        resolve_tune_ref("not-a-ref")
    with pytest.raises(ValueError, match="pass a job id"):
        resolve_tune_ref("  ")


def test_prompt_advice_decision_default_is_no() -> None:
    assert prompt_advice_decision(prompt=lambda _: "") == "no"
    assert prompt_advice_decision(prompt=lambda _: "N") == "no"


def test_prompt_advice_decision_yes_and_skip_all() -> None:
    assert prompt_advice_decision(prompt=lambda _: "y") == "yes"
    assert prompt_advice_decision(prompt=lambda _: "skip-all") == "skip_all"


def test_char_delta_label() -> None:
    assert char_delta_label("abcdefghij", "abcd") == "-6 chars"
    assert char_delta_label("hi", "hello") == "+3 chars"
    assert char_delta_label("same", "same") == "same length"


def test_backup_profile_writes_timestamped_copy(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("name: Ana\n", encoding="utf-8")

    backup = backup_profile(profile)

    assert backup.is_file()
    assert backup.name.startswith("profile.yaml.bak.")
    assert backup.read_text(encoding="utf-8") == "name: Ana\n"
    assert profile.read_text(encoding="utf-8") == "name: Ana\n"

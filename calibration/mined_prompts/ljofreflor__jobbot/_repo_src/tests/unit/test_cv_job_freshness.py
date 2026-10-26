"""A job CV on disk is stale once profile.yaml is newer than it."""

from __future__ import annotations

import os
from pathlib import Path

from jobbot.cv.build import should_rebuild_job_cv


def test_missing_cv_is_rebuilt(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("name: Ana\n", encoding="utf-8")
    assert should_rebuild_job_cv(tmp_path / "job", profile) is True


def test_existing_cv_older_than_profile_is_rebuilt(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    pdf = job_dir / "cv.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    profile = tmp_path / "profile.yaml"
    profile.write_text("name: Ana\n", encoding="utf-8")
    os.utime(pdf, (1_000, 1_000))
    os.utime(profile, (2_000, 2_000))
    assert should_rebuild_job_cv(job_dir, profile) is True


def test_cv_newer_than_profile_is_kept(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    pdf = job_dir / "cv.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    profile = tmp_path / "profile.yaml"
    profile.write_text("name: Ana\n", encoding="utf-8")
    os.utime(profile, (1_000, 1_000))
    os.utime(pdf, (2_000, 2_000))
    assert should_rebuild_job_cv(job_dir, profile) is False

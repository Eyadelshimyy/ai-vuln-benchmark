"""Torre application adapter: detect apply methods and handle redirects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobbot.adapters.base import ApplyMethod
from jobbot.adapters.torre.apply import TorreApplicationAdapter
from jobbot.adapters.torre.jobs import job_from_api_item
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def direct_torre_job(project_root: Path) -> JobPosting:
    """A job hosted directly on Torre (quickApply, no external redirect)."""
    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_opportunity.json").read_text(
            encoding="utf-8"
        )
    )
    return job_from_api_item(payload)


@pytest.fixture
def external_ats_job(project_root: Path) -> JobPosting:
    """A job that redirects to an external ATS (Greenhouse, Workday, etc.)."""
    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_external_redirect.json").read_text(
            encoding="utf-8"
        )
    )
    return job_from_api_item(payload)


def test_adapter_name() -> None:
    adapter = TorreApplicationAdapter()
    assert adapter.name == "torre"


def test_direct_torre_job_returns_unknown_method(direct_torre_job: JobPosting) -> None:
    """Direct Torre jobs need manual browser application (not yet automated)."""
    adapter = TorreApplicationAdapter()

    method = adapter.detect_method(direct_torre_job)

    assert method == ApplyMethod.UNKNOWN
    assert direct_torre_job.ats_kind == AtsKind.TORRE.value


def test_external_ats_job_returns_external_ats_method(external_ats_job: JobPosting) -> None:
    """Torre jobs that redirect to Greenhouse/Workday/etc should be flagged."""
    adapter = TorreApplicationAdapter()

    method = adapter.detect_method(external_ats_job)

    assert method == ApplyMethod.EXTERNAL_ATS
    assert external_ats_job.ats_kind == AtsKind.GREENHOUSE.value
    assert "greenhouse.io" in external_ats_job.url


def test_job_without_ats_kind_returns_unknown(direct_torre_job: JobPosting) -> None:
    """If ats_kind is missing, default to UNKNOWN."""
    direct_torre_job.ats_kind = None
    adapter = TorreApplicationAdapter()

    method = adapter.detect_method(direct_torre_job)

    assert method == ApplyMethod.UNKNOWN


def test_torre_job_with_torre_ats_kind_returns_unknown(direct_torre_job: JobPosting) -> None:
    """Even if ats_kind is explicitly TORRE, we return UNKNOWN (no automation yet)."""
    direct_torre_job.ats_kind = AtsKind.TORRE.value
    adapter = TorreApplicationAdapter()

    method = adapter.detect_method(direct_torre_job)

    assert method == ApplyMethod.UNKNOWN

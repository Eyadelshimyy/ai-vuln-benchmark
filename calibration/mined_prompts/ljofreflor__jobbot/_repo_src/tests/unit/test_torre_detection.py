"""Torre portal detection: recognize Torre URLs for the learning system."""

from __future__ import annotations

import pytest

from jobbot.portals.detect import AtsKind
from jobbot.portals.detectors.torre_detector import TorrePortalDetector


@pytest.fixture
def detector() -> TorrePortalDetector:
    return TorrePortalDetector()


def test_torre_ai_domain_is_detected(detector: TorrePortalDetector) -> None:
    url = "https://torre.ai/post/abc123-techcorp-senior-engineer"

    result = detector.detect(url)

    assert result.is_job_portal is True
    assert result.confidence == 1.0
    assert result.ats_kind == AtsKind.TORRE.value
    assert "Torre.ai" in result.evidence


def test_torre_co_domain_is_detected(detector: TorrePortalDetector) -> None:
    url = "https://torre.co/jobs/abc123"

    result = detector.detect(url)

    assert result.is_job_portal is True
    assert result.confidence == 1.0
    assert result.ats_kind == AtsKind.TORRE.value


def test_torre_subdomain_is_detected(detector: TorrePortalDetector) -> None:
    url = "https://jobs.torre.ai/search"

    result = detector.detect(url)

    assert result.is_job_portal is True
    assert result.confidence == 1.0


def test_case_insensitive_detection(detector: TorrePortalDetector) -> None:
    url = "https://TORRE.AI/post/xyz789"

    result = detector.detect(url)

    assert result.is_job_portal is True


def test_non_torre_url_is_not_detected(detector: TorrePortalDetector) -> None:
    url = "https://linkedin.com/jobs/view/123456"

    result = detector.detect(url)

    assert result.is_job_portal is False
    assert result.confidence == 0.0
    assert result.ats_kind is None


def test_torre_like_domain_is_not_detected(detector: TorrePortalDetector) -> None:
    """Avoid false positives from domains that contain 'torre' but aren't Torre."""
    url = "https://torrent.example.com/jobs"

    result = detector.detect(url)

    assert result.is_job_portal is False


def test_html_is_not_required_for_detection(detector: TorrePortalDetector) -> None:
    """Detection is URL-based, so HTML is optional."""
    url = "https://torre.ai/jobs/search"

    result_without_html = detector.detect(url, html=None)
    result_with_html = detector.detect(url, html="<html>...</html>")

    assert result_without_html.is_job_portal is True
    assert result_with_html.is_job_portal is True
    assert result_without_html.confidence == result_with_html.confidence

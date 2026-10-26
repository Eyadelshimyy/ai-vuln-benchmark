"""Test Indeed URL canonicalization and jk validation (#211)."""

from __future__ import annotations

import pytest

from jobbot.jobs.indeed_url import (
    IndeedUrlError,
    canonical_indeed_job_url,
    extract_indeed_jk,
    indeed_jk_rejection,
)


def test_canonical_url_strips_tracking_params() -> None:
    """Tracking params (from, tk, rjptk, etc.) must be stripped."""
    url = "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email&tk=abc&rjptk=def"

    canonical = canonical_indeed_job_url(url)

    assert canonical == "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    assert "from=" not in canonical
    assert "tk=" not in canonical


def test_canonical_url_preserves_country_host() -> None:
    """The original country subdomain must be preserved."""
    us_url = "https://www.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    cl_url = "https://cl.indeed.com/viewjob?jk=25db4649b81bfb9d"
    mx_url = "https://mx.indeed.com/viewjob?jk=b81bfb9d25db4649"

    assert canonical_indeed_job_url(us_url) == "https://www.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    assert canonical_indeed_job_url(cl_url) == "https://cl.indeed.com/viewjob?jk=25db4649b81bfb9d"
    assert canonical_indeed_job_url(mx_url) == "https://mx.indeed.com/viewjob?jk=b81bfb9d25db4649"


def test_canonical_url_rejects_missing_jk() -> None:
    """URL without jk parameter is a validation error."""
    with pytest.raises(IndeedUrlError, match="Missing 'jk'"):
        canonical_indeed_job_url("https://cl.indeed.com/viewjob?foo=bar")


def test_canonical_url_rejects_empty_jk() -> None:
    """Empty jk parameter is invalid."""
    with pytest.raises(IndeedUrlError, match="Missing 'jk'"):
        canonical_indeed_job_url("https://cl.indeed.com/viewjob?jk=")


def test_canonical_url_rejects_non_hex_jk() -> None:
    """jk must be a hex string of length 16."""
    with pytest.raises(IndeedUrlError, match="formato"):
        canonical_indeed_job_url("https://cl.indeed.com/viewjob?jk=not-hex-123")


def test_canonical_url_rejects_short_jk() -> None:
    with pytest.raises(IndeedUrlError, match="formato"):
        canonical_indeed_job_url("https://cl.indeed.com/viewjob?jk=abc123")


def test_canonical_url_rejects_non_indeed_host() -> None:
    """Only indeed.com domains are valid."""
    with pytest.raises(IndeedUrlError, match="Not an Indeed URL"):
        canonical_indeed_job_url("https://linkedin.com/jobs/view/123")

    with pytest.raises(IndeedUrlError, match="Not an Indeed URL"):
        canonical_indeed_job_url(
            "https://fake-indeed.com/viewjob?jk=8a5fab1a7c476a97"
        )


def test_extract_jk() -> None:
    """extract_indeed_jk returns just the job key."""
    url = "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email"

    jk = extract_indeed_jk(url)

    assert jk == "8a5fab1a7c476a97"


def test_extract_jk_raises_on_invalid_url() -> None:
    """extract_indeed_jk validates the URL."""
    with pytest.raises(IndeedUrlError):
        extract_indeed_jk("https://linkedin.com/jobs/123")


@pytest.mark.parametrize(
    ("jk", "reason"),
    [
        ("123456789abcdef0", "secuencia"),
        ("fedcba9876543210", "secuencia"),
        ("0123456789abcdef", "secuencia"),
        ("aaaaaaaaaaaaaaaa", "repetido"),
        ("abc123", "formato"),
        ("", "formato"),
    ],
)
def test_indeed_jk_rejection_reasons(jk: str, reason: str) -> None:
    assert indeed_jk_rejection(jk) == reason


@pytest.mark.parametrize(
    "jk",
    [
        "8a5fab1a7c476a97",
        "25db4649b81bfb9d",
        "b81bfb9d25db4649",
    ],
)
def test_indeed_jk_accepts_real_keys(jk: str) -> None:
    assert indeed_jk_rejection(jk) is None

"""Indeed job URL canonicalization and validation."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

# Real Indeed job keys are 16 hex digits. Observed decoys and obvious sequences fail.
_JK_HEX = re.compile(r"^[a-f0-9]{16}$", re.IGNORECASE)


class IndeedUrlError(ValueError):
    """Invalid Indeed job URL or job key."""


def indeed_jk_rejection(jk: str) -> str | None:
    """Return why ``jk`` is not a real Indeed job key, or ``None`` if it is valid.

    Reasons: ``formato`` (not exactly 16 hex), ``secuencia`` (ascending/descending
    run such as the observed decoys), ``repetido`` (one hex digit repeated).
    """
    if not jk or not _JK_HEX.fullmatch(jk):
        return "formato"
    lowered = jk.casefold()
    if len(set(lowered)) == 1:
        return "repetido"
    if _is_hex_sequence(lowered):
        return "secuencia"
    return None


def validate_indeed_jk(jk: str) -> str:
    """Return the job key when valid; raise ``IndeedUrlError`` with the rejection reason."""
    reason = indeed_jk_rejection(jk)
    if reason is not None:
        raise IndeedUrlError(f"Invalid 'jk' ({reason}): {jk}")
    return jk.casefold()


def canonical_indeed_job_url(url: str) -> str:
    """Extract canonical Indeed job URL (country host + jk only).

    Args:
        url: Indeed job URL (viewjob or with tracking params)

    Returns:
        Canonical URL: https://<country>.indeed.com/viewjob?jk=<hex>

    Raises:
        IndeedUrlError: URL is not a valid Indeed job URL or missing/invalid jk

    Examples:
        >>> canonical_indeed_job_url(
        ...     "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email"
        ... )
        'https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97'
    """
    if not url:
        raise IndeedUrlError("Empty URL")

    parsed = urlparse(url)

    if not parsed.hostname:
        raise IndeedUrlError(f"No hostname in URL: {url}")

    host_lower = parsed.hostname.lower()
    if not host_lower.endswith(".indeed.com") and host_lower != "indeed.com":
        raise IndeedUrlError(f"Not an Indeed URL: {url}")

    query_params = parse_qs(parsed.query)
    jk_values = query_params.get("jk", [])

    if not jk_values or not jk_values[0]:
        raise IndeedUrlError(f"Missing 'jk' parameter in URL: {url}")

    jk = validate_indeed_jk(jk_values[0])

    scheme = parsed.scheme or "https"
    host = parsed.hostname

    return f"{scheme}://{host}/viewjob?jk={jk}"


def extract_indeed_jk(url: str) -> str:
    """Extract the jk (job key) from an Indeed URL.

    Args:
        url: Indeed job URL

    Returns:
        The jk parameter value (lowercase)

    Raises:
        IndeedUrlError: URL is invalid or missing jk
    """
    canonical = canonical_indeed_job_url(url)
    match = re.search(r"jk=([a-f0-9]{16})", canonical, re.IGNORECASE)
    if not match:
        raise IndeedUrlError(f"Could not extract jk from URL: {url}")
    return match.group(1).casefold()


def _is_hex_sequence(jk: str) -> bool:
    """True when each digit is the previous ±1 mod 16 (wraps, so decoys match)."""
    digits = [int(char, 16) for char in jk]
    if len(digits) < 2:
        return False
    diffs = [((digits[i + 1] - digits[i]) % 16) for i in range(len(digits) - 1)]
    return all(diff == 1 for diff in diffs) or all(diff == 15 for diff in diffs)


__all__ = [
    "IndeedUrlError",
    "canonical_indeed_job_url",
    "extract_indeed_jk",
    "indeed_jk_rejection",
    "validate_indeed_jk",
]

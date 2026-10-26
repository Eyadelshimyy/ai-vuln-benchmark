"""Core types and registry for OtterWorks DAST probes.

A probe is a single, targeted attack attempt against the *running* application.
It is deliberately narrower than a crawler: each probe encodes one concrete
abuse case, states what evidence proves the vulnerability, and returns a
machine-readable verdict.

The verdict is the verification loop. A probe that reports ``VULNERABLE`` is a
reproduction of the attack; the same probe reporting ``SECURE`` after a code
change is programmatic proof that the finding is closed.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

#: Reports are uploaded as CI artifacts and echoed into job summaries, so evidence
#: must never carry a usable credential. Auth responses contain `accessToken`, and
#: probes deliberately attack the login path, so redaction happens at capture time
#: rather than at each call site.
#: Matched as substrings of the key, not whole keys: DAST-SENSITIVE-DATA-EXPOSURE
#: fires precisely when a login response carries `passwordHash`, `password_hash`,
#: `salt` or `mfaSecret`, and attaches that response as its evidence.
SECRET_FIELDS = (
    "token",
    "password",
    "passwd",
    "secret",
    "salt",
    "credential",
    "authorization",
    "apikey",
    "api_key",
    "privatekey",
)
_SECRET_KEY = rf'"[A-Za-z0-9_-]*(?:{"|".join(SECRET_FIELDS)})[A-Za-z0-9_-]*"\s*:\s*'
#: The closing quote is optional so a value cut off by truncation is still caught.
_SECRET_JSON = re.compile(rf'({_SECRET_KEY}")[^"]*("|$)', re.IGNORECASE)
#: The same key with an unquoted value (a numeric salt, a null hash).
_SECRET_JSON_BARE = re.compile(rf'({_SECRET_KEY})([^"\s,}}\]]+)', re.IGNORECASE)
_SECRET_QUERY = re.compile(
    rf"([A-Za-z0-9_-]*(?:{'|'.join(SECRET_FIELDS)})[A-Za-z0-9_-]*=)[^&\s]+",
    re.IGNORECASE,
)
#: A bare JWT, in case it appears under a field name we do not know about. The
#: trailing segments are optional so a token cut off mid-string is still caught.
_JWT = re.compile(r"\bey[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*){0,2}")


def redact(text: str) -> str:
    """Strip credential material from anything destined for a report."""
    text = _SECRET_JSON.sub("\\1[REDACTED]\\2", text)
    text = _SECRET_JSON_BARE.sub(r"\1[REDACTED]", text)
    text = _SECRET_QUERY.sub(r"\1[REDACTED]", text)
    return _JWT.sub("[REDACTED-JWT]", text)


class Severity(enum.StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Verdict(enum.StrEnum):
    VULNERABLE = "vulnerable"
    SECURE = "secure"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


SEVERITY_ORDER = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


def unavailable(response: httpx.Response) -> bool:
    """Whether the response came from something other than the control under test.

    A 5xx is a broken or unreachable backend and a 429 is the limiter, not a
    decision about the request. Either way nothing was assessed, so a probe must
    not count it towards a pass.
    """
    return response.status_code >= 500 or response.status_code == 429


@dataclass
class Evidence:
    """The request/response pair that demonstrates the verdict."""

    request: str
    response_status: int | None = None
    response_excerpt: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        self.request = redact(self.request)
        self.response_excerpt = redact(self.response_excerpt)
        self.note = redact(self.note)

    @classmethod
    def from_response(cls, response: httpx.Response, note: str = "", limit: int = 400) -> Evidence:
        return cls(
            request=f"{response.request.method} {response.request.url}",
            response_status=response.status_code,
            # Redact before truncating: a cut halfway through a token would
            # otherwise leave a fragment that matches no pattern.
            response_excerpt=redact(response.text)[:limit],
            note=note,
        )


@dataclass
class Result:
    """The outcome of running one probe."""

    finding_id: str
    title: str
    severity: Severity
    owasp: str
    cwe: str
    service: str
    verdict: Verdict
    detail: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    remediation: str = ""

    def __post_init__(self) -> None:
        self.detail = redact(self.detail)

    @property
    def is_finding(self) -> bool:
        return self.verdict is Verdict.VULNERABLE

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        payload["verdict"] = self.verdict.value
        return payload


@dataclass
class Probe:
    """A registered attack case."""

    finding_id: str
    title: str
    severity: Severity
    owasp: str
    cwe: str
    service: str
    remediation: str
    run: Callable[..., Result]
    #: False for probes that attack the unauthenticated surface. The runner
    #: reports the rest as inconclusive when identity seeding failed, so an
    #: unauthenticated 401 can never be mistaken for a passing attack.
    requires_identity: bool = True

    def result(
        self,
        verdict: Verdict,
        detail: str = "",
        evidence: list[Evidence] | None = None,
    ) -> Result:
        return Result(
            finding_id=self.finding_id,
            title=self.title,
            severity=self.severity,
            owasp=self.owasp,
            cwe=self.cwe,
            service=self.service,
            verdict=verdict,
            detail=detail,
            evidence=evidence or [],
            remediation=self.remediation,
        )


REGISTRY: dict[str, Probe] = {}


def probe(
    *,
    finding_id: str,
    title: str,
    severity: Severity,
    owasp: str,
    cwe: str,
    service: str,
    remediation: str,
    requires_identity: bool = True,
) -> Callable[[Callable[..., Result]], Callable[..., Result]]:
    """Register an attack case under a stable finding ID."""

    def decorator(fn: Callable[..., Result]) -> Callable[..., Result]:
        if finding_id in REGISTRY:
            raise ValueError(f"duplicate DAST finding id: {finding_id}")
        entry = Probe(
            finding_id=finding_id,
            title=title,
            severity=severity,
            owasp=owasp,
            cwe=cwe,
            service=service,
            remediation=remediation,
            run=fn,
            requires_identity=requires_identity,
        )
        REGISTRY[finding_id] = entry
        fn.probe = entry  # type: ignore[attr-defined]
        return fn

    return decorator

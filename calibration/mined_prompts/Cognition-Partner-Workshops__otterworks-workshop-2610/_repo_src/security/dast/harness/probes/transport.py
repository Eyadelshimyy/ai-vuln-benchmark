"""Configuration and transport-level attack cases (OWASP API7/API8)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from httpx import HTTPError, Response

from .base import Evidence, Result, Severity, Verdict, probe, unavailable
from .context import ScanContext

REQUIRED_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": None,
    "content-security-policy": None,
    "referrer-policy": None,
}

# HSTS is only meaningful, and only expected, on a TLS listener.
TLS_ONLY_HEADERS = {"strict-transport-security": None}

#: How much more throughput a spoofed burst must win before it counts as a bypass.
#: The unspoofed burst is the control, so this is a ratio rather than an absolute:
#: a limiter behaving identically under both bursts lands at ~1.0.
BYPASS_MARGIN = 1.25


def _last_evidence(responses: list[Response | None], note: str) -> list[Evidence]:
    """Evidence from the last request that actually got a response."""
    last = next((r for r in reversed(responses) if r is not None), None)
    return [Evidence.from_response(last, note=note)] if last is not None else []


@probe(
    finding_id="DAST-MISSING-SECURITY-HEADERS",
    title="API responses omit baseline browser security headers",
    severity=Severity.MEDIUM,
    owasp="API8:2023 Security Misconfiguration",
    cwe="CWE-693",
    service="api-gateway",
    remediation=(
        "Add a security-headers middleware to the gateway stack that sets "
        "X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, "
        "Referrer-Policy, and Strict-Transport-Security on every response."
    ),
    requires_identity=False,  # the headers are expected on the rejection too
)
def missing_security_headers(ctx: ScanContext) -> Result:
    self = missing_security_headers.probe
    response = ctx.get("/api/v1/documents/", identity=ctx.attacker)
    expectations = dict(REQUIRED_HEADERS)
    if ctx.base_url.startswith("https://"):
        expectations.update(TLS_ONLY_HEADERS)
    missing = []
    for header, expected in expectations.items():
        value = response.headers.get(header)
        if value is None or (expected is not None and expected not in value.lower()):
            missing.append(header)
    if missing and unavailable(response):
        # A 429 is written by the limiter and a 5xx by a failing hop, both before any
        # response-header middleware runs, so their headers describe nothing.
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the request returned {response.status_code} without reaching the header "
            "middleware, so the headers cannot be assessed",
            [Evidence.from_response(response)],
        )
    if missing:
        return self.result(
            Verdict.VULNERABLE,
            f"missing or weak headers: {', '.join(sorted(missing))}",
            [Evidence.from_response(response, note=f"headers seen: {sorted(response.headers)}")],
        )
    return self.result(
        Verdict.SECURE,
        "all baseline security headers present",
        [Evidence.from_response(response)],
    )


@probe(
    finding_id="DAST-CORS-ORIGIN-REFLECTION",
    title="CORS policy reflects an arbitrary origin with credentials",
    severity=Severity.HIGH,
    owasp="API8:2023 Security Misconfiguration",
    cwe="CWE-942",
    service="api-gateway",
    remediation=(
        "Only echo Access-Control-Allow-Origin for origins on the configured allowlist, and "
        "never combine a wildcard origin with Access-Control-Allow-Credentials: true."
    ),
    requires_identity=False,  # the CORS headers are on the rejection too
)
def cors_origin_reflection(ctx: ScanContext) -> Result:
    self = cors_origin_reflection.probe
    evil = "https://otterworks-attacker.example"
    response = ctx.get("/api/v1/documents/", identity=ctx.attacker, headers={"Origin": evil})
    allow_origin = response.headers.get("access-control-allow-origin", "")
    allow_creds = response.headers.get("access-control-allow-credentials", "").lower() == "true"
    evidence = [
        Evidence.from_response(
            response,
            note=f"Origin: {evil} -> Allow-Origin: {allow_origin or '(none)'}, "
            f"Allow-Credentials: {allow_creds}",
        )
    ]
    if allow_origin in (evil, "*") and allow_creds:
        return self.result(
            Verdict.VULNERABLE,
            "untrusted origin reflected alongside credentials",
            evidence,
        )
    if unavailable(response):
        # The limiter sits in front of the CORS middleware, so a 429 (like a 5xx from
        # further back) carries no Access-Control-* headers at all: their absence says
        # nothing about the policy.
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the request returned {response.status_code} without reaching the CORS "
            "middleware, so the policy cannot be assessed",
            evidence,
        )
    return self.result(Verdict.SECURE, "untrusted origin was not granted access", evidence)


@probe(
    finding_id="DAST-RATE-LIMIT-BYPASS",
    title="Per-IP rate limiting is bypassable with a spoofed forwarding header",
    severity=Severity.HIGH,
    owasp="API4:2023 Unrestricted Resource Consumption",
    cwe="CWE-770",
    service="api-gateway",
    remediation=(
        "Only honour X-Forwarded-For / X-Real-IP from trusted proxy hops, and key the rate "
        "limiter on an identity the client cannot choose (authenticated subject or peer IP)."
    ),
    requires_identity=False,
)
def rate_limit_bypass(ctx: ScanContext) -> Result:
    """Burst past the limiter, then repeat the burst rotating X-Forwarded-For."""
    self = rate_limit_bypass.probe
    burst = ctx.rate_limit_burst

    def hit(index: int, spoof: bool) -> Response | None:
        headers = {"X-Forwarded-For": f"10.9.{index // 256}.{index % 256}"} if spoof else None
        try:
            return ctx.get("/health", headers=headers)
        except HTTPError:
            # A reset or timeout is the expected shape of flooding a gateway. It is
            # neither served nor throttled, so it is counted rather than allowed to
            # abort the burst and discard every other observation.
            return None

    # The limiter is a token bucket refilling at RATE_LIMIT_RPS per second, so a
    # sequential loop only ever drains it when the round trip is faster than the
    # refill — true locally, false against any deployed tenant. Issue the burst
    # concurrently so the request rate beats the refill regardless of latency.
    with ThreadPoolExecutor(max_workers=ctx.rate_limit_workers) as pool:
        baseline = list(pool.map(lambda i: hit(i, False), range(burst)))
        baseline_errors = sum(1 for r in baseline if r is None)
        if baseline_errors > burst // 2:
            return self.result(
                Verdict.INCONCLUSIVE,
                f"{baseline_errors}/{burst} unspoofed requests failed at the transport level, "
                "so the limiter's behaviour cannot be measured",
            )
        if not any(r is not None and r.status_code == 429 for r in baseline):
            return self.result(
                Verdict.INCONCLUSIVE,
                f"{burst} concurrent requests never drew a 429, so either no limiter is "
                "configured at this edge or its allowance exceeds the burst; the bypass "
                "cannot be tested",
                _last_evidence(baseline, "last unspoofed request"),
            )

        spoofed = list(pool.map(lambda i: hit(i, True), range(burst)))

    spoofed_errors = sum(1 for r in spoofed if r is None)
    if spoofed_errors > burst // 2:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"{spoofed_errors}/{burst} spoofed requests failed at the transport level, so the "
            "two bursts are not comparable",
        )

    # Only a request that came back says anything about the limiter, and the two bursts
    # do not error equally (the unspoofed one runs first, on a cold pool). Comparing raw
    # counts would let the control's own transport errors read as throttling and turn a
    # working limiter into a bypass, so both sides are fractions of what responded.
    served = sum(1 for r in spoofed if r is not None and r.status_code != 429)
    baseline_served = sum(1 for r in baseline if r is not None and r.status_code != 429)
    spoofed_rate = served / (burst - spoofed_errors)
    baseline_rate = baseline_served / (burst - baseline_errors)

    note = (
        f"{served}/{burst - spoofed_errors} answered requests served while rotating "
        f"X-Forwarded-For vs {baseline_served}/{burst - baseline_errors} unspoofed"
        + (
            f", {spoofed_errors + baseline_errors} transport errors"
            if spoofed_errors or baseline_errors
            else ""
        )
    )
    evidence = _last_evidence(spoofed, note)
    # Two ways to be a bypass. Absolute: the spoofed burst drew no 429 at all while
    # the unspoofed one did, so rotating the header removed the limiter outright —
    # this is the case the ratio cannot see, since the rate is capped at 1. It still
    # needs the unspoofed burst to have been materially throttled: a limiter that served
    # 1499/1500 unspoofed and 1500/1500 spoofed is jitter, not a bypass.
    if spoofed_rate == 1 and baseline_rate * BYPASS_MARGIN < 1:
        return self.result(
            Verdict.VULNERABLE,
            f"rotating X-Forwarded-For served every request that came back "
            f"({served}/{burst - spoofed_errors}) while the same unspoofed burst was "
            f"throttled to {baseline_served}/{burst - baseline_errors}",
            evidence,
        )
    # Relative: a partial bypass is still a bypass if spoofing bought materially more
    # throughput. Some other layer (ingress, a global bucket) can still return the odd
    # 429, so requiring every request through would report a near-total bypass as safe.
    if spoofed_rate > baseline_rate * BYPASS_MARGIN:
        return self.result(
            Verdict.VULNERABLE,
            f"rotating X-Forwarded-For served {spoofed_rate:.0%} of the requests that came "
            f"back against {baseline_rate:.0%} for the same unspoofed burst ({note})",
            evidence,
        )
    if baseline_rate * BYPASS_MARGIN >= 1:
        # The limiter's allowance is wide enough that the ratio test could not have
        # fired whatever the spoofed burst did, and it was not a clean sweep either:
        # the burst is too small to separate a bypass from a generous limit.
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the unspoofed burst was barely throttled ({baseline_rate:.0%} of answered "
            f"requests served), so a burst of {burst} cannot distinguish a bypass from a "
            "generous allowance; raise rate_limit_burst to test this target",
            evidence,
        )
    return self.result(
        Verdict.SECURE,
        f"the limiter throttled the spoofed burst comparably to the unspoofed one ({note})",
        evidence,
    )


@probe(
    finding_id="DAST-EXPOSED-TELEMETRY",
    title="Operational endpoints are exposed unauthenticated at the edge",
    severity=Severity.MEDIUM,
    owasp="API8:2023 Security Misconfiguration",
    cwe="CWE-497",
    service="api-gateway",
    remediation=(
        "Serve /metrics and other operational endpoints on an internal listener or behind "
        "an authenticated ingress rule rather than the public gateway."
    ),
    requires_identity=False,
)
def exposed_telemetry(ctx: ScanContext) -> Result:
    self = exposed_telemetry.probe
    evidence: list[Evidence] = []
    exposed: list[str] = []
    unreached: list[str] = []
    for index, (path, marker) in enumerate(
        (
            ("/metrics", "go_goroutines"),
            ("/actuator/env", "propertySources"),
        )
    ):
        # A fresh forwarding address per request: the rate-limit probe runs just
        # before this one and leaves the scanner's own bucket drained.
        response = ctx.get(path, headers={"X-Forwarded-For": f"198.51.100.{index + 1}"})
        if response.status_code == 200 and marker in response.text:
            exposed.append(path)
            evidence.append(Evidence.from_response(response, note=path))
        elif unavailable(response):
            unreached.append(f"{path} -> {response.status_code}")
            evidence.append(Evidence.from_response(response, note=path))
    if exposed:
        return self.result(
            Verdict.VULNERABLE,
            f"unauthenticated telemetry at {', '.join(exposed)}",
            evidence,
        )
    if unreached:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"throttled or erroring, so exposure cannot be assessed: {', '.join(unreached)}",
            evidence,
        )
    return self.result(Verdict.SECURE, "no unauthenticated telemetry endpoints found")

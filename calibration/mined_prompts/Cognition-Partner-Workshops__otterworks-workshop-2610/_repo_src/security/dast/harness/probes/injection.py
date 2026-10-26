"""Injection and information-disclosure attack cases (OWASP API3/API8)."""

from __future__ import annotations

import re

from .base import Evidence, Result, Severity, Verdict, probe, unavailable
from .context import ScanContext

SQL_ERROR_SIGNATURES = (
    "syntax error at or near",
    "unterminated quoted string",
    "psycopg2",
    "sqlalchemy.exc",
    "org.postgresql.util.psqlexception",
    "sqlstate",
)

LEAK_SIGNATURES = (
    "traceback (most recent call last)",
    "at org.springframework",
    "goroutine 1 [running]",
    "panic: runtime error",
    "jdbc:postgresql://",
    "otterworks_dev",
)

SQLI_PAYLOADS = ("'", "1' OR '1'='1", "') OR ('a'='a", "'; SELECT pg_sleep(0) --")


@probe(
    finding_id="DAST-SQLI-ERROR-BASED",
    title="SQL error surfaced from an injected query parameter",
    severity=Severity.CRITICAL,
    owasp="API3:2023 Broken Object Property Level Authorization",
    cwe="CWE-89",
    service="search-service, document-service",
    remediation=(
        "Bind every user-supplied value as a query parameter instead of interpolating it "
        "into SQL, and return a generic 400 for malformed input."
    ),
)
def sqli_error_based(ctx: ScanContext) -> Result:
    self = sqli_error_based.probe
    targets = [
        ("/api/v1/search/", "q"),
        ("/api/v1/documents/", "title"),
        # No trailing slash: file-service registers the collection as "" inside
        # web::scope("/api/v1/files"), so "/api/v1/files/" is a 404 that never
        # reaches a query.
        ("/api/v1/files", "folder_id"),
    ]
    evidence: list[Evidence] = []
    # A payload that never reached the query layer cannot surface an error from it,
    # so track whether any endpoint actually processed one.
    reached: list[str] = []
    unreached: list[str] = []
    for path, param in targets:
        for payload in SQLI_PAYLOADS:
            response = ctx.get(path, params={param: payload}, identity=ctx.attacker)
            body = response.text.lower()
            hit = next((sig for sig in SQL_ERROR_SIGNATURES if sig in body), None)
            if hit:
                evidence.append(
                    Evidence.from_response(response, note=f"{param}={payload!r} leaked {hit!r}")
                )
                return self.result(
                    Verdict.VULNERABLE,
                    f"{path} leaked a SQL error for {param}={payload!r}",
                    evidence,
                )
            # A 404/405 is the router refusing the request, so like a refusal or a
            # throttle it says nothing about how the query layer handles the payload.
            if unavailable(response) or response.status_code in (401, 403, 404, 405):
                unreached.append(f"{path} -> {response.status_code}")
            else:
                reached.append(path)

    if not reached:
        return self.result(
            Verdict.INCONCLUSIVE,
            "no injected parameter reached a query: every request was refused, throttled or "
            f"failed ({', '.join(sorted(set(unreached)))})",
        )
    # Unlike the admin probe, a target that refused the payload does not undermine the
    # claim about the ones that took it — but the claim only covers those, so say so.
    scope = f" ({', '.join(sorted(set(unreached)))} never took a payload)" if unreached else ""
    return self.result(Verdict.SECURE, f"no SQL errors surfaced from injected parameters{scope}")


@probe(
    finding_id="DAST-VERBOSE-ERRORS",
    title="Unhandled input returns a stack trace or internal connection detail",
    severity=Severity.MEDIUM,
    owasp="API8:2023 Security Misconfiguration",
    cwe="CWE-209",
    service="all",
    remediation=(
        "Return a generic JSON error envelope on unhandled exceptions and log the detail "
        "server-side; disable debug/development error pages in every deployed profile."
    ),
)
def verbose_errors(ctx: ScanContext) -> Result:
    self = verbose_errors.probe
    cases = [
        ("GET", "/api/v1/documents/not-a-uuid", None),
        ("GET", "/api/v1/files/%00", None),
        ("POST", "/api/v1/documents/", {"title": {"nested": [1, 2]}, "content": None}),
        ("GET", "/api/v1/search/?page=notanumber", None),
    ]
    evidence: list[Evidence] = []
    # Only a response the owning service produced says anything about its error
    # handling: a gateway 5xx or a blanket refusal never exercised the handler.
    reached = 0
    unreached: list[str] = []
    for method, path, body in cases:
        response = ctx.request(method, path, identity=ctx.attacker, json=body)
        lowered = response.text.lower()
        hit = next((sig for sig in LEAK_SIGNATURES if sig in lowered), None)
        if hit:
            evidence.append(Evidence.from_response(response, note=f"leaked {hit!r}"))
            return self.result(
                Verdict.VULNERABLE,
                f"{method} {path} leaked internal detail ({hit!r})",
                evidence,
            )
        if unavailable(response) or response.status_code in (401, 403, 404):
            unreached.append(f"{method} {path} -> {response.status_code}")
        else:
            reached += 1

    if not reached:
        return self.result(
            Verdict.INCONCLUSIVE,
            "no malformed request reached the owning service: every case was refused, "
            f"throttled or failed ({', '.join(unreached)})",
        )
    scope = f" ({', '.join(unreached)} never reached a handler)" if unreached else ""
    return self.result(Verdict.SECURE, f"malformed input produced no internal detail{scope}")


@probe(
    finding_id="DAST-STORED-XSS-DOCUMENTS",
    title="Document content is served back as renderable HTML",
    severity=Severity.HIGH,
    owasp="API8:2023 Security Misconfiguration",
    cwe="CWE-79",
    service="document-service",
    remediation=(
        "Always serve API payloads as application/json with X-Content-Type-Options: nosniff "
        "and escape user content wherever it is rendered."
    ),
)
def stored_xss_documents(ctx: ScanContext) -> Result:
    self = stored_xss_documents.probe
    payload = f"<script>alert('{ctx.run_id}')</script>"
    created = ctx.create_document(ctx.attacker, title=f"xss-{ctx.run_id}", content=payload)
    if created is None:
        return self.result(Verdict.INCONCLUSIVE, "could not create a document to test")

    response = ctx.get(f"/api/v1/documents/{created['id']}", identity=ctx.attacker)
    content_type = response.headers.get("content-type", "")
    reflected = payload in response.text
    evidence = [Evidence.from_response(response, note=f"content-type: {content_type}")]
    if reflected and "text/html" in content_type:
        return self.result(
            Verdict.VULNERABLE, "payload reflected in an HTML-typed response", evidence
        )
    if response.status_code != 200:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the read-back returned {response.status_code}, so the stored content was never "
            "served and the response typing cannot be assessed",
            evidence,
        )
    return self.result(
        Verdict.SECURE, "payload only returned inside a JSON-typed response", evidence
    )


@probe(
    finding_id="DAST-CREDENTIAL-BRUTE-FORCE",
    title="Login accepts unlimited failed attempts against one account",
    severity=Severity.HIGH,
    owasp="API2:2023 Broken Authentication",
    cwe="CWE-307",
    service="auth-service",
    remediation=(
        "Track failed attempts per account and per source, and apply exponential backoff or "
        "temporary lockout after a small threshold."
    ),
)
def credential_brute_force(ctx: ScanContext) -> Result:
    self = credential_brute_force.probe
    attempts = ctx.brute_force_attempts
    statuses = []
    last = None
    # Aimed at the burner, not the victim: a target that correctly locks the
    # account must not strand the later probes that log in as the victim.
    target = ctx.burner
    # Control request first, so "the correct password stops working" afterwards reads as
    # a lockout rather than an account that never worked.
    before = ctx.login_response(target.email, target.password)
    if before.status_code != 200:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the burner's own password returned {before.status_code} before the attack, so "
            "nothing that happens to it during the attempts can be attributed to a lockout",
            [Evidence.from_response(before, note="pre-attack control login")],
        )
    for i in range(attempts):
        last = ctx.request(
            "POST",
            "/api/v1/auth/login",
            json={"email": target.email, "password": f"wrong-password-{i}"},
            headers={"X-Forwarded-For": "203.0.113.7"},
        )
        statuses.append(last.status_code)
        if last.status_code == 423:
            return self.result(
                Verdict.SECURE,
                f"the account was locked after {i + 1} failed attempts",
                [Evidence.from_response(last)],
            )
        if last.status_code == 429:
            # A per-source throttle is not a per-account control: it is keyed on the
            # forwarding header DAST-RATE-LIMIT-BYPASS exists to show is spoofable,
            # so an attacker rotates around it and the lockout stays untested.
            return self.result(
                Verdict.INCONCLUSIVE,
                f"the request limiter answered after {i + 1} attempts; that throttle is keyed "
                "on the source, not the account, so lockout cannot be assessed",
                [Evidence.from_response(last)],
            )

    after = ctx.login_response(target.email, target.password)
    evidence = [
        Evidence.from_response(
            last,
            note=f"{attempts} failed attempts, statuses seen: {sorted(set(statuses))}",
        ),
        Evidence.from_response(after, note="post-attack control login"),
    ]
    if unavailable(after):
        # The attempts were spoofed onto one source and this control login goes out from
        # the scanner's own, so a 429 here is a different bucket entirely — and a 5xx is
        # a broken service. Neither is the account refusing a correct password.
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the post-attack control login returned {after.status_code}, which is the "
            "limiter or a failing service rather than the account, so a lockout cannot be "
            "distinguished from an absent one",
            evidence,
        )
    if after.status_code != 200:
        # The password worked before the attack and is refused now: the account was
        # locked. Most implementations signal that with the same generic 401 as any
        # other failure, so the status codes during the loop would never show it.
        return self.result(
            Verdict.SECURE,
            f"the correct password was refused ({after.status_code}) after {attempts} failed "
            "attempts but succeeded before them, so the account was locked",
            evidence,
        )
    return self.result(
        Verdict.VULNERABLE,
        f"{attempts} consecutive failed logins were accepted without throttling or lockout",
        evidence,
    )


@probe(
    finding_id="DAST-SENSITIVE-DATA-IN-RESPONSE",
    title="Authentication responses expose sensitive user fields",
    severity=Severity.MEDIUM,
    owasp="API3:2023 Broken Object Property Level Authorization",
    cwe="CWE-213",
    service="auth-service",
    remediation=(
        "Serialize responses from an explicit allowlist DTO so credential material and "
        "internal columns can never be returned."
    ),
)
def sensitive_data_in_response(ctx: ScanContext) -> Result:
    self = sensitive_data_in_response.probe
    response = ctx.request(
        "POST",
        "/api/v1/auth/login",
        json={"email": ctx.victim.email, "password": ctx.victim.password},
    )
    if response.status_code != 200:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"login returned {response.status_code}",
            [Evidence.from_response(response)],
        )
    leaked = [
        field
        for field in ("passwordHash", "password_hash", "password", "salt", "mfaSecret")
        if re.search(rf'"{field}"\s*:', response.text)
    ]
    if leaked:
        return self.result(
            Verdict.VULNERABLE,
            f"login response contained {', '.join(leaked)}",
            [Evidence.from_response(response)],
        )
    return self.result(Verdict.SECURE, "login response exposed no credential material")


#: A file that exists on every container image and belongs to no export archive,
#: so serving it is unambiguous proof the handler left its own directory. Each
#: entry is the ``name`` value exactly as it must appear on the wire, so the
#: percent-encoded variant reaches the handler still encoded - passing it as a
#: parameter value would re-encode the ``%`` and test nothing.
TRAVERSAL_TARGETS = (
    ("../../../../etc/passwd", "root:x:0:0"),
    ("..%2f..%2f..%2f..%2fetc%2fpasswd", "root:x:0:0"),
    ("../../../../etc/hostname", ""),
)

#: An export the image seeds into the archive root. Reading it proves the archive
#: root exists and the handler can serve from it, which is what makes a refusal
#: meaningful: on a target with no archive directory every traversal fails at the
#: missing directory component and a vulnerable build looks identical to a fixed
#: one, so that target is inconclusive rather than secure.
CONTROL_EXPORT = "dast-control.txt"


@probe(
    finding_id="DAST-PATH-TRAVERSAL-EXPORT",
    title="Export archive serves a file from outside the archive root",
    severity=Severity.CRITICAL,
    owasp="API1:2023 Broken Object Level Authorization",
    cwe="CWE-22",
    service="document-service",
    remediation=(
        "Resolve the requested path and require the result to stay inside the resolved "
        "archive root before opening it; reject anything else as a 404."
    ),
)
def path_traversal_export(ctx: ScanContext) -> Result:
    """Attacker asks the export route for a path that climbs out of the archive."""
    self = path_traversal_export.probe
    # Control first: FastAPI answers a missing required query parameter with 422, so a
    # 404 here means the route does not exist on this target and every traversal 404
    # below would be the router refusing, not the handler defending.
    control = ctx.get("/api/v1/documents/exports", identity=ctx.attacker)
    if control.status_code != 422:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the export route is not present on this target (control returned "
            f"{control.status_code}, expected 422 for a missing name)",
            [Evidence.from_response(control)],
        )

    legitimate = ctx.get(
        f"/api/v1/documents/exports?name={CONTROL_EXPORT}", identity=ctx.attacker
    )
    if legitimate.status_code != 200:
        return self.result(
            Verdict.INCONCLUSIVE,
            f"the export archive served nothing for the seeded control export "
            f"{CONTROL_EXPORT!r} (status {legitimate.status_code}), so a refused traversal "
            "cannot be told apart from an archive root that does not exist",
            [Evidence.from_response(legitimate)],
        )

    evidence: list[Evidence] = []
    for name, signature in TRAVERSAL_TARGETS:
        response = ctx.get(
            f"/api/v1/documents/exports?name={name}", identity=ctx.attacker
        )
        if unavailable(response):
            return self.result(
                Verdict.INCONCLUSIVE,
                f"the export route was unavailable or throttled (status {response.status_code})",
                [Evidence.from_response(response)],
            )
        served = response.status_code == 200 and (
            signature in response.text if signature else bool(response.text.strip())
        )
        if served:
            evidence.append(Evidence.from_response(response, note=f"name={name!r} was served"))
            return self.result(
                Verdict.VULNERABLE,
                f"name={name!r} returned a file from outside the export archive",
                evidence,
            )
    return self.result(Verdict.SECURE, "every traversal attempt was refused")

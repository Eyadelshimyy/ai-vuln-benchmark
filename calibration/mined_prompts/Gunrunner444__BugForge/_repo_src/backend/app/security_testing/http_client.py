"""HTTP client that cannot send traffic without ScopeGuard + SafetyController.

Redirects are disabled by default. Each Location is normalized and
re-authorized (ScopeGuard + SafetyController) before it is followed.
The first authorized URL never covers the rest of a redirect chain.

Redirect method policy (documented):
- 303 → GET, body dropped
- 307/308 → preserve method and body
- 301/302 → convert to GET and drop the body (historical user-agent policy)

Cross-origin redirects strip Authorization, Cookie, proxy credentials, and
other sensitive headers. Same-origin (scheme + host + port) may keep them.

Live DNS: the addresses authorized at ScopeGuard time are confirmed immediately
before connect. HTTP connections pin to the authorized IP. Live HTTPS to a
hostname still uses the stack resolver after a confirm-pin check; a residual
TOCTOU window is documented and fail-closed when the confirmed set diverges.
Lab mode is exempt.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import httpx

from app.domain.http import HttpBodyMeta, HttpExchange, HttpHeader
from app.security_testing.errors import AuthorizationDeniedError, SafetyLimitExceededError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.secrets import redact_exchange, redact_text
from app.security_testing.target import TargetNormalizer, try_ip

if TYPE_CHECKING:
    from app.security_testing.engine import SecurityTestEngine

_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-access-token",
    }
)


class GatedHttpClient:
    """The only HTTP sender used by fuzzing, API tests, and PoC requests."""

    def __init__(self, engine: SecurityTestEngine, *, tool: str) -> None:
        self._engine = engine
        self._tool = tool
        self._normalizer = TargetNormalizer()

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        content: str | bytes | None = None,
        timeout: float | None = None,
        destructive: bool = False,
        active: bool = True,
        follow_redirects: bool = False,
    ) -> HttpExchange | ToolExecutionResult:
        payload: str | bytes | None = content if content is not None else b""
        payload_bytes = len(payload.encode("utf-8") if isinstance(payload, str) else payload or b"")
        limits = self._engine.safety.limits
        timeout_s = timeout if timeout is not None else limits.timeout_seconds
        current = url
        hop_method = method.upper()
        hop_headers = dict(headers or {})
        remaining = limits.max_redirects if follow_redirects else 0
        seen: set[str] = set()
        response: httpx.Response | None = None
        last_decision_reason = ""
        hops = 0

        while True:
            decision = self._engine.authorize(
                current,
                method=hop_method,
                tool=self._tool if hops == 0 else f"{self._tool}_redirect",
                active=active,
                payload_bytes=payload_bytes if payload else 0,
                destructive=destructive,
            )
            last_decision_reason = decision.reason
            if not decision.allowed:
                self._engine.audit.record(
                    project=self._engine.project_id,
                    target=current,
                    scope_decision=decision.reason,
                    tool=self._tool,
                    action=f"{hop_method} denied" if hops == 0 else "redirect denied",
                    result="denied",
                    human_approval=decision.approval_state,
                )
                raise AuthorizationDeniedError(decision.reason, target=current, tool=self._tool)

            if self._engine.safety.dry_run or decision.dry_run:
                self._engine.audit.record(
                    project=self._engine.project_id,
                    target=current,
                    scope_decision=decision.reason,
                    tool=self._tool,
                    action=f"{hop_method} dry-run",
                    result="dry_run",
                )
                return ToolExecutionResult(
                    tool=self._tool,
                    state=ToolExecutionState.DRY_RUN,
                    detail=f"Would send {hop_method} {current}",
                )

            try:
                self._engine.rate_limiter.require()
            except SafetyLimitExceededError as exc:
                self._engine.audit.record(
                    project=self._engine.project_id,
                    target=current,
                    scope_decision=decision.reason,
                    tool=self._tool,
                    action=f"{hop_method} rate-limited",
                    rate_limit_decision=str(exc),
                    result="rate_limited",
                )
                raise

            connect_url, connect_headers = self._pin_connection(current, hop_headers)

            self._engine.safety.acquire(current)
            try:
                async with httpx.AsyncClient(
                    timeout=timeout_s,
                    follow_redirects=False,
                    max_redirects=0,
                ) as client:
                    response = await client.request(
                        hop_method,
                        connect_url,
                        headers=connect_headers,
                        content=payload if payload else None,
                    )
            except httpx.TimeoutException as exc:
                return ToolExecutionResult(
                    tool=self._tool, state=ToolExecutionState.TIMEOUT, detail=str(exc)
                )
            except httpx.HTTPError as exc:
                return ToolExecutionResult(
                    tool=self._tool, state=ToolExecutionState.NETWORK_FAILURE, detail=str(exc)
                )
            finally:
                self._engine.safety.release()

            assert response is not None
            if not _is_redirect(response):
                break
            location = response.headers.get("location")
            if not location:
                break
            nxt = self._normalize_redirect(current, location)
            if nxt in seen or current in seen:
                return ToolExecutionResult(
                    tool=self._tool,
                    state=ToolExecutionState.EXECUTION_ERROR,
                    detail=f"Redirect loop involving {current}",
                )
            seen.add(current)
            if remaining <= 0:
                if follow_redirects:
                    return ToolExecutionResult(
                        tool=self._tool,
                        state=ToolExecutionState.EXECUTION_ERROR,
                        detail="Maximum redirects exceeded",
                    )
                break
            remaining -= 1
            hops += 1
            hop_method, payload = _redirect_method_and_body(
                response.status_code, hop_method, payload
            )
            payload_bytes = len(
                payload.encode("utf-8") if isinstance(payload, str) else payload or b""
            )
            hop_headers = _headers_for_redirect(current, nxt, hop_headers)
            current = nxt

        assert response is not None
        body = response.content[: limits.max_response_bytes]
        truncated = len(response.content) > limits.max_response_bytes
        original_payload = content if content is not None else b""
        exchange = HttpExchange(
            method=method.upper(),
            url=str(response.url),
            request_headers=tuple(HttpHeader(name=k, value=v) for k, v in (headers or {}).items()),
            request_body=original_payload.decode("utf-8", "replace")
            if isinstance(original_payload, bytes)
            else original_payload or None,
            response_status=response.status_code,
            response_headers=tuple(
                HttpHeader(name=k, value=v) for k, v in response.headers.items()
            ),
            response_body=body.decode("utf-8", "replace"),
            response_body_meta=HttpBodyMeta(
                content_type=response.headers.get("content-type"),
                byte_length=len(response.content),
                truncated=truncated,
            ),
            elapsed_ms=(response.elapsed.total_seconds() * 1000 if response.elapsed else None),
            source_tool=self._tool,
            scope_decision=last_decision_reason,
        )
        redacted = redact_exchange(exchange)
        self._engine.audit.record(
            project=self._engine.project_id,
            target=url,
            scope_decision=last_decision_reason,
            tool=self._tool,
            action=f"{method.upper()} {url}",
            request_id=redacted.request_id,
            result=f"status={redacted.response_status}",
        )
        return redacted

    def _pin_connection(self, url: str, headers: dict[str, str]) -> tuple[str, dict[str, str]]:
        """Confirm authorized DNS and pin HTTP to that IP. Lab is exempt."""
        if self._engine.session.mode.value != "live":
            return url, dict(headers)
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return url, dict(headers)
        if try_ip(hostname) is not None:
            return url, dict(headers)
        decision = self._engine.dns.authorize(
            url, lab_mode=False, lab_hosts=self._engine.session.scope.lab_hosts
        )
        if not decision.allowed:
            raise AuthorizationDeniedError(decision.reason, target=url, tool=self._tool)
        confirmed = tuple(self._engine.dns.resolver(hostname))
        overlap = [item for item in confirmed if item in decision.addresses]
        if decision.addresses and not overlap:
            raise AuthorizationDeniedError(
                "DNS rebinding detected: resolved addresses changed after authorization",
                target=url,
                tool=self._tool,
            )
        pin = overlap[0] if overlap else (decision.addresses[0] if decision.addresses else "")
        if not pin:
            raise AuthorizationDeniedError(
                "Live connection refused: no pinned address",
                target=url,
                tool=self._tool,
            )
        if (parsed.scheme or "https").lower() == "https":
            # Residual TOCTOU: httpx will resolve HTTPS hostnames again. We fail
            # closed if the confirmed set diverged; we cannot IP-pin TLS SNI
            # with the current HTTP stack without disabling certificate checks.
            return url, dict(headers)
        netloc = pin if parsed.port is None else f"{pin}:{parsed.port}"
        pinned = parsed._replace(netloc=netloc).geturl()
        outgoing = dict(headers)
        outgoing.setdefault(
            "Host", hostname if parsed.port is None else f"{hostname}:{parsed.port}"
        )
        return pinned, outgoing

    def _normalize_redirect(self, current: str, location: str) -> str:
        joined = urljoin(current, location.strip())
        try:
            normalized = self._normalizer.normalize(joined, default_scheme=_scheme_of(current))
            return normalized.url
        except ValueError as exc:
            raise AuthorizationDeniedError(
                f"Malformed redirect target {redact_text(location)} ({exc})",
                target=joined,
                tool=self._tool,
            ) from exc


def _is_redirect(response: httpx.Response) -> bool:
    return response.status_code in {301, 302, 303, 307, 308}


def _scheme_of(url: str) -> str:
    if url.startswith("http://"):
        return "http"
    return "https"


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    port = parsed.port
    if port is None:
        port = 80 if scheme == "http" else 443
    return scheme, host, port


def _same_origin(left: str, right: str) -> bool:
    return _origin(left) == _origin(right)


def _headers_for_redirect(current: str, nxt: str, headers: Mapping[str, str]) -> dict[str, str]:
    outgoing = dict(headers)
    if _same_origin(current, nxt):
        return outgoing
    stripped = {
        key: value for key, value in outgoing.items() if key.lower() not in _SENSITIVE_HEADERS
    }
    return stripped


def _redirect_method_and_body(
    status: int, method: str, body: str | bytes | None
) -> tuple[str, str | bytes | None]:
    if status in {307, 308}:
        return method, body
    # 303 always GET. 301/302 use the documented historical GET conversion.
    return "GET", None

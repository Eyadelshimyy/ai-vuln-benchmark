"""HackerOne Hacker API HTTP client.

Isolated JSON:API + Basic auth. Rate-limited independently of target scanning.
Requests may only target the configured HackerOne API origin. Absolute URLs
to other hosts are rejected before Basic authentication is attached.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse

import httpx

from app.adapters.hackerone.credentials import HackerOneCredentials
from app.adapters.hackerone.errors import (
    HackerOneAuthError,
    HackerOneError,
    HackerOneIdentityVerificationError,
    HackerOneRateLimitError,
    HackerOneUrlRejectedError,
)
from app.adapters.hackerone.models import node_id, parse_hackerone_id
from app.security_testing.secrets import redact_text

_IDENTITY_HINTS = (
    "identity verification",
    "verify your identity",
    "id verification",
    "kyc",
)
_UNSAFE_POST_RETRY_MARKERS = ("/hackers/reports", "/hackers/report_intents")


class HackerOneApiClient:
    def __init__(
        self,
        credentials: HackerOneCredentials,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
        min_interval_seconds: float = 0.25,
        max_retries: int = 3,
    ) -> None:
        self._credentials = credentials
        self._base = credentials.base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries
        self._last_call = 0.0
        self._transport = transport
        self.calls: list[tuple[str, str]] = []
        self._origin = _origin_parts(self._base)
        self.last_status: int | None = None
        self.last_sanitized_body: dict[str, Any] | None = None

    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {
            "base_url": self._base,
            "auth": self._credentials.auth_tuple(),
            "timeout": self._timeout,
            "headers": {"Accept": "application/json", "Content-Type": "application/json"},
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        relative, extra_params = self._authorized_relative(path)
        merged = dict(extra_params)
        if params:
            merged.update({str(key): value for key, value in params.items()})
        self.calls.append((method.upper(), _path_only(urljoin(self._base, relative))))
        attempt = 0
        while True:
            self._wait()
            attempt += 1
            try:
                with self._client() as client:
                    response = client.request(
                        method.upper(),
                        relative,
                        json=json_body,
                        params=merged or None,
                    )
            except httpx.TimeoutException as exc:
                raise HackerOneError("HackerOne API timed out", code="timeout") from exc
            except httpx.HTTPError as exc:
                raise HackerOneError("HackerOne API network failure", code="network") from exc
            if (
                response.status_code == 429
                and attempt <= self._max_retries
                and _retry_safe(method, relative)
            ):
                retry_after = _retry_after(response)
                time.sleep(retry_after)
                continue
            return self._handle(response)

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", path, params=params)

    def post(self, path: str, *, json_body: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("POST", path, json_body=json_body)

    def patch(self, path: str, *, json_body: Mapping[str, Any]) -> dict[str, Any]:
        return self.request("PATCH", path, json_body=json_body)

    def delete(self, path: str) -> dict[str, Any]:
        return self.request("DELETE", path)

    def paginate(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        continue_with_id_gt: bool = True,
        page_cap: int = 100,
    ) -> list[Any]:
        items, _meta = self.paginate_with_meta(
            path, params=params, continue_with_id_gt=continue_with_id_gt, page_cap=page_cap
        )
        return items

    def paginate_with_meta(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        continue_with_id_gt: bool = True,
        page_cap: int = 100,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Paginate a JSON:API collection and report whether the sync is complete.

        HackerOne documents that ordinary ``links.next`` pagination can stop at
        10,000 structured scopes. When that happens, continue with
        ``filter[id__gt]`` using the last seen numeric id.
        ``scope_sync_complete`` is true only when every page was fetched.
        """
        items: list[Any] = []
        seen: set[str] = set()
        next_path: str | None = path
        query = dict(params or {})
        last_numeric = 0
        last_id = ""
        pages = 0
        last_page_count = 0
        error: str | None = None
        continuation_state = "links.next"
        had_next = False
        try:
            while next_path and pages < page_cap:
                payload = self.get(next_path, params=query or None)
                page_items = _collection(payload)
                last_page_count = len(page_items)
                for row in page_items:
                    ident = node_id(row) or str(id(row))
                    if ident in seen:
                        continue
                    seen.add(ident)
                    items.append(row)
                    numeric = parse_hackerone_id(ident)
                    if numeric is not None and numeric >= last_numeric:
                        last_numeric = numeric
                        last_id = ident
                pages += 1
                links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
                nxt = links.get("next") if isinstance(links, dict) else None
                if not nxt:
                    next_path = None
                    query = {}
                    break
                had_next = True
                next_path, extra = self._authorized_relative(str(nxt))
                query = dict(extra)
            hit_cap = bool(next_path) and pages >= page_cap
            truncated = hit_cap or (had_next and pages >= page_cap and last_page_count > 0)
            continued = False
            if continue_with_id_gt and last_numeric and (truncated or hit_cap):
                continuation_state = "filter[id__gt]"
                continued, last_numeric, last_id = self._continue_id_gt(
                    path, params or {}, items, seen, last_numeric, last_id
                )
                if continued:
                    truncated = False
                    hit_cap = False
            complete = error is None and not truncated and not hit_cap
            if truncated or hit_cap:
                continuation_state = f"incomplete:pages={pages}:last_id={last_id}"
            elif continued:
                continuation_state = "complete:id_gt"
            else:
                continuation_state = "complete"
        except HackerOneError as exc:
            error = str(exc)
            complete = False
            continuation_state = f"error:{exc.code or 'error'}:last_id={last_id}"
        return items, {
            "count": len(items),
            "pages_fetched": pages,
            "last_scope_id": last_id or None,
            "last_numeric_id": last_numeric or None,
            "continuation_state": continuation_state,
            "scope_sync_complete": complete,
            "error": error,
        }

    def create_report_intent(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.post("hackers/report_intents", json_body=dict(payload))

    def get_report_intent(self, intent_id: str) -> dict[str, Any]:
        return self.get(f"hackers/report_intents/{intent_id}")

    def patch_report_intent(self, intent_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.patch(f"hackers/report_intents/{intent_id}", json_body=dict(payload))

    def delete_report_intent(self, intent_id: str) -> dict[str, Any]:
        return self.delete(f"hackers/report_intents/{intent_id}")

    def submit_report_intent(self, intent_id: str) -> dict[str, Any]:
        return self.post(f"hackers/report_intents/{intent_id}/submit", json_body={})

    def list_report_intent_attachments(self, intent_id: str) -> dict[str, Any]:
        return self.get(f"hackers/report_intents/{intent_id}/attachments")

    def create_report_intent_attachment(
        self, intent_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.post(f"hackers/report_intents/{intent_id}/attachments", json_body=dict(payload))

    def delete_report_intent_attachment(self, intent_id: str, attachment_id: str) -> dict[str, Any]:
        return self.delete(f"hackers/report_intents/{intent_id}/attachments/{attachment_id}")

    def list_hacker_reports(self, *, params: Mapping[str, Any] | None = None) -> list[Any]:
        return self.paginate("hackers/reports", params=params, continue_with_id_gt=False)

    def _continue_id_gt(
        self,
        path: str,
        params: Mapping[str, Any],
        items: list[Any],
        seen: set[str],
        last_numeric: int,
        last_id: str,
    ) -> tuple[bool, int, str]:
        current = last_numeric
        current_id = last_id
        progressed_any = False
        while True:
            query = dict(params)
            query["filter[id__gt]"] = current
            payload = self.get(path, params=query)
            page_items = _collection(payload)
            if not page_items:
                return progressed_any, current, current_id
            progressed = False
            for row in page_items:
                ident = node_id(row) or str(id(row))
                if ident in seen:
                    continue
                seen.add(ident)
                items.append(row)
                numeric = parse_hackerone_id(ident)
                if numeric is not None and numeric > current:
                    current = numeric
                    current_id = ident
                    progressed = True
                    progressed_any = True
            if not progressed:
                return progressed_any, current, current_id

    def _authorized_relative(self, path: str) -> tuple[str, dict[str, str]]:
        """Return a relative path + query that is locked to the configured origin.

        Absolute URLs are accepted only when they match scheme, hostname, port,
        and API base path of the configured HackerOne origin. Anything else is
        rejected *before* Basic auth is attached.
        """
        raw = (path or "").strip()
        if not raw:
            raise HackerOneUrlRejectedError("HackerOne request path is empty")
        if raw.startswith("//") or raw.startswith("\\"):
            raise HackerOneUrlRejectedError("HackerOne client rejects protocol-relative URLs")
        if "://" in raw or raw.lower().startswith("http"):
            parsed = urlparse(raw)
            self._assert_origin(parsed)
            relative = _strip_base_path(parsed.path, self._origin["base_path"])
            extra = {key: value for key, value in parse_qsl(parsed.query, keep_blank_values=True)}
            return relative.lstrip("/"), extra
        if ".." in raw.split("/"):
            raise HackerOneUrlRejectedError("HackerOne client rejects path traversal")
        return raw.lstrip("/"), {}

    def _assert_origin(self, parsed: Any) -> None:
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()
        default_port = 443 if scheme == "https" else 80
        port = parsed.port or default_port
        if scheme != self._origin["scheme"]:
            raise HackerOneUrlRejectedError(
                "HackerOne client rejected URL with a non-configured scheme"
            )
        if hostname != self._origin["hostname"]:
            raise HackerOneUrlRejectedError(
                "HackerOne client rejected URL with a non-configured hostname"
            )
        if port != self._origin["port"]:
            raise HackerOneUrlRejectedError(
                "HackerOne client rejected URL with a non-configured port"
            )
        path = parsed.path or ""
        base_path = self._origin["base_path"]
        if base_path and not path.startswith(base_path):
            raise HackerOneUrlRejectedError(
                "HackerOne client rejected URL outside the configured API base path"
            )

    def _wait(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _handle(self, response: httpx.Response) -> dict[str, Any]:
        status = response.status_code
        text = redact_text(response.text or "")
        body: dict[str, Any]
        try:
            parsed = response.json()
            body = parsed if isinstance(parsed, dict) else {"data": parsed}
        except ValueError:
            body = {"raw": text[:500]}
        self.last_status = status
        self.last_sanitized_body = _sanitize_response_body(body)
        errors = body.get("errors")
        message = _error_message(errors) or text[:300] or f"HTTP {status}"
        lowered = message.lower()
        if status in {401}:
            raise HackerOneAuthError(
                "Invalid HackerOne API credentials", status_code=401, code="invalid_credentials"
            )
        if status == 403 and any(hint in lowered for hint in _IDENTITY_HINTS):
            raise HackerOneIdentityVerificationError()
        if status == 403:
            raise HackerOneAuthError(
                "HackerOne API forbidden. The account may lack permission for this action.",
                status_code=403,
                code="forbidden",
            )
        if status == 404:
            raise HackerOneError("HackerOne resource not found", status_code=404, code="not_found")
        if status == 409:
            raise HackerOneError(
                "HackerOne reported a conflict (possible duplicate)",
                status_code=409,
                code="conflict",
            )
        if status == 422:
            code = _classify_unprocessable(message)
            raise HackerOneError(
                f"HackerOne rejected the payload: {message}", status_code=422, code=code
            )
        if status == 400:
            raise HackerOneError(
                f"HackerOne request was invalid: {message}", status_code=400, code="bad_request"
            )
        if status == 429:
            raise HackerOneRateLimitError(
                "HackerOne API rate limited", retry_after=_retry_after(response)
            )
        if status >= 500:
            raise HackerOneError(
                "HackerOne API server error", status_code=status, code="server_error"
            )
        if status >= 400:
            raise HackerOneError(message, status_code=status, code="error")
        return body


def _sanitize_response_body(body: dict[str, Any]) -> dict[str, Any]:
    blocked = {
        "authorization",
        "token",
        "password",
        "cookie",
        "api_token",
        "secret",
        "hackerone_api_token",
    }
    cleaned: dict[str, Any] = {}
    for key, value in body.items():
        lowered = str(key).lower()
        if any(part in lowered for part in blocked):
            continue
        if isinstance(value, dict):
            cleaned[key] = _sanitize_response_body(value)
        elif isinstance(value, str):
            cleaned[key] = redact_text(value)
        else:
            cleaned[key] = value
    return cleaned


def _retry_safe(method: str, path: str) -> bool:
    if method.upper() != "GET":
        # Never blindly retry report creation.
        lowered = path.lower()
        if any(marker in lowered for marker in _UNSAFE_POST_RETRY_MARKERS):
            return False
        return False
    return True


def _retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After") or response.headers.get("retry-after") or "1"
    try:
        return max(0.5, float(raw))
    except ValueError:
        return 1.0


def _error_message(errors: object) -> str:
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            detail = first.get("detail") or first.get("title") or first.get("code")
            return str(detail or "")
        return str(first)
    if isinstance(errors, dict):
        return str(errors.get("detail") or errors.get("title") or "")
    return ""


def _classify_unprocessable(message: str) -> str:
    lowered = message.lower()
    if "severity" in lowered:
        return "severity_required"
    if "weakness" in lowered:
        return "weakness_invalid"
    if "structured_scope" in lowered or "structured scope" in lowered:
        return "structured_scope_invalid"
    if "identity" in lowered:
        return "identity_verification_required"
    if "permission" in lowered or "forbidden" in lowered:
        return "permission_denied"
    if "duplicate" in lowered or "conflict" in lowered:
        return "duplicate"
    return "unprocessable"


def _path_only(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path or url


def _collection(payload: Mapping[str, Any]) -> list[Any]:
    data = payload.get("data")
    if isinstance(data, list):
        return list(data)
    if data is None:
        return []
    return [data]


def _origin_parts(base_url: str) -> dict[str, Any]:
    parsed = urlparse(base_url)
    scheme = (parsed.scheme or "https").lower()
    hostname = (parsed.hostname or "").lower()
    default_port = 443 if scheme == "https" else 80
    port = parsed.port or default_port
    base_path = (parsed.path or "").rstrip("/") or ""
    return {
        "scheme": scheme,
        "hostname": hostname,
        "port": port,
        "base_path": base_path,
    }


def _strip_base_path(path: str, base_path: str) -> str:
    if base_path and path.startswith(base_path):
        return path[len(base_path) :] or "/"
    return path

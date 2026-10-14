"""Chat-completions transport for OpenAI-compatible endpoints.

Covers every endpoint the comparison targets — OpenRouter (managed and
open-weight candidates behind one key) and a self-served vLLM endpoint — because
all of them accept the same ``response_format`` JSON Schema and return the same
message shape.

Cost is read from whatever the endpoint reports (OpenRouter returns
``usage.cost`` when usage accounting is requested) and is never estimated from
token counts, so no price table lives in this repo.
"""

from __future__ import annotations

from typing import Any

import httpx

from field_notes.adapters.base import AdapterUnavailableError, TokenUsage
from field_notes.adapters.schema_output import TransportResponse

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
SCHEMA_NAME = "field_note_turn_result"


class OpenAICompatibleTransport:
    """Calls ``POST {base_url}/chat/completions`` once per turn.

    Retries are deliberately narrow: a transport-level or 5xx/429 failure is
    retried with a fixed backoff, while a schema-invalid payload is returned
    as-is so the suite scores it instead of hiding it behind a retry.
    """

    def __init__(
        self,
        *,
        default_base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
        max_attempts: int = 3,
        backoff_seconds: float = 3.0,
        strict_schema: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._default_base_url = default_base_url
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._backoff = backoff_seconds
        self._strict_schema = strict_schema
        self._headers = dict(headers or {})

    def complete(
        self,
        *,
        request_id: str,
        base_url: str | None,
        api_key: str | None,
        model: str,
        system: str,
        user_content: str,
        json_schema: dict[str, Any],
        parameters: dict[str, Any],
    ) -> TransportResponse:
        """Return the candidate's raw JSON payload plus reported cost and usage."""
        del request_id
        if api_key is None:
            raise AdapterUnavailableError("no API key resolved for a live transport")
        url = f"{(base_url or self._default_base_url).rstrip('/')}/chat/completions"
        body = _request_body(
            model=model,
            system=system,
            user_content=user_content,
            json_schema=json_schema,
            parameters=parameters,
            strict_schema=self._strict_schema,
        )
        headers = {"Authorization": f"Bearer {api_key}", **self._headers}
        payload = self._post(url, headers, body)
        return _to_response(payload)

    def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = httpx.post(  # noqa: S113 - timeout supplied below
                    url, headers=headers, json=body, timeout=self._timeout
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableStatusError(response.status_code, response.text[:200])
                response.raise_for_status()
                decoded = response.json()
                if not isinstance(decoded, dict):
                    raise AdapterUnavailableError("endpoint returned a non-object response")
                return decoded
            except (httpx.HTTPError, _RetryableStatusError) as exc:
                last_error = exc
                if attempt == self._max_attempts:
                    break
                _sleep(self._backoff * attempt)
        raise AdapterUnavailableError(
            f"endpoint call failed after {self._max_attempts} attempts: {last_error}"
        )


class _RetryableStatusError(Exception):
    """A status the endpoint may recover from on a later attempt."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def _request_body(
    *,
    model: str,
    system: str,
    user_content: str,
    json_schema: dict[str, Any],
    parameters: dict[str, Any],
    strict_schema: bool,
) -> dict[str, Any]:
    """Build the chat-completions body, holding decoding settings as configured."""
    extra = dict(parameters)
    max_output_tokens = extra.pop("max_output_tokens", None)
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "strict": strict_schema,
                "schema": json_schema,
            },
        },
        "usage": {"include": True},
    }
    if max_output_tokens is not None:
        body["max_tokens"] = max_output_tokens
    body.update(extra)
    return body


def _to_response(payload: dict[str, Any]) -> TransportResponse:
    """Extract the message content and the reported cost/usage."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AdapterUnavailableError(f"endpoint returned no choices: {str(payload)[:200]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    raw = content if isinstance(content, str) else ""
    usage_payload = payload.get("usage")
    usage: TokenUsage | None = None
    cost: float | None = None
    if isinstance(usage_payload, dict):
        usage = TokenUsage(
            input_tokens=_as_int(usage_payload.get("prompt_tokens")),
            output_tokens=_as_int(usage_payload.get("completion_tokens")),
            total_tokens=_as_int(usage_payload.get("total_tokens")),
        )
        cost = _as_float(usage_payload.get("cost"))
    return TransportResponse(raw=raw, cost_usd=cost, usage=usage)


def _as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None

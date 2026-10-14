"""Adapter shell for schema-constrained candidates.

One adapter covers both intended targets, because both accept the same
Pydantic-derived JSON Schema and return the same typed contract:

* managed structured-output APIs (Anthropic tool use, OpenAI Structured Outputs,
  Gemini structured output, Bedrock validated JSON schemas);
* an OpenAI-compatible endpoint served by vLLM for open-weight candidates.

Transport is injected, so this module contains no HTTP client, no provider SDK, and
no credential handling. Endpoint and model configuration is read from environment
variable *names* supplied by a manifest — never from values committed to the repo.
Until a transport is supplied, running a turn raises
:class:`AdapterUnavailableError`; nothing here can reach a paid API by accident.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from field_notes.adapters.base import (
    AdapterUnavailableError,
    TokenUsage,
    TurnOutcome,
    TurnRequest,
)
from field_notes.prompt import build_user_content, system_prompt, turn_result_json_schema
from field_notes.schema import TurnResult


class TransportResponse(BaseModel):
    """What one candidate call returned.

    ``cost_usd`` and ``usage`` are how a real run populates the scorecard's operational
    slots. A transport reports what its provider billed or metered; the suite never
    estimates cost from token counts, since price tables belong outside this repo.
    """

    model_config = ConfigDict(extra="forbid")

    raw: str
    cost_usd: float | None = None
    usage: TokenUsage | None = None


class SchemaCompletionTransport(Protocol):
    """Minimal call surface a real run must implement."""

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
        """Return the candidate's raw JSON text plus whatever cost/usage it reported."""
        ...


class EndpointConfig(BaseModel):
    """External, secret-free endpoint description."""

    model_config = ConfigDict(extra="forbid")

    base_url_env: str | None = None
    api_key_env: str | None = None

    def resolve(self) -> tuple[str | None, str | None]:
        """Read the configured environment variables, requiring any that are named."""
        base_url = os.environ.get(self.base_url_env) if self.base_url_env else None
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if self.base_url_env and not base_url:
            raise AdapterUnavailableError(f"environment variable {self.base_url_env} is not set")
        if self.api_key_env and not api_key:
            raise AdapterUnavailableError(f"environment variable {self.api_key_env} is not set")
        return base_url, api_key


class SchemaOutputAdapter:
    """Drives a schema-constrained candidate through the runner contract."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        endpoint: EndpointConfig | None = None,
        parameters: dict[str, Any] | None = None,
        transport: SchemaCompletionTransport | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._endpoint = endpoint or EndpointConfig()
        self._parameters = dict(parameters or {})
        self._transport = transport

    @property
    def name(self) -> str:
        return self._name

    def run_turn(self, request: TurnRequest) -> TurnOutcome:
        if self._transport is None:
            raise AdapterUnavailableError(
                f"candidate {self._name!r} has no transport; this suite ships only "
                "deterministic local adapters and never calls a paid API"
            )
        base_url, api_key = self._endpoint.resolve()
        user_content = build_user_content(request.prior_fields, request.history, request.transcript)
        started = time.perf_counter()
        response = self._transport.complete(
            request_id=f"{request.case_id}:{request.turn_index}",
            base_url=base_url,
            api_key=api_key,
            model=self._model,
            system=system_prompt(),
            user_content=user_content,
            json_schema=turn_result_json_schema(),
            parameters=self._parameters,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return _parse(response, latency_ms)


def _parse(response: TransportResponse, latency_ms: float) -> TurnOutcome:
    """Turn a transport response into an outcome, keeping cost/usage either way.

    A schema failure still cost money, so the operational slots are populated even when
    the payload is unusable.
    """
    try:
        payload = json.loads(response.raw)
        result = TurnResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        return TurnOutcome(
            result=None,
            schema_valid=False,
            error=str(exc),
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
            usage=response.usage,
            raw_response=response.raw,
        )
    return TurnOutcome(
        result=result,
        schema_valid=True,
        latency_ms=latency_ms,
        cost_usd=response.cost_usd,
        usage=response.usage,
        raw_response=response.raw,
    )


class RecordedTransport:
    """Replays responses keyed by ``case_id:turn_index``.

    Lets a future real run's captured outputs — including the cost and usage it
    reported — be re-scored offline, and lets tests exercise the parsing path with no
    network access. Bare strings are accepted for the common case of replaying only the
    payload.
    """

    def __init__(
        self,
        responses: dict[str, TransportResponse | str],
        fallback: TransportResponse | str | None = None,
    ) -> None:
        self._responses = {key: _as_response(value) for key, value in responses.items()}
        self._fallback = None if fallback is None else _as_response(fallback)

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
        del base_url, api_key, model, system, user_content, json_schema, parameters
        recorded = self._responses.get(request_id, self._fallback)
        if recorded is None:
            raise AdapterUnavailableError(f"no recorded response for {request_id!r}")
        return recorded


def _as_response(value: TransportResponse | str) -> TransportResponse:
    return value if isinstance(value, TransportResponse) else TransportResponse(raw=value)


class TransportParameters(BaseModel):
    """Decoding settings held fixed across candidates in a real run."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = 0.0
    max_output_tokens: int = 4096
    extra: dict[str, Any] = Field(default_factory=dict)

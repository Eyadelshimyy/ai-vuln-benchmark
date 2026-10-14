"""A replaceable completion transport beneath shared parsing and tool iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from litellm import ModelResponse
from litellm.utils import type_to_response_format_param
from pydantic import BaseModel, JsonValue, TypeAdapter

_PAYLOAD = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True)
class CompletionRequest:
    """One actual provider attempt, including its full bounded request parameters.

    This is below tool iteration and retry, not an observer. A transport failure
    prevents the request from succeeding and is never swallowed as telemetry.
    """

    model: str
    messages: list[dict[str, Any]]
    parameters: dict[str, Any]

    def as_json(self) -> dict[str, JsonValue]:
        """Serialize a completion for a typed remote or governed transport.

        Native Pydantic structured-output requests become the equivalent explicit
        JSON schema; arbitrary Python objects are refused, not stringified.
        """
        parameters = dict(self.parameters)
        if {"model", "messages"} & parameters.keys():
            raise ValueError("Completion parameters cannot replace the model or messages.")
        schema = parameters.get("response_format")
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            parameters["response_format"] = type_to_response_format_param(schema)
        return _PAYLOAD.validate_python({
            "model": self.model,
            "messages": self.messages,
            **parameters,
        })


class CompletionTransport(Protocol):
    """Execute a single non-streaming completion and return its provider response.

    Implementations can admit and settle requests at an external boundary.
    They must not retry: the client owns retry and calls the transport anew for
    each attempt. Streaming is deliberately unavailable with this transport;
    it must never fall through to an ungoverned direct provider request.
    """

    async def complete(self, request: CompletionRequest) -> ModelResponse: ...


__all__ = ["CompletionRequest", "CompletionTransport"]

"""Structured-output helpers built on litellm's native ``response_format``.

Modern litellm supports passing a Pydantic ``BaseModel`` subclass to the
``response_format`` parameter of ``litellm.acompletion``. Providers that
support JSON-mode (OpenAI, Anthropic, Mistral, Gemini, …) will return a
JSON document validating against the schema; providers that don't will
raise an exception, which we surface as
:class:`StructuredOutputUnsupportedError`.

This module deliberately does *not* re-implement schema-to-prompt
conversion. Hand-rolled "please respond with JSON like this…" prompts
are inferior — they don't constrain generation, only post-hoc parsing.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

import litellm
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ..exceptions import LLMError, StructuredOutputUnsupportedError

T = TypeVar("T", bound=BaseModel)


def supports_structured_output(model: str) -> bool:
    """Return whether *model* supports litellm's native ``response_format``."""
    try:
        return bool(litellm.supports_response_schema(model=model))
    except Exception:
        # Older litellm versions or unknown models — be conservative and
        # fall through to the attempt, which will raise if unsupported.
        return True


def ensure_structured_support(model: str) -> None:
    """Hard-fail when *model* cannot produce native structured outputs."""
    if not supports_structured_output(model):
        raise StructuredOutputUnsupportedError(
            f"Model {model!r} does not support native structured outputs. "
            "Choose a model with JSON-mode / response_format support."
        )


def extract_json_payload(content: str) -> str:
    """Extract a JSON document from a response that may include code fences."""
    stripped = content.strip()
    if "```json" in stripped:
        match = re.search(r"```json\s*\n(.*?)\n```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
    if "```" in stripped:
        match = re.search(r"```\s*\n(.*?)\n```", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
    return stripped


def parse_structured_content(
    content: str,
    response_model: type[T],
    *,
    error_context: str = "Failed to parse structured response",
) -> T:
    """Parse JSON *content* into an instance of *response_model*.

    Raises:
        LLMError: If *content* is not valid JSON or does not validate
            against the model's schema.
    """
    payload = extract_json_payload(content)
    try:
        data = json.loads(payload)
        return response_model.model_validate(data)
    except (json.JSONDecodeError, PydanticValidationError, ValueError) as exc:
        preview = payload[:200] if len(payload) > 200 else payload
        raise LLMError(
            f"{error_context}: {exc}\nContent preview: {preview}"
        ) from exc


__all__ = [
    "ensure_structured_support",
    "extract_json_payload",
    "parse_structured_content",
    "supports_structured_output",
]

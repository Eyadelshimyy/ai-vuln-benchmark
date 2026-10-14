"""Placeholder for an external Cursor controller. This is not a language model.

Cursor performs the reasoning. Anything that asks this object to generate
text, hypotheses, or tool plans is refused. It must not fall back to mock,
OpenAI, Anthropic, local, or MLX providers.
"""

from __future__ import annotations

import re

from app.ai.provider import (
    AICapabilities,
    CompletionRequest,
    CompletionResponse,
    DebuggingRequest,
    LLMProvider,
    ProviderResponse,
    StructuredTextResponse,
    TestGenerationResponse,
)
from app.security_testing.errors import RestrictedActivityError

CURSOR_EXTERNAL_PROVIDER = "cursor_external"
CURSOR_SELECTED_MODEL = "cursor-selected-model"
_MODEL_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")
_SECRET_HINTS = ("sk-", "api_key", "token", "secret", "bearer ")


class ExternalControllerError(RestrictedActivityError):
    """Raised when Cursor mode is asked to run a BugForge language model."""


def cursor_model_label(raw: str | None) -> str:
    """Keep a caller-supplied Cursor model id, or the explicit unknown label.

    Do not invent a vendor model name. Reject values that look like secrets.
    """

    text = (raw or "").strip()
    lowered = text.lower()
    if not text or any(hint in lowered for hint in _SECRET_HINTS):
        return CURSOR_SELECTED_MODEL
    if not _MODEL_LABEL.fullmatch(text):
        return CURSOR_SELECTED_MODEL
    return text


class ExternalControllerProvider(LLMProvider):
    """Metadata placeholder. Generation methods always refuse."""

    def __init__(self, model_name: str = CURSOR_SELECTED_MODEL) -> None:
        self._model_name = cursor_model_label(model_name)

    @property
    def provider_name(self) -> str:
        return CURSOR_EXTERNAL_PROVIDER

    @property
    def model_name(self) -> str:
        return self._model_name

    def capabilities(self) -> AICapabilities:
        return AICapabilities(
            chat=False,
            structured_output=False,
            tool_calls=False,
            thinking=False,
            thinking_can_disable=False,
            max_context_tokens=0,
            max_output_tokens=0,
            supports_local_models=False,
            text_generation=False,
            structured_generation=False,
            reasoning=False,
            notes=("external Cursor controller; BugForge does not call a model",),
        )

    async def is_available(self) -> bool:
        return False

    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        del request
        raise ExternalControllerError("cursor_external_refuses_generation")

    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        del system_prompt, user_message
        raise ExternalControllerError("cursor_external_refuses_generation")

    async def generate_structured(
        self, system_prompt: str, user_message: str
    ) -> StructuredTextResponse:
        del system_prompt, user_message
        raise ExternalControllerError("cursor_external_refuses_generation")

    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        del system_prompt, user_message
        raise ExternalControllerError("cursor_external_refuses_generation")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        del request
        raise ExternalControllerError("cursor_external_refuses_generation")

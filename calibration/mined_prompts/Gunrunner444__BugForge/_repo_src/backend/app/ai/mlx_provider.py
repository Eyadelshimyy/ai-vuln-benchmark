"""MLX local AI provider — OpenAI-compatible HTTP on a loopback server.

Default endpoint: http://127.0.0.1:8080/v1
The model name is configurable. A typical local install is
Qwen3.6-35B-A3B-8bit, but that string is not hard-coded in the engine.

Works without a cloud API key. Thinking traces are split from final content.
Structured output is parsed defensively rather than assuming native
``response_format`` support.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import httpx

from app.ai.health import AIHealthStatus, probe_openai_compatible
from app.ai.provider import (
    AICapabilities,
    AIUsage,
    CompletionRequest,
    CompletionResponse,
    DebuggingRequest,
    LLMProvider,
    ProviderResponse,
    StructuredTextResponse,
    TestGenerationResponse,
)
from app.ai.structured import StructuredParseError, extract_json_object
from app.ai.thinking import split_thinking

logger = logging.getLogger(__name__)

DEFAULT_MLX_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MLX_MODEL = "Qwen3.6-35B-A3B-8bit"


class MlxProvider(LLMProvider):
    """Local MLX OpenAI-compatible server (typically mlx_lm.server)."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_MLX_BASE_URL,
        *,
        api_key: str = "",
        max_tokens: int = 2_000,
        temperature: float = 0.1,
        timeout_seconds: int = 120,
        max_retries: int = 1,
        thinking_enabled: bool = False,
        native_json_mode: bool = False,
        max_context_tokens: int | None = 8192,
        provider_name: str = "mlx",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/") or DEFAULT_MLX_BASE_URL
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._thinking_enabled = thinking_enabled
        self._native_json_mode = native_json_mode
        self._max_context_tokens = max_context_tokens
        self._provider_name = provider_name

    @classmethod
    def from_settings(cls, settings: object) -> MlxProvider:
        from app.core.config import Settings

        assert isinstance(settings, Settings)
        model = settings.ai_model
        if not model or model == "gpt-4o-mini":
            model = settings.mlx_model
        base_url = settings.ai_base_url or settings.mlx_base_url
        return cls(
            model=model,
            base_url=base_url,
            api_key=settings.ai_api_key,
            max_tokens=settings.ai_max_output_tokens,
            temperature=settings.ai_temperature,
            timeout_seconds=max(settings.ai_timeout_seconds, 30),
            max_retries=settings.ai_max_retries,
            thinking_enabled=settings.ai_thinking_enabled,
            native_json_mode=settings.ai_native_json_mode,
            max_context_tokens=settings.ai_max_context_tokens,
            provider_name="mlx",
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    def capabilities(self) -> AICapabilities:
        return AICapabilities(
            chat=True,
            structured_output=True,
            structured_generation=True,
            text_generation=True,
            tool_calls=True,
            thinking=True,
            thinking_can_disable=True,
            reasoning=True,
            max_output_tokens=self._max_tokens,
            max_context_tokens=self._max_context_tokens,
            supports_local_models=True,
            local_execution=True,
            long_context=bool(self._max_context_tokens and self._max_context_tokens >= 16_000),
            native_json_response_format=self._native_json_mode,
            notes=(
                "Local MLX OpenAI-compatible HTTP. No cloud API key required. "
                "Thinking traces are stripped from the final answer. "
                "Structured JSON is extracted defensively. "
                "Not every OpenAI-compatible local server implements "
                "response_format or native tool_calls.",
            ),
        )

    async def discover_capabilities(self) -> AICapabilities:
        """Probe the local server; fall back to configured capabilities."""
        caps = self.capabilities()
        try:
            health = await self.health()
        except Exception:
            return caps
        if not health.reachable:
            return caps
        return caps

    async def is_available(self) -> bool:
        if not _http_url_configured(self._base_url):
            return False
        status = await self._probe()
        return status.reachable

    async def health(self) -> AIHealthStatus:
        if not _http_url_configured(self._base_url):
            return AIHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                is_local=True,
                reachable=False,
                configured=False,
                model_available=None,
                error="invalid_or_missing_base_url",
                capabilities=list(self.capabilities().advertised()),
            )
        status = await self._probe()
        status.capabilities = list(self.capabilities().advertised())
        return status

    async def _probe(self) -> AIHealthStatus:
        return await probe_openai_compatible(
            provider=self.provider_name,
            model=self.model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            is_local=True,
            configured=_http_url_configured(self._base_url),
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.monotonic()
        thinking_flag = self._thinking_enabled if request.thinking is None else request.thinking
        try:
            content, thinking, usage, native_tools = await self._chat(
                system_prompt=request.system_prompt,
                user_message=request.user_message,
                json_mode=request.json_mode,
                thinking=thinking_flag,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                tools=request.tools,
            )
        except Exception as exc:
            return CompletionResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                usage=AIUsage(),
                duration_seconds=time.monotonic() - start,
                error=_format_error(exc),
            )
        if request.json_mode:
            try:
                extract_json_object(content)
            except StructuredParseError as exc:
                return CompletionResponse(
                    content=content,
                    provider=self.provider_name,
                    model=self.model_name,
                    usage=usage,
                    duration_seconds=time.monotonic() - start,
                    thinking=thinking,
                    error=f"malformed_structured_output:{exc}",
                )
        tool_calls: tuple[Mapping[str, Any], ...] = native_tools
        if not tool_calls and request.tools:
            try:
                parsed = extract_json_object(content)
                if isinstance(parsed, dict) and (parsed.get("tool") or parsed.get("name")):
                    tool_calls = (parsed,)
            except StructuredParseError:
                tool_calls = ()
        return CompletionResponse(
            content=content,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            duration_seconds=time.monotonic() - start,
            thinking=thinking,
            tool_calls=tool_calls,
        )

    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        from app.ai.openai_provider import _parse_hypotheses
        from app.ai.prompt_builder import PromptBuilder

        prompt = PromptBuilder().build(request)
        start = time.monotonic()
        try:
            content, _thinking, usage, _native = await self._chat(
                system_prompt=prompt.system_message,
                user_message=prompt.user_message,
                json_mode=True,
                thinking=self._thinking_enabled,
            )
        except Exception as exc:
            return ProviderResponse(
                hypotheses=[],
                provider=self.provider_name,
                model=self.model_name,
                usage=AIUsage(),
                duration_seconds=time.monotonic() - start,
                error=_format_error(exc),
            )
        hypotheses, parse_error = _parse_hypotheses(content, request.max_hypotheses)
        return ProviderResponse(
            hypotheses=hypotheses,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            duration_seconds=time.monotonic() - start,
            raw_json=content,
            error=parse_error,
        )

    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        start = time.monotonic()
        try:
            content, _thinking, usage, _native = await self._chat(
                system_prompt, user_message, json_mode=True, thinking=False
            )
        except Exception as exc:
            return TestGenerationResponse(
                candidates_json="",
                provider=self.provider_name,
                model=self.model_name,
                usage=AIUsage(),
                duration_seconds=time.monotonic() - start,
                error=_format_error(exc),
            )
        return TestGenerationResponse(
            candidates_json=content,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            duration_seconds=time.monotonic() - start,
        )

    async def generate_structured(
        self, system_prompt: str, user_message: str
    ) -> StructuredTextResponse:
        start = time.monotonic()
        try:
            content, _thinking, usage, _native = await self._chat(
                system_prompt, user_message, json_mode=True, thinking=self._thinking_enabled
            )
        except Exception as exc:
            return StructuredTextResponse(
                content="",
                provider=self.provider_name,
                model=self.model_name,
                usage=AIUsage(),
                duration_seconds=time.monotonic() - start,
                error=_format_error(exc),
            )
        return StructuredTextResponse(
            content=content,
            provider=self.provider_name,
            model=self.model_name,
            usage=usage,
            duration_seconds=time.monotonic() - start,
        )

    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        return await self.generate_structured(system_prompt, user_message)

    async def _chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        json_mode: bool,
        thinking: bool,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[Mapping[str, Any]] | tuple[Any, ...] | None = None,
    ) -> tuple[str, str | None, AIUsage, tuple[Mapping[str, Any], ...]]:
        url = _completions_url(self._base_url)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "enable_thinking": thinking,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "description": str(item.get("description") or ""),
                        "parameters": item.get("parameters") or {},
                    },
                }
                for item in tools
                if isinstance(item, Mapping)
            ]
        if json_mode and self._native_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if not thinking:
            # Ask for a clean final response when thinking is disabled.
            payload["messages"][0]["content"] = (
                system_prompt
                + "\n\nRespond with the final answer only. Do not include a reasoning trace."
            )

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                return _parse_chat_response(response)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                logger.warning("MLX attempt %d failed: %s", attempt + 1, exc)
        assert last_exc is not None
        raise last_exc


def _parse_chat_response(
    response: httpx.Response,
) -> tuple[str, str | None, AIUsage, tuple[Mapping[str, Any], ...]]:
    if response.status_code == 401:
        raise httpx.HTTPStatusError("Unauthorized", request=response.request, response=response)
    if response.status_code == 429:
        raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
    if response.status_code == 404:
        raise httpx.HTTPStatusError("model_missing", request=response.request, response=response)
    if response.status_code >= 500:
        raise httpx.HTTPStatusError(
            f"server_error:{response.status_code}", request=response.request, response=response
        )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    raw_content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    thinking, content = split_thinking(raw_content, reasoning_field=reasoning)
    raw_usage = data.get("usage") or {}
    usage = AIUsage(
        prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
        completion_tokens=int(raw_usage.get("completion_tokens") or 0),
    )
    native: list[Mapping[str, Any]] = []
    for item in message.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        native.append({"tool": fn.get("name"), "arguments": args, "reason": "native_tool_call"})
    return content, thinking, usage, tuple(native)


def _completions_url(base: str) -> str:
    cleaned = base.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return cleaned + "/chat/completions"


def _http_url_configured(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _format_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return f"timeout:{exc}"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else "?"
        return f"http_{status}:{exc}"
    return str(exc)

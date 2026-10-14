"""Local AI provider — OpenAI-compatible HTTP backends (Ollama, LM Studio, MLX).

``backend`` selects the transport:
  * ``openai_compatible`` (default) — Ollama, LM Studio, llama.cpp server
  * ``mlx`` — :class:`MlxProvider` against a loopback OpenAI-compatible server
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.ai.health import AIHealthStatus, probe_openai_compatible
from app.ai.openai_provider import OpenAIProvider
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

_OLLAMA_DEFAULT_BASE = "http://localhost:11434/v1"


class LocalAIProvider(LLMProvider):
    """Local-model facade over an OpenAI-compatible HTTP endpoint or MLX."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "",
        max_tokens: int = 2_000,
        temperature: float = 0.1,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        *,
        backend: str = "openai_compatible",
        provider_name: str = "local",
        thinking_enabled: bool = False,
        native_json_mode: bool = False,
        max_context_tokens: int | None = None,
    ) -> None:
        if backend == "mlx":
            from app.ai.mlx_provider import DEFAULT_MLX_BASE_URL, MlxProvider

            self._mlx: MlxProvider | None = MlxProvider(
                model=model,
                base_url=base_url or DEFAULT_MLX_BASE_URL,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                thinking_enabled=thinking_enabled,
                native_json_mode=native_json_mode,
                max_context_tokens=max_context_tokens,
                provider_name=provider_name if provider_name != "local" else "mlx",
            )
            self._inner: OpenAIProvider | None = None
            self._provider_name = self._mlx.provider_name
            self._base_url = base_url or DEFAULT_MLX_BASE_URL
            self._api_key = api_key
            self._max_tokens = max_tokens
            self._backend = "mlx"
            return
        if backend != "openai_compatible":
            raise ValueError(
                f"Unknown local AI backend {backend!r}. Supported: 'openai_compatible', 'mlx'."
            )
        resolved_base = base_url or _OLLAMA_DEFAULT_BASE
        resolved_key = api_key or "ollama"
        self._mlx = None
        self._inner = OpenAIProvider(
            api_key=resolved_key,
            model=model,
            base_url=resolved_base,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self._provider_name = provider_name
        self._base_url = resolved_base
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._backend = "openai_compatible"

    @classmethod
    def from_settings(cls, settings: object) -> LocalAIProvider:
        from app.core.config import Settings

        assert isinstance(settings, Settings)
        provider_name = settings.ai_provider
        if provider_name == "mlx":
            from app.ai.mlx_provider import MlxProvider

            mlx = MlxProvider.from_settings(settings)
            instance = cls.__new__(cls)
            instance._mlx = mlx
            instance._inner = None
            instance._provider_name = "mlx"
            instance._base_url = settings.ai_base_url or settings.mlx_base_url
            instance._api_key = settings.ai_api_key
            instance._max_tokens = settings.ai_max_output_tokens
            instance._backend = "mlx"
            return instance

        base_url = settings.ai_base_url
        if not base_url and provider_name == "ollama":
            base_url = _OLLAMA_DEFAULT_BASE
        api_key = settings.ai_api_key
        if not api_key and provider_name == "ollama":
            api_key = "ollama"
        return cls(
            api_key=api_key,
            model=settings.ai_model,
            base_url=base_url,
            max_tokens=settings.ai_max_output_tokens,
            temperature=settings.ai_temperature,
            timeout_seconds=settings.ai_timeout_seconds,
            max_retries=settings.ai_max_retries,
            backend="openai_compatible",
            provider_name=provider_name if provider_name != "local" else "local",
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        if self._mlx is not None:
            return self._mlx.model_name
        assert self._inner is not None
        return self._inner.model_name

    def capabilities(self) -> AICapabilities:
        if self._mlx is not None:
            return self._mlx.capabilities()
        return AICapabilities(
            chat=True,
            structured_output=True,
            structured_generation=True,
            text_generation=True,
            tool_calls=False,
            thinking=False,
            thinking_can_disable=True,
            max_output_tokens=self._max_tokens,
            supports_local_models=True,
            local_execution=True,
            native_json_response_format=True,
            notes=("OpenAI-compatible local HTTP (Ollama, LM Studio, llama.cpp).",),
        )

    def _endpoint_configured(self) -> bool:
        return _http_url_configured(self._base_url)

    async def is_available(self) -> bool:
        if self._mlx is not None:
            return await self._mlx.is_available()
        if not self._endpoint_configured():
            return False
        status = await self._probe()
        return status.reachable

    async def _probe(self) -> AIHealthStatus:
        key = self._api_key or ("ollama" if self._provider_name == "ollama" else "")
        return await probe_openai_compatible(
            provider=self.provider_name,
            model=self.model_name,
            base_url=self._base_url,
            api_key=key,
            is_local=True,
            configured=self._endpoint_configured(),
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self._mlx is not None:
            return await self._mlx.complete(request)
        return await super().complete(request)

    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        if self._mlx is not None:
            return await self._mlx.analyze(request)
        assert self._inner is not None
        response = await self._inner.analyze(request)
        response.provider = self.provider_name
        return response

    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        if self._mlx is not None:
            return await self._mlx.generate_tests(system_prompt, user_message)
        assert self._inner is not None
        response = await self._inner.generate_tests(system_prompt, user_message)
        response.provider = self.provider_name
        return response

    async def generate_structured(
        self, system_prompt: str, user_message: str
    ) -> StructuredTextResponse:
        if self._mlx is not None:
            return await self._mlx.generate_structured(system_prompt, user_message)
        assert self._inner is not None
        response = await self._inner.generate_structured(system_prompt, user_message)
        response.provider = self.provider_name
        return response

    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        return await self.generate_structured(system_prompt, user_message)

    async def health(self) -> AIHealthStatus:
        if self._mlx is not None:
            return await self._mlx.health()
        if not self._endpoint_configured():
            return AIHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                is_local=True,
                reachable=False,
                configured=False,
                model_available=None,
                error="invalid_or_missing_base_url",
            )
        return await self._probe()


def _http_url_configured(url: str) -> bool:
    raw = (url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

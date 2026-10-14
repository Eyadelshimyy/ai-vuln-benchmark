from app.ai.anthropic_provider import AnthropicProvider
from app.ai.context_builder import ContextBuilder
from app.ai.local_provider import LocalAIProvider
from app.ai.mlx_provider import MlxProvider
from app.ai.mock_provider import MockLLMProvider
from app.ai.openai_provider import OpenAIProvider
from app.ai.prompt_builder import PromptBuilder
from app.ai.provider import (
    AICapabilities,
    AIUsage,
    CompletionRequest,
    CompletionResponse,
    DebuggingRequest,
    HypothesisResult,
    LLMProvider,
    ProviderResponse,
    SourceFileEvidence,
    StaticFindingEvidence,
    TestFailureEvidence,
)
from app.ai.restore import restore_provider, snapshot_provider

# Canonical architecture name; existing LLMProvider remains the ABC.
AIProvider = LLMProvider
MockAIProvider = MockLLMProvider


def get_provider() -> LLMProvider:
    """Return the configured AI provider based on settings."""
    from app.ai.factory import create_provider
    from app.core.config import settings

    return create_provider(settings)


__all__ = [
    "AICapabilities",
    "AIProvider",
    "LLMProvider",
    "CompletionRequest",
    "CompletionResponse",
    "DebuggingRequest",
    "ProviderResponse",
    "HypothesisResult",
    "TestFailureEvidence",
    "StaticFindingEvidence",
    "SourceFileEvidence",
    "AIUsage",
    "MockAIProvider",
    "MockLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalAIProvider",
    "MlxProvider",
    "ContextBuilder",
    "PromptBuilder",
    "get_provider",
    "restore_provider",
    "snapshot_provider",
]

"""Tests for the curated public root API in ellements.core.

The root surface is deliberately small — happy-path entry points only.
Specialized types (sources, events, dialects, image types, config
loaders…) live in their canonical subpackage and are imported from
there. See the ``ellements.core`` module docstring for the full
subpackage map.
"""

from __future__ import annotations

import ellements.core as core


def test_curated_root_surface_advertises_happy_path() -> None:
    """The root surface covers what 90% of callers reach for."""
    expected_subset = {
        # Clients + messages
        "LLMClient",
        "LLMClientProtocol",
        "Conversation",
        "Message",
        "MessageContent",
        "MessageInput",
        # Tools
        "Tool",
        "ToolSpec",
        "ToolRegistry",
        "ToolCallRecord",
        "ToolCallResponse",
        "SimpleTool",
        # Prompting + templating
        "PersonaLibrary",
        "GuidelineLibrary",
        "TemplateRenderer",
        # Observability
        "JsonlPromptLogger",
        "LLMObserver",
        # Error taxonomy
        "EllementsError",
        "EllementsValidationError",
        "LLMError",
        "ConversationError",
        "MaxToolIterationsError",
        "StructuredOutputUnsupportedError",
        "LogprobsUnsupportedError",
        "BudgetExceededError",
        "PromptLibraryError",
        "PersonaNotFoundError",
        "GuidelineNotFoundError",
        "PromptKeyMissingError",
    }
    missing = expected_subset - set(core.__all__)
    assert not missing, f"missing from ellements.core.__all__: {sorted(missing)}"


def test_specialized_types_live_in_subpackages() -> None:
    """Specialized types must NOT be re-exported at the root.

    They are reachable from their canonical subpackage; the root is
    kept small on purpose.
    """
    for name in (
        # Prompting internals
        "PersonaSource",
        "GuidelineSource",
        "PromptContext",
        # Observability event types
        "LLMRequestEvent",
        "LLMResponseEvent",
        "AgentEvent",
        "format_log_markdown",
        # Config helpers
        "ConfigError",
        "load_json",
        "load_toml",
        "overlay",
        # Async helpers
        "parallel_map",
        # Chunking
        "TextProcessor",
        "count_tokens",
        # Image types
        "ImageInput",
        "ImageGenerationResponse",
        "GeneratedImage",
        # Composition layers
        "CachingLLMClient",
        "InMemoryCache",
        "RateLimitedLLMClient",
        "BudgetedLLMClient",
        # Tool dialects
        "OpenAIChatDialect",
        "AnthropicDialect",
        "default_dialect_for_model",
    ):
        assert name not in core.__all__, (
            f"{name} should live in its subpackage, not at the root"
        )


def test_removed_legacy_aliases_are_absent() -> None:
    for name in (
        "ContentProcessor",
        "create_llm_summarizer",
        "VisionClient",
        "PersonaManager",
        "GuidelineManager",
        "ValidationError",  # renamed to EllementsValidationError
        "RateLimiter",  # renamed to RateLimiterProtocol
        "BudgetTracker",  # renamed to BudgetTrackerProtocol
    ):
        assert not hasattr(core, name), f"{name} should no longer be exported"


def test_legacy_module_paths_are_gone() -> None:
    import importlib

    for module in (
        "ellements.core.clients",
        "ellements.core.vision",
        "ellements.core.prompt_log",
        "ellements.core.web_utils",
    ):
        try:
            importlib.import_module(module)
        except ImportError:
            continue
        raise AssertionError(f"{module} should no longer exist")

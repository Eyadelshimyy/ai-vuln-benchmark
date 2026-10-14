"""Core primitives and reusable foundations for ellements.

The top-level surface is intentionally small — the **happy path** of
client + messages + tools + prompts + observability + the full error
taxonomy. Specialized layers live in subpackages and are imported
directly from them. The subpackage path is the canonical one; the
top-level re-exports below exist purely to keep the most common 90%
of code short::

    from ellements.core import LLMClient, Conversation, Tool, ToolRegistry

Subpackage map:

* :mod:`ellements.core.caching` — caching wrapper + backends
  (``CachingLLMClient``, ``InMemoryCache``, ``JsonDiskCache``).
* :mod:`ellements.core.rate_limit` — rate-limit wrapper + token-bucket
  implementation (``RateLimitedLLMClient``, ``TokenBucketRateLimiter``).
* :mod:`ellements.core.budgeting` — budget tracker + budgeted wrapper
  (``BudgetedLLMClient``, ``CallCountBudget``, ``TokenBudget``).
* :mod:`ellements.core.observability` — event types, observers,
  Markdown formatters (``LLMRequestEvent``, ``LLMResponseEvent``,
  ``AgentEvent``, ``format_log_markdown``).
* :mod:`ellements.core.prompting` — persona / guideline / prompt
  context (``PromptContext``, ``PersonaSource``, ``GuidelineSource``).
* :mod:`ellements.core.templating` — Mustache template renderer plus
  ``render_prompt_template`` convenience helper.
* :mod:`ellements.core.config` — JSON / TOML loaders + ``overlay``.
* :mod:`ellements.core.chunking` — text chunking + ``count_tokens``.
* :mod:`ellements.core.async_utils` — ``parallel_map``.
* :mod:`ellements.core.tools` — full tool / dialect / registry surface,
  including the provider dialects (``OpenAIChatDialect``,
  ``OpenAIResponsesDialect``, ``AnthropicDialect``, ``GeminiDialect``,
  ``default_dialect_for_model``).
* :mod:`ellements.core.llm` — extras like multimodal types
  (``ImageInput``, ``ImageURLPart``, ``ImageGenerationResponse``,
  ``GeneratedImage``) and the ``LLMClientWrapper`` base.
* :mod:`ellements.core.exceptions` — full exception hierarchy
  (already re-exported below for the common ones; specialized
  exceptions stay here).

Anything not listed in this module's ``__all__`` is reachable from its
subpackage. There are no hidden names: every subpackage carries its
own ``__all__``.
"""

from __future__ import annotations

from .exceptions import (
    BudgetExceededError,
    ConversationError,
    EllementsError,
    EllementsValidationError,
    GuidelineNotFoundError,
    LLMError,
    LogprobsUnsupportedError,
    MaxToolIterationsError,
    PersonaNotFoundError,
    PromptKeyMissingError,
    PromptLibraryError,
    StructuredOutputUnsupportedError,
)
from .llm import (
    Conversation,
    LLMClient,
    LLMClientProtocol,
    LocalCacheConfig,
    Message,
    MessageContent,
    MessageInput,
)
from .observability import JsonlPromptLogger, LLMObserver
from .prompting import GuidelineLibrary, PersonaLibrary
from .templating import TemplateRenderer
from .tools import (
    SimpleTool,
    Tool,
    ToolCallRecord,
    ToolCallResponse,
    ToolRegistry,
    ToolSpec,
)

__all__ = [
    "BudgetExceededError",
    "Conversation",
    "ConversationError",
    "EllementsError",
    "EllementsValidationError",
    "GuidelineLibrary",
    "GuidelineNotFoundError",
    "JsonlPromptLogger",
    "LLMClient",
    "LLMClientProtocol",
    "LLMError",
    "LLMObserver",
    "LocalCacheConfig",
    "LogprobsUnsupportedError",
    "MaxToolIterationsError",
    "Message",
    "MessageContent",
    "MessageInput",
    "PersonaLibrary",
    "PersonaNotFoundError",
    "PromptKeyMissingError",
    "PromptLibraryError",
    "SimpleTool",
    "StructuredOutputUnsupportedError",
    "TemplateRenderer",
    "Tool",
    "ToolCallRecord",
    "ToolCallResponse",
    "ToolRegistry",
    "ToolSpec",
]

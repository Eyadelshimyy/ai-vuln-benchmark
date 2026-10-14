"""Internal LLM support package for ellements.core."""

from .client import LLMClient
from .images import (
    GeneratedImage,
    ImageGenerationResponse,
    ImageInput,
)
from .local_cache import LocalCacheConfig
from .messages import (
    Conversation,
    ImageURLPart,
    Message,
    MessageContent,
    MessageInput,
    build_multimodal_user_message,
    normalize_message_input,
)
from .model_params import filter_parameters, get_unsupported_parameters
from .protocol import LLMClientProtocol
from .structured import (
    ensure_structured_support,
    parse_structured_content,
    supports_structured_output,
)
from .transport import CompletionRequest, CompletionTransport
from .wrapper import LLMClientWrapper

__all__ = [
    "Conversation",
    "CompletionRequest",
    "CompletionTransport",
    "GeneratedImage",
    "ImageGenerationResponse",
    "ImageInput",
    "ImageURLPart",
    "LLMClient",
    "LLMClientProtocol",
    "LLMClientWrapper",
    "LocalCacheConfig",
    "Message",
    "MessageContent",
    "MessageInput",
    "build_multimodal_user_message",
    "ensure_structured_support",
    "filter_parameters",
    "get_unsupported_parameters",
    "normalize_message_input",
    "parse_structured_content",
    "supports_structured_output",
]

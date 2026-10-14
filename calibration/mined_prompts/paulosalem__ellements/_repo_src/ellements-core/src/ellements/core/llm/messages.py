"""Conversation and multimodal message primitives.

This module defines the typed message model used throughout ellements:

- :class:`MessageContent` — a single content part (text or image)
- :class:`Message` — one message in a conversation
- :class:`Conversation` — an ordered thread of messages plus optional system prompt

Plus helpers for normalizing accepted inputs and building multimodal payloads.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

ImageDetail: TypeAlias = Literal["auto", "low", "high"]

_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})


class ImageURLPart(BaseModel):
    """Typed payload for the ``image_url`` field of an image content part.

    Distinguishes between hosted URLs (``http://`` / ``https://``) and
    data URIs (``data:image/...;base64,...``) only by inspection; the
    distinction is opaque to the provider.
    """

    url: str = Field(description="HTTP(S) URL or data URI to the image")
    detail: ImageDetail | None = Field(default=None, description="Vision detail level")

    @property
    def is_data_uri(self) -> bool:
        """True if ``url`` is a base64-encoded data URI."""
        return self.url.startswith("data:")


class MessageContent(BaseModel):
    """One typed content part: either text or an image reference."""

    type: Literal["text", "image_url"] = Field(description="Content part type")
    text: str | None = Field(default=None, description="Text content when type='text'")
    image_url: ImageURLPart | None = Field(
        default=None,
        description="Image reference when type='image_url'",
    )

    def model_post_init(self, __context: Any) -> None:
        if self.type == "text" and not self.text:
            raise ValueError("text must be provided when type='text'")
        if self.type == "image_url" and self.image_url is None:
            raise ValueError("image_url must be provided when type='image_url'")

    @classmethod
    def text_part(cls, text: str) -> MessageContent:
        """Build a text content part."""
        return cls(type="text", text=text)

    @classmethod
    def image_part(
        cls,
        *,
        url: str,
        detail: ImageDetail | None = None,
    ) -> MessageContent:
        """Build an image content part referencing *url* (URL or data URI)."""
        return cls(type="image_url", image_url=ImageURLPart(url=url, detail=detail))

    def to_litellm_part(self) -> dict[str, Any]:
        """Convert the content part to LiteLLM-compatible payload."""
        if self.type == "text":
            return {"type": "text", "text": self.text}
        assert self.image_url is not None  # checked in model_post_init
        payload: dict[str, Any] = {"url": self.image_url.url}
        if self.image_url.detail is not None:
            payload["detail"] = self.image_url.detail
        return {"type": "image_url", "image_url": payload}


def content_parts_to_litellm(parts: Sequence[MessageContent]) -> list[dict[str, Any]]:
    """Convert content parts into LiteLLM payloads."""
    return [part.to_litellm_part() for part in parts]


class Message(BaseModel):
    """One message in a conversation thread."""

    role: str = Field(description="Role of the message sender")
    content: str | list[MessageContent] = Field(
        description="Message content as plain text or multimodal parts",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        if value not in _VALID_ROLES:
            raise ValueError(
                f"Invalid role {value!r}. "
                f"Expected one of: {sorted(_VALID_ROLES)}."
            )
        return value


class Conversation(BaseModel):
    """Ordered thread of messages plus an optional system prompt."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = Field(default_factory=list)
    model: str = Field(description="The LLM model identifier")
    system_prompt: str | None = Field(default=None)

    def add_message(
        self,
        role: str,
        content: str | list[MessageContent],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Append a message to the thread.

        Raises:
            ValueError: If *role* is not one of ``system``, ``user``,
                ``assistant``, or ``tool``.
        """
        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )
        return self

    def user(
        self,
        content: str | list[MessageContent],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Append a user message."""
        return self.add_message("user", content, metadata)

    def assistant(
        self,
        content: str | list[MessageContent],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Append an assistant message."""
        return self.add_message("assistant", content, metadata)

    def system(
        self,
        content: str | list[MessageContent],
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Append a system message."""
        return self.add_message("system", content, metadata)

    def to_litellm_messages(self) -> list[dict[str, Any]]:
        """Convert the conversation into LiteLLM message payloads."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
                continue
            messages.append(
                {
                    "role": msg.role,
                    "content": content_parts_to_litellm(msg.content),
                }
            )

        return messages


MessageInput: TypeAlias = str | list[dict[str, Any]] | Conversation


def normalize_message_input(messages: MessageInput) -> list[dict[str, Any]]:
    """Normalize accepted message inputs into mutable LiteLLM-style payloads."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, Conversation):
        return messages.to_litellm_messages()
    return list(messages)


def build_multimodal_user_message(
    prompt: str,
    parts: Sequence[MessageContent],
) -> list[dict[str, Any]]:
    """Build a single-user multimodal message payload combining text and parts."""
    return [
        {
            "role": "user",
            "content": content_parts_to_litellm(
                [MessageContent.text_part(prompt), *parts]
            ),
        }
    ]

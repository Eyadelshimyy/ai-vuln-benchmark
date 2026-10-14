"""Image input + image-generation response models and conversion helpers.

This module owns the multimodal *input* type :class:`ImageInput` (URLs,
data URIs, local files) and the *output* types
:class:`GeneratedImage` / :class:`ImageGenerationResponse` for image-generation
APIs. There is no separate vision client — callers compose ``ImageInput``
content parts directly with :meth:`LLMClient.complete` for analysis tasks.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .messages import ImageDetail, MessageContent


@dataclass(frozen=True)
class ImageUpload:
    """Captured upload bytes. This identity asserts neither custody nor permission."""

    filename: str
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}\.(png|jpg|jpeg|webp)", self.filename):
            raise ValueError("An image upload requires a safe explicit PNG/JPEG/WebP filename.")
        if type(self.content) is not bytes or not 0 < len(self.content) < 50_000_000:
            raise ValueError("An image upload requires captured bytes below 50 MB.")

    @property
    def media_type(self) -> str:
        return "image/" + {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}[
            self.filename.rsplit(".", 1)[1]
        ]

    def identity(self) -> dict[str, Any]:
        """JSON-friendly facts about the exact bytes supplied to the SDK."""
        return {
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "size_bytes": len(self.content), "media_type": self.media_type,
        }

_MEDIA_TYPE_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageInput(BaseModel):
    """A multimodal image input for vision-capable LLMs.

    Exactly one source must be provided: ``url`` (HTTP/HTTPS) or ``base64``.
    Use the factory classmethods :meth:`from_url`, :meth:`from_base64`,
    or :meth:`from_path` for convenience.
    """

    url: str | None = Field(default=None, description="HTTP(S) URL to image")
    base64: str | None = Field(default=None, description="Base64-encoded image data")
    media_type: str | None = Field(
        default=None,
        description="MIME type for base64 inputs (e.g. 'image/png')",
    )
    detail: ImageDetail = Field(default="auto", description="Vision detail level")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImageInput:
        if self.url is None and self.base64 is None:
            raise ValueError("Either url or base64 must be provided")
        if self.url is not None and self.base64 is not None:
            raise ValueError("Only one of url or base64 should be provided")
        if self.base64 is not None and not self.media_type:
            raise ValueError("media_type is required when providing base64 data")
        return self

    @classmethod
    def from_url(cls, url: str, detail: ImageDetail = "auto") -> ImageInput:
        """Build an :class:`ImageInput` from an HTTP(S) URL."""
        return cls(url=url, detail=detail)

    @classmethod
    def from_base64(
        cls,
        data: str,
        media_type: str = "image/png",
        detail: ImageDetail = "auto",
    ) -> ImageInput:
        """Build an :class:`ImageInput` from a base64-encoded image payload."""
        return cls(base64=data, media_type=media_type, detail=detail)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        media_type: str | None = None,
        detail: ImageDetail = "auto",
    ) -> ImageInput:
        """Build an :class:`ImageInput` from a local file path."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        resolved_media_type = media_type
        if resolved_media_type is None:
            resolved_media_type = _MEDIA_TYPE_BY_EXT.get(path_obj.suffix.lower())
            if resolved_media_type is None:
                raise ValueError(
                    f"Cannot determine media type for extension: {path_obj.suffix}. "
                    "Please provide media_type explicitly."
                )

        encoded = base64.b64encode(path_obj.read_bytes()).decode("utf-8")
        return cls(base64=encoded, media_type=resolved_media_type, detail=detail)

    def to_message_part(self) -> MessageContent:
        """Convert to the shared multimodal content representation."""
        if self.url:
            return MessageContent.image_part(url=self.url, detail=self.detail)
        data_uri = f"data:{self.media_type};base64,{self.base64}"
        return MessageContent.image_part(url=data_uri, detail=self.detail)

    def to_litellm_part(self) -> dict[str, Any]:
        """Convert to a LiteLLM-compatible message content part payload."""
        return self.to_message_part().to_litellm_part()


class GeneratedImage(BaseModel):
    """A single image returned by an image-generation API."""

    url: str | None = Field(default=None, description="URL to the generated image")
    b64_json: str | None = Field(
        default=None, description="Base64-encoded JSON payload of the image"
    )
    revised_prompt: str | None = Field(
        default=None, description="Provider-revised prompt actually used"
    )


class ImageGenerationResponse(BaseModel):
    """Normalized response from an image-generation API."""

    created: int = Field(description="Unix timestamp when the image was created")
    data: list[GeneratedImage] = Field(description="Generated images")
    model: str | None = Field(default=None, description="Model used for generation")
    usage: Any | None = Field(
        default=None, description="Usage info (shape varies by provider)"
    )
    provider_receipt_ids: tuple[str, ...] = Field(
        default=(), description="Observed provider request identities, never local attempt IDs"
    )
    request_parameters: dict[str, Any] | None = Field(
        default=None, description="Actual public parameters passed to the provider SDK; excludes credentials"
    )


def build_image_generation_request(
    *,
    prompt: str,
    model: str | None,
    n: int,
    size: str | None,
    quality: str | None,
    style: str | None,
    response_format: Literal["url", "b64_json"],
    extra_params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build provider parameters for image-generation calls."""
    target_model = model or "dall-e-3"
    params: dict[str, Any] = {"prompt": prompt, "model": target_model, "n": n}

    if "dall-e" in target_model.lower():
        params["response_format"] = response_format
    if size:
        params["size"] = size
    if quality:
        params["quality"] = quality
    if style:
        params["style"] = style

    params.update(extra_params)
    return target_model, params


def build_image_edit_request(
    *,
    prompt: str,
    model: str | None,
    n: int,
    size: str | None,
    quality: str | None,
    extra_params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build provider parameters for image-*edit* calls (references + prompt).

    The input image files are passed separately to the provider call; this
    returns only the JSON-friendly parameters.
    """
    target_model = model or "gpt-image-1"
    params: dict[str, Any] = {"prompt": prompt, "model": target_model, "n": n}
    if size:
        params["size"] = size
    if quality:
        params["quality"] = quality
    params.update(extra_params)
    return target_model, params


def normalize_usage(usage_obj: Any) -> Any | None:
    """Convert provider-specific usage payloads into serializable objects."""
    if not usage_obj:
        return None
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    if hasattr(usage_obj, "dict"):
        return usage_obj.dict()
    if isinstance(usage_obj, dict):
        return usage_obj
    return {
        key: getattr(usage_obj, key)
        for key in dir(usage_obj)
        if not key.startswith("_") and not callable(getattr(usage_obj, key))
    }


def parse_image_generation_response(
    response: Any, *, target_model: str, provider_receipt_ids: tuple[str, ...] = ()
) -> ImageGenerationResponse:
    """Normalize a LiteLLM image-generation response into a stable model."""
    return ImageGenerationResponse(
        created=response.created,
        data=[GeneratedImage.model_validate(
            img.model_dump() if hasattr(img, "model_dump") else img
        ) for img in response.data],
        model=getattr(response, "model", target_model),
        usage=normalize_usage(getattr(response, "usage", None)),
        provider_receipt_ids=provider_receipt_ids,
    )


__all__ = [
    "ImageUpload",
    "GeneratedImage",
    "ImageGenerationResponse",
    "ImageInput",
    "build_image_edit_request",
    "build_image_generation_request",
    "normalize_usage",
    "parse_image_generation_response",
]

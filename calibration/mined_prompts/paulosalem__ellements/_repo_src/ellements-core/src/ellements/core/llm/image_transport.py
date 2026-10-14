"""Receipt-preserving, single-attempt image generation beneath the shared client."""

from __future__ import annotations

import os
import re
from typing import Any, Protocol

from ellements.core.exceptions import LLMError

from .images import (
    ImageGenerationResponse,
    ImageUpload,
    parse_image_generation_response,
)


class ImageGenerationTransportError(LLMError):
    """Non-content failure evidence; never a settlement or retry authorization."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__("Image provider outcome unknown; no automatic retry.")
        self.diagnostic = diagnostic


class ImageGenerationNotStartedError(ImageGenerationTransportError):
    """The SDK failed before entering the provider-request phase."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__(diagnostic)
        self.args = ("Image provider request did not start; no automatic retry.",)


class ImageGenerationTransport(Protocol):
    async def generate(self, parameters: dict[str, Any]) -> ImageGenerationResponse: ...


class ImageEditTransport(Protocol):
    async def edit(
        self, parameters: dict[str, Any], images: tuple[ImageUpload, ...],
        mask: ImageUpload | None = None,
    ) -> ImageGenerationResponse: ...


#: Output encodings the image endpoints accept. `png` alone carries alpha, so a
#: transparent background requires it.
IMAGE_OUTPUT_FORMATS = frozenset({"png", "jpeg", "webp"})
#: Background handling. `transparent` requires a `png` or `webp` output format.
IMAGE_BACKGROUNDS = frozenset({"opaque", "transparent", "auto"})

#: Quality tiers the current OpenAI image models accept. `xhigh` and `max` arrived
#: with GPT Image 2.5; the older tiers remain valid.
IMAGE_QUALITIES = frozenset({"low", "medium", "high", "xhigh", "max", "auto"})


class OpenAIImageGenerationTransport:
    """Use the official async SDK without retries, URLs or hidden conditioning."""

    def __init__(self, *, timeout_seconds: float) -> None:
        if not 0 < timeout_seconds <= 900:
            raise ValueError("A finite image transport timeout of at most 900 seconds is required.")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def preflight() -> None:
        """Require explicit process credentials without retaining or discovering them."""
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("Image provider credentials are unavailable in this process.")

    async def generate(self, parameters: dict[str, Any]) -> ImageGenerationResponse:
        """Text-only generation across the endpoint's full parameter surface."""
        allowed = {"model", "prompt", "size", "quality", "n", "output_format", "background"}
        if (
            set(parameters) != allowed
            or type(parameters["n"]) is not int or not 1 <= parameters["n"] <= 10
            or parameters["output_format"] not in IMAGE_OUTPUT_FORMATS
            or parameters["background"] not in IMAGE_BACKGROUNDS
            or (parameters["background"] == "transparent"
                and parameters["output_format"] not in {"png", "webp"})
            or parameters["quality"] not in IMAGE_QUALITIES
        ):
            raise ValueError("The image transport requires one explicit text-only request.")
        return await self._request(parameters, images=None)

    async def _request(
        self, parameters: dict[str, Any], *, images: tuple[ImageUpload, ...] | None,
        mask: ImageUpload | None = None,
    ) -> ImageGenerationResponse:
        from openai import AsyncOpenAI

        model = parameters["model"]
        if not isinstance(model, str) or not re.fullmatch(r"openai/[A-Za-z0-9._-]{1,128}", model):
            raise ValueError("The OpenAI transport requires an explicit openai/ model.")
        request = {**parameters, "model": model.removeprefix("openai/")}
        phase = "client_initialization"
        identifier = None
        status = None
        try:
            async with AsyncOpenAI(
                max_retries=0, timeout=self.timeout_seconds, base_url="https://api.openai.com/v1"
            ) as client:
                phase = "request"
                if images is None:
                    raw = await client.images.with_raw_response.generate(**request)
                else:
                    raw = await client.images.with_raw_response.edit(
                        **request,
                        image=[(item.filename, item.content, item.media_type) for item in images],
                        # A mask pins what the model may not touch. Without one a
                        # "local edit" is only a request in prose, and the model
                        # is free to move the subject or repaint the whole frame.
                        **({"mask": (mask.filename, mask.content, mask.media_type)}
                           if mask is not None else {}),
                    )
                phase = "response_parse"
                identifier = raw.headers.get("x-request-id")
                status = raw.status_code
                response = raw.parse()
            result = parse_image_generation_response(
                response, target_model=model, provider_receipt_ids=_receipts(identifier),
            )
        except Exception as error:
            from openai import APIStatusError

            if isinstance(error, APIStatusError):
                status = error.status_code
                identifier = error.request_id
            failure = (
                ImageGenerationNotStartedError
                if phase == "client_initialization" and status is None and identifier is None
                else ImageGenerationTransportError
            )
            raise failure({
                "phase": phase,
                "exception_type": type(error).__name__,
                "cause_type": type(error.__cause__).__name__ if error.__cause__ else None,
                "http_status": status,
                "provider_receipt_ids": list(_receipts(identifier)),
                "timeout_seconds": self.timeout_seconds,
            }) from None
        result.request_parameters = request
        if images is not None:
            result.request_parameters = {
                **request, "images": [item.identity() for item in images],
                "masks": [mask.identity()] if mask is not None else [],
                "history": [], "other_conditioning": [],
            }
        return result


class OpenAIImageEditTransport(OpenAIImageGenerationTransport):
    """One explicit reference-assisted edit, optionally masked; no URLs or retries."""

    async def edit(
        self, parameters: dict[str, Any], images: tuple[ImageUpload, ...],
        mask: ImageUpload | None = None,
    ) -> ImageGenerationResponse:
        """One reference-assisted edit, optionally confined to a mask.

        `input_fidelity` is deliberately absent. Current OpenAI image models
        reject it outright -- gpt-image-2 answers HTTP 400 "does not support the
        'input_fidelity' parameter" -- because reference editing always runs at
        high fidelity. Sending it fails every edit, so it is not accepted here.

        Everything the endpoint does support is available: several variants in
        one call, any output encoding, transparent backgrounds, and a mask whose
        transparent pixels are the only region the model may repaint. The
        parameters are still checked exactly, because an unbounded request
        cannot be priced or reserved honestly.
        """
        allowed = {"model", "prompt", "size", "quality", "n", "output_format", "background"}
        if (
            set(parameters) != allowed
            or type(parameters["n"]) is not int or not 1 <= parameters["n"] <= 10
            or parameters["output_format"] not in IMAGE_OUTPUT_FORMATS
            or parameters["background"] not in IMAGE_BACKGROUNDS
            # Only PNG and WebP carry an alpha channel; JPEG cannot be transparent.
            or (parameters["background"] == "transparent"
                and parameters["output_format"] not in {"png", "webp"})
            or not isinstance(parameters["prompt"], str) or not parameters["prompt"].strip()
            or parameters["quality"] not in IMAGE_QUALITIES
            or not isinstance(parameters["size"], str)
            or not re.fullmatch(r"[1-9][0-9]{2,3}x[1-9][0-9]{2,3}", parameters["size"])
        ):
            raise ValueError("The edit transport requires one bounded explicit reference request.")
        if (
            not isinstance(images, tuple) or not 1 <= len(images) <= 16
            or any(not isinstance(item, ImageUpload) for item in images)
            or sum(len(item.content) for item in images) > 128 * 1024 * 1024
        ):
            raise ValueError("The edit transport requires bounded captured ordered image bytes.")
        # A mask is matched against the first reference -- the image being
        # edited. A mismatched mask silently edits the wrong region, so refuse.
        if mask is not None:
            if not isinstance(mask, ImageUpload) or mask.media_type != "image/png":
                raise ValueError("A mask must be captured PNG bytes carrying an alpha channel.")
            fault = _mask_fault(mask.content, images[0].content)
            if fault is not None:
                raise ValueError(fault)
        return await self._request(parameters, images=images, mask=mask)


def _mask_fault(mask: bytes, image: bytes) -> str | None:
    """Say exactly why a mask is unusable; an unverifiable mask is refused, not guessed."""
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return "A mask cannot be verified in this process; install Pillow or send no mask."
    try:
        with Image.open(BytesIO(mask)) as first, Image.open(BytesIO(image)) as second:
            if "A" not in first.getbands():
                return (
                    "A mask needs an alpha channel: its transparent pixels are the region "
                    "the model may repaint. Save RGBA PNG, not RGB."
                )
            if first.size != second.size:
                return (
                    f"A mask must match the edited image exactly: the mask is "
                    f"{first.width}x{first.height} and the image is {second.width}x{second.height}."
                )
    except Exception:  # noqa: BLE001 - an unreadable mask is a refusal
        return "A mask must be decodable PNG bytes."
    return None


def _receipts(identifier: object) -> tuple[str, ...]:
    return (
        (identifier,) if isinstance(identifier, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/+=-]{0,255}", identifier) else ()
    )


__all__ = [
    "IMAGE_BACKGROUNDS", "IMAGE_OUTPUT_FORMATS", "IMAGE_QUALITIES",
    "ImageEditTransport", "OpenAIImageEditTransport",
    "ImageGenerationTransport", "ImageGenerationTransportError",
    "ImageGenerationNotStartedError", "OpenAIImageGenerationTransport",
]

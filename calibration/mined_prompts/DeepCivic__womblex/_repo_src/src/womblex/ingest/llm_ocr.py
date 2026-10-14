"""LLM/VLM-based OCR engines that produce page-level markdown.

These engines return page text with reading order already resolved, so
layout analysis and region sorting are skipped downstream — see
``OCRPageResult.reading_order_native``.

Two engines live here:

- ``MistralOCRReader`` — Mistral's Pixtral Large VLM inferenced via
  **AWS Bedrock** using the Converse API
  (``bedrock-runtime`` ``converse``). Region is taken from
  ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (default ``us-east-1``); the
  model id defaults to ``mistral.pixtral-large-2502-v1:0``. AWS
  credentials are resolved by the standard boto3 chain (env vars,
  shared config, instance/role profile). Model access must be enabled
  in the Bedrock Model Access console first.
- ``OllamaOCRReader`` — a multimodal LLM served via a local Ollama
  OpenAI-compatible endpoint (``OLLAMA_BASE_URL``, default
  ``http://localhost:11435/v1``). Retained for fully-local runs.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from womblex.ingest.interfaces.protocols import OCRPageResult

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


def _img_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode an RGB/grayscale numpy image as PNG bytes."""
    from PIL import Image

    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(img[:, :, :3], mode="RGB")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _img_to_data_url(img: np.ndarray) -> str:
    """Encode an RGB numpy image as a base64 PNG data URL (for Ollama)."""
    b64 = base64.b64encode(_img_to_png_bytes(img)).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Mistral (Pixtral Large) via AWS Bedrock Converse API
# ---------------------------------------------------------------------------

# The transcription instruction sent alongside each page image. Asks for a
# faithful markdown transcription in reading order — Pixtral resolves layout
# itself, so the output feeds the reading-order-native path downstream.
_MISTRAL_OCR_PROMPT = (
    "Transcribe all text in this document image to markdown, verbatim and "
    "in natural reading order. Preserve headings, lists and tables. Do not "
    "summarise, translate, or add commentary — output only the transcription."
)


class MistralOCRReader:
    """OCR reader backed by Mistral Pixtral Large on AWS Bedrock.

    Invokes the Pixtral Large VLM through the ``bedrock-runtime``
    Converse API. The page image is sent as raw PNG bytes (the Converse
    API takes bytes directly — no base64 wrapping) with a transcription
    prompt; the model returns markdown text in reading order.

    ``OCRPageResult.reading_order_native`` is True so downstream
    strategies bypass layout-model region sorting.

    AWS credentials and region come from the standard boto3 resolution
    chain. Override the region with ``AWS_REGION`` / ``AWS_DEFAULT_REGION``
    (default ``us-east-1``) and the model with the ``model`` kwarg or the
    ``MISTRAL_OCR_MODEL_ID`` env var. Model access must be enabled in the
    Bedrock Model Access console before first use.
    """

    DEFAULT_MODEL_ID = "mistral.pixtral-large-2502-v1:0"
    DEFAULT_REGION = "us-east-1"

    def __init__(
        self,
        model: str | None = None,
        region: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = (
            model or os.environ.get("MISTRAL_OCR_MODEL_ID") or self.DEFAULT_MODEL_ID
        )
        self.region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or self.DEFAULT_REGION
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._client: object | None = None  # lazy boto3 client

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import boto3
        from botocore.config import Config

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(
                read_timeout=self.timeout_s,
                connect_timeout=min(self.timeout_s, 10.0),
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        logger.info(
            "Mistral OCR (Bedrock Pixtral) reader ready: model=%s region=%s",
            self.model_id, self.region,
        )

    @staticmethod
    def _extract_text(response: dict) -> str:
        """Concatenate text blocks from a Bedrock Converse response."""
        content = (
            response.get("output", {}).get("message", {}).get("content", [])
        )
        parts = [
            block["text"] for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(part for part in parts if part).strip()

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """Send the page image to Bedrock and return markdown text."""
        if img is None or img.size == 0:
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        self._ensure_client()
        assert self._client is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"text": _MISTRAL_OCR_PROMPT},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": _img_to_png_bytes(img)},
                        }
                    },
                ],
            }
        ]

        try:
            response = self._client.converse(  # type: ignore[attr-defined]
                modelId=self.model_id,
                messages=messages,
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except Exception as exc:
            logger.warning("Mistral OCR (Bedrock) request failed: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        try:
            text = self._extract_text(response)
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning("Mistral OCR (Bedrock) malformed response: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        # VLMs don't emit per-token confidence; treat a non-empty response
        # as high-confidence at the page level. Empty output → 0.
        confidence = 1.0 if text.strip() else 0.0
        return OCRPageResult(
            regions=[],
            markdown=text,
            reading_order_native=True,
            confidence=confidence,
        )

    def ping(self) -> None:
        """One-token text request: credentials, region and model access, or raise.

        The pre-run model check calls this; ``read_page`` swallows request
        errors so a single bad page does not stop a batch.
        """
        self._ensure_client()
        assert self._client is not None
        self._client.converse(  # type: ignore[attr-defined]
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 1},
        )

    def close(self) -> None:
        # boto3 clients hold no long-lived connection to close explicitly.
        self._client = None


# ---------------------------------------------------------------------------
# Ollama (local, OpenAI-compatible)
# ---------------------------------------------------------------------------

# Documented Ollama VLM prompts. "free" returns plain transcription; the
# markdown variant asks the model to preserve document structure.
OLLAMA_PROMPTS: dict[str, str] = {
    "free": "Transcribe all text in this image verbatim, preserving reading order.",
    "markdown": "Convert this document image to markdown, preserving headings, "
                "lists and tables in reading order.",
}


class OllamaOCRReader:
    """OCR reader backed by a multimodal LLM via a local Ollama endpoint.

    Default target is a local Ollama instance at
    ``http://localhost:11435/v1``. Override with ``OLLAMA_BASE_URL``.

    The model returns markdown/prose with native reading order, so
    ``OCRPageResult.reading_order_native`` is set to True and downstream
    strategies bypass layout-model region sorting.
    """

    DEFAULT_BASE_URL = "http://localhost:11435/v1"
    DEFAULT_MODEL = "llama3.2-vision"
    DEFAULT_PROMPT_KEY = "free"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        prompt: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        env_base = os.environ.get("OLLAMA_BASE_URL")
        resolved_base = base_url or env_base or self.DEFAULT_BASE_URL
        self.base_url = resolved_base.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.prompt = prompt or OLLAMA_PROMPTS[self.DEFAULT_PROMPT_KEY]
        self.timeout_s = timeout_s
        self._client: httpx.Client | None = None  # lazy

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import httpx

        self._client = httpx.Client(timeout=self.timeout_s)
        logger.info(
            "Ollama OCR reader ready: model=%s base_url=%s",
            self.model, self.base_url,
        )

    def read_page(self, img: np.ndarray) -> OCRPageResult:
        """Send the page image to the LLM and return markdown text."""
        if img is None or img.size == 0:
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        self._ensure_client()
        assert self._client is not None

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _img_to_data_url(img)},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Ollama OCR request failed: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("Ollama OCR malformed response: %s", exc)
            return OCRPageResult(reading_order_native=True, confidence=0.0)

        # LLMs don't emit per-token confidence; treat a non-empty response
        # as high-confidence at the page level. Empty output → 0.
        confidence = 1.0 if text.strip() else 0.0
        return OCRPageResult(
            regions=[],
            markdown=text,
            reading_order_native=True,
            confidence=confidence,
        )

    def ping(self) -> None:
        """Raise unless the server answers and has the model pulled (free, no inference)."""
        self._ensure_client()
        assert self._client is not None
        resp = self._client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        served = {m.get("id", "") for m in resp.json().get("data", [])}
        if not served & {self.model, f"{self.model}:latest"}:
            raise RuntimeError(
                f"Ollama at {self.base_url} does not serve {self.model!r} "
                f"(has: {sorted(served)}); `ollama pull {self.model}`"
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ------------------------------------------------------------------
# Module-level caches
# ------------------------------------------------------------------

_mistral_readers: dict[str, MistralOCRReader] = {}
_ollama_readers: dict[str, OllamaOCRReader] = {}


def get_mistral_reader(
    model: str | None = None,
    region: str | None = None,
) -> MistralOCRReader:
    """Return a cached Mistral-OCR (Bedrock) reader keyed on (model, region)."""
    key = f"{model or ''}|{region or ''}"
    if key not in _mistral_readers:
        _mistral_readers[key] = MistralOCRReader(model=model, region=region)
    return _mistral_readers[key]


def get_ollama_reader(
    model: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
) -> OllamaOCRReader:
    """Return a cached Ollama OCR reader keyed on (model, base_url, prompt)."""
    key = f"{model or ''}|{base_url or ''}|{prompt or ''}"
    if key not in _ollama_readers:
        _ollama_readers[key] = OllamaOCRReader(
            model=model, base_url=base_url, prompt=prompt,
        )
    return _ollama_readers[key]

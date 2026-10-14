"""OpenAI audio transcription.

The audio never lands in GVAS storage: the channel's attachment access adapter
hands over bytes in memory, they are posted to the transcription endpoint, and
only the resulting text is persisted by the application. OpenAI is used for
transcription alone; review, reporting and quote drafting stay deterministic.
"""

import math
from datetime import UTC, datetime
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from gvas.config import OpenAISettings
from gvas.domain.messages import AudioReference, TranscriptResult
from gvas.domain.ports import AttachmentAccessPort
from gvas.domain.usage import UsageKind, UsageLedgerPort

TRANSCRIPTION_PATH: Final = "/audio/transcriptions"
DEFAULT_FILENAME: Final = "audio"
DEFAULT_MIME_TYPE: Final = "application/octet-stream"
# verbose_json is the documented format that returns the audio duration the
# usage ledger is counted in; the plain json format omits it.
RESPONSE_FORMAT: Final = "verbose_json"


class TranscriptionError(RuntimeError):
    """Raised when transcription should be retried by the dispatcher."""


class OpenAITranscriptionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str
    language: str | None = None
    duration: float | None = None


class OpenAITranscriber:
    """Transcribes owner audio through the documented transcription endpoint."""

    def __init__(
        self,
        settings: OpenAISettings,
        client: httpx.AsyncClient,
        attachment_access: AttachmentAccessPort,
        usage_ledger: UsageLedgerPort | None = None,
    ) -> None:
        if not settings.is_configured:
            raise TranscriptionError("openai api key is not configured")
        self._settings = settings
        self._client = client
        self._attachment_access = attachment_access
        self._usage_ledger = usage_ledger

    async def transcribe(self, audio: AudioReference) -> TranscriptResult:
        payload = await self._attachment_access.fetch(audio.attachment)
        if not payload.content:
            raise TranscriptionError("audio attachment carried no bytes")
        if len(payload.content) > self._settings.max_audio_bytes:
            raise TranscriptionError("audio attachment is larger than the configured limit")
        files = {
            "file": (
                payload.filename or DEFAULT_FILENAME,
                payload.content,
                payload.mime_type or DEFAULT_MIME_TYPE,
            )
        }
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}{TRANSCRIPTION_PATH}",
                data={
                    "model": self._settings.transcription_model,
                    "response_format": RESPONSE_FORMAT,
                },
                files=files,
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=self._settings.timeout_seconds,
            )
        except httpx.HTTPError as error:
            raise TranscriptionError("openai was unreachable") from error
        transcript = _transcript(response)
        await self._record_usage(audio, transcript)
        return transcript

    async def _record_usage(self, audio: AudioReference, transcript: TranscriptResult) -> None:
        if (
            self._usage_ledger is None
            or audio.business_id is None
            or transcript.duration_seconds is None
        ):
            return
        await self._usage_ledger.record(
            audio.business_id,
            UsageKind.TRANSCRIPTION_AUDIO_SECONDS,
            math.ceil(transcript.duration_seconds),
            at=datetime.now(UTC),
        )


def _transcript(response: httpx.Response) -> TranscriptResult:
    if response.status_code >= 400:
        raise TranscriptionError(f"openai returned http {response.status_code}")
    try:
        parsed = OpenAITranscriptionResponse.model_validate(response.json())
    except (ValueError, ValidationError) as error:
        raise TranscriptionError("openai returned an unreadable transcription") from error
    return TranscriptResult(
        text=parsed.text,
        language=parsed.language,
        duration_seconds=parsed.duration,
    )

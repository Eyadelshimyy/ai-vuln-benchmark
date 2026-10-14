import base64
import json
from uuid import uuid4

import httpx
import pytest

from gvas.config import OpenAISettings, ResendSettings
from gvas.domain.enums import MediaKind, RecipientAddressKind
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import (
    AttachmentPayload,
    AttachmentReference,
    AudioReference,
    CustomerDeliveryRequest,
    CustomerRecipient,
)
from gvas.domain.reporting import DOCX_MEDIA_TYPE, RenderedReportArtifact, ReportEmailRequest
from gvas.infrastructure.hosted_links import PORTAL_LOGIN_LINK_REFERENCE
from gvas.infrastructure.openai_transcription import OpenAITranscriber, TranscriptionError
from gvas.infrastructure.resend import (
    ResendDeliveryError,
    ResendQuoteDeliveryAdapter,
    ResendReportEmailAdapter,
)

OPENAI_KEY = "sk-test"
RESEND_KEY = "re-test"
BUSINESS_ID = BusinessId(uuid4())


class StubAttachmentAccess:
    def __init__(self, payload: AttachmentPayload) -> None:
        self._payload = payload

    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        return self._payload


def audio_reference() -> AudioReference:
    return AudioReference(
        attachment=AttachmentReference(
            attachment_id=uuid4(),
            media_kind=MediaKind.AUDIO,
            locator="channel-file:F1",
            mime_type="audio/mp4",
            filename="note.m4a",
        )
    )


def openai_settings(max_audio_bytes: int = 25 * 1024 * 1024) -> OpenAISettings:
    return OpenAISettings(api_key=OPENAI_KEY, max_audio_bytes=max_audio_bytes)


def resend_settings(reply_to_address: str = "") -> ResendSettings:
    return ResendSettings(
        api_key=RESEND_KEY,
        from_address="quotes@gudvector.com",
        reply_to_address=reply_to_address,
    )


def delivery_request(link: str = PORTAL_LOGIN_LINK_REFERENCE) -> CustomerDeliveryRequest:
    return CustomerDeliveryRequest(
        business_id=BUSINESS_ID,
        recipient=CustomerRecipient(
            address="person@example.com", address_kind=RecipientAddressKind.EMAIL
        ),
        idempotency_key="quote:1:v2",
        subject="Your quote",
        body_text="2 x Air sampling",
        links=(link,),
    )


@pytest.mark.asyncio
async def test_transcription_posts_audio_bytes_as_multipart_and_parses_the_text() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "job walk complete", "language": "en"})

    payload = AttachmentPayload(content=b"audio-bytes", mime_type="audio/mp4", filename="note.m4a")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transcriber = OpenAITranscriber(openai_settings(), client, StubAttachmentAccess(payload))
        result = await transcriber.transcribe(audio_reference())

    assert result.text == "job walk complete"
    request = seen[0]
    assert request.url.path.endswith("/audio/transcriptions")
    assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    body = request.read()
    assert b"audio-bytes" in body
    assert b"whisper-1" in body


@pytest.mark.asyncio
async def test_transcription_refuses_audio_over_the_configured_limit() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise AssertionError("oversized audio must not reach the provider")

    payload = AttachmentPayload(content=b"x" * 32, mime_type="audio/mp4")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transcriber = OpenAITranscriber(openai_settings(16), client, StubAttachmentAccess(payload))
        with pytest.raises(TranscriptionError):
            await transcriber.transcribe(audio_reference())


@pytest.mark.asyncio
async def test_transcription_error_is_sanitized() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {OPENAI_KEY}"}})

    payload = AttachmentPayload(content=b"audio", mime_type="audio/mp4")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transcriber = OpenAITranscriber(openai_settings(), client, StubAttachmentAccess(payload))
        with pytest.raises(TranscriptionError) as error:
            await transcriber.transcribe(audio_reference())

    assert OPENAI_KEY not in str(error.value)


@pytest.mark.asyncio
async def test_quote_email_carries_idempotency_key_and_resolved_portal_link() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "email-1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = ResendQuoteDeliveryAdapter(resend_settings("owner@gudvector.com"), client)
        receipt = await adapter.deliver(delivery_request())

    assert receipt.provider_message_id == "email-1"
    request = seen[0]
    assert request.headers["idempotency-key"] == "quote:1:v2"
    body = request.read().decode()
    assert "https://gudvector.com/portal/login" in body
    assert PORTAL_LOGIN_LINK_REFERENCE not in body
    assert "owner@gudvector.com" in body


@pytest.mark.asyncio
async def test_quote_email_refuses_an_unknown_hosted_link_reference() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise AssertionError("an unresolved link must not be emailed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = ResendQuoteDeliveryAdapter(resend_settings(), client)
        with pytest.raises(ResendDeliveryError):
            await adapter.deliver(delivery_request(link="unknown-link"))


@pytest.mark.asyncio
async def test_quote_email_refuses_a_non_email_recipient() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise AssertionError("resend only delivers email")

    request = CustomerDeliveryRequest(
        business_id=BUSINESS_ID,
        recipient=CustomerRecipient(
            address="+15555550100", address_kind=RecipientAddressKind.PHONE
        ),
        idempotency_key="quote:1:v2",
        body_text="2 x Air sampling",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = ResendQuoteDeliveryAdapter(resend_settings(), client)
        with pytest.raises(ResendDeliveryError):
            await adapter.deliver(request)


@pytest.mark.asyncio
async def test_quote_email_provider_error_is_sanitized() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"message": f"invalid key {RESEND_KEY}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = ResendQuoteDeliveryAdapter(resend_settings(), client)
        with pytest.raises(ResendDeliveryError) as error:
            await adapter.deliver(delivery_request())

    assert RESEND_KEY not in str(error.value)


def report_email_request() -> ReportEmailRequest:
    return ReportEmailRequest(
        business_id=BUSINESS_ID,
        recipient_address="client@example.com",
        idempotency_key="field_notes_report_email:v:client@example.com:m1",
        subject="Roof Inspection — report version 2",
        body_text="The approved report is attached.",
        artifact=RenderedReportArtifact(
            filename="roof-inspection-v2.docx",
            media_type=DOCX_MEDIA_TYPE,
            content=b"PK\x03\x04docx-bytes",
        ),
    )


@pytest.mark.asyncio
async def test_report_email_posts_the_docx_as_a_base64_attachment() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "email_9"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = ResendReportEmailAdapter(resend_settings("office@gudvector.com"), client)
        receipt = await adapter.deliver(report_email_request())

    assert receipt.provider_message_id == "email_9"
    request = seen[0]
    assert request.url.path.endswith("/emails")
    assert request.headers["authorization"] == f"Bearer {RESEND_KEY}"
    assert request.headers["idempotency-key"] == "field_notes_report_email:v:client@example.com:m1"
    body = json.loads(request.read())
    assert body["from"] == "quotes@gudvector.com"
    assert body["reply_to"] == "office@gudvector.com"
    assert body["to"] == ["client@example.com"]
    assert body["subject"] == "Roof Inspection — report version 2"
    assert body["text"] == "The approved report is attached."
    assert body["attachments"] == [
        {
            "filename": "roof-inspection-v2.docx",
            "content": base64.b64encode(b"PK\x03\x04docx-bytes").decode("ascii"),
            "content_type": DOCX_MEDIA_TYPE,
        }
    ]


@pytest.mark.asyncio
async def test_report_email_provider_error_is_sanitized() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": f"boom {RESEND_KEY}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = ResendReportEmailAdapter(resend_settings(), client)
        with pytest.raises(ResendDeliveryError) as error:
            await adapter.deliver(report_email_request())

    assert RESEND_KEY not in str(error.value)
    assert "boom" not in str(error.value)

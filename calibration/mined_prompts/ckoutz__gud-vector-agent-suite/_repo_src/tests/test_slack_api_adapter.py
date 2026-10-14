from uuid import uuid4

import httpx
import pytest

from gvas.domain.enums import MediaKind
from gvas.domain.messages import AttachmentReference
from gvas.infrastructure.slack.api import (
    SlackAttachmentError,
    SlackFileAttachmentAccess,
    SlackWebApiChatPoster,
    SlackWebApiFileUploader,
)
from gvas.infrastructure.slack.config import SlackSettings
from gvas.infrastructure.slack.delivery import (
    SlackChatPostRequest,
    SlackDeliveryError,
    SlackFileUploadRequest,
    SlackUploadFile,
)

BOT_TOKEN = "xoxb-test-token"  # noqa: S105
FILE_ID = "F123"


def settings(attachment_max_bytes: int = 25 * 1024 * 1024) -> SlackSettings:
    return SlackSettings(
        signing_secret="secret",  # noqa: S106
        bot_token=BOT_TOKEN,
        installations="T1:U1:11111111-1111-1111-1111-111111111111",
        attachment_max_bytes=attachment_max_bytes,
    )


def client_for(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


def audio_attachment(locator: str = f"slack-file:{FILE_ID}") -> AttachmentReference:
    return AttachmentReference(
        attachment_id=uuid4(),
        media_kind=MediaKind.AUDIO,
        locator=locator,
        mime_type="audio/mp4",
        filename="note.m4a",
    )


@pytest.mark.asyncio
async def test_chat_post_sends_thread_and_delivery_key_and_returns_ts() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "ts": "1700000000.000100"})

    async with client_for(httpx.MockTransport(handle)) as client:
        poster = SlackWebApiChatPoster(settings(), client)
        result = await poster.post_message(
            SlackChatPostRequest(
                channel="C1",
                thread_ts="1699999999.000100",
                text="hello",
                idempotency_key="business:C1:correlation",
            )
        )

    assert result.message_ts == "1700000000.000100"
    request = seen[0]
    assert request.url.path.endswith("/chat.postMessage")
    assert request.headers["authorization"] == f"Bearer {BOT_TOKEN}"
    body = request.read().decode()
    assert '"thread_ts":"1699999999.000100"' in body.replace(" ", "")
    assert "business:C1:correlation" in body


@pytest.mark.asyncio
async def test_chat_post_reports_slack_rejection_without_provider_payload() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    async with client_for(httpx.MockTransport(handle)) as client:
        result = await SlackWebApiChatPoster(settings(), client).post_message(
            SlackChatPostRequest(channel="C1", text="hello", idempotency_key="key")
        )

    assert result.message_ts is None
    assert result.detail is not None
    assert BOT_TOKEN not in result.detail


@pytest.mark.asyncio
async def test_chat_post_transport_failure_is_retryable_and_redacted() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection to {BOT_TOKEN} refused")

    async with client_for(httpx.MockTransport(handle)) as client:
        poster = SlackWebApiChatPoster(settings(), client)
        with pytest.raises(SlackDeliveryError) as error:
            await poster.post_message(
                SlackChatPostRequest(channel="C1", text="hello", idempotency_key="key")
            )

    assert BOT_TOKEN not in str(error.value)


@pytest.mark.asyncio
async def test_attachment_fetch_validates_metadata_and_returns_bytes() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files.info"):
            assert request.url.params["file"] == FILE_ID
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "id": FILE_ID,
                        "name": "note.m4a",
                        "mimetype": "audio/mp4",
                        "size": 5,
                        "url_private_download": "https://files.slack.com/note.m4a",
                    },
                },
            )
        assert request.headers["authorization"] == f"Bearer {BOT_TOKEN}"
        return httpx.Response(200, content=b"bytes")

    async with client_for(httpx.MockTransport(handle)) as client:
        payload = await SlackFileAttachmentAccess(settings(), client).fetch(audio_attachment())

    assert payload.content == b"bytes"
    assert payload.mime_type == "audio/mp4"


@pytest.mark.asyncio
async def test_attachment_fetch_rejects_a_locator_from_another_channel() -> None:
    async with client_for(httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        with pytest.raises(SlackAttachmentError):
            await SlackFileAttachmentAccess(settings(), client).fetch(
                audio_attachment(locator="telnyx-media:1")
            )


@pytest.mark.asyncio
async def test_attachment_fetch_rejects_a_file_larger_than_the_limit() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "file": {
                    "id": FILE_ID,
                    "mimetype": "audio/mp4",
                    "size": 4096,
                    "url_private_download": "https://files.slack.com/note.m4a",
                },
            },
        )

    async with client_for(httpx.MockTransport(handle)) as client:
        access = SlackFileAttachmentAccess(settings(1024), client)
        with pytest.raises(SlackAttachmentError):
            await access.fetch(audio_attachment())


@pytest.mark.asyncio
async def test_attachment_fetch_refuses_a_download_host_outside_slack() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files.info"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "id": FILE_ID,
                        "mimetype": "audio/mp4",
                        "size": 5,
                        "url_private_download": "https://attacker.example.com/note.m4a",
                    },
                },
            )
        raise AssertionError("the download must not be attempted")

    async with client_for(httpx.MockTransport(handle)) as client:
        with pytest.raises(SlackAttachmentError):
            await SlackFileAttachmentAccess(settings(), client).fetch(audio_attachment())


@pytest.mark.asyncio
async def test_attachment_fetch_refuses_a_redirected_download() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files.info"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "id": FILE_ID,
                        "mimetype": "audio/mp4",
                        "size": 5,
                        "url_private_download": "https://files.slack.com/note.m4a",
                    },
                },
            )
        return httpx.Response(302, headers={"location": "https://attacker.example.com/note.m4a"})

    async with client_for(httpx.MockTransport(handle)) as client:
        with pytest.raises(SlackAttachmentError):
            await SlackFileAttachmentAccess(settings(), client).fetch(audio_attachment())


@pytest.mark.asyncio
async def test_attachment_fetch_refuses_non_audio_metadata_for_an_audio_reference() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "file": {
                    "id": FILE_ID,
                    "mimetype": "application/pdf",
                    "size": 5,
                    "url_private_download": "https://files.slack.com/note.pdf",
                },
            },
        )

    async with client_for(httpx.MockTransport(handle)) as client:
        with pytest.raises(SlackAttachmentError):
            await SlackFileAttachmentAccess(settings(), client).fetch(audio_attachment())


def upload_request(**overrides: object) -> SlackFileUploadRequest:
    base: dict[str, object] = {
        "channel": "C1",
        "thread_ts": "1699999999.000100",
        "files": (SlackUploadFile(filename="report-v1.docx", content=b"PK\x03\x04docx"),),
        "initial_comment": "Approved report",
        "idempotency_key": "business:C1:publish",
    }
    base.update(overrides)
    return SlackFileUploadRequest.model_validate(base)


@pytest.mark.asyncio
async def test_file_upload_reserves_puts_and_completes_into_the_thread() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/files.getUploadURLExternal"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "upload_url": "https://files.slack.com/upload/v1/abc",
                    "file_id": FILE_ID,
                },
            )
        if request.url.host == "files.slack.com":
            return httpx.Response(200, text="OK - 8")
        if request.url.path.endswith("/files.completeUploadExternal"):
            return httpx.Response(200, json={"ok": True, "files": [{"id": FILE_ID}]})
        raise AssertionError(request.url)

    async with client_for(httpx.MockTransport(handle)) as client:
        uploader = SlackWebApiFileUploader(settings(), client)
        result = await uploader.upload_files(upload_request())

    assert result.file_ids == (FILE_ID,)
    reserve, put, complete = seen
    assert reserve.headers["authorization"] == f"Bearer {BOT_TOKEN}"
    reserve_body = reserve.read().decode()
    assert "filename=report-v1.docx" in reserve_body
    assert "length=8" in reserve_body
    assert put.read() == b"PK\x03\x04docx"
    assert "authorization" not in put.headers
    assert complete.headers["authorization"] == f"Bearer {BOT_TOKEN}"
    complete_body = complete.read().decode().replace(" ", "")
    assert f'"files":[{{"id":"{FILE_ID}"}}]' in complete_body
    assert '"channel_id":"C1"' in complete_body
    assert '"thread_ts":"1699999999.000100"' in complete_body
    assert '"initial_comment":"Approvedreport"' in complete_body


@pytest.mark.asyncio
async def test_file_upload_refuses_an_upload_location_outside_slack() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files.getUploadURLExternal"):
            return httpx.Response(
                200,
                json={"ok": True, "upload_url": "https://evil.example/upload", "file_id": FILE_ID},
            )
        raise AssertionError(request.url)

    async with client_for(httpx.MockTransport(handle)) as client:
        uploader = SlackWebApiFileUploader(settings(), client)
        with pytest.raises(SlackDeliveryError, match="outside slack"):
            await uploader.upload_files(upload_request())


@pytest.mark.asyncio
async def test_file_upload_reports_completion_rejection_without_provider_payload() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files.getUploadURLExternal"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "upload_url": "https://files.slack.com/upload/v1/abc",
                    "file_id": FILE_ID,
                },
            )
        if request.url.host == "files.slack.com":
            return httpx.Response(200)
        return httpx.Response(
            200, json={"ok": False, "error": "not_in_channel", "secret": "do-not-leak"}
        )

    async with client_for(httpx.MockTransport(handle)) as client:
        uploader = SlackWebApiFileUploader(settings(), client)
        result = await uploader.upload_files(upload_request())

    assert result.file_ids == ()
    assert result.detail == "slack rejected the file upload: not_in_channel"


@pytest.mark.asyncio
async def test_file_upload_transport_failure_is_retryable_and_redacted() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("token xoxb-secret leaked in connect error")

    async with client_for(httpx.MockTransport(handle)) as client:
        uploader = SlackWebApiFileUploader(settings(), client)
        with pytest.raises(SlackDeliveryError) as error:
            await uploader.upload_files(upload_request())

    assert "xoxb" not in str(error.value)

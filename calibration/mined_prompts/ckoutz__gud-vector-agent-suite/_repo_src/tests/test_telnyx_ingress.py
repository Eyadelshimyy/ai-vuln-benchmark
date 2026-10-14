import base64
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.enums import SenderRole
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import TextPart
from gvas.infrastructure.models import InboundMessage, OutboxMessage, WorkflowRun
from gvas.infrastructure.telnyx.config import TelnyxSettings
from gvas.infrastructure.telnyx.events import inbound_message_of, parse_webhook
from gvas.infrastructure.telnyx.ingress import TelnyxIngressResult
from gvas.infrastructure.telnyx.installations import (
    StaticTelnyxInstallationDirectory,
    TelnyxInstallationError,
)
from gvas.infrastructure.telnyx.normalization import TelnyxNormalizationError
from gvas.infrastructure.telnyx.signature import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    TelnyxSignatureError,
    verify_signature,
)
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.telnyx import create_telnyx_router
from telnyx_fixtures import (
    EVENT_ID,
    MESSAGE_ID,
    OTHER_PUBLIC_KEY,
    OWNER_NUMBER,
    PUBLIC_KEY,
    REQUEST_NOW,
    REQUEST_TIMESTAMP,
    STRANGER_NUMBER,
    TELNYX_NUMBER,
    build_ingress,
    message_payload,
    normalize,
    seed_business,
    sign,
    signed_request,
    status_payload,
)

MAX_AGE = timedelta(seconds=300)


def verify(body: bytes, signature: str | None, timestamp: str | None, **overrides: object) -> None:
    kwargs: dict[str, object] = {
        "public_key": PUBLIC_KEY,
        "body": body,
        "signature": signature,
        "timestamp": timestamp,
        "now": REQUEST_NOW,
        "max_age": MAX_AGE,
    }
    kwargs.update(overrides)
    verify_signature(**kwargs)  # type: ignore[arg-type]


def test_signature_verification_accepts_the_official_ed25519_format() -> None:
    body, headers = signed_request(message_payload())
    verify(body, headers[SIGNATURE_HEADER], headers[TIMESTAMP_HEADER])


def test_signature_verification_rejects_tampering_wrong_key_and_missing_headers() -> None:
    body, headers = signed_request(message_payload())
    signature = headers[SIGNATURE_HEADER]
    with pytest.raises(TelnyxSignatureError):
        verify(body + b" ", signature, REQUEST_TIMESTAMP)
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, str(int(REQUEST_TIMESTAMP) + 1))
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, REQUEST_TIMESTAMP, public_key=OTHER_PUBLIC_KEY)
    with pytest.raises(TelnyxSignatureError):
        verify(body, None, REQUEST_TIMESTAMP)
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, None)
    with pytest.raises(TelnyxSignatureError):
        verify(body, "not base64!", REQUEST_TIMESTAMP)
    with pytest.raises(TelnyxSignatureError):
        verify(body, base64.b64encode(b"short").decode(), REQUEST_TIMESTAMP)
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, "not-a-timestamp")
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, REQUEST_TIMESTAMP, public_key="not a key")


def test_signature_verification_rejects_stale_and_future_timestamps() -> None:
    body, headers = signed_request(message_payload())
    signature = headers[SIGNATURE_HEADER]
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, REQUEST_TIMESTAMP, now=REQUEST_NOW + timedelta(seconds=301))
    with pytest.raises(TelnyxSignatureError):
        verify(body, signature, REQUEST_TIMESTAMP, now=REQUEST_NOW - timedelta(seconds=301))
    verify(body, signature, REQUEST_TIMESTAMP, now=REQUEST_NOW + timedelta(seconds=299))
    stale = str(int(REQUEST_TIMESTAMP) - 600)
    with pytest.raises(TelnyxSignatureError):
        verify(body, sign(body, stale), stale)


def test_normalization_maps_a_text_to_the_shared_inbound_envelope() -> None:
    business_id = BusinessId(uuid4())
    inbound = normalize(message_payload(text="  quote: replace the water heater  "), business_id)

    assert inbound.endpoint.source_namespace == "telnyx"
    assert inbound.endpoint.external_endpoint_id == TELNYX_NUMBER
    assert inbound.endpoint.business_id == business_id
    assert inbound.message.message_key == f"{TELNYX_NUMBER}:{MESSAGE_ID}"
    assert inbound.message.conversation_ref.external_conversation_id == (
        f"{TELNYX_NUMBER}:{OWNER_NUMBER}"
    )
    assert inbound.message.sender.external_id == OWNER_NUMBER
    assert inbound.message.sender.role is SenderRole.OWNER
    assert inbound.message.received_at == datetime(2025, 1, 1, 0, 0, 0, 100000, tzinfo=UTC)
    assert inbound.message.reply_to is None
    assert inbound.message.parts == (TextPart(text="quote: replace the water heater"),)
    assert inbound.routing["from"] == TELNYX_NUMBER
    assert inbound.routing["to"] == OWNER_NUMBER
    assert inbound.routing["event_id"] == EVENT_ID
    assert inbound.routing["messaging_profile_id"] == "profile-1"


def test_every_text_from_the_owner_continues_the_same_conversation() -> None:
    business_id = BusinessId(uuid4())
    first = normalize(message_payload(), business_id)
    second = normalize(message_payload(id="second-message", text="approve"), business_id)

    assert first.message.conversation_ref == second.message.conversation_ref
    assert first.message.message_key != second.message.message_key


def test_mms_media_never_enters_the_domain_and_media_only_messages_have_no_content() -> None:
    business_id = BusinessId(uuid4())
    media = [
        {"url": "https://media.telnyx.example/photo.jpg", "content_type": "image/jpeg"},
        {"url": "https://media.telnyx.example/memo.m4a", "content_type": "audio/mp4"},
    ]
    inbound = normalize(
        message_payload(type="MMS", text="quote: fix the gate", media=media), business_id
    )

    assert inbound.message.parts == (TextPart(text="quote: fix the gate"),)
    assert inbound.routing["media_count"] == 2
    assert "telnyx.example" not in repr(inbound)
    with pytest.raises(TelnyxNormalizationError):
        normalize(message_payload(type="MMS", text=None, media=media), business_id)
    with pytest.raises(TelnyxNormalizationError):
        normalize(message_payload(text="   "), business_id)


def test_status_events_carry_no_owner_message() -> None:
    assert inbound_message_of(parse_webhook(status_payload("message.sent"))) is None
    assert inbound_message_of(parse_webhook(status_payload("message.finalized"))) is None
    outbound_received = message_payload(direction="outbound")
    assert inbound_message_of(parse_webhook(outbound_received)) is None


@pytest.mark.asyncio
async def test_installation_directory_parses_e164_numbers_and_business_ids() -> None:
    business_id = uuid4()
    directory = StaticTelnyxInstallationDirectory.from_setting(
        f" {OWNER_NUMBER}={business_id}:{TELNYX_NUMBER} "
    )
    found = await directory.find(TELNYX_NUMBER)
    assert found is not None
    assert found.business_id == business_id
    assert found.owner_numbers == frozenset({OWNER_NUMBER})
    assert found.is_authorized_owner(OWNER_NUMBER)
    assert not found.is_authorized_owner(STRANGER_NUMBER)
    assert await directory.find(STRANGER_NUMBER) is None
    with pytest.raises(TelnyxInstallationError):
        StaticTelnyxInstallationDirectory.from_setting(f"{OWNER_NUMBER}=not-a-uuid:{TELNYX_NUMBER}")
    with pytest.raises(TelnyxInstallationError):
        StaticTelnyxInstallationDirectory.from_setting(OWNER_NUMBER)
    with pytest.raises(TelnyxInstallationError):
        StaticTelnyxInstallationDirectory.from_setting(f"{OWNER_NUMBER}={business_id}")
    with pytest.raises(TelnyxInstallationError):
        StaticTelnyxInstallationDirectory.from_setting(f"5550100001={business_id}:{TELNYX_NUMBER}")
    with pytest.raises(TelnyxInstallationError):
        StaticTelnyxInstallationDirectory.from_setting(f"{OWNER_NUMBER}={business_id}:15550100000")


def test_settings_are_optional_as_a_set() -> None:
    assert not TelnyxSettings().is_configured
    assert not TelnyxSettings().is_partially_configured
    partial = TelnyxSettings(public_key=PUBLIC_KEY)
    assert partial.is_partially_configured
    assert not partial.is_configured
    complete = TelnyxSettings(
        public_key=PUBLIC_KEY,
        api_key="KEY",
        installations=f"{OWNER_NUMBER}={uuid4()}:{TELNYX_NUMBER}",
    )
    assert complete.is_configured
    assert not complete.is_partially_configured


@pytest.mark.asyncio
async def test_ingress_persists_one_process_command_and_no_processing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    body, headers = signed_request(message_payload())

    outcome = await ingress.handle(
        body=body, signature=headers[SIGNATURE_HEADER], timestamp=headers[TIMESTAMP_HEADER]
    )

    assert outcome.result is TelnyxIngressResult.ACCEPTED
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 0
        command = await session.scalar(select(OutboxMessage))
        assert command is not None
        assert command.command_type == "owner_message.process"


@pytest.mark.asyncio
async def test_telnyx_redelivery_of_the_same_message_is_deduplicated(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    body, headers = signed_request(message_payload())

    first = await ingress.handle(
        body=body, signature=headers[SIGNATURE_HEADER], timestamp=headers[TIMESTAMP_HEADER]
    )
    second = await ingress.handle(
        body=body, signature=headers[SIGNATURE_HEADER], timestamp=headers[TIMESTAMP_HEADER]
    )

    assert first.result is TelnyxIngressResult.ACCEPTED
    assert second.result is TelnyxIngressResult.DUPLICATE
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 1


@pytest.mark.asyncio
async def test_unknown_senders_and_numbers_are_ignored_without_persistence(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)

    body, headers = signed_request(message_payload(**{"from": {"phone_number": STRANGER_NUMBER}}))
    stranger = await ingress.handle(
        body=body, signature=headers[SIGNATURE_HEADER], timestamp=headers[TIMESTAMP_HEADER]
    )
    body, headers = signed_request(message_payload(to=[{"phone_number": "+15550109999"}]))
    unknown_number = await ingress.handle(
        body=body, signature=headers[SIGNATURE_HEADER], timestamp=headers[TIMESTAMP_HEADER]
    )
    body, headers = signed_request(status_payload())
    status = await ingress.handle(
        body=body, signature=headers[SIGNATURE_HEADER], timestamp=headers[TIMESTAMP_HEADER]
    )

    assert stranger.result is TelnyxIngressResult.UNAUTHORIZED_SENDER
    assert unknown_number.result is TelnyxIngressResult.UNKNOWN_INSTALLATION
    assert status.result is TelnyxIngressResult.IGNORED
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 0


@pytest.mark.asyncio
async def test_http_route_maps_signature_payload_and_ignored_outcomes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    app = create_app(routers=(create_telnyx_router(build_ingress(session_factory, business_id)),))

    with TestClient(app) as client:
        body, headers = signed_request(message_payload())
        accepted = client.post("/telnyx/messaging", content=body, headers=headers)
        assert accepted.status_code == 200
        assert accepted.json()["status"] == "accepted"

        body, headers = signed_request(
            message_payload(**{"from": {"phone_number": STRANGER_NUMBER}})
        )
        stranger = client.post("/telnyx/messaging", content=body, headers=headers)
        assert stranger.status_code == 200
        assert stranger.json()["status"] == "unauthorized_sender"

        body, headers = signed_request(status_payload())
        status = client.post("/telnyx/messaging", content=body, headers=headers)
        assert status.status_code == 200
        assert status.json()["status"] == "ignored"

        body, headers = signed_request(message_payload())
        headers[SIGNATURE_HEADER] = sign(body + b" ")
        forged = client.post("/telnyx/messaging", content=body, headers=headers)
        assert forged.status_code == 401
        assert forged.json() == {"status": "invalid_signature"}

        stale = str(int(REQUEST_TIMESTAMP) - 3600)
        body, headers = signed_request(message_payload(), timestamp=stale)
        replayed = client.post("/telnyx/messaging", content=body, headers=headers)
        assert replayed.status_code == 401

        body, headers = signed_request({"data": {"event_type": "message.received"}})
        malformed = client.post("/telnyx/messaging", content=body, headers=headers)
        assert malformed.status_code == 400
        assert malformed.json() == {"status": "invalid_payload"}

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1

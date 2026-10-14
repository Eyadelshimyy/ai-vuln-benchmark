from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.enums import MediaKind, SenderRole
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentPart, TextPart
from gvas.infrastructure.models import InboundMessage, OutboxMessage, WorkflowRun
from gvas.infrastructure.slack.config import SlackSettings
from gvas.infrastructure.slack.events import parse_envelope
from gvas.infrastructure.slack.ingress import SlackIngressResult
from gvas.infrastructure.slack.installations import (
    SlackInstallationError,
    StaticSlackInstallationDirectory,
)
from gvas.infrastructure.slack.normalization import SlackNormalizationError
from gvas.infrastructure.slack.signature import (
    SlackSignatureError,
    compute_signature,
    verify_signature,
)
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.slack import create_slack_router
from slack_fixtures import (
    APP_ID,
    CHANNEL,
    NON_OWNER_USER,
    OWNER_USER,
    REQUEST_NOW,
    REQUEST_TIMESTAMP,
    ROOT_TS,
    SIGNING_SECRET,
    TEAM_ID,
    build_ingress,
    message_payload,
    normalize,
    seed_business,
    signed_request,
)


def test_signature_verification_rejects_tampered_body_and_stale_timestamp() -> None:
    body, headers = signed_request(message_payload())
    with pytest.raises(SlackSignatureError):
        verify_signature(
            signing_secret=SIGNING_SECRET,
            body=body + b" ",
            signature=headers["X-Slack-Signature"],
            timestamp=REQUEST_TIMESTAMP,
            now=REQUEST_NOW,
            max_age=timedelta(seconds=300),
        )
    with pytest.raises(SlackSignatureError):
        verify_signature(
            signing_secret=SIGNING_SECRET,
            body=body,
            signature=headers["X-Slack-Signature"],
            timestamp=REQUEST_TIMESTAMP,
            now=REQUEST_NOW + timedelta(seconds=301),
            max_age=timedelta(seconds=300),
        )
    with pytest.raises(SlackSignatureError):
        verify_signature(
            signing_secret=SIGNING_SECRET,
            body=body,
            signature=None,
            timestamp=REQUEST_TIMESTAMP,
            now=REQUEST_NOW,
            max_age=timedelta(seconds=300),
        )
    verify_signature(
        signing_secret=SIGNING_SECRET,
        body=body,
        signature=headers["X-Slack-Signature"],
        timestamp=REQUEST_TIMESTAMP,
        now=REQUEST_NOW,
        max_age=timedelta(seconds=300),
    )


def test_normalization_maps_slack_message_to_inbound_envelope() -> None:
    business_id = BusinessId(uuid4())
    inbound = normalize(message_payload(), business_id)

    assert inbound.endpoint.source_namespace == "slack"
    assert inbound.endpoint.external_endpoint_id == f"{TEAM_ID}/{APP_ID}"
    assert inbound.message.message_key == f"{CHANNEL}:{ROOT_TS}"
    assert inbound.message.conversation_ref.external_conversation_id == (f"{CHANNEL}:{ROOT_TS}")
    assert inbound.message.sender.role is SenderRole.OWNER
    assert inbound.message.received_at == datetime(2025, 1, 1, 0, 0, 0, 100, tzinfo=UTC)
    assert inbound.message.reply_to is None
    assert inbound.message.parts == (TextPart(text="replace the water heater"),)
    assert inbound.routing["channel"] == CHANNEL
    assert inbound.routing["thread_ts"] == ROOT_TS
    assert inbound.routing["event_id"] == "Ev00000FAKE"


def test_normalization_unwraps_slack_mailto_links() -> None:
    business_id = BusinessId(uuid4())
    inbound = normalize(
        message_payload(
            text=(
                "quote:\n"
                "customer: <mailto:person@example.com|person@example.com>\n"
                "currency: USD\n"
                "item: 1 | Inspection | 125.00"
            )
        ),
        business_id,
    )

    assert inbound.message.parts == (
        TextPart(
            text=(
                "quote:\ncustomer: person@example.com\ncurrency: USD\nitem: 1 | Inspection | 125.00"
            )
        ),
    )


def test_thread_replies_share_the_root_conversation_and_carry_reply_correlation() -> None:
    business_id = BusinessId(uuid4())
    root = normalize(message_payload(), business_id)
    reply = normalize(
        message_payload(
            ts="1735689700.000200",
            event_ts="1735689700.000200",
            thread_ts=ROOT_TS,
        ),
        business_id,
    )

    assert reply.message.conversation_ref == root.message.conversation_ref
    assert reply.message.message_key != root.message.message_key
    assert reply.message.reply_to is not None
    assert reply.message.reply_to.external_message_id == ROOT_TS


def test_files_and_voice_memos_become_ordered_attachment_parts() -> None:
    business_id = BusinessId(uuid4())
    inbound = normalize(
        message_payload(
            subtype="file_share",
            text="field notes: replaced the anode rod",
            files=[
                {
                    "id": "F00000AUDIO",
                    "name": "audio_message.webm",
                    "mimetype": "audio/webm",
                    "subtype": "slack_audio",
                    "size": 2048,
                },
                {"id": "F00000PHOTO", "name": "before.jpg", "mimetype": "image/jpeg"},
            ],
        ),
        business_id,
    )

    assert isinstance(inbound.message.parts[0], TextPart)
    attachments = [
        part.attachment for part in inbound.message.parts if isinstance(part, AttachmentPart)
    ]
    assert [attachment.media_kind for attachment in attachments] == [
        MediaKind.AUDIO,
        MediaKind.IMAGE,
    ]
    assert [attachment.locator for attachment in attachments] == [
        "slack-file:F00000AUDIO",
        "slack-file:F00000PHOTO",
    ]
    repeated = normalize(
        message_payload(
            subtype="file_share",
            text="field notes: replaced the anode rod",
            files=[
                {
                    "id": "F00000AUDIO",
                    "name": "audio_message.webm",
                    "mimetype": "audio/webm",
                    "subtype": "slack_audio",
                    "size": 2048,
                },
                {"id": "F00000PHOTO", "name": "before.jpg", "mimetype": "image/jpeg"},
            ],
        ),
        business_id,
    )
    assert repeated.message.parts == inbound.message.parts


def test_normalization_rejects_events_without_content() -> None:
    business_id = BusinessId(uuid4())
    with pytest.raises(SlackNormalizationError):
        normalize(message_payload(text="   "), business_id)


def test_unsupported_envelopes_and_events_are_not_parsed() -> None:
    assert parse_envelope({"type": "event_callback", "event": {"type": "reaction_added"}}) is None
    assert parse_envelope({"type": "something_else"}) is None


@pytest.mark.asyncio
async def test_installation_directory_parses_configured_workspaces_and_owner_users() -> None:
    business_id = uuid4()
    directory = StaticSlackInstallationDirectory.from_setting(
        f" {TEAM_ID}={business_id}:{OWNER_USER}|{OWNER_USER} "
    )
    found = await directory.find(TEAM_ID, APP_ID)
    assert found is not None
    assert found.business_id == business_id
    assert found.owner_user_ids == frozenset({OWNER_USER})
    assert found.is_authorized_owner(OWNER_USER)
    assert not found.is_authorized_owner(NON_OWNER_USER)
    assert await directory.find("T00000OTHER", APP_ID) is None
    with pytest.raises(SlackInstallationError):
        StaticSlackInstallationDirectory.from_setting(f"T1=not-a-uuid:{OWNER_USER}")
    with pytest.raises(SlackInstallationError):
        StaticSlackInstallationDirectory.from_setting("T1")
    with pytest.raises(SlackInstallationError):
        StaticSlackInstallationDirectory.from_setting(f"T1={business_id}")
    with pytest.raises(SlackInstallationError):
        StaticSlackInstallationDirectory.from_setting(f"T1={business_id}: ")


@pytest.mark.asyncio
async def test_ingress_persists_one_process_command_and_no_processing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    body, headers = signed_request(message_payload())

    outcome = await ingress.handle(
        body=body,
        signature=headers["X-Slack-Signature"],
        timestamp=headers["X-Slack-Request-Timestamp"],
    )

    assert outcome.result is SlackIngressResult.ACCEPTED
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 0
        command = await session.scalar(select(OutboxMessage))
        assert command is not None
        assert command.command_type == "owner_message.process"


@pytest.mark.asyncio
async def test_slack_retry_of_the_same_event_is_deduplicated(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    body, headers = signed_request(message_payload())

    first = await ingress.handle(
        body=body,
        signature=headers["X-Slack-Signature"],
        timestamp=headers["X-Slack-Request-Timestamp"],
    )
    retried = await ingress.handle(
        body=body,
        signature=headers["X-Slack-Signature"],
        timestamp=headers["X-Slack-Request-Timestamp"],
        retry_num="1",
        retry_reason="http_timeout",
    )

    assert first.result is SlackIngressResult.ACCEPTED
    assert retried.result is SlackIngressResult.DUPLICATE
    assert retried.detail == "retry 1: http_timeout"
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 1


@pytest.mark.asyncio
async def test_unknown_workspace_is_acknowledged_without_persistence(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    body, headers = signed_request(message_payload() | {"team_id": "T00000OTHER"})

    outcome = await ingress.handle(
        body=body,
        signature=headers["X-Slack-Signature"],
        timestamp=headers["X-Slack-Request-Timestamp"],
    )

    assert outcome.result is SlackIngressResult.UNKNOWN_INSTALLATION
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 0


@pytest.mark.asyncio
async def test_unauthorized_workspace_member_is_not_ingested(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    body, headers = signed_request(message_payload(user=NON_OWNER_USER))

    outcome = await ingress.handle(
        body=body,
        signature=headers["X-Slack-Signature"],
        timestamp=headers["X-Slack-Request-Timestamp"],
    )

    assert outcome.result is SlackIngressResult.UNAUTHORIZED_SENDER
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 0


def test_normalization_rejects_a_sender_outside_the_configured_owners() -> None:
    with pytest.raises(SlackNormalizationError):
        normalize(message_payload(user=NON_OWNER_USER), BusinessId(uuid4()))


@pytest.mark.asyncio
async def test_bot_messages_and_edits_are_ignored(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    for payload in (
        message_payload(bot_id="B00000FAKE"),
        message_payload(subtype="message_changed"),
    ):
        body, headers = signed_request(payload)
        outcome = await ingress.handle(
            body=body,
            signature=headers["X-Slack-Signature"],
            timestamp=headers["X-Slack-Request-Timestamp"],
        )
        assert outcome.result is SlackIngressResult.IGNORED
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 0


@pytest.mark.asyncio
async def test_request_url_endpoint_acknowledges_challenge_and_events(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingress = build_ingress(session_factory, business_id)
    app = create_app(routers=(create_slack_router(ingress, path="/slack/events"),))

    with TestClient(app) as client:
        challenge_body, challenge_headers = signed_request(
            {"type": "url_verification", "challenge": "fake-challenge"}
        )
        challenge = client.post("/slack/events", content=challenge_body, headers=challenge_headers)

        body, headers = signed_request(message_payload())
        accepted = client.post("/slack/events", content=body, headers=headers)
        invalid = client.post(
            "/slack/events",
            content=body,
            headers={**headers, "X-Slack-Signature": "v0=deadbeef"},
        )
        malformed_body = b"not-json"
        malformed = client.post(
            "/slack/events",
            content=malformed_body,
            headers={
                **headers,
                "X-Slack-Signature": compute_signature(
                    SIGNING_SECRET, REQUEST_TIMESTAMP, malformed_body
                ),
            },
        )

    assert challenge.status_code == 200
    assert challenge.json() == {"challenge": "fake-challenge"}
    assert accepted.status_code == 200
    assert accepted.json() == {"status": "accepted"}
    assert invalid.status_code == 401
    assert malformed.status_code == 400
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 0


def test_slack_router_is_not_mounted_by_default() -> None:
    paths = [route.path for route in create_app().routes if isinstance(route, APIRoute)]
    assert paths == ["/healthz"]


def test_slack_settings_use_environment_names_only() -> None:
    settings = SlackSettings(_env_file=None)
    assert settings.signing_secret == ""
    assert settings.events_path == "/slack/events"
    assert settings.request_max_age_seconds == 300
    assert settings.installations == ""

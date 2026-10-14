import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import InboundOwnerMessage
from gvas.infrastructure.models import Business
from gvas.infrastructure.slack.events import SlackEventCallback, parse_envelope
from gvas.infrastructure.slack.ingress import SlackEventIngress
from gvas.infrastructure.slack.installations import (
    SlackInstallation,
    StaticSlackInstallationDirectory,
)
from gvas.infrastructure.slack.normalization import normalize_event
from gvas.infrastructure.slack.signature import compute_signature
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory

SIGNING_SECRET = "fake-signing-secret"  # noqa: S105 - fake value for tests
TEAM_ID = "T00000FAKE"
APP_ID = "A00000FAKE"
CHANNEL = "C00000FAKE"
OWNER_USER = "U00000FAKE"
NON_OWNER_USER = "U00001FAKE"
ROOT_TS = "1735689600.000100"
REQUEST_NOW = datetime(2025, 1, 1, tzinfo=UTC)
REQUEST_TIMESTAMP = str(int(REQUEST_NOW.timestamp()))


def message_payload(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "message",
        "channel": CHANNEL,
        "channel_type": "im",
        "user": OWNER_USER,
        "text": "replace the water heater",
        "ts": ROOT_TS,
        "event_ts": ROOT_TS,
    }
    event.update(overrides)
    return {
        "type": "event_callback",
        "team_id": TEAM_ID,
        "api_app_id": APP_ID,
        "event_id": "Ev00000FAKE",
        "event_time": 1735689600,
        "event": event,
    }


def signed_request(payload: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload).encode()
    return body, {
        "X-Slack-Signature": compute_signature(SIGNING_SECRET, REQUEST_TIMESTAMP, body),
        "X-Slack-Request-Timestamp": REQUEST_TIMESTAMP,
        "Content-Type": "application/json",
    }


def installation(business_id: BusinessId) -> SlackInstallation:
    return SlackInstallation(
        business_id=business_id,
        team_id=TEAM_ID,
        api_app_id=APP_ID,
        owner_user_ids=frozenset({OWNER_USER}),
    )


def build_ingress(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> SlackEventIngress:
    return SlackEventIngress(
        IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)),
        StaticSlackInstallationDirectory((installation(business_id),)),
        signing_secret=SIGNING_SECRET,
        request_max_age=timedelta(seconds=300),
        clock=lambda: REQUEST_NOW,
    )


async def seed_business(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> None:
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"business-{business_id}",
                name="Business",
                created_at=REQUEST_NOW,
                updated_at=REQUEST_NOW,
            )
        )
        await session.commit()


def normalize(payload: dict[str, Any], business_id: BusinessId) -> InboundOwnerMessage:
    envelope = parse_envelope(payload)
    assert isinstance(envelope, SlackEventCallback)
    return normalize_event(envelope, installation(business_id))

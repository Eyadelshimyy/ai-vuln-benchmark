import base64
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import InboundOwnerMessage
from gvas.infrastructure.models import Business
from gvas.infrastructure.telnyx.events import inbound_message_of, parse_webhook
from gvas.infrastructure.telnyx.ingress import TelnyxMessagingIngress
from gvas.infrastructure.telnyx.installations import (
    StaticTelnyxInstallationDirectory,
    TelnyxInstallation,
)
from gvas.infrastructure.telnyx.normalization import normalize_event
from gvas.infrastructure.telnyx.signature import SIGNATURE_HEADER, TIMESTAMP_HEADER
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory

TELNYX_NUMBER = "+15550100000"
OWNER_NUMBER = "+15550100001"
STRANGER_NUMBER = "+15550100002"
MESSAGE_ID = "40385f64-5717-4562-b3fc-2c963f66afa6"
EVENT_ID = "9a1b2c3d-0000-4000-8000-000000000001"
REQUEST_NOW = datetime(2025, 1, 1, tzinfo=UTC)
REQUEST_TIMESTAMP = str(int(REQUEST_NOW.timestamp()))

_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x07" * 32)
PUBLIC_KEY = base64.b64encode(
    _PRIVATE_KEY.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
).decode()
OTHER_PUBLIC_KEY = base64.b64encode(
    Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)
    .public_key()
    .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
).decode()


def message_payload(**overrides: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "id": MESSAGE_ID,
        "record_type": "message",
        "direction": "inbound",
        "type": "SMS",
        "from": {"phone_number": OWNER_NUMBER, "carrier": "Carrier", "line_type": "Wireless"},
        "to": [{"phone_number": TELNYX_NUMBER, "status": "webhook_delivered"}],
        "text": "replace the water heater",
        "media": [],
        "received_at": "2025-01-01T00:00:00.100000+00:00",
        "messaging_profile_id": "profile-1",
    }
    message.update(overrides)
    return {
        "data": {
            "event_type": "message.received",
            "id": EVENT_ID,
            "occurred_at": "2025-01-01T00:00:00.200000+00:00",
            "record_type": "event",
            "payload": message,
        },
        "meta": {"attempt": 1, "delivered_to": "https://example.test/telnyx/messaging"},
    }


def status_payload(event_type: str = "message.finalized") -> dict[str, Any]:
    payload = message_payload(direction="outbound", **{"from": {"phone_number": TELNYX_NUMBER}})
    payload["data"]["event_type"] = event_type
    payload["data"]["payload"]["to"] = [{"phone_number": OWNER_NUMBER, "status": "delivered"}]
    return payload


def sign(body: bytes, timestamp: str = REQUEST_TIMESTAMP) -> str:
    return base64.b64encode(_PRIVATE_KEY.sign(timestamp.encode() + b"|" + body)).decode()


def signed_request(
    payload: dict[str, Any], timestamp: str = REQUEST_TIMESTAMP
) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(payload).encode()
    return body, {
        SIGNATURE_HEADER: sign(body, timestamp),
        TIMESTAMP_HEADER: timestamp,
        "Content-Type": "application/json",
    }


def installation(business_id: BusinessId) -> TelnyxInstallation:
    return TelnyxInstallation(
        business_id=business_id,
        telnyx_number=TELNYX_NUMBER,
        owner_numbers=frozenset({OWNER_NUMBER}),
    )


def build_ingress(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> TelnyxMessagingIngress:
    return TelnyxMessagingIngress(
        IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)),
        StaticTelnyxInstallationDirectory((installation(business_id),)),
        public_key=PUBLIC_KEY,
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
    event = inbound_message_of(parse_webhook(payload))
    assert event is not None
    return normalize_event(event, installation(business_id))

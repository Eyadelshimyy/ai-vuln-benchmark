import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from gvas.application.ingestion import IngestionStatus, IngestOwnerMessageService
from gvas.domain.identifiers import MessageId
from gvas.infrastructure.telnyx.events import (
    TelnyxInboundMessageEvent,
    TelnyxPayloadError,
    inbound_message_of,
    parse_webhook,
)
from gvas.infrastructure.telnyx.installations import TelnyxInstallationDirectory
from gvas.infrastructure.telnyx.normalization import TelnyxNormalizationError, normalize_event
from gvas.infrastructure.telnyx.signature import verify_signature


class TelnyxIngressResult(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    IGNORED = "ignored"
    UNKNOWN_INSTALLATION = "unknown_installation"
    UNAUTHORIZED_SENDER = "unauthorized_sender"


@dataclass(frozen=True)
class TelnyxIngressOutcome:
    result: TelnyxIngressResult
    message_id: MessageId | None = None
    detail: str | None = None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class TelnyxMessagingIngress:
    """Verifies Telnyx webhooks and performs only the durable ingress handoff.

    Every outcome other than a failed signature or a malformed body is
    acknowledged, so Telnyx stops redelivering: status events for our own
    messages, texts from numbers that are not the configured owner, and
    messages with no text are all dropped silently. Unknown senders are never
    replied to.
    """

    def __init__(
        self,
        ingest_service: IngestOwnerMessageService,
        installations: TelnyxInstallationDirectory,
        *,
        public_key: str,
        request_max_age: timedelta,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._ingest_service = ingest_service
        self._installations = installations
        self._public_key = public_key
        self._request_max_age = request_max_age
        self._clock = clock

    async def handle(
        self, *, body: bytes, signature: str | None, timestamp: str | None
    ) -> TelnyxIngressOutcome:
        verify_signature(
            public_key=self._public_key,
            body=body,
            signature=signature,
            timestamp=timestamp,
            now=self._clock(),
            max_age=self._request_max_age,
        )
        webhook = parse_webhook(_decode(body))
        event = inbound_message_of(webhook)
        if event is None:
            return TelnyxIngressOutcome(TelnyxIngressResult.IGNORED, detail=webhook.data.event_type)
        return await self._ingest(event)

    async def _ingest(self, event: TelnyxInboundMessageEvent) -> TelnyxIngressOutcome:
        message = event.message
        installation = await self._installations.find(message.business_number)
        if installation is None:
            return TelnyxIngressOutcome(
                TelnyxIngressResult.UNKNOWN_INSTALLATION, detail="unknown telnyx number"
            )
        if not installation.is_authorized_owner(message.sender_number):
            return TelnyxIngressOutcome(
                TelnyxIngressResult.UNAUTHORIZED_SENDER, detail="sender is not an authorized owner"
            )
        try:
            inbound = normalize_event(event, installation)
        except TelnyxNormalizationError as error:
            return TelnyxIngressOutcome(TelnyxIngressResult.IGNORED, detail=str(error))
        outcome = await self._ingest_service.ingest(inbound)
        if outcome.status is IngestionStatus.DUPLICATE:
            return TelnyxIngressOutcome(TelnyxIngressResult.DUPLICATE)
        return TelnyxIngressOutcome(TelnyxIngressResult.ACCEPTED, message_id=outcome.message_id)


def _decode(body: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TelnyxPayloadError("request body is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TelnyxPayloadError("request body is not a JSON object")
    return payload

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from gvas.application.ingestion import IngestionStatus, IngestOwnerMessageService
from gvas.domain.identifiers import MessageId
from gvas.infrastructure.slack.events import (
    SlackEventCallback,
    SlackPayloadError,
    SlackUrlVerification,
    is_supported_owner_message,
    parse_envelope,
)
from gvas.infrastructure.slack.installations import SlackInstallationDirectory
from gvas.infrastructure.slack.normalization import SlackNormalizationError, normalize_event
from gvas.infrastructure.slack.signature import verify_signature


class SlackIngressResult(StrEnum):
    CHALLENGE = "challenge"
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    IGNORED = "ignored"
    UNKNOWN_INSTALLATION = "unknown_installation"
    UNAUTHORIZED_SENDER = "unauthorized_sender"


@dataclass(frozen=True)
class SlackIngressOutcome:
    result: SlackIngressResult
    challenge: str | None = None
    message_id: MessageId | None = None
    detail: str | None = None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class SlackEventIngress:
    """Verifies Slack requests and performs only the durable ingress handoff.

    Workflow processing happens asynchronously through the outbox command that
    ingestion enqueues, so the HTTP request can be acknowledged immediately.
    Intent selection is never performed here.
    """

    def __init__(
        self,
        ingest_service: IngestOwnerMessageService,
        installations: SlackInstallationDirectory,
        *,
        signing_secret: str,
        request_max_age: timedelta,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._ingest_service = ingest_service
        self._installations = installations
        self._signing_secret = signing_secret
        self._request_max_age = request_max_age
        self._clock = clock

    async def handle(
        self,
        *,
        body: bytes,
        signature: str | None,
        timestamp: str | None,
        retry_num: str | None = None,
        retry_reason: str | None = None,
    ) -> SlackIngressOutcome:
        verify_signature(
            signing_secret=self._signing_secret,
            body=body,
            signature=signature,
            timestamp=timestamp,
            now=self._clock(),
            max_age=self._request_max_age,
        )
        envelope = parse_envelope(_decode(body))
        if envelope is None:
            return SlackIngressOutcome(SlackIngressResult.IGNORED, detail="unsupported envelope")
        if isinstance(envelope, SlackUrlVerification):
            return SlackIngressOutcome(SlackIngressResult.CHALLENGE, challenge=envelope.challenge)
        return await self._ingest(envelope, retry_num=retry_num, retry_reason=retry_reason)

    async def _ingest(
        self, callback: SlackEventCallback, *, retry_num: str | None, retry_reason: str | None
    ) -> SlackIngressOutcome:
        if not is_supported_owner_message(callback.event):
            return SlackIngressOutcome(SlackIngressResult.IGNORED, detail="unsupported message")
        installation = await self._installations.find(callback.team_id, callback.api_app_id)
        if installation is None:
            return SlackIngressOutcome(
                SlackIngressResult.UNKNOWN_INSTALLATION, detail=callback.team_id
            )
        sender = callback.event.user
        if sender is None or not installation.is_authorized_owner(sender):
            return SlackIngressOutcome(
                SlackIngressResult.UNAUTHORIZED_SENDER, detail="sender is not an authorized owner"
            )
        try:
            inbound = normalize_event(callback, installation)
        except SlackNormalizationError as error:
            return SlackIngressOutcome(SlackIngressResult.IGNORED, detail=str(error))
        outcome = await self._ingest_service.ingest(inbound)
        detail = _retry_detail(retry_num, retry_reason)
        if outcome.status is IngestionStatus.DUPLICATE:
            return SlackIngressOutcome(SlackIngressResult.DUPLICATE, detail=detail)
        return SlackIngressOutcome(
            SlackIngressResult.ACCEPTED, message_id=outcome.message_id, detail=detail
        )


def _retry_detail(retry_num: str | None, retry_reason: str | None) -> str | None:
    if retry_num is None:
        return None
    return f"retry {retry_num}" + (f": {retry_reason}" if retry_reason else "")


def _decode(body: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SlackPayloadError("request body is not valid JSON") from error
    if not isinstance(payload, dict):
        raise SlackPayloadError("request body is not a JSON object")
    return payload

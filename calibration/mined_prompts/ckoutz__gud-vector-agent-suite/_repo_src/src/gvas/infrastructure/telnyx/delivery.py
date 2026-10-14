from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import RoutingData
from gvas.domain.messages import (
    ConversationRef,
    DeliveryReceipt,
    OutboundOwnerMessage,
    TextPart,
)


class TelnyxDeliveryError(RuntimeError):
    """Raised when a Telnyx delivery attempt should be retried by the dispatcher."""


class TelnyxRoutingError(TelnyxDeliveryError):
    pass


class TelnyxModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TelnyxConversationRouting(TelnyxModel):
    """Persisted at ingest: ``from`` is our Telnyx number, ``to`` the owner handset."""

    from_number: str = Field(min_length=1)
    to_number: str = Field(min_length=1)
    messaging_profile_id: str | None = None

    @classmethod
    def from_routing(cls, routing: RoutingData) -> "TelnyxConversationRouting":
        from_number = routing.get("from")
        to_number = routing.get("to")
        profile = routing.get("messaging_profile_id")
        if not isinstance(from_number, str) or not from_number:
            raise TelnyxRoutingError("persisted routing has no telnyx sending number")
        if not isinstance(to_number, str) or not to_number:
            raise TelnyxRoutingError("persisted routing has no owner phone number")
        if profile is not None and not isinstance(profile, str):
            raise TelnyxRoutingError("persisted routing has an invalid messaging profile")
        return cls(from_number=from_number, to_number=to_number, messaging_profile_id=profile)


class TelnyxSendRequest(TelnyxModel):
    from_number: str = Field(min_length=1)
    to_number: str = Field(min_length=1)
    text: str = Field(min_length=1)
    messaging_profile_id: str | None = None
    idempotency_key: str = Field(min_length=1)


class TelnyxSendResult(TelnyxModel):
    message_id: str | None = None
    detail: str | None = None


class TelnyxMessageSender(Protocol):
    async def send_message(self, request: TelnyxSendRequest) -> TelnyxSendResult: ...


class TelnyxRoutingResolver(Protocol):
    async def resolve(
        self, conversation_ref: ConversationRef
    ) -> TelnyxConversationRouting | None: ...


class TelnyxDeliveryLedger(Protocol):
    """At-least-once: a recorded receipt suppresses a resend, a crash before ``record`` does not."""

    async def find(self, key: str) -> DeliveryReceipt | None: ...

    async def record(self, key: str, receipt: DeliveryReceipt) -> None: ...


class InMemoryTelnyxDeliveryLedger:
    def __init__(self) -> None:
        self._receipts: dict[str, DeliveryReceipt] = {}

    async def find(self, key: str) -> DeliveryReceipt | None:
        return self._receipts.get(key)

    async def record(self, key: str, receipt: DeliveryReceipt) -> None:
        self._receipts.setdefault(key, receipt)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def render_text(message: OutboundOwnerMessage) -> str:
    """SMS carries text only; attachment parts are not deliverable over this channel."""

    return "\n".join(part.text for part in message.parts if isinstance(part, TextPart))


def delivery_key(conversation_ref: ConversationRef, message: OutboundOwnerMessage) -> str:
    return (
        f"{conversation_ref.business_id}:"
        f"{conversation_ref.external_conversation_id}:{message.correlation_id}"
    )


class TelnyxOwnerReplyAdapter:
    """Delivers owner replies as SMS to the persisted owner handset."""

    def __init__(
        self,
        sender: TelnyxMessageSender,
        routing_resolver: TelnyxRoutingResolver,
        ledger: TelnyxDeliveryLedger,
        *,
        messaging_profile_id: str | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._sender = sender
        self._routing_resolver = routing_resolver
        self._ledger = ledger
        self._messaging_profile_id = messaging_profile_id or None
        self._clock = clock

    async def send(
        self, conversation_ref: ConversationRef, message: OutboundOwnerMessage
    ) -> DeliveryReceipt:
        key = delivery_key(conversation_ref, message)
        recorded = await self._ledger.find(key)
        if recorded is not None:
            return recorded
        routing = await self._routing_resolver.resolve(conversation_ref)
        if routing is None:
            raise TelnyxRoutingError(
                f"no telnyx routing for conversation {conversation_ref.external_conversation_id}"
            )
        text = render_text(message)
        if not text:
            raise TelnyxDeliveryError("owner reply has no deliverable sms content")
        result = await self._sender.send_message(
            TelnyxSendRequest(
                from_number=routing.from_number,
                to_number=routing.to_number,
                text=text,
                messaging_profile_id=self._messaging_profile_id or routing.messaging_profile_id,
                idempotency_key=key,
            )
        )
        if result.message_id is None:
            raise TelnyxDeliveryError(result.detail or "telnyx rejected the message")
        receipt = DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id=result.message_id,
            occurred_at=self._clock(),
            detail=result.detail,
        )
        await self._ledger.record(key, receipt)
        return receipt

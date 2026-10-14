"""Telnyx messaging webhook envelopes.

Telnyx wraps every event as ``{"data": {"event_type", "id", "occurred_at",
"payload"}, "meta": {...}}``. Only ``message.received`` carries an owner
message; ``message.sent`` and ``message.finalized`` report delivery status of
our own outbound messages and are acknowledged without further work.
"""

from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MESSAGE_RECEIVED = "message.received"
INBOUND_DIRECTION = "inbound"


class TelnyxPayloadError(ValueError):
    pass


class TelnyxEventModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class TelnyxPhoneNumber(TelnyxEventModel):
    phone_number: str = Field(min_length=1)


class TelnyxMedia(TelnyxEventModel):
    url: str | None = None
    content_type: str | None = None
    size: int | None = None


class TelnyxMessage(TelnyxEventModel):
    id: str = Field(min_length=1)
    direction: str = Field(min_length=1)
    from_: TelnyxPhoneNumber = Field(alias="from")
    to: tuple[TelnyxPhoneNumber, ...] = Field(min_length=1)
    text: str | None = None
    media: tuple[TelnyxMedia, ...] = ()
    received_at: datetime | None = None
    type: str | None = None
    messaging_profile_id: str | None = None

    @property
    def sender_number(self) -> str:
        return self.from_.phone_number

    @property
    def business_number(self) -> str:
        return self.to[0].phone_number


class TelnyxEventData(TelnyxEventModel):
    event_type: str = Field(min_length=1)
    id: str = Field(min_length=1)
    occurred_at: datetime
    payload: Mapping[str, object]


class TelnyxWebhook(TelnyxEventModel):
    data: TelnyxEventData


class TelnyxInboundMessageEvent(TelnyxEventModel):
    event_id: str
    occurred_at: datetime
    message: TelnyxMessage


def parse_webhook(payload: Mapping[str, object]) -> TelnyxWebhook:
    try:
        return TelnyxWebhook.model_validate(payload)
    except ValidationError as error:
        raise TelnyxPayloadError("telnyx webhook envelope is malformed") from error


def inbound_message_of(webhook: TelnyxWebhook) -> TelnyxInboundMessageEvent | None:
    """The owner message an event carries, or ``None`` for status-only events."""

    if webhook.data.event_type != MESSAGE_RECEIVED:
        return None
    try:
        message = TelnyxMessage.model_validate(webhook.data.payload)
    except ValidationError as error:
        raise TelnyxPayloadError("telnyx message payload is malformed") from error
    if message.direction != INBOUND_DIRECTION:
        return None
    return TelnyxInboundMessageEvent(
        event_id=webhook.data.id, occurred_at=webhook.data.occurred_at, message=message
    )

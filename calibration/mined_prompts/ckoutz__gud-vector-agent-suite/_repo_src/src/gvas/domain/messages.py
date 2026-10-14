from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gvas.domain.enums import DeliveryStatus, MediaKind, RecipientAddressKind, SenderRole
from gvas.domain.identifiers import (
    BusinessId,
    MessageKey,
    RoutingData,
)


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AttachmentReference(DomainModel):
    attachment_id: UUID
    media_kind: MediaKind
    locator: str
    mime_type: str | None = None
    filename: str | None = None
    byte_size: int | None = Field(default=None, ge=0)

    @field_validator("locator")
    @classmethod
    def locator_is_opaque(cls, value: str) -> str:
        if "://" in value or value.lower().startswith(("http:", "https:", "www.")):
            raise ValueError("attachment locator must be an opaque adapter token")
        if not value.strip():
            raise ValueError("attachment locator must not be empty")
        return value


class TextPart(DomainModel):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)


class AttachmentPart(DomainModel):
    kind: Literal["attachment"] = "attachment"
    attachment: AttachmentReference


ContentPart = Annotated[TextPart | AttachmentPart, Field(discriminator="kind")]


class SenderRef(DomainModel):
    external_id: str = Field(min_length=1)
    role: SenderRole


class ConversationRef(DomainModel):
    business_id: BusinessId
    external_conversation_id: str = Field(min_length=1)


class ReplyRef(DomainModel):
    correlation_id: str = Field(min_length=1)
    external_message_id: str | None = None


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class NormalizedOwnerMessage(DomainModel):
    message_key: MessageKey
    business_id: BusinessId
    conversation_ref: ConversationRef
    sender: SenderRef
    received_at: datetime
    parts: tuple[ContentPart, ...] = Field(min_length=1)
    reply_to: ReplyRef | None = None

    _received_at_aware = field_validator("received_at")(_aware)

    @model_validator(mode="after")
    def business_matches_conversation(self) -> "NormalizedOwnerMessage":
        if self.conversation_ref.business_id != self.business_id:
            raise ValueError("conversation business must match message business")
        return self


class ChannelEndpointRef(DomainModel):
    """Opaque adapter-owned endpoint identity; core never branches on its namespace."""

    business_id: BusinessId
    source_namespace: str = Field(min_length=1)
    external_endpoint_id: str = Field(min_length=1)


class InboundOwnerMessage(DomainModel):
    message: NormalizedOwnerMessage
    endpoint: ChannelEndpointRef
    routing: RoutingData

    @model_validator(mode="after")
    def business_matches_endpoint(self) -> "InboundOwnerMessage":
        if self.endpoint.business_id != self.message.business_id:
            raise ValueError("endpoint business must match message business")
        return self


class OutboundOwnerMessage(DomainModel):
    business_id: BusinessId
    conversation_ref: ConversationRef
    parts: tuple[ContentPart, ...] = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    reply_to: ReplyRef | None = None

    @model_validator(mode="after")
    def business_matches_conversation(self) -> "OutboundOwnerMessage":
        if self.conversation_ref.business_id != self.business_id:
            raise ValueError("conversation business must match message business")
        return self


class DeliveryReceipt(DomainModel):
    status: DeliveryStatus
    provider_message_id: str | None = None
    occurred_at: datetime
    detail: str | None = None
    # Where the customer can view what was delivered, when the delivery created
    # a hosted page rather than sending the content itself. Customer-facing, not
    # secret, so the owner may be shown it.
    customer_link: str | None = None
    # Whether the delivery emailed the customer; ``None`` when the channel does
    # not report it.
    emailed: bool | None = None

    _occurred_at_aware = field_validator("occurred_at")(_aware)


class AudioReference(DomainModel):
    attachment: AttachmentReference
    # Present when the audio belongs to a tenant, so a metered adapter can
    # attribute what the call consumed to that business.
    business_id: BusinessId | None = None

    @field_validator("attachment")
    @classmethod
    def must_be_audio(cls, value: AttachmentReference) -> AttachmentReference:
        if value.media_kind is not MediaKind.AUDIO:
            raise ValueError("audio reference requires an audio attachment")
        return value


class TranscriptResult(DomainModel):
    text: str
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    duration_seconds: float | None = Field(default=None, ge=0)
    provider_ref: str | None = None


class AttachmentPayload(DomainModel):
    content: bytes
    mime_type: str | None = None
    filename: str | None = None


class CustomerRecipient(DomainModel):
    address: str = Field(min_length=1)
    address_kind: RecipientAddressKind
    display_name: str | None = None
    # A second way to reach the customer, E.164, when ``address`` is not a phone.
    phone: str | None = None
    # Where the work happens, when known (e.g. from the booking).
    service_address: str | None = None

    @property
    def email_address(self) -> str | None:
        return self.address if self.address_kind is RecipientAddressKind.EMAIL else None

    @property
    def phone_number(self) -> str | None:
        if self.address_kind is RecipientAddressKind.PHONE:
            return self.address
        return self.phone


class CustomerDeliveryLineItem(DomainModel):
    description: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    # Signed so a discount can travel as its own line.
    unit_price_minor: int


class CustomerDeliveryRequest(DomainModel):
    business_id: BusinessId
    recipient: CustomerRecipient
    idempotency_key: str = Field(min_length=1)
    subject: str | None = None
    body_text: str = Field(min_length=1)
    links: tuple[str, ...] = Field(default_factory=tuple)
    attachments: tuple[AttachmentReference, ...] = Field(default_factory=tuple)
    # The same content as ``body_text``, structured, for channels that render
    # the quote themselves rather than forwarding the text.
    line_items: tuple[CustomerDeliveryLineItem, ...] = Field(default_factory=tuple)
    currency: str | None = None
    # Set when the quote is hosted on the business's own site: the adapter is
    # asked to render the link into the message and report it on the receipt.
    quote_url: str | None = None
    # The name the customer-facing message should come from, when configured.
    business_name: str | None = None
    note: str | None = None


class CustomerTextRequest(DomainModel):
    """One short text message to a customer's phone."""

    business_id: BusinessId
    phone_number: str = Field(min_length=1)
    text: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)

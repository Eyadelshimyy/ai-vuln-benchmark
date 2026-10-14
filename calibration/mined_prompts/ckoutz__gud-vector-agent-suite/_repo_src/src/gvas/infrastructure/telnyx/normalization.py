"""Telnyx ``message.received`` events as channel-neutral inbound owner messages.

A conversation is the ``(telnyx number, owner number)`` pair, so every text
from the owner to the business number continues the same conversation; SMS has
no threads to key on. MMS media is not surfaced: the quote workflow this
channel is scoped to is text-only, and media URLs may never enter the domain as
attachment locators. A media-only message therefore has no content and is
ignored by ingress.
"""

from gvas.domain.enums import SenderRole
from gvas.domain.identifiers import JsonValue, MessageKey
from gvas.domain.messages import (
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    SenderRef,
    TextPart,
)
from gvas.infrastructure.telnyx.events import TelnyxInboundMessageEvent
from gvas.infrastructure.telnyx.installations import (
    TELNYX_SOURCE_NAMESPACE,
    TelnyxInstallation,
)


class TelnyxNormalizationError(ValueError):
    pass


def conversation_id_for(telnyx_number: str, owner_number: str) -> str:
    return f"{telnyx_number}:{owner_number}"


def normalize_event(
    event: TelnyxInboundMessageEvent, installation: TelnyxInstallation
) -> InboundOwnerMessage:
    message = event.message
    sender = message.sender_number
    if not installation.is_authorized_owner(sender):
        raise TelnyxNormalizationError("telnyx sender is not an authorized owner")
    if message.business_number != installation.telnyx_number:
        raise TelnyxNormalizationError("telnyx message was not sent to the installed number")
    text = (message.text or "").strip()
    if not text:
        raise TelnyxNormalizationError("telnyx message has no text content")
    endpoint = ChannelEndpointRef(
        business_id=installation.business_id,
        source_namespace=TELNYX_SOURCE_NAMESPACE,
        external_endpoint_id=installation.external_endpoint_id,
    )
    conversation_ref = ConversationRef(
        business_id=installation.business_id,
        external_conversation_id=conversation_id_for(installation.telnyx_number, sender),
    )
    normalized = NormalizedOwnerMessage(
        message_key=MessageKey(f"{installation.telnyx_number}:{message.id}"),
        business_id=installation.business_id,
        conversation_ref=conversation_ref,
        sender=SenderRef(external_id=sender, role=SenderRole.OWNER),
        received_at=message.received_at or event.occurred_at,
        parts=(TextPart(text=text),),
    )
    routing: dict[str, JsonValue] = {
        "from": installation.telnyx_number,
        "to": sender,
        "event_id": event.event_id,
        "message_id": message.id,
        "media_count": len(message.media),
    }
    if message.messaging_profile_id is not None:
        routing["messaging_profile_id"] = message.messaging_profile_id
    return InboundOwnerMessage(message=normalized, endpoint=endpoint, routing=routing)

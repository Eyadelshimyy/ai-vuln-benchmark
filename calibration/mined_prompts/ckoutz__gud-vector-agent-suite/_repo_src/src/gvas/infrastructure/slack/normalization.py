import re
from datetime import UTC, datetime
from uuid import UUID, uuid5

from gvas.domain.enums import MediaKind, SenderRole
from gvas.domain.identifiers import JsonValue, MessageKey
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentReference,
    ChannelEndpointRef,
    ContentPart,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    ReplyRef,
    SenderRef,
    TextPart,
)
from gvas.infrastructure.slack.events import SlackEventCallback, SlackFile, SlackMessageEvent
from gvas.infrastructure.slack.installations import SLACK_SOURCE_NAMESPACE, SlackInstallation

ATTACHMENT_NAMESPACE = UUID("2b4c7f1e-0f45-4f4e-9c0a-6f5b1d3a8e21")
ATTACHMENT_LOCATOR_PREFIX = "slack-file"
VOICE_MEMO_SUBTYPE = "slack_audio"
MAILTO_LINK_PATTERN = re.compile(r"<mailto:([^>|]+)(?:\|[^>]*)?>")


class SlackNormalizationError(ValueError):
    pass


def normalize_event(
    callback: SlackEventCallback, installation: SlackInstallation
) -> InboundOwnerMessage:
    event = callback.event
    if event.user is None:
        raise SlackNormalizationError("slack message event has no sender")
    if not installation.is_authorized_owner(event.user):
        raise SlackNormalizationError("slack sender is not an authorized owner")
    endpoint = ChannelEndpointRef(
        business_id=installation.business_id,
        source_namespace=SLACK_SOURCE_NAMESPACE,
        external_endpoint_id=installation.external_endpoint_id,
    )
    conversation_ref = ConversationRef(
        business_id=installation.business_id,
        external_conversation_id=f"{event.channel}:{event.thread_root_ts}",
    )
    parts = _content_parts(callback, event)
    if not parts:
        raise SlackNormalizationError("slack message event has no supported content")
    message = NormalizedOwnerMessage(
        message_key=MessageKey(f"{event.channel}:{event.ts}"),
        business_id=installation.business_id,
        conversation_ref=conversation_ref,
        sender=SenderRef(external_id=event.user, role=SenderRole.OWNER),
        received_at=_timestamp(event.event_ts or event.ts),
        parts=parts,
        reply_to=_reply_ref(event),
    )
    return InboundOwnerMessage(
        message=message, endpoint=endpoint, routing=_routing(callback, event)
    )


def _content_parts(
    callback: SlackEventCallback, event: SlackMessageEvent
) -> tuple[ContentPart, ...]:
    parts: list[ContentPart] = []
    text = (event.text or "").strip()
    if text:
        parts.append(TextPart(text=MAILTO_LINK_PATTERN.sub(r"\1", text)))
    for file in event.files:
        parts.append(AttachmentPart(attachment=_attachment(callback, event, file)))
    return tuple(parts)


def _attachment(
    callback: SlackEventCallback, event: SlackMessageEvent, file: SlackFile
) -> AttachmentReference:
    name = f"{callback.team_id}:{event.channel}:{event.ts}:{file.id}"
    return AttachmentReference(
        attachment_id=uuid5(ATTACHMENT_NAMESPACE, name),
        media_kind=_media_kind(file),
        locator=f"{ATTACHMENT_LOCATOR_PREFIX}:{file.id}",
        mime_type=file.mimetype,
        filename=file.name or file.title,
        byte_size=file.size,
    )


def _media_kind(file: SlackFile) -> MediaKind:
    if file.subtype == VOICE_MEMO_SUBTYPE:
        return MediaKind.AUDIO
    mime_type = (file.mimetype or "").lower()
    if mime_type.startswith("audio/"):
        return MediaKind.AUDIO
    if mime_type.startswith("image/"):
        return MediaKind.IMAGE
    if mime_type.startswith("video/"):
        return MediaKind.VIDEO
    if mime_type.startswith("text/") or mime_type.startswith("application/"):
        return MediaKind.DOCUMENT
    return MediaKind.OTHER


def _reply_ref(event: SlackMessageEvent) -> ReplyRef | None:
    if not event.is_thread_reply:
        return None
    return ReplyRef(
        correlation_id=f"{event.channel}:{event.thread_root_ts}",
        external_message_id=event.thread_root_ts,
    )


def _routing(callback: SlackEventCallback, event: SlackMessageEvent) -> dict[str, JsonValue]:
    """Adapter-owned opaque routing; core code stores it without interpretation."""

    return {
        "team_id": callback.team_id,
        "api_app_id": callback.api_app_id,
        "enterprise_id": callback.enterprise_id,
        "event_id": callback.event_id,
        "channel": event.channel,
        "channel_type": event.channel_type,
        "message_ts": event.ts,
        "thread_ts": event.thread_root_ts,
    }


def _timestamp(value: str) -> datetime:
    try:
        seconds = float(value)
    except ValueError as error:
        raise SlackNormalizationError(f"invalid slack timestamp {value!r}") from error
    return datetime.fromtimestamp(seconds, tz=UTC)

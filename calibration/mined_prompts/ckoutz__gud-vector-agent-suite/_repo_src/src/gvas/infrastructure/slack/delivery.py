from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import RoutingData
from gvas.domain.messages import (
    AttachmentPart,
    ConversationRef,
    DeliveryReceipt,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.ports import AttachmentAccessPort


class SlackDeliveryError(RuntimeError):
    """Raised when a Slack delivery attempt should be retried by the dispatcher."""


class SlackRoutingError(SlackDeliveryError):
    pass


class SlackModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SlackConversationRouting(SlackModel):
    channel: str = Field(min_length=1)
    thread_ts: str | None = None

    @classmethod
    def from_routing(cls, routing: RoutingData) -> "SlackConversationRouting":
        channel = routing.get("channel")
        thread_ts = routing.get("thread_ts")
        if not isinstance(channel, str) or not channel:
            raise SlackRoutingError("persisted routing has no slack channel")
        if thread_ts is not None and not isinstance(thread_ts, str):
            raise SlackRoutingError("persisted routing has an invalid slack thread timestamp")
        return cls(channel=channel, thread_ts=thread_ts)


class SlackChatPostRequest(SlackModel):
    channel: str = Field(min_length=1)
    text: str = Field(min_length=1)
    thread_ts: str | None = None
    idempotency_key: str = Field(min_length=1)


class SlackChatPostResult(SlackModel):
    message_ts: str | None = None
    detail: str | None = None


class SlackChatPoster(Protocol):
    """Outbound Slack transport; no provider client is selected in this round."""

    async def post_message(self, request: SlackChatPostRequest) -> SlackChatPostResult: ...


class SlackUploadFile(SlackModel):
    filename: str = Field(min_length=1)
    content: bytes = Field(min_length=1)
    title: str | None = None


class SlackFileUploadRequest(SlackModel):
    channel: str = Field(min_length=1)
    files: tuple[SlackUploadFile, ...] = Field(min_length=1)
    thread_ts: str | None = None
    initial_comment: str | None = None
    idempotency_key: str = Field(min_length=1)


class SlackFileUploadResult(SlackModel):
    file_ids: tuple[str, ...] = Field(default_factory=tuple)
    detail: str | None = None


class SlackFileUploader(Protocol):
    """Shares real files into a channel or thread; text-only posters need not implement it."""

    async def upload_files(self, request: SlackFileUploadRequest) -> SlackFileUploadResult: ...


class SlackRoutingResolver(Protocol):
    async def resolve(
        self, conversation_ref: ConversationRef
    ) -> SlackConversationRouting | None: ...


class SlackDeliveryLedger(Protocol):
    """Records completed deliveries so a retried command usually skips reposting.

    Delivery is at-least-once. A recorded receipt suppresses the repost, but a
    process that crashes after Slack accepts the post and before ``record``
    commits leaves no receipt, and the retry posts again. Implementations must
    not claim to close that window; the post carries a delivery key in its
    message metadata so duplicates can be reconciled after the fact.
    """

    async def find(self, key: str) -> DeliveryReceipt | None: ...

    async def record(self, key: str, receipt: DeliveryReceipt) -> None: ...


class InMemorySlackDeliveryLedger:
    def __init__(self) -> None:
        self._receipts: dict[str, DeliveryReceipt] = {}

    async def find(self, key: str) -> DeliveryReceipt | None:
        return self._receipts.get(key)

    async def record(self, key: str, receipt: DeliveryReceipt) -> None:
        self._receipts.setdefault(key, receipt)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def render_text(message: OutboundOwnerMessage) -> str:
    lines: list[str] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            lines.append(part.text)
        elif isinstance(part, AttachmentPart):
            attachment = part.attachment
            label = attachment.filename or attachment.media_kind.value
            lines.append(f"[{attachment.media_kind.value}: {label}]")
    return "\n".join(lines)


def delivery_key(conversation_ref: ConversationRef, message: OutboundOwnerMessage) -> str:
    return (
        f"{conversation_ref.business_id}:"
        f"{conversation_ref.external_conversation_id}:{message.correlation_id}"
    )


def render_comment(message: OutboundOwnerMessage) -> str | None:
    text = "\n".join(part.text for part in message.parts if isinstance(part, TextPart))
    return text or None


class SlackOwnerReplyAdapter:
    """Delivers owner replies to Slack using persisted routing only.

    Replies that carry attachments are shared as real files when an uploader and
    an attachment source are wired; the text parts become the file's comment.
    Without them, attachments degrade to a text label so delivery never depends
    on an optional capability.
    """

    def __init__(
        self,
        poster: SlackChatPoster,
        routing_resolver: SlackRoutingResolver,
        ledger: SlackDeliveryLedger,
        *,
        uploader: SlackFileUploader | None = None,
        attachments: AttachmentAccessPort | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._poster = poster
        self._routing_resolver = routing_resolver
        self._ledger = ledger
        self._uploader = uploader
        self._attachments = attachments
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
            raise SlackRoutingError(
                f"no slack routing for conversation {conversation_ref.external_conversation_id}"
            )
        attachments = [part for part in message.parts if isinstance(part, AttachmentPart)]
        if attachments and self._uploader is not None and self._attachments is not None:
            provider_id, detail = await self._upload(
                key, routing, message, attachments, self._uploader, self._attachments
            )
        else:
            provider_id, detail = await self._post(key, routing, message)
        receipt = DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id=provider_id,
            occurred_at=self._clock(),
            detail=detail,
        )
        await self._ledger.record(key, receipt)
        return receipt

    async def _post(
        self, key: str, routing: SlackConversationRouting, message: OutboundOwnerMessage
    ) -> tuple[str, str | None]:
        text = render_text(message)
        if not text:
            raise SlackDeliveryError("owner reply has no deliverable slack content")
        result = await self._poster.post_message(
            SlackChatPostRequest(
                channel=routing.channel,
                text=text,
                thread_ts=_thread_ts(routing, message),
                idempotency_key=key,
            )
        )
        if result.message_ts is None:
            raise SlackDeliveryError(result.detail or "slack rejected the message")
        return result.message_ts, result.detail

    async def _upload(
        self,
        key: str,
        routing: SlackConversationRouting,
        message: OutboundOwnerMessage,
        attachments: list[AttachmentPart],
        uploader: SlackFileUploader,
        source: AttachmentAccessPort,
    ) -> tuple[str, str | None]:
        files: list[SlackUploadFile] = []
        for part in attachments:
            payload = await source.fetch(part.attachment)
            filename = payload.filename or part.attachment.filename
            if not filename:
                raise SlackDeliveryError("owner reply attachment has no filename")
            files.append(SlackUploadFile(filename=filename, content=payload.content))
        result = await uploader.upload_files(
            SlackFileUploadRequest(
                channel=routing.channel,
                files=tuple(files),
                thread_ts=_thread_ts(routing, message),
                initial_comment=render_comment(message),
                idempotency_key=key,
            )
        )
        if not result.file_ids:
            raise SlackDeliveryError(result.detail or "slack rejected the file upload")
        return ",".join(result.file_ids), result.detail


def _thread_ts(routing: SlackConversationRouting, message: OutboundOwnerMessage) -> str | None:
    if message.reply_to is not None and message.reply_to.external_message_id:
        return message.reply_to.external_message_id
    return routing.thread_ts

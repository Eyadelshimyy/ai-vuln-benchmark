from datetime import UTC, datetime
from inspect import Parameter, signature
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.enums import DeliveryStatus, MediaKind
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentPayload,
    AttachmentReference,
    ConversationRef,
    OutboundOwnerMessage,
    ReplyRef,
    TextPart,
)
from gvas.domain.ports import OwnerReplyPort
from gvas.infrastructure.slack.composition import build_slack_owner_reply_adapter
from gvas.infrastructure.slack.delivery import (
    InMemorySlackDeliveryLedger,
    SlackChatPostRequest,
    SlackChatPostResult,
    SlackConversationRouting,
    SlackDeliveryError,
    SlackFileUploadRequest,
    SlackFileUploadResult,
    SlackOwnerReplyAdapter,
    SlackRoutingError,
    delivery_key,
)
from gvas.infrastructure.slack.routing import SqlSlackRoutingResolver
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from slack_fixtures import (
    CHANNEL,
    ROOT_TS,
    message_payload,
    normalize,
    seed_business,
)

DELIVERED_AT = datetime(2025, 2, 1, tzinfo=UTC)


class RecordingPoster:
    def __init__(self, results: list[SlackChatPostResult] | None = None) -> None:
        self.requests: list[SlackChatPostRequest] = []
        self._results = results

    async def post_message(self, request: SlackChatPostRequest) -> SlackChatPostResult:
        self.requests.append(request)
        if self._results:
            return self._results.pop(0)
        return SlackChatPostResult(message_ts="1735689800.000300")


class StaticRoutingResolver:
    def __init__(self, routing: SlackConversationRouting | None) -> None:
        self._routing = routing
        self.calls = 0

    async def resolve(self, conversation_ref: ConversationRef) -> SlackConversationRouting | None:
        self.calls += 1
        return self._routing


def reply(business_id: BusinessId, *, correlation_id: str = "reply-1") -> OutboundOwnerMessage:
    conversation_ref = ConversationRef(
        business_id=business_id, external_conversation_id=f"{CHANNEL}:{ROOT_TS}"
    )
    return OutboundOwnerMessage(
        business_id=business_id,
        conversation_ref=conversation_ref,
        parts=(
            TextPart(text="On it."),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(),
                    media_kind=MediaKind.DOCUMENT,
                    locator="slack-file:F00000FAKE",
                    filename="quote.pdf",
                )
            ),
        ),
        correlation_id=correlation_id,
        reply_to=ReplyRef(
            correlation_id=f"{CHANNEL}:{ROOT_TS}",
            external_message_id=ROOT_TS,
        ),
    )


def build_adapter(
    resolver: StaticRoutingResolver, poster: RecordingPoster
) -> SlackOwnerReplyAdapter:
    return SlackOwnerReplyAdapter(
        poster, resolver, InMemorySlackDeliveryLedger(), clock=lambda: DELIVERED_AT
    )


@pytest.mark.asyncio
async def test_reply_is_posted_to_the_persisted_thread() -> None:
    business_id = BusinessId(uuid4())
    poster = RecordingPoster()
    resolver = StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL, thread_ts=None))
    adapter: OwnerReplyPort = build_adapter(resolver, poster)
    message = reply(business_id)

    receipt = await adapter.send(message.conversation_ref, message)

    assert receipt.status is DeliveryStatus.ACCEPTED
    assert receipt.provider_message_id == "1735689800.000300"
    assert receipt.occurred_at == DELIVERED_AT
    assert len(poster.requests) == 1
    assert poster.requests[0].channel == CHANNEL
    assert poster.requests[0].thread_ts == ROOT_TS
    assert poster.requests[0].text == "On it.\n[document: quote.pdf]"


@pytest.mark.asyncio
async def test_retried_delivery_posts_once_and_returns_the_recorded_receipt() -> None:
    business_id = BusinessId(uuid4())
    poster = RecordingPoster()
    resolver = StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL))
    adapter = build_adapter(resolver, poster)
    message = reply(business_id)

    first = await adapter.send(message.conversation_ref, message)
    retried = await adapter.send(message.conversation_ref, message)

    assert first == retried
    assert len(poster.requests) == 1
    assert poster.requests[0].idempotency_key.endswith("reply-1")


@pytest.mark.asyncio
async def test_failed_post_is_retryable_and_not_recorded() -> None:
    business_id = BusinessId(uuid4())
    poster = RecordingPoster(
        [
            SlackChatPostResult(detail="ratelimited"),
            SlackChatPostResult(message_ts="1735689900.000400"),
        ]
    )
    resolver = StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL))
    adapter = build_adapter(resolver, poster)
    message = reply(business_id)

    with pytest.raises(SlackDeliveryError):
        await adapter.send(message.conversation_ref, message)
    receipt = await adapter.send(message.conversation_ref, message)

    assert receipt.provider_message_id == "1735689900.000400"
    assert len(poster.requests) == 2


@pytest.mark.asyncio
async def test_missing_routing_is_a_delivery_error() -> None:
    business_id = BusinessId(uuid4())
    poster = RecordingPoster()
    adapter = build_adapter(StaticRoutingResolver(None), poster)
    message = reply(business_id)

    with pytest.raises(SlackRoutingError):
        await adapter.send(message.conversation_ref, message)
    assert poster.requests == []


def test_routing_requires_a_channel() -> None:
    with pytest.raises(SlackRoutingError):
        SlackConversationRouting.from_routing({"thread_ts": ROOT_TS})
    with pytest.raises(SlackRoutingError):
        SlackConversationRouting.from_routing({"channel": CHANNEL, "thread_ts": 17})


@pytest.mark.asyncio
async def test_owner_reply_adapter_composition_requires_an_explicit_ledger(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    parameter = signature(build_slack_owner_reply_adapter).parameters["ledger"]
    assert parameter.default is Parameter.empty

    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    inbound = normalize(message_payload(), business_id)
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(inbound)
    poster = RecordingPoster()
    ledger = InMemorySlackDeliveryLedger()
    adapter = build_slack_owner_reply_adapter(poster, session_factory, ledger)
    message = reply(business_id)

    first = await adapter.send(inbound.message.conversation_ref, message)
    second = await adapter.send(inbound.message.conversation_ref, message)

    assert first == second
    assert len(poster.requests) == 1
    assert await ledger.find(poster.requests[0].idempotency_key) == first


@pytest.mark.asyncio
async def test_persisted_slack_routing_is_resolved_for_ingested_conversations(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    inbound = normalize(message_payload(), business_id)
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(inbound)

    resolver = SqlSlackRoutingResolver(session_factory)
    routing = await resolver.resolve(inbound.message.conversation_ref)

    assert routing == SlackConversationRouting(channel=CHANNEL, thread_ts=ROOT_TS)
    assert (
        await resolver.resolve(
            ConversationRef(business_id=business_id, external_conversation_id="C0:1")
        )
        is None
    )


class RecordingUploader:
    def __init__(self, result: SlackFileUploadResult | None = None) -> None:
        self.requests: list[SlackFileUploadRequest] = []
        self._result = result or SlackFileUploadResult(file_ids=("F0DOC",))

    async def upload_files(self, request: SlackFileUploadRequest) -> SlackFileUploadResult:
        self.requests.append(request)
        return self._result


class StaticAttachmentSource:
    def __init__(self, content: bytes = b"PK\x03\x04docx") -> None:
        self.fetched: list[AttachmentReference] = []
        self._content = content

    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        self.fetched.append(attachment)
        return AttachmentPayload(
            content=self._content, mime_type=attachment.mime_type, filename=attachment.filename
        )


@pytest.mark.asyncio
async def test_attachment_reply_is_uploaded_as_a_file_into_the_thread() -> None:
    business_id = BusinessId(uuid4())
    poster = RecordingPoster()
    uploader = RecordingUploader()
    source = StaticAttachmentSource()
    adapter = SlackOwnerReplyAdapter(
        poster,
        StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL, thread_ts=ROOT_TS)),
        InMemorySlackDeliveryLedger(),
        uploader=uploader,
        attachments=source,
        clock=lambda: DELIVERED_AT,
    )
    message = reply(business_id)

    receipt = await adapter.send(message.conversation_ref, message)

    assert poster.requests == []
    assert len(uploader.requests) == 1
    request = uploader.requests[0]
    assert request.channel == CHANNEL
    assert request.thread_ts == ROOT_TS
    assert request.initial_comment == "On it."
    assert [file.filename for file in request.files] == ["quote.pdf"]
    assert request.files[0].content == b"PK\x03\x04docx"
    assert source.fetched == [message.parts[1].attachment]  # type: ignore[union-attr]
    assert receipt.provider_message_id == "F0DOC"
    assert receipt.status is DeliveryStatus.ACCEPTED


@pytest.mark.asyncio
async def test_retried_upload_is_suppressed_by_the_ledger() -> None:
    business_id = BusinessId(uuid4())
    uploader = RecordingUploader()
    adapter = SlackOwnerReplyAdapter(
        RecordingPoster(),
        StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL, thread_ts=ROOT_TS)),
        InMemorySlackDeliveryLedger(),
        uploader=uploader,
        attachments=StaticAttachmentSource(),
        clock=lambda: DELIVERED_AT,
    )
    message = reply(business_id)

    first = await adapter.send(message.conversation_ref, message)
    second = await adapter.send(message.conversation_ref, message)

    assert first == second
    assert len(uploader.requests) == 1


@pytest.mark.asyncio
async def test_rejected_upload_is_retryable_and_not_recorded() -> None:
    business_id = BusinessId(uuid4())
    uploader = RecordingUploader(SlackFileUploadResult(detail="slack rejected the file upload"))
    ledger = InMemorySlackDeliveryLedger()
    adapter = SlackOwnerReplyAdapter(
        RecordingPoster(),
        StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL, thread_ts=ROOT_TS)),
        ledger,
        uploader=uploader,
        attachments=StaticAttachmentSource(),
    )
    message = reply(business_id)

    with pytest.raises(SlackDeliveryError):
        await adapter.send(message.conversation_ref, message)

    assert await ledger.find(delivery_key(message.conversation_ref, message)) is None


@pytest.mark.asyncio
async def test_text_only_reply_does_not_use_the_uploader() -> None:
    business_id = BusinessId(uuid4())
    poster = RecordingPoster()
    uploader = RecordingUploader()
    adapter = SlackOwnerReplyAdapter(
        poster,
        StaticRoutingResolver(SlackConversationRouting(channel=CHANNEL, thread_ts=ROOT_TS)),
        InMemorySlackDeliveryLedger(),
        uploader=uploader,
        attachments=StaticAttachmentSource(),
    )
    conversation_ref = ConversationRef(
        business_id=business_id, external_conversation_id=f"{CHANNEL}:{ROOT_TS}"
    )
    message = OutboundOwnerMessage(
        business_id=business_id,
        conversation_ref=conversation_ref,
        parts=(TextPart(text="Just text."),),
        correlation_id="text-1",
    )

    await adapter.send(conversation_ref, message)

    assert uploader.requests == []
    assert [request.text for request in poster.requests] == ["Just text."]

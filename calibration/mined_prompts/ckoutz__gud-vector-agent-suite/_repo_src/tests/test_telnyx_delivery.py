import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.enums import DeliveryStatus, MediaKind
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentReference,
    ConversationRef,
    DeliveryReceipt,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.ports import OwnerReplyPort
from gvas.infrastructure.delivery_ledger import SqlChannelDeliveryLedger
from gvas.infrastructure.owner_reply_routing import ChannelOwnerReplyRouter, OwnerReplyRoutingError
from gvas.infrastructure.telnyx.api import TelnyxMessagingApiSender
from gvas.infrastructure.telnyx.composition import build_telnyx_owner_reply_adapter
from gvas.infrastructure.telnyx.config import TelnyxSettings
from gvas.infrastructure.telnyx.delivery import (
    InMemoryTelnyxDeliveryLedger,
    TelnyxConversationRouting,
    TelnyxDeliveryError,
    TelnyxOwnerReplyAdapter,
    TelnyxRoutingError,
    TelnyxSendRequest,
    TelnyxSendResult,
    delivery_key,
)
from gvas.infrastructure.telnyx.routing import SqlTelnyxRoutingResolver
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from telnyx_fixtures import (
    OWNER_NUMBER,
    TELNYX_NUMBER,
    message_payload,
    normalize,
    seed_business,
)

DELIVERED_AT = datetime(2025, 2, 1, tzinfo=UTC)
API_KEY = "KEY0123456789"  # noqa: S105 - fake value for tests


class RecordingSender:
    def __init__(self, results: list[TelnyxSendResult] | None = None) -> None:
        self.requests: list[TelnyxSendRequest] = []
        self._results = results

    async def send_message(self, request: TelnyxSendRequest) -> TelnyxSendResult:
        self.requests.append(request)
        if self._results:
            return self._results.pop(0)
        return TelnyxSendResult(message_id="msg-1")


class StaticRoutingResolver:
    def __init__(self, routing: TelnyxConversationRouting | None) -> None:
        self._routing = routing

    async def resolve(self, conversation_ref: ConversationRef) -> TelnyxConversationRouting | None:
        return self._routing


def reply(business_id: BusinessId, *, correlation_id: str = "reply-1") -> OutboundOwnerMessage:
    conversation_ref = ConversationRef(
        business_id=business_id, external_conversation_id=f"{TELNYX_NUMBER}:{OWNER_NUMBER}"
    )
    return OutboundOwnerMessage(
        business_id=business_id,
        conversation_ref=conversation_ref,
        parts=(
            TextPart(text="Quote sent."),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(),
                    media_kind=MediaKind.DOCUMENT,
                    locator="report:1",
                    filename="report.docx",
                )
            ),
        ),
        correlation_id=correlation_id,
    )


def build_adapter(resolver: StaticRoutingResolver, sender: RecordingSender) -> OwnerReplyPort:
    return TelnyxOwnerReplyAdapter(
        sender, resolver, InMemoryTelnyxDeliveryLedger(), clock=lambda: DELIVERED_AT
    )


@pytest.mark.asyncio
async def test_reply_is_sent_as_text_only_to_the_persisted_owner_number() -> None:
    business_id = BusinessId(uuid4())
    sender = RecordingSender()
    resolver = StaticRoutingResolver(
        TelnyxConversationRouting(from_number=TELNYX_NUMBER, to_number=OWNER_NUMBER)
    )
    adapter = build_adapter(resolver, sender)
    message = reply(business_id)

    receipt = await adapter.send(message.conversation_ref, message)

    assert receipt.status is DeliveryStatus.ACCEPTED
    assert receipt.provider_message_id == "msg-1"
    assert receipt.occurred_at == DELIVERED_AT
    assert len(sender.requests) == 1
    assert sender.requests[0].from_number == TELNYX_NUMBER
    assert sender.requests[0].to_number == OWNER_NUMBER
    assert sender.requests[0].text == "Quote sent."
    assert sender.requests[0].idempotency_key == delivery_key(message.conversation_ref, message)


@pytest.mark.asyncio
async def test_retried_delivery_sends_once_and_returns_the_recorded_receipt() -> None:
    business_id = BusinessId(uuid4())
    sender = RecordingSender()
    resolver = StaticRoutingResolver(
        TelnyxConversationRouting(from_number=TELNYX_NUMBER, to_number=OWNER_NUMBER)
    )
    adapter = build_adapter(resolver, sender)
    message = reply(business_id)

    first = await adapter.send(message.conversation_ref, message)
    retried = await adapter.send(message.conversation_ref, message)
    await adapter.send(message.conversation_ref, reply(business_id, correlation_id="r2"))

    assert first == retried
    assert len(sender.requests) == 2
    assert sender.requests[0].idempotency_key.endswith("reply-1")
    assert sender.requests[1].idempotency_key.endswith("r2")


@pytest.mark.asyncio
async def test_failed_or_unrouted_send_is_retryable_and_not_recorded() -> None:
    business_id = BusinessId(uuid4())
    sender = RecordingSender([TelnyxSendResult(detail="telnyx returned http 500")])
    resolver = StaticRoutingResolver(
        TelnyxConversationRouting(from_number=TELNYX_NUMBER, to_number=OWNER_NUMBER)
    )
    adapter = build_adapter(resolver, sender)
    message = reply(business_id)

    with pytest.raises(TelnyxDeliveryError):
        await adapter.send(message.conversation_ref, message)
    await adapter.send(message.conversation_ref, message)
    assert len(sender.requests) == 2

    with pytest.raises(TelnyxRoutingError):
        await build_adapter(StaticRoutingResolver(None), RecordingSender()).send(
            message.conversation_ref, message
        )


@pytest.mark.asyncio
async def test_sql_routing_and_ledger_deliver_once_from_ingested_routing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    inbound = normalize(message_payload(), business_id)
    await ingest.ingest(inbound)
    sender = RecordingSender()
    adapter = build_telnyx_owner_reply_adapter(
        sender,
        session_factory,
        SqlChannelDeliveryLedger(session_factory),
        messaging_profile_id="configured-profile",
    )
    message = reply(business_id)

    resolved = await SqlTelnyxRoutingResolver(session_factory).resolve(message.conversation_ref)
    assert resolved == TelnyxConversationRouting(
        from_number=TELNYX_NUMBER, to_number=OWNER_NUMBER, messaging_profile_id="profile-1"
    )
    first = await adapter.send(message.conversation_ref, message)
    second = await adapter.send(message.conversation_ref, message)

    assert first == second
    assert len(sender.requests) == 1
    assert sender.requests[0].messaging_profile_id == "configured-profile"


class NamespaceReplyFake:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[OutboundOwnerMessage] = []

    async def send(
        self, conversation_ref: ConversationRef, message: OutboundOwnerMessage
    ) -> DeliveryReceipt:
        self.sent.append(message)
        return DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id=self.name,
            occurred_at=DELIVERED_AT,
        )


@pytest.mark.asyncio
async def test_owner_reply_router_picks_the_adapter_of_the_conversation_channel(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        normalize(message_payload(), business_id)
    )
    telnyx = NamespaceReplyFake("telnyx")
    other = NamespaceReplyFake("other")
    router = ChannelOwnerReplyRouter(session_factory, {"telnyx": telnyx, "other": other})
    message = reply(business_id)

    receipt = await router.send(message.conversation_ref, message)

    assert receipt.provider_message_id == "telnyx"
    assert telnyx.sent == [message]
    assert other.sent == []
    unknown = ConversationRef(business_id=business_id, external_conversation_id="nowhere")
    with pytest.raises(OwnerReplyRoutingError):
        await router.send(unknown, message)
    with pytest.raises(OwnerReplyRoutingError):
        await ChannelOwnerReplyRouter(session_factory, {"other": other}).send(
            message.conversation_ref, message
        )


def api_settings() -> TelnyxSettings:
    return TelnyxSettings(
        public_key="cHVibGlj",
        api_key=API_KEY,
        installations=f"{OWNER_NUMBER}={uuid4()}:{TELNYX_NUMBER}",
        api_base_url="https://telnyx.test/v2",
    )


def request_for(profile: str | None = None) -> TelnyxSendRequest:
    return TelnyxSendRequest(
        from_number=TELNYX_NUMBER,
        to_number=OWNER_NUMBER,
        text="Quote sent.",
        messaging_profile_id=profile,
        idempotency_key="key",
    )


@pytest.mark.asyncio
async def test_api_sender_posts_a_bearer_authorized_message_and_returns_its_id() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": {"id": "tx-1", "record_type": "message"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await TelnyxMessagingApiSender(api_settings(), client).send_message(
            request_for("profile-1")
        )

    assert result.message_id == "tx-1"
    assert seen[0].url == httpx.URL("https://telnyx.test/v2/messages")
    assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(seen[0].content) == {
        "from": TELNYX_NUMBER,
        "to": OWNER_NUMBER,
        "text": "Quote sent.",
        "messaging_profile_id": "profile-1",
    }


@pytest.mark.asyncio
async def test_api_sender_hides_provider_errors_and_the_api_key() -> None:
    def rejected(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"errors": [{"detail": f"secret leak {API_KEY} invalid number"}]}
        )

    def unreadable(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    def unreachable(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(rejected)) as client:
        result = await TelnyxMessagingApiSender(api_settings(), client).send_message(request_for())
    assert result.message_id is None
    assert result.detail == "telnyx returned http 422"

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreadable)) as client:
        with pytest.raises(TelnyxDeliveryError) as error:
            await TelnyxMessagingApiSender(api_settings(), client).send_message(request_for())
    assert API_KEY not in str(error.value)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
        with pytest.raises(TelnyxDeliveryError):
            await TelnyxMessagingApiSender(api_settings(), client).send_message(request_for())

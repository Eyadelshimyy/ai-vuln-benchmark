from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.quotes import (
    DeliverApprovedQuoteService,
    QuoteDeliveryStatus,
    QuoteIntentSelector,
    QuoteWorkflowHandler,
)
from gvas.domain.enums import (
    DeliveryStatus,
    HostedLinkKind,
    QuoteSendAction,
    QuoteStatus,
    RecipientAddressKind,
    SenderRole,
)
from gvas.domain.identifiers import (
    BusinessId,
    ConversationId,
    EndpointId,
    MessageKey,
    WorkflowRunId,
)
from gvas.domain.messages import (
    ConversationRef,
    CustomerDeliveryRequest,
    CustomerRecipient,
    DeliveryReceipt,
    NormalizedOwnerMessage,
    ReplyRef,
    SenderRef,
    TextPart,
)
from gvas.domain.quotes import (
    QUOTE_INTENT,
    HostedLinkReference,
    OwnerApprovalRequiredPolicy,
    Quote,
    QuoteDraftProposal,
    QuoteDraftRequest,
    QuoteLineItem,
    QuoteSendAssessment,
    QuoteSendDecision,
    new_quote,
    quote_delivery_command,
)
from gvas.domain.repositories import CrossBusinessReferenceError, UnitOfWork
from gvas.domain.workflows import WorkflowContext
from gvas.infrastructure.models import (
    Business,
    Conversation,
    OutboxMessage,
    OwnerChannelEndpoint,
)
from gvas.infrastructure.unit_of_work import SqlUnitOfWork, SqlUnitOfWorkFactory

NOW = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)


@dataclass
class UnitOfWorkTracker:
    active: int = 0


class TrackedSqlUnitOfWork(SqlUnitOfWork):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        tracker: UnitOfWorkTracker,
    ) -> None:
        super().__init__(session_factory)
        self._tracker = tracker

    async def __aenter__(self) -> "TrackedSqlUnitOfWork":
        self._tracker.active += 1
        try:
            await super().__aenter__()
        except BaseException:
            self._tracker.active -= 1
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        try:
            await super().__aexit__(exc_type, exc, traceback)
        finally:
            self._tracker.active -= 1


class TrackedUnitOfWorkFactory:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        tracker: UnitOfWorkTracker,
    ) -> None:
        self._session_factory = session_factory
        self._tracker = tracker

    def __call__(self) -> UnitOfWork:
        return TrackedSqlUnitOfWork(self._session_factory, self._tracker)


class DraftingFake:
    def __init__(self, tracker: UnitOfWorkTracker, *, failures: int = 0) -> None:
        self.tracker = tracker
        self.failures = failures
        self.requests: list[QuoteDraftRequest] = []

    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal:
        assert self.tracker.active == 0
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise RuntimeError("drafting unavailable")
        return QuoteDraftProposal(
            quote_id=request.quote_id,
            business_id=request.business_id,
            recipient=CustomerRecipient(
                address="customer@example.test",
                address_kind=RecipientAddressKind.EMAIL,
                display_name="Customer",
            ),
            currency="usd",
            line_items=(
                QuoteLineItem(
                    description=f"Revision {request.revision}",
                    quantity=2,
                    unit_price_minor=1250,
                ),
            ),
            tax_minor=200,
            discount_minor=100,
            owner_note="We'll be there Tuesday",
            hosted_links=(
                HostedLinkReference(
                    kind=HostedLinkKind.PAYMENT,
                    reference="payment-link-token",
                ),
            ),
            confidence=0.75,
        )


class DeliveryFake:
    def __init__(self, tracker: UnitOfWorkTracker) -> None:
        self.tracker = tracker
        self.requests: list[CustomerDeliveryRequest] = []

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        assert self.tracker.active == 0
        self.requests.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.DELIVERED,
            provider_message_id="delivery-reference",
            occurred_at=NOW,
        )


class AutoSendPolicy:
    def decide(self, assessment: QuoteSendAssessment) -> QuoteSendDecision:
        return QuoteSendDecision(action=QuoteSendAction.AUTO_SEND)


async def seed_conversation(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[BusinessId, ConversationId]:
    business_id = BusinessId(uuid4())
    endpoint_id = EndpointId(uuid4())
    conversation_id = ConversationId(uuid4())
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"business-{business_id}",
                name="Business",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            OwnerChannelEndpoint(
                id=endpoint_id,
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id=f"endpoint-{endpoint_id}",
                owner_external_id="owner",
                routing={},
            )
        )
        session.add(
            Conversation(
                id=conversation_id,
                business_id=business_id,
                endpoint_id=endpoint_id,
                external_conversation_id=f"conversation-{conversation_id}",
                routing={},
            )
        )
        await session.commit()
    return business_id, conversation_id


def owner_message(
    business_id: BusinessId,
    conversation_id: ConversationId,
    message_key: str,
    text: str,
    *,
    reply_to: str | None = None,
) -> NormalizedOwnerMessage:
    return NormalizedOwnerMessage(
        message_key=MessageKey(message_key),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id,
            external_conversation_id=f"conversation-{conversation_id}",
        ),
        sender=SenderRef(external_id="owner", role=SenderRole.OWNER),
        received_at=NOW,
        parts=(TextPart(text=text),),
        reply_to=ReplyRef(correlation_id=reply_to) if reply_to is not None else None,
    )


def workflow_context(
    message: NormalizedOwnerMessage, conversation_id: ConversationId
) -> WorkflowContext:
    return WorkflowContext(
        run_id=WorkflowRunId(uuid4()),
        intent=QUOTE_INTENT,
        message=message,
        conversation_id=conversation_id,
    )


async def active_quote(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    conversation_id: ConversationId,
) -> Quote | None:
    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        quote = await unit_of_work.quotes.get_active(business_id, conversation_id)
        await unit_of_work.commit()
    return quote


def test_quote_totals_validation_and_initial_policy() -> None:
    item = QuoteLineItem(description="Service", quantity=3, unit_price_minor=400)
    assert item.total_minor == 1200
    proposal = QuoteDraftProposal(
        quote_id=uuid4(),
        business_id=uuid4(),
        recipient=CustomerRecipient(
            address="customer@example.test",
            address_kind=RecipientAddressKind.EMAIL,
        ),
        currency="usd",
        line_items=(item,),
        tax_minor=100,
        discount_minor=50,
    )
    assert proposal.currency == "USD"
    assert proposal.subtotal_minor == 1200
    assert proposal.total_minor == 1250
    decision = OwnerApprovalRequiredPolicy().decide(
        QuoteSendAssessment(confidence=1, risk_flags=(), suspected_mistake=False)
    )
    assert decision.action is QuoteSendAction.REQUIRE_APPROVAL

    with pytest.raises(ValidationError, match="quote total must not be negative"):
        QuoteDraftProposal(
            quote_id=uuid4(),
            business_id=uuid4(),
            recipient=proposal.recipient,
            currency="USD",
            line_items=(QuoteLineItem(description="Free", quantity=1, unit_price_minor=0),),
            discount_minor=1,
        )
    with pytest.raises(ValidationError, match="opaque token"):
        HostedLinkReference(
            kind=HostedLinkKind.SIGNUP,
            reference="https://provider.example/signup",
        )


def test_intent_selection_is_explicit_and_conversation_scoped() -> None:
    business_id = BusinessId(uuid4())
    conversation_id = ConversationId(uuid4())
    selector = QuoteIntentSelector()
    explicit = owner_message(business_id, conversation_id, "message-1", " quote: service")
    resolution = selector.select(explicit, conversation_id, None)
    assert resolution is not None
    assert resolution.intent == QUOTE_INTENT
    assert (
        selector.select(
            owner_message(business_id, conversation_id, "message-2", "service"),
            conversation_id,
            None,
        )
        is None
    )
    quote = new_quote(
        business_id=business_id,
        conversation_id=conversation_id,
        conversation_ref=explicit.conversation_ref,
        message_key=explicit.message_key,
        request_text="service",
        now=NOW,
    )
    reply = owner_message(business_id, conversation_id, "message-3", "approve")
    resolution = selector.select(reply, conversation_id, quote)
    assert resolution is not None
    assert resolution.intent == QUOTE_INTENT
    with pytest.raises(ValueError, match="normalized conversation"):
        selector.select(reply, ConversationId(uuid4()), quote)


@pytest.mark.asyncio
async def test_workflow_retries_thread_scoped_approval_and_delivery_are_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id, conversation_id = await seed_conversation(session_factory)
    tracker = UnitOfWorkTracker()
    unit_of_work_factory = TrackedUnitOfWorkFactory(session_factory, tracker)
    drafting = DraftingFake(tracker)
    handler = QuoteWorkflowHandler(unit_of_work_factory, drafting)
    initial = owner_message(
        business_id,
        conversation_id,
        "quote-message",
        "quote: two service visits",
    )

    first = await handler.handle(workflow_context(initial, conversation_id))
    retry = await handler.handle(workflow_context(initial, conversation_id))
    assert first == retry
    assert len(drafting.requests) == 1
    assert drafting.requests[0].idempotency_key.endswith(":1")

    slack_thread_correlation = "slack-thread:1712345678.000100"
    assert slack_thread_correlation != first.replies[0].correlation_id
    approval = owner_message(
        business_id,
        conversation_id,
        "approval-message",
        "approve",
        reply_to=slack_thread_correlation,
    )
    approved = await handler.handle(workflow_context(approval, conversation_id))
    approval_retry = await handler.handle(workflow_context(approval, conversation_id))
    assert approved == approval_retry
    async with session_factory() as session:
        command_count = await session.scalar(select(func.count()).select_from(OutboxMessage))
    assert command_count == 1

    late_rejection = owner_message(
        business_id,
        conversation_id,
        "late-rejection",
        "reject",
        reply_to=slack_thread_correlation,
    )
    late_result = await handler.handle(workflow_context(late_rejection, conversation_id))
    late_part = late_result.replies[0].parts[0]
    assert isinstance(late_part, TextPart)
    assert late_part.text == "No quote is awaiting approval in this conversation."
    async with session_factory() as session:
        command_count = await session.scalar(select(func.count()).select_from(OutboxMessage))
    assert command_count == 1

    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is not None
    assert quote.status is QuoteStatus.APPROVED
    command = quote_delivery_command(quote)
    assert command.payload == {"quote_id": str(quote.quote_id)}
    assert command.dedup_key == f"quote_delivery:{quote.quote_id}"

    delivery = DeliveryFake(tracker)
    service = DeliverApprovedQuoteService(unit_of_work_factory, delivery)
    wrong_business = await service.deliver(BusinessId(uuid4()), quote.quote_id)
    assert wrong_business.status is QuoteDeliveryStatus.MISSING
    completed = await service.deliver(business_id, quote.quote_id)
    duplicate = await service.deliver(business_id, quote.quote_id)
    assert completed.status is QuoteDeliveryStatus.COMPLETED
    assert duplicate.status is QuoteDeliveryStatus.ALREADY_COMPLETED
    assert len(delivery.requests) == 1
    assert delivery.requests[0].idempotency_key == f"quote-delivery:{quote.quote_id}"
    assert delivery.requests[0].links == ("payment-link-token",)
    assert delivery.requests[0].note == "We'll be there Tuesday"


@pytest.mark.asyncio
async def test_thread_scoped_correction_and_rejection_use_current_active_revision(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id, conversation_id = await seed_conversation(session_factory)
    tracker = UnitOfWorkTracker()
    unit_of_work_factory = TrackedUnitOfWorkFactory(session_factory, tracker)
    drafting = DraftingFake(tracker)
    handler = QuoteWorkflowHandler(unit_of_work_factory, drafting)
    initial = owner_message(business_id, conversation_id, "initial", "quote: service")
    first = await handler.handle(workflow_context(initial, conversation_id))

    slack_thread_correlation = "slack-thread:1712345678.000100"
    assert slack_thread_correlation != first.replies[0].correlation_id
    correction = owner_message(
        business_id,
        conversation_id,
        "correction",
        "correct: use three visits",
        reply_to=slack_thread_correlation,
    )
    corrected = await handler.handle(workflow_context(correction, conversation_id))
    correction_retry = await handler.handle(workflow_context(correction, conversation_id))
    assert corrected == correction_retry
    assert [request.revision for request in drafting.requests] == [1, 2]
    assert corrected.replies[0].correlation_id != first.replies[0].correlation_id

    rejection = owner_message(
        business_id,
        conversation_id,
        "rejection",
        "reject",
        reply_to=slack_thread_correlation,
    )
    rejected = await handler.handle(workflow_context(rejection, conversation_id))
    rejection_retry = await handler.handle(workflow_context(rejection, conversation_id))
    assert rejected == rejection_retry
    assert await active_quote(session_factory, business_id, conversation_id) is None

    no_pending_quote = owner_message(
        business_id,
        conversation_id,
        "no-pending-quote",
        "approve",
        reply_to=slack_thread_correlation,
    )
    result = await handler.handle(workflow_context(no_pending_quote, conversation_id))
    part = result.replies[0].parts[0]
    assert isinstance(part, TextPart)
    assert part.text == "No quote is awaiting approval in this conversation."
    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is None


@pytest.mark.asyncio
async def test_failed_draft_retries_with_same_request_identity(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id, conversation_id = await seed_conversation(session_factory)
    tracker = UnitOfWorkTracker()
    drafting = DraftingFake(tracker, failures=1)
    handler = QuoteWorkflowHandler(
        TrackedUnitOfWorkFactory(session_factory, tracker),
        drafting,
    )
    message = owner_message(business_id, conversation_id, "initial", "quote: service")

    with pytest.raises(RuntimeError, match="drafting unavailable"):
        await handler.handle(workflow_context(message, conversation_id))
    await handler.handle(workflow_context(message, conversation_id))

    assert len(drafting.requests) == 2
    assert drafting.requests[0] == drafting.requests[1]


@pytest.mark.asyncio
async def test_auto_send_policy_cannot_bypass_initial_approval(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id, conversation_id = await seed_conversation(session_factory)
    tracker = UnitOfWorkTracker()
    handler = QuoteWorkflowHandler(
        TrackedUnitOfWorkFactory(session_factory, tracker),
        DraftingFake(tracker),
        AutoSendPolicy(),
    )
    result = await handler.handle(
        workflow_context(
            owner_message(business_id, conversation_id, "initial", "quote: service"),
            conversation_id,
        )
    )

    quote = await active_quote(session_factory, business_id, conversation_id)
    assert quote is not None
    assert quote.status is QuoteStatus.AWAITING_APPROVAL
    part = result.replies[0].parts[0]
    assert isinstance(part, TextPart)
    assert "disabled until a later phase" in part.text
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 0


@pytest.mark.asyncio
async def test_quote_repository_rejects_cross_business_conversation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first_business, conversation_id = await seed_conversation(session_factory)
    second_business, _ = await seed_conversation(session_factory)
    quote = new_quote(
        business_id=second_business,
        conversation_id=conversation_id,
        conversation_ref=ConversationRef(
            business_id=second_business,
            external_conversation_id=f"conversation-{conversation_id}",
        ),
        message_key=MessageKey("cross-business"),
        request_text="service",
        now=NOW,
    )

    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        with pytest.raises(CrossBusinessReferenceError):
            await unit_of_work.quotes.add(quote)
    assert first_business != second_business

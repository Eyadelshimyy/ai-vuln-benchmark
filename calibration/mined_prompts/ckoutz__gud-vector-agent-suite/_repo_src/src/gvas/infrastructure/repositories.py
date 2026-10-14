from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from gvas.domain.enums import DeliveryStatus, OutboxStatus, QuoteStatus, WorkflowRunStatus
from gvas.domain.identifiers import (
    BusinessId,
    ConversationId,
    CustomerId,
    EndpointId,
    MessageId,
    MessageKey,
    QuoteId,
    RoutingData,
    WorkflowIntent,
    WorkflowRunId,
)
from gvas.domain.intake import IntakeProfile
from gvas.domain.messages import (
    ChannelEndpointRef,
    ConversationRef,
    DeliveryReceipt,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
)
from gvas.domain.outbox import (
    DEFAULT_MAX_ATTEMPTS,
    LostOutboxLeaseError,
    OutboxCommand,
    OutboxRecord,
)
from gvas.domain.owner_email import OWNER_EMAIL_SOURCE_NAMESPACE
from gvas.domain.quotes import (
    QUOTE_CLAIMABLE_STATUSES,
    Quote,
    QuoteConcurrencyError,
    normalize_customer_email,
)
from gvas.domain.repositories import (
    BusinessRecord,
    CrossBusinessReferenceError,
    EndpointBusinessMismatchError,
    InboundProcessingRecord,
    LostWorkflowLeaseError,
    OutboundDeliveryRecord,
    OwnerChannelEndpointRecord,
    WorkflowClaimResult,
    WorkflowRunClaim,
)
from gvas.infrastructure.models import (
    Business,
    Conversation,
    InboundMessage,
    OutboundMessage,
    OutboxMessage,
    OwnerChannelEndpoint,
    QuoteRecord,
    WorkflowRun,
)


def _validate_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _rowcount(result: Result[Any]) -> int:
    return cast(CursorResult[Any], result).rowcount


def _aware_or_none(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class SqlBusinessRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: Business) -> BusinessRecord:
        return BusinessRecord(
            business_id=BusinessId(row.id),
            slug=row.slug,
            name=row.name,
            site_url=row.site_url,
            display_name=row.display_name,
            calendly_url=row.calendly_url,
            stripe_account_id=row.stripe_account_id,
            public_key=row.public_key,
            intake_profile=IntakeProfile(
                brief=row.intake_brief,
                questions=row.intake_questions,
                opening=row.intake_opening,
                offer_line=row.intake_offer_line,
                visit_question=row.intake_visit_question,
            ),
            notification_email=row.notification_email,
            owner_email=row.owner_email,
            calendar_feed_url=row.calendar_feed_url,
            timezone=row.timezone,
            invoice_prefix=row.invoice_prefix,
        )

    async def get(self, business_id: BusinessId) -> BusinessRecord | None:
        row = await self.session.scalar(select(Business).where(Business.id == business_id))
        return None if row is None else self._record(row)

    async def lock(self, business_id: BusinessId) -> BusinessRecord | None:
        row = await self.session.scalar(
            select(Business).where(Business.id == business_id).with_for_update()
        )
        return None if row is None else self._record(row)

    async def ensure(
        self, business_id: BusinessId, slug: str, name: str, *, now: datetime
    ) -> BusinessRecord:
        row = await self.session.scalar(select(Business).where(Business.id == business_id))
        if row is None:
            row = Business(id=business_id, slug=slug, name=name, created_at=now, updated_at=now)
            self.session.add(row)
        elif row.slug != slug or row.name != name:
            row.slug = slug
            row.name = name
            row.updated_at = now
        await self.session.flush()
        return self._record(row)

    async def get_by_public_key(self, public_key: str) -> BusinessRecord | None:
        row = await self.session.scalar(select(Business).where(Business.public_key == public_key))
        return None if row is None else self._record(row)

    async def list_site_urls(self) -> tuple[str, ...]:
        rows = await self.session.scalars(
            select(Business.site_url).where(Business.site_url.is_not(None))
        )
        return tuple(url for url in rows.all() if url)

    async def configure_site(
        self,
        business_id: BusinessId,
        *,
        site_url: str | None = None,
        display_name: str | None = None,
        calendly_url: str | None = None,
        stripe_account_id: str | None = None,
        public_key: str | None = None,
        intake_brief: str | None = None,
        intake_questions: str | None = None,
        intake_opening: str | None = None,
        intake_offer_line: str | None = None,
        intake_visit_question: str | None = None,
        invoice_prefix: str | None = None,
        notification_email: str | None = None,
        owner_email: str | None = None,
        calendar_feed_url: str | None = None,
        timezone: str | None = None,
        now: datetime,
    ) -> BusinessRecord:
        row = await self.session.scalar(select(Business).where(Business.id == business_id))
        if row is None:
            raise BusinessNotFoundError(f"business {business_id} does not exist")
        if site_url is not None:
            row.site_url = site_url
        if display_name is not None:
            row.display_name = display_name
        if calendly_url is not None:
            row.calendly_url = calendly_url
        if stripe_account_id is not None:
            row.stripe_account_id = stripe_account_id
        if public_key is not None:
            row.public_key = public_key
        if intake_brief is not None:
            row.intake_brief = intake_brief or None
        if intake_questions is not None:
            row.intake_questions = intake_questions or None
        if intake_opening is not None:
            row.intake_opening = intake_opening or None
        if intake_offer_line is not None:
            row.intake_offer_line = intake_offer_line or None
        if intake_visit_question is not None:
            row.intake_visit_question = intake_visit_question or None
        if invoice_prefix is not None:
            row.invoice_prefix = invoice_prefix or None
        if notification_email is not None:
            row.notification_email = notification_email or None
        if owner_email is not None:
            row.owner_email = owner_email
        if calendar_feed_url is not None:
            row.calendar_feed_url = calendar_feed_url or None
        if timezone is not None:
            row.timezone = timezone or None
        row.updated_at = now
        try:
            async with self.session.begin_nested():
                await self.session.flush()
        except IntegrityError as error:
            raise BusinessPublicKeyConflictError(
                "another business already holds that public key"
            ) from error
        return self._record(row)

    async def adopt_timezone(self, business_id: BusinessId, timezone: str, now: datetime) -> bool:
        result = cast(
            CursorResult[Any],
            await self.session.execute(
                update(Business)
                .where(Business.id == business_id, Business.timezone.is_(None))
                .values(timezone=timezone, updated_at=now)
                .execution_options(synchronize_session="fetch")
            ),
        )
        return result.rowcount == 1


class BusinessNotFoundError(ValueError):
    pass


class BusinessPublicKeyConflictError(ValueError):
    pass


class SqlOwnerChannelEndpointRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _record(row: OwnerChannelEndpoint) -> OwnerChannelEndpointRecord:
        return OwnerChannelEndpointRecord(
            endpoint_id=EndpointId(row.id),
            business_id=BusinessId(row.business_id),
            source_namespace=row.source_namespace,
            external_endpoint_id=row.external_endpoint_id,
            owner_external_id=row.owner_external_id,
            routing=row.routing,
        )

    async def get(self, endpoint_id: EndpointId) -> OwnerChannelEndpointRecord | None:
        row = await self.session.scalar(
            select(OwnerChannelEndpoint).where(OwnerChannelEndpoint.id == endpoint_id)
        )
        return self._record(row) if row is not None else None

    async def get_or_create(
        self, reference: ChannelEndpointRef, routing: RoutingData
    ) -> EndpointId:
        row = await self.session.scalar(
            select(OwnerChannelEndpoint).where(
                OwnerChannelEndpoint.business_id == reference.business_id,
                OwnerChannelEndpoint.source_namespace == reference.source_namespace,
                OwnerChannelEndpoint.external_endpoint_id == reference.external_endpoint_id,
            )
        )
        if row is not None:
            return EndpointId(row.id)
        row = OwnerChannelEndpoint(
            business_id=reference.business_id,
            source_namespace=reference.source_namespace,
            external_endpoint_id=reference.external_endpoint_id,
            routing=routing,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            existing = await self.session.scalar(
                select(OwnerChannelEndpoint).where(
                    OwnerChannelEndpoint.business_id == reference.business_id,
                    OwnerChannelEndpoint.source_namespace == reference.source_namespace,
                    OwnerChannelEndpoint.external_endpoint_id == reference.external_endpoint_id,
                )
            )
            if existing is None:
                raise
            return EndpointId(existing.id)
        return EndpointId(row.id)


class SqlConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create(
        self, reference: ConversationRef, endpoint_id: EndpointId, routing: RoutingData
    ) -> ConversationId:
        endpoint = await self.session.scalar(
            select(OwnerChannelEndpoint).where(OwnerChannelEndpoint.id == endpoint_id)
        )
        if endpoint is None or endpoint.business_id != reference.business_id:
            raise EndpointBusinessMismatchError(
                "conversation endpoint must belong to the conversation business"
            )
        result = await self.session.execute(
            select(Conversation).where(
                Conversation.endpoint_id == endpoint_id,
                Conversation.external_conversation_id == reference.external_conversation_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            return ConversationId(existing.id)
        conversation = Conversation(
            business_id=reference.business_id,
            endpoint_id=endpoint_id,
            external_conversation_id=reference.external_conversation_id,
            routing=routing,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(conversation)
                await self.session.flush()
        except IntegrityError:
            existing = await self.session.scalar(
                select(Conversation).where(
                    Conversation.endpoint_id == endpoint_id,
                    Conversation.external_conversation_id == reference.external_conversation_id,
                )
            )
            if existing is None:
                raise
            return ConversationId(existing.id)
        return ConversationId(conversation.id)

    async def find_endpoint(self, reference: ConversationRef) -> ChannelEndpointRef | None:
        endpoint = await self.session.scalar(
            select(OwnerChannelEndpoint)
            .join(Conversation, Conversation.endpoint_id == OwnerChannelEndpoint.id)
            .where(
                Conversation.business_id == reference.business_id,
                Conversation.external_conversation_id == reference.external_conversation_id,
            )
            .order_by(Conversation.id)
            .limit(1)
        )
        if endpoint is None:
            return None
        return ChannelEndpointRef(
            business_id=BusinessId(endpoint.business_id),
            source_namespace=endpoint.source_namespace,
            external_endpoint_id=endpoint.external_endpoint_id,
        )


class SqlInboundMessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        message: InboundOwnerMessage,
        conversation_id: ConversationId,
        endpoint_id: EndpointId,
    ) -> MessageId | None:
        conversation = await self.session.scalar(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        endpoint = await self.session.scalar(
            select(OwnerChannelEndpoint).where(OwnerChannelEndpoint.id == endpoint_id)
        )
        if (
            conversation is None
            or endpoint is None
            or conversation.business_id != message.message.business_id
            or endpoint.business_id != message.message.business_id
            or conversation.endpoint_id != endpoint_id
        ):
            raise CrossBusinessReferenceError(
                "inbound message references a conversation or endpoint from another business"
            )
        existing = await self.session.scalar(
            select(InboundMessage).where(
                InboundMessage.endpoint_id == endpoint_id,
                InboundMessage.message_key == message.message.message_key,
            )
        )
        if existing is not None:
            return None
        normalized = message.message
        row = InboundMessage(
            business_id=normalized.business_id,
            endpoint_id=endpoint_id,
            conversation_id=conversation_id,
            message_key=normalized.message_key,
            sender_external_id=normalized.sender.external_id,
            sender_role=normalized.sender.role.value,
            received_at=normalized.received_at,
            parts=[part.model_dump(mode="json") for part in normalized.parts],
            reply_to=normalized.reply_to.model_dump(mode="json") if normalized.reply_to else None,
            routing=message.routing,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            return None
        return MessageId(row.id)

    async def get_for_processing(
        self, inbound_message_id: MessageId
    ) -> InboundProcessingRecord | None:
        result = await self.session.execute(
            select(InboundMessage, Conversation)
            .join(Conversation, Conversation.id == InboundMessage.conversation_id)
            .where(InboundMessage.id == inbound_message_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        inbound, conversation = row
        return _processing_record(inbound, conversation)

    async def find_by_key(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        message_key: MessageKey,
    ) -> InboundProcessingRecord | None:
        result = await self.session.execute(
            select(InboundMessage, Conversation)
            .join(Conversation, Conversation.id == InboundMessage.conversation_id)
            .where(
                InboundMessage.business_id == business_id,
                InboundMessage.conversation_id == conversation_id,
                InboundMessage.message_key == message_key,
            )
        )
        row = result.one_or_none()
        if row is None:
            return None
        inbound, conversation = row
        return _processing_record(inbound, conversation)

    async def find_latest_for_business(
        self, business_id: BusinessId
    ) -> InboundProcessingRecord | None:
        result = await self.session.execute(
            select(InboundMessage, Conversation)
            .join(Conversation, Conversation.id == InboundMessage.conversation_id)
            .join(OwnerChannelEndpoint, OwnerChannelEndpoint.id == Conversation.endpoint_id)
            .where(
                InboundMessage.business_id == business_id,
                # Leftover endpoints of the retired e-mail reply channel are
                # never the owner thread new notices anchor to.
                OwnerChannelEndpoint.source_namespace != OWNER_EMAIL_SOURCE_NAMESPACE,
            )
            .order_by(InboundMessage.received_at.desc())
            .limit(1)
        )
        row = result.one_or_none()
        if row is None:
            return None
        inbound, conversation = row
        return _processing_record(inbound, conversation)


def _processing_record(
    inbound: InboundMessage, conversation: Conversation
) -> InboundProcessingRecord:
    message = NormalizedOwnerMessage.model_validate(
        {
            "message_key": inbound.message_key,
            "business_id": inbound.business_id,
            "conversation_ref": {
                "business_id": conversation.business_id,
                "external_conversation_id": conversation.external_conversation_id,
            },
            "sender": {
                "external_id": inbound.sender_external_id,
                "role": inbound.sender_role,
            },
            "received_at": (
                inbound.received_at
                if inbound.received_at.tzinfo is not None
                else inbound.received_at.replace(tzinfo=UTC)
            ),
            "parts": inbound.parts,
            "reply_to": inbound.reply_to,
        }
    )
    return InboundProcessingRecord(
        inbound_message_id=MessageId(inbound.id),
        business_id=BusinessId(inbound.business_id),
        conversation_id=ConversationId(inbound.conversation_id),
        endpoint_id=EndpointId(inbound.endpoint_id),
        message=message,
    )


class SqlOutboundMessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        message: OutboundOwnerMessage,
        conversation_id: ConversationId,
        inbound_message_id: MessageId,
    ) -> MessageId:
        conversation = await self.session.scalar(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        inbound = await self.session.scalar(
            select(InboundMessage).where(InboundMessage.id == inbound_message_id)
        )
        if (
            conversation is None
            or inbound is None
            or conversation.business_id != message.business_id
            or inbound.business_id != message.business_id
            or inbound.conversation_id != conversation_id
        ):
            raise CrossBusinessReferenceError(
                "outbound message references a conversation or inbound message "
                "from another business"
            )
        existing = await self.session.scalar(
            select(OutboundMessage).where(
                OutboundMessage.inbound_message_id == inbound_message_id,
                OutboundMessage.correlation_id == message.correlation_id,
            )
        )
        if existing is not None:
            return MessageId(existing.id)
        row = OutboundMessage(
            business_id=message.business_id,
            conversation_id=conversation_id,
            inbound_message_id=inbound_message_id,
            parts=[part.model_dump(mode="json") for part in message.parts],
            reply_to=message.reply_to.model_dump(mode="json") if message.reply_to else None,
            status=DeliveryStatus.ACCEPTED.value,
            correlation_id=message.correlation_id,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            existing = await self.session.scalar(
                select(OutboundMessage).where(
                    OutboundMessage.inbound_message_id == inbound_message_id,
                    OutboundMessage.correlation_id == message.correlation_id,
                )
            )
            if existing is None:
                raise
            return MessageId(existing.id)
        return MessageId(row.id)

    async def get_for_delivery(
        self, outbound_message_id: MessageId
    ) -> OutboundDeliveryRecord | None:
        result = await self.session.execute(
            select(OutboundMessage, Conversation, OwnerChannelEndpoint)
            .join(Conversation, Conversation.id == OutboundMessage.conversation_id)
            .join(OwnerChannelEndpoint, OwnerChannelEndpoint.id == Conversation.endpoint_id)
            .where(OutboundMessage.id == outbound_message_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        outbound, conversation, endpoint = row
        return OutboundDeliveryRecord(
            outbound_message_id=MessageId(outbound.id),
            message=OutboundOwnerMessage(
                business_id=BusinessId(outbound.business_id),
                conversation_ref=ConversationRef(
                    business_id=BusinessId(conversation.business_id),
                    external_conversation_id=conversation.external_conversation_id,
                ),
                parts=tuple(outbound.parts),
                correlation_id=outbound.correlation_id,
                reply_to=outbound.reply_to,
            ),
            endpoint_id=EndpointId(endpoint.id),
            conversation_routing=conversation.routing,
            endpoint_routing=endpoint.routing,
            status=DeliveryStatus(outbound.status),
        )

    async def find_by_correlation(
        self, business_id: BusinessId, conversation_id: ConversationId, correlation_id: str
    ) -> MessageId | None:
        row_id = await self.session.scalar(
            select(OutboundMessage.id).where(
                OutboundMessage.business_id == business_id,
                OutboundMessage.conversation_id == conversation_id,
                OutboundMessage.correlation_id == correlation_id,
            )
        )
        return None if row_id is None else MessageId(row_id)

    async def record_delivery(
        self, outbound_message_id: MessageId, receipt: DeliveryReceipt
    ) -> None:
        await self.session.execute(
            update(OutboundMessage)
            .where(OutboundMessage.id == outbound_message_id)
            .values(
                status=receipt.status.value,
                provider_message_id=receipt.provider_message_id,
                delivered_at=receipt.occurred_at,
                delivery_detail=receipt.detail,
            )
        )


class SqlWorkflowRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _claim(row: WorkflowRun) -> WorkflowRunClaim:
        return WorkflowRunClaim(
            result=WorkflowClaimResult.ACQUIRED,
            run_id=WorkflowRunId(row.id),
            status=WorkflowRunStatus(row.status),
            intent=None if row.intent is None else WorkflowIntent(row.intent),
            attempts=row.attempts,
            lease_token=row.lease_token,
        )

    async def claim(
        self,
        business_id: BusinessId,
        inbound_message_id: MessageId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> WorkflowRunClaim:
        _validate_aware(now, "now")
        _validate_aware(stale_before, "stale_before")
        row = await self.session.scalar(
            select(WorkflowRun)
            .where(
                WorkflowRun.business_id == business_id,
                WorkflowRun.inbound_message_id == inbound_message_id,
            )
            .with_for_update()
        )
        if row is None:
            row = WorkflowRun(
                business_id=business_id,
                inbound_message_id=inbound_message_id,
                intent=None,
                status=WorkflowRunStatus.RUNNING.value,
                attempts=1,
                started_at=now,
                leased_at=now,
                lease_token=uuid4(),
            )
            try:
                async with self.session.begin_nested():
                    self.session.add(row)
                    await self.session.flush()
            except IntegrityError:
                row = await self.session.scalar(
                    select(WorkflowRun)
                    .where(
                        WorkflowRun.business_id == business_id,
                        WorkflowRun.inbound_message_id == inbound_message_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    raise
                return self._claim_existing(row, now, stale_before)
            return self._claim(row)
        return self._claim_existing(row, now, stale_before)

    def _claim_existing(
        self, row: WorkflowRun, now: datetime, stale_before: datetime
    ) -> WorkflowRunClaim:
        if row.status == WorkflowRunStatus.SUCCEEDED.value:
            return WorkflowRunClaim(
                result=WorkflowClaimResult.TERMINAL,
                run_id=WorkflowRunId(row.id),
                status=WorkflowRunStatus(row.status),
                intent=None if row.intent is None else WorkflowIntent(row.intent),
                attempts=row.attempts,
            )
        leased_at = row.leased_at
        if leased_at is not None and leased_at.tzinfo is None:
            leased_at = leased_at.replace(tzinfo=UTC)
        if row.status == WorkflowRunStatus.RUNNING.value and (
            leased_at is not None and leased_at > stale_before
        ):
            return WorkflowRunClaim(
                result=WorkflowClaimResult.BUSY,
                run_id=WorkflowRunId(row.id),
                status=WorkflowRunStatus(row.status),
                intent=None if row.intent is None else WorkflowIntent(row.intent),
                attempts=row.attempts,
            )
        row.status = WorkflowRunStatus.RUNNING.value
        row.attempts += 1
        row.leased_at = now
        row.lease_token = uuid4()
        row.finished_at = None
        row.error = None
        return self._claim(row)

    @staticmethod
    def _lease_filter(claim: WorkflowRunClaim) -> tuple[ColumnElement[bool], ...]:
        if claim.result is not WorkflowClaimResult.ACQUIRED or claim.lease_token is None:
            raise LostWorkflowLeaseError("workflow claim does not hold an active lease")
        return (
            WorkflowRun.id == claim.run_id,
            WorkflowRun.lease_token == claim.lease_token,
            WorkflowRun.status == WorkflowRunStatus.RUNNING.value,
        )

    async def set_intent(self, claim: WorkflowRunClaim, intent: WorkflowIntent) -> None:
        result = await self.session.execute(
            update(WorkflowRun).where(*self._lease_filter(claim)).values(intent=intent)
        )
        if _rowcount(result) != 1:
            raise LostWorkflowLeaseError("workflow claim is no longer active")

    async def set_error(self, claim: WorkflowRunClaim, error: str) -> None:
        result = await self.session.execute(
            update(WorkflowRun).where(*self._lease_filter(claim)).values(error=error)
        )
        if _rowcount(result) != 1:
            raise LostWorkflowLeaseError("workflow claim is no longer active")

    async def finish(
        self,
        claim: WorkflowRunClaim,
        status: WorkflowRunStatus,
        error: str | None = None,
    ) -> None:
        result = await self.session.execute(
            update(WorkflowRun)
            .where(*self._lease_filter(claim))
            .values(status=status.value, finished_at=datetime.now(UTC), error=error)
        )
        if _rowcount(result) != 1:
            raise LostWorkflowLeaseError("workflow claim is no longer active")


class SqlQuoteRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _quote(row: QuoteRecord) -> Quote:
        created_at = row.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        updated_at = row.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        return Quote.model_validate(
            {
                "quote_id": row.id,
                "business_id": row.business_id,
                "conversation_id": row.conversation_id,
                "conversation_ref": {
                    "business_id": row.business_id,
                    "external_conversation_id": row.external_conversation_id,
                },
                "status": row.status,
                "revision": row.revision,
                "source_message_key": row.source_message_key,
                "last_message_key": row.last_message_key,
                "pending_request_text": row.pending_request_text,
                "draft": row.draft,
                "approval_correlation_id": row.approval_correlation_id,
                "delivery_receipt": row.delivery_receipt,
                "customer_appointment": row.customer_appointment,
                "customer_candidates": row.customer_candidates,
                "claim_token": row.claim_token,
                "claim_token_hash": row.claim_token_hash,
                "customer_status": row.customer_status,
                "approved_at": _aware_or_none(row.approved_at),
                "texted_at": _aware_or_none(row.texted_at),
                "customer_id": row.customer_id,
                "version": row.version,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )

    @staticmethod
    def _customer_columns(quote: Quote) -> dict[str, object]:
        return {
            "customer_appointment": (
                quote.customer_appointment.model_dump(mode="json")
                if quote.customer_appointment is not None
                else None
            ),
            "customer_candidates": (
                [candidate.model_dump(mode="json") for candidate in quote.customer_candidates]
                if quote.customer_candidates is not None
                else None
            ),
            "claim_token": quote.claim_token,
            "claim_token_hash": quote.claim_token_hash,
            "customer_status": (
                quote.customer_status.value if quote.customer_status is not None else None
            ),
            "approved_at": quote.approved_at,
            "customer_id": quote.customer_id,
            "billing": (quote.draft.billing.value if quote.draft is not None else "one_time"),
            "billing_interval": (
                quote.draft.interval.value
                if quote.draft is not None and quote.draft.interval is not None
                else None
            ),
        }

    async def get(self, business_id: BusinessId, quote_id: QuoteId) -> Quote | None:
        row = await self.session.scalar(
            select(QuoteRecord).where(
                QuoteRecord.business_id == business_id,
                QuoteRecord.id == quote_id,
            )
        )
        return None if row is None else self._quote(row)

    async def get_active(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> Quote | None:
        row = await self.session.scalar(
            select(QuoteRecord).where(
                QuoteRecord.business_id == business_id,
                QuoteRecord.active_conversation_id == conversation_id,
            )
        )
        return None if row is None else self._quote(row)

    async def get_by_message(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        message_key: MessageKey,
    ) -> Quote | None:
        row = await self.session.scalar(
            select(QuoteRecord).where(
                QuoteRecord.business_id == business_id,
                QuoteRecord.conversation_id == conversation_id,
                QuoteRecord.last_message_key == message_key,
            )
        )
        return None if row is None else self._quote(row)

    async def get_by_claim_hash(self, claim_token_hash: str) -> Quote | None:
        row = await self.session.scalar(
            select(QuoteRecord).where(QuoteRecord.claim_token_hash == claim_token_hash)
        )
        return None if row is None else self._quote(row)

    async def list_for_customer(
        self, business_id: BusinessId, customer_id: CustomerId
    ) -> tuple[Quote, ...]:
        rows = await self.session.scalars(
            select(QuoteRecord)
            .where(
                QuoteRecord.business_id == business_id,
                QuoteRecord.customer_id == customer_id,
            )
            .order_by(QuoteRecord.created_at.desc())
        )
        return tuple(self._quote(row) for row in rows)

    async def list_for_business(
        self, business_id: BusinessId, *, limit: int, status: QuoteStatus | None = None
    ) -> tuple[Quote, ...]:
        query = select(QuoteRecord).where(QuoteRecord.business_id == business_id)
        if status is not None:
            query = query.where(QuoteRecord.status == status.value)
        rows = await self.session.scalars(
            query.order_by(QuoteRecord.created_at.desc()).limit(limit)
        )
        return tuple(self._quote(row) for row in rows)

    async def list_unlinked_for_email(
        self, business_id: BusinessId, email: str
    ) -> tuple[Quote, ...]:
        # The recipient lives inside the draft document, so the tenant's
        # unlinked claimable quotes are narrowed in SQL and matched here.
        rows = await self.session.scalars(
            select(QuoteRecord).where(
                QuoteRecord.business_id == business_id,
                QuoteRecord.customer_id.is_(None),
                QuoteRecord.status.in_([status.value for status in QUOTE_CLAIMABLE_STATUSES]),
            )
        )
        wanted = normalize_customer_email(email)
        quotes = (self._quote(row) for row in rows)
        return tuple(quote for quote in quotes if quote.recipient_email == wanted)

    async def add(self, quote: Quote) -> None:
        conversation = await self.session.scalar(
            select(Conversation).where(Conversation.id == quote.conversation_id)
        )
        if (
            conversation is None
            or conversation.business_id != quote.business_id
            or conversation.external_conversation_id
            != quote.conversation_ref.external_conversation_id
        ):
            raise CrossBusinessReferenceError(
                "quote references a conversation from another business"
            )
        row = QuoteRecord(
            id=quote.quote_id,
            business_id=quote.business_id,
            conversation_id=quote.conversation_id,
            active_conversation_id=(quote.conversation_id if quote.is_active else None),
            external_conversation_id=quote.conversation_ref.external_conversation_id,
            status=quote.status.value,
            revision=quote.revision,
            source_message_key=quote.source_message_key,
            last_message_key=quote.last_message_key,
            pending_request_text=quote.pending_request_text,
            draft=(quote.draft.model_dump(mode="json") if quote.draft is not None else None),
            approval_correlation_id=quote.approval_correlation_id,
            delivery_receipt=(
                quote.delivery_receipt.model_dump(mode="json")
                if quote.delivery_receipt is not None
                else None
            ),
            version=quote.version,
            created_at=quote.created_at,
            updated_at=quote.updated_at,
            **self._customer_columns(quote),
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError as error:
            raise QuoteConcurrencyError("quote already exists or conversation is active") from error

    async def save(self, quote: Quote, *, expected_version: int) -> None:
        result = await self.session.execute(
            update(QuoteRecord)
            .where(
                QuoteRecord.id == quote.quote_id,
                QuoteRecord.business_id == quote.business_id,
                QuoteRecord.version == expected_version,
            )
            .values(
                active_conversation_id=(quote.conversation_id if quote.is_active else None),
                status=quote.status.value,
                revision=quote.revision,
                last_message_key=quote.last_message_key,
                pending_request_text=quote.pending_request_text,
                draft=(quote.draft.model_dump(mode="json") if quote.draft is not None else None),
                approval_correlation_id=quote.approval_correlation_id,
                delivery_receipt=(
                    quote.delivery_receipt.model_dump(mode="json")
                    if quote.delivery_receipt is not None
                    else None
                ),
                version=quote.version,
                updated_at=quote.updated_at,
                **self._customer_columns(quote),
            )
        )
        if _rowcount(result) != 1:
            raise QuoteConcurrencyError("quote version is no longer current")

    async def record_texted(self, business_id: BusinessId, quote_id: QuoteId, at: datetime) -> None:
        await self.session.execute(
            update(QuoteRecord)
            .where(
                QuoteRecord.id == quote_id,
                QuoteRecord.business_id == business_id,
                QuoteRecord.texted_at.is_(None),
            )
            .values(texted_at=at)
            .execution_options(synchronize_session=False)
        )


class SqlOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def enqueue(self, command: OutboxCommand) -> None:
        if command.outbound_message_id is not None:
            outbound = await self.session.scalar(
                select(OutboundMessage).where(OutboundMessage.id == command.outbound_message_id)
            )
            if outbound is None or outbound.business_id != command.business_id:
                raise CrossBusinessReferenceError(
                    "outbox command references an outbound message from another business"
                )
        if command.inbound_message_id is not None:
            inbound = await self.session.scalar(
                select(InboundMessage).where(InboundMessage.id == command.inbound_message_id)
            )
            if inbound is None or inbound.business_id != command.business_id:
                raise CrossBusinessReferenceError(
                    "outbox command references an inbound message from another business"
                )
        existing = (
            await self.session.scalar(
                select(OutboxMessage).where(
                    OutboxMessage.business_id == command.business_id,
                    OutboxMessage.dedup_key == command.dedup_key,
                )
            )
            if command.dedup_key is not None
            else None
        )
        if existing is not None:
            return
        row = OutboxMessage(
            id=command.command_id,
            business_id=command.business_id,
            outbound_message_id=command.outbound_message_id,
            inbound_message_id=command.inbound_message_id,
            command_type=command.command_type,
            payload=command.payload,
            status=OutboxStatus.PENDING.value,
            attempts=0,
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            available_at=command.not_before or datetime.now(UTC),
            dedup_key=command.dedup_key,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            return

    async def claim_batch(
        self, limit: int, now: datetime, claimed_by: str, *, stale_before: datetime
    ) -> list[OutboxRecord]:
        _validate_aware(now, "now")
        _validate_aware(stale_before, "stale_before")
        rows = list(
            (
                await self.session.scalars(
                    select(OutboxMessage)
                    .where(
                        or_(
                            and_(
                                OutboxMessage.status.in_(
                                    [OutboxStatus.PENDING.value, OutboxStatus.FAILED.value]
                                ),
                                OutboxMessage.available_at <= now,
                            ),
                            and_(
                                OutboxMessage.status == OutboxStatus.IN_PROGRESS.value,
                                OutboxMessage.locked_at.is_not(None),
                                OutboxMessage.locked_at <= stale_before,
                            ),
                        ),
                    )
                    .order_by(OutboxMessage.available_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).all()
        )
        result: list[OutboxRecord] = []
        for row in rows:
            row.status = OutboxStatus.IN_PROGRESS.value
            row.attempts += 1
            row.locked_at = now
            row.locked_by = claimed_by
            result.append(
                OutboxRecord(
                    command=OutboxCommand(
                        command_id=row.id,
                        business_id=row.business_id,
                        command_type=row.command_type,
                        payload=row.payload,
                        dedup_key=row.dedup_key,
                        outbound_message_id=row.outbound_message_id,
                        inbound_message_id=row.inbound_message_id,
                    ),
                    status=OutboxStatus.IN_PROGRESS,
                    attempts=row.attempts,
                    max_attempts=row.max_attempts,
                    available_at=row.available_at,
                    last_error=row.last_error,
                    locked_by=row.locked_by,
                    claim_attempts=row.attempts,
                )
            )
        await self.session.flush()
        return result

    async def update(self, record: OutboxRecord) -> None:
        if (
            record.locked_by is None
            or record.claim_attempts is None
            or record.status is OutboxStatus.PENDING
        ):
            raise LostOutboxLeaseError("outbox record does not hold an active lease")
        result = await self.session.execute(
            update(OutboxMessage)
            .where(
                OutboxMessage.id == record.command.command_id,
                OutboxMessage.status == OutboxStatus.IN_PROGRESS.value,
                OutboxMessage.attempts == record.claim_attempts,
                OutboxMessage.locked_by == record.locked_by,
            )
            .values(
                status=record.status.value,
                attempts=record.attempts,
                max_attempts=record.max_attempts,
                available_at=record.available_at,
                last_error=record.last_error,
                locked_at=None,
                locked_by=None,
            )
        )
        if _rowcount(result) != 1:
            raise LostOutboxLeaseError("outbox record lease is no longer active")

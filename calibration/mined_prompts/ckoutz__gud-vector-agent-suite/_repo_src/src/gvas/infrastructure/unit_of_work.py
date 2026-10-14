from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.calendar_blocks import CalendarBlockRepository
from gvas.domain.completeness_repositories import (
    ChecklistDefinitionRepository,
    CompletenessUnitOfWork,
    FieldNoteReviewRepository,
    FollowUpQuestionRepository,
)
from gvas.domain.customers import (
    CustomerRepository,
    PortalLoginTokenRepository,
    PortalSessionRepository,
    ServiceRequestRepository,
)
from gvas.domain.intake import IntakeConversationRepository, IntakeMessageRepository
from gvas.domain.invoices import InvoiceRepository
from gvas.domain.owner import OwnerLoginTokenRepository, OwnerSessionRepository
from gvas.domain.payments import (
    PaymentEventRepository,
    PaymentLedgerRepository,
    QuotePaymentRepository,
    QuoteSubscriptionRepository,
)
from gvas.domain.quotes import QuoteRepository
from gvas.domain.repositories import (
    BusinessRepository,
    ConversationRepository,
    InboundMessageRepository,
    OutboundMessageRepository,
    OutboxRepository,
    OwnerChannelEndpointRepository,
    UnitOfWork,
    WorkflowRunRepository,
)
from gvas.domain.template_repositories import (
    BusinessTemplateProfileRepository,
    ReportTemplateDefinitionRepository,
    TemplateSetRepository,
)
from gvas.infrastructure.calendar_block_repositories import SqlCalendarBlockRepository
from gvas.infrastructure.completeness_repositories import (
    SqlChecklistDefinitionRepository,
    SqlFieldNoteReviewRepository,
    SqlFollowUpQuestionRepository,
)
from gvas.infrastructure.customer_repositories import (
    SqlCustomerRepository,
    SqlOwnerLoginTokenRepository,
    SqlOwnerSessionRepository,
    SqlPortalLoginTokenRepository,
    SqlPortalSessionRepository,
    SqlServiceRequestRepository,
)
from gvas.infrastructure.intake_repositories import (
    SqlIntakeConversationRepository,
    SqlIntakeMessageRepository,
)
from gvas.infrastructure.payment_repositories import (
    SqlInvoiceRepository,
    SqlPaymentEventRepository,
    SqlPaymentLedgerRepository,
    SqlQuotePaymentRepository,
    SqlQuoteSubscriptionRepository,
)
from gvas.infrastructure.repositories import (
    SqlBusinessRepository,
    SqlConversationRepository,
    SqlInboundMessageRepository,
    SqlOutboundMessageRepository,
    SqlOutboxRepository,
    SqlOwnerChannelEndpointRepository,
    SqlQuoteRepository,
    SqlWorkflowRunRepository,
)
from gvas.infrastructure.template_repositories import (
    SqlBusinessTemplateProfileRepository,
    SqlReportTemplateDefinitionRepository,
    SqlTemplateSetRepository,
)


class SqlUnitOfWork:
    businesses: BusinessRepository
    owner_channel_endpoints: OwnerChannelEndpointRepository
    conversations: ConversationRepository
    inbound_messages: InboundMessageRepository
    outbound_messages: OutboundMessageRepository
    workflow_runs: WorkflowRunRepository
    outbox: OutboxRepository
    quotes: QuoteRepository
    quote_payments: QuotePaymentRepository
    payments: PaymentLedgerRepository
    payment_events: PaymentEventRepository
    customers: CustomerRepository
    portal_login_tokens: PortalLoginTokenRepository
    portal_sessions: PortalSessionRepository
    service_requests: ServiceRequestRepository
    quote_subscriptions: QuoteSubscriptionRepository
    intake_conversations: IntakeConversationRepository
    intake_messages: IntakeMessageRepository
    owner_login_tokens: OwnerLoginTokenRepository
    owner_sessions: OwnerSessionRepository
    calendar_blocks: CalendarBlockRepository
    invoices: InvoiceRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SqlUnitOfWork":
        self._session = self._session_factory()
        self.businesses = SqlBusinessRepository(self._session)
        self.owner_channel_endpoints = SqlOwnerChannelEndpointRepository(self._session)
        self.conversations = SqlConversationRepository(self._session)
        self.inbound_messages = SqlInboundMessageRepository(self._session)
        self.outbound_messages = SqlOutboundMessageRepository(self._session)
        self.workflow_runs = SqlWorkflowRunRepository(self._session)
        self.outbox = SqlOutboxRepository(self._session)
        self.quotes = SqlQuoteRepository(self._session)
        self.quote_payments = SqlQuotePaymentRepository(self._session)
        self.payments = SqlPaymentLedgerRepository(self._session)
        self.payment_events = SqlPaymentEventRepository(self._session)
        self.customers = SqlCustomerRepository(self._session)
        self.portal_login_tokens = SqlPortalLoginTokenRepository(self._session)
        self.portal_sessions = SqlPortalSessionRepository(self._session)
        self.service_requests = SqlServiceRequestRepository(self._session)
        self.quote_subscriptions = SqlQuoteSubscriptionRepository(self._session)
        self.intake_conversations = SqlIntakeConversationRepository(self._session)
        self.intake_messages = SqlIntakeMessageRepository(self._session)
        self.owner_login_tokens = SqlOwnerLoginTokenRepository(self._session)
        self.owner_sessions = SqlOwnerSessionRepository(self._session)
        self.calendar_blocks = SqlCalendarBlockRepository(self._session)
        self.invoices = SqlInvoiceRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            await self._session.rollback()
        await self._session.close()

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.rollback()


class SqlUnitOfWorkFactory:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> UnitOfWork:
        return SqlUnitOfWork(self._session_factory)


class SqlCompletenessUnitOfWork:
    checklists: ChecklistDefinitionRepository
    template_sets: TemplateSetRepository
    report_templates: ReportTemplateDefinitionRepository
    business_template_profiles: BusinessTemplateProfileRepository
    field_note_reviews: FieldNoteReviewRepository
    follow_up_questions: FollowUpQuestionRepository
    outbound_messages: OutboundMessageRepository
    outbox: OutboxRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SqlCompletenessUnitOfWork":
        self._session = self._session_factory()
        self.checklists = SqlChecklistDefinitionRepository(self._session)
        self.template_sets = SqlTemplateSetRepository(self._session)
        self.report_templates = SqlReportTemplateDefinitionRepository(self._session)
        self.business_template_profiles = SqlBusinessTemplateProfileRepository(self._session)
        self.field_note_reviews = SqlFieldNoteReviewRepository(self._session)
        self.follow_up_questions = SqlFollowUpQuestionRepository(self._session)
        self.outbound_messages = SqlOutboundMessageRepository(self._session)
        self.outbox = SqlOutboxRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            await self._session.rollback()
        await self._session.close()

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.rollback()


class SqlCompletenessUnitOfWorkFactory:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> CompletenessUnitOfWork:
        return SqlCompletenessUnitOfWork(self._session_factory)

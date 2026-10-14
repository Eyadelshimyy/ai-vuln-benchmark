from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from gvas.application.calendar_blocks import CalendarBlockHandler
from gvas.application.channel_policy import (
    ChannelScopedIntentResolver,
    ChannelUnsupportedMessageHandler,
    ChannelWorkflowPolicy,
)
from gvas.application.completeness import FieldNoteCompletenessService
from gvas.application.docx_report import DocxReportRenderer
from gvas.application.field_note_transcription import (
    FieldNoteTranscriptService,
    TranscribeFieldNoteAudioService,
)
from gvas.application.field_notes import (
    CloseFieldNoteCaseHandler,
    FieldNoteIntakeHandler,
    FieldNoteIntentContribution,
)
from gvas.application.ingestion import IngestOwnerMessageService
from gvas.application.intake import (
    ArrangeIntakeBookingService,
    BookingDecisionHandler,
    CancelIntakeBookingService,
    IntakeBookingEventService,
    IntakeDecisionLinkService,
    IntakeService,
    IntakeVisitor,
    SendIntakeCustomerEmailService,
    SendIntakeCustomerTextService,
    SendOwnerEmailService,
)
from gvas.application.manual_payments import ManualPaymentEffectsService
from gvas.application.outbox_service import OutboxService
from gvas.application.owner import OwnerService
from gvas.application.owner_assistant import OwnerAssistant, OwnerReadTools
from gvas.application.owner_reply_delivery import DeliverOwnerReplyService
from gvas.application.plan_custody import (
    CopyPlanSetIntoCustodyService,
    RegisterPlanSetUploadService,
)
from gvas.application.portal import PortalService
from gvas.application.processing import ProcessOwnerMessageService
from gvas.application.public_quotes import DEFAULT_DEPLOYMENT, PublicQuoteService
from gvas.application.quotes import (
    DeliverApprovedQuoteService,
    QuoteIntentSelector,
    QuoteWorkflowHandler,
    TextDeliveredQuoteService,
)
from gvas.application.report_approval import ApproveFieldNoteReportHandler
from gvas.application.report_email import SendFieldNoteReportHandler
from gvas.application.report_generation import GenerateFieldNotesReportService
from gvas.application.templates import PublishTemplateSetService, TemplateResolver
from gvas.application.unmatched_messages import UnmatchedMessageHandler
from gvas.application.workflow_conflicts import WorkflowConflictHandler
from gvas.composition.dispatcher import OutboxCommandDispatcher, OutboxWorker
from gvas.composition.failure_notices import NotifyExhaustedCommandService
from gvas.composition.field_note_workflow import FieldNoteWorkflowHandler
from gvas.composition.intents import DeterministicIntentResolver
from gvas.composition.report_delivery import DeliverFieldNotesReportService
from gvas.composition.report_email import EmailFieldNotesReportService
from gvas.composition.report_publication import (
    PublishFieldNotesReportService,
    ReportArtifactAccess,
)
from gvas.composition.review import CoordinateFieldNoteReviewService
from gvas.composition.snapshots import BuildFieldNoteCaseSnapshotService
from gvas.config import IntakeSettings, Settings
from gvas.domain.completeness import CompletenessReviewPort
from gvas.domain.identifiers import BusinessId
from gvas.domain.owner_assistant import OwnerAssistantPort
from gvas.domain.ports import (
    AppointmentLookupPort,
    AttachmentAccessPort,
    AvailabilityPort,
    BillingAccountPort,
    BookedEventsPort,
    CalendarFeedPort,
    ChecklistEvidencePort,
    CustomerQuoteDeliveryPort,
    CustomerTextDeliveryPort,
    IntakeAgentPort,
    IntentResolutionPort,
    ObjectStoragePort,
    OwnerEmailPort,
    OwnerReplyPort,
    PaymentCheckoutPort,
    PortalLoginEmailPort,
    QuoteDraftingPort,
    ScheduleBlockPort,
    TranscriptionPort,
)
from gvas.domain.reporting import ReportEmailPort, ReportGenerationPort
from gvas.domain.usage import UsageCeilingGuard, UsageCeilings, UsageLedgerPort
from gvas.domain.workflows import WorkflowRouter
from gvas.infrastructure.db import create_engine, create_session_factory
from gvas.infrastructure.field_note_repositories import SqlFieldNoteUnitOfWorkFactory
from gvas.infrastructure.plan_repositories import SqlPlanCustodyUnitOfWorkFactory
from gvas.infrastructure.reporting_unit_of_work import SqlReportUnitOfWorkFactory
from gvas.infrastructure.unit_of_work import (
    SqlCompletenessUnitOfWorkFactory,
    SqlUnitOfWorkFactory,
)
from gvas.infrastructure.usage_ledger import SqlUsageLedger


@dataclass(frozen=True)
class ApplicationPorts:
    """Provider-neutral ports the composition root requires.

    Production adapters are injected here; the repository ships no provider
    implementation, so tests and local runs supply deterministic fakes.
    """

    owner_replies: OwnerReplyPort
    quote_drafting: QuoteDraftingPort
    quote_delivery: CustomerQuoteDeliveryPort
    transcription: TranscriptionPort
    completeness_review: CompletenessReviewPort
    checklist_evidence: ChecklistEvidencePort
    report_generation: ReportGenerationPort
    source_attachments: AttachmentAccessPort | None = None
    object_storage: ObjectStoragePort | None = None
    report_email: ReportEmailPort | None = None
    # When present, a delivered quote whose receipt carries a customer link is
    # also texted to the customer's phone; when absent nothing is texted.
    customer_text: CustomerTextDeliveryPort | None = None
    # When absent a quote must carry ``customer:``; when present the quote
    # workflow may resolve the customer from the owner's appointments instead.
    appointment_lookup: AppointmentLookupPort | None = None
    channel_policies: tuple[ChannelWorkflowPolicy, ...] = ()
    # When present, the public API can open a hosted checkout on quote accept;
    # when absent the accept route answers 503 and the rest still serves.
    payment_checkout: PaymentCheckoutPort | None = None
    # When present, recurring quotes can be accepted (a provider customer is
    # created on first accept) and the portal can open a billing portal.
    billing_accounts: BillingAccountPort | None = None
    # When present the worker sends portal magic-link e-mails.
    portal_login_email: PortalLoginEmailPort | None = None
    # Website booking chat: the model that answers turns, the calendar that
    # reports openings and books approved slots, and the plain e-mail sender
    # for customer notices (kept apart from quote_delivery, which may route to
    # an external portal).
    intake_agent: IntakeAgentPort | None = None
    # Owner messages that match no command get a plain-language, read-only
    # answer from this model; absent means the owner gets the command list.
    owner_assistant: OwnerAssistantPort | None = None
    availability: AvailabilityPort | None = None
    customer_email: CustomerQuoteDeliveryPort | None = None
    # The ledger the metered adapters write to; the ceiling guard reads the same
    # one. Defaults to the SQL ledger on the application's sessions.
    usage_ledger: UsageLedgerPort | None = None
    # Owner dashboard calendar: Calendly bookings and the owner's own
    # calendar feed. Either may be absent; the dashboard shows what it has.
    booked_events: BookedEventsPort | None = None
    calendar_feed: CalendarFeedPort | None = None
    # Owner-requested blocks of time (``unavailable 8-12``) on the booking
    # calendar; absent means the owner is told blocking isn't set up.
    schedule_blocks: ScheduleBlockPort | None = None
    # Owner notices as text + HTML; absent means they go out text-only
    # through ``customer_email``.
    owner_email: OwnerEmailPort | None = None


@dataclass(frozen=True)
class Application:
    engine: AsyncEngine | None
    session_factory: async_sessionmaker[AsyncSession]
    unit_of_work_factory: SqlUnitOfWorkFactory
    field_note_unit_of_work_factory: SqlFieldNoteUnitOfWorkFactory
    completeness_unit_of_work_factory: SqlCompletenessUnitOfWorkFactory
    report_unit_of_work_factory: SqlReportUnitOfWorkFactory
    plan_custody_unit_of_work_factory: SqlPlanCustodyUnitOfWorkFactory
    router: WorkflowRouter
    intent_resolver: IntentResolutionPort
    ingest_service: IngestOwnerMessageService
    processing_service: ProcessOwnerMessageService
    owner_reply_service: DeliverOwnerReplyService
    quote_delivery_service: DeliverApprovedQuoteService
    transcription_service: TranscribeFieldNoteAudioService
    transcript_service: FieldNoteTranscriptService
    template_resolver: TemplateResolver
    template_publisher: PublishTemplateSetService
    completeness_service: FieldNoteCompletenessService
    review_service: CoordinateFieldNoteReviewService
    snapshot_service: BuildFieldNoteCaseSnapshotService
    report_service: GenerateFieldNotesReportService
    report_delivery_service: DeliverFieldNotesReportService
    report_publication_service: PublishFieldNotesReportService
    report_email_service: EmailFieldNotesReportService
    report_artifacts: ReportArtifactAccess
    public_quotes: PublicQuoteService
    portal: PortalService
    owner: OwnerService
    owner_assistant: OwnerAssistant | None
    intake: IntakeService | None
    intake_booking_events: IntakeBookingEventService
    # Wired only when the intake decision-link secret and base URL are set.
    intake_decision_links: IntakeDecisionLinkService | None
    failure_notice_service: NotifyExhaustedCommandService
    usage_ledger: UsageLedgerPort
    usage_ceilings: UsageCeilings
    plan_set_upload_service: RegisterPlanSetUploadService
    plan_custody_service: CopyPlanSetIntoCustodyService | None
    outbox: OutboxService
    dispatcher: OutboxCommandDispatcher
    worker: OutboxWorker


def _utcnow() -> datetime:
    return datetime.now(UTC)


def build_application(
    ports: ApplicationPorts,
    settings: Settings | None = None,
    *,
    intent_resolver: IntentResolutionPort | None = None,
    now: Callable[[], datetime] = _utcnow,
    lease_ttl: timedelta = timedelta(minutes=5),
    engine: AsyncEngine | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    ceilings: UsageCeilings | None = None,
    intake_settings: IntakeSettings | None = None,
    payment_deployment: str = DEFAULT_DEPLOYMENT,
    intake_message_budget: Callable[[UUID, datetime], Awaitable[str | None]] | None = None,
    intake_visitor: Callable[[UUID], Awaitable[IntakeVisitor | None]] | None = None,
    payments_off: Callable[[BusinessId], Awaitable[bool]] | None = None,
) -> Application:
    resolved_engine = engine
    if resolved_engine is None and session_factory is None:
        resolved_engine = create_engine((settings or Settings()).database_url)
    if session_factory is not None:
        sessions = session_factory
    elif resolved_engine is not None:
        sessions = create_session_factory(resolved_engine)
    else:
        raise ValueError("an engine or session factory is required")

    unit_of_work_factory = SqlUnitOfWorkFactory(sessions)
    field_note_unit_of_work_factory = SqlFieldNoteUnitOfWorkFactory(sessions)
    completeness_unit_of_work_factory = SqlCompletenessUnitOfWorkFactory(sessions)
    report_unit_of_work_factory = SqlReportUnitOfWorkFactory(sessions)
    plan_custody_unit_of_work_factory = SqlPlanCustodyUnitOfWorkFactory(sessions)
    usage_ledger = ports.usage_ledger or SqlUsageLedger(sessions)
    usage_ceilings = ceilings or UsageCeilings()
    ceiling_guard = UsageCeilingGuard(usage_ledger, usage_ceilings)

    intake = FieldNoteIntakeHandler(field_note_unit_of_work_factory, now=now)
    closure = CloseFieldNoteCaseHandler(field_note_unit_of_work_factory, now=now)
    approval = ApproveFieldNoteReportHandler(
        field_note_unit_of_work_factory, report_unit_of_work_factory
    )
    plan_set_uploads = RegisterPlanSetUploadService(plan_custody_unit_of_work_factory)
    plan_custody = (
        CopyPlanSetIntoCustodyService(
            plan_custody_unit_of_work_factory, ports.source_attachments, ports.object_storage
        )
        if ports.source_attachments is not None and ports.object_storage is not None
        else None
    )
    report_send = SendFieldNoteReportHandler(
        field_note_unit_of_work_factory, report_unit_of_work_factory, unit_of_work_factory
    )
    field_note_handler = FieldNoteWorkflowHandler(
        intake,
        closure,
        approval,
        report_send,
        field_note_unit_of_work_factory,
        completeness_unit_of_work_factory,
        now=now,
        plan_set_uploads=plan_set_uploads if plan_custody is not None else None,
    )
    quote_handler = QuoteWorkflowHandler(
        unit_of_work_factory,
        ports.quote_drafting,
        appointment_lookup=ports.appointment_lookup,
    )
    booking_handler = BookingDecisionHandler(unit_of_work_factory, now=now)
    calendar_block_handler = CalendarBlockHandler(
        unit_of_work_factory, ports.schedule_blocks, now=now
    )
    resolved_intake_settings = intake_settings or IntakeSettings()
    intake_service = (
        IntakeService(
            unit_of_work_factory,
            agent=ports.intake_agent,
            availability=ports.availability,
            ceiling=ceiling_guard,
            max_conversations_per_day=resolved_intake_settings.max_conversations_per_day,
            max_user_messages=resolved_intake_settings.max_messages_per_conversation,
            message_budget=intake_message_budget,
            visitor=intake_visitor,
            decision_link_secret=resolved_intake_settings.decision_link_secret,
            decision_link_base_url=resolved_intake_settings.decision_link_base_url,
            now=now,
        )
        if ports.intake_agent is not None
        else None
    )
    intake_decision_links = (
        IntakeDecisionLinkService(
            unit_of_work_factory,
            secret=resolved_intake_settings.decision_link_secret,
            now=now,
        )
        if resolved_intake_settings.decision_links_enabled
        else None
    )
    public_quotes = PublicQuoteService(
        unit_of_work_factory,
        checkout=ports.payment_checkout,
        billing_accounts=ports.billing_accounts,
        deployment=payment_deployment,
        payments_off=payments_off,
    )
    portal = PortalService(unit_of_work_factory, billing_accounts=ports.billing_accounts, now=now)
    owner = OwnerService(
        unit_of_work_factory,
        booked_events=ports.booked_events,
        calendar_feed=ports.calendar_feed,
        now=now,
    )
    owner_assistant = (
        OwnerAssistant(
            unit_of_work_factory,
            ports.owner_assistant,
            OwnerReadTools(unit_of_work_factory, owner, now=now),
            ceilings=ceiling_guard,
            now=now,
        )
        if ports.owner_assistant is not None
        else None
    )
    router = WorkflowRouter(
        [
            quote_handler,
            booking_handler,
            calendar_block_handler,
            field_note_handler,
            WorkflowConflictHandler(),
            UnmatchedMessageHandler(owner_assistant),
            *(ChannelUnsupportedMessageHandler(policy) for policy in ports.channel_policies),
        ]
    )

    resolver: IntentResolutionPort = intent_resolver or DeterministicIntentResolver(
        FieldNoteIntentContribution(field_note_unit_of_work_factory),
        QuoteIntentSelector(),
        field_note_unit_of_work_factory,
        unit_of_work_factory,
    )
    if ports.channel_policies:
        resolver = ChannelScopedIntentResolver(
            resolver, ports.channel_policies, unit_of_work_factory
        )

    transcripts = FieldNoteTranscriptService(field_note_unit_of_work_factory)
    template_resolver = TemplateResolver(completeness_unit_of_work_factory)
    template_publisher = PublishTemplateSetService(completeness_unit_of_work_factory)
    completeness = FieldNoteCompletenessService(
        completeness_unit_of_work_factory, ports.completeness_review, template_resolver
    )
    review = CoordinateFieldNoteReviewService(
        field_note_unit_of_work_factory,
        unit_of_work_factory,
        transcripts,
        completeness,
        ceilings=ceiling_guard,
    )
    snapshots = BuildFieldNoteCaseSnapshotService(
        completeness_unit_of_work_factory, ports.checklist_evidence, template_resolver
    )
    reports = GenerateFieldNotesReportService(
        report_unit_of_work_factory, ports.report_generation, template_resolver
    )
    processing = ProcessOwnerMessageService(unit_of_work_factory, router, resolver)
    owner_replies = DeliverOwnerReplyService(unit_of_work_factory, ports.owner_replies)
    quote_delivery = DeliverApprovedQuoteService(
        unit_of_work_factory, ports.quote_delivery, ports.customer_text
    )
    quote_text = (
        TextDeliveredQuoteService(unit_of_work_factory, ports.customer_text)
        if ports.customer_text is not None
        else None
    )
    transcription = TranscribeFieldNoteAudioService(
        field_note_unit_of_work_factory, ports.transcription, ceilings=ceiling_guard
    )
    report_delivery = DeliverFieldNotesReportService(
        field_note_unit_of_work_factory, unit_of_work_factory
    )
    report_renderer = DocxReportRenderer()
    report_publication = PublishFieldNotesReportService(
        report_renderer,
        report_unit_of_work_factory,
        field_note_unit_of_work_factory,
        unit_of_work_factory,
        ports.object_storage,
    )
    report_email = EmailFieldNotesReportService(
        report_renderer,
        ports.report_email,
        report_unit_of_work_factory,
        field_note_unit_of_work_factory,
        unit_of_work_factory,
    )
    report_artifacts = ReportArtifactAccess(report_renderer, report_unit_of_work_factory)
    failure_notices = NotifyExhaustedCommandService(
        unit_of_work_factory, field_note_unit_of_work_factory
    )
    outbox = OutboxService(unit_of_work_factory)
    dispatcher = OutboxCommandDispatcher(
        processing=processing,
        owner_replies=owner_replies,
        quote_delivery=quote_delivery,
        transcription=transcription,
        review=review,
        snapshots=snapshots,
        reports=reports,
        report_delivery=report_delivery,
        report_publication=report_publication,
        report_email=report_email,
        outbox=outbox,
        now=now,
        plan_custody=plan_custody,
        lease_ttl=lease_ttl,
        ceiling_notices=failure_notices,
        quote_text=quote_text,
        portal_login_email=ports.portal_login_email,
        intake_booking=ArrangeIntakeBookingService(
            unit_of_work_factory, availability=ports.availability, now=now
        ),
        intake_email=(
            SendIntakeCustomerEmailService(ports.customer_email)
            if ports.customer_email is not None
            else None
        ),
        intake_text=(
            SendIntakeCustomerTextService(ports.customer_text, unit_of_work_factory)
            if ports.customer_text is not None
            else None
        ),
        intake_booking_cancel=CancelIntakeBookingService(availability=ports.availability),
        owner_email=(
            SendOwnerEmailService(ports.owner_email) if ports.owner_email is not None else None
        ),
        manual_payments=ManualPaymentEffectsService(
            unit_of_work_factory,
            checkout=ports.payment_checkout,
            receipts=(
                SendIntakeCustomerEmailService(ports.customer_email).send
                if ports.customer_email is not None
                else None
            ),
        ),
    )
    return Application(
        engine=resolved_engine,
        session_factory=sessions,
        unit_of_work_factory=unit_of_work_factory,
        field_note_unit_of_work_factory=field_note_unit_of_work_factory,
        completeness_unit_of_work_factory=completeness_unit_of_work_factory,
        report_unit_of_work_factory=report_unit_of_work_factory,
        plan_custody_unit_of_work_factory=plan_custody_unit_of_work_factory,
        router=router,
        intent_resolver=resolver,
        ingest_service=IngestOwnerMessageService(unit_of_work_factory),
        processing_service=processing,
        owner_reply_service=owner_replies,
        quote_delivery_service=quote_delivery,
        transcription_service=transcription,
        transcript_service=transcripts,
        template_resolver=template_resolver,
        template_publisher=template_publisher,
        completeness_service=completeness,
        review_service=review,
        snapshot_service=snapshots,
        report_service=reports,
        report_delivery_service=report_delivery,
        report_publication_service=report_publication,
        report_email_service=report_email,
        report_artifacts=report_artifacts,
        public_quotes=public_quotes,
        portal=portal,
        owner=owner,
        owner_assistant=owner_assistant,
        intake=intake_service,
        intake_booking_events=IntakeBookingEventService(
            unit_of_work_factory,
            decision_link_secret=resolved_intake_settings.decision_link_secret,
            decision_link_origin=resolved_intake_settings.decision_link_origin(),
            now=now,
        ),
        intake_decision_links=intake_decision_links,
        failure_notice_service=failure_notices,
        usage_ledger=usage_ledger,
        usage_ceilings=usage_ceilings,
        plan_set_upload_service=plan_set_uploads,
        plan_custody_service=plan_custody,
        outbox=outbox,
        dispatcher=dispatcher,
        worker=OutboxWorker(
            outbox,
            dispatcher,
            now=now,
            lease_ttl=lease_ttl,
            failure_notices=failure_notices,
        ),
    )

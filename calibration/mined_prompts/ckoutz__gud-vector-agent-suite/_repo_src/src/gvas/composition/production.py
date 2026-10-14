"""Production wiring: concrete providers, the mounted ingress, and settings.

``build_application`` stays provider-neutral so tests can inject fakes; this
module is the only place that decides which providers the deployment uses. It
also refuses to start when a required setting is absent, because a half
configured process would accept Slack events and then fail every command in the
worker instead of failing the deploy.

Completeness review stays deterministic (marker reviewer), but a review may
only complete once the OpenAI contradiction pass has cleared it. Evidence
attribution stays deterministic (marker attributor); the OpenAI annotator only
adds verbatim supporting excerpts to items the markers satisfied and is skipped
on any failure. Report generation remains deterministic. Swapping a model in or
out is a change to this module and the ports it fills, not to the application.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from gvas.application.channel_policy import ChannelWorkflowPolicy
from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.contradiction_guard import GuardedCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.application.docx_report import DocxReportRenderer
from gvas.application.guarded_checklist_evidence import GuardedChecklistEvidenceAttributor
from gvas.application.public_quotes import DEFAULT_DEPLOYMENT
from gvas.application.quotes import SiteAwareQuoteDelivery
from gvas.composition import Application, ApplicationPorts, build_application
from gvas.composition.report_publication import ReportArtifactAccess
from gvas.config import (
    CostCeilingSettings,
    DatabaseUrlError,
    DemoSettings,
    IntakeSettings,
    ObjectStorageSettings,
    OpenAISettings,
    PublicApiSettings,
    ResendSettings,
    Settings,
    WorkerSettings,
    require_managed_postgres_url,
)
from gvas.domain.ports import (
    CustomerQuoteDeliveryPort,
    CustomerTextDeliveryPort,
    OwnerReplyPort,
    QuoteDraftingPort,
)
from gvas.domain.usage import UsageCeilingGuard, UsageCeilings
from gvas.infrastructure.calendar import IcsCalendarFeed
from gvas.infrastructure.calendly.api import CalendlyAppointmentLookup
from gvas.infrastructure.calendly.availability import CalendlyAvailability
from gvas.infrastructure.calendly.calendar import CalendlyBookedEvents
from gvas.infrastructure.calendly.composition import build_calendly_webhook_router
from gvas.infrastructure.calendly.config import (
    CalendlyInstallationError,
    CalendlySettings,
    parse_calendly_installations,
)
from gvas.infrastructure.db import create_engine, create_session_factory
from gvas.infrastructure.delivery_ledger import SqlChannelDeliveryLedger
from gvas.infrastructure.demo import (
    DemoAvailability,
    DemoBookedEvents,
    LoggedCustomerEmail,
    LoggedCustomerText,
    LoggedOwnerEmail,
    LoggedOwnerReply,
    LoggedPortalLoginEmail,
    LoggedReportEmail,
    NoAttachments,
)
from gvas.infrastructure.object_storage import R2ObjectStorage
from gvas.infrastructure.openai_checklist_evidence import OpenAIChecklistEvidenceAnnotator
from gvas.infrastructure.openai_contradiction_guard import OpenAIContradictionGuard
from gvas.infrastructure.openai_intake_agent import OpenAIIntakeAgent
from gvas.infrastructure.openai_owner_assistant import OpenAIOwnerAssistant
from gvas.infrastructure.openai_quote_drafting import OpenAIFreeTextQuoteDrafter
from gvas.infrastructure.openai_transcription import OpenAITranscriber
from gvas.infrastructure.owner_reply_routing import ChannelOwnerReplyRouter
from gvas.infrastructure.portal import PortalQuoteDelivery, PortalSettings, SqlPortalHandoffLedger
from gvas.infrastructure.quote_drafting import (
    DeterministicQuoteDrafter,
    ModelAssistedQuoteDrafter,
)
from gvas.infrastructure.reporting_unit_of_work import SqlReportUnitOfWorkFactory
from gvas.infrastructure.repositories import SqlBusinessRepository
from gvas.infrastructure.resend import (
    ResendOwnerEmailAdapter,
    ResendPortalLoginEmailAdapter,
    ResendQuoteDeliveryAdapter,
    ResendReportEmailAdapter,
)
from gvas.infrastructure.slack.api import (
    SlackFileAttachmentAccess,
    SlackWebApiChatPoster,
    SlackWebApiFileUploader,
)
from gvas.infrastructure.slack.composition import (
    build_slack_event_router,
    build_slack_owner_reply_adapter,
)
from gvas.infrastructure.slack.config import SlackSettings
from gvas.infrastructure.slack.installations import (
    SLACK_SOURCE_NAMESPACE,
    SlackInstallationError,
    parse_slack_installations,
)
from gvas.infrastructure.stripe import (
    StripeCheckout,
    StripeSettings,
    StripeWebhookVerifier,
)
from gvas.infrastructure.telnyx.api import TelnyxMessagingApiSender
from gvas.infrastructure.telnyx.composition import (
    build_telnyx_owner_reply_adapter,
    build_telnyx_webhook_router,
    sms_quotes_only_policy,
)
from gvas.infrastructure.telnyx.config import TelnyxSettings
from gvas.infrastructure.telnyx.customer_text import TelnyxCustomerTextAdapter
from gvas.infrastructure.telnyx.installations import (
    TELNYX_SOURCE_NAMESPACE,
    TelnyxInstallationError,
    parse_telnyx_installations,
)
from gvas.infrastructure.usage_ledger import SqlUsageLedger
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.owner import create_owner_router
from gvas.interfaces.http.portal import create_portal_router
from gvas.interfaces.http.public import PerIpRateLimiter, create_public_router
from gvas.interfaces.logging_setup import configure_logging

if TYPE_CHECKING:
    from gvas.interfaces.demo_sandboxes import DemoSandboxes

logger = logging.getLogger(__name__)

R2_SETTING_NAMES = (
    "GVAS_R2_ACCOUNT_ID",
    "GVAS_R2_BUCKET",
    "GVAS_R2_ACCESS_KEY_ID",
    "GVAS_R2_SECRET_ACCESS_KEY",
)


class ProductionConfigurationError(RuntimeError):
    """Raised at startup when required settings are missing or malformed.

    The message names the environment variables only; values never appear.
    """


@dataclass(frozen=True)
class ProductionSettings:
    app: Settings
    slack: SlackSettings
    openai: OpenAISettings
    resend: ResendSettings
    worker: WorkerSettings
    storage: ObjectStorageSettings = field(default_factory=ObjectStorageSettings)
    telnyx: TelnyxSettings = field(default_factory=TelnyxSettings)
    calendly: CalendlySettings = field(default_factory=CalendlySettings)
    portal: PortalSettings = field(default_factory=PortalSettings)
    stripe: StripeSettings = field(default_factory=StripeSettings)
    public_api: PublicApiSettings = field(default_factory=PublicApiSettings)
    cost_ceilings: CostCeilingSettings = field(default_factory=CostCeilingSettings)
    intake: IntakeSettings = field(default_factory=IntakeSettings)
    demo: DemoSettings = field(default_factory=DemoSettings)

    def usage_ceilings(self) -> UsageCeilings:
        return UsageCeilings(
            transcription_seconds=self.cost_ceilings.transcription_seconds,
            review_tokens=self.cost_ceilings.review_tokens,
        )


def load_production_settings() -> ProductionSettings:
    settings = ProductionSettings(
        app=Settings(),
        slack=SlackSettings(),
        openai=OpenAISettings(),
        resend=ResendSettings(),
        worker=WorkerSettings(),
        storage=ObjectStorageSettings(),
        telnyx=TelnyxSettings(),
        calendly=CalendlySettings(),
        portal=PortalSettings(),
        stripe=StripeSettings(),
        public_api=PublicApiSettings(),
        cost_ceilings=CostCeilingSettings(),
        intake=IntakeSettings(),
        demo=DemoSettings(),
    )
    if settings.demo.mode:
        _require_demo_isolation(settings)
        return settings
    missing = [
        name
        for name, present in (
            # The localhost default exists for development; a deployed process
            # that inherited it would quietly run against nothing.
            (
                "GVAS_DATABASE_URL or DATABASE_URL",
                "database_url" in settings.app.model_fields_set and bool(settings.app.database_url),
            ),
            ("GVAS_SLACK_SIGNING_SECRET", bool(settings.slack.signing_secret)),
            ("GVAS_SLACK_BOT_TOKEN", bool(settings.slack.bot_token)),
            ("GVAS_SLACK_INSTALLATIONS", bool(settings.slack.installations)),
            ("GVAS_OPENAI_API_KEY", settings.openai.is_configured),
            ("GVAS_RESEND_API_KEY", bool(settings.resend.api_key)),
            ("GVAS_RESEND_FROM_ADDRESS", bool(settings.resend.from_address)),
        )
        if not present
    ]
    if missing:
        raise ProductionConfigurationError(f"missing required settings: {', '.join(missing)}")
    _require_managed_database(settings.app.database_url)
    _require_single_owner(settings.slack.installations)
    _require_complete_object_storage(settings.storage)
    _require_complete_telnyx_channel(settings.telnyx)
    _require_complete_calendly_lookup(settings.calendly)
    _require_complete_portal_handoff(settings.portal)
    _require_complete_stripe_checkout(settings.stripe)
    return settings


# Stripe test-mode secret and restricted keys; anything else could be live.
STRIPE_TEST_KEY_PREFIXES = ("sk_test_", "rk_test_")


def _require_demo_isolation(settings: ProductionSettings) -> None:
    """A demo runs a fictional business with nothing sent, so it must not
    hold a single credential that could reach a real person or account.

    It needs only its own database and the model key behind Gus. E-mail,
    texts, Slack, Calendly, the external portal and object storage must be
    unset (the demo logs or generates them instead), and Stripe, if set at
    all, must be a test-mode key. Names are reported, never values.
    """

    missing = [
        name
        for name, present in (
            (
                "GVAS_DATABASE_URL or DATABASE_URL",
                "database_url" in settings.app.model_fields_set and bool(settings.app.database_url),
            ),
            ("GVAS_OPENAI_API_KEY", settings.openai.is_configured),
        )
        if not present
    ]
    if missing:
        raise ProductionConfigurationError(f"missing required settings: {', '.join(missing)}")
    _require_managed_database(settings.app.database_url)
    storage = settings.storage
    held = [
        name
        for name, present in (
            ("GVAS_SLACK_SIGNING_SECRET", bool(settings.slack.signing_secret)),
            ("GVAS_SLACK_BOT_TOKEN", bool(settings.slack.bot_token)),
            ("GVAS_SLACK_INSTALLATIONS", bool(settings.slack.installations)),
            ("GVAS_RESEND_API_KEY", bool(settings.resend.api_key)),
            *settings.telnyx.required_settings.items(),
            ("GVAS_TELNYX_MESSAGING_PROFILE_ID", bool(settings.telnyx.messaging_profile_id)),
            *settings.calendly.required_settings.items(),
            ("GVAS_CALENDLY_WEBHOOK_SIGNING_KEY", bool(settings.calendly.webhook_signing_key)),
            *settings.portal.required_settings.items(),
            *zip(
                R2_SETTING_NAMES,
                (
                    bool(storage.account_id),
                    bool(storage.bucket),
                    bool(storage.access_key_id),
                    bool(storage.secret_access_key),
                ),
                strict=True,
            ),
        )
        if present
    ]
    if held:
        raise ProductionConfigurationError(
            f"demo mode sends nothing, so these must be unset: {', '.join(held)}"
        )
    _require_complete_stripe_checkout(settings.stripe)
    secret_key = settings.stripe.secret_key
    if secret_key and not secret_key.startswith(STRIPE_TEST_KEY_PREFIXES):
        raise ProductionConfigurationError(
            "demo mode only accepts a Stripe test-mode key in GVAS_STRIPE_SECRET_KEY"
        )
    if settings.stripe.is_configured and settings.stripe.deployment == DEFAULT_DEPLOYMENT:
        # The demo shares production's Stripe test account; untagged or
        # production-tagged events are production's, so the demo needs its own name.
        raise ProductionConfigurationError(
            "demo mode with Stripe needs its own GVAS_STRIPE_DEPLOYMENT (e.g. demo)"
        )


def _require_complete_object_storage(storage: ObjectStorageSettings) -> None:
    """Object storage is optional, but half of it is a misconfiguration.

    With no ``GVAS_R2_*`` set the DOCX is delivered to the channel only; with
    all of them set it is also kept in the bucket. Some-but-not-all means the
    operator intended durability and would silently not get it.
    """

    present = (
        bool(storage.account_id),
        bool(storage.bucket),
        bool(storage.access_key_id),
        bool(storage.secret_access_key),
    )
    if any(present) and not all(present):
        missing = [name for name, ok in zip(R2_SETTING_NAMES, present, strict=True) if not ok]
        raise ProductionConfigurationError(
            f"object storage is partially configured; missing: {', '.join(missing)}"
        )


def _require_complete_calendly_lookup(settings: CalendlySettings) -> None:
    """Calendly is optional as a set: with neither variable set a quote must
    carry ``customer:``; with only one set the deployment must not start."""

    if settings.is_partially_configured:
        missing = [name for name, present in settings.required_settings.items() if not present]
        raise ProductionConfigurationError(
            f"calendly lookup is partially configured; missing: {', '.join(missing)}"
        )
    if not settings.is_configured:
        if settings.webhook_signing_key:
            raise ProductionConfigurationError(
                "GVAS_CALENDLY_WEBHOOK_SIGNING_KEY requires "
                "GVAS_CALENDLY_TOKEN and GVAS_CALENDLY_INSTALLATIONS"
            )
        return
    try:
        parse_calendly_installations(settings.installations)
    except CalendlyInstallationError as error:
        raise ProductionConfigurationError(f"GVAS_CALENDLY_INSTALLATIONS: {error}") from error


def _require_complete_portal_handoff(settings: PortalSettings) -> None:
    """The portal is optional as a set: with neither variable set approved
    quotes are emailed; with only one set the deployment must not start."""

    if settings.is_partially_configured:
        missing = [name for name, present in settings.required_settings.items() if not present]
        raise ProductionConfigurationError(
            f"portal handoff is partially configured; missing: {', '.join(missing)}"
        )


def _require_complete_stripe_checkout(settings: StripeSettings) -> None:
    """Card checkout is optional as a set: with neither variable set the quote
    accept route answers 503; with only one set the deployment must not start."""

    if settings.is_partially_configured:
        missing = [name for name, present in settings.required_settings.items() if not present]
        raise ProductionConfigurationError(
            f"card checkout is partially configured; missing: {', '.join(missing)}"
        )


def _require_complete_telnyx_channel(settings: TelnyxSettings) -> None:
    """Telnyx is optional as a set: all of it or none of it.

    A deployment that set the webhook key but not the API key would ingest
    texts and then fail every reply in the worker, so it must not start.
    """

    if settings.is_partially_configured:
        missing = [name for name, present in settings.required_settings.items() if not present]
        raise ProductionConfigurationError(
            f"telnyx channel is partially configured; missing: {', '.join(missing)}"
        )
    if not settings.is_configured:
        return
    try:
        installations = parse_telnyx_installations(settings.installations)
    except TelnyxInstallationError as error:
        raise ProductionConfigurationError(f"GVAS_TELNYX_INSTALLATIONS: {error}") from error
    if len(installations) != 1 or len(installations[0].owner_numbers) != 1:
        raise ProductionConfigurationError(
            "GVAS_TELNYX_INSTALLATIONS must configure exactly one number "
            "with exactly one owner number"
        )


def _require_managed_database(url: str) -> None:
    try:
        require_managed_postgres_url(url)
    except DatabaseUrlError as error:
        raise ProductionConfigurationError(f"GVAS_DATABASE_URL or DATABASE_URL: {error}") from error


def _require_single_owner(value: str) -> None:
    """The accepted pilot boundary: one ProTech workspace, one owner user.

    The parser stays general so later tenants need no new code, but a
    deployment that authorized a second workspace or a second owner would go
    past what this pilot was approved for, so it must not start.
    """

    try:
        installations = parse_slack_installations(value)
    except SlackInstallationError as error:
        raise ProductionConfigurationError(f"GVAS_SLACK_INSTALLATIONS: {error}") from error
    if len(installations) != 1 or len(installations[0].owner_user_ids) != 1:
        raise ProductionConfigurationError(
            "GVAS_SLACK_INSTALLATIONS must configure exactly one installation "
            "with exactly one owner user"
        )


def worker_identity(prefix: str) -> str:
    """Each replica claims outbox rows under its own name.

    Replicas that shared one identity would steal each other's leases, so the
    hostname the platform assigns is appended.
    """

    return f"{prefix}-{os.uname().nodename}-{os.getpid()}"


@dataclass(frozen=True)
class ProductionRuntime:
    settings: ProductionSettings
    application: Application
    app: FastAPI
    http_client: httpx.AsyncClient
    engine: AsyncEngine
    sandboxes: "DemoSandboxes | None" = None

    async def aclose(self) -> None:
        await self.http_client.aclose()
        await self.engine.dispose()


def _quote_drafting(
    settings: ProductionSettings, client: httpx.AsyncClient, usage_ledger: SqlUsageLedger
) -> QuoteDraftingPort:
    quote_drafting: QuoteDraftingPort = DeterministicQuoteDrafter()
    if settings.openai.is_configured:
        return ModelAssistedQuoteDrafter(
            quote_drafting,
            OpenAIFreeTextQuoteDrafter(settings.openai, client, usage_ledger=usage_ledger),
            ceilings=UsageCeilingGuard(usage_ledger, settings.usage_ceilings()),
        )
    logger.warning("openai not configured; quotes accept the structured format only")
    return quote_drafting


def build_demo_ports(
    settings: ProductionSettings,
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> ApplicationPorts:
    """The same workflows with nothing sent: see ``gvas.infrastructure.demo``.

    Only the model behind Gus and, when a test key is set, Stripe test mode
    are real. Startup (``_require_demo_isolation``) has already refused every
    other provider credential.
    """

    logger.warning("demo mode: e-mail, texts and Slack are logged, not sent")
    usage_ledger = SqlUsageLedger(session_factory)
    attachments = NoAttachments()
    customer_email = LoggedCustomerEmail(settings.resend.portal_url)
    owner_channel = LoggedOwnerReply()
    availability = DemoAvailability(settings.demo, session_factory)
    payment_checkout = (
        StripeCheckout(settings.stripe, client) if settings.stripe.is_configured else None
    )
    return ApplicationPorts(
        owner_replies=ChannelOwnerReplyRouter(
            session_factory,
            {SLACK_SOURCE_NAMESPACE: owner_channel, TELNYX_SOURCE_NAMESPACE: owner_channel},
        ),
        quote_drafting=_quote_drafting(settings, client, usage_ledger),
        availability=availability,
        booked_events=DemoBookedEvents(session_factory),
        intake_agent=OpenAIIntakeAgent(settings.openai, client, usage_ledger=usage_ledger),
        owner_assistant=OpenAIOwnerAssistant(settings.openai, client, usage_ledger=usage_ledger),
        customer_email=customer_email,
        owner_email=LoggedOwnerEmail(),
        quote_delivery=SiteAwareQuoteDelivery(customer_email),
        customer_text=LoggedCustomerText(),
        payment_checkout=payment_checkout,
        billing_accounts=payment_checkout,
        portal_login_email=LoggedPortalLoginEmail(),
        report_email=LoggedReportEmail(),
        transcription=OpenAITranscriber(
            settings.openai, client, attachments, usage_ledger=usage_ledger
        ),
        completeness_review=GuardedCompletenessReviewer(
            MarkerCompletenessReviewer(),
            OpenAIContradictionGuard(settings.openai, client, usage_ledger=usage_ledger),
        ),
        checklist_evidence=GuardedChecklistEvidenceAttributor(
            MarkerChecklistEvidenceAttributor(),
            OpenAIChecklistEvidenceAnnotator(settings.openai, client),
        ),
        report_generation=DeterministicReportGenerator(),
        source_attachments=attachments,
        usage_ledger=usage_ledger,
    )


def build_production_ports(
    settings: ProductionSettings,
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> ApplicationPorts:
    if settings.demo.mode:
        return build_demo_ports(settings, client, session_factory)
    poster = SlackWebApiChatPoster(settings.slack, client)
    attachments = SlackFileAttachmentAccess(settings.slack, client)
    usage_ledger = SqlUsageLedger(session_factory)
    report_artifacts = ReportArtifactAccess(
        DocxReportRenderer(), SqlReportUnitOfWorkFactory(session_factory)
    )
    object_storage = R2ObjectStorage(settings.storage) if settings.storage.is_configured else None
    if object_storage is None:
        logger.warning("object storage not configured; published reports live in Slack only")
    ledger = SqlChannelDeliveryLedger(session_factory)
    owner_replies: dict[str, OwnerReplyPort] = {
        SLACK_SOURCE_NAMESPACE: build_slack_owner_reply_adapter(
            poster,
            session_factory,
            ledger,
            uploader=SlackWebApiFileUploader(settings.slack, client),
            attachments=report_artifacts,
        )
    }
    channel_policies: tuple[ChannelWorkflowPolicy, ...] = ()
    customer_text: CustomerTextDeliveryPort | None = None
    if settings.telnyx.is_configured:
        telnyx_sender = TelnyxMessagingApiSender(settings.telnyx, client)
        owner_replies[TELNYX_SOURCE_NAMESPACE] = build_telnyx_owner_reply_adapter(
            telnyx_sender,
            session_factory,
            ledger,
            messaging_profile_id=settings.telnyx.messaging_profile_id or None,
        )
        channel_policies = (
            sms_quotes_only_policy("Slack", owner_assistant=settings.openai.is_configured),
        )
        customer_text = TelnyxCustomerTextAdapter(
            telnyx_sender,
            parse_telnyx_installations(settings.telnyx.installations),
            ledger,
            messaging_profile_id=settings.telnyx.messaging_profile_id or None,
        )
    resend_quotes = ResendQuoteDeliveryAdapter(settings.resend, client)
    owner_email = ResendOwnerEmailAdapter(settings.resend, client)
    portal_quotes: CustomerQuoteDeliveryPort | None = None
    if settings.portal.is_configured:
        portal_quotes = PortalQuoteDelivery(
            settings.portal, client, SqlPortalHandoffLedger(session_factory)
        )
    # A hosted quote (business site_url set) always goes to the email adapter so
    # its own link is rendered; otherwise the external portal, when configured,
    # takes the handoff, else the generic email link is sent.
    quote_delivery: CustomerQuoteDeliveryPort = SiteAwareQuoteDelivery(resend_quotes, portal_quotes)
    if portal_quotes is not None and customer_text is None:
        logger.warning("portal configured without telnyx; quote links are emailed only")
    payment_checkout = (
        StripeCheckout(settings.stripe, client) if settings.stripe.is_configured else None
    )
    appointment_lookup = (
        CalendlyAppointmentLookup(settings.calendly, client)
        if settings.calendly.is_configured
        else None
    )
    availability = (
        CalendlyAvailability(settings.calendly, client) if settings.calendly.is_configured else None
    )
    booked_events = (
        CalendlyBookedEvents(settings.calendly, client) if settings.calendly.is_configured else None
    )
    intake_agent = (
        OpenAIIntakeAgent(settings.openai, client, usage_ledger=usage_ledger)
        if settings.openai.is_configured
        else None
    )
    if intake_agent is None:
        logger.warning("openai not configured; the website booking chat is off")
    owner_assistant = (
        OpenAIOwnerAssistant(settings.openai, client, usage_ledger=usage_ledger)
        if settings.openai.is_configured
        else None
    )
    quote_drafting = _quote_drafting(settings, client, usage_ledger)
    return ApplicationPorts(
        owner_replies=ChannelOwnerReplyRouter(session_factory, owner_replies),
        quote_drafting=quote_drafting,
        appointment_lookup=appointment_lookup,
        availability=availability,
        booked_events=booked_events,
        schedule_blocks=availability,
        calendar_feed=IcsCalendarFeed(client),
        intake_agent=intake_agent,
        owner_assistant=owner_assistant,
        customer_email=resend_quotes,
        owner_email=owner_email,
        quote_delivery=quote_delivery,
        customer_text=customer_text,
        payment_checkout=payment_checkout,
        billing_accounts=payment_checkout,
        portal_login_email=ResendPortalLoginEmailAdapter(settings.resend, client),
        report_email=ResendReportEmailAdapter(settings.resend, client),
        transcription=OpenAITranscriber(
            settings.openai, client, attachments, usage_ledger=usage_ledger
        ),
        completeness_review=GuardedCompletenessReviewer(
            MarkerCompletenessReviewer(),
            OpenAIContradictionGuard(settings.openai, client, usage_ledger=usage_ledger),
        ),
        checklist_evidence=GuardedChecklistEvidenceAttributor(
            MarkerChecklistEvidenceAttributor(),
            OpenAIChecklistEvidenceAnnotator(settings.openai, client),
        ),
        report_generation=DeterministicReportGenerator(),
        source_attachments=attachments,
        object_storage=object_storage,
        channel_policies=channel_policies,
        usage_ledger=usage_ledger,
    )


def build_production_runtime(settings: ProductionSettings | None = None) -> ProductionRuntime:
    resolved = settings or load_production_settings()
    engine = create_engine(resolved.app.database_url)
    session_factory = create_session_factory(engine)
    # Redirects are refused so a provider cannot move an authenticated request.
    client = httpx.AsyncClient(follow_redirects=False)
    sandboxes = None
    if resolved.demo.mode and resolved.demo.sandbox_template_slug:
        # Imported here: sandboxes reuse the demo seed, which imports this module.
        from gvas.interfaces.demo_sandboxes import DemoSandboxes

        sandboxes = DemoSandboxes(resolved.demo, session_factory)
    application = build_application(
        build_production_ports(resolved, client, session_factory),
        resolved.app,
        session_factory=session_factory,
        lease_ttl=timedelta(seconds=resolved.worker.lease_seconds),
        ceilings=resolved.usage_ceilings(),
        intake_settings=resolved.intake,
        payment_deployment=resolved.stripe.deployment,
        intake_message_budget=None if sandboxes is None else sandboxes.refusal,
        intake_visitor=None if sandboxes is None else sandboxes.visitor,
        payments_off=None if sandboxes is None else sandboxes.is_sandbox,
    )
    on_activity = None if sandboxes is None else sandboxes.touch
    # A demo has no Slack workspace, so it mounts no Slack Request URL.
    routers = (
        []
        if resolved.demo.mode
        else [build_slack_event_router(application.ingest_service, resolved.slack)]
    )
    if resolved.telnyx.is_configured:
        routers.append(build_telnyx_webhook_router(application.ingest_service, resolved.telnyx))
    if resolved.calendly.webhook_signing_key:
        routers.append(
            build_calendly_webhook_router(application.intake_booking_events, resolved.calendly)
        )

    async def cors_origins() -> frozenset[str]:
        async with session_factory() as session:
            site_urls = await SqlBusinessRepository(session).list_site_urls()
        return frozenset(site_urls) | resolved.public_api.extra_origins()

    routers.append(
        create_public_router(
            application.public_quotes,
            webhook_verifier=(
                StripeWebhookVerifier(resolved.stripe.webhook_secret)
                if resolved.stripe.is_configured
                else None
            ),
            rate_limiter=PerIpRateLimiter(resolved.public_api.rate_limit_per_minute),
            intake=application.intake,
            decision_links=application.intake_decision_links,
            on_activity=on_activity,
        )
    )
    routers.append(
        create_portal_router(
            application.portal,
            rate_limiter=PerIpRateLimiter(resolved.public_api.rate_limit_per_minute),
            intake=application.intake,
            owner=application.owner,
        )
    )
    routers.append(
        create_owner_router(
            application.owner,
            rate_limiter=PerIpRateLimiter(resolved.public_api.rate_limit_per_minute),
            on_activity=on_activity,
        )
    )
    if sandboxes is not None:
        from gvas.interfaces.http.sandbox import create_sandbox_router

        routers.append(
            create_sandbox_router(
                sandboxes,
                per_ip_per_hour=resolved.demo.sandbox_per_ip_per_hour,
                rate_limiter=PerIpRateLimiter(resolved.public_api.rate_limit_per_minute),
                owner_assistant=application.owner_assistant,
                owner_texts_per_sandbox=resolved.demo.sandbox_messages_per_visitor,
            )
        )
    return ProductionRuntime(
        settings=resolved,
        application=application,
        app=create_app(resolved.app, tuple(routers), cors_origins=cors_origins),
        http_client=client,
        engine=engine,
        sandboxes=sandboxes,
    )


def create_production_app() -> FastAPI:
    """Uvicorn target for the web service; mounts the Slack Request URL and, when
    configured, the Telnyx messaging webhook."""

    runtime = build_production_runtime()
    configure_logging(runtime.settings.app.log_level)
    runtime.app.add_event_handler("shutdown", runtime.aclose)
    return runtime.app

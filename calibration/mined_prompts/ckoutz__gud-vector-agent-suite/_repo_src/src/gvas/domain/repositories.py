from datetime import datetime
from enum import StrEnum
from ipaddress import ip_address
from typing import Protocol
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gvas.domain.calendar_blocks import CalendarBlockRepository
from gvas.domain.customers import (
    CustomerRepository,
    PortalLoginTokenRepository,
    PortalSessionRepository,
    ServiceRequestRepository,
)
from gvas.domain.enums import DeliveryStatus, WorkflowRunStatus
from gvas.domain.identifiers import (
    BusinessId,
    ConversationId,
    EndpointId,
    MessageId,
    MessageKey,
    RoutingData,
    WorkflowIntent,
    WorkflowRunId,
)
from gvas.domain.intake import (
    IntakeConversationRepository,
    IntakeMessageRepository,
    IntakeProfile,
)
from gvas.domain.invoices import InvoiceRepository
from gvas.domain.messages import (
    ChannelEndpointRef,
    ConversationRef,
    DeliveryReceipt,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
)
from gvas.domain.outbox import OutboxCommand, OutboxRecord
from gvas.domain.owner import OwnerLoginTokenRepository, OwnerSessionRepository
from gvas.domain.payments import (
    PaymentEventRepository,
    PaymentLedgerRepository,
    QuotePaymentRepository,
    QuoteSubscriptionRepository,
)
from gvas.domain.quotes import QuoteRepository


def normalize_site_url(value: str) -> str:
    """One spelling for a business's public origin so links and CORS agree.

    Strictly an origin: scheme + host (optional port), no path, query,
    fragment or credentials — anything more would break both the
    ``<site>/q/<token>`` link shape and CORS origin matching. Claim tokens
    ride in those links, so a real site must use ``https``; plain ``http``
    only passes for a local development host.
    """

    try:
        parts = urlsplit(value.strip())
    except ValueError as error:
        raise ValueError("site url is not a parseable URL") from error
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("site url must be an absolute http(s) origin")
    try:
        host = parts.hostname
        parts.port  # noqa: B018 - property access raises on a malformed port
    except ValueError as error:
        raise ValueError("site url has a malformed host or port") from error
    if not host or parts.username or parts.password:
        raise ValueError("site url must be a bare host with no credentials")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("site url must be an origin: no path, query or fragment")
    if parts.scheme.lower() == "http" and not is_local_host(host):
        raise ValueError("site url must use https outside local development")
    netloc = parts.netloc.lower()
    return f"{parts.scheme.lower()}://{netloc}"


def is_local_host(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


class BusinessRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    business_id: BusinessId
    slug: str
    name: str
    # Hosted-quote config: the public site the quote page runs on, the name
    # customers read, a booking link, a publishable (non-secret) key, and the
    # business's future connected-account id (storage only for now).
    site_url: str | None = None
    display_name: str | None = None
    calendly_url: str | None = None
    stripe_account_id: str | None = None
    public_key: str | None = None
    # What the website booking agent says the business is, asks and opens
    # with; empty means the generic default.
    intake_profile: IntakeProfile = IntakeProfile()
    # Owner inbox that gets a copy of every owner notice raised by the
    # website (booking requests, escalations, service requests, payments).
    notification_email: str | None = None
    # The address that signs in to the owner dashboard (set by the operator).
    owner_email: str | None = None
    # The owner's private calendar subscription link: a credential, read only
    # by the feed adapter and never returned by any route.
    calendar_feed_url: str | None = Field(default=None, repr=False)
    # The IANA zone the business works in (``America/Los_Angeles``): filled
    # from Calendly, editable by the owner, and used for every time label.
    timezone: str | None = None
    # Invoices are on only where this is set; numbers read ``LL-1042``.
    invoice_prefix: str | None = None

    @field_validator("site_url")
    @classmethod
    def site_url_is_an_origin(cls, value: str | None) -> str | None:
        return None if value is None else normalize_site_url(value)


class OwnerChannelEndpointRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint_id: EndpointId
    business_id: BusinessId
    source_namespace: str
    external_endpoint_id: str
    owner_external_id: str | None
    routing: RoutingData


class OutboundDeliveryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outbound_message_id: MessageId
    message: OutboundOwnerMessage
    endpoint_id: EndpointId
    conversation_routing: RoutingData
    endpoint_routing: RoutingData
    status: DeliveryStatus


class WorkflowClaimResult(StrEnum):
    ACQUIRED = "acquired"
    TERMINAL = "terminal"
    BUSY = "busy"


class InboundProcessingRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inbound_message_id: MessageId
    business_id: BusinessId
    conversation_id: ConversationId
    endpoint_id: EndpointId
    message: NormalizedOwnerMessage


class WorkflowRunClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result: WorkflowClaimResult
    run_id: WorkflowRunId
    status: WorkflowRunStatus
    intent: WorkflowIntent | None
    attempts: int
    lease_token: UUID | None = None

    @model_validator(mode="after")
    def validate_lease_token(self) -> "WorkflowRunClaim":
        if (self.result is WorkflowClaimResult.ACQUIRED) != (self.lease_token is not None):
            raise ValueError("only acquired workflow claims may carry a lease token")
        return self


class LostWorkflowLeaseError(ValueError):
    pass


class EndpointBusinessMismatchError(ValueError):
    pass


class CrossBusinessReferenceError(ValueError):
    pass


class BusinessRepository(Protocol):
    async def get(self, business_id: BusinessId) -> BusinessRecord | None: ...

    async def lock(self, business_id: BusinessId) -> BusinessRecord | None:
        """Row-level lock: callers that guard on per-business state (daily
        caps, quotas) hold this for the length of their transaction so
        concurrent writers serialize."""
        ...

    async def ensure(
        self, business_id: BusinessId, slug: str, name: str, *, now: datetime
    ) -> BusinessRecord: ...

    async def get_by_public_key(self, public_key: str) -> BusinessRecord | None: ...

    async def list_site_urls(self) -> tuple[str, ...]:
        """Every configured ``site_url``; feeds the public API's CORS set."""
        ...

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
        """Set hosted-quote and intake-profile fields; ``None`` arguments
        leave stored values and an empty ``calendar_feed_url`` clears it."""
        ...

    async def adopt_timezone(self, business_id: BusinessId, timezone: str, now: datetime) -> bool:
        """Set the zone only while it is still unset, in one conditional
        write, so an owner's own choice is never overwritten."""
        ...


class OwnerChannelEndpointRepository(Protocol):
    async def get(self, endpoint_id: EndpointId) -> OwnerChannelEndpointRecord | None: ...

    async def get_or_create(
        self, reference: ChannelEndpointRef, routing: RoutingData
    ) -> EndpointId: ...


class ConversationRepository(Protocol):
    """Endpoint and conversation references must belong to the same business."""

    async def get_or_create(
        self, reference: ConversationRef, endpoint_id: EndpointId, routing: RoutingData
    ) -> ConversationId: ...

    async def find_endpoint(self, reference: ConversationRef) -> ChannelEndpointRef | None: ...


class InboundMessageRepository(Protocol):
    """Inbound links must reference the message business and endpoint."""

    async def create(
        self, message: InboundOwnerMessage, conversation_id: ConversationId, endpoint_id: EndpointId
    ) -> MessageId | None: ...

    async def get_for_processing(
        self, inbound_message_id: MessageId
    ) -> InboundProcessingRecord | None: ...

    async def find_by_key(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        message_key: MessageKey,
    ) -> InboundProcessingRecord | None: ...

    async def find_latest_for_business(
        self, business_id: BusinessId
    ) -> InboundProcessingRecord | None:
        """The business's newest inbound message, whichever channel it rode."""
        ...


class OutboundMessageRepository(Protocol):
    """Outbound links must reference the same business as the reply."""

    async def create(
        self,
        message: OutboundOwnerMessage,
        conversation_id: ConversationId,
        inbound_message_id: MessageId,
    ) -> MessageId: ...

    async def get_for_delivery(
        self, outbound_message_id: MessageId
    ) -> OutboundDeliveryRecord | None: ...

    async def find_by_correlation(
        self, business_id: BusinessId, conversation_id: ConversationId, correlation_id: str
    ) -> MessageId | None: ...

    async def record_delivery(
        self, outbound_message_id: MessageId, receipt: DeliveryReceipt
    ) -> None: ...


class WorkflowRunRepository(Protocol):
    async def claim(
        self,
        business_id: BusinessId,
        inbound_message_id: MessageId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> WorkflowRunClaim: ...

    async def set_intent(self, claim: WorkflowRunClaim, intent: WorkflowIntent) -> None: ...

    async def set_error(self, claim: WorkflowRunClaim, error: str) -> None: ...

    async def finish(
        self,
        claim: WorkflowRunClaim,
        status: WorkflowRunStatus,
        error: str | None = None,
    ) -> None: ...


class OutboxRepository(Protocol):
    async def enqueue(self, command: OutboxCommand) -> None: ...
    async def claim_batch(
        self, limit: int, now: datetime, claimed_by: str, *, stale_before: datetime
    ) -> list[OutboxRecord]: ...
    async def update(self, record: OutboxRecord) -> None: ...


class UnitOfWork(Protocol):
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

    async def __aenter__(self) -> "UnitOfWork": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...

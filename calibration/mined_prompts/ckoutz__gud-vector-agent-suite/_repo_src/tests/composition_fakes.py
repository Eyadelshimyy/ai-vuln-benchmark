from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.composition import ApplicationPorts
from gvas.domain.enums import DeliveryStatus, HostedLinkKind, RecipientAddressKind
from gvas.domain.identifiers import JsonValue
from gvas.domain.messages import (
    AudioReference,
    ConversationRef,
    CustomerDeliveryRequest,
    CustomerRecipient,
    DeliveryReceipt,
    OutboundOwnerMessage,
    TranscriptResult,
)
from gvas.domain.quotes import (
    HostedLinkReference,
    QuoteDraftProposal,
    QuoteDraftRequest,
    QuoteLineItem,
)
from gvas.domain.reporting import ReportEmailRequest, ReportGenerationRequest

FAKE_NOW = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)


class OwnerReplyFake:
    """Records owner replies instead of calling a chat transport."""

    def __init__(self) -> None:
        self.sent: list[tuple[ConversationRef, OutboundOwnerMessage]] = []

    async def send(
        self, conversation_ref: ConversationRef, message: OutboundOwnerMessage
    ) -> DeliveryReceipt:
        self.sent.append((conversation_ref, message))
        return DeliveryReceipt(
            status=DeliveryStatus.DELIVERED,
            provider_message_id=f"owner-reply-{len(self.sent)}",
            occurred_at=FAKE_NOW,
        )


class QuoteDraftingFake:
    def __init__(self) -> None:
        self.requests: list[QuoteDraftRequest] = []

    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal:
        self.requests.append(request)
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
                    description=request.request_text,
                    quantity=1,
                    unit_price_minor=25_000,
                ),
            ),
            hosted_links=(
                HostedLinkReference(kind=HostedLinkKind.PAYMENT, reference="hosted-link"),
            ),
            confidence=1,
        )


class CustomerDeliveryFake:
    def __init__(self) -> None:
        self.requests: list[CustomerDeliveryRequest] = []

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        self.requests.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.DELIVERED,
            provider_message_id=f"customer-delivery-{len(self.requests)}",
            occurred_at=FAKE_NOW,
        )


class ReportEmailFake:
    def __init__(self, fail_times: int = 0) -> None:
        self.requests: list[ReportEmailRequest] = []
        self.fail_times = fail_times

    async def deliver(self, request: ReportEmailRequest) -> DeliveryReceipt:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("resend returned http 500 api_key=re_secret")
        self.requests.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.ACCEPTED,
            provider_message_id=f"report-email-{len(self.requests)}",
            occurred_at=FAKE_NOW,
        )


class TranscriptionFake:
    """Returns configured transcripts and observes committed state while called.

    ``observer`` runs on a separate session, so what it sees proves whether the
    claim transaction was committed before the provider call.
    """

    def __init__(
        self,
        transcripts: dict[str, str],
        observer: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._transcripts = transcripts
        self._observer = observer
        self.calls: list[str] = []
        self.observed_states: list[str | None] = []

    async def transcribe(self, audio: AudioReference) -> TranscriptResult:
        self.calls.append(audio.attachment.locator)
        if self._observer is not None:
            self.observed_states.append(await self._observer())
        return TranscriptResult(text=self._transcripts[audio.attachment.locator])


class ReportGenerationFake:
    def __init__(self) -> None:
        self.requests: list[ReportGenerationRequest] = []

    async def generate(self, request: ReportGenerationRequest) -> dict[str, JsonValue]:
        """Renders the structure of the pinned template, as a real generator must."""
        self.requests.append(request)
        return {
            "schema_version": "field-notes-report/v1",
            "title": request.report_template.title,
            "sections": [
                {
                    "section_key": section.section_key,
                    "heading": section.heading,
                    "blocks": [
                        {
                            "kind": "text",
                            "text": request.source.canonical_transcript,
                            "evidence_refs": [{"source": "transcript", "key": "canonical"}],
                        }
                    ],
                }
                for section in request.report_template.sections
            ],
        }


def application_ports(
    *,
    owner_replies: OwnerReplyFake,
    quote_drafting: QuoteDraftingFake,
    quote_delivery: CustomerDeliveryFake,
    transcription: TranscriptionFake,
    report_generation: ReportGenerationFake,
    report_email: ReportEmailFake | None = None,
) -> ApplicationPorts:
    return ApplicationPorts(
        owner_replies=owner_replies,
        quote_drafting=quote_drafting,
        quote_delivery=quote_delivery,
        transcription=transcription,
        completeness_review=MarkerCompletenessReviewer(),
        checklist_evidence=MarkerChecklistEvidenceAttributor(),
        report_generation=report_generation,
        report_email=report_email,
    )

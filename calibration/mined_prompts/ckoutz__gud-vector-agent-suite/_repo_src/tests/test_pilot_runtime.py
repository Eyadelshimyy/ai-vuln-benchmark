"""Deterministic pilot runtime: report delivery, failure notices, bootstrap."""

from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    FAKE_NOW,
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    TranscriptionFake,
)
from gvas.application.checklist_evidence import MarkerChecklistEvidenceAttributor
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.deterministic_report import DeterministicReportGenerator
from gvas.composition import Application, ApplicationPorts, build_application
from gvas.composition.dispatcher import OutboxWorker
from gvas.domain.enums import DeliveryStatus, OutboxStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import (
    AudioReference,
    CustomerDeliveryRequest,
    DeliveryReceipt,
    TextPart,
    TranscriptResult,
)
from gvas.domain.ports import CustomerQuoteDeliveryPort, TranscriptionPort
from gvas.domain.templates import IndustryKey
from gvas.infrastructure.models import Business, OutboxMessage
from gvas.interfaces.bootstrap import BootstrapRequest, run_bootstrap
from test_composition import (
    Clock,
    audio_part,
    configure_checklist,
    inbound,
    seed_business,
)


class FailingTranscription:
    """Stands in for a provider outage that outlives every retry."""

    async def transcribe(self, audio: AudioReference) -> TranscriptResult:
        raise RuntimeError("provider token 'xoxb-secret' rejected")


class RecoveringTranscription:
    """Fails the recordings named in ``failing`` and transcribes the rest."""

    def __init__(self, failing: frozenset[str], text: str) -> None:
        self._failing = failing
        self._text = text

    async def transcribe(self, audio: AudioReference) -> TranscriptResult:
        if audio.attachment.locator in self._failing:
            raise RuntimeError("provider token 'xoxb-secret' rejected")
        return TranscriptResult(text=self._text)


class RecoveringCustomerDelivery:
    """Fails the first quote it is asked to send and delivers later ones."""

    def __init__(self) -> None:
        self.delivered: list[CustomerDeliveryRequest] = []
        self._doomed: str | None = None

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        if self._doomed is None:
            self._doomed = request.idempotency_key
        if request.idempotency_key == self._doomed:
            raise RuntimeError("provider key 're-secret' rejected")
        self.delivered.append(request)
        return DeliveryReceipt(
            status=DeliveryStatus.DELIVERED,
            provider_message_id=f"customer-delivery-{len(self.delivered)}",
            occurred_at=FAKE_NOW,
        )


def deterministic_ports(
    owner_replies: OwnerReplyFake,
    transcription: TranscriptionPort,
    quote_delivery: CustomerQuoteDeliveryPort,
) -> ApplicationPorts:
    """The pilot wiring: deterministic review and reporting, no inference model."""

    return ApplicationPorts(
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=quote_delivery,
        transcription=transcription,
        completeness_review=MarkerCompletenessReviewer(),
        checklist_evidence=MarkerChecklistEvidenceAttributor(),
        report_generation=DeterministicReportGenerator(),
    )


def build(
    session_factory: async_sessionmaker[AsyncSession],
    owner_replies: OwnerReplyFake,
    transcription: TranscriptionPort,
    quote_delivery: CustomerQuoteDeliveryPort | None = None,
) -> Application:
    return build_application(
        deterministic_ports(owner_replies, transcription, quote_delivery or CustomerDeliveryFake()),
        session_factory=session_factory,
        now=Clock(),
    )


def immediate_worker(application: Application) -> OutboxWorker:
    """Retries without delay so a test can reach the terminal state."""

    return OutboxWorker(
        application.outbox,
        application.dispatcher,
        now=Clock(),
        retry_in=timedelta(seconds=0),
        failure_notices=application.failure_notice_service,
    )


def texts_of(owner_replies: OwnerReplyFake, prefix: str) -> list[str]:
    return [
        part.text
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, TextPart) and part.text.startswith(prefix)
    ]


@pytest.mark.asyncio
async def test_completed_report_is_delivered_to_the_case_thread_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(session_factory, owner_replies, TranscriptionFake({}))
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north tower", message_key="notes-1")
    )
    await application.worker.drain()
    await application.ingest_service.ingest(
        inbound(business_id, "Replaced the gutter run", message_key="notes-1-answer")
    )
    await application.worker.drain()
    await application.worker.drain()

    reports = texts_of(owner_replies, "Field Notes Report")
    assert len(reports) == 1
    assert reports[0].splitlines()[:4] == [
        "Field Notes Report",
        "Report version 1",
        "",
        "Site and Work",
    ]
    assert "Which site was visited?" in reports[0]
    assert "observed" not in reports[0]
    assert "Replaced the gutter run" in reports[0]
    assert reports[0].count("Replaced the gutter run") == 1
    assert {ref.external_conversation_id for ref, _ in owner_replies.sent} == {"conversation"}


@pytest.mark.asyncio
async def test_dead_command_notifies_the_owner_once_without_leaking_provider_detail(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(session_factory, owner_replies, FailingTranscription())
    await configure_checklist(application, business_id)
    worker = immediate_worker(application)

    await application.ingest_service.ingest(
        inbound(
            business_id,
            "field notes: site visit",
            message_key="notes-audio",
            attachments=(audio_part("channel-file:F1"),),
        )
    )
    await worker.drain()
    await worker.drain()

    async with session_factory() as session:
        dead = await session.scalar(
            select(func.count())
            .select_from(OutboxMessage)
            .where(OutboxMessage.status == OutboxStatus.DEAD.value)
        )
    assert dead == 1

    notices = texts_of(owner_replies, "A voice note could not be transcribed.")
    assert len(notices) == 1
    assert "xoxb" not in notices[0]
    assert "close notes" in notices[0]
    assert "new thread" in notices[0]


@pytest.mark.asyncio
async def test_the_audio_recovery_guidance_actually_recovers_the_case(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Close the notes, start again in a new thread, re-upload: a report follows."""

    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(
        session_factory,
        owner_replies,
        RecoveringTranscription(
            frozenset({"channel-file:F1"}), "site: 123 Main. work: replaced the gutter run."
        ),
    )
    await configure_checklist(application, business_id)
    worker = immediate_worker(application)

    await application.ingest_service.ingest(
        inbound(
            business_id,
            "field notes: site visit",
            message_key="notes-audio",
            attachments=(audio_part("channel-file:F1"),),
        )
    )
    await worker.drain()
    await worker.drain()
    assert texts_of(owner_replies, "A voice note could not be transcribed.")

    await application.ingest_service.ingest(
        inbound(business_id, "close notes", message_key="notes-close")
    )
    await worker.drain()
    await application.ingest_service.ingest(
        inbound(
            business_id,
            "field notes: site visit",
            message_key="notes-audio-retry",
            conversation="conversation-2",
            attachments=(audio_part("channel-file:F2"),),
        )
    )
    for _ in range(4):
        await worker.drain()

    reports = texts_of(owner_replies, "Field Notes Report")
    assert len(reports) == 1
    assert "site: 123 Main." in reports[0]


@pytest.mark.asyncio
async def test_a_dead_quote_delivery_needs_the_new_quote_the_guidance_asks_for(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    delivery = RecoveringCustomerDelivery()
    application = build(session_factory, owner_replies, TranscriptionFake({}), delivery)
    worker = immediate_worker(application)

    async def quote(thread: str, key: str) -> None:
        await application.ingest_service.ingest(
            inbound(business_id, "quote: two gutters", message_key=key, conversation=thread)
        )
        await worker.drain()
        await application.ingest_service.ingest(
            inbound(business_id, "approve", message_key=f"{key}-approve", conversation=thread)
        )
        for _ in range(3):
            await worker.drain()

    await quote("conversation", "quote-1")
    notices = texts_of(owner_replies, "The approved quote could not be emailed")
    assert len(notices) == 1
    assert "re-secret" not in notices[0]
    assert "new quote in a new thread" in notices[0]
    assert delivery.delivered == []

    # The guidance says the thread is spent, so prove that it is.
    await application.ingest_service.ingest(
        inbound(business_id, "send it", message_key="quote-1-nudge")
    )
    await worker.drain()
    assert delivery.delivered == []

    await quote("conversation-2", "quote-2")
    assert len(delivery.delivered) == 1


@pytest.mark.asyncio
async def test_bootstrap_is_idempotent_and_updates_identity_in_place(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    request = BootstrapRequest(
        business_id=business_id,
        slug="protech",
        name="ProTech",
        industry=IndustryKey("environmental_testing"),
    )

    first = await run_bootstrap(request, session_factory)
    second = await run_bootstrap(request, session_factory)
    renamed = await run_bootstrap(
        BootstrapRequest(
            business_id=business_id,
            slug="protech",
            name="ProTech Environmental",
            industry=IndustryKey("environmental_testing"),
        ),
        session_factory,
    )

    assert first.template_set == second.template_set == renamed.template_set
    async with session_factory() as session:
        rows = list((await session.scalars(select(Business))).all())
    assert [(row.slug, row.name) for row in rows] == [("protech", "ProTech Environmental")]

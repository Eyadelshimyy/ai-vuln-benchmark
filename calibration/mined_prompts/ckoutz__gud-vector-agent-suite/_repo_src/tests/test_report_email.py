from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    ReportEmailFake,
    ReportGenerationFake,
    TranscriptionFake,
    application_ports,
)
from gvas.application.docx_report import render_report_docx
from gvas.application.report_email import (
    INVALID_ADDRESS_REPLY,
    NO_OPEN_CASE_REPLY,
    NOT_PUBLISHED_REPLY,
    report_email_queued_reply,
    report_email_sent_reply,
)
from gvas.application.unmatched_messages import UNMATCHED_MESSAGE_REPLY
from gvas.composition import Application, build_application
from gvas.composition.dispatcher import OutboxWorker
from gvas.composition.failure_notices import SEND_REPORT_AGAIN_IN_THREAD
from gvas.domain.enums import OutboxStatus
from gvas.domain.field_notes import (
    FieldNoteCaseStatus,
    match_field_note_report_send_trigger,
)
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import NormalizedOwnerMessage
from gvas.domain.outbox import DEFAULT_MAX_ATTEMPTS
from gvas.domain.reporting import (
    DOCX_MEDIA_TYPE,
    FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE,
    field_notes_report_email_command,
    normalize_email_address,
)
from gvas.infrastructure.object_storage import InMemoryObjectStorage
from test_composition import (
    Clock,
    case_rows,
    configure_checklist,
    drain,
    inbound,
    outbox_rows,
    reply_texts,
    report_versions,
    seed_business,
    unsucceeded_outbox,
)

RECIPIENT = "client@example.com"


def message(text: str) -> NormalizedOwnerMessage:
    return inbound(BusinessId(uuid4()), text, message_key=f"m-{uuid4()}").message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("send report to client@example.com", "client@example.com"),
        ("  Send Report To Client@Example.com  ", "Client@Example.com"),
        ("send report to", ""),
        ("send report to  a@b.co  c@d.co", "a@b.co  c@d.co"),
    ],
)
def test_send_trigger_returns_the_typed_recipient_text(text: str, expected: str) -> None:
    assert match_field_note_report_send_trigger(message(text)) == expected


@pytest.mark.parametrize("text", ["send report together", "please send report to a@b.co", "hi"])
def test_send_trigger_does_not_match_other_messages(text: str) -> None:
    assert match_field_note_report_send_trigger(message(text)) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Client@Example.com", "client@example.com"),
        (" first.last+tag@sub.example.co.uk ", "first.last+tag@sub.example.co.uk"),
        ("", None),
        ("client", None),
        ("client@localhost", None),
        ("client@example.com.", None),
        ("<client@example.com>", None),
        ("Client Name <client@example.com>", None),
        ("a@b.co, c@d.co", None),
        ("a@b.co c@d.co", None),
        ("a@b.co;c@d.co", None),
        ("a" * 250 + "@b.co", None),
    ],
)
def test_email_address_normalization_accepts_exactly_one_address(
    value: str, expected: str | None
) -> None:
    assert normalize_email_address(value) == expected


def test_email_command_identity_is_pinned_to_version_recipient_and_request() -> None:
    business_id = BusinessId(uuid4())
    case_id = uuid4()
    report_version_id = uuid4()

    first = field_notes_report_email_command(
        business_id, case_id, report_version_id, RECIPIENT, "k1"
    )
    again = field_notes_report_email_command(
        business_id, case_id, report_version_id, RECIPIENT, "k1"
    )
    retry = field_notes_report_email_command(
        business_id, case_id, report_version_id, RECIPIENT, "k2"
    )
    other_recipient = field_notes_report_email_command(
        business_id, case_id, report_version_id, "other@example.com", "k1"
    )
    other_version = field_notes_report_email_command(business_id, case_id, uuid4(), RECIPIENT, "k1")

    assert first == again
    assert first.command_type == FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE
    assert first.payload == {
        "field_note_case_id": str(case_id),
        "report_version_id": str(report_version_id),
        "recipient_address": RECIPIENT,
        "request_key": "k1",
    }
    assert first.dedup_key == f"field_notes_report_email:{report_version_id}:{RECIPIENT}:k1"
    commands = (first, retry, other_recipient, other_version)
    assert len({command.command_id for command in commands}) == 4
    assert len({command.dedup_key for command in commands}) == 4


def emailing_application(
    session_factory: async_sessionmaker[AsyncSession],
    owner_replies: OwnerReplyFake,
    report_email: ReportEmailFake,
) -> Application:
    ports = application_ports(
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
        report_email=report_email,
    )
    return build_application(
        replace(ports, object_storage=InMemoryObjectStorage()),
        session_factory=session_factory,
        now=Clock(),
    )


async def open_and_publish(
    application: Application,
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
) -> None:
    await configure_checklist(application, business_id)
    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-1")
    )
    await drain(application)


@pytest.mark.asyncio
async def test_send_report_emails_the_published_docx_once_and_confirms_in_thread(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    report_email = ReportEmailFake()
    application = emailing_application(session_factory, owner_replies, report_email)
    await open_and_publish(application, session_factory, business_id)
    versions = await report_versions(session_factory)
    assert len(versions) == 1

    request = inbound(business_id, "  Send Report To Client@Example.COM ", message_key="send-1")
    await application.ingest_service.ingest(request)
    await application.ingest_service.ingest(request)
    await drain(application)

    replies = reply_texts(owner_replies)
    assert replies.count(report_email_queued_reply(1, RECIPIENT)) == 1
    assert replies.count(report_email_sent_reply(1, RECIPIENT)) == 1
    emails = await outbox_rows(session_factory, FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE)
    assert len(emails) == 1
    assert emails[0].payload["report_version_id"] == str(versions[0].id)
    assert emails[0].payload["recipient_address"] == RECIPIENT
    assert await unsucceeded_outbox(session_factory) == 0

    assert len(report_email.requests) == 1
    sent = report_email.requests[0]
    assert sent.business_id == business_id
    assert sent.recipient_address == RECIPIENT
    assert sent.idempotency_key == emails[0].dedup_key
    assert "version 1" in sent.subject
    assert sent.artifact.media_type == DOCX_MEDIA_TYPE
    assert sent.artifact.filename.endswith("-v1.docx")
    async with application.report_unit_of_work_factory() as pinned:
        version = await pinned.reports.get_version(business_id, versions[0].id)
        await pinned.commit()
    assert version is not None
    assert sent.artifact.content == render_report_docx(version)

    assert [case.status for case in await case_rows(session_factory)] == [
        FieldNoteCaseStatus.OPEN.value
    ]

    await application.ingest_service.ingest(
        inbound(business_id, f"send report to {RECIPIENT}", message_key="send-2")
    )
    await drain(application)

    assert len(await outbox_rows(session_factory, FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE)) == 2
    assert len(report_email.requests) == 2
    assert reply_texts(owner_replies).count(report_email_sent_reply(1, RECIPIENT)) == 2


@pytest.mark.asyncio
async def test_dead_email_tells_the_owner_and_asking_again_recovers_it(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    report_email = ReportEmailFake(fail_times=DEFAULT_MAX_ATTEMPTS)
    application = emailing_application(session_factory, owner_replies, report_email)
    worker = OutboxWorker(
        application.outbox,
        application.dispatcher,
        now=Clock(),
        retry_in=timedelta(seconds=0),
        failure_notices=application.failure_notice_service,
    )
    await configure_checklist(application, business_id)
    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await worker.drain()
    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-1")
    )
    await worker.drain()

    await application.ingest_service.ingest(
        inbound(business_id, f"send report to {RECIPIENT}", message_key="send-1")
    )
    await worker.drain()
    await worker.drain()

    emails = await outbox_rows(session_factory, FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE)
    assert [row.status for row in emails] == [OutboxStatus.DEAD.value]
    assert report_email.fail_times == 0
    assert report_email.requests == []
    notices = [
        text
        for text in reply_texts(owner_replies)
        if text.startswith("The published report could not be emailed")
    ]
    assert len(notices) == 1
    assert SEND_REPORT_AGAIN_IN_THREAD in notices[0]
    assert "re_secret" not in notices[0]
    assert "http 500" not in notices[0]
    assert report_email_sent_reply(1, RECIPIENT) not in reply_texts(owner_replies)

    await application.ingest_service.ingest(
        inbound(business_id, f"send report to {RECIPIENT}", message_key="send-2")
    )
    await worker.drain()

    emails = await outbox_rows(session_factory, FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE)
    assert sorted(row.status for row in emails) == [
        OutboxStatus.DEAD.value,
        OutboxStatus.SUCCEEDED.value,
    ]
    assert len(report_email.requests) == 1
    assert reply_texts(owner_replies).count(report_email_sent_reply(1, RECIPIENT)) == 1


@pytest.mark.asyncio
async def test_send_report_before_publication_explains_approve_report_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    report_email = ReportEmailFake()
    application = emailing_application(session_factory, owner_replies, report_email)
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    assert len(await report_versions(session_factory)) == 1

    request = inbound(business_id, f"send report to {RECIPIENT}", message_key="send-early")
    await application.ingest_service.ingest(request)
    await application.ingest_service.ingest(request)
    await drain(application)

    assert reply_texts(owner_replies).count(NOT_PUBLISHED_REPLY) == 1
    assert "approve report" in NOT_PUBLISHED_REPLY
    assert await outbox_rows(session_factory, FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE) == []
    assert report_email.requests == []
    assert await unsucceeded_outbox(session_factory) == 0


@pytest.mark.asyncio
async def test_send_report_rejects_bad_addresses_and_missing_cases_without_sending(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    report_email = ReportEmailFake()
    application = emailing_application(session_factory, owner_replies, report_email)

    await application.ingest_service.ingest(
        inbound(business_id, f"send report to {RECIPIENT}", message_key="send-no-case")
    )
    await drain(application)
    assert reply_texts(owner_replies) == [NO_OPEN_CASE_REPLY]

    await open_and_publish(application, session_factory, business_id)
    before = len(reply_texts(owner_replies))
    for key, text in [
        ("bad-1", "send report to"),
        ("bad-2", "send report to a@b.co, c@d.co"),
        ("bad-3", "send report to Client <client@example.com>"),
    ]:
        await application.ingest_service.ingest(inbound(business_id, text, message_key=key))
    await drain(application)

    assert reply_texts(owner_replies)[before:] == [INVALID_ADDRESS_REPLY] * 3
    assert await outbox_rows(session_factory, FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE) == []
    assert report_email.requests == []
    assert await unsucceeded_outbox(session_factory) == 0


def test_unmatched_help_mentions_the_send_trigger() -> None:
    assert "`send report to <address>` emails the published report" in UNMATCHED_MESSAGE_REPLY

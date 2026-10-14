"""Plan sets uploaded into an open field-note case thread (audit follow-up #4)."""

from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    ReportGenerationFake,
    TranscriptionFake,
    application_ports,
)
from gvas.composition import Application, build_application
from gvas.composition.dispatcher import OutboxWorker
from gvas.composition.failure_notices import (
    FAILURE_GUIDANCE,
    UPLOAD_PLAN_SET_AGAIN_IN_THREAD,
    NotifyExhaustedCommandService,
)
from gvas.composition.field_note_workflow import PLAN_CUSTODY_NOT_ENABLED_REPLY
from gvas.domain.enums import MediaKind, OutboxStatus
from gvas.domain.field_notes import FieldNoteCaseId
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentPart, AttachmentReference
from gvas.domain.outbox import DEFAULT_MAX_ATTEMPTS, OutboxRecord
from gvas.domain.plans import (
    PLAN_SET_COPY_COMMAND_TYPE,
    PlanSetUploadId,
    PlanSetUploadStatus,
    field_note_case_site_id,
    plan_set_copy_command,
)
from gvas.infrastructure.object_storage import InMemoryObjectStorage, tenant_key_prefix
from gvas.infrastructure.plan_models import SitePlanSetUploadRow, SiteRow
from test_composition import (
    Clock,
    case_rows,
    configure_checklist,
    drain,
    inbound,
    outbox_rows,
    reply_texts,
    seed_business,
    unsucceeded_outbox,
)
from test_plan_custody import FailingAttachments, StaticAttachments


def custody_application(
    session_factory: async_sessionmaker[AsyncSession],
    owner_replies: OwnerReplyFake,
    *,
    storage: InMemoryObjectStorage | None,
    attachments: StaticAttachments | FailingAttachments | None,
) -> Application:
    ports = application_ports(
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    return build_application(
        replace(ports, object_storage=storage, source_attachments=attachments),
        session_factory=session_factory,
        now=Clock(),
    )


def plan_pdf(attachment_id: UUID | None = None, locator: str = "slack-file:F1") -> AttachmentPart:
    return AttachmentPart(
        attachment=AttachmentReference(
            attachment_id=attachment_id or uuid4(),
            media_kind=MediaKind.DOCUMENT,
            locator=locator,
            mime_type="application/pdf",
            filename="plans.pdf",
        )
    )


async def upload_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[SitePlanSetUploadRow]:
    async with session_factory() as session:
        return list((await session.scalars(select(SitePlanSetUploadRow))).all())


async def site_rows(session_factory: async_sessionmaker[AsyncSession]) -> list[SiteRow]:
    async with session_factory() as session:
        return list((await session.scalars(select(SiteRow))).all())


def test_plan_set_copy_notice_names_the_thread_and_the_retry_path() -> None:
    summary, recovery = FAILURE_GUIDANCE[PLAN_SET_COPY_COMMAND_TYPE]
    assert "plan set" in summary
    assert recovery == UPLOAD_PLAN_SET_AGAIN_IN_THREAD
    assert "Upload the plan set file again in this thread" in recovery


def test_plan_set_copy_command_identity_ignores_the_case_it_is_reported_to() -> None:
    business_id = BusinessId(uuid4())
    upload_id = PlanSetUploadId(uuid4())
    bare = plan_set_copy_command(business_id, upload_id)
    anchored = plan_set_copy_command(business_id, upload_id, field_note_case_id=uuid4())
    assert anchored.command_id == bare.command_id
    assert anchored.dedup_key == bare.dedup_key
    assert "field_note_case_id" not in bare.payload
    assert anchored.payload["plan_set_upload_id"] == str(upload_id)


@pytest.mark.asyncio
async def test_pdf_in_open_case_thread_is_copied_into_custody_once_under_replay(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    storage = InMemoryObjectStorage()
    attachments = StaticAttachments()
    application = custody_application(
        session_factory, owner_replies, storage=storage, attachments=attachments
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    pdf = plan_pdf()
    upload = inbound(business_id, None, message_key="plans-1", attachments=(pdf,))
    await application.ingest_service.ingest(upload)
    await drain(application)
    await application.ingest_service.ingest(upload)
    await drain(application)

    copies = await outbox_rows(session_factory, PLAN_SET_COPY_COMMAND_TYPE)
    assert [row.status for row in copies] == [OutboxStatus.SUCCEEDED.value]
    case_id = FieldNoteCaseId((await case_rows(session_factory))[0].id)
    assert copies[0].payload["field_note_case_id"] == str(case_id)
    uploads = await upload_rows(session_factory)
    assert [row.status for row in uploads] == [PlanSetUploadStatus.STORED.value]
    assert uploads[0].business_id == business_id
    assert uploads[0].site_id == field_note_case_site_id(business_id, case_id)
    assert [row.business_id for row in await site_rows(session_factory)] == [business_id]
    assert attachments.calls == 1
    assert len(storage.keys) == 1
    assert all(key.startswith(tenant_key_prefix(business_id)) for key in storage.keys)
    assert reply_texts(owner_replies).count("Queued 1 plan set file(s) for storage.") == 1
    assert PLAN_CUSTODY_NOT_ENABLED_REPLY not in reply_texts(owner_replies)
    assert await unsucceeded_outbox(session_factory) == 0


@pytest.mark.asyncio
async def test_same_file_in_two_tenants_yields_two_scoped_uploads(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first, second = BusinessId(uuid4()), BusinessId(uuid4())
    await seed_business(session_factory, first)
    await seed_business(session_factory, second)
    owner_replies = OwnerReplyFake()
    application = custody_application(
        session_factory,
        owner_replies,
        storage=InMemoryObjectStorage(),
        attachments=StaticAttachments(),
    )
    for business_id in (first, second):
        await configure_checklist(application, business_id)
        await application.ingest_service.ingest(
            inbound(business_id, "field notes: site: north", message_key=f"notes-{business_id}")
        )
        await drain(application)
        await application.ingest_service.ingest(
            inbound(
                business_id,
                None,
                message_key=f"plans-{business_id}",
                attachments=(plan_pdf(attachment_id=UUID(int=7)),),
            )
        )
        await drain(application)

    uploads = await upload_rows(session_factory)
    assert sorted(row.business_id for row in uploads) == sorted((first, second))
    assert len({row.id for row in uploads}) == 2
    assert await unsucceeded_outbox(session_factory) == 0


@pytest.mark.asyncio
async def test_pdf_without_custody_wired_gets_one_not_enabled_reply_and_no_command(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = custody_application(
        session_factory, owner_replies, storage=InMemoryObjectStorage(), attachments=None
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north", message_key="notes-1")
    )
    await drain(application)
    upload = inbound(business_id, None, message_key="plans-1", attachments=(plan_pdf(),))
    await application.ingest_service.ingest(upload)
    await drain(application)
    await application.ingest_service.ingest(upload)
    await drain(application)

    assert reply_texts(owner_replies).count(PLAN_CUSTODY_NOT_ENABLED_REPLY) == 1
    assert await outbox_rows(session_factory, PLAN_SET_COPY_COMMAND_TYPE) == []
    assert await upload_rows(session_factory) == []
    assert await unsucceeded_outbox(session_factory) == 0


@pytest.mark.asyncio
async def test_text_only_note_never_mentions_plan_custody(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = custody_application(
        session_factory, owner_replies, storage=None, attachments=None
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north", message_key="notes-1")
    )
    await drain(application)

    assert PLAN_CUSTODY_NOT_ENABLED_REPLY not in reply_texts(owner_replies)
    assert await outbox_rows(session_factory, PLAN_SET_COPY_COMMAND_TYPE) == []


@pytest.mark.asyncio
async def test_dead_plan_set_copy_tells_the_owner_once_without_the_provider_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = custody_application(
        session_factory,
        owner_replies,
        storage=InMemoryObjectStorage(),
        attachments=FailingAttachments(),
    )
    await configure_checklist(application, business_id)
    worker = OutboxWorker(
        application.outbox,
        application.dispatcher,
        now=Clock(),
        retry_in=timedelta(seconds=0),
        failure_notices=application.failure_notice_service,
    )

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north", message_key="notes-1")
    )
    await worker.drain()
    await application.ingest_service.ingest(
        inbound(business_id, None, message_key="plans-1", attachments=(plan_pdf(),))
    )
    for _ in range(DEFAULT_MAX_ATTEMPTS + 2):
        await worker.drain()

    copies = await outbox_rows(session_factory, PLAN_SET_COPY_COMMAND_TYPE)
    assert [row.status for row in copies] == [OutboxStatus.DEAD.value]
    summary, recovery = FAILURE_GUIDANCE[PLAN_SET_COPY_COMMAND_TYPE]
    notices = [text for text in reply_texts(owner_replies) if text.startswith(summary)]
    assert notices == [f"{summary}\n{recovery}"]
    assert "source channel is unavailable" not in "\n".join(reply_texts(owner_replies))
    assert [row.status for row in await upload_rows(session_factory)] == [
        PlanSetUploadStatus.FAILED.value
    ]


@pytest.mark.asyncio
async def test_plan_set_copy_notice_without_a_case_anchor_is_skipped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    application = custody_application(
        session_factory, OwnerReplyFake(), storage=None, attachments=None
    )
    service: NotifyExhaustedCommandService = application.failure_notice_service
    command = plan_set_copy_command(business_id, PlanSetUploadId(uuid4()))
    record = OutboxRecord(
        command=command,
        status=OutboxStatus.DEAD,
        attempts=DEFAULT_MAX_ATTEMPTS,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        available_at=Clock()(),
    )

    assert await service.notify(record) is None

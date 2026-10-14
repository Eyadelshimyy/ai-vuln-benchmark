import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from uuid import UUID, uuid4
from xml.etree import ElementTree

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    ReportGenerationFake,
    TranscriptionFake,
    application_ports,
)
from gvas.application.docx_report import (
    DocxReportRenderer,
    render_report_docx,
    report_docx_filename,
)
from gvas.application.report_approval import (
    NO_OPEN_CASE_REPLY,
    NO_REPORT_REPLY,
    approved_report_reply,
)
from gvas.composition import Application, build_application
from gvas.composition.dispatcher import OutboxWorker
from gvas.composition.report_publication import (
    REPORT_ARTIFACT_CUSTODY_SCOPE,
    ReportVersionNotFoundError,
)
from gvas.domain.enums import MediaKind, OutboxStatus
from gvas.domain.field_notes import FieldNoteCaseStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentPart, AttachmentReference
from gvas.domain.object_storage import ObjectCustodyError, ObjectCustodyRequest, StoredObject
from gvas.domain.outbox import DEFAULT_MAX_ATTEMPTS
from gvas.domain.reporting import (
    DOCX_MEDIA_TYPE,
    FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE,
    FieldNotesReportDocument,
    FieldNotesReportVersion,
    ReportArtifactLocator,
    field_notes_report_publish_command,
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

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NOW = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)


def version(business_id: BusinessId | None = None, number: int = 2) -> FieldNotesReportVersion:
    return FieldNotesReportVersion(
        business_id=business_id or BusinessId(uuid4()),
        case_id=uuid4(),
        report_id=uuid4(),
        report_version_id=uuid4(),
        version=number,
        source_fingerprint="a" * 64,
        generated_at=NOW,
        document=FieldNotesReportDocument.model_validate(
            {
                "schema_version": "field-notes-report/v1",
                "title": "Roof Inspection & Repair",
                "sections": [
                    {
                        "section_key": "observations",
                        "heading": "Observations",
                        "blocks": [
                            {
                                "kind": "text",
                                "text": "Panel was secured <and> checked.",
                                "evidence_refs": [{"source": "transcript", "key": "canonical"}],
                            }
                        ],
                    },
                    {
                        "section_key": "work",
                        "heading": "Work performed",
                        "blocks": [
                            {
                                "kind": "text",
                                "text": "Replaced the downpipe.",
                                "evidence_refs": [{"source": "transcript", "key": "canonical"}],
                            }
                        ],
                    },
                ],
            }
        ),
    )


def paragraphs(docx: bytes) -> list[str]:
    with zipfile.ZipFile(BytesIO(docx)) as package:
        root = ElementTree.fromstring(package.read("word/document.xml"))  # noqa: S314
    return [
        "".join(run.text or "" for run in paragraph.iter(f"{WORD_NS}t"))
        for paragraph in root.iter(f"{WORD_NS}p")
    ]


def test_docx_is_a_valid_package_with_title_version_headings_and_blocks() -> None:
    docx = render_report_docx(version())

    with zipfile.ZipFile(BytesIO(docx)) as package:
        assert package.testzip() is None
        names = set(package.namelist())
        assert {
            "[Content_Types].xml",
            "_rels/.rels",
            "word/document.xml",
            "word/_rels/document.xml.rels",
            "word/styles.xml",
        } <= names
        for name in names:
            ElementTree.fromstring(package.read(name))  # noqa: S314
    assert paragraphs(docx) == [
        "Roof Inspection & Repair",
        "Report version 2",
        "Observations",
        "Panel was secured <and> checked.",
        "Work performed",
        "Replaced the downpipe.",
    ]


def test_docx_rendering_is_byte_for_byte_reproducible() -> None:
    v = version()
    assert render_report_docx(v) == render_report_docx(v)


def test_docx_renderer_names_and_types_the_artifact() -> None:
    artifact = DocxReportRenderer().render(version())

    assert artifact.media_type == DOCX_MEDIA_TYPE
    assert artifact.filename == "roof-inspection-repair-v2.docx"
    assert report_docx_filename(version(number=7)) == "roof-inspection-repair-v7.docx"
    assert artifact.content.startswith(b"PK")


def test_publish_command_identity_is_pinned_to_the_version_and_the_approval() -> None:
    business_id = BusinessId(uuid4())
    case_id = uuid4()
    report_version_id = uuid4()

    first = field_notes_report_publish_command(business_id, case_id, report_version_id, "ok-1")
    again = field_notes_report_publish_command(business_id, case_id, report_version_id, "ok-1")
    retry = field_notes_report_publish_command(business_id, case_id, report_version_id, "ok-2")
    other = field_notes_report_publish_command(business_id, case_id, uuid4(), "ok-1")

    assert first == again
    assert first.command_type == FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE
    assert first.payload == {
        "field_note_case_id": str(case_id),
        "report_version_id": str(report_version_id),
    }
    assert first.dedup_key == f"field_notes_report_publish:{report_version_id}:ok-1"
    assert retry.payload == first.payload
    assert len({first.command_id, retry.command_id, other.command_id}) == 3
    assert len({first.dedup_key, retry.dedup_key, other.dedup_key}) == 3


def test_report_artifact_locator_round_trips_and_rejects_foreign_locators() -> None:
    locator = ReportArtifactLocator(business_id=BusinessId(uuid4()), report_version_id=uuid4())

    assert ReportArtifactLocator.decode(locator.encode()) == locator
    assert ReportArtifactLocator.decode("slack-file:F123") is None
    assert ReportArtifactLocator.decode("field-notes-report:not-a-uuid:nope") is None
    assert ReportArtifactLocator.decode("field-notes-report:") is None


def publishing_application(
    session_factory: async_sessionmaker[AsyncSession],
    owner_replies: OwnerReplyFake,
    storage: InMemoryObjectStorage,
) -> Application:
    ports = application_ports(
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    return build_application(
        replace(ports, object_storage=storage),
        session_factory=session_factory,
        now=Clock(),
    )


def attachments(owner_replies: OwnerReplyFake) -> list[AttachmentReference]:
    return [
        part.attachment
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, AttachmentPart)
    ]


@pytest.mark.asyncio
async def test_approve_report_publishes_the_docx_once_and_keeps_the_case_open(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    storage = InMemoryObjectStorage()
    application = publishing_application(session_factory, owner_replies, storage)
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    versions = await report_versions(session_factory)
    assert [row.version for row in versions] == [1]
    text_replies_before = len(reply_texts(owner_replies))

    approve = inbound(business_id, "  Approve Report ", message_key="approve-1")
    await application.ingest_service.ingest(approve)
    await application.ingest_service.ingest(approve)
    await drain(application)

    assert reply_texts(owner_replies).count(approved_report_reply(1)) == 1
    publishes = await outbox_rows(session_factory, FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE)
    assert len(publishes) == 1
    assert publishes[0].payload["report_version_id"] == str(versions[0].id)
    assert await unsucceeded_outbox(session_factory) == 0

    published = attachments(owner_replies)
    assert len(published) == 1
    attachment = published[0]
    assert attachment.media_kind is MediaKind.DOCUMENT
    assert attachment.mime_type == DOCX_MEDIA_TYPE
    assert attachment.filename is not None and attachment.filename.endswith("-v1.docx")
    locator = ReportArtifactLocator.decode(attachment.locator)
    assert locator is not None
    assert locator.business_id == business_id
    assert locator.report_version_id == versions[0].id
    assert len(reply_texts(owner_replies)) == text_replies_before + 2

    payload = await application.report_artifacts.fetch(attachment)
    assert payload.mime_type == DOCX_MEDIA_TYPE
    assert "Report version 1" in paragraphs(payload.content)
    assert attachment.byte_size == len(payload.content)

    assert len(storage.keys) == 1
    stored_key = storage.keys[0]
    assert str(business_id) in stored_key
    assert REPORT_ARTIFACT_CUSTODY_SCOPE in stored_key
    assert stored_key.endswith(attachment.filename)

    assert [case.status for case in await case_rows(session_factory)] == [
        FieldNoteCaseStatus.OPEN.value
    ]

    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-2")
    )
    await drain(application)

    assert reply_texts(owner_replies).count(approved_report_reply(1)) == 2
    assert len(await outbox_rows(session_factory, FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE)) == 2
    assert await unsucceeded_outbox(session_factory) == 0
    assert len(attachments(owner_replies)) == 1
    assert len(storage.keys) == 1


class RecoveringObjectStorage(InMemoryObjectStorage):
    """Refuses custody until told otherwise, the way an outage would."""

    def __init__(self) -> None:
        super().__init__()
        self.available = False
        self.refused = 0

    async def put(self, request: ObjectCustodyRequest) -> StoredObject:
        if not self.available:
            self.refused += 1
            raise ObjectCustodyError("bucket unavailable: secret-bucket-name")
        return await super().put(request)


@pytest.mark.asyncio
async def test_dead_publish_tells_the_owner_and_approving_again_recovers_it(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    storage = RecoveringObjectStorage()
    application = publishing_application(session_factory, owner_replies, storage)
    await configure_checklist(application, business_id)
    worker = OutboxWorker(
        application.outbox,
        application.dispatcher,
        now=Clock(),
        retry_in=timedelta(seconds=0),
        failure_notices=application.failure_notice_service,
    )

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await worker.drain()
    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-1")
    )
    await worker.drain()
    await worker.drain()

    publishes = await outbox_rows(session_factory, FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE)
    assert [row.status for row in publishes] == [OutboxStatus.DEAD.value]
    assert storage.refused == DEFAULT_MAX_ATTEMPTS
    notices = [
        text
        for text in reply_texts(owner_replies)
        if text.startswith("The approved report document could not be posted")
    ]
    assert len(notices) == 1
    assert "approve report" in notices[0]
    assert "secret-bucket-name" not in notices[0]
    assert attachments(owner_replies) == []
    assert [case.status for case in await case_rows(session_factory)] == [
        FieldNoteCaseStatus.OPEN.value
    ]

    storage.available = True
    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-2")
    )
    await worker.drain()

    publishes = await outbox_rows(session_factory, FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE)
    assert sorted(row.status for row in publishes) == [
        OutboxStatus.DEAD.value,
        OutboxStatus.SUCCEEDED.value,
    ]
    assert len(attachments(owner_replies)) == 1
    assert len(storage.keys) == 1


@pytest.mark.asyncio
async def test_approve_report_without_an_active_case_or_report_replies_without_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = publishing_application(session_factory, owner_replies, InMemoryObjectStorage())
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-none")
    )
    await drain(application)

    assert reply_texts(owner_replies) == [NO_OPEN_CASE_REPLY]
    assert await unsucceeded_outbox(session_factory) == 0
    assert await outbox_rows(session_factory, FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE) == []


@pytest.mark.asyncio
async def test_approve_report_before_the_report_exists_asks_the_owner_to_wait(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = publishing_application(session_factory, owner_replies, InMemoryObjectStorage())
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north", message_key="notes-open")
    )
    await drain(application)
    assert await report_versions(session_factory) == []

    await application.ingest_service.ingest(
        inbound(business_id, "approve report", message_key="approve-early")
    )
    await drain(application)

    assert NO_REPORT_REPLY in reply_texts(owner_replies)
    assert await outbox_rows(session_factory, FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE) == []


@pytest.mark.asyncio
async def test_report_artifact_access_refuses_foreign_and_cross_tenant_locators(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = publishing_application(session_factory, owner_replies, InMemoryObjectStorage())
    await configure_checklist(application, business_id)
    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    report_version_id: UUID = (await report_versions(session_factory))[0].id

    def reference(locator: str) -> AttachmentReference:
        return AttachmentReference(
            attachment_id=uuid4(), media_kind=MediaKind.DOCUMENT, locator=locator
        )

    foreign = ReportArtifactLocator(
        business_id=BusinessId(uuid4()), report_version_id=report_version_id
    )
    with pytest.raises(ReportVersionNotFoundError):
        await application.report_artifacts.fetch(reference(foreign.encode()))
    with pytest.raises(ReportVersionNotFoundError):
        await application.report_artifacts.fetch(reference("slack-file:F1"))

    own = ReportArtifactLocator(business_id=business_id, report_version_id=report_version_id)
    payload = await application.report_artifacts.fetch(reference(own.encode()))
    assert isinstance(payload.content, bytes) and payload.content.startswith(b"PK")

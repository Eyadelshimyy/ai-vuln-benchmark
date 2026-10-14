from typing import Protocol
from uuid import UUID, uuid5

from gvas.composition.report_delivery import FieldNoteUnitOfWorkFactory, MessageUnitOfWorkFactory
from gvas.domain.enums import MediaKind
from gvas.domain.field_notes import FieldNoteCaseId, FieldNoteCaseNotFoundError
from gvas.domain.identifiers import BusinessId, MessageId
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentPayload,
    AttachmentReference,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.object_storage import ObjectCustodyRequest
from gvas.domain.outbox import owner_reply_command
from gvas.domain.ports import ObjectStoragePort
from gvas.domain.reporting import (
    FieldNotesReportVersion,
    ReportArtifactLocator,
    ReportArtifactRendererPort,
    ReportUnitOfWork,
)

REPORT_ARTIFACT_NAMESPACE = UUID("3f6b8d21-9c4e-5a70-b1d2-8e5f7a9c0b64")
REPORT_ARTIFACT_CUSTODY_SCOPE = "field-notes-reports"


class ReportUnitOfWorkFactory(Protocol):
    def __call__(self) -> ReportUnitOfWork: ...


class ReportVersionNotFoundError(LookupError):
    """The publish command names a report version this business does not hold."""


def report_artifact_reference(
    version: FieldNotesReportVersion, media_type: str, filename: str, byte_size: int
) -> AttachmentReference:
    locator = ReportArtifactLocator(
        business_id=version.business_id, report_version_id=version.report_version_id
    )
    return AttachmentReference(
        attachment_id=uuid5(REPORT_ARTIFACT_NAMESPACE, locator.encode()),
        media_kind=MediaKind.DOCUMENT,
        locator=locator.encode(),
        mime_type=media_type,
        filename=filename,
        byte_size=byte_size,
    )


class PublishFieldNotesReportService:
    """Posts the approved report version into its conversation as a document.

    The artifact is rendered from the pinned version, kept in managed object
    storage when a store is configured, and handed to the owner reply outbox as
    an attachment whose locator names the version. The outbound message is keyed
    on the version, so a replayed publish command reuses the one message and the
    channel adapter's delivery ledger keeps the file from being posted twice.
    The case stays open; only ``close notes`` closes it.
    """

    def __init__(
        self,
        renderer: ReportArtifactRendererPort,
        report_unit_of_work_factory: ReportUnitOfWorkFactory,
        field_note_unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        message_unit_of_work_factory: MessageUnitOfWorkFactory,
        object_storage: ObjectStoragePort | None = None,
    ) -> None:
        self._renderer = renderer
        self._reports = report_unit_of_work_factory
        self._field_notes = field_note_unit_of_work_factory
        self._messages = message_unit_of_work_factory
        self._storage = object_storage

    async def publish(self, business_id: BusinessId, report_version_id: UUID) -> MessageId:
        async with self._reports() as report_unit_of_work:
            version = await report_unit_of_work.reports.get_version(business_id, report_version_id)
            await report_unit_of_work.commit()
        if version is None:
            raise ReportVersionNotFoundError("report version was not found")
        case_id = FieldNoteCaseId(version.case_id)
        async with self._field_notes() as unit_of_work:
            case = await unit_of_work.field_note_cases.get(business_id, case_id)
            await unit_of_work.commit()
        if case is None:
            raise FieldNoteCaseNotFoundError("field-note case was not found")

        artifact = self._renderer.render(version)
        if self._storage is not None:
            await self._storage.put(
                ObjectCustodyRequest(
                    business_id=business_id,
                    scope=REPORT_ARTIFACT_CUSTODY_SCOPE,
                    name=f"{version.report_id}/v{version.version}/{artifact.filename}",
                    content=artifact.content,
                    media_kind=MediaKind.DOCUMENT,
                    media_type=artifact.media_type,
                    filename=artifact.filename,
                )
            )

        message = OutboundOwnerMessage(
            business_id=business_id,
            conversation_ref=case.conversation_ref,
            parts=(
                TextPart(
                    text=f"{version.document.title} — approved report version {version.version}"
                ),
                AttachmentPart(
                    attachment=report_artifact_reference(
                        version, artifact.media_type, artifact.filename, len(artifact.content)
                    )
                ),
            ),
            correlation_id=f"field_notes_report_publish:{version.report_version_id}",
        )
        async with self._messages() as unit_of_work:
            outbound_message_id = await unit_of_work.outbound_messages.create(
                message, case.conversation_id, case.origin_inbound_message_id
            )
            await unit_of_work.outbox.enqueue(owner_reply_command(business_id, outbound_message_id))
            await unit_of_work.commit()
        return outbound_message_id


class ReportArtifactAccess:
    """Serves published report bytes to channel adapters by opaque locator.

    Adapters see only an ``AttachmentReference``; this resolves the version it
    names, within the tenant it names, and renders the same bytes the publish
    step produced. Unknown locators are refused rather than guessed.
    """

    def __init__(
        self,
        renderer: ReportArtifactRendererPort,
        report_unit_of_work_factory: ReportUnitOfWorkFactory,
    ) -> None:
        self._renderer = renderer
        self._reports = report_unit_of_work_factory

    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        locator = ReportArtifactLocator.decode(attachment.locator)
        if locator is None:
            raise ReportVersionNotFoundError("attachment is not a published report")
        async with self._reports() as unit_of_work:
            version = await unit_of_work.reports.get_version(
                locator.business_id, locator.report_version_id
            )
            await unit_of_work.commit()
        if version is None:
            raise ReportVersionNotFoundError("report version was not found")
        artifact = self._renderer.render(version)
        return AttachmentPayload(
            content=artifact.content, mime_type=artifact.media_type, filename=artifact.filename
        )

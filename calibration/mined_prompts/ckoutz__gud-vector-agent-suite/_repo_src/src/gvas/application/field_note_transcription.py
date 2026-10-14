from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from gvas.domain.field_note_repositories import (
    FieldNoteUnitOfWork,
    LostTranscriptionLeaseError,
    TranscriptionClaimResult,
)
from gvas.domain.field_notes import (
    CanonicalFieldNoteTranscript,
    FieldNoteCase,
    FieldNoteCaseId,
    FieldNoteCaseNotFoundError,
    FieldNotePart,
    FieldNotePartId,
    FieldNotePartKind,
    UnsupportedFieldNoteMediaError,
    build_canonical_transcript,
)
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import AttachmentPayload, AttachmentReference
from gvas.domain.ports import AttachmentAccessPort, TranscriptionPort
from gvas.domain.usage import UsageCeilingGuard, UsageKind


class FieldNoteMediaHandoff:
    def __init__(self, attachments: AttachmentAccessPort) -> None:
        self._attachments = attachments

    @staticmethod
    def redact_attachment(attachment: AttachmentReference) -> str:
        return f"attachment {attachment.attachment_id} ({attachment.media_kind})"

    async def open_audio(self, part: FieldNotePart) -> AttachmentPayload:
        if part.attachment is None or part.kind is not FieldNotePartKind.AUDIO:
            attachment = part.attachment
            detail = (
                self.redact_attachment(attachment)
                if attachment is not None
                else "field-note part without attachment"
            )
            raise UnsupportedFieldNoteMediaError(f"{detail} is not audio")
        try:
            return await self._attachments.fetch(part.attachment)
        except Exception as error:
            raise UnsupportedFieldNoteMediaError(
                f"unable to access {self.redact_attachment(part.attachment)}"
            ) from error


class TranscriptionOutcome(StrEnum):
    TRANSCRIBED = "transcribed"
    ALREADY_TRANSCRIBED = "already_transcribed"
    BUSY = "busy"
    MISSING = "missing"
    FAILED = "failed"
    LEASE_LOST = "lease_lost"
    CEILING_REACHED = "ceiling_reached"


@dataclass(frozen=True)
class TranscriptionResultReport:
    outcome: TranscriptionOutcome
    part_id: FieldNotePartId | None = None
    attempts: int = 0
    detail: str | None = None


class FieldNoteUnitOfWorkFactory(Protocol):
    def __call__(self) -> FieldNoteUnitOfWork: ...


class TranscribeFieldNoteAudioService:
    def __init__(
        self,
        unit_of_work_factory: FieldNoteUnitOfWorkFactory,
        transcription: TranscriptionPort,
        ceilings: UsageCeilingGuard | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._transcription = transcription
        self._ceilings = ceilings or UsageCeilingGuard()

    async def transcribe(
        self,
        business_id: BusinessId,
        part_id: FieldNotePartId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> TranscriptionResultReport:
        # Checked before the claim so a part held back by the ceiling keeps its
        # PENDING status and attempt count untouched; nothing is leased.
        if await self._ceilings.is_reached(
            business_id, UsageKind.TRANSCRIPTION_AUDIO_SECONDS, now=now
        ):
            return TranscriptionResultReport(
                TranscriptionOutcome.CEILING_REACHED,
                part_id=part_id,
                detail="monthly transcription ceiling reached",
            )
        async with self._unit_of_work_factory() as unit_of_work:
            claim = await unit_of_work.field_note_transcriptions.claim(
                business_id, part_id, now=now, stale_before=stale_before
            )
            await unit_of_work.commit()
        if claim.result is TranscriptionClaimResult.TERMINAL:
            return TranscriptionResultReport(
                TranscriptionOutcome.ALREADY_TRANSCRIBED,
                part_id=claim.part_id,
                attempts=claim.attempts,
            )
        if claim.result is TranscriptionClaimResult.BUSY:
            return TranscriptionResultReport(
                TranscriptionOutcome.BUSY,
                part_id=claim.part_id,
                attempts=claim.attempts,
            )
        if claim.result is TranscriptionClaimResult.MISSING:
            return TranscriptionResultReport(TranscriptionOutcome.MISSING)
        if claim.audio is None or claim.part_id is None:
            return TranscriptionResultReport(
                TranscriptionOutcome.LEASE_LOST, detail="acquired claim is incomplete"
            )
        try:
            transcript = await self._transcription.transcribe(claim.audio)
        except Exception as error:
            detail = repr(error)
            async with self._unit_of_work_factory() as unit_of_work:
                try:
                    await unit_of_work.field_note_transcriptions.record_failure(claim, detail)
                    await unit_of_work.commit()
                except LostTranscriptionLeaseError:
                    await unit_of_work.rollback()
                    return TranscriptionResultReport(
                        TranscriptionOutcome.LEASE_LOST,
                        part_id=claim.part_id,
                        attempts=claim.attempts,
                        detail=detail,
                    )
            return TranscriptionResultReport(
                TranscriptionOutcome.FAILED,
                part_id=claim.part_id,
                attempts=claim.attempts,
                detail=detail,
            )
        async with self._unit_of_work_factory() as unit_of_work:
            try:
                await unit_of_work.field_note_transcriptions.record_success(claim, transcript)
                await unit_of_work.commit()
            except LostTranscriptionLeaseError:
                await unit_of_work.rollback()
                return TranscriptionResultReport(
                    TranscriptionOutcome.LEASE_LOST,
                    part_id=claim.part_id,
                    attempts=claim.attempts,
                )
        return TranscriptionResultReport(
            TranscriptionOutcome.TRANSCRIBED,
            part_id=claim.part_id,
            attempts=claim.attempts,
        )


class FieldNoteTranscriptService:
    def __init__(self, unit_of_work_factory: FieldNoteUnitOfWorkFactory) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def canonical_transcript(
        self, business_id: BusinessId, case_id: FieldNoteCaseId
    ) -> CanonicalFieldNoteTranscript:
        async with self._unit_of_work_factory() as unit_of_work:
            case = await unit_of_work.field_note_cases.get(business_id, case_id)
            await unit_of_work.commit()
        if case is None:
            raise FieldNoteCaseNotFoundError("field-note case was not found")
        return build_canonical_transcript(
            FieldNoteCase(
                case_id=case.case_id,
                business_id=case.business_id,
                conversation_ref=case.conversation_ref,
                origin_inbound_message_id=case.origin_inbound_message_id,
                status=case.status,
                parts=case.parts,
            )
        )

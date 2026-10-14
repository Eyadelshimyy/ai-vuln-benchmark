from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from gvas.domain.enums import MediaKind
from gvas.domain.field_note_repositories import (
    AmbiguousFieldNoteMessageError,
    CrossBusinessFieldNoteError,
    FieldNoteCaseClosureResult,
    FieldNoteCaseRecord,
    FieldNoteCaseRepository,
    FieldNoteConversationStateRepository,
    FieldNoteIntakeResult,
    FieldNoteMessageLocation,
    FieldNoteMessageLocator,
    FieldNotePartDraft,
    FieldNoteTranscriptionRepository,
    FieldNoteUnitOfWork,
    LostTranscriptionLeaseError,
    TranscriptionClaim,
    TranscriptionClaimResult,
)
from gvas.domain.field_notes import (
    FieldNoteCaseId,
    FieldNoteCaseNotFoundError,
    FieldNoteCaseStatus,
    FieldNotePart,
    FieldNotePartId,
    FieldNotePartKind,
    TranscriptionStatus,
)
from gvas.domain.identifiers import (
    BusinessId,
    ConversationId,
    EndpointId,
    MessageId,
    MessageKey,
)
from gvas.domain.messages import (
    AttachmentReference,
    AudioReference,
    ConversationRef,
    TranscriptResult,
)
from gvas.infrastructure.field_note_models import FieldNoteCase as FieldNoteCaseRow
from gvas.infrastructure.field_note_models import (
    FieldNoteConversationState,
    FieldNotePartRow,
)
from gvas.infrastructure.models import Conversation, InboundMessage


def _rowcount(result: Result[Any]) -> int:
    return cast(CursorResult[Any], result).rowcount


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _with_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _attachment(row: FieldNotePartRow) -> AttachmentReference | None:
    if row.attachment_id is None:
        return None
    if row.media_kind is None or row.attachment_locator is None:
        raise ValueError("stored attachment is incomplete")
    return AttachmentReference(
        attachment_id=row.attachment_id,
        media_kind=MediaKind(row.media_kind),
        locator=row.attachment_locator,
        mime_type=row.mime_type,
        filename=row.filename,
        byte_size=row.byte_size,
    )


def _part(row: FieldNotePartRow) -> FieldNotePart:
    if (
        row.transcription_status == TranscriptionStatus.SUCCEEDED.value
        and row.transcript_text is None
    ):
        raise ValueError("stored transcript is incomplete")
    transcript = (
        TranscriptResult(
            text=row.transcript_text or "",
            language=row.transcript_language,
            confidence=row.transcript_confidence,
            duration_seconds=row.transcript_duration_seconds,
            provider_ref=row.transcript_provider_ref,
        )
        if row.transcription_status == TranscriptionStatus.SUCCEEDED.value
        and row.transcript_text is not None
        else None
    )
    return FieldNotePart(
        part_id=FieldNotePartId(row.id),
        case_id=FieldNoteCaseId(row.case_id),
        business_id=BusinessId(row.business_id),
        sequence=row.sequence,
        kind=FieldNotePartKind(row.kind),
        text=row.text,
        attachment=_attachment(row),
        transcription_status=TranscriptionStatus(row.transcription_status),
        transcript=transcript,
        attempts=row.attempts,
        last_error=row.last_error,
    )


class SqlFieldNoteMessageLocator:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def locate(
        self, business_id: BusinessId, conversation_ref: ConversationRef, message_key: MessageKey
    ) -> FieldNoteMessageLocation | None:
        if conversation_ref.business_id != business_id:
            raise CrossBusinessFieldNoteError("conversation reference belongs to another business")
        result = await self.session.execute(
            select(InboundMessage, Conversation)
            .join(Conversation, Conversation.id == InboundMessage.conversation_id)
            .where(
                InboundMessage.business_id == business_id,
                InboundMessage.message_key == message_key,
                Conversation.business_id == business_id,
                Conversation.external_conversation_id == conversation_ref.external_conversation_id,
                Conversation.endpoint_id == InboundMessage.endpoint_id,
            )
        )
        rows = result.all()
        if len(rows) > 1:
            raise AmbiguousFieldNoteMessageError(
                "field note message identity matches multiple endpoint-scoped messages"
            )
        if not rows:
            return None
        inbound, conversation = rows[0]
        return FieldNoteMessageLocation(
            business_id=business_id,
            endpoint_id=EndpointId(inbound.endpoint_id),
            conversation_id=ConversationId(conversation.id),
            inbound_message_id=MessageId(inbound.id),
        )


class SqlFieldNoteCaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _record(self, row: FieldNoteCaseRow) -> FieldNoteCaseRecord:
        conversation = await self.session.scalar(
            select(Conversation).where(
                Conversation.business_id == row.business_id,
                Conversation.id == row.conversation_id,
            )
        )
        if conversation is None:
            raise CrossBusinessFieldNoteError("field-note case references another business")
        parts = tuple(
            await self.session.scalars(
                select(FieldNotePartRow)
                .where(
                    FieldNotePartRow.business_id == row.business_id,
                    FieldNotePartRow.case_id == row.id,
                )
                .order_by(FieldNotePartRow.sequence)
            )
        )
        return FieldNoteCaseRecord(
            case_id=FieldNoteCaseId(row.id),
            business_id=BusinessId(row.business_id),
            conversation_id=ConversationId(row.conversation_id),
            conversation_ref=ConversationRef(
                business_id=BusinessId(conversation.business_id),
                external_conversation_id=conversation.external_conversation_id,
            ),
            origin_inbound_message_id=MessageId(row.origin_inbound_message_id),
            status=FieldNoteCaseStatus(row.status),
            parts=tuple(_part(item) for item in parts),
        )

    async def get(
        self, business_id: BusinessId, case_id: FieldNoteCaseId
    ) -> FieldNoteCaseRecord | None:
        row = await self.session.scalar(
            select(FieldNoteCaseRow).where(
                FieldNoteCaseRow.business_id == business_id,
                FieldNoteCaseRow.id == case_id,
            )
        )
        return await self._record(row) if row is not None else None

    async def record_intake(
        self,
        *,
        location: FieldNoteMessageLocation,
        parts: Sequence[FieldNotePartDraft],
        case_id: FieldNoteCaseId | None,
    ) -> FieldNoteIntakeResult:
        conversation = await self.session.scalar(
            select(Conversation).where(Conversation.id == location.conversation_id)
        )
        inbound = await self.session.scalar(
            select(InboundMessage).where(InboundMessage.id == location.inbound_message_id)
        )
        if (
            conversation is None
            or inbound is None
            or conversation.business_id != location.business_id
            or inbound.business_id != location.business_id
            or inbound.conversation_id != location.conversation_id
            or conversation.endpoint_id != location.endpoint_id
            or inbound.endpoint_id != location.endpoint_id
        ):
            raise CrossBusinessFieldNoteError(
                "field-note message location references another business"
            )
        if case_id is not None:
            case = await self.session.scalar(
                select(FieldNoteCaseRow)
                .where(
                    FieldNoteCaseRow.business_id == location.business_id,
                    FieldNoteCaseRow.id == case_id,
                )
                .with_for_update()
            )
            if case is None:
                any_case = await self.session.scalar(
                    select(FieldNoteCaseRow).where(FieldNoteCaseRow.id == case_id)
                )
                if any_case is not None:
                    raise CrossBusinessFieldNoteError("field-note case belongs to another business")
                raise FieldNoteCaseNotFoundError("field-note case was not found")
            if case.status != FieldNoteCaseStatus.OPEN.value:
                raise FieldNoteCaseNotFoundError("field-note case is not open")
            if case.conversation_id != location.conversation_id:
                raise CrossBusinessFieldNoteError("field-note case belongs to another conversation")
            existing_delivery = await self.session.scalars(
                select(FieldNotePartRow)
                .where(
                    FieldNotePartRow.business_id == location.business_id,
                    FieldNotePartRow.case_id == case_id,
                    FieldNotePartRow.source_inbound_message_id == location.inbound_message_id,
                )
                .order_by(FieldNotePartRow.sequence)
            )
            delivery_parts = tuple(existing_delivery)
            if delivery_parts:
                return FieldNoteIntakeResult(
                    case=await self._record(case),
                    created_case=False,
                    created_part_ids=(),
                    audio_part_ids=tuple(
                        FieldNotePartId(item.id)
                        for item in delivery_parts
                        if item.kind == FieldNotePartKind.AUDIO.value
                        and item.transcription_status == TranscriptionStatus.PENDING.value
                    ),
                )
        else:
            case = await self.session.scalar(
                select(FieldNoteCaseRow)
                .where(
                    FieldNoteCaseRow.business_id == location.business_id,
                    FieldNoteCaseRow.origin_inbound_message_id == location.inbound_message_id,
                )
                .with_for_update()
            )
            if case is not None:
                delivery_parts = tuple(
                    await self.session.scalars(
                        select(FieldNotePartRow)
                        .where(
                            FieldNotePartRow.business_id == location.business_id,
                            FieldNotePartRow.case_id == case.id,
                            FieldNotePartRow.source_inbound_message_id
                            == location.inbound_message_id,
                        )
                        .order_by(FieldNotePartRow.sequence)
                    )
                )
                if delivery_parts:
                    return FieldNoteIntakeResult(
                        case=await self._record(case),
                        created_case=False,
                        created_part_ids=(),
                        audio_part_ids=tuple(
                            FieldNotePartId(item.id)
                            for item in delivery_parts
                            if item.kind == FieldNotePartKind.AUDIO.value
                            and item.transcription_status == TranscriptionStatus.PENDING.value
                        ),
                    )
            if case is None:
                try:
                    async with self.session.begin_nested():
                        case = FieldNoteCaseRow(
                            id=uuid4(),
                            business_id=location.business_id,
                            conversation_id=location.conversation_id,
                            origin_inbound_message_id=location.inbound_message_id,
                            status=FieldNoteCaseStatus.OPEN.value,
                            created_at=datetime.now(UTC),
                            updated_at=datetime.now(UTC),
                        )
                        self.session.add(case)
                        await self.session.flush()
                except IntegrityError:
                    case = await self.session.scalar(
                        select(FieldNoteCaseRow).where(
                            FieldNoteCaseRow.business_id == location.business_id,
                            FieldNoteCaseRow.origin_inbound_message_id
                            == location.inbound_message_id,
                        )
                    )
                    if case is None:
                        raise
                    delivery_parts = tuple(
                        await self.session.scalars(
                            select(FieldNotePartRow)
                            .where(
                                FieldNotePartRow.business_id == location.business_id,
                                FieldNotePartRow.case_id == case.id,
                                FieldNotePartRow.source_inbound_message_id
                                == location.inbound_message_id,
                            )
                            .order_by(FieldNotePartRow.sequence)
                        )
                    )
                    if delivery_parts:
                        return FieldNoteIntakeResult(
                            case=await self._record(case),
                            created_case=False,
                            created_part_ids=(),
                            audio_part_ids=tuple(
                                FieldNotePartId(item.id)
                                for item in delivery_parts
                                if item.kind == FieldNotePartKind.AUDIO.value
                                and item.transcription_status == TranscriptionStatus.PENDING.value
                            ),
                        )
        max_sequence = await self.session.scalar(
            select(func.max(FieldNotePartRow.sequence)).where(
                FieldNotePartRow.business_id == location.business_id,
                FieldNotePartRow.case_id == case.id,
            )
        )
        rows = [
            FieldNotePartRow(
                id=uuid4(),
                business_id=location.business_id,
                case_id=case.id,
                source_inbound_message_id=location.inbound_message_id,
                sequence=(max_sequence if max_sequence is not None else 0) + index + 1,
                kind=draft.kind.value,
                text=draft.text,
                attachment_id=draft.attachment.attachment_id if draft.attachment else None,
                media_kind=draft.attachment.media_kind.value if draft.attachment else None,
                attachment_locator=draft.attachment.locator if draft.attachment else None,
                mime_type=draft.attachment.mime_type if draft.attachment else None,
                filename=draft.attachment.filename if draft.attachment else None,
                byte_size=draft.attachment.byte_size if draft.attachment else None,
                transcription_status=(
                    TranscriptionStatus.PENDING.value
                    if draft.kind is FieldNotePartKind.AUDIO
                    else TranscriptionStatus.NOT_REQUIRED.value
                ),
                attempts=0,
                lease_token=uuid4(),
            )
            for index, draft in enumerate(parts)
        ]
        try:
            async with self.session.begin_nested():
                self.session.add_all(rows)
                case.updated_at = datetime.now(UTC)
                await self.session.flush()
        except IntegrityError:
            existing_rows = tuple(
                await self.session.scalars(
                    select(FieldNotePartRow)
                    .where(
                        FieldNotePartRow.business_id == location.business_id,
                        FieldNotePartRow.case_id == case.id,
                        FieldNotePartRow.source_inbound_message_id == location.inbound_message_id,
                    )
                    .order_by(FieldNotePartRow.sequence)
                )
            )
            if not existing_rows:
                raise
            return FieldNoteIntakeResult(
                case=await self._record(case),
                created_case=False,
                created_part_ids=(),
                audio_part_ids=tuple(
                    FieldNotePartId(item.id)
                    for item in existing_rows
                    if item.kind == FieldNotePartKind.AUDIO.value
                    and item.transcription_status == TranscriptionStatus.PENDING.value
                ),
            )
        return FieldNoteIntakeResult(
            case=await self._record(case),
            created_case=case_id is None,
            created_part_ids=tuple(FieldNotePartId(row.id) for row in rows),
            audio_part_ids=tuple(
                FieldNotePartId(row.id) for row in rows if row.kind == FieldNotePartKind.AUDIO.value
            ),
        )

    async def close(
        self, business_id: BusinessId, case_id: FieldNoteCaseId, *, now: datetime
    ) -> FieldNoteCaseClosureResult:
        _aware(now, "now")
        case = await self.session.scalar(
            select(FieldNoteCaseRow)
            .where(
                FieldNoteCaseRow.business_id == business_id,
                FieldNoteCaseRow.id == case_id,
            )
            .with_for_update()
        )
        if case is None:
            return FieldNoteCaseClosureResult.MISSING
        if case.status != FieldNoteCaseStatus.OPEN.value:
            return FieldNoteCaseClosureResult.ALREADY_CLOSED
        case.status = FieldNoteCaseStatus.CLOSED.value
        case.closed_at = now
        case.updated_at = now
        await self.session.flush()
        return FieldNoteCaseClosureResult.CLOSED


class SqlFieldNoteConversationStateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_active_case_id(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> FieldNoteCaseId | None:
        row = await self.session.scalar(
            select(FieldNoteConversationState).where(
                FieldNoteConversationState.business_id == business_id,
                FieldNoteConversationState.conversation_id == conversation_id,
            )
        )
        return FieldNoteCaseId(row.active_case_id) if row and row.active_case_id else None

    async def set_active_case(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        case_id: FieldNoteCaseId,
        *,
        now: datetime,
    ) -> None:
        _aware(now, "now")
        conversation = await self.session.scalar(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        case = await self.session.scalar(
            select(FieldNoteCaseRow).where(FieldNoteCaseRow.id == case_id)
        )
        if (
            conversation is None
            or case is None
            or conversation.business_id != business_id
            or case.business_id != business_id
            or case.conversation_id != conversation_id
        ):
            raise CrossBusinessFieldNoteError("active field-note case is outside the business")
        row = await self.session.scalar(
            select(FieldNoteConversationState).where(
                FieldNoteConversationState.business_id == business_id,
                FieldNoteConversationState.conversation_id == conversation_id,
            )
        )
        if row is None:
            row = FieldNoteConversationState(
                business_id=business_id,
                conversation_id=conversation_id,
                active_case_id=case_id,
                updated_at=now,
            )
            try:
                async with self.session.begin_nested():
                    self.session.add(row)
                    await self.session.flush()
            except IntegrityError:
                row = await self.session.scalar(
                    select(FieldNoteConversationState).where(
                        FieldNoteConversationState.business_id == business_id,
                        FieldNoteConversationState.conversation_id == conversation_id,
                    )
                )
                if row is None:
                    raise
        if row is not None:
            row.active_case_id = case_id
            row.updated_at = now

    async def clear_active_case(
        self, business_id: BusinessId, conversation_id: ConversationId, *, now: datetime
    ) -> None:
        _aware(now, "now")
        await self.session.execute(
            update(FieldNoteConversationState)
            .where(
                FieldNoteConversationState.business_id == business_id,
                FieldNoteConversationState.conversation_id == conversation_id,
            )
            .values(active_case_id=None, updated_at=now)
        )


class SqlFieldNoteTranscriptionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def claim(
        self,
        business_id: BusinessId,
        part_id: FieldNotePartId,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> TranscriptionClaim:
        _aware(now, "now")
        _aware(stale_before, "stale_before")
        row = await self.session.scalar(
            select(FieldNotePartRow)
            .where(
                FieldNotePartRow.business_id == business_id,
                FieldNotePartRow.id == part_id,
            )
            .with_for_update()
        )
        if row is None or row.kind != FieldNotePartKind.AUDIO.value:
            return TranscriptionClaim(result=TranscriptionClaimResult.MISSING)
        if row.transcription_status == TranscriptionStatus.SUCCEEDED.value:
            return TranscriptionClaim(
                result=TranscriptionClaimResult.TERMINAL,
                part_id=FieldNotePartId(row.id),
                business_id=BusinessId(row.business_id),
            )
        leased_at = _with_utc(row.leased_at) if row.leased_at is not None else None
        if (
            row.transcription_status == TranscriptionStatus.IN_PROGRESS.value
            and leased_at is not None
            and leased_at >= stale_before
        ):
            return TranscriptionClaim(
                result=TranscriptionClaimResult.BUSY,
                part_id=FieldNotePartId(row.id),
                business_id=BusinessId(row.business_id),
            )
        attachment = _attachment(row)
        if attachment is None:
            return TranscriptionClaim(result=TranscriptionClaimResult.MISSING)
        row.transcription_status = TranscriptionStatus.IN_PROGRESS.value
        row.attempts += 1
        row.leased_at = now
        row.lease_token = uuid4()
        row.last_error = None
        return TranscriptionClaim(
            result=TranscriptionClaimResult.ACQUIRED,
            part_id=FieldNotePartId(row.id),
            business_id=BusinessId(row.business_id),
            audio=AudioReference(attachment=attachment, business_id=BusinessId(row.business_id)),
            attempts=row.attempts,
            lease_token=row.lease_token,
        )

    @staticmethod
    def _lease_filter(claim: TranscriptionClaim) -> tuple[ColumnElement[bool], ...]:
        if (
            claim.result is not TranscriptionClaimResult.ACQUIRED
            or claim.part_id is None
            or claim.business_id is None
            or claim.lease_token is None
        ):
            raise LostTranscriptionLeaseError("transcription claim does not hold an active lease")
        return (
            FieldNotePartRow.business_id == claim.business_id,
            FieldNotePartRow.id == claim.part_id,
            FieldNotePartRow.lease_token == claim.lease_token,
            FieldNotePartRow.transcription_status == TranscriptionStatus.IN_PROGRESS.value,
        )

    async def record_success(self, claim: TranscriptionClaim, transcript: TranscriptResult) -> None:
        result = await self.session.execute(
            update(FieldNotePartRow)
            .where(*self._lease_filter(claim))
            .values(
                transcription_status=TranscriptionStatus.SUCCEEDED.value,
                transcript_text=transcript.text,
                transcript_language=transcript.language,
                transcript_confidence=transcript.confidence,
                transcript_duration_seconds=transcript.duration_seconds,
                transcript_provider_ref=transcript.provider_ref,
                leased_at=None,
                last_error=None,
            )
        )
        if _rowcount(result) != 1:
            raise LostTranscriptionLeaseError("transcription claim is no longer active")

    async def record_failure(self, claim: TranscriptionClaim, error: str) -> None:
        result = await self.session.execute(
            update(FieldNotePartRow)
            .where(*self._lease_filter(claim))
            .values(
                transcription_status=TranscriptionStatus.FAILED.value,
                transcript_text=None,
                transcript_language=None,
                transcript_confidence=None,
                transcript_duration_seconds=None,
                transcript_provider_ref=None,
                leased_at=None,
                last_error=error,
            )
        )
        if _rowcount(result) != 1:
            raise LostTranscriptionLeaseError("transcription claim is no longer active")


class SqlFieldNoteUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SqlFieldNoteUnitOfWork":
        session: AsyncSession = self._session_factory()
        self._session = session
        self.field_note_cases: FieldNoteCaseRepository = SqlFieldNoteCaseRepository(session)
        self.field_note_messages: FieldNoteMessageLocator = SqlFieldNoteMessageLocator(session)
        self.field_note_conversation_states: FieldNoteConversationStateRepository = (
            SqlFieldNoteConversationStateRepository(session)
        )
        self.field_note_transcriptions: FieldNoteTranscriptionRepository = (
            SqlFieldNoteTranscriptionRepository(session)
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            await self._session.rollback()
        await self._session.close()

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.rollback()


class SqlFieldNoteUnitOfWorkFactory:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> FieldNoteUnitOfWork:
        return SqlFieldNoteUnitOfWork(self._session_factory)

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gvas.domain.completeness import (
    ActiveFieldNoteReviewExistsError,
    ChecklistItem,
    ChecklistItemKey,
    ChecklistKey,
    ChecklistVersionConflictError,
    CompletenessChecklist,
    CorrelatedAnswer,
    FieldNoteReviewId,
    FieldNoteReviewStatus,
    FollowUpQuestionId,
    FollowUpQuestionStatus,
    MissingChecklistItem,
    follow_up_correlation_id,
    transcript_fingerprint,
)
from gvas.domain.completeness_repositories import (
    FieldNoteReviewRecord,
    FollowUpQuestionRecord,
)
from gvas.domain.identifiers import BusinessId, ConversationId, JsonValue, MessageId
from gvas.domain.templates import TemplateSetKey, TemplateSetRef
from gvas.infrastructure.completeness_models import (
    FieldNoteChecklist,
    FieldNoteFollowUpQuestion,
    FieldNoteReview,
    FieldNoteReviewAnswer,
)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _review_record(row: FieldNoteReview) -> FieldNoteReviewRecord:
    return FieldNoteReviewRecord(
        review_id=FieldNoteReviewId(row.id),
        business_id=BusinessId(row.business_id),
        conversation_id=ConversationId(row.conversation_id),
        external_conversation_id=row.external_conversation_id,
        inbound_message_id=MessageId(row.inbound_message_id),
        checklist_key=ChecklistKey(row.checklist_key),
        checklist_version=row.checklist_version,
        template_set_key=(
            None if row.template_set_key is None else TemplateSetKey(row.template_set_key)
        ),
        template_set_version=row.template_set_version,
        transcript_text=row.transcript_text,
        transcript_fingerprint=row.transcript_fingerprint,
        revision=row.revision,
        thread_correlation_id=row.thread_correlation_id,
        status=FieldNoteReviewStatus(row.status),
        round_index=row.round_index,
    )


@dataclass(frozen=True)
class _ReviewPins:
    checklist_key: str
    checklist_version: int
    template_set_key: str
    template_set_version: int


def _pins(
    latest: FieldNoteReview | None,
    checklist_key: ChecklistKey,
    checklist_version: int,
    template_set: TemplateSetRef,
) -> _ReviewPins:
    """Pins for a review row: inherited from the case, else freshly resolved.

    A field-note case stays open until the owner closes it, so every review
    revision of that case reports from the template its first review pinned, even
    once a newer version has been published and become active.
    """
    if latest is not None and latest.template_set_key is not None:
        if latest.template_set_version is None:
            raise ValueError(f"review {latest.id} has a template key without a version")
        return _ReviewPins(
            checklist_key=latest.checklist_key,
            checklist_version=latest.checklist_version,
            template_set_key=latest.template_set_key,
            template_set_version=latest.template_set_version,
        )
    return _ReviewPins(
        checklist_key=checklist_key,
        checklist_version=checklist_version,
        template_set_key=template_set.template_set_key,
        template_set_version=template_set.version,
    )


def _question_record(row: FieldNoteFollowUpQuestion) -> FollowUpQuestionRecord:
    return FollowUpQuestionRecord(
        question_id=FollowUpQuestionId(row.id),
        business_id=BusinessId(row.business_id),
        review_id=FieldNoteReviewId(row.review_id),
        item_key=ChecklistItemKey(row.item_key),
        round_index=row.round_index,
        prompt=row.prompt,
        correlation_id=row.correlation_id,
        status=FollowUpQuestionStatus(row.status),
    )


class SqlChecklistDefinitionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(self, checklist: CompletenessChecklist) -> None:
        row = await self.session.scalar(
            select(FieldNoteChecklist).where(
                FieldNoteChecklist.business_id == checklist.business_id,
                FieldNoteChecklist.checklist_key == checklist.checklist_key,
                FieldNoteChecklist.version == checklist.version,
            )
        )
        items: list[JsonValue] = [item.model_dump(mode="json") for item in checklist.items]
        if row is None:
            self.session.add(
                FieldNoteChecklist(
                    business_id=checklist.business_id,
                    checklist_key=checklist.checklist_key,
                    version=checklist.version,
                    items=items,
                )
            )
        elif row.items != items:
            raise ChecklistVersionConflictError(
                f"{checklist.checklist_key} version {checklist.version} is immutable"
            )

    async def get(
        self,
        business_id: BusinessId,
        checklist_key: ChecklistKey,
        version: int | None = None,
    ) -> CompletenessChecklist | None:
        statement = select(FieldNoteChecklist).where(
            FieldNoteChecklist.business_id == business_id,
            FieldNoteChecklist.checklist_key == checklist_key,
        )
        if version is None:
            statement = statement.order_by(FieldNoteChecklist.version.desc()).limit(1)
        else:
            statement = statement.where(FieldNoteChecklist.version == version)
        row = await self.session.scalar(statement)
        if row is None:
            return None
        return CompletenessChecklist(
            business_id=BusinessId(row.business_id),
            checklist_key=ChecklistKey(row.checklist_key),
            version=row.version,
            items=tuple(ChecklistItem.model_validate(item) for item in row.items),
        )


class SqlFieldNoteReviewRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        external_conversation_id: str,
        inbound_message_id: MessageId,
        checklist_key: ChecklistKey,
        checklist_version: int,
        template_set: TemplateSetRef,
        transcript_text: str,
        thread_correlation_id: str,
    ) -> FieldNoteReviewRecord:
        fingerprint = transcript_fingerprint(transcript_text)
        existing = await self.session.scalar(
            select(FieldNoteReview).where(
                FieldNoteReview.business_id == business_id,
                FieldNoteReview.inbound_message_id == inbound_message_id,
                FieldNoteReview.transcript_fingerprint == fingerprint,
            )
        )
        if existing is not None:
            return _review_record(existing)
        latest = await self._latest(business_id, inbound_message_id, lock=True)
        if latest is not None and latest.status != FieldNoteReviewStatus.COMPLETE.value:
            return _review_record(latest)
        active = await self.session.scalar(
            select(FieldNoteReview).where(
                FieldNoteReview.business_id == business_id,
                FieldNoteReview.active_conversation_id == conversation_id,
            )
        )
        if active is not None:
            raise ActiveFieldNoteReviewExistsError(
                f"conversation {conversation_id} already has an active field-note review"
            )
        now = datetime.now(UTC)
        pins = _pins(latest, checklist_key, checklist_version, template_set)
        row = FieldNoteReview(
            business_id=business_id,
            conversation_id=conversation_id,
            active_conversation_id=conversation_id,
            external_conversation_id=external_conversation_id,
            inbound_message_id=inbound_message_id,
            checklist_key=pins.checklist_key,
            checklist_version=pins.checklist_version,
            template_set_key=pins.template_set_key,
            template_set_version=pins.template_set_version,
            transcript_text=transcript_text,
            transcript_fingerprint=fingerprint,
            revision=1 if latest is None else latest.revision + 1,
            thread_correlation_id=thread_correlation_id,
            status=FieldNoteReviewStatus.AWAITING_REVIEW.value,
            round_index=0,
            created_at=now,
            updated_at=now,
        )
        try:
            async with self.session.begin_nested():
                self.session.add(row)
                await self.session.flush()
        except IntegrityError:
            concurrent = await self.session.scalar(
                select(FieldNoteReview).where(
                    FieldNoteReview.business_id == business_id,
                    FieldNoteReview.inbound_message_id == inbound_message_id,
                    FieldNoteReview.transcript_fingerprint == fingerprint,
                )
            )
            if concurrent is None:
                raise
            return _review_record(concurrent)
        return _review_record(row)

    async def get(
        self, business_id: BusinessId, review_id: FieldNoteReviewId
    ) -> FieldNoteReviewRecord | None:
        row = await self.session.scalar(
            select(FieldNoteReview).where(
                FieldNoteReview.id == review_id,
                FieldNoteReview.business_id == business_id,
            )
        )
        return _review_record(row) if row is not None else None

    async def latest_for_origin(
        self, business_id: BusinessId, inbound_message_id: MessageId
    ) -> FieldNoteReviewRecord | None:
        row = await self._latest(business_id, inbound_message_id)
        return None if row is None else _review_record(row)

    async def _latest(
        self, business_id: BusinessId, inbound_message_id: MessageId, *, lock: bool = False
    ) -> FieldNoteReview | None:
        statement = (
            select(FieldNoteReview)
            .where(
                FieldNoteReview.business_id == business_id,
                FieldNoteReview.inbound_message_id == inbound_message_id,
            )
            .order_by(FieldNoteReview.revision.desc())
            .limit(1)
        )
        row: FieldNoteReview | None = await self.session.scalar(
            statement.with_for_update() if lock else statement
        )
        return row

    async def get_active_for_conversation(
        self, business_id: BusinessId, conversation_id: ConversationId
    ) -> FieldNoteReviewRecord | None:
        row = await self.session.scalar(
            select(FieldNoteReview).where(
                FieldNoteReview.business_id == business_id,
                FieldNoteReview.active_conversation_id == conversation_id,
            )
        )
        return _review_record(row) if row is not None else None

    async def begin_round(self, review: FieldNoteReviewRecord) -> FieldNoteReviewRecord:
        await self.session.execute(
            update(FieldNoteReview)
            .where(
                FieldNoteReview.id == review.review_id,
                FieldNoteReview.business_id == review.business_id,
                FieldNoteReview.status != FieldNoteReviewStatus.COMPLETE.value,
            )
            .values(
                status=FieldNoteReviewStatus.AWAITING_ANSWERS.value,
                round_index=FieldNoteReview.round_index + 1,
                updated_at=datetime.now(UTC),
            )
        )
        row = await self.session.scalar(
            select(FieldNoteReview).where(
                FieldNoteReview.id == review.review_id,
                FieldNoteReview.business_id == review.business_id,
            )
        )
        if row is None:
            raise LookupError("field-note review not found")
        return _review_record(row)

    async def complete(self, review: FieldNoteReviewRecord) -> FieldNoteReviewRecord:
        await self.session.execute(
            update(FieldNoteReview)
            .where(
                FieldNoteReview.id == review.review_id,
                FieldNoteReview.business_id == review.business_id,
            )
            .values(
                status=FieldNoteReviewStatus.COMPLETE.value,
                active_conversation_id=None,
                updated_at=datetime.now(UTC),
            )
        )
        row = await self.session.scalar(
            select(FieldNoteReview).where(
                FieldNoteReview.id == review.review_id,
                FieldNoteReview.business_id == review.business_id,
            )
        )
        if row is None:
            raise LookupError("field-note review not found")
        return _review_record(row)


class SqlFollowUpQuestionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create_many(
        self, review: FieldNoteReviewRecord, missing_items: tuple[MissingChecklistItem, ...]
    ) -> tuple[FollowUpQuestionRecord, ...]:
        existing = {
            (row.round_index, row.item_key): row
            for row in (
                await self.session.scalars(
                    select(FieldNoteFollowUpQuestion).where(
                        FieldNoteFollowUpQuestion.business_id == review.business_id,
                        FieldNoteFollowUpQuestion.review_id == review.review_id,
                    )
                )
            ).all()
        }
        rows: list[FieldNoteFollowUpQuestion] = []
        for item in missing_items:
            key = (review.round_index, item.item_key)
            row = existing.get(key)
            if row is None:
                row = FieldNoteFollowUpQuestion(
                    business_id=review.business_id,
                    review_id=review.review_id,
                    item_key=item.item_key,
                    round_index=review.round_index,
                    prompt=item.prompt,
                    correlation_id=follow_up_correlation_id(
                        review.review_id, review.round_index, item.item_key
                    ),
                    status=FollowUpQuestionStatus.PENDING.value,
                )
                self.session.add(row)
                await self.session.flush()
            rows.append(row)
        return tuple(_question_record(row) for row in rows)

    async def list_for_review(
        self, business_id: BusinessId, review_id: FieldNoteReviewId
    ) -> tuple[FollowUpQuestionRecord, ...]:
        rows = (
            await self.session.scalars(
                select(FieldNoteFollowUpQuestion)
                .where(
                    FieldNoteFollowUpQuestion.business_id == business_id,
                    FieldNoteFollowUpQuestion.review_id == review_id,
                )
                .order_by(
                    FieldNoteFollowUpQuestion.round_index,
                    FieldNoteFollowUpQuestion.item_key,
                )
            )
        ).all()
        return tuple(_question_record(row) for row in rows)

    async def mark_asked(self, question: FollowUpQuestionRecord) -> None:
        await self.session.execute(
            update(FieldNoteFollowUpQuestion)
            .where(
                FieldNoteFollowUpQuestion.id == question.question_id,
                FieldNoteFollowUpQuestion.business_id == question.business_id,
                FieldNoteFollowUpQuestion.status == FollowUpQuestionStatus.PENDING.value,
            )
            .values(status=FollowUpQuestionStatus.ASKED.value, asked_at=datetime.now(UTC))
        )

    async def record_answer(
        self,
        question: FollowUpQuestionRecord,
        inbound_message_id: MessageId,
        text: str,
        received_at: datetime,
    ) -> bool:
        existing = await self.session.scalar(
            select(FieldNoteReviewAnswer).where(
                FieldNoteReviewAnswer.business_id == question.business_id,
                FieldNoteReviewAnswer.review_id == question.review_id,
                FieldNoteReviewAnswer.inbound_message_id == inbound_message_id,
            )
        )
        if existing is not None:
            return False
        try:
            async with self.session.begin_nested():
                self.session.add(
                    FieldNoteReviewAnswer(
                        business_id=question.business_id,
                        review_id=question.review_id,
                        question_id=question.question_id,
                        inbound_message_id=inbound_message_id,
                        item_key=question.item_key,
                        text=text,
                        received_at=received_at,
                    )
                )
                await self.session.flush()
        except IntegrityError:
            return False
        await self.session.execute(
            update(FieldNoteFollowUpQuestion)
            .where(
                FieldNoteFollowUpQuestion.id == question.question_id,
                FieldNoteFollowUpQuestion.business_id == question.business_id,
                FieldNoteFollowUpQuestion.status == FollowUpQuestionStatus.ASKED.value,
            )
            .values(status=FollowUpQuestionStatus.ANSWERED.value)
        )
        return True

    async def answer_exists_for_inbound(
        self,
        business_id: BusinessId,
        review_id: FieldNoteReviewId,
        inbound_message_id: MessageId,
    ) -> bool:
        row = await self.session.scalar(
            select(FieldNoteReviewAnswer.id).where(
                FieldNoteReviewAnswer.business_id == business_id,
                FieldNoteReviewAnswer.review_id == review_id,
                FieldNoteReviewAnswer.inbound_message_id == inbound_message_id,
            )
        )
        return row is not None

    async def answers_for_review(
        self, business_id: BusinessId, review_id: FieldNoteReviewId
    ) -> tuple[CorrelatedAnswer, ...]:
        rows = (
            await self.session.scalars(
                select(FieldNoteReviewAnswer)
                .where(
                    FieldNoteReviewAnswer.business_id == business_id,
                    FieldNoteReviewAnswer.review_id == review_id,
                )
                .order_by(FieldNoteReviewAnswer.received_at, FieldNoteReviewAnswer.item_key)
            )
        ).all()
        return tuple(
            CorrelatedAnswer(
                item_key=ChecklistItemKey(row.item_key),
                text=row.text,
                received_at=_as_utc(row.received_at),
            )
            for row in rows
        )

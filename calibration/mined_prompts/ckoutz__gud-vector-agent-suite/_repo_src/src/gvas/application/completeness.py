from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from gvas.domain.completeness import (
    ChecklistItemKey,
    CompletenessChecklist,
    CompletenessReviewOutcome,
    CompletenessReviewPort,
    CompletenessReviewRequest,
    FieldNoteReviewId,
    FieldNoteReviewStatus,
    FollowUpQuestionStatus,
    InvalidCompletenessReviewOutcomeError,
    MissingItemReason,
    UnknownChecklistError,
    UnknownChecklistItemError,
    field_note_thread_correlation_id,
)
from gvas.domain.completeness_repositories import (
    CompletenessUnitOfWork,
    FieldNoteReviewRecord,
    FollowUpQuestionRecord,
)
from gvas.domain.identifiers import BusinessId, ConversationId, MessageId
from gvas.domain.messages import (
    ConversationRef,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    ReplyRef,
    TextPart,
)
from gvas.domain.outbox import owner_reply_command
from gvas.domain.templates import TemplateResolutionPort, TemplateSetRef


class CompletenessStatus(StrEnum):
    QUESTIONS_SENT = "questions_sent"
    AWAITING_ANSWERS = "awaiting_answers"
    COMPLETE = "complete"
    ALREADY_COMPLETE = "already_complete"
    NO_ACTIVE_REVIEW = "no_active_review"
    DUPLICATE_REPLY = "duplicate_reply"
    UNCORRELATED_REPLY = "uncorrelated_reply"


@dataclass(frozen=True)
class CompletenessOutcome:
    status: CompletenessStatus
    review_id: FieldNoteReviewId | None = None
    round_index: int | None = None
    missing_item_keys: tuple[ChecklistItemKey, ...] = ()
    questions_sent: int = 0
    detail: str | None = None


class CompletenessUnitOfWorkFactory(Protocol):
    def __call__(self) -> CompletenessUnitOfWork: ...


def _answer_text(message: NormalizedOwnerMessage) -> str:
    return "\n".join(
        part.text.strip()
        for part in message.parts
        if isinstance(part, TextPart) and part.text.strip()
    )


class FieldNoteCompletenessService:
    """Channel-neutral completeness review loop.

    The review port is invoked with no unit of work open. Follow-up messages and
    owner-reply outbox commands are persisted atomically with deterministic
    correlation IDs, and replies follow the review's persisted thread state.

    The template set is resolved once, when the review row is created, and pinned
    on it. Every later round reads the pinned checklist, so publishing a new
    template version cannot change a case already under review.
    """

    def __init__(
        self,
        unit_of_work_factory: CompletenessUnitOfWorkFactory,
        review_port: CompletenessReviewPort,
        templates: TemplateResolutionPort,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._review_port = review_port
        self._templates = templates

    async def _pin(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        inbound_message_id: MessageId,
    ) -> TemplateSetRef:
        """The case's existing pin, or a freshly resolved one for a first review.

        A case stays open until the owner closes it, so a later transcript
        revision must keep reporting from the template its first review pinned,
        even once that version is deprecated and no active version resolves.
        """
        async with self._unit_of_work_factory() as unit_of_work:
            latest = await unit_of_work.field_note_reviews.latest_for_origin(
                business_id, inbound_message_id
            )
            await unit_of_work.commit()
        pinned = None if latest is None else latest.template_set
        if pinned is not None:
            return pinned
        return await self._templates.resolve_for_new_case(business_id, conversation_id)

    async def start_review(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        external_conversation_id: str,
        inbound_message_id: MessageId,
        transcript_text: str,
    ) -> CompletenessOutcome:
        resolved = await self._pin(business_id, conversation_id, inbound_message_id)
        template_set = await self._templates.load(resolved)
        async with self._unit_of_work_factory() as unit_of_work:
            review = await unit_of_work.field_note_reviews.get_or_create(
                business_id,
                conversation_id,
                external_conversation_id,
                inbound_message_id,
                template_set.checklist_key,
                template_set.checklist_version,
                resolved,
                transcript_text,
                field_note_thread_correlation_id(inbound_message_id),
            )
            checklist = await unit_of_work.checklists.get(
                business_id, review.checklist_key, review.checklist_version
            )
            if checklist is None:
                await unit_of_work.rollback()
                raise UnknownChecklistError(f"{review.checklist_key} is not configured")
            questions = await unit_of_work.follow_up_questions.list_for_review(
                business_id, review.review_id
            )
            await unit_of_work.commit()

        if review.status is FieldNoteReviewStatus.COMPLETE:
            return CompletenessOutcome(
                CompletenessStatus.ALREADY_COMPLETE,
                review_id=review.review_id,
                round_index=review.round_index,
            )
        outstanding = _outstanding(questions)
        if outstanding:
            sent = await self._enqueue_questions(review, outstanding)
            return CompletenessOutcome(
                CompletenessStatus.QUESTIONS_SENT if sent else CompletenessStatus.AWAITING_ANSWERS,
                review_id=review.review_id,
                round_index=review.round_index,
                missing_item_keys=tuple(question.item_key for question in outstanding),
                questions_sent=sent,
            )
        return await self._run_round(review, checklist)

    async def record_owner_reply(
        self,
        business_id: BusinessId,
        conversation_id: ConversationId,
        inbound_message_id: MessageId,
        message: NormalizedOwnerMessage,
    ) -> CompletenessOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            review = await unit_of_work.field_note_reviews.get_active_for_conversation(
                business_id, conversation_id
            )
            if review is None:
                await unit_of_work.rollback()
                return CompletenessOutcome(CompletenessStatus.NO_ACTIVE_REVIEW)
            duplicate = await unit_of_work.follow_up_questions.answer_exists_for_inbound(
                business_id, review.review_id, inbound_message_id
            )
            if duplicate:
                questions = await unit_of_work.follow_up_questions.list_for_review(
                    business_id, review.review_id
                )
                checklist = await unit_of_work.checklists.get(
                    business_id, review.checklist_key, review.checklist_version
                )
                if checklist is None:
                    await unit_of_work.rollback()
                    raise UnknownChecklistError(f"{review.checklist_key} is not configured")
                await unit_of_work.commit()
                outstanding = _outstanding(questions)
                if outstanding:
                    sent = await self._enqueue_questions(review, outstanding)
                    return CompletenessOutcome(
                        CompletenessStatus.QUESTIONS_SENT
                        if sent
                        else CompletenessStatus.DUPLICATE_REPLY,
                        review_id=review.review_id,
                        round_index=review.round_index,
                        missing_item_keys=tuple(question.item_key for question in outstanding),
                        questions_sent=sent,
                    )
                return await self._run_round(review, checklist)
            if (
                message.business_id != business_id
                or message.conversation_ref.external_conversation_id
                != review.external_conversation_id
            ):
                await unit_of_work.rollback()
                return CompletenessOutcome(
                    CompletenessStatus.UNCORRELATED_REPLY,
                    review_id=review.review_id,
                    round_index=review.round_index,
                    detail="reply does not belong to the active field-note conversation",
                )
            answer_text = _answer_text(message)
            if not answer_text:
                await unit_of_work.rollback()
                return CompletenessOutcome(
                    CompletenessStatus.UNCORRELATED_REPLY,
                    review_id=review.review_id,
                    round_index=review.round_index,
                    detail="reply contains no text answer",
                )
            questions = await unit_of_work.follow_up_questions.list_for_review(
                business_id, review.review_id
            )
            target = self._select_question(questions)
            if target is None:
                await unit_of_work.rollback()
                return CompletenessOutcome(
                    CompletenessStatus.UNCORRELATED_REPLY,
                    review_id=review.review_id,
                    round_index=review.round_index,
                    detail="no single follow-up question is currently awaiting an answer",
                )
            recorded = await unit_of_work.follow_up_questions.record_answer(
                target, inbound_message_id, answer_text, message.received_at
            )
            if not recorded:
                await unit_of_work.rollback()
                return CompletenessOutcome(
                    CompletenessStatus.DUPLICATE_REPLY,
                    review_id=review.review_id,
                    round_index=review.round_index,
                )
            remaining = tuple(
                question
                for question in questions
                if question.question_id != target.question_id
                and question.status is not FollowUpQuestionStatus.ANSWERED
            )
            checklist = await unit_of_work.checklists.get(
                business_id, review.checklist_key, review.checklist_version
            )
            if checklist is None:
                await unit_of_work.rollback()
                raise UnknownChecklistError(f"{review.checklist_key} is not configured")
            await unit_of_work.commit()

        if remaining:
            sent = await self._enqueue_questions(review, remaining)
            return CompletenessOutcome(
                CompletenessStatus.QUESTIONS_SENT if sent else CompletenessStatus.AWAITING_ANSWERS,
                review_id=review.review_id,
                round_index=review.round_index,
                missing_item_keys=tuple(question.item_key for question in remaining),
                questions_sent=sent,
            )
        return await self._run_round(review, checklist)

    def _select_question(
        self, questions: tuple[FollowUpQuestionRecord, ...]
    ) -> FollowUpQuestionRecord | None:
        asked = _outstanding(questions, asked_only=True)
        return asked[0] if len(asked) == 1 else None

    async def _run_round(
        self, review: FieldNoteReviewRecord, checklist: CompletenessChecklist
    ) -> CompletenessOutcome:
        async with self._unit_of_work_factory() as unit_of_work:
            answers = await unit_of_work.follow_up_questions.answers_for_review(
                review.business_id, review.review_id
            )
            await unit_of_work.commit()

        outcome = await self._review_port.review(
            CompletenessReviewRequest(
                business_id=review.business_id,
                checklist=checklist,
                transcript_text=review.transcript_text,
                answers=answers,
                round_index=review.round_index,
            )
        )
        _validate_outcome(outcome, checklist)

        async with self._unit_of_work_factory() as unit_of_work:
            if outcome.is_complete:
                completed = await unit_of_work.field_note_reviews.complete(review)
                await unit_of_work.commit()
                return CompletenessOutcome(
                    CompletenessStatus.COMPLETE,
                    review_id=completed.review_id,
                    round_index=completed.round_index,
                    detail=outcome.detail,
                )
            next_round = await unit_of_work.field_note_reviews.begin_round(review)
            questions = await unit_of_work.follow_up_questions.get_or_create_many(
                next_round, outcome.missing_items
            )
            await unit_of_work.commit()

        sent = await self._enqueue_questions(next_round, _outstanding(questions))
        return CompletenessOutcome(
            CompletenessStatus.QUESTIONS_SENT,
            review_id=next_round.review_id,
            round_index=next_round.round_index,
            missing_item_keys=tuple(item.item_key for item in outcome.missing_items),
            questions_sent=sent,
            detail=outcome.detail,
        )

    async def _enqueue_questions(
        self, review: FieldNoteReviewRecord, questions: tuple[FollowUpQuestionRecord, ...]
    ) -> int:
        if any(question.status is FollowUpQuestionStatus.ASKED for question in questions):
            return 0
        question = next(
            (
                question
                for question in sorted(
                    questions, key=lambda item: (item.round_index, item.item_key)
                )
                if question.status is FollowUpQuestionStatus.PENDING
            ),
            None,
        )
        if question is None:
            return 0
        conversation_ref = ConversationRef(
            business_id=review.business_id,
            external_conversation_id=review.external_conversation_id,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            message = OutboundOwnerMessage(
                business_id=review.business_id,
                conversation_ref=conversation_ref,
                parts=(TextPart(text=question.prompt),),
                correlation_id=question.correlation_id,
                reply_to=ReplyRef(correlation_id=review.thread_correlation_id),
            )
            outbound_message_id = await unit_of_work.outbound_messages.create(
                message, review.conversation_id, review.inbound_message_id
            )
            await unit_of_work.outbox.enqueue(
                owner_reply_command(review.business_id, outbound_message_id)
            )
            await unit_of_work.follow_up_questions.mark_asked(question)
            await unit_of_work.commit()
        return 1


def _outstanding(
    questions: tuple[FollowUpQuestionRecord, ...], *, asked_only: bool = False
) -> tuple[FollowUpQuestionRecord, ...]:
    wanted = (
        (FollowUpQuestionStatus.ASKED,)
        if asked_only
        else (FollowUpQuestionStatus.PENDING, FollowUpQuestionStatus.ASKED)
    )
    return tuple(
        question
        for question in sorted(questions, key=lambda item: (item.round_index, item.item_key))
        if question.status in wanted
    )


def _validate_outcome(outcome: CompletenessReviewOutcome, checklist: CompletenessChecklist) -> None:
    unknown = [
        item.item_key for item in outcome.missing_items if checklist.item(item.item_key) is None
    ]
    if unknown:
        raise UnknownChecklistItemError(
            f"review reported items outside checklist {checklist.checklist_key}: {unknown}"
        )
    invalid: list[ChecklistItemKey] = []
    for item in outcome.missing_items:
        configured = checklist.item(item.item_key)
        if configured is None or item.reason is MissingItemReason.CONTRADICTION:
            continue
        if item.item_key not in checklist.required_item_keys or item.prompt != configured.prompt:
            invalid.append(item.item_key)
    if invalid:
        raise InvalidCompletenessReviewOutcomeError(
            f"review reported invalid missing items for {checklist.checklist_key}: {invalid}"
        )

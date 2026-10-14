from datetime import datetime
from typing import Protocol

from gvas.domain.completeness import (
    FieldNoteReviewId,
    FieldNoteReviewStatus,
    UnknownChecklistError,
)
from gvas.domain.completeness_repositories import CompletenessUnitOfWork
from gvas.domain.field_notes import FieldNoteCaseId, FieldNoteCaseNotFoundError
from gvas.domain.identifiers import BusinessId
from gvas.domain.ports import ChecklistEvidencePort
from gvas.domain.reporting import (
    ChecklistEvidenceRequest,
    CorrelatedAnswer,
    FieldNoteCaseSnapshot,
    FieldNoteCaseStatus,
)
from gvas.domain.templates import TemplateResolutionPort


class IncompleteFieldNoteReviewError(ValueError):
    pass


class UnknownFieldNoteReviewError(LookupError):
    pass


class UnattributedChecklistEvidenceError(ValueError):
    pass


class CompletenessUnitOfWorkFactory(Protocol):
    def __call__(self) -> CompletenessUnitOfWork: ...


class BuildFieldNoteCaseSnapshotService:
    """Assembles the report source for a completed field-note review.

    The source is the transcript the review actually ran against, not the live
    canonical transcript, so a retried report command cannot pick up notes that
    arrived after the review and change the snapshot fingerprint. Checklist
    evidence is attributed through a provider-neutral port because the accepted
    completeness outcome only reports items that are still missing. The report
    template pin comes from the review row, so a report regenerated after a newer
    template version is published still renders from the pinned version.
    """

    def __init__(
        self,
        completeness_unit_of_work_factory: CompletenessUnitOfWorkFactory,
        evidence: ChecklistEvidencePort,
        templates: TemplateResolutionPort,
    ) -> None:
        self._completeness = completeness_unit_of_work_factory
        self._evidence = evidence
        self._templates = templates

    async def build(
        self,
        business_id: BusinessId,
        case_id: FieldNoteCaseId,
        review_id: FieldNoteReviewId,
        *,
        completed_at: datetime,
    ) -> FieldNoteCaseSnapshot:
        async with self._completeness() as unit_of_work:
            review = await unit_of_work.field_note_reviews.get(business_id, review_id)
            if review is None:
                await unit_of_work.rollback()
                raise UnknownFieldNoteReviewError("field-note review was not found")
            checklist = await unit_of_work.checklists.get(
                business_id, review.checklist_key, review.checklist_version
            )
            if checklist is None:
                await unit_of_work.rollback()
                raise UnknownChecklistError(f"{review.checklist_key} is not configured")
            answers = await unit_of_work.follow_up_questions.answers_for_review(
                business_id, review_id
            )
            await unit_of_work.commit()
        if review.status is not FieldNoteReviewStatus.COMPLETE:
            raise IncompleteFieldNoteReviewError("only completed reviews produce report snapshots")

        pinned = review.template_set
        template_set = None if pinned is None else await self._templates.load(pinned)

        transcript_text = review.transcript_text
        if not transcript_text:
            raise FieldNoteCaseNotFoundError("field-note case has no canonical transcript")
        prompts = {item.key: item.prompt for item in checklist.items}
        correlated = tuple(
            CorrelatedAnswer(
                question_key=answer.item_key,
                question=prompts.get(answer.item_key, answer.item_key),
                answer=answer.text,
            )
            for answer in answers
        )
        evidence = await self._evidence.attribute(
            ChecklistEvidenceRequest(
                business_id=business_id,
                case_id=case_id,
                checklist=checklist,
                canonical_transcript=transcript_text,
                correlated_answers=correlated,
            )
        )
        unknown = sorted(item.item_key for item in evidence if item.item_key not in prompts)
        if unknown:
            raise UnattributedChecklistEvidenceError(
                f"evidence reported items outside checklist {review.checklist_key}: {unknown}"
            )
        return FieldNoteCaseSnapshot(
            business_id=business_id,
            case_id=case_id,
            status=FieldNoteCaseStatus.COMPLETED,
            completed_at=completed_at,
            canonical_transcript=transcript_text,
            checklist_evidence=evidence,
            correlated_answers=correlated,
            report_template_key=None if template_set is None else template_set.report_template_key,
            report_template_version=(
                None if template_set is None else template_set.report_template_version
            ),
        )

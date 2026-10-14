from gvas.domain.completeness import (
    CompletenessReviewOutcome,
    CompletenessReviewPort,
    CompletenessReviewRequest,
    ContradictionGuardPort,
    MissingChecklistItem,
    MissingItemReason,
)


class GuardedCompletenessReviewer:
    """Runs a focused contradiction pass before a review may complete.

    The primary reviewer decides which checklist items are still missing; while
    any are, its outcome is returned unchanged so critical gaps are asked about
    first. Only once it reports the note complete does the guard run, and a hard
    contradiction it finds becomes exactly one follow-up question. Contradictions
    on an item the owner has already answered are not re-asked, so a resolved
    conflict cannot loop, and contradictions on items outside the pinned
    checklist are dropped rather than trusted.
    """

    def __init__(self, primary: CompletenessReviewPort, guard: ContradictionGuardPort) -> None:
        self._primary = primary
        self._guard = guard

    async def review(self, request: CompletenessReviewRequest) -> CompletenessReviewOutcome:
        outcome = await self._primary.review(request)
        if not outcome.is_complete:
            return outcome
        guard = await self._guard.detect(request)
        answered = {answer.item_key for answer in request.answers}
        raised = next(
            (
                contradiction
                for contradiction in guard.contradictions
                if request.checklist.item(contradiction.item_key) is not None
                and contradiction.item_key not in answered
            ),
            None,
        )
        if raised is None:
            return CompletenessReviewOutcome(detail=outcome.detail or guard.detail)
        return CompletenessReviewOutcome(
            missing_items=(
                MissingChecklistItem(
                    item_key=raised.item_key,
                    prompt=raised.question,
                    detail=raised.detail,
                    reason=MissingItemReason.CONTRADICTION,
                ),
            ),
            detail=guard.detail,
        )

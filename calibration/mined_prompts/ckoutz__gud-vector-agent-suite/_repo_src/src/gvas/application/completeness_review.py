from gvas.domain.completeness import (
    ChecklistItemRequirement,
    CompletenessReviewOutcome,
    CompletenessReviewRequest,
    MissingChecklistItem,
)


class MarkerCompletenessReviewer:
    """Deterministic reviewer used until a review provider is selected.

    An item counts as satisfied when a correlated answer exists for it or when any
    of its configured evidence markers appears in the transcript. Markers come from
    the checklist definition, so no operational policy is encoded here.
    """

    async def review(self, request: CompletenessReviewRequest) -> CompletenessReviewOutcome:
        answered = {answer.item_key for answer in request.answers}
        transcript = request.transcript_text.casefold()
        missing = tuple(
            MissingChecklistItem(item_key=item.key, prompt=item.prompt)
            for item in request.checklist.items
            if item.requirement is ChecklistItemRequirement.REQUIRED
            and item.key not in answered
            and not any(marker.casefold() in transcript for marker in item.evidence_markers)
        )
        return CompletenessReviewOutcome(missing_items=missing)

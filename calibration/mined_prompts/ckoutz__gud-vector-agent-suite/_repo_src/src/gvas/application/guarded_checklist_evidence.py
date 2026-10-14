import logging

from gvas.domain.ports import ChecklistEvidenceAnnotatorPort, ChecklistEvidencePort
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistEvidenceAnnotation,
    ChecklistEvidenceRequest,
    ChecklistOutcome,
)

logger = logging.getLogger(__name__)


def verbatim_excerpts(transcript: str, excerpts: tuple[str, ...]) -> tuple[str, ...]:
    """Keep only excerpts that appear verbatim in the transcript, in the order given.

    An excerpt is trusted only if the exact text is a substring of the note; a
    paraphrase, a merged span, or an invented value is dropped without being
    reported as evidence. Duplicates collapse to the first occurrence.
    """

    return tuple(
        dict.fromkeys(
            stripped
            for excerpt in excerpts
            if (stripped := excerpt.strip()) and stripped in transcript
        )
    )


class GuardedChecklistEvidenceAttributor:
    """Deterministic-first evidence attribution with a model-annotated evidence layer.

    The primary attributor decides which checklist items are satisfied and its
    outcomes are returned unchanged. The annotator may only add supporting
    excerpts to items the primary already marked observed, and each excerpt is
    checked against the transcript here rather than trusted. Any annotator
    failure is logged and the primary evidence is returned as-is, so a model
    outage can never fail a report.
    """

    def __init__(
        self, primary: ChecklistEvidencePort, annotator: ChecklistEvidenceAnnotatorPort
    ) -> None:
        self._primary = primary
        self._annotator = annotator

    async def attribute(self, request: ChecklistEvidenceRequest) -> tuple[ChecklistEvidence, ...]:
        attributed = await self._primary.attribute(request)
        try:
            annotations = await self._annotator.annotate(request, attributed)
        except Exception:
            logger.warning(
                "checklist evidence annotation failed; using marker evidence only",
                extra={"case_id": str(request.case_id)},
                exc_info=True,
            )
            return attributed
        return merge_annotations(request.canonical_transcript, attributed, annotations)


def merge_annotations(
    transcript: str,
    attributed: tuple[ChecklistEvidence, ...],
    annotations: tuple[ChecklistEvidenceAnnotation, ...],
) -> tuple[ChecklistEvidence, ...]:
    excerpts_by_item: dict[str, tuple[str, ...]] = {}
    for annotation in annotations:
        excerpts_by_item.setdefault(
            annotation.item_key, verbatim_excerpts(transcript, annotation.excerpts)
        )
    merged: list[ChecklistEvidence] = []
    for item in attributed:
        excerpts = excerpts_by_item.get(item.item_key, ())
        if item.outcome is not ChecklistOutcome.OBSERVED or not excerpts:
            merged.append(item)
            continue
        evidence = tuple(dict.fromkeys((*item.evidence, *excerpts)))
        merged.append(item.model_copy(update={"evidence": evidence}))
    return tuple(merged)

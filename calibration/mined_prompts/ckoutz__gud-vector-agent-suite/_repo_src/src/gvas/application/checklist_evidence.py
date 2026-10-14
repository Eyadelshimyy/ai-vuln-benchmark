import re

from gvas.domain.completeness import ChecklistItemRequirement
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistEvidenceRequest,
    ChecklistOutcome,
)

#: Sentence boundaries need trailing whitespace, so "3.5 ppm" stays one segment.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _segments(transcript: str) -> tuple[str, ...]:
    """Split the canonical transcript into the units evidence is quoted from."""

    parts = (
        segment.strip()
        for line in transcript.splitlines()
        for segment in _SENTENCE_BOUNDARY.split(line)
    )
    return tuple(part for part in parts if part)


def _quoted_evidence(markers: tuple[str, ...], segments: tuple[str, ...]) -> tuple[str, ...]:
    """Quote the transcript text around each marker, not the marker label.

    A report that cites ``site:`` tells the reader nothing; the segment that
    contains it carries the value the owner dictated. Every matching segment is
    quoted, because a note dictated across several messages repeats a marker
    per observation and stopping at the first one would drop the rest. Segments
    are quoted verbatim in transcript order and deduplicated, so nothing is
    invented and a segment matched by two markers is not repeated.
    """

    folded_markers = tuple(marker.casefold() for marker in markers)
    return tuple(
        dict.fromkeys(
            segment
            for segment in segments
            if any(marker in segment.casefold() for marker in folded_markers)
        )
    )


class MarkerChecklistEvidenceAttributor:
    """Deterministic evidence attribution used until a review provider is selected.

    Evidence comes from the correlated answer for an item, otherwise from the
    transcript text around the configured evidence markers. Markers are
    checklist configuration, so no operational policy is encoded here.
    """

    async def attribute(self, request: ChecklistEvidenceRequest) -> tuple[ChecklistEvidence, ...]:
        answers = {answer.question_key: answer.answer for answer in request.correlated_answers}
        transcript = request.canonical_transcript.casefold()
        segments = _segments(request.canonical_transcript)
        attributed: list[ChecklistEvidence] = []
        for item in request.checklist.items:
            answer = answers.get(item.key)
            markers = _quoted_evidence(
                tuple(
                    marker for marker in item.evidence_markers if marker.casefold() in transcript
                ),
                segments,
            )
            evidence: tuple[str, ...]
            if answer is not None:
                outcome = ChecklistOutcome.OBSERVED
                evidence = (answer,)
            elif markers:
                outcome = ChecklistOutcome.OBSERVED
                evidence = markers
            else:
                outcome = (
                    ChecklistOutcome.NOT_OBSERVED
                    if item.requirement is ChecklistItemRequirement.REQUIRED
                    else ChecklistOutcome.NOT_APPLICABLE
                )
                evidence = ()
            attributed.append(
                ChecklistEvidence(
                    item_key=item.key,
                    prompt=item.prompt,
                    outcome=outcome,
                    evidence=evidence,
                )
            )
        return tuple(attributed)

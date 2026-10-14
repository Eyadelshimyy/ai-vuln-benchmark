"""Deterministic field-notes report generation and rendering.

No model is involved. The report is a rearrangement of what was already
persisted: the pinned template supplies the title, section order and headings,
and each section reproduces the evidence bound to it by checklist item key. The
generator invents nothing, so a report is reproducible from its snapshot.

``ReportGenerationPort`` is the seam an inference-backed generator would take
over later; the pinned-template and evidence validation in
``FieldNotesReportDocument`` applies to any implementation.
"""

from typing import Final

from gvas.domain.identifiers import JsonValue
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistOutcome,
    CorrelatedAnswer,
    EvidenceSource,
    FieldNotesReportDocument,
    FieldNotesReportVersion,
    ReportBlock,
    ReportEvidenceReference,
    ReportGenerationRequest,
    ReportSection,
)
from gvas.domain.templates import ReportTemplateSection

EMPTY_SECTION_TEXT: Final = "No evidence was recorded for this section."

#: An item backed by quoted evidence needs no status word; the evidence is the
#: statement. Only the absence of evidence has to be said out loud, so the
#: reader can tell a gap from an item that does not apply to the job.
OUTCOME_LABELS: Final[dict[ChecklistOutcome, str]] = {
    ChecklistOutcome.NOT_OBSERVED: "not recorded",
    ChecklistOutcome.NOT_APPLICABLE: "not applicable",
}


class DeterministicReportGenerator:
    """Renders the pinned template from persisted evidence only."""

    async def generate(self, request: ReportGenerationRequest) -> dict[str, JsonValue]:
        evidence = {item.item_key: item for item in request.source.checklist_evidence}
        answers = {answer.question_key: answer for answer in request.source.correlated_answers}
        sections = tuple(
            _section(configured, evidence, answers)
            for configured in request.report_template.sections
        )
        document = FieldNotesReportDocument(title=request.report_template.title, sections=sections)
        return document.model_dump(mode="json")


def _section(
    configured: ReportTemplateSection,
    evidence: dict[str, ChecklistEvidence],
    answers: dict[str, CorrelatedAnswer],
) -> ReportSection:
    blocks = [
        block
        for key in configured.checklist_item_keys
        for block in _item_blocks(key, evidence, answers)
    ]
    if not blocks:
        blocks = [ReportBlock(text=EMPTY_SECTION_TEXT)]
    return ReportSection(
        section_key=configured.section_key,
        heading=configured.heading,
        blocks=tuple(blocks),
    )


def _item_blocks(
    key: str,
    evidence: dict[str, ChecklistEvidence],
    answers: dict[str, CorrelatedAnswer],
) -> list[ReportBlock]:
    blocks: list[ReportBlock] = []
    item = evidence.get(key)
    quoted: tuple[str, ...] = ()
    if item is not None:
        quoted = item.evidence
        label = OUTCOME_LABELS.get(item.outcome)
        heading = item.prompt if label is None else f"{item.prompt} — {label}."
        blocks.append(
            ReportBlock(
                text="\n".join((heading, *quoted)),
                evidence_refs=(ReportEvidenceReference(source=EvidenceSource.CHECKLIST, key=key),),
            )
        )
    answer = answers.get(key)
    if answer is not None and not _already_quoted(answer.answer, quoted):
        blocks.append(
            ReportBlock(
                text=f"{answer.question} {answer.answer}",
                evidence_refs=(ReportEvidenceReference(source=EvidenceSource.ANSWER, key=key),),
            )
        )
    return blocks


def _already_quoted(answer: str, quoted: tuple[str, ...]) -> bool:
    """Whether the checklist evidence already carries the owner's answer.

    Evidence is quoted from the canonical transcript, which contains the owner's
    replies, so an answer usually appears verbatim under its own checklist item.
    Repeating it as a second block reads as if the owner said it twice.
    """

    folded = answer.casefold().strip()
    return any(folded in segment.casefold() for segment in quoted)


def render_report_text(version: FieldNotesReportVersion) -> str:
    """Channel-neutral structured text of a completed report version."""

    lines = [version.document.title, f"Report version {version.version}"]
    for section in version.document.sections:
        lines.append("")
        lines.append(section.heading)
        for block in section.blocks:
            lines.append(block.text)
    return "\n".join(lines)

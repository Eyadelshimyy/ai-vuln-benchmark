import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gvas.domain.completeness import CompletenessChecklist
from gvas.domain.identifiers import BusinessId, JsonValue, OutboxCommandId
from gvas.domain.messages import DeliveryReceipt
from gvas.domain.outbox import OutboxCommand
from gvas.domain.templates import ReportTemplateDefinition

REPORT_SCHEMA_VERSION: Literal["field-notes-report/v1"] = "field-notes-report/v1"
_REPORT_NAMESPACE = UUID("7281d38a-7bbb-5d4b-bbf7-adb9de54dcf8")
_REPORT_VERSION_NAMESPACE = UUID("58c1274a-bf3f-5b52-926b-abd5d4113cd3")
FIELD_NOTES_REPORT_COMMAND_TYPE = "field_notes_report.generate"
FIELD_NOTES_REPORT_COMMAND_NAMESPACE = UUID("c1d9a6f2-3b47-5e81-9a2c-6d4f8b0e7315")
FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE = "field_notes_report.publish"
FIELD_NOTES_REPORT_PUBLISH_COMMAND_NAMESPACE = UUID("a94e1c57-2d0b-5f36-8e7a-41c9d2b6f083")
FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE = "field_notes_report.email"
FIELD_NOTES_REPORT_EMAIL_COMMAND_NAMESPACE = UUID("5c2e7b91-4f6a-5d38-b0c7-9a1e3d8f2b46")
REPORT_ARTIFACT_LOCATOR_PREFIX = "field-notes-report"
# One mailbox at one domain with a dot in it; display names, lists and
# whitespace are refused rather than guessed at.
_EMAIL_ADDRESS_PATTERN = re.compile(r"^[^\s@<>,;\"()]+@[^\s@<>,;\"()]+\.[^\s@<>,;\"().]+$")
DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class ReportDomainModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FieldNoteCaseStatus(StrEnum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class ChecklistOutcome(StrEnum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    NOT_APPLICABLE = "not_applicable"


class ReportStatus(StrEnum):
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ReportClaimResult(StrEnum):
    ACQUIRED = "acquired"
    TERMINAL = "terminal"
    BUSY = "busy"


class EvidenceSource(StrEnum):
    TRANSCRIPT = "transcript"
    CHECKLIST = "checklist"
    ANSWER = "answer"


class ChecklistEvidence(ReportDomainModel):
    item_key: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    prompt: str = Field(min_length=1)
    outcome: ChecklistOutcome
    evidence: tuple[str, ...] = Field(default_factory=tuple)


class CorrelatedAnswer(ReportDomainModel):
    question_key: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class FieldNoteCaseSnapshot(ReportDomainModel):
    business_id: BusinessId
    case_id: UUID
    status: FieldNoteCaseStatus
    completed_at: datetime | None = None
    canonical_transcript: str = Field(min_length=1)
    checklist_evidence: tuple[ChecklistEvidence, ...] = Field(default_factory=tuple)
    correlated_answers: tuple[CorrelatedAnswer, ...] = Field(default_factory=tuple)
    report_template_key: str | None = Field(default=None, min_length=1)
    report_template_version: int | None = Field(default=None, ge=1)

    @field_validator("completed_at")
    @classmethod
    def completed_at_is_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @model_validator(mode="after")
    def validate_correlations(self) -> "FieldNoteCaseSnapshot":
        checklist_keys = [item.item_key for item in self.checklist_evidence]
        answer_keys = [answer.question_key for answer in self.correlated_answers]
        if len(checklist_keys) != len(set(checklist_keys)):
            raise ValueError("checklist item keys must be unique")
        if len(answer_keys) != len(set(answer_keys)):
            raise ValueError("answer question keys must be unique")
        if self.status is FieldNoteCaseStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed field-note cases require completed_at")
        if (self.report_template_key is None) != (self.report_template_version is None):
            raise ValueError("report template pins need both a key and a version")
        return self


class ReportEvidenceReference(ReportDomainModel):
    source: EvidenceSource
    key: str = Field(min_length=1)


class ReportBlock(ReportDomainModel):
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)
    evidence_refs: tuple[ReportEvidenceReference, ...] = Field(default_factory=tuple)


class ReportSection(ReportDomainModel):
    section_key: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    heading: str = Field(min_length=1)
    blocks: tuple[ReportBlock, ...] = Field(min_length=1)


class FieldNotesReportDocument(ReportDomainModel):
    schema_version: Literal["field-notes-report/v1"] = REPORT_SCHEMA_VERSION
    title: str = Field(min_length=1)
    sections: tuple[ReportSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def section_keys_are_unique(self) -> "FieldNotesReportDocument":
        section_keys = [section.section_key for section in self.sections]
        if len(section_keys) != len(set(section_keys)):
            raise ValueError("report section keys must be unique")
        return self

    def validate_evidence_against(self, source: FieldNoteCaseSnapshot) -> None:
        valid_keys = {
            EvidenceSource.TRANSCRIPT: {"canonical"},
            EvidenceSource.CHECKLIST: {item.item_key for item in source.checklist_evidence},
            EvidenceSource.ANSWER: {answer.question_key for answer in source.correlated_answers},
        }
        for section in self.sections:
            for block in section.blocks:
                for reference in block.evidence_refs:
                    if reference.key not in valid_keys[reference.source]:
                        raise ValueError(
                            f"unknown {reference.source.value} evidence key: {reference.key}"
                        )

    def validate_against_template(self, template: ReportTemplateDefinition) -> None:
        """Reject a document whose structure is not the one the business configured.

        Without this the pinned definition would be advisory: a generator could
        invent headings or file an item's evidence under a section that does not
        bind it, and the report would stop being reproducible from the template.
        """
        if self.title != template.title:
            raise ValueError(
                f"report title {self.title!r} does not match the configured "
                f"title {template.title!r}"
            )
        expected_keys = [section.section_key for section in template.sections]
        actual_keys = [section.section_key for section in self.sections]
        if actual_keys != expected_keys:
            raise ValueError(
                f"report sections {actual_keys} do not match the configured "
                f"sections {expected_keys}"
            )
        for section, configured in zip(self.sections, template.sections, strict=True):
            if section.heading != configured.heading:
                raise ValueError(
                    f"section {section.section_key} heading {section.heading!r} does not "
                    f"match the configured heading {configured.heading!r}"
                )
            bound = set(configured.checklist_item_keys)
            for block in section.blocks:
                for reference in block.evidence_refs:
                    if reference.source is EvidenceSource.TRANSCRIPT:
                        continue
                    if reference.key not in bound:
                        raise ValueError(
                            f"section {section.section_key} cites {reference.source.value} "
                            f"evidence {reference.key} it does not bind"
                        )


class ChecklistEvidenceRequest(ReportDomainModel):
    """Inputs for attributing a completed review's checklist items to evidence."""

    business_id: BusinessId
    case_id: UUID
    checklist: CompletenessChecklist
    canonical_transcript: str = Field(min_length=1)
    correlated_answers: tuple[CorrelatedAnswer, ...] = Field(default_factory=tuple)


class ChecklistEvidenceAnnotation(ReportDomainModel):
    """Supporting excerpts a review model attached to an already-satisfied item."""

    item_key: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    excerpts: tuple[str, ...] = Field(default_factory=tuple)


class ReportGenerationRequest(ReportDomainModel):
    report_id: UUID
    report_version: int = Field(ge=1)
    source_fingerprint: str = Field(min_length=64, max_length=64)
    source: FieldNoteCaseSnapshot
    report_template: ReportTemplateDefinition

    @model_validator(mode="after")
    def report_template_matches_pin(self) -> "ReportGenerationRequest":
        """The generator renders the case's pinned schema, never the active one."""
        if self.report_template.business_id != self.source.business_id:
            raise ValueError("report template must belong to the case's business")
        pinned_key = self.source.report_template_key
        pinned_version = self.source.report_template_version
        if pinned_key is None or pinned_version is None:
            return self
        if (
            self.report_template.report_template_key != pinned_key
            or self.report_template.version != pinned_version
        ):
            raise ValueError("report template does not match the case's pinned version")
        return self


class FieldNotesReportVersion(ReportDomainModel):
    report_id: UUID
    report_version_id: UUID
    business_id: BusinessId
    case_id: UUID
    version: int = Field(ge=1)
    source_fingerprint: str = Field(min_length=64, max_length=64)
    generated_at: datetime
    document: FieldNotesReportDocument

    _generated_at_aware = field_validator("generated_at")(_aware)


class ReportGenerationClaim(ReportDomainModel):
    result: ReportClaimResult
    report_id: UUID
    business_id: BusinessId
    case_id: UUID
    status: ReportStatus
    attempts: int = Field(ge=1)
    report_version: int = Field(ge=1)
    source_fingerprint: str = Field(min_length=64, max_length=64)
    lease_token: UUID | None = None
    completed_version: FieldNotesReportVersion | None = None

    @model_validator(mode="after")
    def validate_result_payload(self) -> "ReportGenerationClaim":
        acquired = self.result is ReportClaimResult.ACQUIRED
        terminal = self.result is ReportClaimResult.TERMINAL
        if acquired != (self.lease_token is not None):
            raise ValueError("only acquired report claims may carry a lease token")
        if terminal != (self.completed_version is not None):
            raise ValueError("only terminal report claims carry a completed version")
        return self


class IncompleteFieldNoteCaseError(ValueError):
    pass


class ReportGenerationBusyError(RuntimeError):
    pass


class ReportGenerationFailedError(RuntimeError):
    pass


class MalformedGeneratedReportError(ValueError):
    pass


class LostReportLeaseError(ValueError):
    pass


class ReportGenerationPort(Protocol):
    async def generate(self, request: ReportGenerationRequest) -> dict[str, JsonValue]: ...


class FieldNotesReportRepository(Protocol):
    async def claim(
        self,
        source: FieldNoteCaseSnapshot,
        source_fingerprint: str,
        *,
        now: datetime,
        stale_before: datetime,
    ) -> ReportGenerationClaim: ...

    async def complete(
        self,
        claim: ReportGenerationClaim,
        document: FieldNotesReportDocument,
        *,
        generated_at: datetime,
    ) -> FieldNotesReportVersion: ...

    async def fail(
        self, claim: ReportGenerationClaim, error: str, *, failed_at: datetime
    ) -> None: ...

    async def get_completed(
        self, business_id: BusinessId, report_id: UUID
    ) -> FieldNotesReportVersion | None: ...

    async def get_version(
        self, business_id: BusinessId, report_version_id: UUID
    ) -> FieldNotesReportVersion | None: ...


class RenderedReportArtifact(ReportDomainModel):
    """A report version projected into a file the owner can open and edit."""

    content: bytes = Field(min_length=1)
    media_type: str = Field(min_length=1)
    filename: str = Field(min_length=1)


def normalize_email_address(value: str) -> str | None:
    """Returns one lower-cased address, or ``None`` when the text is not exactly one."""

    candidate = value.strip()
    if not candidate or len(candidate) > 254 or not _EMAIL_ADDRESS_PATTERN.fullmatch(candidate):
        return None
    return candidate.lower()


class ReportEmailRequest(ReportDomainModel):
    """One rendered report version addressed to one typed recipient."""

    business_id: BusinessId
    recipient_address: str = Field(min_length=3)
    idempotency_key: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body_text: str = Field(min_length=1)
    artifact: RenderedReportArtifact


class ReportEmailPort(Protocol):
    """Emails a rendered report; adapters raise to request a retry."""

    async def deliver(self, request: ReportEmailRequest) -> DeliveryReceipt: ...


class ReportArtifactRendererPort(Protocol):
    """Projects a persisted report version into its owner-facing document.

    The report document stays canonical; the artifact is derived from it on
    demand, so the same version always renders to the same file.
    """

    def render(self, version: FieldNotesReportVersion) -> RenderedReportArtifact: ...


class ReportArtifactLocator(ReportDomainModel):
    """The opaque attachment locator for a published report version.

    It names the tenant and the version so any channel adapter can ask for the
    bytes through the neutral attachment port without a storage provider.
    """

    business_id: BusinessId
    report_version_id: UUID

    def encode(self) -> str:
        return f"{REPORT_ARTIFACT_LOCATOR_PREFIX}:{self.business_id}:{self.report_version_id}"

    @classmethod
    def decode(cls, locator: str) -> "ReportArtifactLocator | None":
        prefix, _, rest = locator.partition(":")
        if prefix != REPORT_ARTIFACT_LOCATOR_PREFIX or not rest:
            return None
        business, _, version = rest.partition(":")
        try:
            return cls(business_id=BusinessId(UUID(business)), report_version_id=UUID(version))
        except ValueError:
            return None


class ReportUnitOfWork(Protocol):
    reports: FieldNotesReportRepository

    async def __aenter__(self) -> "ReportUnitOfWork": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    async def commit(self) -> None: ...


def field_notes_report_id(business_id: BusinessId, case_id: UUID) -> UUID:
    return uuid5(_REPORT_NAMESPACE, f"{business_id}:{case_id}")


def field_notes_report_version_id(report_id: UUID, version: int, source_fingerprint: str) -> UUID:
    return uuid5(_REPORT_VERSION_NAMESPACE, f"{report_id}:{version}:{source_fingerprint}")


def field_notes_report_command(
    business_id: BusinessId,
    case_id: UUID,
    review_id: UUID,
    completed_at: datetime,
) -> OutboxCommand:
    """Requests one report per completed review revision of a field-note case.

    The command identity is the reviewed revision, so recovery and replay of the
    same completed review reuse one command and one report version, while notes
    added to an open case are reviewed again and produce the next version.
    ``completed_at`` is persisted in the command payload so retries reuse the
    same snapshot fingerprint.
    """

    return OutboxCommand(
        command_id=OutboxCommandId(
            uuid5(FIELD_NOTES_REPORT_COMMAND_NAMESPACE, f"{business_id}:{case_id}:{review_id}")
        ),
        business_id=business_id,
        command_type=FIELD_NOTES_REPORT_COMMAND_TYPE,
        payload={
            "field_note_case_id": str(case_id),
            "field_note_review_id": str(review_id),
            "completed_at": _aware(completed_at).isoformat(),
        },
        dedup_key=f"field_notes_report:{case_id}:{review_id}",
    )


def field_notes_report_publish_command(
    business_id: BusinessId, case_id: UUID, report_version_id: UUID, approval_key: str
) -> OutboxCommand:
    """Publishes one approved report version as a document in the case's channel.

    The command identity is the approved version plus the owner message that
    approved it. A redelivered approval reuses one command, while a fresh
    ``approve report`` is a fresh attempt, which is what lets the owner retry
    after a publish dead-letters. The posted document itself stays single
    because the outbound message is correlated on the version alone.
    """

    return OutboxCommand(
        command_id=OutboxCommandId(
            uuid5(
                FIELD_NOTES_REPORT_PUBLISH_COMMAND_NAMESPACE,
                f"{business_id}:{report_version_id}:{approval_key}",
            )
        ),
        business_id=business_id,
        command_type=FIELD_NOTES_REPORT_PUBLISH_COMMAND_TYPE,
        payload={
            "field_note_case_id": str(case_id),
            "report_version_id": str(report_version_id),
        },
        dedup_key=f"field_notes_report_publish:{report_version_id}:{approval_key}",
    )


def report_publication_correlation_id(report_version_id: UUID) -> str:
    """Correlation of the outbound message that carries a published version.

    Its presence in the case's conversation is what "published" means: the
    publish command ran and the document is posted, or queued to be posted.
    """

    return f"field_notes_report_publish:{report_version_id}"


def field_notes_report_email_command(
    business_id: BusinessId,
    case_id: UUID,
    report_version_id: UUID,
    recipient_address: str,
    request_key: str,
) -> OutboxCommand:
    """Emails one published report version to one typed recipient, once per request.

    Identity is the version, the recipient and the owner message that asked, so
    a redelivered request collapses to one command while a fresh ``send report
    to`` is a fresh attempt that can recover a dead-lettered email.
    """

    key = report_email_idempotency_key(report_version_id, recipient_address, request_key)
    return OutboxCommand(
        command_id=OutboxCommandId(
            uuid5(FIELD_NOTES_REPORT_EMAIL_COMMAND_NAMESPACE, f"{business_id}:{key}")
        ),
        business_id=business_id,
        command_type=FIELD_NOTES_REPORT_EMAIL_COMMAND_TYPE,
        payload={
            "field_note_case_id": str(case_id),
            "report_version_id": str(report_version_id),
            "recipient_address": recipient_address,
            "request_key": request_key,
        },
        dedup_key=key,
    )


def report_email_idempotency_key(
    report_version_id: UUID, recipient_address: str, request_key: str
) -> str:
    """Shared by the command, the provider call and the thread confirmation."""

    return f"field_notes_report_email:{report_version_id}:{recipient_address}:{request_key}"


def field_note_source_fingerprint(source: FieldNoteCaseSnapshot) -> str:
    canonical = json.dumps(
        source.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()

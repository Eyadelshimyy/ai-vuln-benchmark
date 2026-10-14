"""Security finding domain model.

Static-analysis :class:`app.analysis.finding.Finding` records remain the
representation of code-quality/static issues. This model is the foundation
for later security-verification work and must not import vendor SDKs.

Invariant: an AI hypothesis never becomes a verified finding by itself.
Verification is a state transition that requires independent observational
or executable evidence. Status cannot be mutated in place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from app.domain.evidence import (
    VERIFICATION_PROVENANCE,
    Evidence,
    EvidenceBundle,
)
from app.domain.lifecycle_policy import (
    can_corroborate,
    independent_verification_items,
    positive_reproduction,
    reproduction_for_target,
)
from app.domain.security import EvidenceTier
from app.domain.target_identity import semantic_target_identity


class FindingStatus(StrEnum):
    POTENTIAL = "potential"
    CORROBORATED = "corroborated"
    REPRODUCED = "reproduced"
    VERIFIED = "verified"
    HUMAN_ACCEPTED = "human_accepted"
    REJECTED = "rejected"


_TERMINAL_BLOCK = frozenset({FindingStatus.REJECTED})
_VERIFIED_STATUSES = frozenset({FindingStatus.VERIFIED, FindingStatus.HUMAN_ACCEPTED})


class HumanReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"


@dataclass(frozen=True)
class SourceLocation:
    file_path: str
    line: int | None = None
    end_line: int | None = None
    column: int | None = None
    function: str | None = None


@dataclass(frozen=True)
class SecurityFinding:
    """A security (or suspected security) finding with explicit verification state.

    Use the constructors :meth:`potential`, :meth:`verified`, and
    :meth:`rejected`, or the :meth:`verify` transition. Direct status
    assignment is impossible because the dataclass is frozen.
    """

    title: str
    status: FindingStatus
    description: str = ""
    vulnerability_class: str | None = None
    target: str | None = None
    endpoint: str | None = None
    source_location: SourceLocation | None = None
    hypothesis: str | None = None
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    reproduction: str | None = None
    observed_behavior: str | None = None
    expected_behavior: str | None = None
    impact: str | None = None
    confidence: str = "low"
    reproducibility: str | None = None
    tools: tuple[str, ...] = ()
    ai_analysis: str | None = None
    human_review_state: HumanReviewState = HumanReviewState.UNREVIEWED
    evidence_tier: EvidenceTier = EvidenceTier.STATIC_INDICATOR
    rule_ids: tuple[str, ...] = ()
    analyzer: str | None = None
    observation_refs: tuple[str, ...] = ()
    finding_key: str = ""
    project_id: str = ""
    flow_summary: str = ""
    flow_source: str = ""
    flow_sink: str = ""
    field_path: str = ""
    files_crossed: str = ""
    analysis_incomplete: str = ""
    parser_completeness: str = ""
    evidence_summary: str = ""
    related_group: str = ""
    asset: str | None = None
    report_title: str | None = None
    report_description: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Finding title must be non-empty")
        if self.status is FindingStatus.REPRODUCED and not _has_target_reproduction(self):
            raise ValueError(
                "A reproduced finding requires a successful reproduction of this semantic target."
            )
        if self.status is FindingStatus.VERIFIED and not _trusted_for(self):
            raise ValueError(
                "Verification requires an independent observation. "
                "The reproduction record alone cannot verify a finding."
            )
        if self.status is FindingStatus.HUMAN_ACCEPTED and not (
            _has_target_reproduction(self) or _trusted_for(self)
        ):
            raise ValueError(
                "Human acceptance requires a successful reproduction "
                "or an independent verification observation."
            )

    @property
    def is_verified(self) -> bool:
        return self.status in _VERIFIED_STATUSES

    def verify(
        self, evidence: EvidenceBundle | Sequence[Evidence] | None = None
    ) -> SecurityFinding:
        """Transition a potential finding to verified using independent evidence.

        Additional evidence is merged with any evidence already attached.
        Rejected findings cannot be verified.
        """
        if self.status in _TERMINAL_BLOCK:
            raise ValueError("Rejected findings cannot be verified")
        if self.status is FindingStatus.HUMAN_ACCEPTED:
            raise ValueError("Human-accepted findings are already verified")
        merged = (
            self.evidence.extend(_as_bundle(evidence).items)
            if evidence is not None
            else self.evidence
        )
        if not independent_verification_items(merged.items, **_verification_binding(self)):
            raise ValueError(
                "Verification requires an independent observation. "
                "The reproduction record alone cannot verify a finding. "
                "A duplicate of that record is the same observation."
            )
        return replace(
            self,
            status=FindingStatus.VERIFIED,
            evidence=merged,
            evidence_tier=EvidenceTier.VERIFIED,
        )

    def corroborate(self) -> SecurityFinding:
        """Mark independent static observations as a corroborated hypothesis.

        This is not verification. AI-only findings cannot skip this via mutation.
        """
        if self.status in _TERMINAL_BLOCK:
            raise ValueError("Rejected findings cannot be corroborated")
        if self.status in _VERIFIED_STATUSES or self.status is FindingStatus.REPRODUCED:
            raise ValueError("Finding is already beyond corroboration")
        if not can_corroborate(
            self.evidence,
            target_id=semantic_target_identity(self),
            finding_id=str(self.id),
            finding_key=self.finding_key,
            project_id=self.project_id,
        ):
            raise ValueError(
                "Corroboration requires two independent non-AI observations. "
                "One static record, a duplicate of that record, or AI text is not enough."
            )
        return replace(
            self,
            status=FindingStatus.CORROBORATED,
            evidence_tier=EvidenceTier.CORROBORATED,
        )

    def reproduce(
        self, evidence: EvidenceBundle | Sequence[Evidence] | None = None
    ) -> SecurityFinding:
        """Record independent reproduction. Stronger than corroboration, not verified."""
        if self.status in _TERMINAL_BLOCK:
            raise ValueError("Rejected findings cannot be reproduced")
        if self.status in _VERIFIED_STATUSES:
            raise ValueError("Verified findings are already beyond reproduction")
        target = semantic_target_identity(self)
        base = (
            self.evidence.extend(_as_bundle(evidence).items)
            if evidence is not None
            else self.evidence
        )
        # Bind unstamped successes to this target. A record that already names
        # a target, including a previous one, is left unchanged.
        merged = EvidenceBundle.from_items(_stamp_reproductions(base.items, target))
        if not any(reproduction_for_target(item, target) for item in merged.items):
            raise ValueError(
                "Reproduction requires a successful reproduction of this semantic target. "
                "A historical reproduction, a failed attempt, or an unknown outcome is not enough."
            )
        return replace(
            self,
            status=FindingStatus.REPRODUCED,
            evidence=merged,
            evidence_tier=EvidenceTier.REPRODUCED,
        )

    def human_accept(self) -> SecurityFinding:
        """Operator accepts a reproduced or verified finding. AI cannot take this path."""
        if self.status in _TERMINAL_BLOCK:
            raise ValueError("Rejected findings cannot be accepted")
        if self.status not in {
            FindingStatus.REPRODUCED,
            FindingStatus.VERIFIED,
            FindingStatus.HUMAN_ACCEPTED,
        }:
            raise ValueError(
                "Human acceptance requires a reproduced or independently verified finding. "
                "AI hypotheses and static corroboration are not sufficient."
            )
        if not (_has_target_reproduction(self) or _trusted_for(self)):
            raise ValueError(
                "Human acceptance requires a successful reproduction "
                "or an independent verification observation."
            )
        return replace(
            self,
            status=FindingStatus.HUMAN_ACCEPTED,
            human_review_state=HumanReviewState.ACCEPTED,
            evidence_tier=EvidenceTier.VERIFIED,
        )

    def reject(
        self, *, evidence: EvidenceBundle | Sequence[Evidence] | None = None
    ) -> SecurityFinding:
        extra = _as_bundle(evidence)
        merged = self.evidence.extend(extra.items) if extra else self.evidence
        return replace(self, status=FindingStatus.REJECTED, evidence=merged)

    def with_review(self, state: HumanReviewState) -> SecurityFinding:
        """Record human review without changing verification status."""
        return replace(self, human_review_state=state)

    @classmethod
    def potential(
        cls,
        title: str,
        *,
        hypothesis: str | None = None,
        evidence: EvidenceBundle | Sequence[Evidence] | None = None,
        **kwargs: Any,
    ) -> SecurityFinding:
        """Create a potential finding. AI output must use this constructor."""
        return cls(
            title=title,
            status=FindingStatus.POTENTIAL,
            hypothesis=hypothesis,
            evidence=_as_bundle(evidence),
            **kwargs,
        )

    @classmethod
    def from_hypothesis(
        cls,
        title: str,
        hypothesis: str,
        *,
        evidence: EvidenceBundle | Sequence[Evidence] | None = None,
        ai_analysis: str | None = None,
        **kwargs: Any,
    ) -> SecurityFinding:
        """Wrap a model-generated hypothesis. Always potential, never verified."""
        kwargs.setdefault("evidence_tier", EvidenceTier.AI_HYPOTHESIS)
        return cls.potential(
            title,
            hypothesis=hypothesis,
            evidence=evidence,
            ai_analysis=ai_analysis or hypothesis,
            **kwargs,
        )

    @classmethod
    def verified(
        cls,
        title: str,
        *,
        evidence: EvidenceBundle | Sequence[Evidence],
        **kwargs: Any,
    ) -> SecurityFinding:
        bundle = _as_bundle(evidence)
        kwargs = dict(kwargs)
        if "id" not in kwargs:
            adopted = _adopted_finding_id(bundle)
            if adopted is not None:
                kwargs["id"] = adopted
        probe = cls.potential(title, evidence=bundle, **kwargs)
        if not independent_verification_items(bundle.items, **_verification_binding(probe)):
            raise ValueError(
                "Verification requires a server-issued independent observation "
                "for this semantic target."
            )
        kwargs.setdefault("evidence_tier", EvidenceTier.VERIFIED)
        return cls(title=title, status=FindingStatus.VERIFIED, evidence=bundle, **kwargs)

    @classmethod
    def rejected(
        cls,
        title: str,
        *,
        evidence: EvidenceBundle | Sequence[Evidence] | None = None,
        **kwargs: Any,
    ) -> SecurityFinding:
        return cls(
            title=title,
            status=FindingStatus.REJECTED,
            evidence=_as_bundle(evidence),
            **kwargs,
        )


def _as_bundle(evidence: EvidenceBundle | Sequence[Evidence] | None) -> EvidenceBundle:
    if evidence is None:
        return EvidenceBundle()
    if isinstance(evidence, EvidenceBundle):
        return evidence
    return EvidenceBundle.from_items(evidence)


def _verification_binding(finding: SecurityFinding) -> dict[str, str]:
    return {
        "target_id": semantic_target_identity(finding),
        "finding_id": str(finding.id),
        "finding_key": finding.finding_key,
        "project_id": finding.project_id,
    }


def _adopted_finding_id(bundle: EvidenceBundle) -> UUID | None:
    """Use the signed finding id when ``verified()`` is built from that finding."""
    from app.domain.trusted_evidence import is_trusted_observation

    found: set[str] = set()
    for item in bundle.items:
        if not is_trusted_observation(item):
            continue
        value = str(item.metadata.get("finding_id") or "")
        if value:
            found.add(value)
    if len(found) != 1:
        return None
    try:
        return UUID(next(iter(found)))
    except ValueError:
        return None


def _trusted_for(finding: SecurityFinding) -> tuple[Evidence, ...]:
    return independent_verification_items(finding.evidence.items, **_verification_binding(finding))


def _has_target_reproduction(finding: SecurityFinding) -> bool:
    target = semantic_target_identity(finding)
    return any(reproduction_for_target(item, target) for item in finding.evidence.items)


def _stamp_reproductions(
    items: tuple[Evidence, ...] | list[Evidence], target_id: str
) -> list[Evidence]:
    """Bind new successful reproductions to the finding's current target.

    Records that already name a target are left unchanged, including when that
    target is a previous one. Unsuccessful records are not stamped.
    """
    stamped: list[Evidence] = []
    for item in items:
        if (
            positive_reproduction(item)
            and target_id
            and not str(item.metadata.get("observed_target") or "")
        ):
            metadata = dict(item.metadata)
            metadata["observed_target"] = target_id
            stamped.append(replace(item, metadata=metadata))
        else:
            stamped.append(item)
    return stamped


def _has_positive_reproduction(bundle: EvidenceBundle) -> bool:
    return any(positive_reproduction(item) for item in bundle.items)


def _require_verifying_evidence(bundle: EvidenceBundle) -> None:
    allowed = ", ".join(sorted(item.value for item in VERIFICATION_PROVENANCE))
    if not bundle:
        raise ValueError(
            "A verified finding requires evidence. "
            "AI hypotheses must remain potential until independently confirmed."
        )
    if not bundle.verifying_items():
        raise ValueError(
            "Verification requires independent observational or executable evidence "
            f"(provenance: {allowed}). "
            "AI-generated hypotheses, static hints, and source excerpts cannot verify a finding."
        )

"""Production security-finding lifecycle.

Static scans enter through :meth:`FindingLifecycleService.persist_static_scan`.
Later evidence and every status change enter through
:meth:`record_collected_evidence` or :meth:`attach_evidence`. Those methods
correlate, optionally call a domain transition, and merge the result onto the
persisted row. They do not assign ``FindingStatus`` directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum
from uuid import UUID, uuid4

from app.adapters.evidence.attribution import (
    ServerAttribution,
    stamp_server_attribution,
    strip_client_attribution,
)
from app.adapters.evidence.base import EvidenceCollector
from app.domain.evidence import Evidence, EvidenceSource
from app.domain.findings import FindingStatus, SecurityFinding
from app.domain.target_identity import semantic_target_identity
from app.domain.trusted_evidence import issue_server_observation
from app.models.security_finding import DBSecurityFinding
from app.repositories.security_finding_repo import SecurityFindingRepository, to_domain
from app.security.evidence_correlation import correlate_finding
from app.security.lifecycle_rules import independent_verification_items, positive_reproduction


class LifecycleTransition(StrEnum):
    NONE = "none"
    CORROBORATE = "corroborate"
    REPRODUCE = "reproduce"
    VERIFY = "verify"
    HUMAN_ACCEPT = "human_accept"
    REJECT = "reject"


class FindingNotFoundError(LookupError):
    """No persisted security finding matches the requested id."""


class FindingProjectMismatchError(LookupError):
    """The finding exists, but it does not belong to the supplied project."""


_STATIC_SCAN_STATUSES = frozenset({FindingStatus.POTENTIAL, FindingStatus.CORROBORATED})


class FindingLifecycleService:
    def __init__(self, repo: SecurityFindingRepository) -> None:
        self._repo = repo

    async def persist_static_scan(
        self,
        findings: list[SecurityFinding],
        *,
        project_id: UUID | None,
        analysis_id: UUID | None,
    ) -> list[DBSecurityFinding]:
        """Persist potential or corroborated scan output. This is not verification."""
        illegal = [
            item.status.value for item in findings if item.status not in _STATIC_SCAN_STATUSES
        ]
        if illegal:
            raise ValueError(
                "Static scan ingress accepts potential or corroborated findings only. "
                f"Refused: {', '.join(illegal)}."
            )
        return await self._repo.bulk_create(
            findings, project_id=project_id, analysis_id=analysis_id
        )

    async def attach_evidence(
        self,
        finding_id: UUID,
        evidence: Sequence[Evidence],
        *,
        project_id: UUID | None = None,
        analysis_id: UUID | None = None,
        transition: LifecycleTransition = LifecycleTransition.NONE,
    ) -> SecurityFinding:
        """Correlate evidence already built by the server.

        This path strips client identity and does not issue a trusted
        observation. Verification evidence must come from
        :meth:`record_collected_evidence`.
        """
        row, finding = await self._load_for_update(finding_id, project_id=project_id)
        prepared = [strip_client_attribution(item) for item in evidence]
        return await self._apply(
            row,
            finding,
            prepared,
            project_id=row.project_id,
            analysis_id=analysis_id,
            transition=transition,
        )

    async def record_collected_evidence(
        self,
        finding_id: UUID,
        source: EvidenceSource,
        collectors: Sequence[EvidenceCollector],
        *,
        execution_id: str | None = None,
        project_id: UUID | None = None,
        analysis_id: UUID | None = None,
        transition: LifecycleTransition = LifecycleTransition.NONE,
    ) -> SecurityFinding:
        """Collect evidence for one loaded finding and stamp its identity.

        The execution id is created here when the caller does not already have
        one. Collector output does not keep a client-supplied finding key.
        """
        row, finding = await self._load_for_update(finding_id, project_id=project_id)
        server_execution = execution_id or uuid4().hex
        attribution = ServerAttribution.from_finding(finding, server_execution)
        extra = dict(source.extra)
        extra["attribution"] = attribution
        source.extra = extra
        collected: list[Evidence] = []
        for collector in collectors:
            collected.extend(collector.collect(source))
        project = str(row.project_id or "")
        if finding.project_id != project:
            finding = replace(finding, project_id=project)
        target_id = semantic_target_identity(finding)
        stamped: list[Evidence] = []
        for item in collected:
            attributed = stamp_server_attribution(strip_client_attribution(item), attribution)
            if attributed is None:
                continue
            stamped.append(
                issue_server_observation(
                    attributed,
                    execution_id=server_execution,
                    observed_target=target_id,
                    finding_id=attribution.finding_id,
                    finding_key=attribution.finding_key,
                    project_id=project,
                )
            )
        return await self._apply(
            row,
            finding,
            stamped,
            project_id=row.project_id,
            analysis_id=analysis_id,
            transition=transition,
        )

    async def _load_for_update(
        self, finding_id: UUID, *, project_id: UUID | None
    ) -> tuple[DBSecurityFinding, SecurityFinding]:
        row = await self._repo.get_for_update(finding_id)
        if row is None:
            raise FindingNotFoundError(str(finding_id))
        if project_id is not None and row.project_id != project_id:
            raise FindingProjectMismatchError(str(finding_id))
        return row, to_domain(row)

    async def _apply(
        self,
        row: DBSecurityFinding,
        finding: SecurityFinding,
        evidence: Sequence[Evidence],
        *,
        project_id: UUID | None,
        analysis_id: UUID | None,
        transition: LifecycleTransition,
    ) -> SecurityFinding:
        peers = await self._peers(finding, project_id=project_id)
        updated = correlate_finding(finding, evidence, peers=peers)
        if transition is not LifecycleTransition.NONE:
            try:
                updated = apply_lifecycle_transition(updated, transition)
            except ValueError:
                # Keep attributed evidence, including contradictions and failed
                # attempts. The status change is refused by the domain rule.
                await self._repo.save_lifecycle(
                    updated, project_id=project_id, analysis_id=analysis_id
                )
                raise
        persisted = await self._repo.save_lifecycle(
            updated, project_id=project_id, analysis_id=analysis_id
        )
        del row
        return to_domain(persisted)

    async def _peers(
        self, finding: SecurityFinding, *, project_id: UUID | None
    ) -> tuple[SecurityFinding, ...]:
        loc = finding.source_location
        if project_id is None or loc is None or not loc.file_path:
            return ()
        rows = await self._repo.competing_findings(
            project_id,
            file_path=loc.file_path,
            line=loc.line,
            vulnerability_class=finding.vulnerability_class,
        )
        return tuple(to_domain(row) for row in rows if row.id != finding.id)


def apply_lifecycle_transition(
    finding: SecurityFinding, transition: LifecycleTransition
) -> SecurityFinding:
    """Run one domain method. Orchestration cannot assign status itself."""
    if transition is LifecycleTransition.CORROBORATE:
        return finding.corroborate()
    if transition is LifecycleTransition.REPRODUCE:
        if not any(positive_reproduction(item) for item in finding.evidence.items):
            raise ValueError(
                "Reproduction requires a successful reproduction record. "
                "Failed tests, logs, screenshots, generated tests, AI text, "
                "and contradictory results are not reproduction."
            )
        return finding.reproduce()
    if transition is LifecycleTransition.VERIFY:
        if not independent_verification_items(finding):
            raise ValueError(
                "Verification requires an independent observation. "
                "The reproduction record alone cannot be reused as verification."
            )
        return finding.verify()
    if transition is LifecycleTransition.HUMAN_ACCEPT:
        return finding.human_accept()
    if transition is LifecycleTransition.REJECT:
        return finding.reject()
    return finding

from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.evidence import Evidence, EvidenceBundle
from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding
from app.domain.lifecycle_policy import independent_verification_items
from app.domain.security import EvidenceTier
from app.models.security_finding import DBSecurityFinding
from app.repositories.finding_identity import intelligence_with_column_key
from app.schemas.security import SecurityFindingResponse
from app.security.evidence_correlation import evidence_identity


class SecurityFindingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(
        self,
        findings: list[SecurityFinding],
        *,
        project_id: UUID | None,
        analysis_id: UUID | None,
    ) -> list[DBSecurityFinding]:
        existing_by_key = await self._load_existing_by_key(project_id, findings)
        rows: list[DBSecurityFinding] = []
        seen: set[str] = set()
        for finding in findings:
            key = finding.finding_key or ""
            if project_id is not None and key and key in existing_by_key:
                row = existing_by_key[key]
                _reconcile_row(row, finding, analysis_id=analysis_id)
                rows.append(row)
                seen.add(key)
                continue
            if project_id is not None and key and key in seen:
                continue
            row = await self._insert_or_reconcile(
                finding, project_id=project_id, analysis_id=analysis_id
            )
            rows.append(row)
            if key:
                existing_by_key[key] = row
                seen.add(key)
        await self._session.flush()
        return rows

    async def save_domain(
        self,
        finding: SecurityFinding,
        *,
        project_id: UUID | None,
        analysis_id: UUID | None = None,
    ) -> DBSecurityFinding:
        """Persist lifecycle state through the same merge as ``save_lifecycle``.

        There is no overwrite path that can drop evidence or downgrade status.
        ``analysis_id`` is left unchanged unless a new analysis is supplied.
        """
        return await self.save_lifecycle(finding, project_id=project_id, analysis_id=analysis_id)

    async def list_for_project(
        self, project_id: UUID, *, offset: int = 0, limit: int = 100
    ) -> tuple[list[DBSecurityFinding], int]:
        count_result = await self._session.execute(
            select(DBSecurityFinding).where(DBSecurityFinding.project_id == project_id)
        )
        total = len(count_result.scalars().all())
        result = await self._session.execute(
            select(DBSecurityFinding)
            .where(DBSecurityFinding.project_id == project_id)
            .order_by(DBSecurityFinding.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def list_for_analysis(self, analysis_id: UUID) -> list[DBSecurityFinding]:
        result = await self._session.execute(
            select(DBSecurityFinding)
            .where(DBSecurityFinding.analysis_id == analysis_id)
            .order_by(DBSecurityFinding.created_at.desc())
        )
        return list(result.scalars().all())

    async def get(self, finding_id: UUID) -> DBSecurityFinding | None:
        return await self._session.get(DBSecurityFinding, finding_id)

    async def get_for_update(self, finding_id: UUID) -> DBSecurityFinding | None:
        """Lock the finding row on PostgreSQL. SQLite has no row lock."""
        stmt = select(DBSecurityFinding).where(DBSecurityFinding.id == finding_id)
        connection = await self._session.connection()
        if connection.dialect.name == "postgresql":
            stmt = stmt.with_for_update()
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def competing_findings(
        self,
        project_id: UUID,
        *,
        file_path: str,
        line: int | None,
        vulnerability_class: str | None,
    ) -> list[DBSecurityFinding]:
        """Every finding that could compete for evidence at this location.

        This is not the paginated project listing. A competitor past the first
        page still participates.
        """
        stmt = select(DBSecurityFinding).where(DBSecurityFinding.project_id == project_id)
        if vulnerability_class:
            stmt = stmt.where(DBSecurityFinding.vulnerability_class == vulnerability_class)
        if line is not None:
            stmt = stmt.where(DBSecurityFinding.line == line)
        result = await self._session.execute(stmt)
        return [
            row
            for row in result.scalars().all()
            if row.file_path and _same_stored_path(row.file_path, file_path)
        ]

    async def save_lifecycle(
        self,
        finding: SecurityFinding,
        *,
        project_id: UUID | None,
        analysis_id: UUID | None = None,
    ) -> DBSecurityFinding:
        """Merge evidence onto the locked row. Do not last-write-wins evidence away."""
        row = await self.get_for_update(finding.id)
        if row is None and project_id is not None and finding.finding_key:
            row = await self.get_by_key(project_id, finding.finding_key)
        if row is None:
            created = _to_row(finding, project_id=project_id, analysis_id=analysis_id)
            return await self._add_row(created, finding, project_id=project_id)
        current = to_domain(row)
        merged = merge_lifecycle_state(current, finding)
        kept_analysis = row.analysis_id if analysis_id is None else analysis_id
        incoming = _to_row(merged, project_id=row.project_id, analysis_id=kept_analysis)
        _overwrite_row(row, incoming, replace_analysis=analysis_id is not None)
        await self._session.flush()
        return row

    async def get_by_key(self, project_id: UUID, finding_key: str) -> DBSecurityFinding | None:
        result = await self._session.execute(
            select(DBSecurityFinding).where(
                DBSecurityFinding.project_id == project_id,
                DBSecurityFinding.finding_key == finding_key,
            )
        )
        return result.scalars().first()

    async def get_domain(self, finding_id: UUID) -> SecurityFinding | None:
        row = await self.get(finding_id)
        return to_domain(row) if row else None

    async def _load_existing_by_key(
        self, project_id: UUID | None, findings: list[SecurityFinding]
    ) -> dict[str, DBSecurityFinding]:
        keys = [finding.finding_key for finding in findings if finding.finding_key]
        if project_id is None or not keys:
            return {}
        result = await self._session.execute(
            select(DBSecurityFinding).where(
                DBSecurityFinding.project_id == project_id,
                DBSecurityFinding.finding_key.in_(keys),
            )
        )
        return {row.finding_key: row for row in result.scalars().all() if row.finding_key}

    async def _insert_or_reconcile(
        self,
        finding: SecurityFinding,
        *,
        project_id: UUID | None,
        analysis_id: UUID | None,
    ) -> DBSecurityFinding:
        row = _to_row(finding, project_id=project_id, analysis_id=analysis_id)
        return await self._add_row(row, finding, project_id=project_id)

    async def _add_row(
        self,
        row: DBSecurityFinding,
        finding: SecurityFinding,
        *,
        project_id: UUID | None,
    ) -> DBSecurityFinding:
        try:
            async with self._session.begin_nested():
                self._session.add(row)
                await self._session.flush()
            return row
        except IntegrityError:
            if row in self._session:
                self._session.expunge(row)
            if project_id is None or not finding.finding_key:
                raise
            raced = await self.get_by_key(project_id, finding.finding_key)
            if raced is None:
                raise
            _reconcile_row(raced, finding, analysis_id=row.analysis_id)
            return raced


def _to_row(
    finding: SecurityFinding,
    *,
    project_id: UUID | None,
    analysis_id: UUID | None,
) -> DBSecurityFinding:
    loc = finding.source_location
    evidence = [
        {
            "kind": item.kind.value,
            "source": item.source,
            "summary": item.summary,
            "details": item.details,
            "artifact_path": item.artifact_path,
            "provenance": item.provenance.value if item.provenance else None,
            "collected_at": item.collected_at.isoformat(),
            "metadata": _json_metadata(item.metadata),
        }
        for item in finding.evidence.items
    ]
    key = finding.finding_key or None
    return DBSecurityFinding(
        id=finding.id,
        project_id=project_id,
        analysis_id=analysis_id,
        title=finding.title,
        status=finding.status.value,
        vulnerability_class=finding.vulnerability_class,
        evidence_tier=finding.evidence_tier.value,
        confidence=finding.confidence,
        description=finding.description,
        hypothesis=finding.hypothesis,
        ai_analysis=finding.ai_analysis,
        impact=finding.impact,
        file_path=loc.file_path if loc else finding.asset,
        line=loc.line if loc else None,
        language=finding.evidence.items[0].metadata.get("language")
        if finding.evidence.items
        else None,
        analyzer=finding.analyzer,
        parser_backend=(
            finding.evidence.items[0].metadata.get("parser_backend")
            if finding.evidence.items
            else None
        ),
        node_id=finding.evidence.items[0].metadata.get("node_id")
        if finding.evidence.items
        else None,
        taint_path=finding.evidence.items[0].metadata.get("taint_path")
        if finding.evidence.items
        else None,
        rule_ids=",".join(finding.rule_ids),
        observation_refs=",".join(finding.observation_refs),
        evidence_json=json.dumps(evidence),
        intelligence_json=intelligence_with_column_key(
            json.dumps(_intelligence(finding), sort_keys=True), key
        ),
        finding_key=key,
        asset=finding.asset,
        target=finding.target,
        endpoint=finding.endpoint,
        reproduction=finding.reproduction,
        observed_behavior=finding.observed_behavior,
        expected_behavior=finding.expected_behavior,
        report_title=finding.report_title,
        report_description=finding.report_description,
        created_at=finding.created_at,
    )


_INTELLIGENCE_FIELDS = (
    "finding_key",
    "flow_summary",
    "flow_source",
    "flow_sink",
    "field_path",
    "files_crossed",
    "analysis_incomplete",
    "parser_completeness",
    "evidence_summary",
    "related_group",
    "human_review_state",
)


def _stored_intelligence(raw: str | None) -> dict[str, str]:
    try:
        loaded = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        key: str(loaded[key])
        for key in _INTELLIGENCE_FIELDS
        if key in loaded and isinstance(loaded[key], str)
    }


def _row_intelligence(row: DBSecurityFinding) -> dict[str, str]:
    """Column ``finding_key`` is authoritative, including NULL."""
    intel = _stored_intelligence(row.intelligence_json)
    intel.pop("finding_key", None)
    if row.finding_key:
        intel["finding_key"] = row.finding_key
    else:
        intel["finding_key"] = ""
    return intel


def to_security_response(row: DBSecurityFinding) -> SecurityFindingResponse:
    """Map a stored row to the API shape, including additive flow fields."""
    data = {
        "id": row.id,
        "project_id": row.project_id,
        "analysis_id": row.analysis_id,
        "title": row.title,
        "status": row.status,
        "vulnerability_class": row.vulnerability_class,
        "evidence_tier": row.evidence_tier,
        "confidence": row.confidence,
        "description": row.description,
        "hypothesis": row.hypothesis,
        "ai_analysis": row.ai_analysis,
        "impact": row.impact,
        "file_path": row.file_path,
        "line": row.line,
        "analyzer": row.analyzer,
        "rule_ids": row.rule_ids,
        "observation_refs": row.observation_refs,
        "asset": row.asset,
        "created_at": row.created_at,
        "parser_backend": row.parser_backend,
        "node_id": row.node_id,
        "taint_path": row.taint_path,
        "language": row.language,
    }
    data.update(_row_intelligence(row))
    domain = to_domain(row)
    data["status"] = domain.status.value
    data["evidence_tier"] = domain.evidence_tier.value
    return SecurityFindingResponse.model_validate(data)


def _json_metadata(metadata: object) -> dict[str, object]:
    """Keep evidence metadata in the existing JSON blob. Non-JSON values are dropped."""
    from collections.abc import Mapping

    if not isinstance(metadata, Mapping):
        return {}
    try:
        encoded = json.dumps(dict(metadata), sort_keys=True)
    except (TypeError, ValueError):
        return {}
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _intelligence(finding: SecurityFinding) -> dict[str, str]:
    payload = {
        "flow_summary": finding.flow_summary,
        "flow_source": finding.flow_source,
        "flow_sink": finding.flow_sink,
        "field_path": finding.field_path,
        "files_crossed": finding.files_crossed,
        "analysis_incomplete": finding.analysis_incomplete,
        "parser_completeness": finding.parser_completeness,
        "evidence_summary": finding.evidence_summary,
        "related_group": finding.related_group,
        "human_review_state": finding.human_review_state.value,
    }
    if finding.finding_key:
        payload["finding_key"] = finding.finding_key
    return payload


def _evidence_from_stored(item: dict[str, object]) -> Evidence | None:
    """Rebuild one stored record. Unknown or inconsistent records stay non-authoritative."""
    from app.domain.evidence import Evidence, EvidenceKind, EvidenceProvenance
    from app.domain.trusted_evidence import restore_server_observation

    raw_kind = item.get("kind")
    try:
        kind = EvidenceKind(str(raw_kind or ""))
    except ValueError:
        return _quarantine(str(raw_kind or ""))
    derived = {
        EvidenceKind.AI_ANALYSIS: EvidenceProvenance.AI_HYPOTHESIS,
        EvidenceKind.STATIC_ANALYSIS: EvidenceProvenance.STATIC_ANALYSIS,
        EvidenceKind.SOURCE_CODE: EvidenceProvenance.SOURCE_OBSERVATION,
        EvidenceKind.GENERATED_TEST: EvidenceProvenance.SOURCE_OBSERVATION,
        EvidenceKind.TEST_FAILURE: EvidenceProvenance.EXECUTION,
        EvidenceKind.REPRODUCTION: EvidenceProvenance.REPRODUCTION,
        EvidenceKind.BROWSER: EvidenceProvenance.BROWSER_OBSERVATION,
        EvidenceKind.PROXY: EvidenceProvenance.HTTP_OBSERVATION,
        EvidenceKind.HTTP_REQUEST: EvidenceProvenance.HTTP_OBSERVATION,
        EvidenceKind.HTTP_RESPONSE: EvidenceProvenance.HTTP_OBSERVATION,
        EvidenceKind.SCANNER: EvidenceProvenance.SCANNER_RESULT,
        EvidenceKind.SCANNER_PLAN: EvidenceProvenance.SCANNER_PLAN,
        EvidenceKind.TOOL_STATUS: EvidenceProvenance.TOOL_STATUS,
        EvidenceKind.FUZZING: EvidenceProvenance.FUZZING_RESULT,
        EvidenceKind.API_TEST: EvidenceProvenance.API_TEST,
        EvidenceKind.LOG: EvidenceProvenance.LOG,
        EvidenceKind.SCREENSHOT: EvidenceProvenance.SCREENSHOT,
        EvidenceKind.REPLAY: EvidenceProvenance.REPLAY,
    }[kind]
    raw_prov = item.get("provenance")
    if isinstance(raw_prov, str) and raw_prov:
        try:
            stored = EvidenceProvenance(raw_prov)
        except ValueError:
            return _quarantine(str(raw_kind or ""))
        if stored is not derived:
            return _quarantine(str(raw_kind or ""))
    raw_meta = item.get("metadata", {})
    metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}
    collected_at = None
    raw_collected = item.get("collected_at")
    if isinstance(raw_collected, str) and raw_collected:
        try:
            from datetime import datetime

            collected_at = datetime.fromisoformat(raw_collected)
        except ValueError:
            collected_at = None
    evidence_kwargs: dict[str, object] = {
        "kind": kind,
        "source": str(item.get("source") or "stored"),
        "summary": str(item.get("summary") or "stored evidence"),
        "details": str(item.get("details") or ""),
        "artifact_path": item.get("artifact_path"),
        "metadata": metadata,
    }
    if collected_at is not None:
        evidence_kwargs["collected_at"] = collected_at
    try:
        built = Evidence(**evidence_kwargs)  # type: ignore[arg-type]
    except ValueError:
        return _quarantine(str(raw_kind or ""))
    return restore_server_observation(built)


def _quarantine(raw_kind: str) -> Evidence:
    from app.domain.evidence import Evidence, EvidenceKind

    return Evidence(
        kind=EvidenceKind.LOG,
        source="quarantine",
        summary="quarantined evidence",
        metadata={"quarantined": "true", "raw_kind": raw_kind[:80]},
    )


def _reconcile_row(
    row: DBSecurityFinding,
    finding: SecurityFinding,
    *,
    analysis_id: UUID | None,
) -> None:
    """Update static facts without destroying higher-trust lifecycle state.

    A rescan may refresh location and flow fields. Formatting and line
    movement keep the semantic target, so verification stays. A material
    target change keeps the historical evidence and requires a new
    verification: the row returns to the scan's potential or corroborated
    status. ``analysis_id`` is the latest scan that observed this finding.
    """
    from app.domain.target_identity import semantic_target_identity

    project = str(row.project_id or "")
    previous_target = semantic_target_identity(to_domain(row), project_id=project)
    incoming_target = semantic_target_identity(finding, project_id=project)
    target_changed = previous_target != incoming_target
    previous_status = row.status
    incoming = _to_row(finding, project_id=row.project_id, analysis_id=analysis_id)
    row.title = incoming.title
    row.vulnerability_class = incoming.vulnerability_class
    row.confidence = incoming.confidence
    row.description = incoming.description
    row.file_path = incoming.file_path
    row.line = incoming.line
    row.language = incoming.language
    row.analyzer = incoming.analyzer
    row.parser_backend = incoming.parser_backend
    row.node_id = incoming.node_id
    row.taint_path = incoming.taint_path
    row.rule_ids = incoming.rule_ids
    row.observation_refs = incoming.observation_refs
    row.intelligence_json = _merge_intelligence_json(
        row.intelligence_json, incoming.intelligence_json, finding_key=incoming.finding_key
    )
    row.finding_key = incoming.finding_key or row.finding_key
    row.asset = incoming.asset
    row.target = incoming.target
    row.endpoint = incoming.endpoint
    row.report_title = incoming.report_title
    row.report_description = incoming.report_description
    row.evidence_json = _merge_evidence_json(row.evidence_json, incoming.evidence_json)
    if target_changed and previous_status in {
        "corroborated",
        "reproduced",
        "verified",
        "human_accepted",
    }:
        row.status = finding.status.value
        row.evidence_tier = finding.evidence_tier.value
        stored = _stored_intelligence(row.intelligence_json)
        stored["human_review_state"] = "unreviewed"
        row.intelligence_json = intelligence_with_column_key(
            json.dumps(stored, sort_keys=True), incoming.finding_key
        )
    if analysis_id is not None:
        row.analysis_id = analysis_id
    if incoming.hypothesis and not row.hypothesis:
        row.hypothesis = incoming.hypothesis
    if incoming.ai_analysis:
        row.ai_analysis = incoming.ai_analysis
    if incoming.impact:
        row.impact = incoming.impact
    # Same-target rescans keep status, tier, and review. A changed semantic
    # target above replaces verification with the scan's own static status.


def _overwrite_row(
    row: DBSecurityFinding, incoming: DBSecurityFinding, *, replace_analysis: bool
) -> None:
    """Write every persistable field from an explicit domain save."""
    row.title = incoming.title
    row.status = incoming.status
    row.vulnerability_class = incoming.vulnerability_class
    row.evidence_tier = incoming.evidence_tier
    row.confidence = incoming.confidence
    row.description = incoming.description
    row.hypothesis = incoming.hypothesis
    row.ai_analysis = incoming.ai_analysis
    row.impact = incoming.impact
    row.file_path = incoming.file_path
    row.line = incoming.line
    row.language = incoming.language
    row.analyzer = incoming.analyzer
    row.parser_backend = incoming.parser_backend
    row.node_id = incoming.node_id
    row.taint_path = incoming.taint_path
    row.rule_ids = incoming.rule_ids
    row.observation_refs = incoming.observation_refs
    row.evidence_json = incoming.evidence_json
    row.intelligence_json = incoming.intelligence_json
    row.finding_key = incoming.finding_key
    row.asset = incoming.asset
    row.target = incoming.target
    row.endpoint = incoming.endpoint
    row.reproduction = incoming.reproduction
    row.observed_behavior = incoming.observed_behavior
    row.expected_behavior = incoming.expected_behavior
    row.report_title = incoming.report_title
    row.report_description = incoming.report_description
    if replace_analysis:
        row.analysis_id = incoming.analysis_id


def _merge_intelligence_json(
    existing_json: str, incoming_json: str, *, finding_key: str | None
) -> str:
    existing = _stored_intelligence(existing_json)
    incoming = _stored_intelligence(incoming_json)
    merged = dict(incoming)
    existing_review = existing.get("human_review_state", "")
    if existing_review and existing_review != "unreviewed":
        merged["human_review_state"] = existing_review
    return intelligence_with_column_key(json.dumps(merged, sort_keys=True), finding_key)


def _merge_evidence_json(existing_json: str, incoming_json: str) -> str:
    existing = _parse_evidence_list(existing_json)
    incoming = _parse_evidence_list(incoming_json)
    kept = [item for item in existing if str(item.get("kind") or "") != "static_analysis"]
    static_new = [item for item in incoming if str(item.get("kind") or "") == "static_analysis"]
    ai_new = [item for item in incoming if str(item.get("kind") or "") == "ai_analysis"]
    if ai_new:
        kept = [item for item in kept if str(item.get("kind") or "") != "ai_analysis"]
        kept.extend(ai_new)
    return json.dumps(static_new + kept)


def _parse_evidence_list(raw: str) -> list[dict[str, object]]:
    try:
        loaded = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def merge_lifecycle_state(current: SecurityFinding, incoming: SecurityFinding) -> SecurityFinding:
    """Union evidence and keep the row's static facts and stronger lifecycle.

    Lifecycle saves start from the current row. A stale snapshot cannot
    replace location, flow, report text, or AI analysis that the row already
    holds. Empty current fields may be filled from the incoming object.
    Runtime evidence is the union of both sides. ``created_at`` stays original.
    """
    from dataclasses import replace

    seen: set[tuple[str, ...]] = set()
    items = []
    for item in (*current.evidence.items, *incoming.evidence.items):
        key = evidence_identity(item)
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    status = _prefer_status(current.status, incoming.status)
    review = current.human_review_state
    if (
        review is HumanReviewState.UNREVIEWED
        and incoming.human_review_state is not HumanReviewState.UNREVIEWED
    ):
        review = incoming.human_review_state
    if status is FindingStatus.HUMAN_ACCEPTED:
        review = HumanReviewState.ACCEPTED
    return replace(
        current,
        description=_fill(current.description, incoming.description),
        vulnerability_class=_fill(current.vulnerability_class, incoming.vulnerability_class),
        target=_fill(current.target, incoming.target),
        endpoint=_fill(current.endpoint, incoming.endpoint),
        source_location=current.source_location or incoming.source_location,
        hypothesis=_fill(current.hypothesis, incoming.hypothesis),
        evidence=EvidenceBundle.from_items(items),
        reproduction=_fill(current.reproduction, incoming.reproduction),
        observed_behavior=_fill(current.observed_behavior, incoming.observed_behavior),
        expected_behavior=_fill(current.expected_behavior, incoming.expected_behavior),
        impact=_fill(current.impact, incoming.impact),
        confidence=current.confidence or incoming.confidence,
        reproducibility=_fill(current.reproducibility, incoming.reproducibility),
        tools=current.tools or incoming.tools,
        ai_analysis=_fill(current.ai_analysis, incoming.ai_analysis),
        human_review_state=review,
        evidence_tier=_tier_for(status, current=current, incoming=incoming),
        rule_ids=current.rule_ids or incoming.rule_ids,
        analyzer=_fill(current.analyzer, incoming.analyzer),
        observation_refs=current.observation_refs or incoming.observation_refs,
        finding_key=current.finding_key or incoming.finding_key,
        flow_summary=_fill(current.flow_summary, incoming.flow_summary),
        flow_source=_fill(current.flow_source, incoming.flow_source),
        flow_sink=_fill(current.flow_sink, incoming.flow_sink),
        field_path=_fill(current.field_path, incoming.field_path),
        files_crossed=_fill(current.files_crossed, incoming.files_crossed),
        analysis_incomplete=_fill(current.analysis_incomplete, incoming.analysis_incomplete),
        parser_completeness=_fill(current.parser_completeness, incoming.parser_completeness),
        evidence_summary=_fill(current.evidence_summary, incoming.evidence_summary),
        related_group=_fill(current.related_group, incoming.related_group),
        asset=_fill(current.asset, incoming.asset),
        report_title=_fill(current.report_title, incoming.report_title),
        report_description=_fill(current.report_description, incoming.report_description),
        status=status,
        id=current.id,
        created_at=current.created_at,
    )


def _human_accepted_from_stored(title: str, kwargs: dict[str, object]) -> SecurityFinding:
    """Reload an accepted finding without treating reproduction as verification.

    An accepted finding that already has an independent observation reloads
    through ``verified()``. An accepted finding that only has a successful
    reproduction reloads through ``reproduce()``. Neither path assigns
    ``HUMAN_ACCEPTED`` when the stored evidence fails the domain rule.
    The observation must be bound to this finding's semantic target.
    """
    from app.domain.lifecycle_policy import reproduction_for_target
    from app.domain.target_identity import semantic_target_identity

    bundle = kwargs.get("evidence")
    items = bundle.items if isinstance(bundle, EvidenceBundle) else ()
    probe = SecurityFinding.potential(title, **kwargs)  # type: ignore[arg-type]
    target = semantic_target_identity(probe)
    if independent_verification_items(
        items,
        target_id=target,
        finding_id=str(probe.id),
        finding_key=str(probe.finding_key),
        project_id=str(probe.project_id),
    ):
        return SecurityFinding.verified(title, **kwargs).human_accept()  # type: ignore[arg-type]
    if any(reproduction_for_target(item, target) for item in items):
        return probe.reproduce().human_accept()
    raise ValueError("stored acceptance does not match the current semantic target")


def _fill[T](current: T, incoming: T) -> T:
    """Keep a current value. Use the incoming value only when current is empty."""
    if current is None or current == "":
        return incoming
    return current


_STATUS_RANK = {
    FindingStatus.POTENTIAL: 0,
    FindingStatus.CORROBORATED: 1,
    FindingStatus.REPRODUCED: 2,
    FindingStatus.VERIFIED: 3,
    FindingStatus.HUMAN_ACCEPTED: 4,
}


def _prefer_status(current: FindingStatus, incoming: FindingStatus) -> FindingStatus:
    if current is FindingStatus.REJECTED:
        return FindingStatus.REJECTED
    if incoming is FindingStatus.REJECTED:
        return FindingStatus.REJECTED
    if _STATUS_RANK[incoming] >= _STATUS_RANK[current]:
        return incoming
    return current


def _tier_for(
    status: FindingStatus, *, current: SecurityFinding, incoming: SecurityFinding
) -> EvidenceTier:
    if status is FindingStatus.REJECTED:
        return current.evidence_tier
    if status is incoming.status:
        return incoming.evidence_tier
    if status in {FindingStatus.VERIFIED, FindingStatus.HUMAN_ACCEPTED}:
        return EvidenceTier.VERIFIED
    if status is FindingStatus.REPRODUCED:
        return EvidenceTier.REPRODUCED
    if status is FindingStatus.CORROBORATED:
        return EvidenceTier.CORROBORATED
    return current.evidence_tier


def _same_stored_path(left: str, right: str) -> bool:
    norm_left = left.replace("\\", "/").lstrip("./")
    norm_right = right.replace("\\", "/").lstrip("./")
    return (
        norm_left == norm_right
        or norm_left.endswith("/" + norm_right)
        or norm_right.endswith("/" + norm_left)
    )


def to_domain(row: DBSecurityFinding) -> SecurityFinding:
    from app.domain.evidence import EvidenceBundle
    from app.domain.findings import FindingStatus, HumanReviewState, SourceLocation
    from app.domain.security import EvidenceTier

    items: list[Evidence] = []
    try:
        raw = json.loads(row.evidence_json or "[]")
    except json.JSONDecodeError:
        raw = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            loaded = _evidence_from_stored(item)
            if loaded is not None:
                items.append(loaded)
    status = FindingStatus(row.status)
    title = row.title
    intel = _row_intelligence(row)
    review = intel.pop("human_review_state", "")
    finding_key = intel.pop("finding_key", "")
    kwargs: dict[str, object] = {
        "id": row.id,
        "description": row.description,
        "vulnerability_class": row.vulnerability_class,
        "target": row.target,
        "endpoint": row.endpoint,
        "hypothesis": row.hypothesis,
        "evidence": EvidenceBundle.from_items(items),
        "reproduction": row.reproduction,
        "observed_behavior": row.observed_behavior,
        "expected_behavior": row.expected_behavior,
        "impact": row.impact,
        "confidence": row.confidence,
        "ai_analysis": row.ai_analysis,
        "evidence_tier": EvidenceTier(row.evidence_tier),
        "analyzer": row.analyzer,
        "asset": row.asset,
        "report_title": row.report_title,
        "report_description": row.report_description,
        "created_at": row.created_at,
        "finding_key": finding_key,
        **intel,
    }
    if row.rule_ids:
        kwargs["rule_ids"] = tuple(part for part in row.rule_ids.split(",") if part)
    if row.observation_refs:
        kwargs["observation_refs"] = tuple(part for part in row.observation_refs.split(",") if part)
    if row.project_id is not None:
        kwargs["project_id"] = str(row.project_id)
    if row.file_path:
        kwargs["source_location"] = SourceLocation(file_path=row.file_path, line=row.line)
    if review:
        try:
            kwargs["human_review_state"] = HumanReviewState(review)
        except ValueError:
            pass
    if status is FindingStatus.VERIFIED:
        try:
            return SecurityFinding.verified(title, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            return SecurityFinding.potential(title, **kwargs)  # type: ignore[arg-type]
    if status is FindingStatus.HUMAN_ACCEPTED:
        try:
            return _human_accepted_from_stored(title, kwargs)
        except ValueError:
            return SecurityFinding.potential(title, **kwargs)  # type: ignore[arg-type]
    if status is FindingStatus.REJECTED:
        return SecurityFinding.rejected(title, **kwargs)  # type: ignore[arg-type]
    finding = SecurityFinding.potential(title, **kwargs)  # type: ignore[arg-type]
    if status is FindingStatus.CORROBORATED:
        try:
            return finding.corroborate()
        except ValueError:
            return finding
    if status is FindingStatus.REPRODUCED:
        from app.domain.lifecycle_policy import reproduction_for_target
        from app.domain.target_identity import semantic_target_identity

        target = semantic_target_identity(finding)
        if any(reproduction_for_target(item, target) for item in finding.evidence.items):
            try:
                return finding.reproduce()
            except ValueError:
                return finding
        return finding
    return finding

"""Reload keeps target binding and drops invalid lifecycle claims."""

from __future__ import annotations

import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import FindingStatus, SecurityFinding, SourceLocation
from app.domain.target_identity import semantic_target_identity
from app.domain.trusted_evidence import is_trusted_observation, issue_for_finding
from app.models.project import Project
from app.repositories.security_finding_repo import SecurityFindingRepository, to_domain
from app.security_agent.agent import ResearchSession
from app.security_agent.schemas import ResearchHypothesis
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.safety import SafetyLimits
from tests.test_phase26.test_target_binding import _finding, _http, _repro, _static


@pytest.mark.asyncio
async def test_reproduced_target_survives_reload(db_session: AsyncSession) -> None:
    project = Project(name="phase26", repository_path="/tmp/bugforge-phase26")
    db_session.add(project)
    await db_session.flush()
    finding = _finding(_static(), project_id=str(project.id)).reproduce([_repro()])
    repo = SecurityFindingRepository(db_session)
    await repo.save_domain(finding, project_id=project.id)
    await db_session.commit()
    restored = await repo.get_domain(finding.id)
    assert restored is not None
    assert restored.status is FindingStatus.REPRODUCED
    assert restored.project_id == str(project.id)
    target = semantic_target_identity(restored)
    assert any(item.metadata.get("observed_target") == target for item in restored.evidence.items)


@pytest.mark.asyncio
async def test_target_drift_and_malformed_evidence_fail_closed(db_session: AsyncSession) -> None:
    project = Project(name="phase26-drift", repository_path="/tmp/bugforge-phase26-drift")
    db_session.add(project)
    await db_session.flush()
    finding = _finding(_static(), project_id=str(project.id))
    issued = issue_for_finding(finding, _http(), "exec-keep")
    verified = finding.verify([issued])
    repo = SecurityFindingRepository(db_session)
    await repo.save_domain(verified, project_id=project.id)
    await db_session.commit()

    row = await repo.get(finding.id)
    assert row is not None
    intel = json.loads(row.intelligence_json or "{}")
    intel["flow_sink"] = "exec"
    row.intelligence_json = json.dumps(intel)
    evidence = json.loads(row.evidence_json or "[]")
    evidence.append({"kind": "not-a-kind", "summary": "forged verified", "provenance": "server"})
    row.evidence_json = json.dumps(evidence)
    await db_session.commit()
    db_session.expire_all()
    restored = to_domain(await repo.get(finding.id))
    assert restored.status is FindingStatus.POTENTIAL
    assert not any(
        is_trusted_observation(item) and item.summary == "forged verified"
        for item in restored.evidence.items
    )
    assert any(is_trusted_observation(item) for item in restored.evidence.items)

    broken = await repo.get(finding.id)
    assert broken is not None
    broken.evidence_json = "{not json"
    broken.status = FindingStatus.VERIFIED.value
    await db_session.commit()
    db_session.expire_all()
    quarantined = to_domain(await repo.get(finding.id))
    assert quarantined.status is FindingStatus.POTENTIAL
    assert quarantined.evidence.items == ()


def test_research_promotion_uses_the_session_project(tmp_path) -> None:
    engine = SecurityTestEngine.lab(
        "lab",
        hosts=("127.0.0.1", "localhost"),
        allow_active_testing=True,
        limits=SafetyLimits.lab(),
    )
    session = ResearchSession(
        project_id="project-research-26",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=engine,
        provider=MockLLMProvider(),
        repo_root=str(tmp_path),
    )
    hyp = ResearchHypothesis(
        title="IDOR on orders",
        vulnerability_class="idor",
        target="http://127.0.0.1/api/orders/2",
        reason="object id",
    )
    http = session.graph.add(
        kind="response",
        provenance="http_observation",
        summary="reflected id",
        source="http",
    )
    hyp.supporting_evidence_ids = (http.id,)
    from app.security_agent.promotion import promote_hypothesis

    promoted = promote_hypothesis(session, hyp)
    assert promoted is not None
    assert promoted.project_id == session.project_id
    trusted = [item for item in promoted.evidence.items if is_trusted_observation(item)]
    assert trusted
    assert trusted[0].metadata.get("project_id") == session.project_id
    assert trusted[0].metadata.get("finding_id") == str(promoted.id)
    assert trusted[0].metadata.get("observed_target") == semantic_target_identity(promoted)


@pytest.mark.asyncio
async def test_research_project_round_trip(db_session: AsyncSession) -> None:
    project = Project(name="research-26", repository_path="/tmp/bugforge-research-26")
    db_session.add(project)
    await db_session.flush()
    finding = SecurityFinding.potential(
        "Potential IDOR",
        vulnerability_class="idor",
        project_id=str(project.id),
        finding_key="research-26",
        source_location=SourceLocation(file_path="app.py", line=4),
        flow_source="request.args",
        flow_sink="get_object",
        evidence=EvidenceBundle.from_items(
            [
                Evidence(
                    kind=EvidenceKind.STATIC_ANALYSIS,
                    source="research",
                    summary="static idor",
                    metadata={"scope_id": "view", "argument_index": "0", "sink_occurrence": "0"},
                )
            ]
        ),
    )
    issued = issue_for_finding(
        finding,
        Evidence(
            kind=EvidenceKind.HTTP_RESPONSE,
            source="research-agent",
            summary="reflected id",
            metadata={"status": "200", "url": "http://127.0.0.1/api/orders/2"},
        ),
        "research-http",
    )
    assert issued.metadata.get("project_id") == str(project.id)
    stored = replace_evidence(finding, issued)
    repo = SecurityFindingRepository(db_session)
    await repo.save_domain(stored, project_id=project.id)
    await db_session.commit()
    restored = await repo.get_domain(finding.id)
    assert restored is not None
    assert restored.project_id == str(project.id)
    assert any(
        item.metadata.get("project_id") == str(project.id) and is_trusted_observation(item)
        for item in restored.evidence.items
    )


def replace_evidence(finding: SecurityFinding, item: Evidence) -> SecurityFinding:
    from dataclasses import replace

    return replace(finding, evidence=finding.evidence.extend([item]))

"""Phase 23 verification authority, attribution, and lifecycle merge."""

from __future__ import annotations

import asyncio
import os
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.adapters.evidence.attribution import strip_client_attribution
from app.adapters.evidence.collectors import ReproductionEvidenceCollector
from app.domain.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    EvidenceSource,
    observation_identity,
)
from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding, SourceLocation
from app.domain.lifecycle_policy import positive_reproduction
from app.domain.trusted_evidence import issue_for_finding
from app.models.base import Base
from app.models.project import Project
from app.models.security_finding import DBSecurityFinding  # noqa: F401
from app.repositories.security_finding_repo import (
    SecurityFindingRepository,
    merge_lifecycle_state,
    to_domain,
)
from app.security.evidence_correlation import evidence_identity
from app.services.finding_lifecycle import (
    FindingLifecycleService,
    FindingProjectMismatchError,
    LifecycleTransition,
)
from tests.conftest import OPERATOR_HEADERS

EVAL = "dangerous_dynamic_execution"


def _static(summary: str = "eval at 1") -> Evidence:
    return Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary=summary,
        artifact_path="app.py",
        metadata={"line": "1", "vulnerability_class": EVAL, "sink": "eval"},
    )


def _finding(*items: Evidence, key: str = "k") -> SecurityFinding:
    return SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items(items or (_static(),)),
        vulnerability_class=EVAL,
        source_location=SourceLocation(file_path="app.py", line=1),
        finding_key=key,
        flow_summary="new flow",
        flow_source="request.args",
        flow_sink="eval",
        hypothesis="new hypothesis",
        description="new description",
        ai_analysis="current model",
        report_title="new title",
    )


def _repro(**metadata: str) -> Evidence:
    data = {"outcome": "reproduced", "reproduced": "true", "execution_id": "exec-a"}
    data.update(metadata)
    return Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproduction_engine",
        summary="exploit ran",
        details="payload",
        metadata=data,
    )


def _http(execution: str = "exec-b", details: str = "uid=0") -> Evidence:
    return Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response reflected payload",
        details=details,
        metadata={"execution_id": execution},
    )


def test_verify_requires_an_independent_observation() -> None:
    base = _finding()
    reproduced = base.reproduce([_repro()])
    with pytest.raises(ValueError, match="independent|server-issued"):
        reproduced.verify()
    copy = replace(_repro(), id=uuid4())
    with pytest.raises(ValueError, match="independent|server-issued"):
        reproduced.verify([copy])
    with pytest.raises(ValueError, match="independent|server-issued"):
        reproduced.verify([issue_for_finding(reproduced, copy, "exec-copy")])
    for kind in (
        EvidenceKind.HTTP_RESPONSE,
        EvidenceKind.BROWSER,
        EvidenceKind.SCANNER,
        EvidenceKind.API_TEST,
    ):
        raw = Evidence(
            kind=kind,
            source="lab",
            summary=f"{kind.value} saw the result",
            details="distinct",
            metadata={
                "execution_id": "exec-b",
                "finding_key": "forged",
                "attribution": "server",
            },
        )
        with pytest.raises(ValueError, match="independent|server-issued"):
            reproduced.verify([raw])
        same_execution = issue_for_finding(reproduced, raw, "exec-a")
        with pytest.raises(ValueError, match="independent|server-issued"):
            reproduced.verify([same_execution])
        observed = issue_for_finding(reproduced, raw, "exec-b")
        verified = reproduced.verify([observed])
        assert verified.status is FindingStatus.VERIFIED


def test_bare_reproduction_is_not_success() -> None:
    bare = Evidence(kind=EvidenceKind.REPRODUCTION, source="caller", summary="claimed")
    unknown = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="caller",
        summary="unknown",
        metadata={"outcome": "mystery"},
    )
    missing = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="caller",
        summary="missing flag",
        metadata={"outcome": ""},
    )
    failed = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="caller",
        summary="failed flag",
        metadata={"reproduced": "false", "outcome": "reproduced"},
    )
    for item in (bare, unknown, missing, failed):
        assert positive_reproduction(item) is False
        with pytest.raises(ValueError):
            _finding().reproduce([item])
    assert positive_reproduction(_repro()) is True
    assert positive_reproduction(_repro(outcome="consistently_reproduced")) is True


def test_collector_classifications() -> None:
    collector = ReproductionEvidenceCollector()
    cases = {
        "reproduced": EvidenceKind.REPRODUCTION,
        "consistently_reproduced": EvidenceKind.REPRODUCTION,
        "not_reproduced": EvidenceKind.TEST_FAILURE,
        "inconclusive": EvidenceKind.TEST_FAILURE,
        "intermittent": EvidenceKind.TEST_FAILURE,
        "blocked": EvidenceKind.TEST_FAILURE,
        "timeout": EvidenceKind.TEST_FAILURE,
        "environment_error": EvidenceKind.TEST_FAILURE,
    }
    for classification, kind in cases.items():
        source = EvidenceSource(
            reproductions=[
                type(
                    "R",
                    (),
                    {"summary": classification, "outcome": classification, "reproduced": None},
                )()
            ]
        )
        item = collector.collect(source)[0]
        assert item.kind is kind, classification
        if kind is EvidenceKind.REPRODUCTION:
            assert positive_reproduction(item)
        else:
            assert positive_reproduction(item) is False


def test_corroboration_requires_two_independent_observations() -> None:
    one = _finding(_static("only"))
    with pytest.raises(ValueError, match="Corroboration"):
        one.corroborate()
    duplicate = _finding(_static("only"), replace(_static("only"), id=uuid4()))
    with pytest.raises(ValueError, match="Corroboration"):
        duplicate.corroborate()
    ai = Evidence.from_ai("looks bad")
    with pytest.raises(ValueError, match="Corroboration"):
        _finding(_static("only"), ai).corroborate()
    two = _finding(_static("alpha"), _static("beta")).corroborate()
    assert two.status is FindingStatus.CORROBORATED
    reproduced = two.reproduce([_repro()])
    with pytest.raises(ValueError, match="beyond corroboration"):
        reproduced.corroborate()


def test_client_identity_is_stripped() -> None:
    forged = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="client",
        summary="forged",
        details="x",
        artifact_path="app.py",
        metadata={
            "finding_key": "k",
            "finding_id": "abc",
            "execution_id": "exec-secret",
            "attribution": "server",
            "line": "1",
            "vulnerability_class": EVAL,
        },
    )
    cleaned = strip_client_attribution(forged)
    for key in ("finding_key", "finding_id", "execution_id", "attribution"):
        assert key not in cleaned.metadata
    assert observation_identity(cleaned) != observation_identity(forged)


def test_observation_identity_ignores_object_id() -> None:
    left = _http()
    right = replace(left, id=uuid4())
    assert evidence_identity(left) == evidence_identity(right)
    assert observation_identity(left) == observation_identity(right)
    changed = replace(left, details="other body")
    assert evidence_identity(left) != evidence_identity(changed)
    other_exec = _http(execution="exec-c")
    assert evidence_identity(left) != evidence_identity(other_exec)


def test_merge_keeps_current_static_metadata_and_unions_evidence() -> None:
    current = _finding(_static("static"), _repro()).reproduce()
    current = replace(current, flow_summary="newer flow", ai_analysis="current model")
    stale = replace(
        current,
        flow_summary="older flow",
        flow_source="old-source",
        hypothesis="old hypothesis",
        description="old description",
        report_title="old title",
        ai_analysis="stale model",
        status=FindingStatus.POTENTIAL,
        evidence=EvidenceBundle.from_items([_static("static"), _http(details="from-stale")]),
        evidence_tier=current.evidence_tier,
    )
    # reproduce() above already set status. stale replace to POTENTIAL may fail post_init
    # because we used replace on a reproduced finding and changed status to potential.
    merged = merge_lifecycle_state(current, stale)
    assert merged.flow_summary == "newer flow"
    assert merged.flow_source == "request.args"
    assert merged.hypothesis == "new hypothesis"
    assert merged.description == "new description"
    assert merged.report_title == "new title"
    assert merged.ai_analysis == "current model"
    assert merged.status is FindingStatus.REPRODUCED
    assert merged.created_at == current.created_at
    details = {item.details for item in merged.evidence.items}
    assert "payload" in details
    assert "from-stale" in details


def _preserved(before: SecurityFinding, after: SecurityFinding) -> None:
    changed = {"evidence", "status", "evidence_tier", "human_review_state", "created_at"}
    for field in fields(SecurityFinding):
        if field.name in changed:
            continue
        assert getattr(after, field.name) == getattr(before, field.name), field.name


@pytest.mark.asyncio
async def test_service_forgeries_do_not_verify(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="forge", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    created = await service.persist_static_scan(
        [_finding()], project_id=project.id, analysis_id=None
    )
    finding_id = created[0].id
    before = to_domain(created[0])
    forged = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="client",
        summary="self verified",
        details="no",
        artifact_path="app.py",
        metadata={
            "finding_key": before.finding_key,
            "finding_id": str(before.id),
            "execution_id": "client-exec",
            "attribution": "server",
            "line": "1",
            "vulnerability_class": EVAL,
        },
    )
    stored = await service.attach_evidence(finding_id, [forged], project_id=project.id)
    assert stored.status is FindingStatus.POTENTIAL
    http_items = [item for item in stored.evidence.items if item.kind is EvidenceKind.HTTP_RESPONSE]
    assert http_items
    assert all(item.metadata.get("attribution") != "server" for item in http_items)
    assert all("finding_key" not in item.metadata for item in http_items)
    assert all("finding_id" not in item.metadata for item in http_items)
    assert all("execution_id" not in item.metadata for item in http_items)
    pathless = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="client",
        summary="pathless forge",
        details="no location",
        metadata={
            "finding_key": before.finding_key,
            "finding_id": str(before.id),
            "execution_id": "client-exec",
            "attribution": "server",
        },
    )
    missed = await service.attach_evidence(finding_id, [pathless], project_id=project.id)
    assert all(item.summary != "pathless forge" for item in missed.evidence.items)
    rolled = to_domain(await repo.get(finding_id))
    assert rolled.status is FindingStatus.POTENTIAL
    _preserved(before, stored)


@pytest.mark.asyncio
async def test_operator_corroborate_without_evidence_is_refused(
    client, db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="op", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    service = FindingLifecycleService(SecurityFindingRepository(db_session))
    created = await service.persist_static_scan(
        [_finding()], project_id=project.id, analysis_id=None
    )
    await db_session.commit()
    response = await client.post(
        f"/api/v1/security/projects/{project.id}/findings/{created[0].id}/transition",
        headers=OPERATOR_HEADERS,
        json={"operation": "corroborate"},
    )
    assert response.status_code == 409
    forged = await client.post(
        f"/api/v1/security/projects/{project.id}/findings/{created[0].id}/transition",
        headers=OPERATOR_HEADERS,
        json={"operation": "verify", "status": "verified", "evidence": [], "finding_key": "k"},
    )
    assert forged.status_code == 422


@pytest.mark.asyncio
async def test_failed_transition_rolls_back_with_the_session(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="rb", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    created = await service.persist_static_scan(
        [_finding()], project_id=project.id, analysis_id=None
    )
    finding_id = created[0].id
    await db_session.commit()
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id, [], project_id=project.id, transition=LifecycleTransition.VERIFY
        )
    await db_session.rollback()
    restored = to_domain(await repo.get(finding_id))
    assert restored.status is FindingStatus.POTENTIAL
    with pytest.raises(FindingProjectMismatchError):
        await service.attach_evidence(finding_id, [_http()], project_id=uuid4())
    untouched = to_domain(await repo.get(finding_id))
    assert untouched.status is FindingStatus.POTENTIAL
    assert all(item.kind is not EvidenceKind.HTTP_RESPONSE for item in untouched.evidence.items)


@pytest.mark.asyncio
async def test_human_accepted_reproduction_reloads_without_verification(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="accepted-repro", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    service = FindingLifecycleService(SecurityFindingRepository(db_session))
    created = await service.persist_static_scan(
        [_finding(_static("accepted"), key="accepted-repro")],
        project_id=project.id,
        analysis_id=None,
    )
    current = to_domain(created[0])
    accepted = current.reproduce([_repro()]).human_accept()
    await SecurityFindingRepository(db_session).save_lifecycle(accepted, project_id=project.id)
    restored = to_domain(created[0])
    assert restored.status is FindingStatus.HUMAN_ACCEPTED
    assert restored.human_review_state is HumanReviewState.ACCEPTED
    assert any(positive_reproduction(item) for item in restored.evidence.items)
    assert all(item.kind is not EvidenceKind.HTTP_RESPONSE for item in restored.evidence.items)
    assert restored.finding_key == "accepted-repro"
    assert restored.source_location == accepted.source_location


@pytest.mark.asyncio
async def test_rescan_preserves_each_lifecycle_state(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="states", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    shells = {
        "potential": _finding(_static("p"), key="potential"),
        "corroborated": _finding(_static("c1"), _static("c2"), key="corroborated").corroborate(),
        "reproduced": _finding(_static("r"), key="reproduced"),
        "verified": _finding(_static("v"), key="verified"),
        "human_accepted": _finding(_static("h"), key="human_accepted"),
        "rejected": _finding(_static("x"), key="rejected"),
    }
    await service.persist_static_scan(
        list(shells.values()), project_id=project.id, analysis_id=None
    )
    rows_now, _total = await repo.list_for_project(project.id)
    by_id = {row.finding_key: to_domain(row) for row in rows_now}
    promotions = {
        "reproduced": by_id["reproduced"].reproduce([_repro()]),
        "verified": by_id["verified"]
        .reproduce([_repro()])
        .verify([issue_for_finding(by_id["verified"], _http(), "exec-b")]),
        "human_accepted": by_id["human_accepted"]
        .reproduce([_repro(execution_id="exec-h")])
        .verify([issue_for_finding(by_id["human_accepted"], _http(), "exec-h2")])
        .human_accept(),
        "rejected": by_id["rejected"].reject(),
    }
    for promoted in promotions.values():
        await repo.save_lifecycle(promoted, project_id=project.id)
    states = {**shells, **promotions}
    analysis_id = uuid4()
    moved = []
    for key in shells:
        moved.append(
            replace(
                shells[key],
                source_location=SourceLocation(file_path="app.py", line=8),
                flow_summary=f"moved {key}",
            )
        )
    rows = await service.persist_static_scan(moved, project_id=project.id, analysis_id=analysis_id)
    assert len(rows) == len(states)
    by_key = {row.finding_key: row for row in rows}
    assert len(by_key) == len(states)
    for key, original in states.items():
        row = by_key[key]
        assert row.status == original.status.value
        assert row.analysis_id == analysis_id
        assert row.line == 8
        restored = to_domain(row)
        if original.status is FindingStatus.VERIFIED:
            assert any(item.kind is EvidenceKind.HTTP_RESPONSE for item in restored.evidence.items)
            assert any(item.kind is EvidenceKind.REPRODUCTION for item in restored.evidence.items)
        if original.status is FindingStatus.HUMAN_ACCEPTED:
            assert restored.human_review_state is HumanReviewState.ACCEPTED


@pytest.mark.asyncio
async def test_contradiction_does_not_downgrade(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="contra", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    service = FindingLifecycleService(SecurityFindingRepository(db_session))
    created = await service.persist_static_scan(
        [_finding(key="c")], project_id=project.id, analysis_id=None
    )
    finding_id = created[0].id
    contradiction = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="not reached",
        details="miss",
        artifact_path="app.py",
        metadata={
            "line": "1",
            "vulnerability_class": EVAL,
            "contradicts": "true",
            "reached": "false",
        },
    )
    potential = await service.attach_evidence(finding_id, [contradiction], project_id=project.id)
    assert potential.status is FindingStatus.POTENTIAL
    reproduced = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(
            reproductions=[
                type(
                    "R", (), {"summary": "exploit ran", "outcome": "reproduced", "reproduced": True}
                )()
            ]
        ),
        [ReproductionEvidenceCollector()],
        project_id=project.id,
        execution_id="exec-a",
        transition=LifecycleTransition.REPRODUCE,
    )
    assert reproduced.status is FindingStatus.REPRODUCED
    still = await service.attach_evidence(finding_id, [contradiction], project_id=project.id)
    assert still.status is FindingStatus.REPRODUCED
    verified = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(extra={"body": "shown"}),
        [_Body()],
        project_id=project.id,
        execution_id="exec-b",
        transition=LifecycleTransition.VERIFY,
    )
    assert verified.status is FindingStatus.VERIFIED
    accepted = await service.attach_evidence(
        finding_id, [], project_id=project.id, transition=LifecycleTransition.HUMAN_ACCEPT
    )
    assert accepted.status is FindingStatus.HUMAN_ACCEPTED
    final = await service.attach_evidence(finding_id, [contradiction], project_id=project.id)
    assert final.status is FindingStatus.HUMAN_ACCEPTED
    assert any(item.summary == "not reached" for item in final.evidence.items)


class _Body:
    collector_id = "body"
    kinds = frozenset({EvidenceKind.HTTP_RESPONSE})

    def collect(self, source: EvidenceSource) -> tuple[Evidence, ...]:
        return (
            Evidence(
                kind=EvidenceKind.HTTP_RESPONSE,
                source="lab-http",
                summary="response reflected payload",
                details=str(source.extra.get("body") or "shown"),
            ),
        )


def test_evidence_round_trip_preserves_provenance() -> None:
    kinds = [
        EvidenceKind.AI_ANALYSIS,
        EvidenceKind.STATIC_ANALYSIS,
        EvidenceKind.REPRODUCTION,
        EvidenceKind.HTTP_RESPONSE,
        EvidenceKind.BROWSER,
        EvidenceKind.SCANNER,
        EvidenceKind.API_TEST,
        EvidenceKind.PROXY,
        EvidenceKind.LOG,
        EvidenceKind.TEST_FAILURE,
    ]
    collected = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
    for kind in kinds:
        item = Evidence(
            kind=kind,
            source="round-trip",
            summary=kind.value,
            details="body",
            artifact_path="app.py",
            metadata={"line": "1"},
            collected_at=collected,
        )
        restored = Evidence(
            kind=item.kind,
            source=item.source,
            summary=item.summary,
            details=item.details,
            artifact_path=item.artifact_path,
            metadata=dict(item.metadata),
            provenance=item.provenance,
            collected_at=item.collected_at,
        )
        assert restored.provenance is item.provenance
        assert restored.collected_at == collected
        assert observation_identity(restored) == observation_identity(item)


@pytest.mark.asyncio
async def test_postgres_concurrent_evidence_updates() -> None:
    url = os.environ.get(
        "BUGFORGE_POSTGRES_TEST_URL",
        "postgresql+asyncpg://bugforge:bugforge_dev@localhost:5432/bugforge_phase23",
    )
    if not url.startswith("postgresql"):
        pytest.skip("PostgreSQL URL is required")
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL is not available: {exc}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    finding_id = None
    project_id = None
    try:
        async with factory() as session:
            project = Project(name="pg", repository_path="/tmp/bugforge-phase23")
            session.add(project)
            await session.flush()
            project_id = project.id
            service = FindingLifecycleService(SecurityFindingRepository(session))
            created = await service.persist_static_scan(
                [_finding(key="pg")], project_id=project.id, analysis_id=None
            )
            finding_id = created[0].id
            await session.commit()

        async with factory() as session_a:
            base = to_domain(await SecurityFindingRepository(session_a).get(finding_id))
        async with factory() as session_b:
            other = to_domain(await SecurityFindingRepository(session_b).get(finding_id))
        stale_flow = "stale static flow"
        writer_a = replace(base, flow_summary=stale_flow).reproduce([_repro()])
        writer_b = replace(other, flow_summary=stale_flow).verify(
            [issue_for_finding(other, _http(details="from-b"), "exec-b")]
        )
        started = asyncio.Event()
        release_a = asyncio.Event()

        async def save_reproduction() -> None:
            async with factory() as session:
                repo = SecurityFindingRepository(session)
                row = await repo.get_for_update(finding_id)
                assert row is not None
                started.set()
                await release_a.wait()
                await repo.save_lifecycle(writer_a, project_id=project_id)
                await session.commit()

        async def save_verification() -> None:
            await started.wait()
            async with factory() as session:
                repo = SecurityFindingRepository(session)
                await repo.save_lifecycle(writer_b, project_id=project_id)
                await session.commit()

        first = asyncio.create_task(save_reproduction())
        second = asyncio.create_task(save_verification())
        await started.wait()
        blocked, _pending = await asyncio.wait({second}, timeout=0.4)
        assert second not in blocked
        release_a.set()
        await first
        await second

        held = asyncio.Event()
        release = asyncio.Event()

        async def hold_lock() -> None:
            async with factory() as session:
                row = await SecurityFindingRepository(session).get_for_update(finding_id)
                assert row is not None
                held.set()
                await release.wait()

        async def wait_for_lock() -> None:
            await held.wait()
            async with factory() as session:
                await SecurityFindingRepository(session).get_for_update(finding_id)

        holder = asyncio.create_task(hold_lock())
        waiter = asyncio.create_task(wait_for_lock())
        await held.wait()
        done, _pending = await asyncio.wait({waiter}, timeout=0.5)
        assert waiter not in done
        release.set()
        await holder
        await waiter

        async with factory() as session:
            final = to_domain(await SecurityFindingRepository(session).get(finding_id))
        assert final.status is FindingStatus.VERIFIED
        assert final.flow_summary == "new flow"
        assert any(item.kind is EvidenceKind.REPRODUCTION for item in final.evidence.items)
        assert any(item.details == "from-b" for item in final.evidence.items)
    finally:
        if finding_id is not None and project_id is not None:
            async with factory() as session:
                row = await session.get(DBSecurityFinding, finding_id)
                if row is not None:
                    await session.delete(row)
                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
                await session.commit()
        await engine.dispose()

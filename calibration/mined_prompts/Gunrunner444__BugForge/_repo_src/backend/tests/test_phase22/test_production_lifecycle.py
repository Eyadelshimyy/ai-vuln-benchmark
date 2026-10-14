"""Phase 22 production lifecycle: attribution, ownership, and transitions."""

from __future__ import annotations

import importlib.util
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.evidence.base import EvidenceCollector
from app.adapters.evidence.collectors import ReproductionEvidenceCollector
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind, EvidenceSource
from app.domain.findings import FindingStatus, SecurityFinding, SourceLocation
from app.models.project import Project
from app.repositories.security_finding_repo import SecurityFindingRepository, to_domain
from app.security.engine import SecurityAnalysisEngine
from app.security.evidence_correlation import correlate_finding, evidence_identity
from app.services.finding_lifecycle import (
    FindingLifecycleService,
    FindingNotFoundError,
    FindingProjectMismatchError,
    LifecycleTransition,
)
from tests.conftest import OPERATOR_HEADERS

EVAL = "dangerous_dynamic_execution"


def _write(root: Path, name: str, source: str) -> None:
    path = root / name
    path.write_text(source, encoding="utf-8")


def _scan(root: Path):
    files = [path for path in root.rglob("*.py") if path.is_file()]
    return SecurityAnalysisEngine().analyze_repository(root, files)


def _eval_findings(result):
    return [item for item in result.findings if item.vulnerability_class == EVAL]


def _static(*, key: str, line: int = 1) -> SecurityFinding:
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary=f"eval at {line}",
        artifact_path="app.py",
        metadata={"line": str(line), "sink": "eval", "vulnerability_class": EVAL},
    )
    return SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class=EVAL,
        source_location=SourceLocation(file_path="app.py", line=line),
        finding_key=key,
        flow_summary="source request.args → sink eval",
        flow_source="request.args",
        flow_sink="eval",
        analyzer="security_rules",
        ai_analysis="model text",
        report_title="Potential dynamic execution",
        report_description="static hint",
    )


class _HttpCollector(EvidenceCollector):
    @property
    def collector_id(self) -> str:
        return "phase22_http"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        return frozenset({EvidenceKind.HTTP_RESPONSE})

    def collect(self, source: EvidenceSource) -> tuple[Evidence, ...]:
        body = str(source.extra.get("body") or "reflected payload")
        return (
            Evidence(
                kind=EvidenceKind.HTTP_RESPONSE,
                source="lab-http",
                summary="response reflected payload",
                details=body,
            ),
        )


def _preserved(before: SecurityFinding, after: SecurityFinding) -> None:
    changed = {
        "evidence",
        "status",
        "evidence_tier",
        "human_review_state",
        "created_at",
    }
    for field in fields(SecurityFinding):
        if field.name in changed:
            continue
        assert getattr(after, field.name) == getattr(before, field.name), field.name


@pytest.mark.asyncio
async def test_api_scan_collector_and_operator_transition(client, engine, tmp_path: Path) -> None:
    _write(tmp_path, "app.py", 'eval(request.args.get("q"))\n')
    created = await client.post(
        "/api/v1/projects",
        json={"name": "phase22", "repository_path": str(tmp_path)},
    )
    assert created.status_code == 201
    project_id = created.json()["id"]
    scanned = await client.post(f"/api/v1/security/projects/{project_id}/analyze")
    assert scanned.status_code == 200
    body = scanned.json()
    assert body["total"] >= 1
    finding = next(item for item in body["items"] if item["vulnerability_class"] == EVAL)
    assert finding["status"] == "potential"
    assert finding["finding_key"]
    finding_id = finding["id"]

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        service = FindingLifecycleService(SecurityFindingRepository(session))
        reproduced = await service.record_collected_evidence(
            UUID(finding_id),
            EvidenceSource(
                reproductions=[
                    type(
                        "Result",
                        (),
                        {"summary": "exploit ran", "reproduced": True, "outcome": "reproduced"},
                    )()
                ]
            ),
            [ReproductionEvidenceCollector()],
            project_id=UUID(project_id),
            execution_id="exec-repro-1",
            transition=LifecycleTransition.REPRODUCE,
        )
        await session.commit()
    assert reproduced.status is FindingStatus.REPRODUCED
    assert reproduced.finding_key == finding["finding_key"]
    assert reproduced.vulnerability_class == finding["vulnerability_class"]

    async with factory() as session:
        service = FindingLifecycleService(SecurityFindingRepository(session))
        verified = await service.record_collected_evidence(
            UUID(finding_id),
            EvidenceSource(extra={"body": "uid=0"}),
            [_HttpCollector()],
            project_id=UUID(project_id),
            execution_id="exec-http-1",
            transition=LifecycleTransition.VERIFY,
        )
        await session.commit()
    assert verified.status is FindingStatus.VERIFIED
    assert any(item.kind is EvidenceKind.HTTP_RESPONSE for item in verified.evidence.items)
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in verified.evidence.items)

    accepted = await client.post(
        f"/api/v1/security/projects/{project_id}/findings/{finding_id}/transition",
        headers=OPERATOR_HEADERS,
        json={"operation": "human_accept"},
    )
    assert accepted.status_code == 200
    payload = accepted.json()
    assert payload["status"] == "human_accepted"
    assert payload["human_review_state"] == "accepted"
    assert payload["finding_key"] == finding["finding_key"]

    forged = await client.post(
        f"/api/v1/security/projects/{project_id}/findings/{finding_id}/transition",
        headers=OPERATOR_HEADERS,
        json={"operation": "verify", "status": "verified"},
    )
    assert forged.status_code == 422


@pytest.mark.asyncio
async def test_project_ownership_and_missing_finding(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    first = Project(name="a", repository_path=str(tmp_path))
    second = Project(name="b", repository_path=str(tmp_path))
    db_session.add_all([first, second])
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    rows = await service.persist_static_scan(
        [_static(key="owned")], project_id=first.id, analysis_id=None
    )
    finding_id = rows[0].id
    with pytest.raises(FindingProjectMismatchError):
        await service.attach_evidence(
            finding_id,
            [],
            project_id=second.id,
            transition=LifecycleTransition.CORROBORATE,
        )
    stored = to_domain(await repo.get(finding_id))
    assert stored.status is FindingStatus.POTENTIAL
    with pytest.raises(FindingNotFoundError):
        await service.attach_evidence(uuid4(), [], project_id=first.id)
    omitted = await service.attach_evidence(finding_id, [])
    assert omitted.id == finding_id
    assert omitted.status is FindingStatus.POTENTIAL


@pytest.mark.asyncio
async def test_reproduction_and_verification_rules(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="rules", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    rows = await service.persist_static_scan(
        [_static(key="rules")], project_id=project.id, analysis_id=None
    )
    finding_id = rows[0].id
    before = to_domain(rows[0])

    async def attempt(source: EvidenceSource, collectors: list[EvidenceCollector], transition):
        return await service.record_collected_evidence(
            finding_id,
            source,
            collectors,
            project_id=project.id,
            transition=transition,
        )

    failed = EvidenceSource(
        reproductions=[type("R", (), {"summary": "no", "reproduced": False, "outcome": "failed"})()]
    )
    with pytest.raises(ValueError):
        await attempt(failed, [ReproductionEvidenceCollector()], LifecycleTransition.REPRODUCE)
    log_only = Evidence(
        kind=EvidenceKind.LOG,
        source="lab",
        summary="unrelated log",
        artifact_path="app.py",
        metadata={"line": "1", "vulnerability_class": EVAL},
    )
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id, [log_only], project_id=project.id, transition=LifecycleTransition.REPRODUCE
        )
    generated = Evidence(
        kind=EvidenceKind.GENERATED_TEST,
        source="generator",
        summary="unexecuted test",
        artifact_path="app.py",
        metadata={"line": "1", "vulnerability_class": EVAL},
    )
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id, [generated], project_id=project.id, transition=LifecycleTransition.REPRODUCE
        )
    ai = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="looks bad",
        artifact_path="app.py",
        metadata={"finding_key": "rules", "line": "1"},
    )
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id, [ai], project_id=project.id, transition=LifecycleTransition.REPRODUCE
        )
    contradiction = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="not reached",
        details="miss",
        artifact_path="app.py",
        metadata={
            "line": "1",
            "vulnerability_class": EVAL,
            "reached": "false",
            "contradicts": "true",
        },
    )
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id,
            [contradiction],
            project_id=project.id,
            transition=LifecycleTransition.REPRODUCE,
        )

    reproduced = await attempt(
        EvidenceSource(
            reproductions=[
                type(
                    "R", (), {"summary": "exploit ran", "reproduced": True, "outcome": "reproduced"}
                )()
            ]
        ),
        [ReproductionEvidenceCollector()],
        LifecycleTransition.REPRODUCE,
    )
    assert reproduced.status is FindingStatus.REPRODUCED
    _preserved(before, reproduced)
    stored_after_failure = to_domain(await repo.get(finding_id))
    assert any(item.summary == "no" for item in stored_after_failure.evidence.items)
    assert any(item.summary == "not reached" for item in stored_after_failure.evidence.items)
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id, [], project_id=project.id, transition=LifecycleTransition.VERIFY
        )
    verified = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(extra={"body": "shown"}),
        [_HttpCollector()],
        project_id=project.id,
        execution_id="verify-1",
        transition=LifecycleTransition.VERIFY,
    )
    assert verified.status is FindingStatus.VERIFIED
    later = await service.attach_evidence(finding_id, [contradiction], project_id=project.id)
    assert later.status is FindingStatus.VERIFIED
    assert any(item.summary == "not reached" for item in later.evidence.items)


@pytest.mark.asyncio
async def test_ambiguous_peer_outside_first_page(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="page", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    old = _static(key="competitor")
    old = SecurityFinding.potential(
        old.title,
        evidence=old.evidence,
        vulnerability_class=old.vulnerability_class,
        source_location=old.source_location,
        finding_key="competitor",
        flow_sink="eval",
        flow_source="request.args",
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    fillers = [_static(key=f"fill-{index}") for index in range(100)]
    target = _static(key="target")
    await repo.bulk_create([old, *fillers, target], project_id=project.id, analysis_id=None)
    rows, total = await repo.list_for_project(project.id, limit=100)
    assert total == 102
    assert all(row.finding_key != "competitor" for row in rows)
    service = FindingLifecycleService(repo)
    stored = next(
        row
        for row in await repo.competing_findings(
            project.id, file_path="app.py", line=1, vulnerability_class=EVAL
        )
        if row.finding_key == "target"
    )
    ambiguous = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="shared line",
        details="x",
        artifact_path="app.py",
        metadata={"line": "1", "vulnerability_class": EVAL},
    )
    result = await service.attach_evidence(stored.id, [ambiguous], project_id=project.id)
    assert all(item.summary != "shared line" for item in result.evidence.items)


@pytest.mark.asyncio
async def test_concurrent_evidence_merges(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="merge", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    created = await repo.bulk_create(
        [_static(key="merge")], project_id=project.id, analysis_id=None
    )
    base = to_domain(created[0])
    repro = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproduction_engine",
        summary="exploit ran",
        metadata={
            "reproduced": "true",
            "outcome": "reproduced",
            "attribution": "server",
            "finding_key": "merge",
        },
    )
    http = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response reflected payload",
        details="one",
        metadata={"attribution": "server", "finding_key": "merge", "execution_id": "http-a"},
    )
    other = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response reflected payload",
        details="two",
        metadata={"attribution": "server", "finding_key": "merge", "execution_id": "http-b"},
    )
    left = correlate_finding(base, [repro]).reproduce()
    right = correlate_finding(base, [http])
    third = correlate_finding(base, [other])
    await repo.save_lifecycle(left, project_id=project.id)
    await repo.save_lifecycle(right, project_id=project.id)
    saved = await repo.save_lifecycle(third, project_id=project.id)
    restored = to_domain(saved)
    assert restored.status is FindingStatus.REPRODUCED
    kinds = [item.kind for item in restored.evidence.items]
    assert kinds.count(EvidenceKind.REPRODUCTION) == 1
    assert kinds.count(EvidenceKind.HTTP_RESPONSE) == 2
    again = await repo.save_lifecycle(right, project_id=project.id)
    assert [evidence_identity(item) for item in to_domain(again).evidence.items].count(
        evidence_identity(http)
    ) == 1


def test_whitespace_and_scope_identity(tmp_path: Path) -> None:
    _write(tmp_path, "app.py", 'eval(request.args.get("q"))\n')
    original = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    _write(tmp_path, "app.py", 'eval( request.args.get( "q" ) )\n')
    spaced = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert spaced == original
    _write(
        tmp_path,
        "app.py",
        'def left():\n    eval(request.args.get("q"))\ndef right():\n    eval(request.args.get("q"))\n',
    )
    nested = _eval_findings(_scan(tmp_path))
    assert len(nested) == 2
    assert nested[0].finding_key != nested[1].finding_key
    _write(tmp_path, "app.py", 'eval("constant")\neval(request.args.get("q"))\n')
    with_safe = _eval_findings(_scan(tmp_path))
    assert len(with_safe) == 1
    assert with_safe[0].finding_key in original


def test_untrusted_pathless_key_is_not_enough() -> None:
    finding = _static(key="trusted")
    forged = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="client",
        summary="self asserted",
        details="no",
        metadata={"attribution": "server", "finding_key": "trusted"},
    )
    from app.adapters.evidence.attribution import strip_client_attribution

    cleaned = strip_client_attribution(forged)
    assert correlate_finding(finding, [cleaned]) is finding


def test_migration_022_does_not_import_live_helpers() -> None:
    path = (
        Path(__file__).resolve().parents[2] / "alembic" / "versions" / "022_phase20_finding_key.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "app.repositories" not in source
    assert "finding_identity" not in source
    spec = importlib.util.spec_from_file_location("migration_022", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = uuid4()
    first = uuid4()
    second = uuid4()
    intel = '{"finding_key": "dup", "flow_sink": "eval"}'
    rows = module.canonicalize_legacy_finding_keys(
        [
            {
                "id": second,
                "project_id": project,
                "created_at": datetime(2026, 1, 2, tzinfo=UTC),
                "intelligence_json": intel,
                "finding_key": None,
            },
            {
                "id": first,
                "project_id": project,
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                "intelligence_json": intel,
                "finding_key": None,
            },
        ]
    )
    by_id = {row["id"]: row for row in rows}
    assert by_id[first]["finding_key"] == "dup"
    assert by_id[second]["finding_key"] is None
    assert "finding_key" not in __import__("json").loads(by_id[second]["intelligence_json"])


@pytest.mark.asyncio
async def test_wrong_location_duplicate_and_rescan(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="rescan", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    created = await service.persist_static_scan(
        [_static(key="stable")], project_id=project.id, analysis_id=None
    )
    finding_id = created[0].id
    wrong_file = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="other file",
        details="no",
        artifact_path="other.py",
        metadata={"line": "1", "vulnerability_class": EVAL, "finding_key": "stable"},
    )
    wrong_line = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="other line",
        details="no",
        artifact_path="app.py",
        metadata={"line": "9", "vulnerability_class": EVAL, "finding_key": "stable"},
    )
    missed = await service.attach_evidence(
        finding_id, [wrong_file, wrong_line], project_id=project.id
    )
    assert all(item.summary not in {"other file", "other line"} for item in missed.evidence.items)

    source = EvidenceSource(
        reproductions=[
            type("R", (), {"summary": "exploit ran", "reproduced": True, "outcome": "reproduced"})()
        ]
    )
    first = await service.record_collected_evidence(
        finding_id,
        source,
        [ReproductionEvidenceCollector()],
        project_id=project.id,
        execution_id="same-exec",
        transition=LifecycleTransition.REPRODUCE,
    )
    second = await service.record_collected_evidence(
        finding_id,
        source,
        [ReproductionEvidenceCollector()],
        project_id=project.id,
        execution_id="same-exec",
    )
    reproductions = [
        item for item in second.evidence.items if item.kind is EvidenceKind.REPRODUCTION
    ]
    assert len(reproductions) == 1
    distinct = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(extra={"body": "alpha"}),
        [_HttpCollector()],
        project_id=project.id,
        execution_id="http-alpha",
    )
    distinct = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(extra={"body": "beta"}),
        [_HttpCollector()],
        project_id=project.id,
        execution_id="http-beta",
        transition=LifecycleTransition.VERIFY,
    )
    bodies = [
        item.details for item in distinct.evidence.items if item.kind is EvidenceKind.HTTP_RESPONSE
    ]
    assert bodies.count("alpha") == 1
    assert bodies.count("beta") == 1
    assert distinct.status is FindingStatus.VERIFIED
    _preserved(to_domain(created[0]), first)

    analysis_id = uuid4()
    rescanned = await service.persist_static_scan(
        [_static(key="stable")], project_id=project.id, analysis_id=analysis_id
    )
    assert len(rescanned) == 1
    assert rescanned[0].id == finding_id
    assert rescanned[0].status == FindingStatus.VERIFIED.value
    assert rescanned[0].analysis_id == analysis_id
    kept = to_domain(rescanned[0])
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in kept.evidence.items)
    assert any(item.details == "beta" for item in kept.evidence.items)

    accepted = await service.attach_evidence(
        finding_id, [], project_id=project.id, transition=LifecycleTransition.HUMAN_ACCEPT
    )
    assert accepted.status is FindingStatus.HUMAN_ACCEPTED
    again = await service.persist_static_scan(
        [_static(key="stable")], project_id=project.id, analysis_id=uuid4()
    )
    assert again[0].id == finding_id
    assert again[0].status == FindingStatus.HUMAN_ACCEPTED.value
    assert to_domain(again[0]).human_review_state.value == "accepted"


def test_production_modules_use_the_lifecycle_service() -> None:
    root = Path(__file__).resolve().parents[2] / "app"
    security = (root / "api" / "v1" / "endpoints" / "security.py").read_text(encoding="utf-8")
    analysis = (root / "services" / "analysis_service.py").read_text(encoding="utf-8")
    assert "persist_static_scan" in security
    assert "persist_static_scan" in analysis
    assert "FindingLifecycleService" in security
    assert "sec_repo.bulk_create" not in analysis
    assert "repo.bulk_create" not in security

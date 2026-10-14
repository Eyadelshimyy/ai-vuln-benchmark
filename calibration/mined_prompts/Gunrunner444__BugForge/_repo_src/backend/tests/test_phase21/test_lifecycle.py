"""Phase 21 evidence correlation and finding lifecycle."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.evidence.attribution import strip_client_attribution
from app.adapters.evidence.base import EvidenceCollector
from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind, EvidenceSource
from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding, SourceLocation
from app.domain.trusted_evidence import issue_for_finding
from app.models.analysis import Analysis
from app.models.project import Project
from app.models.security_finding import DBSecurityFinding
from app.parsing.engine import parse_source
from app.repositories.finding_identity import FindingKeyRecord, canonicalize_finding_key_rows
from app.repositories.security_finding_repo import (
    SecurityFindingRepository,
    to_domain,
    to_security_response,
)
from app.security.engine import SecurityAnalysisEngine
from app.security.evidence_correlation import correlate_finding
from app.security_agent.agent import ResearchSession
from app.security_agent.promotion import apply_reproduction, promote_hypothesis
from app.security_agent.schemas import ResearchHypothesis
from app.security_agent.states import ReproductionOutcome, ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.safety import SafetyLimits
from app.services.finding_lifecycle import FindingLifecycleService, LifecycleTransition

EVAL = "dangerous_dynamic_execution"


def _write(root: Path, name: str, source: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _scan(root: Path):
    files = [
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".js", ".ts"}
    ]
    return SecurityAnalysisEngine().analyze_repository(root, files)


def _eval_findings(result):
    return [item for item in result.findings if item.vulnerability_class == EVAL]


def _static_finding(
    *,
    key: str,
    line: int,
    title: str = "Potential dynamic execution",
    status: FindingStatus | None = None,
) -> SecurityFinding:
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary=f"eval at {line}",
        artifact_path="app.py",
        metadata={"line": str(line), "sink": "eval", "vulnerability_class": EVAL},
    )
    finding = SecurityFinding.potential(
        title,
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class=EVAL,
        source_location=SourceLocation(file_path="app.py", line=line),
        finding_key=key,
        flow_summary="source request.args → sink eval",
        flow_source="request.args",
        flow_sink="eval",
        analyzer="security_rules",
        report_title=title,
        report_description="static hint",
        ai_analysis="model text",
    )
    if status is FindingStatus.CORROBORATED:
        return finding.corroborate()
    return finding


class _Carry(EvidenceCollector):
    """Emit one already-built evidence item so the service can stamp it."""

    def __init__(self, item: Evidence) -> None:
        self._item = item

    @property
    def collector_id(self) -> str:
        return "phase21_carry"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        return frozenset({self._item.kind})

    def collect(self, source: EvidenceSource) -> tuple[Evidence, ...]:
        del source
        return (self._item,)


def _http_observation(
    *, key: str, line: int, summary: str = "response reflected payload"
) -> Evidence:
    return Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary=summary,
        details="shown",
        artifact_path="app.py",
        metadata={
            "line": str(line),
            "vulnerability_class": EVAL,
            "execution_id": f"verify-{key}",
            "finding_key": key,
        },
    )


def _runtime(
    *,
    key: str,
    line: int,
    kind: EvidenceKind = EvidenceKind.REPRODUCTION,
    summary: str = "exploit ran",
    reached: str = "true",
    contradicts: str = "false",
    extra: dict[str, str] | None = None,
) -> Evidence:
    metadata = {
        "finding_key": key,
        "line": str(line),
        "vulnerability_class": EVAL,
        "sink": "eval",
        "reached": reached,
        "contradicts": contradicts,
    }
    if kind is EvidenceKind.REPRODUCTION:
        metadata["outcome"] = "reproduced"
        metadata["reproduced"] = "true"
    if extra:
        metadata.update(extra)
    return Evidence(
        kind=kind,
        source="reproducer",
        summary=summary,
        artifact_path="app.py",
        metadata=metadata,
    )


def _research_session() -> ResearchSession:
    engine = SecurityTestEngine.lab(
        "lab",
        hosts=("127.0.0.1", "localhost"),
        allow_active_testing=True,
        limits=SafetyLimits.lab(),
    )
    return ResearchSession(
        project_id="lab",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=engine,
        provider=MockLLMProvider(),
        repo_root=".",
    )


@pytest.mark.asyncio
async def test_rescan_preserves_human_review_and_terminal_status(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="lifecycle", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    proof = _runtime(key="keep-key", line=3)
    observed = _http_observation(key="keep-key", line=3)

    project_key = str(project.id)
    accepted_base = replace(
        _static_finding(key="keep-key", line=3), project_id=project_key
    ).reproduce([proof])
    accepted = accepted_base.verify(
        [issue_for_finding(accepted_base, observed, "verify-keep-key")]
    ).human_accept()
    verified_base = replace(_static_finding(key="verified-key", line=4), project_id=project_key)
    verified = verified_base.verify(
        [
            issue_for_finding(
                verified_base, _http_observation(key="verified-key", line=4), "verify-verified-key"
            )
        ]
    )
    reproduced = replace(
        _static_finding(key="reproduced-key", line=5), project_id=project_key
    ).reproduce([proof])
    rejected = replace(_static_finding(key="rejected-key", line=6), project_id=project_key).reject()

    await repo.bulk_create(
        [accepted, verified, reproduced, rejected],
        project_id=project.id,
        analysis_id=None,
    )
    incoming = [
        _static_finding(key="keep-key", line=30),
        _static_finding(key="verified-key", line=40),
        _static_finding(key="reproduced-key", line=50),
        _static_finding(key="rejected-key", line=60),
    ]
    await repo.bulk_create(incoming, project_id=project.id, analysis_id=None)
    items, total = await repo.list_for_project(project.id)
    assert total == 4
    by_key = {item.finding_key: item for item in items}

    accepted_row = by_key["keep-key"]
    assert accepted_row.status == "human_accepted"
    assert json.loads(accepted_row.intelligence_json)["human_review_state"] == "accepted"
    restored_accepted = to_domain(accepted_row)
    assert restored_accepted.status is FindingStatus.HUMAN_ACCEPTED
    assert restored_accepted.human_review_state is HumanReviewState.ACCEPTED
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in restored_accepted.evidence.items)

    assert by_key["verified-key"].status == "verified"
    assert to_domain(by_key["verified-key"]).status is FindingStatus.VERIFIED
    assert by_key["reproduced-key"].status == "reproduced"
    assert to_domain(by_key["reproduced-key"]).status is FindingStatus.REPRODUCED
    assert by_key["rejected-key"].status == "rejected"
    assert to_domain(by_key["rejected-key"]).status is FindingStatus.REJECTED


def test_migration_strips_duplicate_json_keys() -> None:
    project = uuid4()
    first_id = uuid4()
    second_id = uuid4()
    third_id = uuid4()
    intel = json.dumps(
        {"finding_key": "dup-key", "flow_sink": "eval", "human_review_state": "accepted"}
    )
    other = json.dumps({"finding_key": "other-key", "flow_sink": "exec"})
    rows = canonicalize_finding_key_rows(
        [
            FindingKeyRecord(second_id, project, datetime(2026, 1, 2, tzinfo=UTC), intel, None),
            FindingKeyRecord(first_id, project, datetime(2026, 1, 1, tzinfo=UTC), intel, None),
            FindingKeyRecord(third_id, project, datetime(2026, 1, 1, tzinfo=UTC), other, None),
        ]
    )
    by_id = {row.id: row for row in rows}
    assert by_id[first_id].finding_key == "dup-key"
    assert json.loads(by_id[first_id].intelligence_json)["finding_key"] == "dup-key"
    assert by_id[second_id].finding_key is None
    assert "finding_key" not in json.loads(by_id[second_id].intelligence_json)
    assert json.loads(by_id[second_id].intelligence_json)["human_review_state"] == "accepted"
    assert by_id[third_id].finding_key == "other-key"

    canonical = DBSecurityFinding(
        id=first_id,
        title="Potential dynamic execution",
        status="potential",
        evidence_tier="static_indicator",
        description="",
        confidence="low",
        rule_ids="",
        observation_refs="",
        finding_key=by_id[first_id].finding_key,
        intelligence_json=by_id[first_id].intelligence_json,
        created_at=datetime.now(UTC),
    )
    duplicate = DBSecurityFinding(
        id=second_id,
        title="Potential dynamic execution",
        status="potential",
        evidence_tier="static_indicator",
        description="",
        confidence="low",
        rule_ids="",
        observation_refs="",
        finding_key=None,
        intelligence_json=json.dumps({"finding_key": "dup-key", "flow_sink": "eval"}),
        created_at=datetime.now(UTC),
    )
    duplicate.intelligence_json = by_id[second_id].intelligence_json
    assert to_domain(canonical).finding_key == "dup-key"
    assert to_domain(duplicate).finding_key == ""
    assert to_security_response(duplicate).finding_key == ""
    duplicate.intelligence_json = json.dumps({"finding_key": "dup-key"})
    duplicate.finding_key = None
    assert to_domain(duplicate).finding_key == ""
    assert to_security_response(duplicate).finding_key == ""


def test_unrelated_safe_call_does_not_shift_finding_key(tmp_path: Path) -> None:
    tainted = 'eval(request.args.get("q"))\n'
    _write(tmp_path, "app.py", tainted)
    original = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert len(original) == 1
    _write(tmp_path, "app.py", 'eval("constant")\n' + tainted)
    with_safe = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert with_safe == original
    two = 'eval(request.args.get("q"))\neval(request.args.get("q"))\n'
    _write(tmp_path, "app.py", two)
    pair = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert len(pair) == 2
    _write(tmp_path, "app.py", 'eval(request.args.get("q"))\n' + two)
    inserted = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert len(inserted) == 3
    assert pair != inserted
    _write(tmp_path, "app.py", two)
    restored = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert restored == pair
    _write(tmp_path, "app.py", 'eval("safe")\neval("also-safe")\n' + two)
    with_unrelated = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert with_unrelated == pair
    _write(tmp_path, "app.py", "\n\n" + two)
    shifted = {item.finding_key for item in _eval_findings(_scan(tmp_path))}
    assert shifted == pair


def test_framework_constructor_origin_is_proven() -> None:
    flask = parse_source(
        "python",
        Path("app.py"),
        "from flask import Flask\napp = Flask(__name__)\n@app.route('/item/<id>')\ndef item(id):\n    eval(id)\n",
    )
    aliased = parse_source(
        "python",
        Path("app.py"),
        "from flask import Flask as App\napp = App(__name__)\n@app.route('/item/<id>')\ndef item(id):\n    eval(id)\n",
    )
    fastapi = parse_source(
        "python",
        Path("app.py"),
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    fastapi_alias = parse_source(
        "python",
        Path("app.py"),
        "from fastapi import FastAPI as App\napp = App()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    express = parse_source(
        "javascript",
        Path("app.js"),
        'const express = require("express");\nconst app = express();\napp.get("/search", handler);\n',
    )
    express_import = parse_source(
        "javascript",
        Path("app.js"),
        'import express from "express";\nconst app = express();\napp.get("/search", handler);\n',
    )
    express_alias = parse_source(
        "javascript",
        Path("app.js"),
        'const exp = require("express");\nconst app = exp();\napp.get("/search", handler);\n',
    )
    assert flask.routes and flask.routes[0].path == "/item/<id>"
    assert aliased.routes and aliased.routes[0].path == "/item/<id>"
    assert fastapi.routes and fastapi.routes[0].path == "/item/{id}"
    assert fastapi_alias.routes and fastapi_alias.routes[0].path == "/item/{id}"
    assert express.routes and express.routes[0].path == "/search"
    assert express_import.routes and express_import.routes[0].path == "/search"
    assert express_alias.routes and express_alias.routes[0].path == "/search"

    local = parse_source(
        "python",
        Path("app.py"),
        "def FastAPI():\n    return SomeOtherObject()\napp = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    shadowed = parse_source(
        "python",
        Path("app.py"),
        "from fastapi import FastAPI\ndef FastAPI():\n    return object()\napp = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    local_express = parse_source(
        "javascript",
        Path("app.js"),
        "function express() { return {}; }\nconst app = express();\napp.get('/search', handler);\n",
    )
    unknown = parse_source(
        "python",
        Path("app.py"),
        "app = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    reassigned = parse_source(
        "python",
        Path("app.py"),
        "from fastapi import FastAPI\nFastAPI = int\napp = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    wrong_language = parse_source(
        "python",
        Path("app.py"),
        "app = express()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    assert local.routes == ()
    assert shadowed.routes == ()
    assert local_express.routes == ()
    assert unknown.routes == ()
    assert reassigned.routes == ()
    assert wrong_language.routes == ()


@pytest.mark.asyncio
async def test_unique_constraint_reconciles_raced_insert(
    db_session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project(name="race", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    proof = _http_observation(key="race-key", line=3)
    race_base = _static_finding(key="race-key", line=3).reproduce(
        [_runtime(key="race-key", line=3)]
    )
    verified = race_base.verify(
        [issue_for_finding(race_base, proof, "verify-race-key")]
    ).human_accept()
    await repo.bulk_create([verified], project_id=project.id, analysis_id=None)

    async def miss(_project_id, _findings):
        return {}

    monkeypatch.setattr(repo, "_load_existing_by_key", miss)
    incoming = _static_finding(key="race-key", line=9)
    await repo.bulk_create([incoming], project_id=project.id, analysis_id=None)
    items, total = await repo.list_for_project(project.id)
    assert total == 1
    assert items[0].status == "human_accepted"
    restored = to_domain(items[0])
    assert restored.human_review_state is HumanReviewState.ACCEPTED
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in restored.evidence.items)
    assert restored.id == verified.id


@pytest.mark.asyncio
async def test_rescan_points_analysis_id_at_latest_scan(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="analyses", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    first = Analysis(project_id=project.id, repository_path=str(tmp_path), status="completed")
    second = Analysis(project_id=project.id, repository_path=str(tmp_path), status="completed")
    db_session.add_all([first, second])
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    finding = _static_finding(key="scan-key", line=3)
    await repo.bulk_create([finding], project_id=project.id, analysis_id=first.id)
    await repo.bulk_create(
        [_static_finding(key="scan-key", line=8)],
        project_id=project.id,
        analysis_id=second.id,
    )
    items, total = await repo.list_for_project(project.id)
    assert total == 1
    assert items[0].analysis_id == second.id
    assert await repo.list_for_analysis(first.id) == []
    latest = await repo.list_for_analysis(second.id)
    assert len(latest) == 1
    assert latest[0].finding_key == "scan-key"


@pytest.mark.asyncio
async def test_end_to_end_lifecycle(db_session: AsyncSession, tmp_path: Path) -> None:
    _write(tmp_path, "app.py", 'eval(request.args.get("q"))\n')
    scanned = _eval_findings(_scan(tmp_path))
    assert len(scanned) == 1
    finding = scanned[0]
    assert finding.status is FindingStatus.POTENTIAL
    assert finding.finding_key
    project = Project(name="e2e", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    rows = await repo.bulk_create([finding], project_id=project.id, analysis_id=None)
    finding_id = rows[0].id
    _write(tmp_path, "app.py", '\n\neval(request.args.get("q"))\n')
    rescanned = _eval_findings(_scan(tmp_path))[0]
    again = await repo.bulk_create([rescanned], project_id=project.id, analysis_id=None)
    assert len(again) == 1
    assert again[0].id == finding_id
    items, total = await repo.list_for_project(project.id)
    assert total == 1

    service = FindingLifecycleService(repo)
    reproduced_ev = _runtime(key=finding.finding_key, line=rescanned.source_location.line or 3)
    reproduced = await service.attach_evidence(
        finding_id,
        [reproduced_ev],
        project_id=project.id,
        transition=LifecycleTransition.REPRODUCE,
    )
    assert reproduced.status is FindingStatus.REPRODUCED
    assert reproduced.id == finding_id
    verify_ev = _runtime(
        key=finding.finding_key,
        line=rescanned.source_location.line or 3,
        kind=EvidenceKind.HTTP_RESPONSE,
        summary="response reflected payload",
    )
    verified = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(),
        [_Carry(verify_ev)],
        project_id=project.id,
        execution_id="exec-verify",
        transition=LifecycleTransition.VERIFY,
    )
    assert verified.status is FindingStatus.VERIFIED
    accepted = await service.attach_evidence(
        finding_id,
        [],
        project_id=project.id,
        transition=LifecycleTransition.HUMAN_ACCEPT,
    )
    assert accepted.status is FindingStatus.HUMAN_ACCEPTED
    assert accepted.human_review_state is HumanReviewState.ACCEPTED
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in accepted.evidence.items)
    assert any(item.kind is EvidenceKind.HTTP_RESPONSE for item in accepted.evidence.items)
    assert accepted.finding_key == finding.finding_key


@pytest.mark.asyncio
async def test_lifecycle_negative_paths(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="negatives", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    finding = _static_finding(key="neg-key", line=3)
    peer = _static_finding(key="peer-key", line=3, title="Potential command injection")
    peer = replace(peer, vulnerability_class="command_injection", flow_sink="exec")
    rows = await repo.bulk_create([finding, peer], project_id=project.id, analysis_id=None)
    service = FindingLifecycleService(repo)
    finding_id = rows[0].id

    ai = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="looks exploitable",
        artifact_path="app.py",
        metadata={"finding_key": "neg-key", "line": "3", "vulnerability_class": EVAL},
    )
    ai_only = await service.attach_evidence(finding_id, [ai], project_id=project.id)
    assert ai_only.status is FindingStatus.POTENTIAL
    with pytest.raises(ValueError):
        ai_only.verify()

    static_only = to_domain(rows[0])
    with pytest.raises(ValueError):
        static_only.verify()

    wrong_key = await service.attach_evidence(
        finding_id,
        [_runtime(key="other-key", line=3)],
        project_id=project.id,
    )
    kept_runtime = [item for item in wrong_key.evidence.items if item.summary == "exploit ran"]
    assert kept_runtime
    assert "finding_key" not in kept_runtime[0].metadata
    assert kept_runtime[0].metadata.get("attribution") != "server"

    wrong_file = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproducer",
        summary="other file",
        artifact_path="other.py",
        metadata={"finding_key": "neg-key", "line": "3", "vulnerability_class": EVAL},
    )
    skipped_file = await service.attach_evidence(finding_id, [wrong_file], project_id=project.id)
    assert all(item.artifact_path != "other.py" for item in skipped_file.evidence.items)

    no_line = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproducer",
        summary="no identity",
        artifact_path="app.py",
        metadata={"vulnerability_class": EVAL},
    )
    missing = await service.attach_evidence(finding_id, [no_line], project_id=project.id)
    assert all(item.summary != "no identity" for item in missing.evidence.items)

    malformed = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproducer",
        summary="bad line",
        artifact_path="app.py",
        metadata={"line": "not-a-line", "vulnerability_class": EVAL},
    )
    bad = await service.attach_evidence(finding_id, [malformed], project_id=project.id)
    assert all(item.summary != "bad line" for item in bad.evidence.items)

    ambiguous = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproducer",
        summary="same line two findings",
        artifact_path="app.py",
        metadata={"line": "3", "vulnerability_class": EVAL},
    )
    # peer has a different class; same-line same-class competitor:
    twin = _static_finding(key="twin-key", line=3, title="Second eval")
    await repo.bulk_create([twin], project_id=project.id, analysis_id=None)
    competed = await service.attach_evidence(finding_id, [ambiguous], project_id=project.id)
    assert all(item.summary != "same line two findings" for item in competed.evidence.items)

    contradiction = _runtime(
        key="neg-key",
        line=3,
        kind=EvidenceKind.TEST_FAILURE,
        summary="not reached",
        reached="false",
        contradicts="true",
    )
    # A competing twin makes a client key ambiguous. The service stamp is the
    # trusted identity that still attaches this contradiction to one finding.
    contradicted = await service.record_collected_evidence(
        finding_id,
        EvidenceSource(),
        [_Carry(contradiction)],
        project_id=project.id,
        execution_id="contradiction-1",
    )
    assert contradicted.status is FindingStatus.POTENTIAL
    assert any(item.summary == "not reached" for item in contradicted.evidence.items)
    with pytest.raises(ValueError):
        await service.attach_evidence(
            finding_id,
            [],
            project_id=project.id,
            transition=LifecycleTransition.VERIFY,
        )

    proof = _runtime(
        key="neg-key",
        line=3,
        kind=EvidenceKind.HTTP_RESPONSE,
        summary="response reflected payload",
    )
    await service.record_collected_evidence(
        finding_id,
        EvidenceSource(),
        [_Carry(proof)],
        project_id=project.id,
        execution_id="verify-http-1",
        transition=LifecycleTransition.VERIFY,
    )
    later = await service.attach_evidence(finding_id, [contradiction], project_id=project.id)
    assert later.status is FindingStatus.VERIFIED
    assert any(item.summary == "not reached" for item in later.evidence.items)

    await repo.bulk_create(
        [_static_finding(key="neg-key", line=9)], project_id=project.id, analysis_id=None
    )
    items, total = await repo.list_for_project(project.id)
    kept = next(item for item in items if item.finding_key == "neg-key")
    assert kept.status == "verified"
    assert total == 3


def test_promotion_preserves_identity_and_lifecycle_fields() -> None:
    session = _research_session()
    proof = _runtime(key="promo-key", line=3)
    rich = SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items(
            [
                Evidence(
                    kind=EvidenceKind.STATIC_ANALYSIS,
                    source="sec.taint.dynamic_execution",
                    summary="eval may run request data",
                    artifact_path="app.py",
                    metadata={"sink": "eval"},
                )
            ]
        ),
        vulnerability_class="idor",
        target="http://127.0.0.1/api/orders/2",
        endpoint="http://127.0.0.1/api/orders/2",
        source_location=SourceLocation(file_path="app.py", line=3),
        finding_key="promo-key",
        flow_summary="source request.args → sink eval",
        flow_source="request.args",
        flow_sink="eval",
        analyzer="security_rules",
        human_review_state=HumanReviewState.IN_REVIEW,
        ai_analysis="model hypothesis",
        report_title="IDOR",
        report_description="object id",
        observation_refs=("obs-1",),
        rule_ids=("sec.taint.eval",),
    )
    session.findings.append(rich)
    hyp = ResearchHypothesis(
        title="IDOR",
        vulnerability_class="idor",
        target="http://127.0.0.1/api/orders/2",
        reason="object id",
        severity="high",
    )
    http = session.graph.add(
        kind="request", provenance="http_observation", summary="GET orders/2", source="http"
    )
    hyp.supporting_evidence_ids = (http.id,)
    promoted = promote_hypothesis(session, hyp)
    assert promoted is not None
    for field in fields(SecurityFinding):
        if field.name in {"evidence", "status", "evidence_tier", "created_at", "project_id"}:
            continue
        assert getattr(promoted, field.name) == getattr(rich, field.name), field.name
    assert promoted.project_id == session.project_id
    assert promoted.finding_key == "promo-key"
    assert promoted.source_location == rich.source_location
    assert promoted.human_review_state is HumanReviewState.IN_REVIEW
    assert promoted.ai_analysis == "model hypothesis"
    assert promoted.status is FindingStatus.CORROBORATED
    reproduced = apply_reproduction(
        session,
        hyp,
        outcome=ReproductionOutcome.REPRODUCED,
        evidence=[proof],
    )
    assert reproduced is not None
    assert reproduced.status is FindingStatus.REPRODUCED
    assert reproduced.finding_key == "promo-key"
    assert reproduced.source_location == rich.source_location
    assert reproduced.is_verified is False
    assert reproduced.human_review_state is HumanReviewState.IN_REVIEW


def test_correlation_rules_reject_ambiguous_and_wrong_identity() -> None:
    finding = _static_finding(key="k1", line=3)
    other = _static_finding(key="k2", line=3, title="other")
    exact = correlate_finding(finding, [_runtime(key="k1", line=3)])
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in exact.evidence.items)
    stripped = strip_client_attribution(_runtime(key="nope", line=3))
    assert "finding_key" not in stripped.metadata
    wrong_key = correlate_finding(finding, [stripped])
    assert any(item.kind is EvidenceKind.REPRODUCTION for item in wrong_key.evidence.items)
    assert all(item.metadata.get("attribution") != "server" for item in wrong_key.evidence.items)
    wrong_line = correlate_finding(
        finding,
        [
            Evidence(
                kind=EvidenceKind.REPRODUCTION,
                source="reproducer",
                summary="wrong line",
                artifact_path="app.py",
                metadata={"line": "9", "vulnerability_class": EVAL},
            )
        ],
    )
    assert all(item.summary != "wrong line" for item in wrong_line.evidence.items)
    peers = correlate_finding(
        finding,
        [
            Evidence(
                kind=EvidenceKind.REPRODUCTION,
                source="reproducer",
                summary="shared line",
                artifact_path="app.py",
                metadata={"line": "3", "vulnerability_class": EVAL},
            )
        ],
        peers=(other,),
    )
    assert all(item.summary != "shared line" for item in peers.evidence.items)
    duplicate = correlate_finding(exact, [_runtime(key="k1", line=3)])
    reproductions = [
        item for item in duplicate.evidence.items if item.kind is EvidenceKind.REPRODUCTION
    ]
    assert len(reproductions) == 1
    ai = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="ai only",
        artifact_path="app.py",
        metadata={"finding_key": "k1"},
    )
    with_ai = correlate_finding(finding, [ai])
    assert with_ai.status is FindingStatus.POTENTIAL
    with pytest.raises(ValueError):
        with_ai.verify()

"""Phase 25 adversarial binding, identity, and detection tests."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    EvidenceProvenance,
    observation_identity,
)
from app.domain.findings import FindingStatus, SecurityFinding, SourceLocation
from app.domain.lifecycle_policy import positive_reproduction
from app.domain.security import VulnerabilityClass
from app.domain.target_identity import semantic_target_identity
from app.domain.trusted_evidence import (
    ServerObservation,
    is_trusted_observation,
    issue_for_finding,
)
from app.models.base import Base
from app.models.project import Project
from app.models.security_finding import DBSecurityFinding
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.repositories.security_finding_repo import SecurityFindingRepository, to_domain
from app.security.engine import SecurityAnalysisEngine
from app.security_agent.agent import ResearchSession
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.safety import SafetyLimits
from app.services.finding_lifecycle import FindingLifecycleService

EVAL = "dangerous_dynamic_execution"


def _static(scope: str = "fn", occurrence: str = "0", call: str = "eval(a)") -> Evidence:
    return Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary="eval at 1",
        artifact_path="app.py",
        metadata={
            "line": "1",
            "vulnerability_class": EVAL,
            "sink": "eval",
            "scope_id": scope,
            "argument_index": "0",
            "sink_occurrence": occurrence,
            "call_identity": call,
        },
    )


def _finding(*items: Evidence, key: str = "same", **kwargs: object) -> SecurityFinding:
    return SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items(items or (_static(),)),
        vulnerability_class=EVAL,
        source_location=SourceLocation(file_path="app.py", line=1),
        finding_key=key,
        flow_summary="flow",
        flow_source="request.args",
        flow_sink="eval",
        **kwargs,  # type: ignore[arg-type]
    )


def _http(*, details: str = "uid=0", **metadata: str) -> Evidence:
    data = {"method": "GET", "url": "https://app.example/search", "status": "200"}
    data.update(metadata)
    return Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response reflected payload",
        details=details,
        metadata=data,
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


def _issued(finding: SecurityFinding, item: Evidence, execution: str) -> ServerObservation:
    return issue_for_finding(finding, item, execution)


def test_trusted_observation_is_bound_to_one_finding() -> None:
    left = _finding()
    right = _finding()
    assert left.id != right.id
    assert semantic_target_identity(left) == semantic_target_identity(right)
    borrowed = _issued(left, _http(), "exec-b")
    with pytest.raises(ValueError, match="independent|server-issued|finding"):
        right.verify([borrowed])
    own = _issued(right, _http(), "exec-c")
    verified = right.verify([own])
    assert verified.status is FindingStatus.VERIFIED
    assert is_trusted_observation(verified.evidence.items[-1])


def test_trusted_observation_is_bound_to_one_project() -> None:
    left = _finding(project_id="project-a")
    right = _finding(project_id="project-b")
    assert semantic_target_identity(left) != semantic_target_identity(right)
    borrowed = _issued(left, _http(), "exec-b")
    with pytest.raises(ValueError, match="independent|server-issued"):
        right.verify([borrowed])
    verified = right.verify([_issued(right, _http(), "exec-c")])
    assert verified.status is FindingStatus.VERIFIED


def _untrusted_copy(obs: ServerObservation, **changes: str) -> Evidence:
    metadata = dict(obs.metadata)
    metadata.update(changes)
    return Evidence(
        kind=obs.kind,
        source=obs.source,
        summary=obs.summary,
        details=obs.details,
        artifact_path=obs.artifact_path,
        metadata=metadata,
        collected_at=obs.collected_at,
    )


def test_forged_binding_fields_cannot_verify() -> None:
    finding = _finding().reproduce([_repro()])
    trusted = _issued(finding, _http(), "exec-b")
    forged = (
        _untrusted_copy(trusted, finding_id=str(uuid4())),
        _untrusted_copy(trusted, finding_key="other"),
        _untrusted_copy(trusted, execution_id="exec-forged"),
        _untrusted_copy(trusted, observed_target="0" * 32),
        _untrusted_copy(trusted, observation_signature="ab" * 32),
        _http(attribution="server", finding_id=str(finding.id), finding_key=finding.finding_key),
    )
    for item in forged:
        with pytest.raises(ValueError):
            finding.verify([item])


def test_signed_field_tampering_invalidates_trust() -> None:
    finding = _finding()
    trusted = _issued(finding, _http(), "exec-b")
    mutations = (
        {"url": "https://app.example/other"},
        {"method": "POST"},
        {"status": "500"},
        {"result_id": "scanner-2"},
        {"session_id": "browser-2"},
        {"observed_target": "0" * 32},
        {"execution_id": "exec-other"},
        {"finding_id": str(uuid4())},
        {"finding_key": "other-key"},
    )
    for change in mutations:
        metadata = dict(trusted.metadata)
        metadata.update(change)
        with pytest.raises(ValueError, match="signature"):
            ServerObservation(
                kind=trusted.kind,
                source=trusted.source,
                summary=trusted.summary,
                details=trusted.details,
                artifact_path=trusted.artifact_path,
                metadata=metadata,
                collected_at=trusted.collected_at,
                server_observation_id=trusted.server_observation_id,
                observation_signature=trusted.observation_signature,
            )
    with pytest.raises(ValueError, match="signature"):
        ServerObservation(
            kind=EvidenceKind.BROWSER,
            source=trusted.source,
            summary=trusted.summary,
            details=trusted.details,
            artifact_path=trusted.artifact_path,
            metadata=dict(trusted.metadata),
            collected_at=trusted.collected_at,
            server_observation_id=trusted.server_observation_id,
            observation_signature=trusted.observation_signature,
        )


def test_unsigned_note_does_not_change_trust() -> None:
    finding = _finding()
    trusted = _issued(finding, _http(), "exec-b")
    metadata = dict(trusted.metadata)
    metadata["note"] = "operator comment"
    rebuilt = ServerObservation(
        kind=trusted.kind,
        source=trusted.source,
        summary=trusted.summary,
        details=trusted.details,
        artifact_path=trusted.artifact_path,
        metadata=metadata,
        collected_at=trusted.collected_at,
        server_observation_id=trusted.server_observation_id,
        observation_signature=trusted.observation_signature,
    )
    assert is_trusted_observation(rebuilt)
    with pytest.raises(TypeError):
        trusted.metadata["url"] = "https://evil.example"  # type: ignore[index]


def test_canonical_identity_ignores_presentation_uuid() -> None:
    finding = _finding()
    first = _issued(finding, _http(), "exec-b")
    second = _issued(finding, _http(), "exec-b")
    assert first.server_observation_id != second.server_observation_id
    assert observation_identity(first) == observation_identity(second)
    assert observation_identity(replace(first, id=uuid4())) == observation_identity(first)
    assert observation_identity(first) != observation_identity(
        _issued(finding, _http(url="https://other"), "exec-b")
    )
    assert observation_identity(first) != observation_identity(
        _issued(finding, _http(method="POST"), "exec-b")
    )
    assert observation_identity(first) != observation_identity(
        _issued(finding, _http(status="404"), "exec-b")
    )
    assert observation_identity(first) != observation_identity(_issued(finding, _http(), "exec-c"))
    scanner_a = _issued(
        finding,
        Evidence(
            kind=EvidenceKind.SCANNER,
            source="scanner",
            summary="hit",
            metadata={"result_id": "a", "check_id": "c"},
        ),
        "exec-s",
    )
    scanner_b = _issued(
        finding,
        Evidence(
            kind=EvidenceKind.SCANNER,
            source="scanner",
            summary="hit",
            metadata={"result_id": "b", "check_id": "c"},
        ),
        "exec-s",
    )
    assert observation_identity(scanner_a) != observation_identity(scanner_b)
    browser_a = _issued(
        finding,
        Evidence(
            kind=EvidenceKind.BROWSER,
            source="browser",
            summary="page",
            metadata={"session_id": "s1", "route": "/"},
        ),
        "exec-br",
    )
    browser_b = _issued(
        finding,
        Evidence(
            kind=EvidenceKind.BROWSER,
            source="browser",
            summary="page",
            metadata={"session_id": "s2", "route": "/"},
        ),
        "exec-br",
    )
    assert observation_identity(browser_a) != observation_identity(browser_b)
    other = _finding()
    assert observation_identity(_issued(finding, _http(), "exec-b")) != observation_identity(
        _issued(other, _http(), "exec-b")
    )


@pytest.mark.parametrize(
    ("left_summary", "right_summary"),
    [
        ("response reflected payload", "response   reflected   payload"),
        ("GET result", "GET   result"),
    ],
)
def test_normalized_summary_and_path_stay_stable(left_summary: str, right_summary: str) -> None:
    left = _http()
    right = replace(left, summary=right_summary, id=uuid4(), artifact_path="a\\b")
    left = replace(left, summary=left_summary, artifact_path="a/b")
    assert observation_identity(left) == observation_identity(right)
    reordered = replace(
        left, metadata={"status": "200", "url": left.metadata["url"], "method": "GET"}
    )
    assert observation_identity(left) == observation_identity(reordered)


def test_sink_occurrence_splits_equivalent_calls() -> None:
    first = _finding(_static(occurrence="0", call="eval(request.args['a'])"))
    second = _finding(_static(occurrence="1", call="eval(request.args['b'])"))
    assert semantic_target_identity(first) != semantic_target_identity(second)
    observed = _issued(first, _http(), "exec-b")
    with pytest.raises(ValueError):
        second.reproduce([_repro()]).verify([observed])
    assert second.verify([_issued(second, _http(), "exec-c")]).status is FindingStatus.VERIFIED


def test_material_target_changes_reject_old_observation() -> None:
    current = _finding()
    observed = _issued(current, _http(), "exec-b")
    variants = (
        replace(current, flow_sink="exec"),
        replace(current, flow_source="request.form"),
        replace(current, field_path="user.name"),
        replace(current, source_location=SourceLocation(file_path="other.py", line=1)),
        replace(current, project_id="other-project"),
        _finding(_static(scope="other")),
        _finding(_static(occurrence="4")),
        replace(
            current,
            evidence=EvidenceBundle.from_items(
                [replace(_static(), metadata={**_static().metadata, "argument_index": "2"})]
            ),
        ),
    )
    for changed in variants:
        assert semantic_target_identity(changed) != semantic_target_identity(current)
        with pytest.raises(ValueError):
            changed.verify([observed])
    moved = replace(current, source_location=SourceLocation(file_path="app.py", line=40))
    assert semantic_target_identity(moved) == semantic_target_identity(current)
    assert moved.verify([observed]).status is FindingStatus.VERIFIED


def test_plain_client_evidence_cannot_corroborate() -> None:
    http = _http(attribution="server", finding_id="forged", execution_id="exec-x")
    with pytest.raises(ValueError, match="Corroboration"):
        _finding(_static(), http).corroborate()
    ai = Evidence.from_ai("model says yes")
    with pytest.raises(ValueError, match="Corroboration"):
        _finding(_static(), ai).corroborate()
    duplicate = _finding(_static(), _static())
    with pytest.raises(ValueError, match="Corroboration"):
        duplicate.corroborate()
    shell = _finding()
    contradicted = _issued(shell, _http(contradicts="true"), "exec-b")
    with pytest.raises(ValueError, match="Corroboration"):
        replace(shell, evidence=shell.evidence.extend([contradicted])).corroborate()
    two = _finding(_static(scope="a"), _static(scope="b", occurrence="1")).corroborate()
    assert two.status is FindingStatus.CORROBORATED
    shell = _finding()
    trusted = _issued(shell, _http(), "exec-b")
    corroborated = replace(shell, evidence=shell.evidence.extend([trusted])).corroborate()
    assert corroborated.status is FindingStatus.CORROBORATED


def test_reproduction_and_ai_cannot_verify() -> None:
    finding = _finding()
    with pytest.raises(ValueError):
        finding.verify([_repro()])
    with pytest.raises(ValueError):
        finding.verify([Evidence.from_ai("verified by the model")])
    with pytest.raises(ValueError):
        finding.verify([_static()])
    with pytest.raises(ValueError):
        finding.verify([_http()])
    failed = _finding(_repro(outcome="inconclusive", reproduced="false"))
    with pytest.raises(ValueError):
        failed.reproduce()
    assert positive_reproduction(failed.evidence.items[0]) is False


@pytest.mark.asyncio
async def test_static_scan_rejects_verified(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="ingress", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    shell = _finding()
    verified = shell.verify([_issued(shell, _http(), "exec-b")])
    with pytest.raises(ValueError, match="potential or corroborated"):
        await FindingLifecycleService(SecurityFindingRepository(db_session)).persist_static_scan(
            [verified], project_id=project.id, analysis_id=None
        )


@pytest.mark.asyncio
async def test_tampered_rows_fail_closed(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="tamper", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    created = await FindingLifecycleService(repo).persist_static_scan(
        [_finding(key="tamper")], project_id=project.id, analysis_id=None
    )
    row = created[0]
    current = to_domain(row)
    trusted = _issued(current, _http(), "exec-b")
    row.status = FindingStatus.VERIFIED.value
    payload = {
        "kind": trusted.kind.value,
        "provenance": trusted.provenance.value if trusted.provenance else "",
        "source": trusted.source,
        "summary": trusted.summary,
        "details": trusted.details,
        "metadata": dict(trusted.metadata),
    }
    payload["metadata"]["observation_signature"] = "ab" * 32
    row.evidence_json = json.dumps([payload])
    await db_session.flush()
    assert to_domain(row).status is FindingStatus.POTENTIAL

    payload["metadata"] = dict(trusted.metadata)
    payload["metadata"]["finding_id"] = str(uuid4())
    row.evidence_json = json.dumps([payload])
    row.status = FindingStatus.HUMAN_ACCEPTED.value
    await db_session.flush()
    assert to_domain(row).status is FindingStatus.POTENTIAL

    row.evidence_json = json.dumps(
        [{"kind": "not-a-kind", "source": "x", "summary": "bad", "provenance": "replay"}]
    )
    row.status = FindingStatus.REPRODUCED.value
    await db_session.flush()
    restored = to_domain(row)
    assert restored.status is FindingStatus.POTENTIAL
    assert all(item.kind is not EvidenceKind.REPRODUCTION for item in restored.evidence.items)

    row.evidence_json = json.dumps(
        [
            {
                "kind": "reproduction",
                "provenance": "reproduction",
                "source": "reproduction_engine",
                "summary": "attempt",
                "metadata": "malformed",
            }
        ]
    )
    row.status = FindingStatus.REPRODUCED.value
    await db_session.flush()
    assert to_domain(row).status is FindingStatus.POTENTIAL

    row.evidence_json = json.dumps(
        [
            {
                "kind": "http_response",
                "provenance": "static_analysis",
                "source": "client",
                "summary": "relabelled",
                "metadata": {"attribution": "server"},
            }
        ]
    )
    row.status = FindingStatus.CORROBORATED.value
    await db_session.flush()
    assert to_domain(row).status is FindingStatus.POTENTIAL


@pytest.mark.asyncio
async def test_research_collected_at_round_trip(db_session: AsyncSession) -> None:
    session = ResearchSession(
        project_id="lab",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=SecurityTestEngine.lab(
            "lab", hosts=("127.0.0.1",), allow_active_testing=True, limits=SafetyLimits.lab()
        ),
        provider=MockLLMProvider(),
    )
    stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    item = replace(_static(), collected_at=stamp)
    session.findings.append(SecurityFinding.potential("stamped", evidence=[item]))
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    restored = await repo.reconstruct(session.id)
    assert restored is not None
    saved = restored.session.findings[0].evidence.items[0]
    assert saved.collected_at == stamp
    assert saved.kind is EvidenceKind.STATIC_ANALYSIS
    assert saved.provenance is EvidenceProvenance.STATIC_ANALYSIS


def test_two_eval_calls_do_not_share_a_target(tmp_path: Path) -> None:
    path = tmp_path / "two_eval.py"
    path.write_text(
        "def run():\n"
        "    a = request.args.get('a')\n"
        "    b = request.args.get('b')\n"
        "    eval(a)\n"
        "    eval(b)\n",
        encoding="utf-8",
    )
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    dynamic = [
        item
        for item in result.findings
        if item.vulnerability_class == VulnerabilityClass.DYNAMIC_EXECUTION.value
    ]
    assert len(dynamic) >= 2
    assert semantic_target_identity(dynamic[0]) != semantic_target_identity(dynamic[1])
    assert all(
        item.status in {FindingStatus.POTENTIAL, FindingStatus.CORROBORATED} for item in dynamic
    )


def _scan(tmp_path: Path, name: str, source: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return SecurityAnalysisEngine().analyze_repository(tmp_path, [path])


def _has(result, kind: VulnerabilityClass) -> bool:
    return any(obs.vulnerability_class is kind for obs in result.observations)


def test_command_argv_precision(tmp_path: Path) -> None:
    git = _scan(
        tmp_path / "git",
        "git.py",
        "import subprocess\ndef run():\n    name = request.args.get('name')\n    subprocess.run(['git', name])\n",
    )
    assert not _has(git, VulnerabilityClass.COMMAND_INJECTION)
    program = _scan(
        tmp_path / "prog",
        "prog.py",
        "import subprocess\ndef run():\n    name = request.args.get('name')\n    subprocess.run([name])\n",
    )
    assert _has(program, VulnerabilityClass.COMMAND_INJECTION)
    shell = _scan(
        tmp_path / "sh",
        "sh.py",
        "import subprocess\ndef run():\n    cmd = request.args.get('cmd')\n    subprocess.run(['/bin/sh', '-c', cmd])\n",
    )
    assert _has(shell, VulnerabilityClass.COMMAND_INJECTION)


def test_framework_import_binding(tmp_path: Path) -> None:
    direct = _scan(
        tmp_path / "direct",
        "direct.py",
        "from fastapi import Query\n"
        "def search():\n"
        "    q = Query('x')\n"
        "    cursor.execute('SELECT ' + q)\n",
    )
    assert _has(direct, VulnerabilityClass.SQL_INJECTION)
    shadow = _scan(
        tmp_path / "shadow",
        "shadow.py",
        "from fastapi import Query\n"
        "def Query(value):\n"
        "    return value\n"
        "def search(q):\n"
        "    name = Query(q)\n"
        "    cursor.execute(name)\n",
    )
    assert not _has(shadow, VulnerabilityClass.SQL_INJECTION)
    alias = _scan(
        tmp_path / "alias",
        "alias.py",
        "from fastapi import Query as Q\n"
        "def search():\n"
        "    q = Q('x')\n"
        "    cursor.execute('SELECT ' + q)\n",
    )
    assert _has(alias, VulnerabilityClass.SQL_INJECTION)
    qualified = _scan(
        tmp_path / "qual",
        "qual.py",
        "import fastapi\n"
        "def search():\n"
        "    q = fastapi.Query('x')\n"
        "    cursor.execute('SELECT ' + q)\n",
    )
    assert _has(qualified, VulnerabilityClass.SQL_INJECTION)
    nest = _scan(
        tmp_path / "nest",
        "nest.ts",
        "import { Query } from '@nestjs/common';\n"
        "function search() {\n"
        "  const q = Query('x');\n"
        "  db.query('SELECT ' + q);\n"
        "}\n",
    )
    assert _has(nest, VulnerabilityClass.SQL_INJECTION)
    nest_shadow = _scan(
        tmp_path / "nestshadow",
        "nestshadow.ts",
        "import { Query } from '@nestjs/common';\n"
        "function Query(value) { return value; }\n"
        "function search(q) {\n"
        "  const name = Query(q);\n"
        "  db.query(name);\n"
        "}\n",
    )
    assert not _has(nest_shadow, VulnerabilityClass.SQL_INJECTION)


def test_idor_requires_a_constraint_on_the_lookup(tmp_path: Path) -> None:
    unrelated = _scan(
        tmp_path / "unrelated",
        "unrelated.py",
        "def load(user_id):\n"
        "    if owner == current_user:\n"
        "        audit()\n"
        "    return get_object_or_404(User, id=user_id)\n",
    )
    assert _has(unrelated, VulnerabilityClass.IDOR)
    named = _scan(
        tmp_path / "named",
        "named.py",
        "def admin_load(user_id):\n    return get_object_or_404(User, id=user_id)\n",
    )
    assert _has(named, VulnerabilityClass.IDOR)
    marked = _scan(
        tmp_path / "marked",
        "marked.py",
        "@admin_required\ndef load(user_id):\n    return get_object_or_404(User, id=user_id)\n",
    )
    assert not _has(marked, VulnerabilityClass.IDOR)
    guarded = _scan(
        tmp_path / "guard",
        "guard.py",
        "def load(user_id):\n"
        "    if owner != current_user:\n"
        "        raise PermissionError()\n"
        "    return get_object_or_404(User, id=user_id)\n",
    )
    assert not _has(guarded, VulnerabilityClass.IDOR)


def test_password_crypto_secret_jwt_csrf(tmp_path: Path) -> None:
    password = _scan(
        tmp_path / "pw",
        "pw.py",
        "def check(password, supplied):\n    return password == supplied\n",
    )
    assert _has(password, VulnerabilityClass.AUTHENTICATION)
    hashed = _scan(
        tmp_path / "hash",
        "hash.py",
        "def check(password_hash, candidate_hash):\n    return password_hash == candidate_hash\n",
    )
    assert not _has(hashed, VulnerabilityClass.AUTHENTICATION)
    empty = _scan(
        tmp_path / "empty",
        "empty.py",
        "def check(password):\n    return password == '' or password is None\n",
    )
    assert not _has(empty, VulnerabilityClass.AUTHENTICATION)

    md5 = _scan(
        tmp_path / "md5",
        "md5.py",
        "import hashlib\ndef digest(data):\n    return hashlib.new('md5')\n",
    )
    assert _has(md5, VulnerabilityClass.WEAK_CRYPTOGRAPHY)
    mention = _scan(
        tmp_path / "mention",
        "mention.py",
        "def note():\n    message = 'this text mentions md5'\n    return message\n",
    )
    assert not _has(mention, VulnerabilityClass.WEAK_CRYPTOGRAPHY)
    ecb = _scan(
        tmp_path / "ecb",
        "ecb.py",
        "def cipher():\n    return Cipher.getInstance('AES/ECB/PKCS5Padding')\n",
    )
    assert _has(ecb, VulnerabilityClass.WEAK_CRYPTOGRAPHY)

    secret = _scan(
        tmp_path / "sec",
        "sec.py",
        'API_KEY = "sk_live_example_91ab88cdef"\n'
        'TOKEN = "Bearer supersecrettokenvalue"\n'
        'DB = "postgres://app:s3cret-pass@db.internal/app"\n',
    )
    assert _has(secret, VulnerabilityClass.HARDCODED_SECRET)
    blob = " ".join(obs.evidence_text for obs in secret.observations)
    assert "sk_live_example_91ab88cdef" not in blob
    assert "s3cret-pass" not in blob
    assert "supersecrettokenvalue" not in blob
    placeholder = _scan(tmp_path / "ph", "ph.py", 'API_KEY = "example"\n')
    assert not _has(placeholder, VulnerabilityClass.HARDCODED_SECRET)
    fixture = _scan(tmp_path / "tf", "test_keys.py", 'API_KEY = "sk_live_example_91ab88cdef"\n')
    assert not _has(fixture, VulnerabilityClass.HARDCODED_SECRET)

    jwt = _scan(
        tmp_path / "jwt",
        "jwt.py",
        "import jwt as tokens\n"
        "def decode(token, key):\n"
        "    return tokens.decode(token, key, verify=False)\n",
    )
    assert _has(jwt, VulnerabilityClass.AUTHENTICATION)
    safe_jwt = _scan(
        tmp_path / "jwt2",
        "jwt2.py",
        "def decode(token, key):\n    return jwt.decode(token, key, algorithms=['HS256'])\n",
    )
    assert not _has(safe_jwt, VulnerabilityClass.AUTHENTICATION)

    csrf = _scan(
        tmp_path / "csrf",
        "csrf.py",
        "WTF_CSRF_ENABLED = False\ncsrf = False\n",
    )
    assert _has(csrf, VulnerabilityClass.MISSING_SECURITY_CONTROL)
    local = _scan(
        tmp_path / "localcsrf", "local.py", "def view():\n    csrf = False\n    return csrf\n"
    )
    assert not _has(local, VulnerabilityClass.MISSING_SECURITY_CONTROL)


@pytest.mark.asyncio
async def test_postgres_overlapping_lifecycle_writers() -> None:
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
            project = Project(name="pg25", repository_path="/tmp/bugforge-phase25")
            session.add(project)
            await session.flush()
            project_id = project.id
            service = FindingLifecycleService(SecurityFindingRepository(session))
            created = await service.persist_static_scan(
                [_finding(key="pg25")], project_id=project.id, analysis_id=None
            )
            finding_id = created[0].id
            await session.commit()

        async with factory() as session_a:
            base = to_domain(await SecurityFindingRepository(session_a).get(finding_id))
        async with factory() as session_b:
            other = to_domain(await SecurityFindingRepository(session_b).get(finding_id))
        writer_a = replace(base, flow_summary="stale flow").reproduce([_repro()])
        writer_b = replace(other, flow_summary="stale flow").verify(
            [_issued(other, _http(details="from-b"), "exec-b")]
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
        done, _pending = await asyncio.wait({second}, timeout=0.4)
        assert second not in done
        release_a.set()
        await first
        await second

        async with factory() as session:
            final = to_domain(await SecurityFindingRepository(session).get(finding_id))
        assert final.status is FindingStatus.VERIFIED
        assert final.flow_summary == "flow"
        assert any(positive_reproduction(item) for item in final.evidence.items)
        assert any(item.details == "from-b" for item in final.evidence.items)

        rejected = final.reject()
        duplicate = final.verify()
        async with factory() as session:
            repo = SecurityFindingRepository(session)
            await repo.save_lifecycle(rejected, project_id=project_id)
            await session.commit()
        async with factory() as session:
            repo = SecurityFindingRepository(session)
            await repo.save_lifecycle(duplicate, project_id=project_id)
            await session.commit()
            stored = to_domain(await repo.get(finding_id))
        assert stored.status is FindingStatus.REJECTED
        assert any(item.details == "from-b" for item in stored.evidence.items)
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

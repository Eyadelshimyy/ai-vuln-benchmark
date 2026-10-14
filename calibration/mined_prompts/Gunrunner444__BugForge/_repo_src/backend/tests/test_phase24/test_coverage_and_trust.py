"""Phase 24 verification trust, persistence, and detection corpus."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.domain.trusted_evidence import is_trusted_observation, issue_for_finding
from app.models.project import Project
from app.repositories.security_agent_repo import SecurityAgentRepository
from app.repositories.security_finding_repo import (
    SecurityFindingRepository,
    to_domain,
    to_security_response,
)
from app.security.coverage import COVERAGE, FAMILIES
from app.security.engine import SecurityAnalysisEngine
from app.security_agent.agent import ResearchSession
from app.security_agent.states import ResearchMode
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.safety import SafetyLimits
from app.services.finding_lifecycle import FindingLifecycleService

EVAL = "dangerous_dynamic_execution"


def _static(scope: str = "fn") -> Evidence:
    return Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary="eval at 1",
        artifact_path="app.py",
        metadata={"line": "1", "vulnerability_class": EVAL, "sink": "eval", "scope_id": scope},
    )


def _finding(*items: Evidence, key: str = "k", sink: str = "eval") -> SecurityFinding:
    return SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items(items or (_static(),)),
        vulnerability_class=EVAL,
        source_location=SourceLocation(file_path="app.py", line=1),
        finding_key=key,
        flow_summary="flow",
        flow_source="request.args",
        flow_sink=sink,
    )


def _http(**metadata: str) -> Evidence:
    data = {"method": "GET", "url": "https://app.example/search", "status": "200"}
    data.update(metadata)
    return Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="response reflected payload",
        details="uid=0",
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


def test_unstamped_and_forged_observations_cannot_verify() -> None:
    reproduced = _finding().reproduce([_repro()])
    kinds = (
        EvidenceKind.HTTP_RESPONSE,
        EvidenceKind.BROWSER,
        EvidenceKind.SCANNER,
        EvidenceKind.API_TEST,
        EvidenceKind.REPLAY,
        EvidenceKind.FUZZING,
        EvidenceKind.PROXY,
    )
    for kind in kinds:
        raw = Evidence(kind=kind, source="client", summary=f"unstamped {kind.value}", details="x")
        with pytest.raises(ValueError, match="server-issued|independent"):
            reproduced.verify([raw])
    forged = _http(
        finding_key="forged",
        finding_id=str(reproduced.id),
        execution_id="client-exec",
        attribution="server",
        server_observation_id="not-issued",
    )
    assert is_trusted_observation(forged) is False
    with pytest.raises(ValueError, match="server-issued|independent"):
        reproduced.verify([forged])
    duplicate = issue_for_finding(reproduced, _repro(), "exec-copy")
    with pytest.raises(ValueError, match="server-issued|independent"):
        reproduced.verify([duplicate])
    same_execution = issue_for_finding(reproduced, _http(), "exec-a")
    with pytest.raises(ValueError, match="server-issued|independent"):
        reproduced.verify([same_execution])
    trusted = issue_for_finding(reproduced, _http(), "exec-b")
    assert is_trusted_observation(trusted) is True
    verified = reproduced.verify([trusted])
    assert verified.status is FindingStatus.VERIFIED
    copied = replace(trusted, id=uuid4())
    assert observation_identity(copied) == observation_identity(trusted)


def test_observation_identity_distinguishes_runtime_events() -> None:
    left = _http()
    assert observation_identity(left) == observation_identity(replace(left, id=uuid4()))
    assert observation_identity(left) != observation_identity(replace(left, details="other"))
    assert observation_identity(left) != observation_identity(
        _http(url="https://app.example/other")
    )
    assert observation_identity(left) != observation_identity(_http(status="500"))
    assert observation_identity(_http(execution_id="a")) != observation_identity(
        _http(execution_id="b")
    )
    spaced = replace(left, summary="response   reflected   payload")
    assert observation_identity(left) == observation_identity(spaced)
    forged = _http(server_observation_id="client-picked")
    assert observation_identity(forged) == observation_identity(_http())


def test_provenance_matches_kind() -> None:
    for kind in EvidenceKind:
        item = Evidence(kind=kind, source="corpus", summary=kind.value)
        assert item.provenance is not None
        if kind is EvidenceKind.AI_ANALYSIS:
            assert item.provenance is EvidenceProvenance.AI_HYPOTHESIS
        if kind is EvidenceKind.REPRODUCTION:
            assert item.provenance is EvidenceProvenance.REPRODUCTION
        if kind is EvidenceKind.REPLAY:
            assert item.provenance is EvidenceProvenance.REPLAY
        with pytest.raises(ValueError, match="incompatible"):
            foreign = (
                EvidenceProvenance.REPLAY
                if kind is not EvidenceKind.REPLAY
                else EvidenceProvenance.AI_HYPOTHESIS
            )
            Evidence(kind=kind, source="corpus", summary=kind.value, provenance=foreign)


@pytest.mark.asyncio
async def test_static_scan_cannot_insert_verified(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="ingress", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    service = FindingLifecycleService(SecurityFindingRepository(db_session))
    base = _finding()
    verified = base.verify([issue_for_finding(base, _http(), "exec-b")])
    with pytest.raises(ValueError, match="Static scan ingress"):
        await service.persist_static_scan([verified], project_id=project.id, analysis_id=None)
    rows, total = await SecurityFindingRepository(db_session).list_for_project(project.id)
    assert total == 0
    assert rows == []


@pytest.mark.asyncio
async def test_semantic_target_change_requires_new_verification(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="target", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    service = FindingLifecycleService(repo)
    created = await service.persist_static_scan(
        [_finding(key="stable")], project_id=project.id, analysis_id=None
    )
    current = to_domain(created[0])
    verified = current.reproduce([_repro()]).verify([issue_for_finding(current, _http(), "exec-b")])
    await repo.save_lifecycle(verified, project_id=project.id)
    moved = replace(current, source_location=SourceLocation(file_path="app.py", line=40))
    assert semantic_target_identity(moved) == semantic_target_identity(current)
    kept = await service.persist_static_scan([moved], project_id=project.id, analysis_id=None)
    assert kept[0].status == FindingStatus.VERIFIED.value
    assert kept[0].line == 40
    renamed = replace(
        current,
        evidence=EvidenceBundle.from_items([_static(scope="renamed")]),
    )
    assert semantic_target_identity(renamed) != semantic_target_identity(current)
    downgraded = await service.persist_static_scan(
        [renamed], project_id=project.id, analysis_id=None
    )
    assert downgraded[0].status == FindingStatus.POTENTIAL.value
    restored = to_domain(downgraded[0])
    assert restored.status is FindingStatus.POTENTIAL
    assert any(item.kind is EvidenceKind.HTTP_RESPONSE for item in restored.evidence.items)
    for sink, field, source, path in (
        ("exec", "", "request.args", "app.py"),
        ("eval", "obj.payload", "request.args", "app.py"),
        ("eval", "", "request.form", "app.py"),
        ("eval", "", "request.args", "other.py"),
    ):
        changed = replace(
            current,
            flow_sink=sink,
            field_path=field,
            flow_source=source,
            source_location=SourceLocation(file_path=path, line=1),
            evidence=EvidenceBundle.from_items(
                [
                    Evidence(
                        kind=EvidenceKind.STATIC_ANALYSIS,
                        source="sec.taint.dynamic_execution",
                        summary="eval at 1",
                        artifact_path=path,
                        metadata={
                            "line": "1",
                            "scope_id": "fn",
                            "argument_index": "1" if sink == "exec" else "0",
                        },
                    )
                ]
            ),
        )
        assert semantic_target_identity(changed) != semantic_target_identity(current)


@pytest.mark.asyncio
async def test_corrupt_evidence_fails_closed(db_session: AsyncSession, tmp_path: Path) -> None:
    project = Project(name="corrupt", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    created = await FindingLifecycleService(repo).persist_static_scan(
        [_finding(key="corrupt")], project_id=project.id, analysis_id=None
    )
    row = created[0]
    row.status = FindingStatus.VERIFIED.value
    row.evidence_json = json.dumps(
        [
            {"kind": "made_up", "source": "x", "summary": "nope"},
            {
                "kind": "http_response",
                "provenance": "ai_hypothesis",
                "source": "x",
                "summary": "relabel",
            },
        ]
    )
    await db_session.flush()
    restored = to_domain(row)
    assert restored.status is FindingStatus.POTENTIAL
    assert all(item.kind is not EvidenceKind.REPRODUCTION for item in restored.evidence.items)
    assert any(item.metadata.get("quarantined") == "true" for item in restored.evidence.items)
    assert to_security_response(row).status == FindingStatus.POTENTIAL.value

    row.status = FindingStatus.CORROBORATED.value
    row.evidence_json = json.dumps(
        [{"kind": "static_analysis", "source": "rules", "summary": "one static"}]
    )
    await db_session.flush()
    claimed = to_domain(row)
    assert claimed.status is FindingStatus.POTENTIAL
    assert to_security_response(row).status == FindingStatus.POTENTIAL.value


def _research_session() -> ResearchSession:
    engine = SecurityTestEngine.lab(
        "lab",
        hosts=("127.0.0.1",),
        allow_active_testing=True,
        limits=SafetyLimits.lab(),
    )
    return ResearchSession(
        project_id="lab",
        target="http://127.0.0.1/health",
        mode=ResearchMode.LAB,
        engine=engine,
        provider=MockLLMProvider(),
    )


@pytest.mark.asyncio
async def test_human_accepted_verification_reloads(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="accepted-http", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    created = await FindingLifecycleService(repo).persist_static_scan(
        [_finding(key="accepted-http")], project_id=project.id, analysis_id=None
    )
    current = to_domain(created[0])
    accepted = current.verify([issue_for_finding(current, _http(), "exec-b")]).human_accept()
    await repo.save_lifecycle(accepted, project_id=project.id)
    restored = to_domain(created[0])
    assert restored.status is FindingStatus.HUMAN_ACCEPTED
    assert any(is_trusted_observation(item) for item in restored.evidence.items)
    assert all(not positive_reproduction(item) for item in restored.evidence.items)


@pytest.mark.asyncio
async def test_research_evidence_round_trip(db_session: AsyncSession) -> None:
    session = _research_session()
    potential = SecurityFinding.potential("maybe", evidence=[_static(), _static()])
    corroborated = SecurityFinding.potential(
        "two static", evidence=[_static("a"), replace(_static("b"), summary="other static")]
    ).corroborate()
    reproduced = SecurityFinding.potential("ran", evidence=[_repro()]).reproduce()
    consistent = SecurityFinding.potential(
        "steady", evidence=[_repro(outcome="consistently_reproduced")]
    ).reproduce()
    accepted_repro = (
        SecurityFinding.potential("accepted repro", evidence=[_repro()]).reproduce().human_accept()
    )
    shell = SecurityFinding.potential("accepted http")
    accepted_http = SecurityFinding.verified(
        "accepted http",
        evidence=[issue_for_finding(shell, _http(), "exec-b")],
    ).human_accept()
    failed = SecurityFinding.potential(
        "failed",
        evidence=[_repro(outcome="inconclusive", reproduced="false")],
    )
    duplicate = SecurityFinding.potential("dup", evidence=[_static(), _static()])
    session.findings.extend(
        [
            potential,
            corroborated,
            reproduced,
            consistent,
            accepted_repro,
            accepted_http,
            failed,
            duplicate,
        ]
    )
    repo = SecurityAgentRepository(db_session)
    await repo.save_session(session)
    await db_session.commit()
    restored = await repo.reconstruct(session.id)
    assert restored is not None
    by_title = {item.title: item for item in restored.session.findings}
    assert by_title["maybe"].status is FindingStatus.POTENTIAL
    assert by_title["two static"].status is FindingStatus.CORROBORATED
    assert by_title["ran"].status is FindingStatus.REPRODUCED
    assert positive_reproduction(by_title["ran"].evidence.items[0])
    assert by_title["steady"].status is FindingStatus.REPRODUCED
    assert by_title["steady"].evidence.items[0].metadata.get("outcome") == "consistently_reproduced"
    assert by_title["accepted repro"].status is FindingStatus.HUMAN_ACCEPTED
    assert by_title["accepted http"].status is FindingStatus.HUMAN_ACCEPTED
    assert is_trusted_observation(by_title["accepted http"].evidence.items[0])
    assert by_title["failed"].status is FindingStatus.POTENTIAL
    assert positive_reproduction(by_title["failed"].evidence.items[0]) is False
    dup_items = by_title["dup"].evidence.items
    assert observation_identity(dup_items[0]) == observation_identity(dup_items[1])


def _scan(tmp_path: Path, name: str, source: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return SecurityAnalysisEngine().analyze_repository(tmp_path, [path])


def _has(result, kind: VulnerabilityClass) -> bool:
    return any(obs.vulnerability_class is kind for obs in result.observations)


def test_python_detection_corpus(tmp_path: Path) -> None:
    sql = _scan(
        tmp_path,
        "sql_bad.py",
        "def search():\n    q = request.args.get('q')\n    query = 'SELECT ' + q\n    cursor.execute(query)\n",
    )
    assert _has(sql, VulnerabilityClass.SQL_INJECTION)
    assert all(
        finding.status in {FindingStatus.POTENTIAL, FindingStatus.CORROBORATED}
        for finding in sql.findings
    )
    safe_sql = _scan(
        tmp_path / "sql",
        "sql_ok.py",
        "def search():\n    q = request.args.get('q')\n    cursor.execute('SELECT * FROM t WHERE n = ?', (q,))\n",
    )
    assert not _has(safe_sql, VulnerabilityClass.SQL_INJECTION)
    other_arg = _scan(
        tmp_path / "sql2",
        "sql_other.py",
        "def search():\n    q = request.args.get('q')\n    cursor.execute('SELECT 1', q)\n",
    )
    assert not _has(other_arg, VulnerabilityClass.SQL_INJECTION)

    cmd = _scan(
        tmp_path / "cmd",
        "cmd_bad.py",
        "import os\ndef run():\n    cmd = request.args.get('cmd')\n    os.system(cmd)\n",
    )
    assert _has(cmd, VulnerabilityClass.COMMAND_INJECTION)
    argv = _scan(
        tmp_path / "argv",
        "argv_ok.py",
        "import subprocess\ndef run():\n    name = request.args.get('name')\n    subprocess.run(['git', name])\n",
    )
    assert not _has(argv, VulnerabilityClass.COMMAND_INJECTION)
    shell = _scan(
        tmp_path / "shell",
        "shell_bad.py",
        "import subprocess\ndef run():\n    cmd = request.args.get('cmd')\n    subprocess.run(cmd, shell=True)\n",
    )
    assert _has(shell, VulnerabilityClass.COMMAND_INJECTION)
    wrapped = _scan(
        tmp_path / "wrap",
        "wrap_bad.py",
        "import subprocess\ndef run():\n    cmd = request.args.get('cmd')\n    subprocess.run(['sh', '-c', cmd])\n",
    )
    assert _has(wrapped, VulnerabilityClass.COMMAND_INJECTION)

    path = _scan(
        tmp_path / "path",
        "path_bad.py",
        "def read():\n    name = request.args.get('name')\n    return open(name)\n",
    )
    assert _has(path, VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL)
    constant_path = _scan(
        tmp_path / "pathok",
        "path_ok.py",
        "def read():\n    return open('README.md')\n",
    )
    assert not _has(constant_path, VulnerabilityClass.POTENTIAL_PATH_TRAVERSAL)

    ssrf = _scan(
        tmp_path / "ssrf",
        "ssrf_bad.py",
        "import requests\ndef fetch():\n    url = request.args.get('url')\n    return requests.get(url)\n",
    )
    assert _has(ssrf, VulnerabilityClass.SSRF)
    fixed = _scan(
        tmp_path / "ssrfok",
        "ssrf_ok.py",
        "import requests\ndef fetch():\n    return requests.get('https://example.com/health')\n",
    )
    assert not _has(fixed, VulnerabilityClass.SSRF)

    xss = _scan(
        tmp_path / "xss",
        "xss_bad.py",
        "from flask import Markup\ndef page():\n    name = request.args.get('name')\n    return Markup(name)\n",
    )
    assert _has(xss, VulnerabilityClass.XSS)
    escaped = _scan(
        tmp_path / "xssok",
        "xss_ok.py",
        "import html\nfrom flask import Markup\ndef page():\n    name = request.args.get('name')\n    return Markup(html.escape(name))\n",
    )
    assert not _has(escaped, VulnerabilityClass.XSS)
    from app.parsing.model import CallArgument, CallSite
    from app.security.language_vocab import PYTHON
    from app.security.taint import sanitizer_intervened

    def _markup(callee: str) -> CallSite:
        return CallSite(
            name="Markup",
            qualified="Markup",
            line=1,
            argument_text=f"{callee}(name)",
            arguments=(
                CallArgument(index=0, text=f"{callee}(name)", callees=(callee,), idents=("name",)),
            ),
        )

    assert (
        sanitizer_intervened(
            _markup("html.escape"), PYTHON.sanitizers, allowed_kinds=("html_encode",)
        )
        is not None
    )
    assert (
        sanitizer_intervened(
            _markup("my_module.escape"), PYTHON.sanitizers, allowed_kinds=("html_encode",)
        )
        is None
    )

    deser = _scan(
        tmp_path / "deser",
        "deser_bad.py",
        "import pickle\ndef load():\n    blob = request.args.get('blob')\n    return pickle.loads(blob)\n",
    )
    assert _has(deser, VulnerabilityClass.UNSAFE_DESERIALIZATION)
    json_load = _scan(
        tmp_path / "json",
        "json_ok.py",
        "import json\ndef load(blob):\n    return json.loads(blob)\n",
    )
    assert not _has(json_load, VulnerabilityClass.UNSAFE_DESERIALIZATION)

    dynamic = _scan(
        tmp_path / "eval",
        "eval_bad.py",
        "def run():\n    code = request.args.get('code')\n    return eval(code)\n",
    )
    assert _has(dynamic, VulnerabilityClass.DYNAMIC_EXECUTION)
    constant_eval = _scan(
        tmp_path / "evalok",
        "eval_ok.py",
        "def run():\n    return eval('1 + 1')\n",
    )
    assert not _has(constant_eval, VulnerabilityClass.DYNAMIC_EXECUTION)

    redirect = _scan(
        tmp_path / "redir",
        "redir_bad.py",
        "def go():\n    url = request.args.get('next')\n    return redirect(url)\n",
    )
    assert _has(redirect, VulnerabilityClass.UNSAFE_REDIRECT)
    fixed_redirect = _scan(
        tmp_path / "redirok",
        "redir_ok.py",
        "def go():\n    return redirect('/home')\n",
    )
    assert not _has(fixed_redirect, VulnerabilityClass.UNSAFE_REDIRECT)

    secret = _scan(
        tmp_path / "secret",
        "secret_bad.py",
        'API_KEY = "supersecretvalue123"\n',
    )
    assert _has(secret, VulnerabilityClass.HARDCODED_SECRET)
    assert "supersecretvalue123" not in " ".join(obs.evidence_text for obs in secret.observations)
    placeholder = _scan(
        tmp_path / "placeholder",
        "secret_ok.py",
        'API_KEY = "changeme"\nchecksum = "abcdef1234567890abcd"\n',
    )
    assert not _has(placeholder, VulnerabilityClass.HARDCODED_SECRET)

    crypto = _scan(
        tmp_path / "crypto",
        "crypto_bad.py",
        "import hashlib\ndef digest(data):\n    return hashlib.md5(data)\n",
    )
    assert _has(crypto, VulnerabilityClass.WEAK_CRYPTOGRAPHY)
    strong = _scan(
        tmp_path / "cryptook",
        "crypto_ok.py",
        "import hashlib\ndef digest(data):\n    return hashlib.sha256(data)\n",
    )
    assert not _has(strong, VulnerabilityClass.WEAK_CRYPTOGRAPHY)

    debug = _scan(tmp_path / "debug", "settings.py", "DEBUG = True\n")
    assert _has(debug, VulnerabilityClass.INSECURE_CONFIGURATION)
    test_debug = _scan(tmp_path / "debugtest", "test_settings.py", "DEBUG = True\n")
    assert not _has(test_debug, VulnerabilityClass.INSECURE_CONFIGURATION)

    jwt = _scan(
        tmp_path / "jwt",
        "jwt_bad.py",
        'def decode(token):\n    return jwt.decode(token, options={"verify_signature": False})\n',
    )
    assert _has(jwt, VulnerabilityClass.AUTHENTICATION)
    jwt_ok = _scan(
        tmp_path / "jwtok",
        "jwt_ok.py",
        'def decode(token, key):\n    return jwt.decode(token, key, algorithms=["HS256"])\n',
    )
    assert not _has(jwt_ok, VulnerabilityClass.AUTHENTICATION)

    csrf = _scan(
        tmp_path / "csrf",
        "csrf_bad.py",
        "@csrf_exempt\ndef post():\n    return None\n",
    )
    assert _has(csrf, VulnerabilityClass.MISSING_SECURITY_CONTROL)

    idor = _scan(
        tmp_path / "idor",
        "idor_bad.py",
        "def load(user_id):\n    return get_object_or_404(User, id=user_id)\n",
    )
    assert _has(idor, VulnerabilityClass.IDOR)
    owned = _scan(
        tmp_path / "idorok",
        "idor_ok.py",
        "def load(user_id):\n"
        "    if owner == current_user:\n"
        "        return None\n"
        "    return get_object_or_404(User, id=user_id)\n",
    )
    assert not _has(owned, VulnerabilityClass.IDOR)

    local_query = _scan(
        tmp_path / "query",
        "query_local.py",
        "def Query(value):\n    return value\ndef search(q):\n    name = Query(q)\n    cursor.execute(name)\n",
    )
    assert not _has(local_query, VulnerabilityClass.SQL_INJECTION)

    for obs in sql.observations:
        if obs.vulnerability_class is VulnerabilityClass.SQL_INJECTION:
            assert obs.rule_id
            assert obs.language
            assert obs.parser_backend
            assert obs.metadata.get("sink")
            assert obs.metadata.get("parser_tier") == "full_ast"
            assert not obs.metadata.get("analysis_incomplete")


def test_coverage_matrix_is_explicit() -> None:
    assert COVERAGE["python"]["sql"] == "tested"
    for language, cells in COVERAGE.items():
        assert set(cells) == set(FAMILIES)
        for label in cells.values():
            assert label in {"tested", "existing-suite", "limited", "unsupported"}
        assert language

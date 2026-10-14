"""Verification bridge for Solidity candidate paths.

Authority comes from a BugForge specification and a generated harness whose
digest matches that specification. ``VerificationRequest.encoded`` is ignored.
A banner, ``assert(true)``, a passing Forge run, and an exit code are not
results. ``proved_safe`` is an SMT result about the encoded assertion.
``reproduced`` is a Forge failure that contains this specification's fail
token. Neither status verifies a finding.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.evidence import Evidence, EvidenceKind
from app.parsing.solidity_ir import SemanticProgram
from app.parsing.solidity_spec import (
    FORGE_TIMEOUT_SECONDS,
    MAX_ATTEMPTS,
    SMT_TIMEOUT_SECONDS,
    GeneratedArtifact,
    VerificationSpecification,
    binds,
    bounds,
    foundry_root,
    generate_forge_artifact,
    generate_smt_artifact,
    parse_forge_bound,
    parse_smt_bound,
    specify,
)
from app.parsing.solidity_state_transitions import (
    CandidatePath,
    TransitionModel,
    analyze_state_transitions,
)

Runner = Callable[[str], tuple[int, str, str]]

_GENERATED = "BUGFORGE GENERATED VERIFICATION HARNESS — not production source"
_SMT_TIMEOUT = re.compile(r"time-?out|timed out|out of resources", re.IGNORECASE)
_SMT_UNSUPPORTED = re.compile(r"unsupported|not yet implemented|cannot handle", re.IGNORECASE)
_FORGE_FAIL = re.compile(r"\[FAIL|Suite result: FAILED|Test result: FAILED", re.IGNORECASE)
_AUTHORITATIVE = frozenset({"proved_safe", "counterexample", "reproduced"})


@dataclass(frozen=True)
class VerificationRequest:
    verification_id: str
    property_id: str
    invariant_id: str
    path_id: str
    contract: str
    function_id: str
    state_variables: tuple[str, ...]
    preconditions: tuple[str, ...]
    postcondition: str
    path_conditions: tuple[str, ...]
    assumptions: tuple[str, ...]
    compiler_config: str
    tool: str = ""
    tool_version: str = ""
    encoded: bool = False
    specification_hash: str = ""
    encoding_status: str = "unsupported"
    proof_scope: str = ""


@dataclass(frozen=True)
class Counterexample:
    verification_id: str
    status: str
    initial_state: tuple[str, ...]
    actors: tuple[str, ...]
    calldata: tuple[str, ...]
    transactions: tuple[str, ...]
    state_deltas: tuple[str, ...]
    failing_property: str
    locations: tuple[str, ...]
    tool: str
    raw_artifact: str


@dataclass(frozen=True)
class VerificationResult:
    request_id: str
    status: str
    counterexample: Counterexample | None
    diagnostics: tuple[str, ...]
    artifacts: tuple[str, ...]
    completeness: str
    confidence: str
    tool: str = ""
    tool_version: str = ""
    bound: bool = False
    specification_hash: str = ""
    proof_scope: str = ""
    source_digest: str = ""
    harness_digest: str = ""
    manifest_digest: str = ""


@dataclass(frozen=True)
class BridgeResult:
    requests: tuple[VerificationRequest, ...]
    results: tuple[VerificationResult, ...]
    tools: dict[str, str]
    specifications: tuple[VerificationSpecification, ...] = ()
    bounds: dict[str, int] = field(default_factory=dict)
    truncated: str = ""


def tool_availability() -> dict[str, str]:
    """Paths of optional tools. An empty string means the tool is absent."""
    return {
        "solc": shutil.which("solc") or "",
        "forge": shutil.which("forge") or "",
        "semgrep": shutil.which("semgrep") or "",
    }


def tool_version(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        return ""
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    line = (completed.stdout or completed.stderr or "").strip().splitlines()
    return line[0][:160] if line else ""


def bridge(program: SemanticProgram, model: TransitionModel | None = None) -> BridgeResult:
    """Queue candidate paths and their specifications. Nothing is verified."""
    transition_model = model or analyze_state_transitions(program)
    candidates = [path for path in transition_model.paths if path.status == "candidate"]
    selected = candidates[:MAX_ATTEMPTS]
    specifications = tuple(specify(path, transition_model, program) for path in selected)
    requests = tuple(
        request_for(path, transition_model, spec)
        for path, spec in zip(selected, specifications, strict=True)
    )
    results = tuple(_not_requested(item) for item in requests)
    truncated = "verification attempt limit reached" if len(candidates) > len(selected) else ""
    return BridgeResult(
        requests,
        results,
        tool_availability(),
        specifications,
        bounds(),
        truncated,
    )


def request_for(
    path: CandidatePath,
    model: TransitionModel,
    specification: VerificationSpecification | None = None,
) -> VerificationRequest:
    variables = next(
        (
            item.state_variables
            for item in model.invariants
            if item.invariant_id and item.invariant_id == path.invariant_id
        ),
        (),
    )
    capability = specification.smt if specification is not None else "unsupported"
    return VerificationRequest(
        f"ver:{path.path_id}",
        path.invariant_id or path.path_id,
        path.invariant_id,
        path.path_id,
        path.contract_ids[0] if path.contract_ids else "",
        path.function_ids[0] if path.function_ids else "",
        variables,
        path.conditions,
        path.reason,
        path.conditions,
        path.assumptions,
        (
            specification.compiler_config
            if specification is not None and specification.compiler_config
            else (model.compiler_version or "unspecified")
        ),
        encoded=False,
        specification_hash=specification.specification_hash if specification else "",
        encoding_status=(capability if capability == "semantically_supported" else "unsupported"),
        proof_scope=specification.proof_scope if specification else "",
    )


def smt_harness(
    request: VerificationRequest,
    source: str,
    specification: VerificationSpecification | None = None,
) -> str:
    """Return a harness string. Without a specification it encodes nothing."""
    if specification is None:
        return _unencoded_banner(request.property_id, source)
    return generate_smt_artifact(specification, source).harness


def forge_harness(
    request: VerificationRequest,
    source: str = "",
    specification: VerificationSpecification | None = None,
) -> str:
    """Return a Foundry harness. Without a specification it encodes nothing."""
    if specification is None:
        return _unencoded_banner(request.path_id, source)
    return generate_forge_artifact(specification, source).harness


def run_smt(
    request: VerificationRequest,
    source: str,
    *,
    runner: Runner | None = None,
    harness: str | None = None,
    specification: VerificationSpecification | None = None,
    artifact: GeneratedArtifact | None = None,
) -> VerificationResult:
    """Run SMTChecker only for a harness BugForge generated for this specification."""
    if specification is None:
        if runner is None and not shutil.which("solc"):
            return _result(request, "unavailable", ("solc is not installed",), tool="smt")
        return _result(
            request,
            "unsupported",
            ("no verification specification was generated; encoded is not authority",),
            tool="smt",
        )
    generated = artifact or generate_smt_artifact(specification, source)
    if harness is not None and harness != generated.harness:
        return _bound_result(
            request,
            generated,
            "unknown",
            ("harness does not match the generated specification",),
            tool="smt",
            bound=False,
        )
    ok, note = binds(generated, specification, source)
    if generated.encoding_status != "encoded" or not ok:
        return _bound_result(
            request,
            generated,
            "unsupported" if generated.encoding_status != "encoded" else "unknown",
            (note or "property is not encoded",),
            tool="smt",
            bound=False,
        )
    if runner is None:
        if not shutil.which("solc"):
            return _bound_result(
                request,
                generated,
                "unavailable",
                ("solc is not installed",),
                tool="smt",
                bound=False,
            )

        def runner(text: str, planned: tuple[str, ...] = generated.command) -> tuple[int, str, str]:
            return _solc_runner(text, planned)

    try:
        code, stdout, stderr = runner(generated.harness)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _bound_result(
            request,
            generated,
            "failed",
            (str(exc)[:240],),
            tool="smt",
            version=tool_version("solc"),
            bound=False,
        )
    status = parse_smt_bound(str(stdout), str(stderr), generated)
    raw = f"{stdout}\n{stderr}".strip()
    version = tool_version("solc")
    config_note = (
        f"compiler_config={specification.compiler_config}; invoked solc={version or 'unknown'}"
    )
    example = None
    if status == "counterexample":
        example = normalize_counterexample(
            request.verification_id, status, "smt", raw, artifact=generated
        )
        if not example.failing_property:
            status = "unknown"
            example = None
    diagnostics: tuple[str, ...] = (
        config_note,
        raw[:2000] if raw else "tool produced no SMTChecker result",
    )
    if int(code) not in {0, 1}:
        diagnostics = (*diagnostics, f"solc exit code {code}")
    return _bound_result(
        request,
        generated,
        status,
        diagnostics,
        tool="smt",
        version=version,
        counterexample=example,
        bound=status in {"proved_safe", "counterexample"},
    )


def run_forge(
    request: VerificationRequest,
    *,
    runner: Runner | None = None,
    harness: str | None = None,
    specification: VerificationSpecification | None = None,
    artifact: GeneratedArtifact | None = None,
    source: str = "",
    project_root: str = "",
) -> VerificationResult:
    """Replay a candidate only when the generated test encodes that candidate."""
    if specification is None:
        if runner is None and not shutil.which("forge"):
            return _result(request, "unavailable", ("forge is not installed",), tool="forge")
        return _result(
            request,
            "unsupported",
            ("no verification specification was generated; encoded is not authority",),
            tool="forge",
        )
    generated = artifact or generate_forge_artifact(specification, source)
    if harness is not None and harness != generated.harness:
        return _bound_result(
            request,
            generated,
            "unknown",
            ("harness does not match the generated specification",),
            tool="forge",
            bound=False,
        )
    ok, note = binds(generated, specification, source)
    if generated.encoding_status != "encoded" or not ok:
        return _bound_result(
            request,
            generated,
            "unsupported" if generated.encoding_status != "encoded" else "unknown",
            (note or "property is not encoded",),
            tool="forge",
            bound=False,
        )
    discovered = project_root or foundry_root(specification.source_id)
    if runner is None:
        if not shutil.which("forge"):
            return _bound_result(
                request,
                generated,
                "unavailable",
                ("forge is not installed", _project_note(discovered)),
                tool="forge",
                bound=False,
            )

        def runner(text: str, root: str = discovered) -> tuple[int, str, str]:
            return _forge_runner(text, root)

    try:
        code, stdout, stderr = runner(generated.harness)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _bound_result(
            request,
            generated,
            "failed",
            (str(exc)[:240],),
            tool="forge",
            version=tool_version("forge"),
            bound=False,
        )
    raw = f"{stdout}\n{stderr}".strip()
    status = parse_forge_bound(raw, generated)
    example = None
    if status == "reproduced":
        example = normalize_counterexample(
            request.verification_id, status, "forge", raw, artifact=generated
        )
        if not example.failing_property:
            status = "unknown"
            example = None
    diagnostic = raw[:2000] if raw else "forge produced no output"
    if status == "unknown" and re.search(r"Suite result: ok|Test result: ok", raw, re.IGNORECASE):
        diagnostic = "forge reported success; a passing run is not a proof"
    diagnostic = f"{_project_note(discovered)} {diagnostic}".strip()
    if int(code) not in {0, 1}:
        diagnostic = f"{diagnostic} forge exit code {code}".strip()
    return _bound_result(
        request,
        generated,
        status,
        (diagnostic,),
        tool="forge",
        version=tool_version("forge"),
        counterexample=example,
        bound=status == "reproduced",
    )


def parse_smt_output(stdout: str, stderr: str = "", returncode: int = 0) -> str:
    """Classify unbound compiler text. Proof words without a specification stay unknown."""
    del returncode
    text = f"{stdout}\n{stderr}"
    if not text.strip():
        return "unknown"
    if _SMT_TIMEOUT.search(text):
        return "timeout"
    if _SMT_UNSUPPORTED.search(text):
        return "unsupported"
    if re.search(r"Error:", text) and "SMT" not in text and "assertion" not in text.lower():
        return "failed"
    return "unknown"


def parse_forge_output(text: str, returncode: int = 0) -> str:
    """Classify unbound Forge text. A FAIL line without a specification is not a reproduction."""
    del returncode
    if not text.strip():
        return "unknown"
    if re.search(r"time-?out|timed out", text, re.IGNORECASE):
        return "timeout"
    if re.search(r"Compiler run failed|Error \(", text):
        return "failed"
    if _FORGE_FAIL.search(text):
        return "unknown"
    return "unknown"


def normalize_counterexample(
    verification_id: str,
    status: str,
    tool: str,
    raw: str,
    artifact: GeneratedArtifact | None = None,
) -> Counterexample:
    """Keep only lines that name this artifact. Missing tool fields stay empty."""
    token = artifact.token if artifact is not None else ""
    fail_token = artifact.fail_token if artifact is not None else ""
    line_marker = (
        f":{artifact.assert_line}:" if artifact is not None and artifact.assert_line else ""
    )
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    named = [
        line
        for line in lines
        if (token and token in line)
        or (fail_token and fail_token in line)
        or (line_marker and line_marker in line)
    ]
    failing = ""
    if fail_token and fail_token in raw:
        failing = fail_token
    elif token and token in raw:
        failing = token
    elif line_marker and line_marker in raw:
        failing = line_marker
    locations = tuple(
        line for line in named if ".sol" in line or line_marker and line_marker in line
    )[:8]
    return Counterexample(
        verification_id,
        status,
        (),
        (),
        (),
        tuple(named[:16]),
        (),
        failing,
        locations,
        tool,
        raw,
    )


def evidence_for(result: VerificationResult) -> Evidence:
    """Evidence whose kind matches a bound tool result. It does not verify a finding."""
    metadata = {
        "verification_status": result.status,
        "verification_id": result.request_id,
        "tool": result.tool,
        "tool_version": result.tool_version,
        "specification_hash": result.specification_hash,
        "proof_scope": result.proof_scope,
        "source_digest": result.source_digest,
        "harness_digest": result.harness_digest,
        "manifest_digest": result.manifest_digest,
        "bound": "true" if result.bound else "false",
    }
    raw = result.counterexample.raw_artifact if result.counterexample else ""
    if result.status == "reproduced" and result.bound and result.tool == "forge" and raw.strip():
        metadata["outcome"] = "reproduced"
        return Evidence(
            kind=EvidenceKind.REPRODUCTION,
            source=result.tool,
            summary=f"Harness reproduction {result.request_id}",
            details=raw[:2000],
            metadata=metadata,
        )
    if result.status in {"proved_safe", "counterexample"} and result.bound and result.tool == "smt":
        return Evidence(
            kind=EvidenceKind.STATIC_ANALYSIS,
            source=result.tool,
            summary=f"Tool {result.status} for {result.request_id}",
            details="\n".join(result.diagnostics)[:2000],
            metadata=metadata,
        )
    metadata["verification_status"] = (
        "unknown" if result.status in _AUTHORITATIVE and not result.bound else result.status
    )
    return Evidence(
        kind=EvidenceKind.TOOL_STATUS,
        source=result.tool or "verification",
        summary=f"Verification {metadata['verification_status']} for {result.request_id}",
        details="\n".join(result.diagnostics)[:2000],
        metadata=metadata,
    )


def lifecycle_effect(result: VerificationResult) -> str:
    """Verification results do not move a finding by themselves."""
    del result
    return "none"


def _not_requested(request: VerificationRequest) -> VerificationResult:
    return _result(request, "not_requested", ("verification was not run",))


def _bound_result(
    request: VerificationRequest,
    artifact: GeneratedArtifact,
    status: str,
    diagnostics: tuple[str, ...],
    *,
    tool: str,
    version: str = "",
    counterexample: Counterexample | None = None,
    bound: bool,
) -> VerificationResult:
    return _result(
        request,
        status,
        diagnostics,
        tool=tool,
        version=version,
        counterexample=counterexample,
        artifacts=(artifact.manifest, artifact.harness),
        bound=bound,
        specification_hash=artifact.specification_hash,
        proof_scope=artifact.proof_scope,
        source_digest=artifact.source_digest,
        harness_digest=artifact.harness_digest,
        manifest_digest=artifact.manifest_digest,
    )


def _result(
    request: VerificationRequest,
    status: str,
    diagnostics: tuple[str, ...],
    *,
    tool: str = "",
    version: str = "",
    counterexample: Counterexample | None = None,
    artifacts: tuple[str, ...] = (),
    bound: bool = False,
    specification_hash: str = "",
    proof_scope: str = "",
    source_digest: str = "",
    harness_digest: str = "",
    manifest_digest: str = "",
) -> VerificationResult:
    raw = counterexample.raw_artifact if counterexample else ""
    status, note = _downgrade(status, bound, raw, tool)
    if note:
        diagnostics = (note, *diagnostics)
        bound = False
        if status not in _AUTHORITATIVE:
            counterexample = None
    completeness = "complete" if status in _AUTHORITATIVE and bound else "incomplete"
    confidence = "tool" if status in _AUTHORITATIVE and bound else "none"
    return VerificationResult(
        request.verification_id,
        status,
        counterexample,
        diagnostics,
        artifacts,
        completeness,
        confidence,
        tool,
        version,
        bound and status in _AUTHORITATIVE,
        specification_hash,
        proof_scope,
        source_digest,
        harness_digest,
        manifest_digest,
    )


def _downgrade(status: str, bound: bool, raw: str, tool: str) -> tuple[str, str]:
    if status in {"safe", "verified"}:
        return "unknown", "unknown is not safe"
    if status == "proved_safe" and tool != "smt":
        return "unknown", "proof rejected from a non-SMT tool"
    if status == "reproduced" and tool != "forge":
        return "unknown", "static analysis is not a reproduction"
    if status in _AUTHORITATIVE and not bound:
        return "unknown", "authoritative status rejected without a specification binding"
    if status == "reproduced" and not raw.strip():
        return "unknown", "reproduction rejected without execution output"
    if status == "counterexample" and not raw.strip():
        return "unknown", "counterexample rejected without tool output"
    return status, ""


def _unencoded_banner(label: str, source: str) -> str:
    return (
        f"// {_GENERATED}\n"
        f"// property-id: {label}\n"
        "// encoding: unsupported\n"
        "// This file contains no assertion. Unsupported is not success.\n"
        f"{source}\n"
    )


def _project_note(project_root: str) -> str:
    if not project_root:
        return "foundry project root was not discovered; execution uses an isolated copy"
    return (
        f"discovered foundry root {project_root}; "
        "execution uses an isolated copy and does not modify that project"
    )


def _solc_runner(harness: str, command: tuple[str, ...] = ()) -> tuple[int, str, str]:
    binary = shutil.which("solc")
    if not binary:
        return 127, "", "solc is not installed"
    flags = (
        list(command[:-1])
        if command
        else [
            "--model-checker-engine",
            "chc",
            "--model-checker-show-proved-safe",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="bugforge-smt-") as directory:
        path = Path(directory) / "Harness.sol"
        path.write_text(harness, encoding="utf-8")
        completed = subprocess.run(
            [binary, *flags, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=SMT_TIMEOUT_SECONDS,
        )
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _forge_runner(harness: str, project_root: str = "") -> tuple[int, str, str]:
    """Execute a self-contained harness in a temporary Foundry project.

    The analyzed tree is not modified. Imports are not pulled from
    ``project_root``; a harness that needs them is not encoded.
    """
    binary = shutil.which("forge")
    if not binary:
        return 127, "", "forge is not installed"
    with tempfile.TemporaryDirectory(prefix="bugforge-forge-") as directory:
        root = Path(directory)
        (root / "foundry.toml").write_text(
            '[profile.default]\nsrc = "src"\ntest = "test"\nlibs = []\n',
            encoding="utf-8",
        )
        (root / "src").mkdir()
        test_dir = root / "test"
        test_dir.mkdir()
        (test_dir / "BugforgeReplay.t.sol").write_text(harness, encoding="utf-8")
        completed = subprocess.run(
            [binary, "test", "--match-contract", "BugforgeReplay", "--root", str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=FORGE_TIMEOUT_SECONDS,
            cwd=root,
        )
    note = _project_note(project_root)
    stderr = (completed.stderr or "").strip()
    stderr = f"{stderr}\n{note}".strip()
    return completed.returncode, completed.stdout or "", stderr

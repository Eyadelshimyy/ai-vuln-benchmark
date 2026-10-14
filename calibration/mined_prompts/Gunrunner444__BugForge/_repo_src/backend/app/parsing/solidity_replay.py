"""Bounded project-aware replay of a candidate path.

A transaction sequence is an execution plan, not a finding. ``reproduced``
requires a Foundry run of the generated candidate, bound to that path,
specification, harness, source, project, compiler configuration, and
sequence. A simulated runner cannot produce ``reproduced``. A passing run
is not a proof. Compile, setup, and dependency failures are not reproductions.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from app.core.config import get_settings
from app.domain.evidence import Evidence, EvidenceKind
from app.parsing.comments import strip_comments
from app.parsing.solidity_arguments import parameter_types, preferred_literal
from app.parsing.solidity_identity import function_name_for
from app.parsing.solidity_ir import SemanticProgram
from app.parsing.solidity_project import _read_remappings
from app.parsing.solidity_spec import (
    ASSERTION_MARKER,
    FORGE_TIMEOUT_SECONDS,
    VerificationSpecification,
    foundry_root,
    function_signature,
    observed_configuration,
)
from app.parsing.solidity_state_transitions import CandidatePath
from app.security_testing.exploratory import prepare_foundry_workspace

SCHEMA = "phase44.1"
MAX_TX = 4
_SECRET_LINE = re.compile(r"(?i)(api[_-]?key|mnemonic|private[_-]?key|secret|password|etherscan)")
_IMPORT = re.compile(r"""import\s+(?:[^'"]+?\s+from\s+)?["'](?P<path>[^"']+)["']""")


@dataclass(frozen=True)
class Actor:
    identity: str
    address_placeholder: str
    permissions: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class SetupAction:
    kind: str
    call: str
    justification: str


@dataclass(frozen=True)
class TransactionStep:
    index: int
    contract: str
    function: str
    caller: str
    arguments: tuple[str, ...]
    value: str
    preconditions: tuple[str, ...]
    state_dependencies: tuple[str, ...]
    postconditions: tuple[str, ...]
    source_locations: tuple[str, ...]
    assumptions: tuple[str, ...]
    call: str


@dataclass(frozen=True)
class TransactionSequence:
    sequence_id: str
    candidate_path_id: str
    specification_id: str
    steps: tuple[TransactionStep, ...]
    setup: tuple[SetupAction, ...]
    actors: tuple[Actor, ...]
    result: str
    completeness: str
    bound: int
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayArtifact:
    status: str
    reason: str
    harness: str
    harness_digest: str
    manifest: str
    manifest_digest: str
    sequence_id: str
    specification_hash: str
    path_id: str
    source_digest: str
    project_id: str
    compiler_config: str
    dependency_id: str
    environment: str
    mode: str
    fail_token: str
    command: tuple[str, ...]
    schema: str = SCHEMA


@dataclass(frozen=True)
class ReplayResult:
    status: str
    execution: str
    environment: str
    bound: bool
    diagnostics: tuple[str, ...]
    sequence: TransactionSequence
    sequence_id: str
    specification_hash: str
    harness_digest: str
    source_digest: str
    project_id: str
    compiler_config: str
    dependency_id: str
    path_id: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    state_observations: tuple[str, ...] = ()
    manifest: str = ""
    harness: str = ""


def sequence_limit(requested: int | None = None) -> int:
    """Lengths 1..4. A caller may only tighten the upper bound."""
    if requested is None:
        return MAX_TX
    return max(1, min(int(requested), MAX_TX))


def dependency_identity(source_id: str) -> str:
    """Hash replay inputs. Imported sources count. ``.env`` and secret lines do not."""
    root = foundry_root(source_id)
    if not root:
        return "none"
    parts: list[str] = []
    toml = Path(root) / "foundry.toml"
    remappings = Path(root) / "remappings.txt"
    if toml.is_file() and not toml.is_symlink():
        parts.append("foundry.toml\n" + _toml_without_secrets(toml))
    if remappings.is_file() and not remappings.is_symlink():
        parts.append("remappings\n" + remappings.read_text(encoding="utf-8", errors="replace"))
    observed = {
        key: value
        for key, value in observed_configuration(source_id, "").items()
        if key != "compiler_version"
    }
    if observed:
        parts.append("compiler\n" + json.dumps(observed, sort_keys=True, separators=(",", ":")))
    for path in _imported_sources(source_id, root):
        parts.append(
            "import\n"
            + path.relative_to(Path(root).resolve()).as_posix()
            + "\n"
            + path.read_text(encoding="utf-8", errors="replace")
        )
    if not parts:
        return "none"
    return _digest("dependency\n" + "\n".join(parts))


def prepare_replay(
    path: CandidatePath,
    spec: VerificationSpecification,
    source: str,
    *,
    limit: int | None = None,
    program: SemanticProgram | None = None,
) -> tuple[TransactionSequence, ReplayArtifact]:
    """Build a bounded sequence and a harness. This does not execute it."""
    bound = sequence_limit(limit)
    environment = "environmental/model" if path.path_id.startswith("oracle:") else "real-target"
    actors = _actors(path)
    if any(actor.identity == "attacker" and "owner" in actor.permissions for actor in actors):
        raise RuntimeError("attacker was assigned owner")
    dependency = dependency_identity(spec.source_id)
    blank = _sequence(path, spec, (), (), actors, "not_attempted", "incomplete", bound, ())
    if spec.path_id != path.path_id:
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "specification is not this candidate path",
        )
    if len(path.function_ids) > bound:
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            environment,
            "incomplete",
            "transaction sequence exceeds the bound",
        )
    if path.path_id.startswith("oracle:"):
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            environment,
            "unsupported",
            "an oracle was not substituted; a mock is not a real-target reproduction",
        )
    if path.path_id.startswith("changes-implementation") or "upgrade:" in path.path_id:
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "proxy, implementation, admin, and initializer were not all established",
        )
    if len({item for item in path.contract_ids if item}) > 1:
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "cross-contract replay is not encoded",
        )
    if not spec.contract or spec.predicate_type not in {"equality", "monotonic"}:
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "the candidate property is not an encoded equality or monotonic check",
        )
    if not _public_names(source, _predicate_names(spec)):
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "the property variables are not readable from the replay contract",
        )
    mode, mode_reason = _project_mode(spec.source_id, source)
    if mode == "unsupported":
        return _unsupported(blank, spec, source, dependency, "none", "unsupported", mode_reason)
    constructor = _constructor_parameters(source, spec.contract)
    if constructor is None:
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "constructor arguments are not established",
        )
    reentrant = path.path_id.startswith("reentrancy:")
    steps, step_reason = _steps(path, spec, source, actors, reentrant=reentrant, program=program)
    if steps is None:
        return _unsupported(blank, spec, source, dependency, "none", "unsupported", step_reason)
    if reentrant and not _callback_supported(source, spec.contract, steps[0].function):
        return _unsupported(
            blank,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "the reentrant callback cannot be generated for this function",
        )
    assumptions = ["initializer was not called"] if _declares_initialize(source) else []
    if reentrant:
        assumptions.append("the caller is the attacker contract, not the test contract")
    setup = (
        SetupAction(
            "deploy",
            f"new {spec.contract}()",
            "parameterless constructor or no constructor",
        ),
    )
    sequence = _sequence(
        path,
        spec,
        steps,
        setup,
        actors,
        "not_attempted",
        "complete",
        bound,
        tuple(assumptions),
    )
    harness = _harness(spec, source, sequence, mode, reentrant=reentrant)
    if harness is None or "assertTrue(true)" in harness or "vm.store" in harness:
        return _unsupported(
            sequence,
            spec,
            source,
            dependency,
            "none",
            "unsupported",
            "the replay harness would not check the candidate property",
        )
    return sequence, _encoded_artifact(
        sequence, spec, source, harness, dependency, environment, mode
    )


def binds_replay(
    artifact: ReplayArtifact,
    spec: VerificationSpecification,
    source: str,
    path: CandidatePath,
    sequence: TransactionSequence,
) -> tuple[bool, str]:
    if artifact.schema != SCHEMA:
        return False, "schema mismatch"
    if artifact.status != "encoded":
        return False, "replay is not encoded"
    if artifact.path_id != path.path_id or sequence.candidate_path_id != path.path_id:
        return False, "path mismatch"
    if artifact.specification_hash != spec.specification_hash:
        return False, "specification mismatch"
    if artifact.sequence_id != sequence.sequence_id:
        return False, "sequence mismatch"
    if artifact.source_digest != _digest(source):
        return False, "source mismatch"
    if artifact.project_id != spec.project_id:
        return False, "project mismatch"
    if artifact.compiler_config != spec.compiler_config:
        return False, "compiler mismatch"
    if artifact.dependency_id != dependency_identity(spec.source_id):
        return False, "dependency mismatch"
    if artifact.harness_digest != _digest(artifact.harness):
        return False, "harness mismatch"
    if _digest(artifact.manifest) != artifact.manifest_digest:
        return False, "manifest digest mismatch"
    try:
        manifest = json.loads(artifact.manifest)
    except json.JSONDecodeError:
        return False, "manifest is not JSON"
    expected = _binding(artifact, spec, source, path, sequence)
    if manifest.get("binding") != expected:
        return False, "binding mismatch"
    for key, value in (
        ("specification_hash", spec.specification_hash),
        ("path_id", path.path_id),
        ("sequence_id", sequence.sequence_id),
        ("source_digest", artifact.source_digest),
        ("project_id", spec.project_id),
        ("compiler_config", spec.compiler_config),
        ("dependency_id", artifact.dependency_id),
        ("harness_digest", artifact.harness_digest),
        ("environment", artifact.environment),
        ("fail_token", artifact.fail_token),
    ):
        if manifest.get(key) != value:
            return False, f"manifest {key} mismatch"
    if manifest.get("source_id") != spec.source_id:
        return False, "manifest source_id mismatch"
    if path.path_id.startswith("oracle:"):
        return False, "oracle replay is not a real-target reproduction"
    if artifact.environment != "real-target":
        return False, "environmental result is not a real-target binding"
    marker = f"// {ASSERTION_MARKER} {spec.specification_hash}"
    if artifact.harness.count(marker) != 1:
        return False, "assertion marker is not unique"
    if artifact.fail_token not in artifact.harness:
        return False, "fail token is absent"
    if "assertTrue(true)" in artifact.harness or "vm.store" in artifact.harness:
        return False, "harness is not a property check"
    if artifact.harness.count(f"new {spec.contract}(") != 1:
        return False, "target deployment is not unique"
    cursor = 0
    for step in sequence.steps:
        found = artifact.harness.find(step.call, cursor)
        if found < 0:
            return False, "sequence call is absent"
        cursor = found + len(step.call)
    named_steps = {step.function for step in sequence.steps}
    if ".initialize(" in artifact.harness and "initialize" not in named_steps:
        return False, "initializer was called without being part of the sequence"
    return True, ""


def classify_output(text: str, fail_token: str) -> str:
    """Classify tool text. This function never returns ``reproduced``."""
    if not text.strip():
        return "unknown"
    if re.search(r"time-?out|timed out", text, re.IGNORECASE):
        return "timeout"
    if re.search(r"Compiler run failed|SolcError|Error \(", text):
        return "compile_failed"
    if re.search(r"Setup failed", text):
        return "setup_failed"
    if re.search(r"Failed to resolve|Could not find|not found", text, re.IGNORECASE):
        return "unavailable"
    failed = re.search(r"\[FAIL|Suite result: FAILED|Test result: FAILED", text, re.IGNORECASE)
    if failed and fail_token and fail_token in text:
        return "candidate_violation"
    if re.search(r"\[FAIL|revert", text, re.IGNORECASE):
        return "unknown"
    if re.search(r"Suite result: ok|Test result: ok", text, re.IGNORECASE):
        return "executed_no_violation"
    return "unknown"


# Host forge is not an identity. forge-sandbox is the Docker path.
_FOUNDRY_EXECUTION = frozenset({"forge", "forge-sandbox"})


def promote(status: str, *, execution: str, environment: str, bound: bool) -> str:
    """Upgrade a candidate violation only for a bound controlled Foundry run.

    A reproduced observation is an execution result. It is not a proof, and a
    passing test is not evidence that the contract is safe.
    """
    if status in {"proved_safe", "safe", "verified"}:
        return "unknown"
    if status == "reproduced":
        status = "candidate_violation"
    if (
        status == "candidate_violation"
        and execution in _FOUNDRY_EXECUTION
        and environment == "real-target"
        and bound
    ):
        return "reproduced"
    return status


def run_replay(
    path: CandidatePath,
    spec: VerificationSpecification,
    source: str,
    *,
    runner: object | None = None,
    limit: int | None = None,
) -> ReplayResult:
    """Execute one bounded replay. An injected runner is simulated, not Foundry."""
    sequence, artifact = prepare_replay(path, spec, source, limit=limit)
    if artifact.status != "encoded":
        return _result(
            sequence,
            artifact,
            artifact.status,
            "not_attempted",
            False,
            (artifact.reason,),
        )
    ok, note = binds_replay(artifact, spec, source, path, sequence)
    if not ok:
        return _result(sequence, artifact, "unknown", "not_attempted", False, (note,))
    execution = "simulated"
    if runner is None:
        blocked = blocked_network(artifact.harness)
        if blocked:
            return _result(
                sequence,
                artifact,
                "unsupported",
                "not_attempted",
                False,
                (blocked,),
            )
        problem = sandbox_requirement()
        if problem:
            return _result(
                sequence,
                artifact,
                "unavailable",
                "not_attempted",
                False,
                (problem,),
            )
        execution = "forge-sandbox"

        source_id = spec.source_id

        def runner(text: str, planned: ReplayArtifact = artifact) -> tuple[int, str, str]:
            return _execute(text, planned, source_id)

    try:
        code, stdout, stderr = runner(artifact.harness)  # type: ignore[operator]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _result(
            sequence,
            artifact,
            "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "unknown",
            execution,
            False,
            (str(exc)[:240],),
        )
    raw = f"{stdout}\n{stderr}".strip()
    status = promote(
        classify_output(raw, artifact.fail_token),
        execution=execution,
        environment=artifact.environment,
        bound=True,
    )
    observations = tuple(
        line.strip()
        for line in raw.splitlines()
        if artifact.fail_token in line or ASSERTION_MARKER in line
    )[:8]
    diagnostics: tuple[str, ...] = (raw[:2000] if raw else "forge produced no output",)
    if status == "executed_no_violation":
        diagnostics = ("a passing replay is not a proof", *diagnostics)
    if not observations:
        diagnostics = ("the tool did not report state deltas", *diagnostics)
    return _result(
        sequence,
        artifact,
        status,
        execution,
        status == "reproduced",
        diagnostics,
        stdout=str(stdout),
        stderr=str(stderr),
        exit_code=int(code),
        observations=observations,
    )


def fork_replay() -> ReplayResult:
    """Fork execution is not performed. Public infrastructure is not a target."""
    blank = TransactionSequence("", "", "", (), (), (), "unavailable", "incomplete", MAX_TX)
    return ReplayResult(
        "unavailable",
        "not_attempted",
        "none",
        False,
        ("controlled fork execution is not available; public infrastructure is not targeted",),
        blank,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )


def replay_evidence(result: ReplayResult) -> Evidence:
    metadata = {
        "verification_status": result.status,
        "execution": result.execution,
        "environment": result.environment,
        "specification_hash": result.specification_hash,
        "path_id": result.path_id,
        "sequence_id": result.sequence_id,
        "source_digest": result.source_digest,
        "harness_digest": result.harness_digest,
        "project_id": result.project_id,
        "compiler_config": result.compiler_config,
        "dependency_id": result.dependency_id,
        "bound": "true" if result.bound else "false",
        "lifecycle_effect": "none",
    }
    raw = f"{result.stdout}\n{result.stderr}".strip()
    if (
        result.status == "reproduced"
        and result.bound
        and result.execution in _FOUNDRY_EXECUTION
        and result.environment == "real-target"
        and raw
    ):
        metadata["outcome"] = "reproduced"
        return Evidence(
            kind=EvidenceKind.REPRODUCTION,
            source="forge",
            summary=f"Replay reproduction {result.path_id}",
            details=raw[:2000],
            metadata=metadata,
        )
    metadata["verification_status"] = "unknown" if result.status == "reproduced" else result.status
    return Evidence(
        kind=EvidenceKind.TOOL_STATUS,
        source=result.execution or "replay",
        summary=f"Replay {metadata['verification_status']} for {result.path_id or 'unbound'}",
        details="\n".join(result.diagnostics)[:2000],
        metadata=metadata,
    )


def lifecycle_effect(result: ReplayResult) -> str:
    del result
    return "none"


def _sequence(
    path: CandidatePath,
    spec: VerificationSpecification,
    steps: tuple[TransactionStep, ...],
    setup: tuple[SetupAction, ...],
    actors: tuple[Actor, ...],
    result: str,
    completeness: str,
    bound: int,
    assumptions: tuple[str, ...],
) -> TransactionSequence:
    payload = {
        "actors": [(item.identity, item.address_placeholder, item.permissions) for item in actors],
        "path": path.path_id,
        "setup": [(item.kind, item.call) for item in setup],
        "spec": spec.specification_hash,
        "steps": [
            (
                item.index,
                item.contract,
                item.function,
                item.caller,
                item.arguments,
                item.value,
                item.call,
            )
            for item in steps
        ],
    }
    return TransactionSequence(
        _digest(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
        path.path_id,
        spec.specification_id,
        steps,
        setup,
        actors,
        result,
        completeness,
        bound,
        assumptions,
    )


def _actors(path: CandidatePath) -> tuple[Actor, ...]:
    if path.path_id.startswith("reentrancy:"):
        return (
            Actor(
                "attacker",
                "address(0xA11CE)",
                ("caller",),
                "reentrancy candidate; the attacker is not the owner",
            ),
        )
    if path.path_id.startswith("changes-authority") or "authorization" in path.path_id:
        return (
            Actor(
                "user",
                "address(1)",
                ("caller",),
                "owner is not impersonated; privileged state was not established",
            ),
        )
    return (Actor("user", "address(1)", ("caller",), "ordinary caller"),)


def _steps(
    path: CandidatePath,
    spec: VerificationSpecification,
    source: str,
    actors: tuple[Actor, ...],
    *,
    reentrant: bool,
    program: SemanticProgram | None = None,
) -> tuple[tuple[TransactionStep, ...] | None, str]:
    caller = "user" if not actors or actors[0].identity == "owner" else actors[0].identity
    steps: list[TransactionStep] = []
    for index, function_id in enumerate(path.function_ids):
        name = function_name_for(
            function_id, program=program, source=source, contract=spec.contract
        )
        if not name:
            return None, "a sequence function was not found"
        fact = function_signature(source, spec.contract, name, program)
        if fact is None:
            return None, "a sequence function was not found"
        if fact["visibility"] not in {"public", "external"}:
            return None, "a private or internal function cannot be called from the replay contract"
        arguments = _arguments(fact)
        if arguments is None:
            return None, "argument generation is not sound for this signature"
        rendered = ", ".join(arguments)
        assumptions = list(spec.assumptions)
        value = "0"
        if fact["uses_sender"] and not reentrant:
            assumptions.append(
                "msg.sender is the explicit user actor; this is not proof of authorization"
            )
        if fact["payable"]:
            value = "1"
            call = f"target.{name}{{value: {value}}}({rendered})"
            assumptions.append(
                "transaction value is inbound from the actor; further economic setup is unsupported"
            )
        elif fact["uses_value"]:
            return None, "msg.value use is unsupported for this non-payable function"
        else:
            call = f"target.{name}({rendered})"
        steps.append(
            TransactionStep(
                index,
                spec.contract,
                name,
                caller,
                arguments,
                value,
                spec.preconditions,
                spec.state_variables,
                spec.postconditions,
                (function_id,),
                tuple(assumptions),
                call,
            )
        )
    if reentrant and steps:
        steps.append(replace(steps[0], index=1))
    return tuple(steps), ""


def _arguments(fact: dict[str, object]) -> tuple[str, ...] | None:
    types = fact.get("parameter_types")
    if not isinstance(types, tuple):
        header = str(fact.get("name", "f"))
        types = parameter_types(header) or ()
    literals: list[str] = []
    for type_name in types:
        literal = preferred_literal(str(type_name))
        if literal is None:
            return None
        literals.append(literal)
    return tuple(literals)


def _harness(
    spec: VerificationSpecification,
    source: str,
    sequence: TransactionSequence,
    mode: str,
    *,
    reentrant: bool,
) -> str | None:
    check = _property_check(spec)
    if check is None:
        return None
    if reentrant:
        attack = sequence.steps[0].call
        snapshot = ""
        if spec.predicate_type == "monotonic":
            snapshot = f"uint256 previous = target.{spec.predicate_subject}();\n        "
        body = (
            f"{spec.contract} target = new {spec.contract}();\n"
            f"        {snapshot}BugforgeAttacker attacker = new BugforgeAttacker(target);\n"
            "        attacker.attack();"
        )
        attacker = (
            "contract BugforgeAttacker {\n"
            f"    {spec.contract} target;\n"
            "    bool private entered;\n"
            f"    constructor({spec.contract} _target) {{ target = _target; }}\n"
            f"    function attack() external {{ {attack}; }}\n"
            "    receive() external payable {\n"
            "        if (!entered) { entered = true; " + attack + "; }\n"
            "    }\n"
            "}\n"
        )
    else:
        calls = "\n        ".join(step.call for step in sequence.steps)
        if spec.predicate_type == "monotonic":
            calls = f"uint256 previous = target.{spec.predicate_subject}();\n        {calls}"
        body = f"{spec.contract} target = new {spec.contract}();\n        {calls}"
        attacker = ""
    test = (
        f"{attacker}"
        "contract BugforgeReplay {\n"
        f"    function test_{spec.specification_hash[:8]}() external {{\n"
        f"        {body}\n"
        f"        // {ASSERTION_MARKER} {spec.specification_hash}\n"
        f"        {check}\n"
        "    }\n"
        "}\n"
    )
    header = (
        f"// BUGFORGE_REPLAY {spec.specification_hash}\n"
        f"// path: {sequence.candidate_path_id}\n"
        f"// sequence: {sequence.sequence_id}\n"
        f"// compiler_config: {spec.compiler_config}\n"
        f"// mode: {mode}\n"
    )
    if mode == "project":
        imported = _import_line(spec.source_id, spec.contract)
        if imported is None:
            return None
        return header + imported + "\n" + test
    if _imports(source):
        return None
    return header + source + "\n" + test


def _property_check(spec: VerificationSpecification) -> str | None:
    token = f"bugforge-fail:{spec.specification_hash}"
    if (
        spec.predicate_type == "equality"
        and re.fullmatch(r"[A-Za-z_]\w*", spec.predicate_lhs or "")
        and re.fullmatch(r"[A-Za-z_]\w*", spec.predicate_rhs or "")
    ):
        return (
            f"if (target.{spec.predicate_lhs}() != target.{spec.predicate_rhs}()) "
            f'revert("{token}");'
        )
    if spec.predicate_type == "monotonic" and re.fullmatch(
        r"[A-Za-z_]\w*", spec.predicate_subject or ""
    ):
        return f'if (target.{spec.predicate_subject}() <= previous) revert("{token}");'
    return None


def _encoded_artifact(
    sequence: TransactionSequence,
    spec: VerificationSpecification,
    source: str,
    harness: str,
    dependency: str,
    environment: str,
    mode: str,
) -> ReplayArtifact:
    digest = _digest(harness)
    fail_token = f"bugforge-fail:{spec.specification_hash}"
    command = (
        "forge",
        "test",
        "--match-contract",
        "BugforgeReplay",
        "--match-path",
        "test/BugforgeReplay.t.sol",
    )
    artifact = ReplayArtifact(
        "encoded",
        "",
        harness,
        digest,
        "",
        "",
        sequence.sequence_id,
        spec.specification_hash,
        sequence.candidate_path_id,
        _digest(source),
        spec.project_id,
        spec.compiler_config,
        dependency,
        environment,
        mode,
        fail_token,
        command,
    )
    binding = _binding(artifact, spec, source, _path_view(sequence), sequence)
    manifest_body = {
        "actors": [
            (item.identity, item.address_placeholder, item.permissions) for item in sequence.actors
        ],
        "binding": binding,
        "command": command,
        "compiler_config": spec.compiler_config,
        "dependency_id": dependency,
        "environment": environment,
        "fail_token": fail_token,
        "harness_digest": digest,
        "mode": mode,
        "path_id": sequence.candidate_path_id,
        "predicate": {
            "lhs": spec.predicate_lhs,
            "rhs": spec.predicate_rhs,
            "subject": spec.predicate_subject,
            "type": spec.predicate_type,
        },
        "project_id": spec.project_id,
        "schema": SCHEMA,
        "sequence_id": sequence.sequence_id,
        "setup": [(item.kind, item.call) for item in sequence.setup],
        "source_digest": artifact.source_digest,
        "source_id": spec.source_id,
        "specification_hash": spec.specification_hash,
        "steps": [item.call for item in sequence.steps],
    }
    manifest = json.dumps(manifest_body, sort_keys=True, separators=(",", ":"))
    return replace(artifact, manifest=manifest, manifest_digest=_digest(manifest))


def _unsupported(
    sequence: TransactionSequence,
    spec: VerificationSpecification,
    source: str,
    dependency: str,
    environment: str,
    status: str,
    reason: str,
) -> tuple[TransactionSequence, ReplayArtifact]:
    planned = replace(sequence, result=status, completeness="incomplete")
    artifact = ReplayArtifact(
        status,
        reason,
        f"// replay {status}: {reason[:200]}\n",
        "",
        "",
        "",
        planned.sequence_id,
        spec.specification_hash,
        planned.candidate_path_id,
        _digest(source),
        spec.project_id,
        spec.compiler_config,
        dependency,
        environment,
        "none",
        "",
        (),
    )
    return planned, artifact


def _binding(
    artifact: ReplayArtifact,
    spec: VerificationSpecification,
    source: str,
    path: CandidatePath,
    sequence: TransactionSequence,
) -> str:
    payload = {
        "compiler": spec.compiler_config,
        "dependency": dependency_identity(spec.source_id),
        "harness": artifact.harness_digest,
        "path": path.path_id,
        "project": spec.project_id,
        "sequence": sequence.sequence_id,
        "source": _digest(source),
        "specification": spec.specification_hash,
    }
    return _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _path_view(sequence: TransactionSequence) -> CandidatePath:
    return CandidatePath(
        sequence.candidate_path_id,
        "candidate",
        len(sequence.steps),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        "",
        "complete",
    )


def _result(
    sequence: TransactionSequence,
    artifact: ReplayArtifact,
    status: str,
    execution: str,
    bound: bool,
    diagnostics: tuple[str, ...],
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = None,
    observations: tuple[str, ...] = (),
) -> ReplayResult:
    reported = replace(sequence, result=status)
    return ReplayResult(
        status,
        execution,
        artifact.environment,
        bound,
        diagnostics,
        reported,
        artifact.sequence_id,
        artifact.specification_hash,
        artifact.harness_digest,
        artifact.source_digest,
        artifact.project_id,
        artifact.compiler_config,
        artifact.dependency_id,
        artifact.path_id,
        stdout,
        stderr,
        exit_code,
        observations,
        artifact.manifest,
        artifact.harness,
    )


def blocked_network(text: str) -> str:
    """Reject public RPC, fork, and filesystem cheats before any process starts."""
    from app.security_testing.exploratory import _BLOCKED_CHEATS

    for pattern, reason in _BLOCKED_CHEATS:
        if re.search(pattern, text):
            return reason
    if re.search(r"(?i)(https?://|eth_rpc|fork-url|etherscan|infura|alchemy)", text):
        return "BLOCKED network fork"
    return ""


def sandbox_requirement() -> str:
    """Empty when a local Docker image can run forge. Host forge is never the fallback."""
    image = get_settings().security_agent_replay_image.strip()
    if shutil.which("docker") is None:
        return "sandboxed replay requires Docker; host forge is not started"
    if not image:
        return (
            "sandboxed replay requires a local Foundry image; "
            "host forge is not started and no image is pulled"
        )
    return ""


def _execute(harness: str, artifact: ReplayArtifact, source_id: str) -> tuple[int, str, str]:
    blocked = blocked_network(harness)
    if blocked:
        return 2, "", blocked
    problem = sandbox_requirement()
    if problem:
        return 127, "", problem
    image = get_settings().security_agent_replay_image.strip()
    with tempfile.TemporaryDirectory(prefix="bugforge-replay-") as directory:
        if artifact.mode == "project":
            root = foundry_root(source_id)
            if not root:
                return 2, "", "Failed to resolve foundry root"
            original = Path(source_id).read_bytes()
            project = prepare_foundry_workspace(Path(root), Path(directory), "// removed\n")
            if Path(source_id).read_bytes() != original:
                return 2, "", "analyzed source was modified"
            _sanitize_tree(project)
            exploratory = project / "test" / "Exploratory.t.sol"
            if exploratory.is_file():
                exploratory.unlink()
            if not _copy_has_target(project, root, source_id):
                return 2, "", "Failed to resolve analyzed source in the isolated copy"
            destination = project / "test" / "BugforgeReplay.t.sol"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(harness, encoding="utf-8")
        else:
            project = Path(directory) / "project"
            project.mkdir(parents=True, exist_ok=True)
            (project / "foundry.toml").write_text(
                '[profile.default]\nsrc = "src"\ntest = "test"\nlibs = []\n',
                encoding="utf-8",
            )
            (project / "src").mkdir()
            test_dir = project / "test"
            test_dir.mkdir()
            (test_dir / "BugforgeReplay.t.sol").write_text(harness, encoding="utf-8")
        return _run_in_docker(image, Path(directory))


def _run_in_docker(image: str, scratch: Path) -> tuple[int, str, str]:
    """Forge inside Docker with the network off. This is not a host execution."""
    import asyncio

    from app.execution.base import ExecutionConfig
    from app.execution.docker_executor import DockerTestExecutor

    config = ExecutionConfig(
        command=[
            "forge",
            "test",
            "--offline",
            "--root",
            "/bugforge-output/project",
            "--match-contract",
            "BugforgeReplay",
            "--match-path",
            "test/BugforgeReplay.t.sol",
            "--out",
            "/bugforge-output/out",
            "--cache-path",
            "/bugforge-output/cache",
        ],
        working_directory=str(scratch),
        timeout_seconds=FORGE_TIMEOUT_SECONDS,
        memory_limit_mb=512,
        cpu_limit=1.0,
        environment={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"},
        allow_network=False,
        output_dir=str(scratch),
    )
    if config.allow_network:
        return 2, "", "BLOCKED network fork"
    result = asyncio.run(DockerTestExecutor(image).execute(config))
    if result.timed_out:
        raise subprocess.TimeoutExpired(config.command, FORGE_TIMEOUT_SECONDS)
    return result.exit_code or 0, result.stdout, result.stderr


def _imported_sources(source_id: str, root: str) -> tuple[Path, ...]:
    if not source_id or not Path(source_id).is_file():
        return ()
    text = Path(source_id).read_text(encoding="utf-8", errors="replace")
    pending = list(_imports(text))
    seen: list[Path] = []
    known: set[str] = set()
    while pending and len(seen) < 32:
        imported = pending.pop(0)
        resolved = _resolve_import(source_id, imported, root)
        if not resolved or resolved in known:
            continue
        path = Path(resolved)
        if path.name == ".env" or path.suffix != ".sol":
            continue
        known.add(resolved)
        seen.append(path)
        nested = _imports(path.read_text(encoding="utf-8", errors="replace"))
        pending.extend(nested)
    seen.sort(key=lambda item: item.as_posix())
    return tuple(seen)


def _toml_without_secrets(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = [line for line in lines if not _SECRET_LINE.search(line)]
    return "\n".join(kept)


def _copy_has_target(project: Path, root: str, source_id: str) -> bool:
    try:
        relative = Path(source_id).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return (project / relative).is_file()


def _sanitize_tree(project: Path) -> None:
    for path in list(project.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == ".env":
            path.unlink()
            continue
        if path.name != "foundry.toml":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        kept = [line for line in lines if not _SECRET_LINE.search(line)]
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _project_mode(source_id: str, source: str) -> tuple[str, str]:
    root = foundry_root(source_id)
    imports = _imports(source)
    if root:
        if imports and not _imports_resolve(source_id, source, root):
            return "unsupported", "an import could not be resolved in the foundry project"
        if source_id and not Path(source_id).is_file():
            return "unsupported", "the analyzed source is not a project file"
        try:
            Path(source_id).resolve().relative_to(Path(root).resolve())
        except ValueError:
            return "unsupported", "the analyzed source is outside the foundry project"
        return "project", ""
    if imports:
        return "unsupported", "imports are not reproduced without a foundry project"
    return "isolated", ""


def _imports(source: str) -> tuple[str, ...]:
    stripped = strip_comments(source, line_comment="//", block_comment=("/*", "*/"))
    return tuple(match.group("path") for match in _IMPORT.finditer(stripped))


def _imports_resolve(source_id: str, source: str, root: str) -> bool:
    for imported in _imports(source):
        if not _resolve_import(source_id, imported, root):
            return False
    return True


def _resolve_import(source_id: str, imported: str, root: str) -> str:
    if imported.startswith("."):
        candidate = (Path(source_id).parent / imported).resolve()
        root_path = Path(root).resolve()
        if candidate.is_file() and candidate.is_relative_to(root_path):
            return str(candidate)
        return ""
    remappings, _notes = _read_remappings(Path(root))
    for mapping in remappings:
        prefix, dest = mapping.split("=", 1)
        if imported.startswith(prefix):
            relative = imported[len(prefix) :]
            candidate = (Path(root) / dest / relative).resolve()
            if candidate.is_file() and candidate.is_relative_to(Path(root).resolve()):
                return str(candidate)
    for base in (Path(root) / "src", Path(root)):
        candidate = (base / imported).resolve()
        if candidate.is_file() and candidate.is_relative_to(Path(root).resolve()):
            return str(candidate)
    return ""


def _import_line(source_id: str, contract: str) -> str | None:
    root = foundry_root(source_id)
    if not root or not source_id:
        return None
    try:
        relative = Path(source_id).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return None
    target = "../" + "/".join(relative.parts)
    return f'import {{{contract}}} from "{target}";'


def _constructor_parameters(source: str, contract: str) -> tuple[str, ...] | None:
    body = _contract_body(source, contract)
    if body is None:
        return None
    match = re.search(r"constructor\s*\((?P<args>[^)]*)\)", body)
    if match is None:
        return ()
    raw = match.group("args").strip()
    if not raw:
        return ()
    return None


def _declares_initialize(source: str) -> bool:
    stripped = strip_comments(source, line_comment="//", block_comment=("/*", "*/"))
    return re.search(r"function\s+initialize\s*\(", stripped) is not None


def _callback_supported(source: str, contract: str, name: str) -> bool:
    fact = function_signature(source, contract, name)
    if fact is None or not fact["parameterless"] or fact["uses_value"]:
        return False
    if fact["visibility"] not in {"public", "external"}:
        return False
    body = _contract_body(source, contract) or ""
    stripped = strip_comments(body, line_comment="//", block_comment=("/*", "*/"))
    return "msg.sender.call" in stripped


def _predicate_names(spec: VerificationSpecification) -> tuple[str, ...]:
    if spec.predicate_type == "equality":
        return (spec.predicate_lhs, spec.predicate_rhs)
    if spec.predicate_type == "monotonic":
        return (spec.predicate_subject,)
    return ()


def _public_names(source: str, names: tuple[str, ...]) -> bool:
    for name in names:
        if not name or not re.search(
            rf"\b{re.escape(name)}\b[^;]*\bpublic\b|\bpublic\b[^;]*\b{re.escape(name)}\b",
            source,
        ):
            return False
    return True


def _contract_body(source: str, contract: str) -> str | None:
    match = re.search(rf"\bcontract\s+{re.escape(contract)}\b", source)
    if match is None:
        return None
    start = source.find("{", match.end())
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return None


def _split_args(args: str) -> tuple[str, ...]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in args:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

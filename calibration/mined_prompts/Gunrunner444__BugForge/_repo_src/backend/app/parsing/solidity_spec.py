"""Verification specifications and bound harnesses.

A harness is authoritative only when BugForge generated it for one
specification and the tool output names that specification. ``encoded=True``
on a request is not authority. A banner string is not a predicate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.parsing.comments import strip_comments
from app.parsing.solidity_identity import declared_name, function_name_for
from app.parsing.solidity_ir import SemanticProgram
from app.parsing.solidity_project import _read_remappings, _static_build_config
from app.parsing.solidity_state_transitions import (
    AbstractPredicate,
    CandidatePath,
    TransitionModel,
)

SCHEMA = "phase43.2"
ASSERTION_MARKER = "BUGFORGE_ASSERTION"
_LEVELS = ("unsupported", "syntactically_supported", "semantically_supported", "verified_capable")
MAX_HARNESS = 16_000
MAX_SEQUENCE = 8
MAX_ACTORS = 4
MAX_ATTEMPTS = 8
SMT_TIMEOUT_SECONDS = 30
FORGE_TIMEOUT_SECONDS = 60
_MATRIX = {
    "equality-scalar": {"smt": "semantically_supported", "foundry": "semantically_supported"},
    "equality-mapping-sum": {"smt": "unsupported", "foundry": "unsupported"},
    "monotonic-increment": {"smt": "semantically_supported", "foundry": "semantically_supported"},
    "asset-share": {"smt": "unsupported", "foundry": "unsupported"},
    "authorization": {"smt": "unsupported", "foundry": "unsupported"},
    "reentrancy": {"smt": "unsupported", "foundry": "unsupported"},
    "external-call-preservation": {"smt": "unsupported", "foundry": "unsupported"},
    "allowance": {"smt": "unsupported", "foundry": "unsupported"},
    "debt": {"smt": "unsupported", "foundry": "unsupported"},
    "reserve": {"smt": "unsupported", "foundry": "unsupported"},
    "pause": {"smt": "unsupported", "foundry": "unsupported"},
    "erc4626-conservation": {"smt": "unsupported", "foundry": "unsupported"},
    "external-function-smt": {"smt": "unsupported", "foundry": "semantically_supported"},
    "private-function-foundry": {"smt": "semantically_supported", "foundry": "unsupported"},
    "msg-value": {"smt": "unsupported", "foundry": "unsupported"},
    "unspecified": {"smt": "unsupported", "foundry": "unsupported"},
}


@dataclass(frozen=True)
class VerificationSpecification:
    specification_id: str
    path_id: str
    invariant_id: str
    property_category: str
    predicate_type: str
    predicate_lhs: str
    predicate_rhs: str
    predicate_subject: str
    contract: str
    function_ids: tuple[str, ...]
    operation_ids: tuple[str, ...]
    state_variables: tuple[str, ...]
    declaration_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    sequence: tuple[str, ...]
    actors: tuple[str, ...]
    source_id: str
    source_digest: str
    compiler_config: str
    smt: str
    foundry: str
    proof_scope: str
    specification_hash: str
    unsupported: tuple[str, ...] = ()
    compiler_observed: str = ""
    project_id: str = ""
    source_identity: str = ""


@dataclass(frozen=True)
class GeneratedArtifact:
    verification_id: str
    specification_hash: str
    source_digest: str
    compiler_config: str
    harness: str
    harness_digest: str
    manifest: str
    manifest_digest: str
    tool: str
    encoding_status: str
    token: str
    fail_token: str
    assert_line: int
    proof_scope: str
    command: tuple[str, ...]
    schema: str = SCHEMA
    assertion_id: str = ""
    project_id: str = ""
    source_id: str = ""


def bounds() -> dict[str, int]:
    return {
        "harness_chars": MAX_HARNESS,
        "sequence": MAX_SEQUENCE,
        "actors": MAX_ACTORS,
        "attempts": MAX_ATTEMPTS,
        "smt_timeout_seconds": SMT_TIMEOUT_SECONDS,
        "forge_timeout_seconds": FORGE_TIMEOUT_SECONDS,
    }


def capability_matrix() -> dict[str, dict[str, str]]:
    """Strongest level this encoder can claim without a live compiler run.

    ``verified_capable`` is not in this table. It is recorded only after the
    installed compiler accepts a preflight-valid harness.
    """
    matrix = {key: dict(value) for key, value in _MATRIX.items()}
    for row in matrix.values():
        for level in row.values():
            if level not in _LEVELS or level == "verified_capable":
                raise RuntimeError(level)
    return matrix


def source_text(program: SemanticProgram) -> str:
    path = Path(program.file)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return "\n".join(function.source for function in program.functions)


def specify(
    path: CandidatePath, model: TransitionModel, program: SemanticProgram
) -> VerificationSpecification:
    invariant = next(
        (item for item in model.invariants if item.invariant_id == path.invariant_id),
        None,
    )
    predicate = invariant.predicate if invariant is not None else None
    source = source_text(program)
    contract = (
        invariant.contract if invariant else (path.contract_ids[0] if path.contract_ids else "")
    )
    smt, foundry, scope, unsupported = _capabilities(
        predicate, source, contract, path.function_ids, program
    )
    sequence = path.function_ids[:MAX_SEQUENCE]
    extra_unsupported = unsupported
    if len(path.function_ids) > MAX_SEQUENCE:
        extra_unsupported = (*unsupported, "transaction sequence was truncated")
    variables = invariant.state_variables if invariant else ()
    declarations = invariant.declarations if invariant else ()
    observed = observed_configuration(program.file, model.compiler_version or "")
    compiler = configuration_hash(observed)
    project = project_identity(program.file)
    source_identity = _digest(f"{program.file}\n{source}")[:16]
    source_digest = _digest(source)
    payload = {
        "assumptions": path.assumptions,
        "category": invariant.category if invariant else path.path_id,
        "compiler": compiler,
        "compiler_observed": observed,
        "contract": contract,
        "project": project,
        "source_identity": source_identity,
        "declarations": declarations,
        "foundry": foundry,
        "functions": sequence,
        "invariant": path.invariant_id,
        "operations": path.operation_ids,
        "path": path.path_id,
        "predicate": _predicate_payload(predicate),
        "schema": SCHEMA,
        "smt": smt,
        "source": source_digest,
        "source_id": program.file,
        "variables": variables,
    }
    spec_hash = _digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return VerificationSpecification(
        f"spec:{spec_hash[:16]}",
        path.path_id,
        path.invariant_id,
        invariant.category if invariant else "unspecified",
        predicate.predicate_type if predicate else "",
        predicate.lhs if predicate else "",
        predicate.rhs if predicate else "",
        predicate.subject if predicate else "",
        contract,
        sequence,
        path.operation_ids,
        variables,
        declarations,
        path.assumptions,
        path.conditions,
        (invariant.relation,) if invariant else (),
        sequence,
        (),
        program.file,
        source_digest,
        compiler,
        smt,
        foundry,
        scope,
        spec_hash,
        extra_unsupported,
        json.dumps(observed, sort_keys=True, separators=(",", ":")),
        project,
        source_identity,
    )


def generate_smt_artifact(spec: VerificationSpecification, source: str) -> GeneratedArtifact:
    token = f"bugforge_{spec.specification_hash[:8]}"
    fail_token = f"bugforge-fail:{spec.specification_hash}"
    command = (
        "solc",
        "--model-checker-engine",
        "chc",
        "--model-checker-show-proved-safe",
        "--model-checker-targets",
        f"{spec.contract}:{token}",
        "Harness.sol",
    )
    if spec.smt != "semantically_supported":
        plain = _unsupported_harness(spec, "smt")
        return _artifact(spec, source, plain, "smt", "unsupported", token, fail_token, 0, command)
    encoded = _encoded_smt_harness(spec, source, token)
    line = _assert_line(encoded or "", spec.specification_hash)
    accepted, _note = preflight_harness(spec, source, encoded or "", "smt")
    if encoded is None or not accepted or line == 0:
        plain = _unsupported_harness(spec, "smt")
        return _artifact(spec, source, plain, "smt", "unsupported", token, fail_token, 0, command)
    return _artifact(spec, source, encoded, "smt", "encoded", token, fail_token, line, command)


def generate_forge_artifact(spec: VerificationSpecification, source: str) -> GeneratedArtifact:
    token = f"bugforge_{spec.specification_hash[:8]}"
    fail_token = f"bugforge-fail:{spec.specification_hash}"
    if spec.foundry != "semantically_supported":
        plain = _unsupported_harness(spec, "forge")
        return _artifact(spec, source, plain, "forge", "unsupported", token, fail_token, 0, ())
    encoded = _encoded_forge_harness(spec, source, fail_token)
    line = _assert_line(encoded or "", spec.specification_hash)
    accepted, _note = preflight_harness(spec, source, encoded or "", "forge")
    if encoded is None or not accepted or line == 0:
        plain = _unsupported_harness(spec, "forge")
        return _artifact(spec, source, plain, "forge", "unsupported", token, fail_token, 0, ())
    command = (
        "forge",
        "test",
        "--match-contract",
        "BugforgeReplay",
        "--match-path",
        "test/BugforgeReplay.t.sol",
    )
    return _artifact(spec, source, encoded, "forge", "encoded", token, fail_token, line, command)


def foundry_root(start: str) -> str:
    if not start:
        return ""
    current = Path(start)
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "foundry.toml").is_file():
            return str(parent)
    return ""


def binds(
    artifact: GeneratedArtifact, spec: VerificationSpecification, source: str
) -> tuple[bool, str]:
    if artifact.specification_hash != spec.specification_hash:
        return False, "specification hash mismatch"
    if artifact.source_digest != _digest(source):
        return False, "source digest mismatch"
    if artifact.compiler_config != spec.compiler_config:
        return False, "compiler configuration mismatch"
    if artifact.harness_digest != _digest(artifact.harness):
        return False, "harness digest mismatch"
    if artifact.encoding_status != "encoded":
        return False, "property is not encoded"
    try:
        manifest = json.loads(artifact.manifest)
    except json.JSONDecodeError:
        return False, "manifest is not JSON"
    if manifest.get("harness_digest") != artifact.harness_digest:
        return False, "manifest harness digest mismatch"
    if manifest.get("specification_hash") != spec.specification_hash:
        return False, "manifest specification mismatch"
    if manifest.get("source_digest") != artifact.source_digest:
        return False, "manifest source mismatch"
    if manifest.get("schema") != SCHEMA:
        return False, "manifest schema mismatch"
    if manifest.get("contract") != spec.contract:
        return False, "manifest contract mismatch"
    if tuple(manifest.get("function_ids") or ()) != spec.function_ids:
        return False, "manifest function mismatch"
    if tuple(manifest.get("operation_ids") or ()) != spec.operation_ids:
        return False, "manifest operation mismatch"
    if manifest.get("path_id") != spec.path_id:
        return False, "manifest path mismatch"
    if manifest.get("invariant_id") != spec.invariant_id:
        return False, "manifest property mismatch"
    if _digest(artifact.manifest) != artifact.manifest_digest:
        return False, "manifest digest mismatch"
    if manifest.get("compiler_config") != spec.compiler_config:
        return False, "manifest compiler mismatch"
    if manifest.get("compiler_observed") != spec.compiler_observed:
        return False, "manifest compiler observation mismatch"
    if manifest.get("source_identity") != spec.source_identity:
        return False, "manifest source identity mismatch"
    if manifest.get("assert_line") != artifact.assert_line:
        return False, "manifest assertion line mismatch"
    if manifest.get("verification_id") != artifact.verification_id:
        return False, "manifest verification mismatch"
    if manifest.get("source_id") != spec.source_id:
        return False, "manifest source identity mismatch"
    if manifest.get("project_id") != spec.project_id:
        return False, "manifest project mismatch"
    if manifest.get("predicate") != _manifest_predicate(spec):
        return False, "manifest predicate mismatch"
    if tuple(manifest.get("declaration_ids") or ()) != spec.declaration_ids:
        return False, "manifest declaration mismatch"
    if tuple(manifest.get("state_variables") or ()) != spec.state_variables:
        return False, "manifest state variable mismatch"
    if manifest.get("proof_scope") != spec.proof_scope:
        return False, "manifest proof scope mismatch"
    if manifest.get("tool") != artifact.tool:
        return False, "manifest tool mismatch"
    if tuple(manifest.get("command") or ()) != artifact.command:
        return False, "manifest command mismatch"
    if manifest.get("encoding_status") != artifact.encoding_status:
        return False, "manifest encoding mismatch"
    if manifest.get("token") != artifact.token:
        return False, "manifest token mismatch"
    if manifest.get("fail_token") != artifact.fail_token:
        return False, "manifest fail token mismatch"
    if manifest.get("assertion_id") != artifact.assertion_id:
        return False, "manifest assertion mismatch"
    if artifact.assertion_id != f"{spec.specification_hash}:{artifact.assert_line}":
        return False, "assertion identity mismatch"
    if artifact.token != f"bugforge_{spec.specification_hash[:8]}":
        return False, "token mismatch"
    if artifact.fail_token != f"bugforge-fail:{spec.specification_hash}":
        return False, "fail token mismatch"
    if artifact.assert_line != _assert_line(artifact.harness, spec.specification_hash):
        return False, "assertion line mismatch"
    if artifact.project_id != spec.project_id:
        return False, "project identity mismatch"
    if artifact.source_id != spec.source_id:
        return False, "source identity mismatch"
    if artifact.tool not in {"smt", "forge"}:
        return False, "tool mismatch"
    if artifact.tool == "smt" and artifact.command[:1] != ("solc",):
        return False, "smt command mismatch"
    if artifact.tool == "forge" and artifact.command[:1] != ("forge",):
        return False, "forge command mismatch"
    marker = f"{ASSERTION_MARKER} {spec.specification_hash}"
    if artifact.harness.count(marker) != 1:
        return False, "assertion marker is not unique"
    if spec.specification_hash not in artifact.harness:
        return False, "harness does not contain the specification hash"
    if "assert(true)" in artifact.harness or "assertTrue(true)" in artifact.harness:
        return False, "encoded harness contains a trivial assertion"
    return True, ""


def parse_smt_bound(stdout: str, stderr: str, artifact: GeneratedArtifact) -> str:
    """Accept a result only when it names this generated function and assert line."""
    text = f"{stdout}\n{stderr}"
    if not text.strip():
        return "unknown"
    if re.search(r"time-?out|timed out|out of resources", text, re.IGNORECASE):
        return "timeout"
    relevant = _bound_windows(text, artifact)
    if re.search(r"assertion violation|counterexample", relevant, re.IGNORECASE):
        return "counterexample"
    if re.search(r"\bproved\b", relevant, re.IGNORECASE):
        return "proved_safe"
    if re.search(r"unsupported|not yet implemented|cannot handle", text, re.IGNORECASE):
        return "unsupported"
    if "Error:" in text:
        return "failed"
    return "unknown"


def _bound_windows(text: str, artifact: GeneratedArtifact) -> str:
    """A window counts only when it contains both the function token and the assert line."""
    if not artifact.token or not artifact.assert_line:
        return ""
    lines = text.splitlines()
    marker = f":{artifact.assert_line}:"
    kept: list[str] = []
    for index in range(len(lines)):
        window = "\n".join(lines[max(0, index - 3) : index + 4])
        if artifact.token in window and marker in window:
            kept.append(window)
    return "\n".join(kept)


def parse_forge_bound(text: str, artifact: GeneratedArtifact) -> str:
    if not text.strip():
        return "unknown"
    if re.search(r"time-?out|timed out", text, re.IGNORECASE):
        return "timeout"
    if re.search(
        r"Compiler run failed|Setup failed|SolcError|Failed to resolve|Error \(",
        text,
    ):
        return "failed"
    failed = re.search(r"\[FAIL|Suite result: FAILED|Test result: FAILED", text, re.IGNORECASE)
    if failed and artifact.fail_token in text:
        return "reproduced"
    return "unknown"


def _capabilities(
    predicate: AbstractPredicate | None,
    source: str,
    contract: str,
    function_ids: tuple[str, ...],
    program: object | None = None,
) -> tuple[str, str, str, tuple[str, ...]]:
    if predicate is None:
        return "unsupported", "unsupported", "none", ("no predicate",)
    if "mapping aggregation" in " ".join(predicate.unsupported):
        return "unsupported", "unsupported", "none", predicate.unsupported
    semantic_program = program if isinstance(program, SemanticProgram) else None
    smt_calls, forge_calls, call_notes = _call_plan(
        source, contract, function_ids, semantic_program
    )
    same_contract = _sequence_in_contract(function_ids, contract)
    if predicate.predicate_type == "equality" and predicate.representation_status == "exact":
        readable = _state_names_in(source, contract, (predicate.lhs, predicate.rhs))
        smt_ok = same_contract and smt_calls and readable and not _has_import(source)
        forge_ok = (
            same_contract
            and forge_calls
            and readable
            and not _has_import(source)
            and _forge_can_read(source, contract, predicate)
        )
        notes = call_notes if not (smt_ok and forge_ok) else ()
        if not notes and not smt_ok and not forge_ok:
            notes = ("the transition cannot be called and read back",)
        scope = "contract-copy-harness" if smt_ok or forge_ok else "none"
        return (
            "semantically_supported" if smt_ok else "unsupported",
            "semantically_supported" if forge_ok else "unsupported",
            scope,
            notes,
        )
    if (
        predicate.predicate_type == "monotonic"
        and same_contract
        and _increment_in(source, predicate.subject)
        and _state_names_in(source, contract, (predicate.subject,))
        and not _has_import(source)
        and (smt_calls or forge_calls)
    ):
        note = (
            "the property is the state value after the calls compared with the value before",
            *call_notes,
        )
        return (
            "semantically_supported" if smt_calls else "unsupported",
            "semantically_supported"
            if forge_calls and _public_names(source, (predicate.subject,))
            else "unsupported",
            "contract-copy-harness",
            note,
        )
    if predicate.representation_status == "partial":
        return (
            "unsupported",
            "unsupported",
            "none",
            predicate.unsupported or ("partial predicate is not encoded",),
        )
    return (
        "unsupported",
        "unsupported",
        "none",
        predicate.unsupported or ("predicate is not encodable",),
    )


def _has_import(source: str) -> bool:
    return re.search(r"^\s*import\b", source, re.MULTILINE) is not None


def _sequence_in_contract(function_ids: tuple[str, ...], contract: str) -> bool:
    if not function_ids or not contract:
        return False
    for function_id in function_ids:
        parsed = declared_name(function_id)
        if parsed is None or parsed[0] != contract:
            return False
    return True


def _state_names_in(source: str, contract: str, names: tuple[str, ...]) -> bool:
    body = _contract_body(source, contract)
    if body is None:
        return False
    for name in names:
        if not name or name.startswith("sum("):
            return False
        if not re.search(rf"\b{re.escape(name)}\b", body):
            return False
    return True


def _public_names(source: str, names: tuple[str, ...]) -> bool:
    for name in names:
        if not name or name.startswith("sum("):
            return False
        if not re.search(
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
    end = _close_brace(source, start)
    if end < 0:
        return None
    return source[start:end]


def _forge_can_read(source: str, contract: str, predicate: AbstractPredicate) -> bool:
    if _has_import(source):
        return False
    return _public_names(source, (predicate.lhs, predicate.rhs)) and _state_names_in(
        source, contract, (predicate.lhs, predicate.rhs)
    )


def _increment_in(source: str, symbol: str) -> bool:
    if not symbol:
        return False
    pattern = rf"\b{re.escape(symbol)}\s*(\+=\s*1|=\s*{re.escape(symbol)}\s*\+\s*1)\b"
    return re.search(pattern, source) is not None


def observed_configuration(source_id: str, compiler_version: str) -> dict[str, object]:
    """Settings that were actually present. Missing settings are omitted, not invented."""
    observed: dict[str, object] = {}
    if compiler_version:
        observed["compiler_version"] = compiler_version
    root = foundry_root(source_id)
    if not root:
        return observed
    build = _static_build_config(Path(root))
    for key, value in build.items():
        if value:
            observed[key] = value
    remappings, notes = _read_remappings(Path(root))
    if remappings:
        observed["remappings"] = remappings
    if notes:
        observed["remapping_notes"] = notes
    toml = Path(root) / "foundry.toml"
    if toml.is_file() and not toml.is_symlink():
        observed["foundry_toml_digest"] = _digest(
            toml.read_text(encoding="utf-8", errors="replace")
        )
    if (Path(root) / "src").is_dir():
        observed["src"] = "src"
    return observed


def configuration_hash(observed: dict[str, object]) -> str:
    if not observed:
        return "unspecified"
    return _digest(json.dumps(observed, sort_keys=True, separators=(",", ":")))


def project_identity(source_id: str) -> str:
    root = foundry_root(source_id)
    if root:
        toml = Path(root) / "foundry.toml"
        text = toml.read_text(encoding="utf-8", errors="replace") if toml.is_file() else ""
        return _digest(f"foundry\n{text}")
    return _digest(f"file\n{source_id}")


def function_signature(
    source: str,
    contract: str,
    name: str,
    program: object | None = None,
) -> dict[str, object] | None:
    """Parser facts win when a semantic program contains the function."""
    if program is not None:
        parsed = _signature_from_program(program, contract, name)
        if parsed is not None:
            return parsed
    return _regex_signature(source, contract, name)


def _signature_from_program(program: object, contract: str, name: str) -> dict[str, object] | None:
    functions = getattr(program, "functions", ())
    matches = [
        item
        for item in functions
        if getattr(item, "contract", "") == contract and getattr(item, "name", "") == name
    ]
    if len(matches) != 1:
        return None
    function = matches[0]
    header = (getattr(function, "source", "") or "").split("{", 1)[0]
    from app.parsing.solidity_arguments import parameter_types

    types = parameter_types(header)
    if types is None:
        return None
    visibility = getattr(function, "visibility", "") or "public"
    mutability = getattr(function, "mutability", "") or ""
    stripped = strip_comments(
        getattr(function, "source", "") or "",
        line_comment="//",
        block_comment=("/*", "*/"),
    )
    payable = mutability == "payable" or "payable" in mutability.split()
    return {
        "name": name,
        "parameterless": not types,
        "parameter_types": types,
        "visibility": visibility,
        "payable": payable,
        "uses_sender": "msg.sender" in stripped,
        "uses_value": "msg.value" in stripped or payable,
        "provenance": "parser",
        "identity": getattr(function, "identity", ""),
        "contract": contract,
    }


def _regex_signature(source: str, contract: str, name: str) -> dict[str, object] | None:
    body = _contract_body(source, contract)
    if body is None or not name:
        return None
    pattern = re.compile(
        rf"function\s+{re.escape(name)}\s*\((?P<args>[^)]*)\)\s*(?P<tail>[^{{;]*)",
        re.MULTILINE,
    )
    match = pattern.search(body)
    if match is None:
        return None
    tail = match.group("tail")
    visibility = "public"
    for word in ("external", "public", "internal", "private"):
        if re.search(rf"\b{word}\b", tail):
            visibility = word
            break
    start = body.find("{", match.end())
    end = _close_brace(body, start) if start >= 0 else -1
    function_text = body[match.start() : end] if end > 0 else tail
    stripped = strip_comments(function_text, line_comment="//", block_comment=("/*", "*/"))
    from app.parsing.solidity_arguments import parameter_types

    types = parameter_types(f"function {name}({match.group('args')})")
    payable = bool(re.search(r"\bpayable\b", tail))
    return {
        "name": name,
        "parameterless": match.group("args").strip() == "",
        "parameter_types": types or (),
        "visibility": visibility,
        "payable": payable,
        "uses_sender": "msg.sender" in stripped,
        "uses_value": "msg.value" in stripped or payable,
        "provenance": "regex",
        "contract": contract,
        "identity": "",
    }


def _call_plan(
    source: str,
    contract: str,
    function_ids: tuple[str, ...],
    program: SemanticProgram | None = None,
) -> tuple[bool, bool, tuple[str, ...]]:
    if not function_ids:
        return False, False, ("the transition cannot be called and read back",)
    notes: list[str] = []
    smt_ok = True
    forge_ok = True
    for function_id in function_ids:
        name = function_name_for(function_id, program=program, source=source, contract=contract)
        if not name:
            return False, False, ("the transition cannot be called and read back",)
        fact = function_signature(source, contract, name, program)
        if fact is None or not fact["parameterless"]:
            return False, False, ("the transition cannot be called and read back",)
        if fact["uses_value"]:
            return False, False, ("msg.value is not part of the specification",)
        if fact["visibility"] == "external":
            smt_ok = False
            notes.append("an external function cannot be called internally")
        if fact["visibility"] in {"internal", "private"}:
            forge_ok = False
            notes.append("a private or internal function cannot be called from another contract")
        if fact["uses_sender"]:
            forge_ok = False
            notes.append("the foundry caller would be the test contract")
    return smt_ok, forge_ok, tuple(dict.fromkeys(notes))


def _internal_call_names(
    source: str, contract: str, function_ids: tuple[str, ...]
) -> tuple[str, ...] | None:
    smt_ok, _forge_ok, _notes = _call_plan(source, contract, function_ids)
    if not smt_ok:
        return None
    return _resolved_names(source, contract, function_ids)


def _external_call_names(
    source: str, contract: str, function_ids: tuple[str, ...]
) -> tuple[str, ...] | None:
    _smt_ok, forge_ok, _notes = _call_plan(source, contract, function_ids)
    if not forge_ok:
        return None
    return _resolved_names(source, contract, function_ids)


def _resolved_names(
    source: str, contract: str, function_ids: tuple[str, ...]
) -> tuple[str, ...] | None:
    names: list[str] = []
    for function_id in function_ids:
        name = function_name_for(function_id, program=None, source=source, contract=contract)
        if not name:
            return None
        names.append(name)
    return tuple(names)


def _assertion_block(spec: VerificationSpecification, expression: str) -> str:
    return f"        // {ASSERTION_MARKER} {spec.specification_hash}\n        {expression}\n"


def preflight_harness(
    spec: VerificationSpecification, source: str, harness: str, tool: str
) -> tuple[bool, str]:
    """Structural checks. They do not prove the harness compiles."""
    if not harness or len(harness) > MAX_HARNESS:
        return False, "harness is empty or too large"
    if not spec.specification_hash or not spec.predicate_type:
        return False, "specification is empty"
    if "assert(true)" in harness or "assertTrue(true)" in harness:
        return False, "trivial assertion"
    if _digest(source) != spec.source_digest:
        return False, "source digest mismatch"
    if f"// compiler_config: {spec.compiler_config}" not in harness:
        return False, "compiler configuration mismatch"
    if harness.count(f"// BUGFORGE_VERIFICATION_SPEC {spec.specification_hash}") != 1:
        return False, "specification metadata is not unique"
    marker = f"{ASSERTION_MARKER} {spec.specification_hash}"
    if harness.count(marker) != 1:
        return False, "assertion marker is not unique"
    if harness.count(f"function bugforge_{spec.specification_hash[:8]}") != 1 and tool == "smt":
        return False, "generated function is not unique"
    if f"contract {spec.contract}" not in harness:
        return False, "target contract is absent"
    if tool == "smt" and _internal_call_names(source, spec.contract, spec.sequence) is None:
        return False, "calls are not visibility-safe"
    if tool == "forge" and _external_call_names(source, spec.contract, spec.sequence) is None:
        return False, "calls are not visibility-safe"
    line = _assert_line(harness, spec.specification_hash)
    if line == 0:
        return False, "generated assertion is absent"
    text = harness.splitlines()[line - 1]
    if tool == "smt":
        if (
            spec.predicate_type == "equality"
            and f"assert({spec.predicate_lhs} == {spec.predicate_rhs})" not in text
        ):
            return False, "predicate is not the generated assertion"
        if spec.predicate_type == "monotonic" and "assert(next > previous)" not in text:
            return False, "predicate is not the generated assertion"
        if "this." in harness.split("function bugforge_")[-1]:
            return False, "external message call was substituted"
    if tool == "forge":
        if f'"{spec.specification_hash}"' not in text and spec.specification_hash not in text:
            return False, "fail token is not the generated check"
        if f"new {spec.contract}()" not in harness:
            return False, "target is not deployed"
    return True, ""


def _manifest_predicate(spec: VerificationSpecification) -> dict[str, str]:
    return {
        "lhs": spec.predicate_lhs,
        "rhs": spec.predicate_rhs,
        "subject": spec.predicate_subject,
        "type": spec.predicate_type,
    }


def _encoded_smt_harness(spec: VerificationSpecification, source: str, token: str) -> str | None:
    names = _internal_call_names(source, spec.contract, spec.sequence)
    if names is None:
        return None
    calls = "\n        ".join(f"{name}();" for name in names)
    if spec.predicate_type == "monotonic" and spec.predicate_subject:
        expression = "assert(next > previous);"
        injected = (
            f"function {token}() external {{\n"
            f"        uint256 previous = {spec.predicate_subject};\n"
            f"        {calls}\n"
            f"        uint256 next = {spec.predicate_subject};\n"
            f"{_assertion_block(spec, expression)}"
            "    }\n"
        )
    elif (
        spec.predicate_type == "equality"
        and re.fullmatch(r"[A-Za-z_]\w*", spec.predicate_lhs or "")
        and re.fullmatch(r"[A-Za-z_]\w*", spec.predicate_rhs or "")
    ):
        expression = f"assert({spec.predicate_lhs} == {spec.predicate_rhs});"
        injected = (
            f"function {token}() external {{\n"
            f"        {calls}\n"
            f"{_assertion_block(spec, expression)}"
            "    }\n"
        )
    else:
        return None
    if "this." in injected:
        return None
    copied = _inject(source, spec.contract, injected)
    if copied is None:
        return None
    return _header(spec, "smt") + copied


def _encoded_forge_harness(
    spec: VerificationSpecification, source: str, fail_token: str
) -> str | None:
    names = _external_call_names(source, spec.contract, spec.sequence)
    if names is None or not spec.contract:
        return None
    calls = "\n        ".join(f"target.{item}();" for item in names)
    if spec.predicate_type == "equality" and spec.predicate_lhs and spec.predicate_rhs:
        expression = (
            f"if (target.{spec.predicate_lhs}() != target.{spec.predicate_rhs}()) "
            f'revert("{fail_token}");'
        )
        body = f"{calls}\n{_assertion_block(spec, expression)}"
    elif spec.predicate_type == "monotonic" and spec.predicate_subject:
        expression = f'if (target.{spec.predicate_subject}() <= previous) revert("{fail_token}");'
        body = (
            f"uint256 previous = target.{spec.predicate_subject}();\n"
            f"        {calls}\n"
            f"{_assertion_block(spec, expression)}"
        )
    else:
        return None
    test = (
        "contract BugforgeReplay {\n"
        f"    function test_{spec.specification_hash[:8]}() external {{\n"
        f"        {spec.contract} target = new {spec.contract}();\n"
        f"        {body}\n"
        "    }\n"
        "}\n"
    )
    return _header(spec, "forge") + source + "\n" + test


def _parameterless_names(source: str, function_ids: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for function_id in function_ids:
        parsed = declared_name(function_id)
        if parsed is None:
            continue
        contract, name = parsed
        if function_name_for(function_id, program=None, source=source, contract=contract) != name:
            continue
        body = _contract_body(source, contract) or ""
        if re.search(rf"function\s+{re.escape(name)}\s*\(\s*\)", body):
            names.append(name)
    return tuple(names)


def _parameterless_name_in(source: str, function_ids: tuple[str, ...]) -> str | None:
    names = _parameterless_names(source, function_ids)
    if len(names) == len(function_ids) and names:
        return names[-1]
    return None


def _inject(source: str, contract: str, function_text: str) -> str | None:
    match = re.search(rf"\bcontract\s+{re.escape(contract)}\b", source)
    if match is None:
        return None
    start = source.find("{", match.end())
    if start < 0:
        return None
    end = _close_brace(source, start)
    if end < 0:
        return None
    return source[: end - 1] + "\n" + function_text + source[end - 1 :]


def _close_brace(source: str, start: int) -> int:
    depth = 0
    index = start
    while index < len(source):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return -1


def _unsupported_harness(spec: VerificationSpecification, tool: str) -> str:
    reason = ", ".join(spec.unsupported) or spec.smt
    return (
        f"// BUGFORGE_VERIFICATION_SPEC {spec.specification_hash}\n"
        f"// tool: {tool}\n"
        f"// encoding: unsupported\n"
        f"// {reason[:240]}\n"
        "// This file contains no assertion. Unsupported is not success.\n"
    )


def _header(spec: VerificationSpecification, tool: str) -> str:
    return (
        f"// BUGFORGE_VERIFICATION_SPEC {spec.specification_hash}\n"
        f"// verification_id: ver:{spec.path_id}\n"
        f"// candidate_id: {spec.path_id}\n"
        f"// property_id: {spec.invariant_id}\n"
        f"// source_digest: {spec.source_digest}\n"
        f"// compiler_config: {spec.compiler_config}\n"
        f"// proof_scope: {spec.proof_scope}\n"
        f"// tool: {tool}\n"
        "// Generated verification harness. Not production source.\n"
    )


def _artifact(
    spec: VerificationSpecification,
    source: str,
    harness: str,
    tool: str,
    encoding_status: str,
    token: str,
    fail_token: str,
    assert_line: int,
    command: tuple[str, ...],
) -> GeneratedArtifact:
    harness_digest = _digest(harness)
    verification_id = f"ver:{spec.path_id}"
    source_digest = _digest(source)
    assertion_id = f"{spec.specification_hash}:{assert_line}" if assert_line else ""
    manifest_body = {
        "assert_line": assert_line,
        "assertion_id": assertion_id,
        "command": command,
        "compiler_config": spec.compiler_config,
        "compiler_observed": spec.compiler_observed,
        "contract": spec.contract,
        "declaration_ids": spec.declaration_ids,
        "encoding_status": encoding_status,
        "fail_token": fail_token,
        "function_ids": spec.function_ids,
        "harness_digest": harness_digest,
        "invariant_id": spec.invariant_id,
        "operation_ids": spec.operation_ids,
        "path_id": spec.path_id,
        "predicate": _manifest_predicate(spec),
        "project_id": spec.project_id,
        "proof_scope": spec.proof_scope,
        "schema": SCHEMA,
        "source_digest": source_digest,
        "source_id": spec.source_id,
        "source_identity": spec.source_identity,
        "specification_hash": spec.specification_hash,
        "state_variables": spec.state_variables,
        "token": token,
        "tool": tool,
        "verification_id": verification_id,
    }
    manifest = json.dumps(manifest_body, sort_keys=True, separators=(",", ":"))
    return GeneratedArtifact(
        verification_id,
        spec.specification_hash,
        source_digest,
        spec.compiler_config,
        harness,
        harness_digest,
        manifest,
        _digest(manifest),
        tool,
        encoding_status,
        token,
        fail_token,
        assert_line,
        spec.proof_scope,
        command,
        assertion_id=assertion_id,
        project_id=spec.project_id,
        source_id=spec.source_id,
    )


def _predicate_payload(predicate: AbstractPredicate | None) -> dict[str, str]:
    if predicate is None:
        return {}
    return {
        "lhs": predicate.lhs,
        "operator": predicate.operator,
        "rhs": predicate.rhs,
        "status": predicate.representation_status,
        "subject": predicate.subject,
        "type": predicate.predicate_type,
    }


def _assert_line(harness: str, specification_hash: str) -> int:
    """Return the line after the unique generated marker. Source asserts do not count."""
    marker = f"// {ASSERTION_MARKER} {specification_hash}"
    lines = harness.splitlines()
    hits = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(hits) != 1:
        return 0
    following = hits[0] + 1
    if following >= len(lines):
        return 0
    text = lines[following]
    if "assert(" not in text and "revert(" not in text:
        return 0
    return following + 1


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

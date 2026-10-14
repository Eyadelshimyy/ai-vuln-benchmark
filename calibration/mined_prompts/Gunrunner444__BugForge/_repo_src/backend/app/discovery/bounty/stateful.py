"""Stateful local execution bridge: VFCS plans -> executable Foundry harnesses.

This is the Phase 52 closed loop's execution leg, hardened. A deterministic VFCS
plan plus its :class:`~app.discovery.bounty.properties.PropertySpec` is compiled
into a Foundry test that deploys the analyzed contract from the *exact* source set
the sequence was built from and runs the plan against a fresh local instance,
entirely offline (``--offline``, ``FOUNDRY_OFFLINE=true``, ffi off, no fs access,
no RPC, no fork, pinned host ``solc``, no auto-detect/download).

The harness never decides anything. It only *reports* structured observations by
emitting ``BugForgeObservation(kind, index, value)`` events; the result is read
from ``forge test --json`` (raw logs filtered to the harness address) and the
property is judged in Python by :func:`properties.evaluate`. Analyzed source code
cannot forge an observation (events from any other address are ignored), and it
cannot choose a Forge flag or a config value: execution policy is server-owned.

Outcomes are distinct and never collapse to a generic failure: PROPERTY_VIOLATED,
PROPERTY_HELD, MECHANISM_OBSERVED, FAILURE_PROPAGATED, SEQUENCE_EXECUTED_NO_ORACLE,
SEQUENCE_REVERTED, COMPILE_FAILED, EXECUTION_FAILED, TIMEOUT, IDENTITY_MISMATCH,
INCONCLUSIVE, UNAVAILABLE. A revert is not a finding. A mechanism observation is
not a security-property violation. A sequence that ran without an oracle is not
a finding. Two agreeing local observations are at most a CORROBORATED_CANDIDATE.
Nothing here marks anything verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.discovery.bounty.properties import (
    OracleKind,
    OracleObservation,
    OracleSpec,
    PropertyDeclaration,
    PropertyEvaluation,
    PropertySpec,
    PropertyVerdict,
    build_property,
    evaluate,
    probe_index,
)
from app.discovery.bounty.vfcs import (
    ATTACKER,
    MAX_MUTATIONS,
    VICTIM,
    FeedbackSignal,
    MinimizationResult,
    Vfcs,
    VfcsCall,
    instance_identities,
    minimize,
    mutate,
)
from app.discovery.process import tool_path
from app.parsing.solidity_research import Param, ResearchModel, RFunction, model_source_problems

MAX_CALLS = 8
MAX_FEEDBACK_ROUNDS = 3
EXEC_TIMEOUT = 120
MAX_TAIL = 2000
MAX_EVENTS = 128
MAX_RUNS = 160  # forge invocations per executor (one campaign request)
HARNESS_SCHEMA = "bugforge.harness/2"
ATTACKER_ADDR = "address(0xA11CE)"
VICTIM_ADDR = "address(0xB0B)"
FIXTURE_AMOUNT = "1000000000000000000"  # 1e18 fixture units; local only
# Forge deploys the test contract at this fixed address (default sender, nonce 1).
# Only events emitted by the harness itself are observations; any other emitter,
# including the analyzed contract, is ignored.
HARNESS_ADDRESS = "0x7fa9385be102ac3eac297483dd6233d62b3e1496"
# keccak256("BugForgeObservation(uint256,uint256,uint256)")
OBSERVATION_TOPIC = "0x05cd64bb6ffe3a3804e150ecc2c28424720f099e6672c02e96eb325669e2f411"

K_DEPLOYED = 1
K_CALL = 2
K_BEFORE = 3
K_AFTER = 4
K_READ_FAILED = 5
K_RECEIVED = 6
K_CREDITED = 7
K_PRIMITIVE = 8
K_DONE = 9


class Outcome(StrEnum):
    PROPERTY_VIOLATED = "property_violated"
    PROPERTY_HELD = "property_held"
    # The callee reverted and the outer call returned. Not a security verdict.
    MECHANISM_OBSERVED = "mechanism_observed"
    # The outer call reverted too, so the swallowed-call mechanism was not reproduced.
    FAILURE_PROPAGATED = "failure_propagated"
    SEQUENCE_EXECUTED_NO_ORACLE = "sequence_executed_no_oracle"
    SEQUENCE_REVERTED = "sequence_reverted"
    COMPILE_FAILED = "compile_failed"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    IDENTITY_MISMATCH = "identity_mismatch"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


# Outcomes where the harness actually ran the sequence to completion.
RUNNABLE = frozenset(
    {
        Outcome.PROPERTY_VIOLATED,
        Outcome.PROPERTY_HELD,
        Outcome.MECHANISM_OBSERVED,
        Outcome.FAILURE_PROPAGATED,
        Outcome.SEQUENCE_EXECUTED_NO_ORACLE,
        Outcome.SEQUENCE_REVERTED,
    }
)
# Outcomes worth a durable reproduction bundle.
BUNDLED = RUNNABLE | {Outcome.COMPILE_FAILED, Outcome.EXECUTION_FAILED, Outcome.INCONCLUSIVE}


class ReplayMode(StrEnum):
    """What a local execution reproduces. Only LOCAL_SOURCE_REPLAY runs here."""

    LOCAL_SOURCE_REPLAY = "local_source_replay"  # fresh deploy of analyzed source
    DEPLOYMENT_REPLAY = "deployment_replay"  # deployed bytecode (needs runtime capture)
    PINNED_FORK_REPLAY = "pinned_fork_replay"  # operator-approved pinned fork only


def replay_modes() -> dict[str, dict[str, str]]:
    """What each replay mode would establish and whether this path can run it.

    The three are never conflated: a local source replay says nothing about the
    deployed bytecode or state; a deployment replay needs captured runtime code and
    state; a pinned fork replay needs an approved fork, which research tooling never
    opens (no RPC).
    """
    tools = tool_status()
    local = (
        "usable"
        if execution_enabled() and tools.available
        else ("blocked_by_policy" if not execution_enabled() else "unavailable")
    )
    return {
        ReplayMode.LOCAL_SOURCE_REPLAY.value: {
            "status": local,
            "establishes": "behavior of the analyzed source, freshly deployed with fixtures",
            "does_not_establish": "deployed bytecode, storage, balances or configuration",
        },
        ReplayMode.DEPLOYMENT_REPLAY.value: {
            "status": "unavailable",
            "establishes": "behavior of the deployed runtime bytecode with captured state",
            "does_not_establish": "anything without a captured runtime and state snapshot",
            "reason": "no runtime/state capture adapter exists in this path",
        },
        ReplayMode.PINNED_FORK_REPLAY.value: {
            "status": "blocked_by_policy",
            "establishes": "behavior against chain state pinned at a block",
            "does_not_establish": "anything here: research tooling opens no RPC or fork",
            "reason": "needs an operator-approved pinned fork outside this path",
        },
    }


@dataclass(frozen=True)
class ToolStatus:
    forge: str
    solc: str
    forge_version: str = ""
    solc_version: str = ""

    @property
    def available(self) -> bool:
        return self.forge == "available" and self.solc == "available"

    def as_dict(self) -> dict[str, str]:
        return {
            "forge": self.forge,
            "solc": self.solc,
            "forge_version": self.forge_version or "unknown",
            "solc_version": self.solc_version or "unknown",
        }


@lru_cache(maxsize=8)
def _version(binary: str) -> str:
    path = tool_path(binary)
    if not path:
        return ""
    try:
        proc = subprocess.run(
            [path, "--version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    text = proc.stdout + proc.stderr
    if binary == "solc":
        match = re.search(r"Version:\s*(\d+\.\d+\.\d+)", text)
        return match.group(1) if match else ""
    match = re.search(r"(\d+\.\d+\.\d+[\w.+-]*)", text)
    return match.group(1) if match else ""


def tool_status() -> ToolStatus:
    forge = "available" if tool_path("forge") else "unavailable"
    solc = "available" if tool_path("solc") else "unavailable"
    return ToolStatus(
        forge=forge,
        solc=solc,
        forge_version=_version("forge") if forge == "available" else "",
        solc_version=_version("solc") if solc == "available" else "",
    )


def execution_enabled() -> bool:
    from app.core.config import get_settings

    return bool(get_settings().bounty_stateful_execution)


def _timeout_setting() -> int:
    from app.core.config import get_settings

    value = int(getattr(get_settings(), "bounty_stateful_timeout_seconds", EXEC_TIMEOUT))
    return max(5, min(value, 600))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tail(text: str, limit: int = MAX_TAIL) -> str:
    return text[-limit:] if text else ""


# ---- sequence identity ------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityContext:
    """What the caller expects the sequence to be bound to (server-owned, not from text)."""

    campaign_id: str = ""
    deployment: str = ""  # chain:address, or "" / "source" when source-only
    chain_id: str = ""
    address: str = ""
    runtime_digest: str = ""
    proxy_kind: str = "none"
    implementation: str = ""
    compiler_configuration: str = ""
    source_snapshot: str = ""
    # path -> sha256 recorded when the sequence's sources were analyzed
    expected_hashes: Mapping[str, str] = field(default_factory=dict)
    expected_file: str = ""


@dataclass(frozen=True)
class ExecutionIdentity:
    campaign_id: str
    source_snapshot: str
    source_files: tuple[str, ...]
    source_hashes: tuple[tuple[str, str], ...]
    source_set_digest: str
    contract: str
    contract_file: str
    chain_id: str
    address: str
    deployment: str
    runtime_digest: str
    proxy_kind: str
    implementation: str
    compiler_configuration: str
    solc_version: str
    property_id: str
    sequence_id: str
    call_instances: tuple[str, ...]
    replay_mode: ReplayMode
    status: str  # bound | identity_mismatch
    mismatches: tuple[str, ...] = ()

    @property
    def bound(self) -> bool:
        return self.status == "bound"

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "source_snapshot": self.source_snapshot,
            "source_files": list(self.source_files),
            "source_hashes": dict(self.source_hashes),
            "source_set_digest": self.source_set_digest,
            "contract": self.contract,
            "contract_file": self.contract_file,
            "chain_id": self.chain_id,
            "address": self.address,
            "deployment": self.deployment,
            "runtime_digest": self.runtime_digest,
            "proxy_kind": self.proxy_kind,
            "implementation": self.implementation,
            "compiler_configuration": self.compiler_configuration,
            "solc_version": self.solc_version,
            "property_id": self.property_id,
            "sequence_id": self.sequence_id,
            "call_instances": list(self.call_instances),
            "replay_mode": self.replay_mode.value,
            "status": self.status,
            "mismatches": list(self.mismatches),
        }


def target_contract(sequence: Vfcs) -> str:
    """The contract a sequence targets: the candidate's contract, never calls[0]."""
    _detector, _, target = sequence.derived_from.partition("@")
    contract = target.partition(".")[0]
    if contract:
        return contract
    for call in sequence.calls:
        if not call.primitive:
            return call.contract
    return ""


def source_set_digest(hashes: Mapping[str, str]) -> str:
    return sha256_text(json.dumps(sorted(hashes.items()), separators=(",", ":")))


def _deployment_problem(sequence_deployment: str, context_deployment: str) -> str:
    """Source-only, unspecified, and a pinned deployment are not interchangeable.

    ``""`` on a sequence is source-oriented (no deployment was known when it was
    built). ``"source"`` is an explicit source-only declaration. Any other value
    is a pinned ``chain:address``. An unspecified context may run a source-oriented
    sequence. It may not run a pinned one, because the deployment was not established.
    """
    seq_source = sequence_deployment in ("", "source")
    if context_deployment == "":
        if seq_source:
            return ""
        return (
            f"the sequence is bound to {sequence_deployment}, and the execution context "
            "does not establish that deployment"
        )
    if context_deployment == "source":
        if seq_source:
            return ""
        return f"a source-only execution cannot run a sequence bound to {sequence_deployment}"
    if seq_source:
        return (
            f"a deployment-specific execution ({context_deployment}) cannot run a "
            "source-only sequence"
        )
    if sequence_deployment != context_deployment:
        return (
            f"the sequence is bound to {sequence_deployment}, not the target {context_deployment}"
        )
    return ""


def _identity_agreement(label: str, recorded: str, supplied: str) -> str:
    """A required compiler or implementation value must be established, not assumed.

    Empty on both sides means the operation does not require that field. A value on
    only one side cannot be verified. Two different values never match.
    """
    if not recorded and not supplied:
        return ""
    if recorded and supplied and recorded == supplied:
        return ""
    if recorded and not supplied:
        return (
            f"the sequence requires {label} {recorded}, and the execution context "
            "does not establish it"
        )
    if supplied and not recorded:
        return (
            f"the execution context requires {label} {supplied}, and the sequence "
            "does not record it"
        )
    return f"the {label} does not match the sequence ({recorded} != {supplied})"


def _reported_deployment(sequence_deployment: str, context_deployment: str) -> str:
    """The deployment this execution actually represents. Source-only stays ``source``."""
    if context_deployment == "source" or (
        context_deployment == "" and sequence_deployment in ("", "source")
    ):
        return "source"
    if context_deployment:
        return context_deployment
    if sequence_deployment and sequence_deployment != "source":
        return sequence_deployment
    return "source"


def bind_identity(
    sequence: Vfcs,
    model: ResearchModel,
    sources: Mapping[str, str],
    context: IdentityContext,
    *,
    property_id: str = "",
    solc_version: str = "",
) -> ExecutionIdentity:
    """Bind a sequence to the exact source set it was built from. Mismatches fail closed."""
    contract = target_contract(sequence)
    item = model.contracts.get(contract)
    contract_file = item.file if item is not None else ""
    hashes = {path: sha256_text(text) for path, text in sorted(sources.items())}
    problems: list[str] = []
    if item is None:
        problems.append(f"contract {contract or '?'} is not in the sequence's model")
    elif contract in model.ambiguous:
        problems.append(f"contract name {contract} is declared in several files")
    if contract_file and contract_file not in sources:
        problems.append(f"{contract_file} (declares {contract}) is not in the source set")
    if item is not None:
        for owner in model.lineage(contract):
            base = model.contracts.get(owner)
            if base is not None and base.file not in sources:
                problems.append(f"{base.file} (declares {owner}) is not in the source set")
    if context.expected_file and contract_file and context.expected_file != contract_file:
        problems.append(
            f"expected source file {context.expected_file} but {contract} is declared in "
            f"{contract_file}"
        )
    # The sequence's own hashes are the content identity it was built from. The
    # caller's snapshot is checked too when present. Neither side is filled in
    # from the text about to be executed.
    # A caller-supplied snapshot is not a substitute for the sequence's own
    # content identity, and missing hashes are never filled from ``sources``.
    recorded = dict(sequence.identity.source_hashes)
    if not recorded:
        problems.append("source content identity was not recorded; execution is refused")
    else:
        for path, digest_value in hashes.items():
            expected = recorded.get(path)
            if expected is None:
                problems.append(f"{path} is not part of the sequence's source identity")
            elif expected != digest_value:
                problems.append(f"{path} changed since the sequence was created")
        for path in recorded:
            if path not in hashes:
                problems.append(f"{path} from the sequence's source identity is missing")
    if context.expected_hashes:
        for path, digest_value in hashes.items():
            expected = context.expected_hashes.get(path)
            if expected is None:
                problems.append(f"{path} is not part of the analyzed snapshot")
            elif expected != digest_value:
                problems.append(f"{path} changed since it was analyzed")
        if recorded:
            for path, expected in recorded.items():
                snapshot = context.expected_hashes.get(path)
                if snapshot is None:
                    problems.append(f"{path} is not part of the analyzed snapshot")
                elif snapshot != expected:
                    problems.append(f"{path} does not match the analyzed snapshot")
    seq_campaign = sequence.identity.campaign_id
    if context.campaign_id and seq_campaign and seq_campaign != context.campaign_id:
        problems.append("the sequence belongs to another campaign")
    seq_deployment = sequence.identity.deployment
    deployment_problem = _deployment_problem(seq_deployment, context.deployment)
    if deployment_problem:
        problems.append(deployment_problem)
    compiler_problem = _identity_agreement(
        "compiler configuration",
        sequence.identity.compiler_configuration,
        context.compiler_configuration,
    )
    if compiler_problem:
        problems.append(compiler_problem)
    implementation_problem = _identity_agreement(
        "implementation",
        sequence.identity.implementation,
        context.implementation,
    )
    if implementation_problem:
        problems.append(implementation_problem)
    problems.extend(model_source_problems(model, sources))
    reported = _reported_deployment(seq_deployment, context.deployment)
    return ExecutionIdentity(
        campaign_id=context.campaign_id or seq_campaign,
        source_snapshot=context.source_snapshot or sequence.identity.source_snapshot,
        source_files=tuple(sorted(sources)),
        source_hashes=tuple(sorted(hashes.items())),
        source_set_digest=source_set_digest(hashes),
        contract=contract,
        contract_file=contract_file,
        chain_id=context.chain_id,
        address=context.address,
        deployment=reported,
        runtime_digest=context.runtime_digest,
        proxy_kind=context.proxy_kind or "none",
        implementation=context.implementation,
        compiler_configuration=context.compiler_configuration
        or sequence.identity.compiler_configuration,
        solc_version=solc_version,
        property_id=property_id,
        sequence_id=sequence.sequence_id,
        call_instances=instance_identities(sequence),
        replay_mode=ReplayMode.LOCAL_SOURCE_REPLAY,
        status="identity_mismatch" if problems else "bound",
        mismatches=tuple(problems[:12]),
    )


# ---- harness synthesis ------------------------------------------------------------------------


@dataclass(frozen=True)
class Primitive:
    """A bounded local environment step. It never widens scope, auth, budget, or network."""

    name: str
    purpose: str
    actor: str
    value: str
    property_relevance: str
    established_by: str
    limits: str = "local test fixture only; fresh chain; no RPC; no real assets"

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "actor": self.actor,
            "value": self.value,
            "property_relevance": self.property_relevance,
            "established_by": self.established_by,
            "limits": self.limits,
        }


MAX_DEPLOYED = 4  # target + at most three graph-justified dependencies
MAX_DEPENDENCY_DEPTH = 2


@dataclass(frozen=True)
class Relationship:
    """Why a non-target contract is deployed in the harness.

    ``basis`` is ``type_relationship`` (a constructor parameter is typed with that
    contract), ``sole_implementation`` (the parameter is an interface and exactly one
    deployable contract in the exact source set implements it), or ``token_fixture``.
    ``graph_edges`` lists Phase 47 protocol-graph edges between the two contracts when
    the graph resolves any; an empty list is reported, never filled in.
    """

    owner: str
    parameter: str
    declared_type: str
    deployed: str
    basis: str
    depth: int
    graph_edges: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "parameter": self.parameter,
            "declared_type": self.declared_type,
            "deployed": self.deployed,
            "basis": self.basis,
            "depth": self.depth,
            "graph_edges": list(self.graph_edges),
        }


@dataclass(frozen=True)
class HarnessArtifact:
    sequence_id: str
    source: str
    test_name: str
    pipeline: str
    reason: str = ""
    reason_code: str = ""
    contract: str = ""
    contract_file: str = ""
    oracle: OracleSpec = field(default_factory=OracleSpec)
    primitives: tuple[Primitive, ...] = ()
    deployed: tuple[str, ...] = ()
    # Why every non-target contract is in the harness (bounded, graph-justified).
    relationships: tuple[Relationship, ...] = ()
    # The run body and imports, so another engine can host the same plan verbatim.
    body: str = ""
    imports: tuple[str, ...] = ()
    fixture: bool = False

    @property
    def buildable(self) -> bool:
        return bool(self.source)

    @property
    def harness_hash(self) -> str:
        return sha256_text(self.source) if self.source else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "schema": HARNESS_SCHEMA,
            "test_name": self.test_name,
            "pipeline": self.pipeline,
            "buildable": self.buildable,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "contract": self.contract,
            "contract_file": self.contract_file,
            "oracle": self.oracle.as_dict(),
            "primitives": [item.as_dict() for item in self.primitives],
            "deployed": list(self.deployed),
            "relationships": [item.as_dict() for item in self.relationships],
            "harness_hash": self.harness_hash,
        }


_ELEMENTARY = re.compile(r"(address|bool|string|bytes(?:[1-9]|[12]\d|3[0-2])?|u?int\d*)")
_LOCATION = re.compile(r"\b(memory|calldata|storage|payable)\b")
_TOKEN_TYPE = re.compile(r"^I?ERC20\w*$|^IERC4626$|^SafeERC20$|^IWETH\w*$")
_AMOUNT_LIKE = frozenset(
    {
        "amount_one",
        "minimum_amount",
        "victim_amount",
        "credited_balance",
        "all_shares",
        "unconstrained",
        "attacker_controlled",
        "maximum",
    }
)
TOKEN_ORACLES = frozenset(
    {
        OracleKind.CREDIT_LE_RECEIVED,
        OracleKind.CREDIT_LE_HOLDINGS,
        OracleKind.BALANCE_NOT_INCREASED,
    }
)


def _strip_type(type_name: str) -> str:
    return " ".join(_LOCATION.sub(" ", type_name).split())


def _is_token_type(model: ResearchModel, type_name: str) -> bool:
    base = re.sub(r"\[.*\]", "", _strip_type(type_name)).strip()
    if _TOKEN_TYPE.match(base):
        return True
    item = model.contracts.get(base)
    if item is None or not item.is_interface:
        return False
    names = {fn.name for fn in item.functions}
    return {"transfer", "transferFrom", "balanceOf"} <= names


def abi_type(model: ResearchModel, type_name: str) -> str | None:
    """The canonical ABI type of a declared Solidity type, or None (structs, mappings...)."""
    text = _strip_type(type_name)
    match = re.fullmatch(r"([A-Za-z_][\w.]*)\s*((?:\[\d*\])*)", text)
    if match is None:
        return None
    base, suffix = match.group(1), match.group(2)
    if _ELEMENTARY.fullmatch(base):
        if base == "uint":
            base = "uint256"
        elif base == "int":
            base = "int256"
        return base + suffix
    name = base.rsplit(".", 1)[-1]
    item = model.contracts.get(name)
    if item is not None and item.kind in {"contract", "interface"}:
        return "address" + suffix
    if _TOKEN_TYPE.match(name):
        return "address" + suffix
    return None


def canonical_signature(model: ResearchModel, function: RFunction) -> str | None:
    types: list[str] = []
    for param in function.params:
        item = abi_type(model, param.type_name)
        if item is None:
            return None
        types.append(item)
    return f"{function.name}({','.join(types)})"


def _deployable(model: ResearchModel, contract: str, sources: Mapping[str, str] | None) -> bool:
    item = model.contracts.get(contract)
    if item is None or item.kind != "contract":
        return False
    text = (sources or {}).get(item.file, "")
    return not re.search(rf"\babstract\s+contract\s+{re.escape(contract)}\b", text)


def _constructor(model: ResearchModel, contract: str) -> RFunction | None:
    for function in model.functions_of(contract, inherited=False):
        if function.kind == "constructor":
            return function
    return None


def _resolve(model: ResearchModel, contract: str, signature: str) -> RFunction | None:
    for function in model.functions_of(contract):
        if function.signature == signature and function.has_body:
            return function
    return None


@lru_cache(maxsize=32)
def _protocol_graph_edges(items: tuple[tuple[str, str], ...]) -> frozenset[tuple[str, str, str]]:
    """Resolved Phase 47 protocol-graph edges (source, target, kind) for a source set."""
    from app.parsing.engine import parse_source
    from app.parsing.solidity_ir import build_semantic_program
    from app.parsing.solidity_protocol import build_protocol_graph

    programs = []
    for rel, text in items:
        try:
            programs.append(build_semantic_program(parse_source("solidity", Path(rel), text)))
        except (OSError, UnicodeError, ValueError, RuntimeError):
            return frozenset()
    if not programs:
        return frozenset()
    graph = build_protocol_graph(
        tuple(programs),
        project="harness",
        source_snapshot=source_set_digest({rel: sha256_text(text) for rel, text in items}),
        compiler_configuration="harness",
        sources=dict(items),
    )
    names = {node.node_id: node.contract for node in graph.nodes}
    return frozenset(
        (names.get(edge.source, ""), names.get(edge.target, ""), str(edge.kind))
        for edge in graph.edges
    )


def protocol_edges(sources: Mapping[str, str], owner: str, other: str) -> tuple[str, ...]:
    """Phase 47 edges between two contracts (either direction), when the graph has any."""
    items = tuple(sorted((k, v) for k, v in sources.items() if k.endswith(".sol")))
    edges = _protocol_graph_edges(items)
    return tuple(sorted(f"{a}->{b}:{kind}" for a, b, kind in edges if {a, b} == {owner, other}))


class _UnbuildableError(Exception):
    def __init__(self, reason: str, code: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


class _Synth:
    """Assembles one harness. Raises _UnbuildableError instead of inventing anything."""

    def __init__(
        self,
        model: ResearchModel,
        sequence: Vfcs,
        contract: str,
        oracle: OracleSpec,
        sources: Mapping[str, str] | None,
        profile: Any = None,
    ) -> None:
        self.model = model
        self.sequence = sequence
        self.contract = contract
        self.oracle = oracle
        self.sources = sources
        self.profile = profile
        self.profile_contracts = ""
        self.profile_imports: tuple[str, ...] = ()
        self.pre_deploy: list[str] = []
        self.post_deploy: list[str] = []
        self.steps: list[str] = []
        self.primitives: list[Primitive] = []
        self.deployed: list[str] = [contract]
        self.relationships: list[Relationship] = []
        self.token_kind = ""  # standard | fee_on_transfer | false_return
        self.token_bound = False  # the target can see the fixture token
        self.failing_callee = False
        self.funded: set[str] = set()
        self.approved: set[str] = set()
        self.counter = 0
        self.body = ""

    # -- token fixture -------------------------------------------------------------------
    def want_token(self, kind: str, why: str) -> None:
        if not self.token_kind:
            self.token_kind = kind or "standard"
            fee, false_return = {
                "standard": ("0", "false"),
                "fee_on_transfer": ("100", "false"),
                "false_return": ("0", "true"),
            }.get(self.token_kind, ("0", "false"))
            self.pre_deploy.append(
                f"        BugForgeFixtureToken bfToken = new BugForgeFixtureToken({fee}, "
                f"{false_return});"
            )
            self.primitives.append(
                Primitive(
                    name="test_deploy:fixture_token",
                    purpose=f"a local {self.token_kind} ERC-20 fixture",
                    actor="harness",
                    value=f"fee_bps={fee}, returns_false={false_return}",
                    property_relevance=why,
                    established_by="oracle token fixture / token-typed parameter",
                )
            )

    def fund(self, actor_name: str, actor_addr: str) -> None:
        if actor_name in self.funded:
            return
        self.funded.add(actor_name)
        self.post_deploy.append(f"        bfToken.mint({actor_addr}, 4 * {FIXTURE_AMOUNT});")
        self.primitives.append(
            Primitive(
                name="funding:fixture_mint",
                purpose=f"give the {actor_name} fixture tokens to act with",
                actor=actor_name,
                value=f"4 * {FIXTURE_AMOUNT}",
                property_relevance="the sequence moves tokens on the actor's behalf",
                established_by="token fixture",
            )
        )

    def approve(self, actor_name: str, actor_addr: str, why: str) -> None:
        if actor_name in self.approved:
            return
        self.approved.add(actor_name)
        self.fund(actor_name, actor_addr)
        self.post_deploy.append(f"        vm.prank({actor_addr});")
        self.post_deploy.append("        bfToken.approve(address(t), type(uint256).max);")
        self.primitives.append(
            Primitive(
                name="approval:fixture_approve",
                purpose=f"the {actor_name} approves the target for the fixture token",
                actor=actor_name,
                value="type(uint256).max (fixture only)",
                property_relevance=why,
                established_by="the target pulls tokens with transferFrom",
            )
        )

    # -- values ---------------------------------------------------------------------------
    def _name(self, stem: str) -> str:
        self.counter += 1
        return f"bf{stem}{self.counter}"

    def literal(self, param: Param, call: VfcsCall, override: str, nested_ok: bool) -> str:
        abi = abi_type(self.model, param.type_name)
        if abi is None:
            raise _UnbuildableError(
                f"parameter {param.name or '?'} ({param.type_name}) is not expressible",
                "param_not_expressible",
            )
        if override.startswith("nested_call:"):
            if not nested_ok or abi not in {"bytes", "bytes[]"}:
                raise _UnbuildableError(
                    "a nested call cannot be encoded here", "param_not_expressible"
                )
            encoded = self.nested(override.split(":", 1)[1])
            if abi == "bytes":
                return encoded
            name = self._name("Arr")
            self.steps.append(f"        bytes[] memory {name} = new bytes[](1);")
            self.steps.append(f"        {name}[0] = {encoded};")
            return name
        if _is_token_type(self.model, param.type_name) and not abi.endswith("]"):
            self.want_token(self.oracle.token_fixture or "standard", "a token-typed parameter")
            self.token_bound = True
            return "address(bfToken)"
        if abi.endswith("[]"):
            inner = abi[:-2]
            if "[" in inner:
                raise _UnbuildableError(
                    "nested arrays are not expressible", "param_not_expressible"
                )
            return f"new {inner}[](0)"
        if "[" in abi:
            raise _UnbuildableError(
                "fixed-size arrays are not expressible", "param_not_expressible"
            )
        if abi == "address":
            if override == "reverting_callee":
                if not self.failing_callee:
                    raise _UnbuildableError(
                        "a reverting callee was requested before the harness deployed one",
                        "unsupported_primitive",
                    )
                return "address(bfCallee)"
            if override in {VICTIM, "victim", "second_owner"}:
                return VICTIM_ADDR
            if override in {"attacker_controlled", "attacker_substituted", ATTACKER}:
                return ATTACKER_ADDR
            return ATTACKER_ADDR if call.actor == ATTACKER else VICTIM_ADDR
        if abi.startswith("uint") or abi.startswith("int"):
            if override == "max_uint":
                return f"type({abi}).max"
            if override == "zero":
                return "0"
            if override == "one":
                return "1"
            if self.token_kind and override in _AMOUNT_LIKE and abi in {"uint256", "uint"}:
                return FIXTURE_AMOUNT
            if override == "amount_one":
                return "1"
            return "0"
        if abi == "bool":
            if override in {"true", "false"}:
                return override
            return "false"
        if abi == "string":
            return '""'
        if abi == "bytes":
            # Empty calldata still hits the reverting callee's fallback.
            return 'bytes("")'
        if abi.startswith("bytes"):
            return f"{abi}(0)"
        raise _UnbuildableError(f"type {abi} is not expressible", "param_not_expressible")

    def nested(self, target: str) -> str:
        contract, _, signature = target.partition(".")
        callee = _resolve(self.model, contract, signature)
        if callee is None or contract != self.contract:
            raise _UnbuildableError(
                "the nested callee is not a function of the target", "nested_unresolved"
            )
        canonical = canonical_signature(self.model, callee)
        if canonical is None:
            raise _UnbuildableError("the nested callee's parameters are not expressible", "param")
        probe = VfcsCall(contract, signature, "nested", ATTACKER)
        args = [self.literal(p, probe, "attacker_controlled", False) for p in callee.params]
        return f"abi.encodeWithSignature({', '.join([json.dumps(canonical), *args])})"

    def encode(self, call: VfcsCall, function: RFunction) -> str:
        canonical = canonical_signature(self.model, function)
        if canonical is None:
            raise _UnbuildableError(
                f"a parameter of {function.name} is not ABI-expressible", "param_not_expressible"
            )
        overrides = dict(call.arguments)
        args = [
            self.literal(param, call, overrides.get(param.name, ""), True)
            for param in function.params
        ]
        return f"abi.encodeWithSignature({', '.join([json.dumps(canonical), *args])})"

    def _from_profile(self, owner: str, param: Param) -> str | None:
        """A typed profile expression, or None when no profile was supplied.

        A profile that names this contract and then fails its binding is an error.
        It is not permission to fill the argument with a guessed value.
        """
        profile = self.profile
        if profile is None or profile.contract != owner:
            return None
        if param.name != profile.parameter:
            raise _UnbuildableError(
                f"constructor parameter {param.name or '?'} is not the field profile "
                f"{profile.profile_id} configures",
                "profile_parameter_mismatch",
            )
        from app.discovery.bounty.constructor_profile import ProfileError, prepare_profile

        try:
            prepared = prepare_profile(profile, self.sources or {})
        except ProfileError as exc:
            raise _UnbuildableError(exc.reason, exc.code) from exc
        stripped = _strip_type(param.type_name)
        if stripped != profile.model_type:
            raise _UnbuildableError(
                f"{param.name} is {stripped}, not profile type {profile.model_type}",
                "profile_type_mismatch",
            )
        self.pre_deploy.extend(prepared.deploy_lines)
        self.profile_contracts = prepared.contracts
        self.profile_imports = prepared.import_paths
        self.primitives.append(
            Primitive(
                name=f"constructor_profile:{profile.profile_id}",
                purpose=f"apply trusted profile {profile.profile_id} to {param.name}",
                actor="harness",
                value=profile.source_sha256,
                property_relevance=profile.property_statement,
                established_by="the profile hashes match the source set",
            )
        )
        return prepared.expression

    # -- constructor ---------------------------------------------------------------------
    def constructor_args(self) -> list[str]:
        return self._ctor_args(self.contract, 0, (self.contract,))

    def _ctor_args(self, contract: str, depth: int, chain: tuple[str, ...]) -> list[str]:
        ctor = _constructor(self.model, contract)
        if ctor is None or not ctor.params:
            return []
        args: list[str] = []
        for param in ctor.params:
            type_text = _strip_type(param.type_name)
            if _is_token_type(self.model, param.type_name) and "[" not in type_text:
                self.want_token(
                    self.oracle.token_fixture or "standard",
                    f"constructor parameter {param.name} of {contract} is a token",
                )
                self.token_bound = True
                args.append(f"{type_text}(address(bfToken))")
                continue
            args.append(self._dependency(contract, param, type_text, depth + 1, chain))
        return args

    def _implementation(self, declared: str) -> tuple[str, str]:
        """(contract to deploy, basis) for a contract-typed parameter, or raise."""
        item = self.model.contracts.get(declared)
        if item is not None and _deployable(self.model, declared, self.sources):
            return declared, "type_relationship"
        if item is None:
            return "", ""
        implementations = sorted(
            name
            for name in self.model.contracts
            if name != declared
            and declared in self.model.bases_of(name)
            and _deployable(self.model, name, self.sources)
        )
        if len(implementations) == 1:
            return implementations[0], "sole_implementation"
        if len(implementations) > 1:
            raise _UnbuildableError(
                f"{declared} has {len(implementations)} implementations in the source set; "
                "the harness does not pick one",
                "ambiguous_dependency",
            )
        return "", ""

    def _dependency(
        self, owner: str, param: Param, type_text: str, depth: int, chain: tuple[str, ...]
    ) -> str:
        declared = self.model.contract_for_type(type_text)
        deployed, basis = self._implementation(declared) if declared else ("", "")
        if not deployed:
            rendered = self._from_profile(owner, param)
            if rendered is not None:
                return rendered
            raise _UnbuildableError(
                f"constructor argument {param.name or '?'} ({param.type_name}) of {owner} is "
                "unknown; it is not invented",
                "unknown_constructor_argument",
            )
        if deployed in chain:
            raise _UnbuildableError(
                f"{owner} and {deployed} need each other at construction; the cycle is not "
                "resolved by guessing an address",
                "cyclic_dependency",
            )
        if depth > MAX_DEPENDENCY_DEPTH or len(self.deployed) >= MAX_DEPLOYED:
            raise _UnbuildableError(
                f"deploying {deployed} exceeds the harness bound ({MAX_DEPLOYED} contracts, "
                f"depth {MAX_DEPENDENCY_DEPTH})",
                "dependency_bound_exceeded",
            )
        args = self._ctor_args(deployed, depth, (*chain, deployed))
        name = self._name("Dep")
        self.pre_deploy.append(f"        {deployed} {name} = new {deployed}({', '.join(args)});")
        self.deployed.append(deployed)
        self.relationships.append(
            Relationship(
                owner=owner,
                parameter=param.name,
                declared_type=type_text,
                deployed=deployed,
                basis=basis,
                depth=depth,
                graph_edges=protocol_edges(self.sources or {}, owner, deployed),
            )
        )
        self.primitives.append(
            Primitive(
                name="test_deploy:graph_dependency",
                purpose=f"deploy {deployed} for constructor parameter {param.name} of {owner}",
                actor="harness",
                value=deployed,
                property_relevance=f"{owner} cannot be deployed without it",
                established_by=f"{basis}: {param.name} is typed {type_text}",
            )
        )
        if deployed != declared:
            return f"{type_text}(address({name}))"
        return name

    # -- primitives ------------------------------------------------------------------------
    def primitive(self, index: int, call: VfcsCall) -> None:
        actor_addr = ATTACKER_ADDR if call.actor == ATTACKER else VICTIM_ADDR
        if call.function == "deploy_reverting()":
            self.failing_callee = True
            self.steps.append(
                "        BugForgeRevertingCallee bfCallee = new BugForgeRevertingCallee();"
            )
            self.steps.append(f"        emit BugForgeObservation({K_PRIMITIVE}, {index}, 1);")
            self.primitives.append(
                Primitive(
                    name="test_deploy:reverting_callee",
                    purpose="a local callee that reverts on every call",
                    actor="harness",
                    value="fallback revert",
                    property_relevance=call.established_by,
                    established_by=call.established_by,
                )
            )
            return
        if call.function == "warp(uint256)":
            self.steps.append("        vm.warp(block.timestamp + 1 days);")
            self.steps.append(f"        emit BugForgeObservation({K_PRIMITIVE}, {index}, 1);")
            self.primitives.append(
                Primitive(
                    name="time_advance:warp",
                    purpose="advance the local clock",
                    actor="harness",
                    value="+1 day",
                    property_relevance=call.established_by,
                    established_by=call.established_by,
                )
            )
            return
        if call.contract == "token" and call.function in {
            "transfer(address,uint256)",
            "approve(address,uint256)",
        }:
            self.want_token(self.oracle.token_fixture or "standard", call.established_by)
            self.fund(call.actor, actor_addr)
            verb = "transfer" if call.function.startswith("transfer") else "approve"
            amount = FIXTURE_AMOUNT if verb == "transfer" else "type(uint256).max"
            ok = self._name("Prim")
            self.steps.append(f"        vm.prank({actor_addr});")
            self.steps.append(
                f"        (bool {ok}, ) = address(bfToken).call(abi.encodeWithSignature("
                f'"{verb}(address,uint256)", address(t), {amount}));'
            )
            self.steps.append(
                f"        emit BugForgeObservation({K_PRIMITIVE}, {index}, {ok} ? 1 : 0);"
            )
            self.primitives.append(
                Primitive(
                    name=f"{'donation' if verb == 'transfer' else 'approval'}:fixture_{verb}",
                    purpose=f"the {call.actor} {verb}s fixture tokens to/for the target",
                    actor=call.actor,
                    value=amount,
                    property_relevance=call.established_by,
                    established_by=call.established_by,
                )
            )
            return
        raise _UnbuildableError(
            f"call {index} is an environment primitive ({call.contract}.{call.function}) "
            "that needs a fixture the local harness does not provide",
            "unsupported_primitive",
        )

    # -- oracle reads ----------------------------------------------------------------------
    def read_getter(self, kind: int, index: int) -> None:
        oracle = self.oracle
        getter = oracle.getter
        if not getter:
            return
        if getter.endswith("(address)"):
            actor = ATTACKER_ADDR if oracle.getter_actor in {"", ATTACKER} else VICTIM_ADDR
            data = f'abi.encodeWithSignature("{getter}", {actor})'
        else:
            data = f'abi.encodeWithSignature("{getter}")'
        ok, value = self._name("Ok"), self._name("Val")
        self.steps.append(f"        (bool {ok}, uint256 {value}) = _bfRead(address(t), {data});")
        self.steps.append(
            f"        if ({ok}) {{ emit BugForgeObservation({kind}, {index}, {value}); }} "
            f"else {{ emit BugForgeObservation({K_READ_FAILED}, {index}, {kind}); }}"
        )

    def read_balance(self, index: int, slot: int) -> None:
        value = self._name("Bal")
        self.steps.append(f"        uint256 {value} = bfToken.balanceOf(address(t));")
        self.steps.append(f"        emit BugForgeObservation({K_RECEIVED}, {slot}, {value});")

    def read_credit(self, slot: int) -> None:
        self.read_getter(K_CREDITED, slot)

    # -- assemble --------------------------------------------------------------------------
    def build(self) -> str:
        oracle = self.oracle
        probe = probe_index(self.sequence, oracle) if oracle.executable else -1
        if oracle.kind in TOKEN_ORACLES:
            self.want_token(oracle.token_fixture or "standard", oracle.expression)
        ctor_args = self.constructor_args()
        if oracle.kind in TOKEN_ORACLES:
            self.approve(ATTACKER, ATTACKER_ADDR, "the probe deposits fixture tokens")
        for index, call in enumerate(self.sequence.calls):
            if call.primitive:
                self.primitive(index, call)
                continue
            function = _resolve(self.model, call.contract, call.function)
            if function is None or call.contract != self.contract:
                raise _UnbuildableError(
                    "a call is not a function of the target contract", "call_outside_target"
                )
            encoded = self.encode(call, function)
            if (
                index == probe
                and oracle.getter
                and oracle.kind
                in {
                    OracleKind.STATE_UNCHANGED,
                    OracleKind.CALL_MUST_FAIL,
                }
            ):
                self.read_getter(K_BEFORE, index)
            if index == probe and oracle.kind in TOKEN_ORACLES:
                self.read_balance(index, 0)
                self.read_credit(0)
            sender = ATTACKER_ADDR if call.actor == ATTACKER else VICTIM_ADDR
            if call.actor not in {ATTACKER, VICTIM}:
                raise _UnbuildableError(
                    f"actor {call.actor!r} is not a harness actor", "unknown_actor"
                )
            self.steps.append(f"        vm.prank({sender});")
            self.steps.append(f"        (bool ok_{index}, ) = address(t).call({encoded});")
            self.steps.append(
                f"        emit BugForgeObservation({K_CALL}, {index}, ok_{index} ? 1 : 0);"
            )
            if (
                index == probe
                and oracle.getter
                and oracle.kind
                in {
                    OracleKind.STATE_UNCHANGED,
                    OracleKind.CALL_MUST_FAIL,
                }
            ):
                self.read_getter(K_AFTER, index)
            if index == probe and oracle.kind in TOKEN_ORACLES:
                self.read_balance(index, 1)
                self.read_credit(1)
        if self.token_kind and not self.token_bound:
            raise _UnbuildableError(
                f"{self.contract} reads its token from state the harness cannot bind "
                "(no token parameter or constructor argument)",
                "unbound_token",
            )
        deploy = f"        {self.contract} t = new {self.contract}({', '.join(ctor_args)});"
        body = "\n".join(
            [
                *self.pre_deploy,
                deploy,
                f"        emit BugForgeObservation({K_DEPLOYED}, 0, 1);",
                *self.post_deploy,
                *self.steps,
                f"        emit BugForgeObservation({K_DONE}, {len(self.sequence.calls)}, 1);",
            ]
        )
        self.body = body
        import_lines = [f'import "src/{path}";\n' for path in self.imports()]
        for path in self.profile_imports:
            line = f'import "src/{path}";\n'
            if line not in import_lines:
                import_lines.append(line)
        return _HARNESS_TEMPLATE.format(
            schema=HARNESS_SCHEMA,
            imports="".join(import_lines),
            fixture=_FIXTURE_TOKEN if self.token_kind else "",
            reverting=_REVERTING_CALLEE if self.failing_callee else "",
            profile_contracts=self.profile_contracts,
            body=body,
        )

    def imports(self) -> tuple[str, ...]:
        """Every declaring file of a deployed contract, target first, de-duplicated."""
        files: list[str] = []
        for name in self.deployed:
            path = self.model.contracts[name].file
            if path not in files:
                files.append(path)
        return tuple(files)


def build_harness(
    sequence: Vfcs,
    model: ResearchModel,
    *,
    oracle: OracleSpec | None = None,
    pipeline: str = "default",
    sources: Mapping[str, str] | None = None,
    source_file: str = "",
    profile: Any = None,
) -> HarnessArtifact:
    """Compile a VFCS plan (and its oracle) into a Foundry test, or explain why not.

    The import is the declaring file of the *target* contract, never a generic first
    file. ``source_file`` is accepted only as an expectation: a different file is an
    identity mismatch and nothing is built.
    """
    oracle = oracle or OracleSpec()
    contract = target_contract(sequence)
    item = model.contracts.get(contract)
    contract_file = item.file if item is not None else ""

    def fail(reason: str, code: str) -> HarnessArtifact:
        return HarnessArtifact(
            sequence.sequence_id,
            "",
            "",
            pipeline,
            reason,
            code,
            contract,
            contract_file,
            oracle,
        )

    if len(sequence.calls) > MAX_CALLS:
        return fail("sequence too long", "too_long")
    if source_file and contract_file and source_file != contract_file:
        return fail(
            f"{contract} is declared in {contract_file}, not {source_file}", "identity_mismatch"
        )
    if item is None or not _deployable(model, contract, sources):
        return fail("target contract is not a deployable source", "not_deployable")
    synth = _Synth(model, sequence, contract, oracle, sources, profile)
    try:
        source = synth.build()
    except _UnbuildableError as exc:
        return fail(exc.reason, exc.code)
    return HarnessArtifact(
        sequence_id=sequence.sequence_id,
        source=source,
        test_name="test_vfcs_execute",
        pipeline=pipeline,
        contract=contract,
        contract_file=contract_file,
        oracle=oracle,
        primitives=tuple(synth.primitives),
        deployed=tuple(synth.deployed),
        relationships=tuple(synth.relationships),
        body=synth.body,
        imports=synth.imports(),
        fixture=bool(synth.token_kind),
    )


_HARNESS_TEMPLATE = """// SPDX-License-Identifier: UNLICENSED
// BUGFORGE LOCAL HARNESS ({schema}). Deterministic local execution only. Not a proof.
// The harness only reports observations; the property is judged outside the EVM.
pragma solidity >=0.7.0;

import {{Test}} from "forge-std/Test.sol";
{imports}{fixture}{reverting}{profile_contracts}
contract VfcsHarness is Test {{
    event BugForgeObservation(uint256 kind, uint256 index, uint256 value);

    function _bfRead(address target, bytes memory data) internal view returns (bool, uint256) {{
        (bool ok, bytes memory out) = target.staticcall(data);
        if (!ok || out.length < 32) return (false, 0);
        return (true, abi.decode(out, (uint256)));
    }}

    function test_vfcs_execute() public {{
{body}
    }}
}}
"""

_FIXTURE_TOKEN = """
contract BugForgeFixtureToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    uint256 public totalSupply;
    uint256 public immutable feeBps;
    bool public immutable returnsFalse;

    constructor(uint256 fee, bool falseReturn) {
        feeBps = fee;
        returnsFalse = falseReturn;
    }

    function decimals() external pure returns (uint8) {
        return 18;
    }

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        return _move(msg.sender, to, amount);
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        if (returnsFalse) return false;
        uint256 allowed = allowance[from][msg.sender];
        require(allowed >= amount, "allowance");
        if (allowed != type(uint256).max) allowance[from][msg.sender] = allowed - amount;
        return _move(from, to, amount);
    }

    function _move(address from, address to, uint256 amount) internal returns (bool) {
        if (returnsFalse) return false;
        require(balanceOf[from] >= amount, "balance");
        uint256 fee = (amount * feeBps) / 10000;
        balanceOf[from] -= amount;
        balanceOf[to] += amount - fee;
        totalSupply -= fee;
        return true;
    }
}
"""

_REVERTING_CALLEE = """
contract BugForgeRevertingCallee {
    fallback() external payable {
        revert("bf-revert");
    }
}
"""

# forge-std is big; BugForge does not vendor it. A tiny local shim provides only the
# Test base and the cheatcodes the harness uses, so execution needs no download.
_FORGE_STD = """// SPDX-License-Identifier: MIT
pragma solidity >=0.7.0;

interface Vm {
    function prank(address) external;
    function warp(uint256) external;
    function roll(uint256) external;
    function deal(address, uint256) external;
}

contract Test {
    Vm internal constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));
}
"""


# ---- structured execution results -------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionResult:
    """What one local run produced, structurally. Built from process status and forge's
    JSON report (raw harness events), never from free-text heuristics."""

    process_status: str  # not_run | completed | timeout | spawn_failed
    compiler_status: str  # not_run | ok | failed
    test_status: str  # not_run | passed | failed | missing
    sequence_status: str  # not_run | completed | reverted | incomplete
    property_status: str  # not_run | violated | held | not_evaluated | no_oracle
    return_code: int | None = None
    reverted_index: int = -1
    call_results: tuple[int, ...] = ()  # per call: 1 ok, 0 reverted, -1 not reached
    events: tuple[tuple[int, int, int], ...] = ()
    assertion_id: str = ""  # the property id the oracle judged
    failure_reason: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    forge_version: str = ""
    solc_version: str = ""
    pipeline: str = "default"
    source_hashes: tuple[tuple[str, str], ...] = ()
    harness_hash: str = ""
    config_hash: str = ""
    command: tuple[str, ...] = ()
    duration_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "process_status": self.process_status,
            "compiler_status": self.compiler_status,
            "test_status": self.test_status,
            "sequence_status": self.sequence_status,
            "property_status": self.property_status,
            "return_code": self.return_code,
            "reverted_index": self.reverted_index,
            "call_results": list(self.call_results),
            "events": [list(item) for item in self.events],
            "assertion_id": self.assertion_id,
            "failure_reason": self.failure_reason,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "forge_version": self.forge_version,
            "solc_version": self.solc_version,
            "pipeline": self.pipeline,
            "source_hashes": dict(self.source_hashes),
            "harness_hash": self.harness_hash,
            "config_hash": self.config_hash,
            "command": list(self.command),
            "duration_ms": self.duration_ms,
        }


def parse_forge_report(stdout: str) -> dict[str, Any] | None:
    """The JSON document ``forge test --json`` printed, or None."""
    start = stdout.find("{")
    if start < 0:
        return None
    try:
        value = json.loads(stdout[start:])
    except ValueError:
        # forge may print one JSON document per line; take the first that parses
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                break
        else:
            return None
    return value if isinstance(value, dict) else None


def harness_test(report: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for suite, body in report.items():
        if not str(suite).endswith(":VfcsHarness") or not isinstance(body, Mapping):
            continue
        tests = body.get("test_results")
        if isinstance(tests, Mapping):
            for name, result in tests.items():
                if str(name).startswith("test_vfcs_execute") and isinstance(result, Mapping):
                    return result
    return None


def decode_events(test: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    """Decode harness observations from raw logs. Only the harness address counts."""
    found: list[tuple[int, int, int]] = []
    for log in test.get("logs") or ():
        if not isinstance(log, Mapping):
            continue
        if str(log.get("address", "")).lower() != HARNESS_ADDRESS:
            continue
        topics = log.get("topics") or ()
        if not topics or str(topics[0]).lower() != OBSERVATION_TOPIC:
            continue
        data = str(log.get("data", ""))
        data = data[2:] if data.startswith("0x") else data
        if len(data) != 192 or not re.fullmatch(r"[0-9a-fA-F]+", data):
            continue
        words = [int(data[i : i + 64], 16) for i in range(0, 192, 64)]
        found.append((words[0], words[1], words[2]))
        if len(found) >= MAX_EVENTS:
            break
    return tuple(found)


def observation_from_events(
    events: tuple[tuple[int, int, int], ...], count: int
) -> tuple[OracleObservation, bool, bool]:
    """(oracle observation, deployed, done) from structured events."""
    calls = [-1] * count
    before = after = ""
    readable = True
    received: dict[int, int] = {}
    credited: dict[int, int] = {}
    deployed = done = False
    for kind, index, value in events:
        if kind == K_DEPLOYED:
            deployed = True
        elif kind in {K_CALL, K_PRIMITIVE} and 0 <= index < count:
            calls[index] = 1 if value == 1 else 0
        elif kind == K_BEFORE:
            before = f"0x{value:064x}"
        elif kind == K_AFTER:
            after = f"0x{value:064x}"
        elif kind == K_READ_FAILED:
            readable = False
        elif kind == K_RECEIVED:
            received[index] = value
        elif kind == K_CREDITED:
            credited[index] = value
        elif kind == K_DONE:
            done = index == count
    observation = OracleObservation(
        call_ok=tuple(calls),
        before=before,
        after=after,
        readable=readable,
        received=received[1] - received[0] if {0, 1} <= set(received) else None,
        credited=credited[1] - credited[0] if {0, 1} <= set(credited) else None,
        holdings_after=received.get(1),
        credit_after=credited.get(1),
    )
    return observation, deployed, done


@dataclass(frozen=True)
class StatefulObservation:
    sequence_id: str
    outcome: Outcome
    reason: str
    pipeline: str
    executed: bool
    return_code: int | None
    reverted_index: int
    property_under_test: str
    property_checked: bool
    deployment: str
    tools: ToolStatus
    stdout_tail: str = ""
    verified: bool = False
    property_id: str = ""
    declaration: str = PropertyDeclaration.NO_PROPERTY_AVAILABLE.value
    verdict: str = PropertyVerdict.NOT_EVALUATED.value
    formulation: str = "primary"
    oracle_kind: str = OracleKind.NONE.value
    evaluation: PropertyEvaluation | None = None
    result: ExecutionResult | None = None
    identity: ExecutionIdentity | None = None
    harness: HarnessArtifact | None = None
    reason_code: str = ""

    @property
    def no_oracle(self) -> bool:
        return self.outcome is Outcome.SEQUENCE_EXECUTED_NO_ORACLE

    @property
    def strong_candidate(self) -> bool:
        """Only an executed, identity-bound oracle violation is a strong candidate."""
        return (
            self.outcome is Outcome.PROPERTY_VIOLATED
            and self.declaration == PropertyDeclaration.PROPERTY_UNDER_TEST.value
            and self.property_checked
            and self.identity is not None
            and self.identity.bound
        )

    def signal(self) -> FeedbackSignal | None:
        """Turn an execution observation into bounded feedback. None = nothing to learn."""
        if self.outcome is Outcome.SEQUENCE_REVERTED and self.reverted_index >= 0:
            return FeedbackSignal(
                kind="near_miss",
                sequence_id=self.sequence_id,
                call_index=self.reverted_index,
                engine="foundry",
            )
        if self.outcome is Outcome.PROPERTY_HELD:
            index = self.evaluation.probe_index if self.evaluation else -1
            return FeedbackSignal(
                kind="near_miss", sequence_id=self.sequence_id, call_index=index, engine="foundry"
            )
        if self.outcome is Outcome.SEQUENCE_EXECUTED_NO_ORACLE:
            return FeedbackSignal(
                kind="coverage_gain", sequence_id=self.sequence_id, engine="foundry"
            )
        # A mechanism probe is evidence, not a near-miss to mutate into a finding.
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "pipeline": self.pipeline,
            "executed": self.executed,
            "return_code": self.return_code,
            "reverted_index": self.reverted_index,
            "property_under_test": self.property_under_test,
            "property_checked": self.property_checked,
            "property_id": self.property_id,
            "declaration": self.declaration,
            "verdict": self.verdict,
            "formulation": self.formulation,
            "oracle_kind": self.oracle_kind,
            "strong_candidate": self.strong_candidate,
            "evaluation": self.evaluation.as_dict() if self.evaluation else None,
            "result": self.result.as_dict() if self.result else None,
            "identity": self.identity.as_dict() if self.identity else None,
            "deployment": self.deployment,
            "tools": self.tools.as_dict(),
            "verified": False,
        }


def classify(
    sequence: Vfcs,
    oracle: OracleSpec,
    observation: OracleObservation,
) -> tuple[Outcome, PropertyEvaluation, str]:
    """Map a completed run to a distinct outcome. Revert != finding; no oracle != finding."""
    calls = observation.call_ok
    reverted = next((i for i, ok in enumerate(calls) if ok == 0), -1)
    if not oracle.executable:
        evaluation = evaluate(sequence, oracle, observation)
        if reverted >= 0:
            return (
                Outcome.SEQUENCE_REVERTED,
                evaluation,
                f"call {reverted} reverted against a fresh local deployment (not a finding)",
            )
        return (
            Outcome.SEQUENCE_EXECUTED_NO_ORACLE,
            evaluation,
            "every call succeeded, but no oracle judged a property: execution only",
        )
    evaluation = evaluate(sequence, oracle, observation)
    if evaluation.verdict is PropertyVerdict.PROPERTY_VIOLATED:
        return Outcome.PROPERTY_VIOLATED, evaluation, evaluation.reason
    if evaluation.verdict is PropertyVerdict.PROPERTY_HELD:
        return Outcome.PROPERTY_HELD, evaluation, evaluation.reason
    if evaluation.verdict is PropertyVerdict.MECHANISM_OBSERVED:
        return Outcome.MECHANISM_OBSERVED, evaluation, evaluation.reason
    if evaluation.verdict is PropertyVerdict.FAILURE_PROPAGATED:
        return Outcome.FAILURE_PROPAGATED, evaluation, evaluation.reason
    return Outcome.INCONCLUSIVE, evaluation, evaluation.reason


# ---- execution --------------------------------------------------------------------------------

_PROXY_KINDS = frozenset({"uups", "transparent", "beacon", "diamond", "clone", "generic_proxy"})


# Untrusted analyzed sources run inside forge/echidna/medusa, where any contract can
# call the cheatcode address. ffi and file access are off by config and the
# environment is minimal, but a fork/RPC cheatcode could still reach the network,
# so sources that reference the cheatcode address or a dangerous cheatcode by name
# are refused before anything runs. (Obfuscated selectors remain a residual risk
# that only an OS-level network sandbox closes; that is outside this path.)
_CHEAT_ADDRESS = re.compile(r"0x7109709ecfa91a80626ff3989d68f67f5b1dd12d|hevm cheat code", re.I)
_CHEAT_CALL = re.compile(
    r"\b(createFork|createSelectFork|selectFork|rollFork|makePersistent|ffi|tryFfi|rpc"
    r"|rpcUrl|rpcUrls|readFile|readFileBinary|readLine|writeFile|writeFileBinary|writeLine"
    r"|removeFile|readDir|envString|envUint|envInt|envAddress|envBytes32|envBytes|envBool"
    r"|envOr|envExists|setEnv|startBroadcast|broadcast|httpGet|httpPost|prompt"
    r"|promptSecret|projectRoot)\s*\("
)


def untrusted_cheatcode_use(sources: Mapping[str, str]) -> tuple[str, ...]:
    """``file: construct`` for every cheatcode reference in untrusted sources."""
    from app.parsing.solidity_research import strip_comments

    found: list[str] = []
    for path, text in sorted(sources.items()):
        clean = strip_comments(text)
        for pattern in (_CHEAT_ADDRESS, _CHEAT_CALL):
            match = pattern.search(clean)
            if match is not None:
                found.append(f"{path}: {match.group(0)[:40]}")
    return tuple(found[:8])


@dataclass(frozen=True)
class ProjectSettings:
    """Compiler settings copied from the campaign manifest, not from source text."""

    optimizer: bool | None = None
    optimizer_runs: int | None = None
    via_ir: bool | None = None
    evm_version: str = ""
    evm_exact: bool = True
    evm_requested: str = ""


def foundry_config(pipeline: str, solc: str, settings: ProjectSettings | None = None) -> str:
    """Server-owned Foundry config. Nothing from the analyzed repository reaches it.

    ``pipeline`` may force a corroboration variant. It is never interpolated.
    An unsupported requested EVM version is omitted rather than rewritten into
    a different version the compiler happens to accept.
    """
    via_ir = "true" if pipeline == "via_ir" else "false"
    optimizer = "false" if pipeline == "no_optimizer" else "true"
    runs_line = ""
    evm_line = ""
    if settings is not None and pipeline not in {"via_ir", "no_optimizer"}:
        if settings.via_ir is not None:
            via_ir = "true" if settings.via_ir else "false"
        if settings.optimizer is not None:
            optimizer = "true" if settings.optimizer else "false"
    if settings is not None and pipeline != "no_optimizer" and settings.optimizer_runs is not None:
        runs = settings.optimizer_runs
        if 0 <= runs <= 1_000_000:
            runs_line = f"optimizer_runs = {runs}\n"
    if (
        settings is not None
        and settings.evm_exact
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,31}", settings.evm_version or "")
    ):
        evm_line = f"evm_version = {json.dumps(settings.evm_version)}\n"
    solc_line = f"solc = {json.dumps(solc)}\n" if solc else ""
    return (
        "[profile.default]\n"
        'src = "src"\n'
        'test = "test"\n'
        'libs = ["lib"]\n'
        "ffi = false\n"
        "fs_permissions = []\n"
        "auto_detect_solc = false\n"
        "offline = true\n"
        "no_storage_caching = true\n"
        "rpc_endpoints = {}\n"
        + solc_line
        + evm_line
        + f"via_ir = {via_ir}\n"
        + f"optimizer = {optimizer}\n"
        + runs_line
        + 'remappings = ["forge-std/=lib/forge-std/src/"]\n'
        + "\n[lint]\nlint_on_build = false\n"
    )


def _safe_env(home: str) -> dict[str, str]:
    """A minimal environment: no RPC URLs, keys, or tokens from the server reach forge."""
    parts = [p for p in os.environ.get("PATH", "").split(":") if p]
    return {
        "PATH": ":".join(parts) or "/usr/bin:/bin",
        "HOME": home,
        "FOUNDRY_OFFLINE": "true",
        "FOUNDRY_FFI": "false",
        "NO_COLOR": "1",
    }


def scaffold(root: Path, sources: Mapping[str, str], harness_source: str, config: str) -> None:
    """Lay out a self-contained Foundry project from the exact source set."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "lib" / "forge-std" / "src").mkdir(parents=True, exist_ok=True)
    base = (root / "src").resolve()
    for rel, text in sorted(sources.items()):
        target = (base / rel).resolve()
        if base not in target.parents:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (root / "test" / "VfcsHarness.t.sol").write_text(harness_source, encoding="utf-8")
    (root / "lib" / "forge-std" / "src" / "Test.sol").write_text(_FORGE_STD, encoding="utf-8")
    (root / "foundry.toml").write_text(config, encoding="utf-8")


BUILD_ARGV = ("forge", "build", "--offline")
TEST_ARGV = (
    "forge",
    "test",
    "--offline",
    "--json",
    "-vv",
    "--match-path",
    "test/VfcsHarness.t.sol",
)


class StatefulExecutor:
    """Runs synthesized harnesses with the installed ``forge``, offline and bounded."""

    def __init__(
        self,
        *,
        timeout: int | None = None,
        tools: ToolStatus | None = None,
        max_runs: int = MAX_RUNS,
        compiler: Any = None,
        project: ProjectSettings | None = None,
        compiler_required: str = "",
        profile: Any = None,
    ) -> None:
        self.timeout = timeout if timeout is not None else _timeout_setting()
        self.tools = tools if tools is not None else tool_status()
        self.compiler = compiler
        self.project = project
        self.compiler_required = compiler_required
        self.profile = profile
        self.evm_exact = True
        self.evm_requested = ""
        if compiler is not None:
            self.tools = replace(
                self.tools, solc="available", solc_version=str(compiler.reported_version)
            )
            if project is not None and project.evm_version:
                from app.discovery.bounty.solc_resolve import evm_version_for

                _applied, exact = evm_version_for(str(compiler.path), project.evm_version)
                self.evm_exact = exact
                self.evm_requested = project.evm_version
                if not exact:
                    self.project = replace(
                        project, evm_version="", evm_exact=False, evm_requested=project.evm_version
                    )
        self.runs = 0
        # A server-owned run budget; nothing a caller sends can raise it.
        self.max_runs = max(0, min(max_runs, MAX_RUNS))

    def available(self) -> bool:
        return execution_enabled() and self.tools.available

    def execute(
        self,
        sequence: Vfcs,
        model: ResearchModel,
        sources: Mapping[str, str],
        *,
        spec: PropertySpec | None = None,
        oracle: OracleSpec | None = None,
        context: IdentityContext | None = None,
        pipeline: str = "default",
        deployment: str = "",
        source_file: str = "",
    ) -> StatefulObservation:
        spec = spec if spec is not None else build_property(sequence, model)
        chosen = oracle if oracle is not None else spec.oracle
        context = context or IdentityContext(deployment=deployment, expected_file=source_file)
        if source_file and not context.expected_file:
            context = replace(context, expected_file=source_file)
        identity = bind_identity(
            sequence,
            model,
            sources,
            context,
            property_id=spec.property_id,
            solc_version=self.tools.solc_version,
        )
        base = _Base(sequence, spec, chosen, pipeline, identity, self.tools)
        # Identity first: a sequence bound to other sources never runs, tools or not.
        if not identity.bound:
            return base.done(
                Outcome.IDENTITY_MISMATCH, "; ".join(identity.mismatches), "identity_mismatch"
            )
        if not execution_enabled():
            return base.done(Outcome.UNAVAILABLE, "local stateful execution is disabled")
        cheats = untrusted_cheatcode_use(sources)
        if cheats:
            return base.done(
                Outcome.INCONCLUSIVE,
                "the analyzed sources reference cheatcodes; untrusted code is not run with "
                f"cheatcode access ({'; '.join(cheats)})",
                "blocked_by_policy:cheatcode_in_source",
            )
        if self.compiler_required and self.compiler is None:
            return base.done(
                Outcome.UNAVAILABLE,
                f"solc {self.compiler_required} is not installed",
                "compiler_unavailable",
            )
        if not self.tools.available:
            missing = "forge" if self.tools.forge != "available" else "solc"
            return base.done(Outcome.UNAVAILABLE, f"{missing} is not installed", "tool_missing")
        if context.proxy_kind in _PROXY_KINDS:
            return base.done(
                Outcome.INCONCLUSIVE,
                f"the target deployment is a {context.proxy_kind} proxy; deploying the "
                "implementation alone is not the deployed behavior",
                "proxy_requires_compatible_harness",
            )
        from app.parsing.solidity_version import pragma_allows

        contract_text = sources.get(identity.contract_file, "")
        if pragma_allows(contract_text, self.tools.solc_version) is False:
            return base.done(
                Outcome.UNAVAILABLE,
                f"installed solc {self.tools.solc_version} is outside the source pragma",
                "compiler_unavailable",
            )
        if self.runs >= self.max_runs:
            return base.done(
                Outcome.INCONCLUSIVE,
                f"the local run budget ({self.max_runs}) is exhausted",
                "budget_exhausted",
            )
        harness = build_harness(
            sequence,
            model,
            oracle=chosen,
            pipeline=pipeline,
            sources=sources,
            profile=self.profile,
        )
        if not harness.buildable:
            outcome = (
                Outcome.IDENTITY_MISMATCH
                if harness.reason_code == "identity_mismatch"
                else Outcome.INCONCLUSIVE
            )
            return base.done(outcome, harness.reason, harness.reason_code, harness=harness)
        return self._run(base, harness, sources)

    def _run(
        self, base: _Base, harness: HarnessArtifact, sources: Mapping[str, str]
    ) -> StatefulObservation:
        self.runs += 1
        forge = tool_path("forge") or "forge"
        solc = str(self.compiler.path) if self.compiler is not None else (tool_path("solc") or "")
        config = foundry_config(harness.pipeline, solc, self.project)
        common: dict[str, Any] = {
            "pipeline": harness.pipeline,
            "forge_version": self.tools.forge_version,
            "solc_version": self.tools.solc_version,
            "source_hashes": base.identity.source_hashes,
            "harness_hash": harness.harness_hash,
            "config_hash": sha256_text(config),
            "command": (*BUILD_ARGV, "&&", *TEST_ARGV),
            "assertion_id": base.spec.property_id if base.oracle.executable else "",
        }
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="bugforge-vfcs-") as tmp:
            root = Path(tmp)
            scaffold(root, sources, harness.source, config)
            env = _safe_env(tmp)
            try:
                build = subprocess.run(
                    [forge, *BUILD_ARGV[1:]],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return base.done(
                    Outcome.TIMEOUT,
                    f"forge build exceeded {self.timeout}s",
                    "timeout",
                    harness=harness,
                    result=ExecutionResult(
                        "timeout", "not_run", "not_run", "not_run", "not_run", **common
                    ),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return base.done(
                    Outcome.EXECUTION_FAILED,
                    f"forge did not start: {str(exc)[:160]}",
                    "spawn_failed",
                    harness=harness,
                    result=ExecutionResult(
                        "spawn_failed", "not_run", "not_run", "not_run", "not_run", **common
                    ),
                )
            if build.returncode != 0:
                return base.done(
                    Outcome.COMPILE_FAILED,
                    "the harness did not compile against this exact source set",
                    "compile_failed",
                    harness=harness,
                    result=ExecutionResult(
                        "completed",
                        "failed",
                        "not_run",
                        "not_run",
                        "not_run",
                        return_code=build.returncode,
                        failure_reason="compilation failed",
                        stdout_tail=_tail(build.stdout),
                        stderr_tail=_tail(build.stderr),
                        duration_ms=int((time.monotonic() - started) * 1000),
                        **common,
                    ),
                )
            try:
                proc = subprocess.run(
                    [forge, *TEST_ARGV[1:]],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return base.done(
                    Outcome.TIMEOUT,
                    f"forge test exceeded {self.timeout}s",
                    "timeout",
                    harness=harness,
                    result=ExecutionResult(
                        "timeout", "ok", "not_run", "not_run", "not_run", **common
                    ),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return base.done(
                    Outcome.EXECUTION_FAILED,
                    f"forge test did not start: {str(exc)[:160]}",
                    "spawn_failed",
                    harness=harness,
                    result=ExecutionResult(
                        "spawn_failed", "ok", "not_run", "not_run", "not_run", **common
                    ),
                )
        duration = int((time.monotonic() - started) * 1000)
        return interpret(
            base,
            harness,
            proc.returncode,
            proc.stdout,
            proc.stderr,
            duration_ms=duration,
            common=common,
        )


@dataclass
class _Base:
    """Shared fields for every observation of one (sequence, oracle, pipeline)."""

    sequence: Vfcs
    spec: PropertySpec
    oracle: OracleSpec
    pipeline: str
    identity: ExecutionIdentity
    tools: ToolStatus

    def done(
        self,
        outcome: Outcome,
        reason: str,
        code: str = "",
        *,
        harness: HarnessArtifact | None = None,
        result: ExecutionResult | None = None,
        evaluation: PropertyEvaluation | None = None,
    ) -> StatefulObservation:
        executed = outcome in RUNNABLE
        declaration = self.spec.declaration.value
        if not self.oracle.executable and declaration == "property_under_test":
            declaration = PropertyDeclaration.NO_PROPERTY_AVAILABLE.value
        verdict = evaluation.verdict.value if evaluation else PropertyVerdict.NOT_EVALUATED.value
        reverted = -1
        if result is not None:
            reverted = result.reverted_index
        return StatefulObservation(
            sequence_id=self.sequence.sequence_id,
            outcome=outcome,
            reason=reason,
            pipeline=self.pipeline,
            executed=executed,
            return_code=result.return_code if result else None,
            reverted_index=reverted,
            property_under_test=self.sequence.property_under_test,
            property_checked=bool(evaluation and evaluation.evaluated),
            deployment=self.identity.deployment,
            tools=self.tools,
            stdout_tail=result.stdout_tail if result else "",
            property_id=self.spec.property_id,
            declaration=declaration,
            verdict=verdict,
            formulation=self.oracle.formulation,
            oracle_kind=self.oracle.kind.value,
            evaluation=evaluation,
            result=result,
            identity=self.identity,
            harness=harness,
            reason_code=code or outcome.value,
        )


def interpret(
    base: _Base,
    harness: HarnessArtifact,
    return_code: int,
    stdout: str,
    stderr: str,
    *,
    duration_ms: int = 0,
    common: Mapping[str, Any] | None = None,
) -> StatefulObservation:
    """Interpret one ``forge test --json`` run structurally. Pure; testable without forge."""
    common = dict(common or {"pipeline": harness.pipeline, "harness_hash": harness.harness_hash})
    count = len(base.sequence.calls)
    report = parse_forge_report(stdout)
    test = harness_test(report) if report is not None else None

    def failed(reason: str, code: str, test_status: str) -> StatefulObservation:
        return base.done(
            Outcome.EXECUTION_FAILED,
            reason,
            code,
            harness=harness,
            result=ExecutionResult(
                "completed",
                "ok",
                test_status,
                "incomplete",
                "not_run",
                return_code=return_code,
                failure_reason=reason[:300],
                stdout_tail=_tail(stdout),
                stderr_tail=_tail(stderr),
                duration_ms=duration_ms,
                **common,
            ),
        )

    if test is None:
        return failed("forge produced no structured harness result", "no_result", "missing")
    status = str(test.get("status", ""))
    events = decode_events(test)
    observation, deployed, done = observation_from_events(events, count)
    if status != "Success":
        reason = str(test.get("reason") or "the harness test failed")[:200]
        if not deployed:
            reason = f"the target could not be deployed locally: {reason}"
        return failed(reason, "harness_failed", "failed")
    if not deployed or not done:
        return failed("the harness did not report a complete run", "incomplete_run", "passed")
    outcome, evaluation, reason = classify(base.sequence, base.oracle, observation)
    reverted = next((i for i, ok in enumerate(observation.call_ok) if ok == 0), -1)
    property_status = {
        PropertyVerdict.PROPERTY_VIOLATED: "violated",
        PropertyVerdict.PROPERTY_HELD: "held",
        PropertyVerdict.MECHANISM_OBSERVED: "mechanism_observed",
        PropertyVerdict.FAILURE_PROPAGATED: "failure_propagated",
        PropertyVerdict.EXECUTION_ONLY_OBSERVATION: "no_oracle",
    }.get(evaluation.verdict, "not_evaluated")
    result = ExecutionResult(
        "completed",
        "ok",
        "passed",
        "reverted" if reverted >= 0 else "completed",
        property_status,
        return_code=return_code,
        reverted_index=reverted,
        call_results=observation.call_ok,
        events=events,
        stdout_tail=_tail(stdout, 600),
        stderr_tail=_tail(stderr, 600),
        duration_ms=duration_ms,
        **common,
    )
    code = outcome.value
    if outcome is Outcome.INCONCLUSIVE and not evaluation.precondition_ok:
        code = "precondition_failed"
    return base.done(outcome, reason, code, harness=harness, result=result, evaluation=evaluation)


# ---- independent checks -----------------------------------------------------------------------


@dataclass(frozen=True)
class CheckPath:
    path: str
    engine: str
    material: bool  # materially independent of the primary oracle
    outcome: str
    verdict: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "engine": self.engine,
            "material": self.material,
            "outcome": self.outcome,
            "verdict": self.verdict,
            "reason": self.reason[:200],
        }


@dataclass(frozen=True)
class Corroboration:
    """Which independent paths agree. Agreement is CORROBORATED_CANDIDATE, never VERIFIED."""

    sequence_id: str
    property_id: str
    status: str  # corroborated_candidate | single_path | disagreement | not_applicable
    paths: tuple[CheckPath, ...] = ()
    note: str = ""
    engine_runs: tuple[Mapping[str, Any], ...] = ()
    verified: bool = False
    # same_toolchain_alternate | second_engine | single_path | disagreement | not_applicable
    independence: str = "not_applicable"

    @property
    def agreeing(self) -> tuple[str, ...]:
        return tuple(
            p.path for p in self.paths if p.verdict == PropertyVerdict.PROPERTY_VIOLATED.value
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "property_id": self.property_id,
            "status": self.status,
            "paths": [p.as_dict() for p in self.paths],
            "agreeing": list(self.agreeing),
            "note": self.note,
            "engine_runs": [dict(item) for item in self.engine_runs],
            "independence": self.independence,
            "verified": False,
        }


ENGINE_PROPERTY_ADAPTERS: dict[str, str] = {
    # Engines that could judge the same property. An engine with an adapter runs only
    # when the registry found it usable (smoke-checked); otherwise, and for engines
    # without an adapter, the path is recorded as unavailable. Nothing is fabricated.
    "echidna": "engine-hosted harness with an independent in-EVM judge",
    "medusa": "engine-hosted harness with an independent in-EVM judge",
    "ityfuzz": "no property adapter for VFCS oracles",
    "symbolic": "no symbolic adapter for VFCS oracles",
}


def _path(obs: StatefulObservation, path: str, material: bool) -> CheckPath:
    return CheckPath(path, "foundry", material, obs.outcome.value, obs.verdict, obs.reason)


def independent_check(
    executor: StatefulExecutor,
    sequence: Vfcs,
    model: ResearchModel,
    sources: Mapping[str, str],
    spec: PropertySpec,
    primary: StatefulObservation,
    *,
    context: IdentityContext | None = None,
    engine_status: Mapping[str, str] | None = None,
    supporting_detectors: tuple[str, ...] = (),
    property_engines: Sequence[Any] = (),
) -> tuple[Corroboration, tuple[StatefulObservation, ...]]:
    """Re-judge a violated property along independent paths and record which agree.

    * ``foundry:pipeline:no_optimizer`` -- same oracle, different compiler pipeline.
      It catches pipeline-dependent behavior but is *not* materially independent.
    * ``foundry:alternate:<oracle>`` -- a different oracle formulation of the same
      property (for example a state read instead of call success). Material.
    * ``echidna:property`` / ``medusa:property`` -- the same harness plan hosted in a
      different EVM implementation with an independently implemented in-EVM judge.
      Material. They run only when passed in (the registry found them usable);
      otherwise they are recorded with their real status and never count.
    * ityfuzz / symbolic -- no adapter; recorded unavailable.
    * supporting static detectors (a *different* detector at the same site) are
      listed but never counted: the same detector twice is not two analyses.
    """
    if primary.outcome is not Outcome.PROPERTY_VIOLATED:
        return (
            Corroboration(sequence.sequence_id, spec.property_id, "not_applicable"),
            (),
        )
    runs: list[StatefulObservation] = []
    paths = [_path(primary, "foundry:primary", True)]
    differential = executor.execute(
        sequence, model, sources, spec=spec, context=context, pipeline="no_optimizer"
    )
    runs.append(differential)
    paths.append(_path(differential, "foundry:pipeline:no_optimizer", False))
    for alternate in spec.alternates:
        obs = executor.execute(
            sequence, model, sources, spec=spec, oracle=alternate, context=context
        )
        runs.append(obs)
        label = f"foundry:alternate:{alternate.kind.value}:{alternate.getter or '-'}"
        paths.append(_path(obs, label, True))
    engine_runs: list[Mapping[str, Any]] = []
    ran: set[str] = set()
    for engine in property_engines:
        if primary.harness is None or not primary.harness.buildable:
            break
        if executor.runs >= executor.max_runs:
            paths.append(
                CheckPath(
                    f"{engine.name}:property",
                    engine.name,
                    True,
                    Outcome.INCONCLUSIVE.value,
                    PropertyVerdict.NOT_EVALUATED.value,
                    "the local run budget is exhausted",
                )
            )
            ran.add(engine.name)
            continue
        executor.runs += 1
        run = engine.run(primary.harness, sequence, sources)
        ran.add(engine.name)
        engine_runs.append(run.as_dict())
        paths.append(
            CheckPath(
                f"{engine.name}:property",
                engine.name,
                True,
                run.outcome.value,
                run.verdict,
                run.reason,
            )
        )
    for engine_name, missing in sorted(ENGINE_PROPERTY_ADAPTERS.items()):
        if engine_name in ran:
            continue
        engine = engine_name
        status = (engine_status or {}).get(engine, "unavailable")
        paths.append(
            CheckPath(
                f"{engine}:property",
                engine,
                False,
                Outcome.UNAVAILABLE.value,
                PropertyVerdict.NOT_EVALUATED.value,
                f"{engine} {status}; {missing}",
            )
        )
    for detector in supporting_detectors:
        paths.append(
            CheckPath(
                f"static:{detector}",
                "bugforge-static",
                False,
                "supporting_static_candidate",
                PropertyVerdict.NOT_EVALUATED.value,
                "a different detector at the same site; static, not an execution",
            )
        )
    judged = [p for p in paths[1:] if p.verdict in {"property_violated", "property_held"}]
    held = [p for p in judged if p.verdict == "property_held"]
    material = [p for p in judged if p.material and p.verdict == "property_violated"]
    engines = {p.engine for p in material}
    if held:
        status = "disagreement"
        note = "an independent path held the property: the contradiction stays visible"
    elif material:
        status = "corroborated_candidate"
        if engines <= {"foundry"}:
            note = (
                "another Foundry formulation agrees; same toolchain, not a second engine; "
                "still a candidate, not verified"
            )
        else:
            note = (
                "a different engine agrees with the local property; still a candidate, not verified"
            )
    else:
        status = "single_path"
        note = "no materially independent path could judge this property"
    if status == "disagreement":
        independence = "disagreement"
    elif status != "corroborated_candidate":
        independence = "not_applicable" if status == "not_applicable" else "single_path"
    elif engines <= {"foundry"}:
        independence = "same_toolchain_alternate"
    else:
        independence = "second_engine"
    return (
        Corroboration(
            sequence.sequence_id,
            spec.property_id,
            status,
            tuple(paths),
            note,
            tuple(engine_runs),
            independence=independence,
        ),
        tuple(runs),
    )


# ---- feedback-directed closed loop ------------------------------------------------------------


@dataclass
class FeedbackLoopResult:
    observations: tuple[StatefulObservation, ...]
    mutated: tuple[Vfcs, ...]
    signals_emitted: tuple[str, ...]
    minimized: dict[str, MinimizationResult] = field(default_factory=dict)
    independent: dict[str, Corroboration] = field(default_factory=dict)
    independent_runs: dict[str, tuple[StatefulObservation, ...]] = field(default_factory=dict)
    specs: dict[str, PropertySpec] = field(default_factory=dict)
    sequences: dict[str, Vfcs] = field(default_factory=dict)
    feedback: tuple[dict[str, Any], ...] = ()

    @property
    def violations(self) -> tuple[StatefulObservation, ...]:
        return tuple(o for o in self.observations if o.outcome is Outcome.PROPERTY_VIOLATED)

    @property
    def mechanism_observations(self) -> tuple[StatefulObservation, ...]:
        """Completed behavioral probes. Evidence only; not security violations."""
        return tuple(
            o
            for o in self.observations
            if o.outcome in {Outcome.MECHANISM_OBSERVED, Outcome.FAILURE_PROPAGATED}
        )

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.observations:
            counts[item.outcome.value] = counts.get(item.outcome.value, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified": False,
            "observations": [o.as_dict() for o in self.observations],
            "outcomes": self.outcome_counts(),
            "mutated": [s.sequence_id for s in self.mutated],
            "signals_emitted": list(self.signals_emitted),
            "feedback": list(self.feedback),
            "minimized": {
                sid: {
                    "original": len(result.original),
                    "minimized": len(result.minimized),
                    "completed": result.completed,
                    "reason": result.reason,
                    "calls": [c.identity for c in result.minimized],
                    "preserved_property": result.preserved_property,
                }
                for sid, result in self.minimized.items()
            },
            "independent_checks": {sid: c.as_dict() for sid, c in self.independent.items()},
            "properties": {sid: spec.as_dict() for sid, spec in self.specs.items()},
        }


def parent_of(child: Vfcs) -> str:
    """The parent sequence id recorded in a mutation origin, or ""."""
    if child.origin.startswith("mutation:"):
        return child.origin.rsplit(":", 1)[-1]
    return ""


def run_feedback_loop(
    executor: StatefulExecutor,
    sequences: tuple[Vfcs, ...],
    models: Mapping[str, ResearchModel],
    sources: Mapping[str, Mapping[str, str]],
    *,
    specs: Mapping[str, PropertySpec] | None = None,
    context: IdentityContext | None = None,
    rounds: int = MAX_FEEDBACK_ROUNDS,
    limit: int = MAX_MUTATIONS,
    engine_status: Mapping[str, str] | None = None,
    supporting: Mapping[str, tuple[str, ...]] | None = None,
    property_engines: Sequence[Any] = (),
) -> FeedbackLoopResult:
    """execute -> observe -> feedback -> mutate -> re-execute -> minimize -> independent check.

    ``models`` and ``sources`` are keyed by sequence id: every sequence runs against
    the exact model and source set it was built from; a mutated child inherits its
    parent's. Idempotent per sequence id within a call.
    """
    models = dict(models)
    sources = dict(sources)
    spec_map: dict[str, PropertySpec] = dict(specs or {})
    observations: list[StatefulObservation] = []
    emitted: list[str] = []
    feedback: list[dict[str, Any]] = []
    minimized: dict[str, MinimizationResult] = {}
    independent: dict[str, Corroboration] = {}
    independent_runs: dict[str, tuple[StatefulObservation, ...]] = {}
    seen: set[str] = set()
    frontier = list(sequences)
    all_mutated: list[Vfcs] = []
    by_id = {s.sequence_id: s for s in sequences}

    for _round in range(max(1, rounds)):
        signals: list[FeedbackSignal] = []
        next_frontier: list[Vfcs] = []
        for sequence in frontier:
            if sequence.sequence_id in seen:
                continue
            seen.add(sequence.sequence_id)
            model = models.get(sequence.sequence_id)
            exact = sources.get(sequence.sequence_id)
            if model is None or exact is None:
                continue
            spec = spec_map.get(sequence.sequence_id) or build_property(sequence, model)
            spec_map[sequence.sequence_id] = spec
            obs = executor.execute(sequence, model, exact, spec=spec, context=context)
            observations.append(obs)
            signal = obs.signal()
            if signal is not None:
                signals.append(signal)
                key = f"{signal.engine}:{signal.kind}:{signal.sequence_id}:{signal.call_index}"
                emitted.append(key)
                feedback.append(
                    {
                        "key": key,
                        "engine": signal.engine,
                        "kind": signal.kind,
                        "sequence_id": signal.sequence_id,
                        "call_index": signal.call_index,
                        "from_outcome": obs.outcome.value,
                    }
                )
            if obs.outcome is Outcome.PROPERTY_VIOLATED:
                result = _minimize_sequence(executor, sequence, model, exact, spec, context)
                if result is not None:
                    minimized[sequence.sequence_id] = result
                corroboration, runs = independent_check(
                    executor,
                    sequence,
                    model,
                    exact,
                    spec,
                    obs,
                    context=context,
                    engine_status=engine_status,
                    supporting_detectors=(supporting or {}).get(sequence.derived_from, ()),
                    property_engines=property_engines,
                )
                independent[sequence.sequence_id] = corroboration
                independent_runs[sequence.sequence_id] = runs
        if not signals:
            break
        children = mutate(list(by_id.values()), signals, limit=limit)
        for child in children:
            if child.sequence_id in by_id:
                continue
            parent = parent_of(child)
            if parent not in models or parent not in sources:
                continue
            by_id[child.sequence_id] = child
            models[child.sequence_id] = models[parent]
            sources[child.sequence_id] = sources[parent]
            all_mutated.append(child)
            next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break

    return FeedbackLoopResult(
        observations=tuple(observations),
        mutated=tuple(all_mutated),
        signals_emitted=tuple(emitted),
        minimized=minimized,
        independent=independent,
        independent_runs=independent_runs,
        specs=spec_map,
        sequences=by_id,
        feedback=tuple(feedback),
    )


def _minimize_sequence(
    executor: StatefulExecutor,
    sequence: Vfcs,
    model: ResearchModel,
    sources: Mapping[str, str],
    spec: PropertySpec,
    context: IdentityContext | None,
) -> MinimizationResult | None:
    def evaluator(calls: tuple[VfcsCall, ...]) -> bool | None:
        trial = replace(sequence, calls=calls)
        obs = executor.execute(trial, model, sources, spec=spec, context=context)
        if obs.outcome is Outcome.PROPERTY_VIOLATED:
            return True
        if obs.outcome in {Outcome.PROPERTY_HELD, Outcome.SEQUENCE_EXECUTED_NO_ORACLE}:
            return False
        return None  # reverted / inconclusive / failed: cannot tell

    return minimize(sequence, evaluator, evaluator_name="foundry_local_oracle", max_attempts=12)


# ---- durable reproduction bundles -------------------------------------------------------------

BUNDLE_SCHEMA = "bugforge.repro_bundle/1"
MAX_BUNDLE_BYTES = 96_000


def _calls_dict(calls: tuple[VfcsCall, ...]) -> list[dict[str, Any]]:
    return [
        {
            "contract": c.contract,
            "function": c.function,
            "role": c.role,
            "actor": c.actor,
            "arguments": [list(item) for item in c.arguments],
            "primitive": c.primitive,
            "established_by": c.established_by,
        }
        for c in calls
    ]


def build_bundle(
    observation: StatefulObservation,
    sequence: Vfcs,
    spec: PropertySpec,
    sources: Mapping[str, str],
    *,
    minimized: MinimizationResult | None = None,
    corroboration: Corroboration | None = None,
    independent_runs: tuple[StatefulObservation, ...] = (),
    created_at: str = "",
) -> dict[str, Any]:
    """A self-contained, hash-addressed reproduction bundle (no secrets, bounded).

    Sources are referenced by sha256 and stored once in the campaign's artifact
    storage (``source_blobs``); every other artifact is inline and hashed. The
    bundle id is the hash of its content (timestamps excluded), so persisting the
    same run twice is idempotent.
    """
    harness = observation.harness
    result = observation.result
    config = foundry_config(observation.pipeline, "<host solc>")
    artifacts: dict[str, str] = {
        "test/VfcsHarness.t.sol": harness.source if harness else "",
        "foundry.toml": config,
        "lib/forge-std/src/Test.sol": _FORGE_STD,
    }
    artifacts = {name: text for name, text in artifacts.items() if text}
    source_hashes = {path: sha256_text(text) for path, text in sorted(sources.items())}
    core: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "sequence": {
            "sequence_id": sequence.sequence_id,
            "template": sequence.template,
            "origin": sequence.origin,
            "derived_from": sequence.derived_from,
            "property_under_test": sequence.property_under_test,
            "calls": _calls_dict(sequence.calls),
            "call_instances": list(instance_identities(sequence)),
            "identity": {
                "campaign_id": sequence.identity.campaign_id,
                "source_snapshot": sequence.identity.source_snapshot,
                "compiler_configuration": sequence.identity.compiler_configuration,
                "fork_reference": sequence.identity.fork_reference,
                "program_context": sequence.identity.program_context,
                "deployment": sequence.identity.deployment,
                "source_hashes": dict(sequence.identity.source_hashes),
                "implementation": sequence.identity.implementation,
            },
        },
        "minimized": (
            {
                "calls": _calls_dict(minimized.minimized),
                "completed": minimized.completed,
                "reason": minimized.reason,
                "attempts": minimized.attempts,
                "evaluator": minimized.evaluator,
            }
            if minimized is not None
            else None
        ),
        "property": spec.as_dict(),
        "oracle": observation.harness.oracle.as_dict() if observation.harness else {},
        "identity": observation.identity.as_dict() if observation.identity else {},
        "harness": harness.as_dict() if harness else {},
        "execution": observation.as_dict() | {"result": result.as_dict() if result else None},
        "independent": corroboration.as_dict() if corroboration else None,
        "independent_runs": [
            {
                "pipeline": run.pipeline,
                "formulation": run.formulation,
                "oracle_kind": run.oracle_kind,
                "outcome": run.outcome.value,
                "verdict": run.verdict,
                "harness_hash": run.harness.harness_hash if run.harness else "",
            }
            for run in independent_runs
        ],
        "command": list(BUILD_ARGV) + ["&&"] + list(TEST_ARGV),
        "environment": {
            "forge_version": observation.tools.forge_version,
            "solc_version": observation.tools.solc_version,
            "offline": True,
            "ffi": False,
            "rpc": "none",
            "replay_mode": ReplayMode.LOCAL_SOURCE_REPLAY.value,
        },
        "source_hashes": source_hashes,
        "artifact_hashes": {name: sha256_text(text) for name, text in artifacts.items()},
        "artifacts": artifacts,
        "note": (
            "deterministic local reproduction of an analyzed source set; offline; ffi "
            "disabled; not a deployment replay; not verified"
        ),
        "verified": False,
    }
    blob = json.dumps(core, sort_keys=True, default=str)
    if len(blob) > MAX_BUNDLE_BYTES:
        execution = dict(core["execution"])
        if execution.get("result"):
            execution["result"] = {
                **execution["result"],
                "stdout_tail": "",
                "stderr_tail": "",
                "events": [],
            }
        core["execution"] = execution
        core["truncated"] = True
    bundle_id = "rb_" + sha256_text(json.dumps(core, sort_keys=True, default=str))[:24]
    return {**core, "bundle_id": bundle_id, "created_at": created_at}


class BundleIntegrityError(ValueError):
    """A stored bundle or source blob does not match its recorded hash."""


def materialize_bundle(
    bundle: Mapping[str, Any], blobs: Mapping[str, str], dest: Path
) -> dict[str, str]:
    """Write a bundle back into a Foundry project, verifying every hash first."""
    sources: dict[str, str] = {}
    for path, digest_value in dict(bundle.get("source_hashes", {})).items():
        text = blobs.get(str(digest_value))
        if text is None or sha256_text(text) != digest_value:
            raise BundleIntegrityError(f"source {path} is missing or does not match its hash")
        sources[str(path)] = text
    artifacts = dict(bundle.get("artifacts", {}))
    for name, digest_value in dict(bundle.get("artifact_hashes", {})).items():
        if sha256_text(str(artifacts.get(name, ""))) != digest_value:
            raise BundleIntegrityError(f"artifact {name} does not match its hash")
    harness = str(artifacts.get("test/VfcsHarness.t.sol", ""))
    if not harness:
        raise BundleIntegrityError("the bundle has no harness")
    pipeline = str(dict(bundle.get("execution", {})).get("pipeline", "default"))
    scaffold(dest, sources, harness, foundry_config(pipeline, tool_path("solc") or ""))
    return sources


def replay_bundle(
    bundle: Mapping[str, Any],
    blobs: Mapping[str, str],
    *,
    timeout: int | None = None,
    mode: str = ReplayMode.LOCAL_SOURCE_REPLAY.value,
) -> dict[str, Any]:
    """Re-run a stored bundle locally and compare the observation with the stored one.

    Only ``local_source_replay`` runs here; any other mode is refused with its status
    instead of silently being replaced by a local replay.
    """
    if mode != ReplayMode.LOCAL_SOURCE_REPLAY.value:
        info = replay_modes().get(mode)
        if info is None:
            return {"status": "refused", "reason": f"unknown replay mode {mode!r}"}
        return {"status": info["status"], "mode": mode, "reason": info.get("reason", "")}
    tools = tool_status()
    if not (execution_enabled() and tools.available):
        return {"status": "unavailable", "reason": "forge/solc unavailable or disabled"}
    artifacts = dict(bundle.get("artifacts", {}))
    stored = {
        str(path): blobs.get(str(value), "")
        for path, value in dict(bundle.get("source_hashes", {})).items()
    }
    cheats = untrusted_cheatcode_use(
        {**stored, "test/VfcsHarness.t.sol": str(artifacts.get("test/VfcsHarness.t.sol", ""))}
    )
    if cheats:
        return {
            "status": "blocked_by_policy",
            "reason": f"cheatcode references in the bundle: {'; '.join(cheats)}",
        }
    limit = timeout if timeout is not None else _timeout_setting()
    forge = tool_path("forge") or "forge"
    with tempfile.TemporaryDirectory(prefix="bugforge-replay-") as tmp:
        root = Path(tmp)
        materialize_bundle(bundle, blobs, root)
        env = _safe_env(tmp)
        try:
            proc = subprocess.run(
                [forge, *TEST_ARGV[1:]],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=limit,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "execution_failed", "reason": str(exc)[:160]}
    report = parse_forge_report(proc.stdout)
    test = harness_test(report) if report is not None else None
    if test is None:
        return {"status": "execution_failed", "reason": "no structured result"}
    events = decode_events(test)
    stored = dict(dict(bundle.get("execution", {})).get("result") or {})
    stored_events: list[tuple[object, ...]] = [tuple(item) for item in stored.get("events", [])]
    calls = len(dict(bundle.get("sequence", {})).get("calls", []))
    observation, _deployed, _done = observation_from_events(events, calls)
    return {
        "status": "replayed",
        "events_match": bool(stored_events) and stored_events == [tuple(e) for e in events],
        "call_results": list(observation.call_ok),
        "stored_call_results": stored.get("call_results", []),
        "verified": False,
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "BundleIntegrityError",
    "CheckPath",
    "Corroboration",
    "ExecutionIdentity",
    "ExecutionResult",
    "FeedbackLoopResult",
    "HarnessArtifact",
    "IdentityContext",
    "Outcome",
    "Primitive",
    "ReplayMode",
    "StatefulExecutor",
    "StatefulObservation",
    "ToolStatus",
    "abi_type",
    "bind_identity",
    "build_bundle",
    "build_harness",
    "canonical_signature",
    "classify",
    "decode_events",
    "execution_enabled",
    "independent_check",
    "interpret",
    "materialize_bundle",
    "replay_bundle",
    "run_feedback_loop",
    "target_contract",
    "tool_status",
]

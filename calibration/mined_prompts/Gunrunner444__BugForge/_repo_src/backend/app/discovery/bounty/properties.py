"""Property / oracle layer for executable VFCS (Phase 52 hardening, Part 2).

A sequence that *runs* is not a vulnerability. Every executable VFCS therefore
declares exactly one of:

* ``property_under_test``   -- a property with an executable oracle;
* ``no_property_available`` -- nothing meaningful can be judged from execution;
* ``property_unsupported``  -- a property exists but the bounded local harness
  cannot express its oracle (it needs a signer, a market, an EntryPoint, ...).

After execution a security property is ``property_violated``, ``property_held``
or an ``execution_only_observation``. A behavioral probe can instead record
``mechanism_observed`` or ``failure_propagated``; neither is a security verdict.
A sequence that executed without an oracle is ``SEQUENCE_EXECUTED_NO_ORACLE``
and is never a strong vulnerability candidate.

Every property is built from the static candidate *and* code evidence (the
function body, the state variable it writes, its public getter) and carries
provenance: file, line range, contract, functions, variables, the candidate it
came from, why it was chosen, its assumptions, and the oracle expression. A
property is a hypothesis until executed evidence exists; nothing here verifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.discovery.bounty.vfcs import ATTACKER, Vfcs
from app.discovery.orchestration.codec import digest
from app.parsing.solidity_research import ResearchModel, RFunction, SemanticCandidate


class PropertyFamily(StrEnum):
    AUTHORIZATION = "authorization"
    INITIALIZATION = "initialization"
    ACCOUNTING = "accounting"
    SOLVENCY = "solvency"
    TOKEN_MOVEMENT = "token_movement"
    ORACLE_SAFETY = "oracle_safety"
    NONCE_REPLAY = "nonce_replay"
    UPGRADE_SAFETY = "upgrade_safety"
    GOVERNANCE = "governance"
    BRIDGE_MESSAGE_BINDING = "bridge_message_binding"
    ACCESS_CONTROL_HIERARCHY = "access_control_hierarchy"
    MECHANISM = "mechanism"
    EIP7702 = "eip7702"
    ERC4337 = "erc4337"
    TRANSIENT_STORAGE = "transient_storage"


class PropertyDeclaration(StrEnum):
    PROPERTY_UNDER_TEST = "property_under_test"
    NO_PROPERTY_AVAILABLE = "no_property_available"
    PROPERTY_UNSUPPORTED = "property_unsupported"
    # An executable behavioral probe. Not a security invariant.
    MECHANISM_OBSERVATION = "mechanism_observation"


class PropertyVerdict(StrEnum):
    PROPERTY_VIOLATED = "property_violated"
    PROPERTY_HELD = "property_held"
    # Behavioral probe results. They are not security-property verdicts.
    MECHANISM_OBSERVED = "mechanism_observed"
    FAILURE_PROPAGATED = "failure_propagated"
    EXECUTION_ONLY_OBSERVATION = "execution_only_observation"
    NOT_EVALUATED = "not_evaluated"


class OracleKind(StrEnum):
    """How the harness judges the property. NONE means execution-only."""

    NONE = "none"
    # the probe call (by an actor who must not be able to make it) must revert
    CALL_MUST_FAIL = "call_must_fail"
    # a second initialization / replay of an authorization must revert
    SECOND_CALL_MUST_FAIL = "second_call_must_fail"
    # a privileged public getter must not change across the probe
    STATE_UNCHANGED = "state_unchanged"
    # the attacker's fixture-token balance must not increase over the sequence
    BALANCE_NOT_INCREASED = "balance_not_increased"
    # credited balance (public getter for the actor) must not exceed tokens received
    CREDIT_LE_RECEIVED = "credit_le_received"
    # solvency formulation: the actor's credit must not exceed what the contract holds
    CREDIT_LE_HOLDINGS = "credit_le_holdings"
    # Behavioral probe: did a reverting callee's failure reach the outer call?
    # Outer success is evidence of a mechanism, not a security-property violation.
    MECHANISM_PROBE = "mechanism_probe"


EXECUTABLE_ORACLES = frozenset(OracleKind) - {OracleKind.NONE}


@dataclass(frozen=True)
class OracleSpec:
    kind: OracleKind = OracleKind.NONE
    # The probe and precondition calls are bound by (role, function), not index,
    # so mutation/minimization cannot silently move the oracle onto another call.
    probe_role: str = ""
    probe_function: str = ""
    precondition_roles: tuple[str, ...] = ()
    getter: str = ""  # e.g. "owner()" or "balances(address)"
    getter_actor: str = ""  # actor passed to a mapping getter
    token_fixture: str = ""  # standard | fee_on_transfer | false_return
    snapshot: str = "before_probe"  # before_probe | before_sequence
    expression: str = ""
    formulation: str = "primary"  # primary | alternate

    @property
    def executable(self) -> bool:
        return self.kind in EXECUTABLE_ORACLES

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "probe_role": self.probe_role,
            "probe_function": self.probe_function,
            "precondition_roles": list(self.precondition_roles),
            "getter": self.getter,
            "getter_actor": self.getter_actor,
            "token_fixture": self.token_fixture,
            "snapshot": self.snapshot,
            "expression": self.expression,
            "formulation": self.formulation,
            "executable": self.executable,
        }


@dataclass(frozen=True)
class PropertyProvenance:
    file: str
    line_start: int
    line_end: int
    contract: str
    functions: tuple[str, ...]
    variables: tuple[str, ...]
    candidate: str
    reason: str
    assumptions: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "contract": self.contract,
            "functions": list(self.functions),
            "variables": list(self.variables),
            "candidate": self.candidate,
            "reason": self.reason,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class PropertySpec:
    property_id: str
    family: PropertyFamily
    statement: str
    declaration: PropertyDeclaration
    oracle: OracleSpec
    provenance: PropertyProvenance
    unsupported_reason: str = ""
    alternates: tuple[OracleSpec, ...] = ()
    status: str = "hypothesis"  # a hypothesis until executed evidence exists
    verified: bool = False

    @property
    def executable(self) -> bool:
        return (
            self.declaration
            in {
                PropertyDeclaration.PROPERTY_UNDER_TEST,
                PropertyDeclaration.MECHANISM_OBSERVATION,
            }
            and self.oracle.executable
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "family": self.family.value,
            "statement": self.statement,
            "declaration": self.declaration.value,
            "oracle": self.oracle.as_dict(),
            "alternates": [item.as_dict() for item in self.alternates],
            "provenance": self.provenance.as_dict(),
            "unsupported_reason": self.unsupported_reason,
            "status": self.status,
            "executable": self.executable,
            "verified": False,
        }


# ---- detector -> family ---------------------------------------------------------------------

_FAMILY_BY_PREFIX: tuple[tuple[str, PropertyFamily], ...] = (
    ("aa.unprotected_account_initializer", PropertyFamily.INITIALIZATION),
    ("aa.", PropertyFamily.ERC4337),
    ("caller_context.swallowed_call_failure", PropertyFamily.MECHANISM),
    ("caller_context.unrestricted_dispatch_with_approval_authority", PropertyFamily.TOKEN_MOVEMENT),
    ("caller_context.trusted_intermediary_actor_parameter", PropertyFamily.AUTHORIZATION),
    ("caller_context.", PropertyFamily.ACCESS_CONTROL_HIERARCHY),
    ("accounting.donation_share_price", PropertyFamily.SOLVENCY),
    ("accounting.fee_on_transfer_mismatch", PropertyFamily.TOKEN_MOVEMENT),
    ("accounting.unchecked_token_return", PropertyFamily.TOKEN_MOVEMENT),
    ("accounting.", PropertyFamily.ACCOUNTING),
    ("message.replay_no_consumption", PropertyFamily.NONCE_REPLAY),
    ("message.nonce_not_in_digest", PropertyFamily.NONCE_REPLAY),
    ("message.missing_expiry", PropertyFamily.NONCE_REPLAY),
    ("message.", PropertyFamily.BRIDGE_MESSAGE_BINDING),
    ("oracle.", PropertyFamily.ORACLE_SAFETY),
    ("arithmetic.", PropertyFamily.ACCOUNTING),
    ("eip7702.", PropertyFamily.EIP7702),
    ("transient_storage.", PropertyFamily.TRANSIENT_STORAGE),
    ("read_only_reentrancy.", PropertyFamily.ACCOUNTING),
    ("proxy.", PropertyFamily.UPGRADE_SAFETY),
    ("upgrade.", PropertyFamily.UPGRADE_SAFETY),
    ("governance.", PropertyFamily.GOVERNANCE),
    ("bridge.", PropertyFamily.BRIDGE_MESSAGE_BINDING),
    ("v4_hooks.", PropertyFamily.ACCESS_CONTROL_HIERARCHY),
)


def family_for(detector: str) -> PropertyFamily:
    for prefix, family in _FAMILY_BY_PREFIX:
        if detector.startswith(prefix):
            return family
    return PropertyFamily.AUTHORIZATION


_OWNER_WRITE = re.compile(r"\b(owner\w*|admin\w*|_owner\w*)\s*=(?!=)")
_STATE_WRITE = r"\b{name}\b\s*(?:\[[^\]]*\]\s*)*(?:=(?!=)|\+=|-=)"
_CREDIT_NAMES = ("balances", "balanceOf", "deposits", "shares", "credit", "credited")


def _resolve(model: ResearchModel, contract: str, signature: str) -> RFunction | None:
    for function in model.functions_of(contract):
        if function.signature == signature and function.has_body:
            return function
    return None


def public_getter(model: ResearchModel, contract: str, variable: str) -> str:
    """The getter signature of a public state variable, or "" when it has none."""
    for owner in model.lineage(contract):
        item = model.contracts.get(owner)
        if item is None:
            continue
        kind = dict(item.state_vars).get(variable)
        if kind is None:
            continue
        if not re.search(rf"\bpublic\b[^;{{}}]*\b{re.escape(variable)}\s*[;=]", item.body):
            return ""
        mapping = re.fullmatch(r"mapping\s*\(\s*(\w+)\s*=>\s*(\w+)\s*\)", kind.replace(" ", " "))
        if kind.startswith("mapping"):
            if mapping is None or mapping.group(1) != "address":
                return ""
            if not re.fullmatch(r"u?int\d*|bool|address", mapping.group(2)):
                return ""
            return f"{variable}(address)"
        if re.fullmatch(r"u?int\d*|bool|address|bytes\d+", kind) or model.contract_for_type(kind):
            return f"{variable}()"
        return ""
    return ""


def _written_public_state(
    model: ResearchModel, contract: str, function: RFunction
) -> tuple[str, str]:
    """A public, scalar state variable this function writes, with its getter."""
    for variable in sorted(model.state_vars(contract)):
        if re.search(_STATE_WRITE.format(name=re.escape(variable)), function.body):
            getter = public_getter(model, contract, variable)
            if getter.endswith("()"):
                return variable, getter
    return "", ""


def _credit_getter(model: ResearchModel, contract: str, function: RFunction) -> tuple[str, str]:
    """The per-account credit mapping a deposit-like function increments for msg.sender."""
    for variable in sorted(model.state_vars(contract)):
        if not re.search(rf"\b{re.escape(variable)}\s*\[\s*msg\.sender\s*\]\s*\+=", function.body):
            continue
        getter = public_getter(model, contract, variable)
        if getter == f"{variable}(address)":
            return variable, getter
    for name in _CREDIT_NAMES:
        getter = public_getter(model, contract, name)
        if getter == f"{name}(address)":
            return name, getter
    return "", ""


def _span(function: RFunction) -> tuple[int, int]:
    return function.line, function.line + max(0, function.body.count("\n"))


def _property_id(sequence_key: str, family: PropertyFamily, oracle: OracleSpec) -> str:
    return "pr_" + digest(
        (sequence_key, family.value, oracle.kind.value, oracle.getter, oracle.probe_role)
    )


def _role_function(sequence: Vfcs, role: str) -> str:
    for call in sequence.calls:
        if call.role == role and not call.primitive:
            return call.function
    return ""


def build_property(
    sequence: Vfcs,
    model: ResearchModel,
    candidate: SemanticCandidate | None = None,
) -> PropertySpec:
    """The property a sequence tests, built from the candidate plus code evidence."""
    detector, _, target = sequence.derived_from.partition("@")
    contract, _, signature = target.partition(".")
    family = family_for(detector)
    function = _resolve(model, contract, signature)
    file = function.file if function else (candidate.file if candidate else "")
    start, end = _span(function) if function else ((candidate.line if candidate else 0),) * 2
    functions = tuple(dict.fromkeys(call.function for call in sequence.calls if not call.primitive))
    builder = _ORACLE_BUILDERS.get(sequence.template)
    if function is None:
        oracle, alternates, variables, reason, assumptions, unsupported = (
            OracleSpec(),
            (),
            (),
            "the candidate function is not in the model",
            (),
            "candidate function unresolved",
        )
    elif builder is None:
        oracle, alternates, variables, reason, assumptions, unsupported = (
            OracleSpec(),
            (),
            (),
            f"no oracle is defined for template {sequence.template!r}",
            (),
            "",
        )
    else:
        oracle, alternates, variables, reason, assumptions, unsupported = builder(
            sequence, model, function, detector
        )
    if oracle.kind is OracleKind.MECHANISM_PROBE and oracle.executable:
        declaration = PropertyDeclaration.MECHANISM_OBSERVATION
    elif oracle.executable:
        declaration = PropertyDeclaration.PROPERTY_UNDER_TEST
    elif unsupported:
        declaration = PropertyDeclaration.PROPERTY_UNSUPPORTED
    else:
        declaration = PropertyDeclaration.NO_PROPERTY_AVAILABLE
    provenance = PropertyProvenance(
        file=file,
        line_start=start,
        line_end=end,
        contract=contract,
        functions=functions,
        variables=tuple(variables),
        candidate=sequence.derived_from,
        reason=reason,
        assumptions=tuple(assumptions),
    )
    return PropertySpec(
        property_id=_property_id(sequence.derived_from, family, oracle),
        family=family,
        statement=sequence.property_under_test,
        declaration=declaration,
        oracle=oracle,
        provenance=provenance,
        unsupported_reason=unsupported,
        alternates=tuple(alternates),
    )


OracleBuilt = tuple[OracleSpec, tuple[OracleSpec, ...], tuple[str, ...], str, tuple[str, ...], str]


def _initializer(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
    primary = OracleSpec(
        kind=OracleKind.SECOND_CALL_MUST_FAIL,
        probe_role="reinitialize",
        probe_function=fn.signature,
        precondition_roles=("initialize",),
        expression=f"after a successful {fn.name}(), a second {fn.name}() must revert",
    )
    variables: tuple[str, ...] = ()
    alternates: list[OracleSpec] = []
    match = _OWNER_WRITE.search(fn.body)
    if match:
        variable = match.group(1)
        getter = public_getter(model, fn.contract, variable)
        if getter.endswith("()"):
            variables = (variable,)
            alternates.append(
                OracleSpec(
                    kind=OracleKind.STATE_UNCHANGED,
                    probe_role="reinitialize",
                    probe_function=fn.signature,
                    precondition_roles=("initialize",),
                    getter=getter,
                    expression=f"{getter} set by the first {fn.name}() must not change",
                    formulation="alternate",
                )
            )
    return (
        primary,
        tuple(alternates),
        variables,
        f"{fn.name} assigns the owner without a one-time guard (code evidence: owner write)",
        ("the first initialization is the legitimate one",),
        "",
    )


def _unauthenticated(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
    if det not in {
        "aa.unauthenticated_account_execution",
        "v4_hooks.callback_without_pool_manager_check",
        "bridge.receiver_without_endpoint_check",
    }:
        return (
            OracleSpec(),
            (),
            (),
            "validateUserOp takes a PackedUserOperation and needs an EntryPoint/userOp fixture",
            (),
            "ERC-4337 validation needs an EntryPoint and a signed user operation fixture",
        )
    return (
        OracleSpec(
            kind=OracleKind.CALL_MUST_FAIL,
            probe_role="execute",
            probe_function=fn.signature,
            expression=f"{fn.name}() called by an arbitrary attacker must revert",
        ),
        (),
        (),
        f"{fn.name} acts on caller-supplied input and has no caller authentication",
        (
            "the attacker is not the trusted caller (account owner/EntryPoint, PoolManager, "
            "or bridge endpoint)",
            "a revert for these fixture inputs holds the property only for these inputs",
        ),
        "",
    )


def _nested_dispatch(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
    nested = ""
    for call in sequence.calls:
        for _name, value in call.arguments:
            if value.startswith("nested_call:"):
                nested = value.split(":", 1)[1]
    callee_contract, _, callee_sig = nested.partition(".")
    callee = _resolve(model, callee_contract, callee_sig) if nested else None
    if callee is None:
        return (OracleSpec(), (), (), "the nested callee is not resolvable", (), "")
    variable, getter = _written_public_state(model, callee_contract, callee)
    if not getter:
        return (
            OracleSpec(),
            (),
            (),
            f"{callee.name} writes no public scalar state an oracle could observe",
            (),
            "no observable privileged state",
        )
    primary = OracleSpec(
        kind=OracleKind.STATE_UNCHANGED,
        probe_role="dispatch",
        probe_function=fn.signature,
        getter=getter,
        snapshot="before_probe",
        expression=(
            f"an attacker dispatching through {fn.name}() must not change {getter} "
            f"(written only by {callee.name})"
        ),
    )
    return (
        primary,
        (),
        (variable,),
        f"{fn.name} re-enters {callee.name}, which writes {variable} behind a caller check",
        ("the nested call is encoded by the attacker",),
        "",
    )


def _unsupported(reason: str, needs: str) -> Any:
    def build(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
        return (OracleSpec(), (), (), reason.format(name=fn.name), (), needs)

    return build


def _no_property(reason: str) -> Any:
    def build(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
        return (OracleSpec(), (), (), reason.format(name=fn.name), (), "")

    return build


def _credit_oracle(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
    """Credited balance must not exceed what the contract actually received.

    Executable only when the contract exposes a public per-account credit mapping
    and the credit function takes the token as a parameter or the harness can
    attach a fixture token; otherwise the property is unsupported (Slice B adds
    the token fixture primitive that makes it executable).
    """
    variable, getter = _credit_getter(model, fn.contract, fn)
    if not getter:
        return (
            OracleSpec(),
            (),
            (),
            f"{fn.name} credits no public per-account mapping an oracle could read",
            (),
            "no observable credited balance",
        )
    fixture = "fee_on_transfer" if det == "accounting.fee_on_transfer_mismatch" else "standard"
    if det == "accounting.unchecked_token_return":
        fixture = "false_return"
    primary = OracleSpec(
        kind=OracleKind.CREDIT_LE_RECEIVED,
        probe_role=_first_role(sequence),
        probe_function=fn.signature,
        getter=getter,
        getter_actor=ATTACKER,
        token_fixture=fixture,
        snapshot="before_probe",
        expression=(
            f"{getter} credited to the attacker must not exceed the tokens {fn.contract} "
            f"received ({fixture} fixture token)"
        ),
    )
    alternate = OracleSpec(
        kind=OracleKind.CREDIT_LE_HOLDINGS,
        probe_role=primary.probe_role,
        probe_function=fn.signature,
        getter=getter,
        getter_actor=ATTACKER,
        token_fixture=fixture,
        snapshot="before_probe",
        expression=(
            f"after the probe, {getter} for the attacker must not exceed the fixture tokens "
            f"{fn.contract} holds (solvency formulation)"
        ),
        formulation="alternate",
    )
    return (
        primary,
        (alternate,),
        (variable,),
        f"{fn.name} credits {variable}[msg.sender] from a caller-supplied amount",
        ("the fixture token is the asset the contract accounts for",),
        "",
    )


def _swallowed(sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str) -> OracleBuilt:
    """Observe whether a reverting callee's failure is swallowed. Not a security invariant."""
    bound = any(call.primitive and call.function == "deploy_reverting()" for call in sequence.calls)
    probe = next((call for call in sequence.calls if call.role == "probe"), None)
    if not bound or probe is None:
        return (
            OracleSpec(),
            (),
            (),
            "no failing callee is bound to a caller-supplied address parameter",
            ("upstream reliance and impact are unknown",),
            "the call receiver is not a parameter the harness can replace",
        )
    return (
        OracleSpec(
            kind=OracleKind.MECHANISM_PROBE,
            probe_role="probe",
            probe_function=probe.function,
            precondition_roles=("failing_callee",),
            expression=(
                f"behavioral probe of {fn.name}(): whether a reverting callee's failure "
                "is swallowed. Outer success is not a security-property violation"
            ),
        ),
        (),
        (),
        f"{fn.name} stores a low-level call success flag and a failure path returns normally",
        (
            "the local callee reverts on every call",
            "outer success is a mechanism observation, not a bounty-eligible finding",
            "application-level impact and intentional design remain unproven",
        ),
        "",
    )


def _first_role(sequence: Vfcs) -> str:
    for call in sequence.calls:
        if not call.primitive:
            return call.role
    return ""


def _body_compact(body: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", body, flags=re.S)
    text = re.sub(r"//.*?$", " ", text, flags=re.M)
    return re.sub(r"\s+", "", text)


def _unauthorized_pause(
    sequence: Vfcs, model: ResearchModel, fn: RFunction, det: str
) -> OracleBuilt:
    """Unauthorized pause must revert and must not change the supply-pause flag.

    The check is read from the function body. A renamed template cannot claim
    this property when the source does not contain the authorization condition.
    """
    del sequence, det
    compact = _body_compact(fn.body)
    authorized = (
        "msg.sender!=governor&&msg.sender!=pauseGuardian" in compact
        or "msg.sender==governor||msg.sender==pauseGuardian" in compact
    )
    getter = next(
        (
            item
            for item in model.functions_of(fn.contract)
            if item.signature == "isSupplyPaused()" and item.has_body
        ),
        None,
    )
    if not authorized or getter is None:
        return (
            OracleSpec(),
            (),
            (),
            f"{fn.name} does not check governor and pauseGuardian against isSupplyPaused()",
            (),
            "the authorization check or isSupplyPaused() is not in this source",
        )
    return (
        OracleSpec(
            kind=OracleKind.CALL_MUST_FAIL,
            probe_role="execute",
            probe_function=fn.signature,
            getter="isSupplyPaused()",
            expression=(
                "an account that is neither governor nor pauseGuardian must not "
                "change the supply pause flag"
            ),
        ),
        (),
        ("pauseFlags",),
        f"{fn.name} reverts unless msg.sender is governor or pauseGuardian",
        (
            "the attacker is neither the configured governor nor the pause guardian",
            "a revert with an unchanged pause flag holds only for this local configuration",
        ),
        "",
    )


_ORACLE_BUILDERS: dict[str, Any] = {
    "initialize→reinitialize": _initializer,
    "unauthenticated-execution": _unauthenticated,
    "AA validation→execution": _unauthenticated,
    "nested-dispatch": _nested_dispatch,
    "forwarded-actor": _unsupported(
        "{name} acts for a caller-named account; judging it needs that account funded",
        "needs a funded victim position in the callee (funding fixture)",
    ),
    "approve→transferFrom": _unsupported(
        "{name} forwards caller-chosen calldata; judging it needs a victim allowance "
        "and an encoded transferFrom",
        "needs a fixture token, a victim approval, and nested transferFrom calldata",
    ),
    "deposit→donate→withdraw": _unsupported(
        "share-price inflation needs a funded asset, a donation, and a second depositor",
        "needs a fixture asset wired into the vault's constructor",
    ),
    "donate→credit": _credit_oracle,
    "deposit": _credit_oracle,
    "deposit→withdraw": _credit_oracle,
    "authorize→execute": _unsupported(
        "{name} consumes a signed authorization; judging it needs a valid signature",
        "needs a signer fixture that produces a valid authorization for this digest",
    ),
    "oracle update→valuation→borrow": _unsupported(
        "{name} reads an external price; judging it needs a controlled price source",
        "needs a controlled oracle/market fixture wired as the price source",
    ),
    "unauthorized-call": _unauthorized_pause,
    "boundary-batch": _no_property(
        "{name}'s totals are internal; no public observable distinguishes a wrap"
    ),
    "failing-callee→swallowed-call": _swallowed,
}


# ---- evaluation -------------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleObservation:
    """The structured observation a harness reported for one oracle (no text parsing)."""

    call_ok: tuple[int, ...]  # per call: 1 ok, 0 reverted, -1 not reached
    before: str = ""  # getter value before (hex word) or ""
    after: str = ""
    readable: bool = True
    received: int | None = None  # tokens the target received across the probe
    credited: int | None = None  # credit the getter added for the actor across the probe
    holdings_after: int | None = None  # tokens the target holds after the probe
    credit_after: int | None = None  # the actor's credit after the probe


@dataclass(frozen=True)
class PropertyEvaluation:
    verdict: PropertyVerdict
    reason: str
    probe_index: int = -1
    precondition_ok: bool = True
    evaluated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "probe_index": self.probe_index,
            "precondition_ok": self.precondition_ok,
            "evaluated": self.evaluated,
            "verified": False,
        }


def probe_index(sequence: Vfcs, oracle: OracleSpec) -> int:
    """The index of the probe call, bound by (role, function), or -1 when lost."""
    for index, call in enumerate(sequence.calls):
        if call.role == oracle.probe_role and call.function == oracle.probe_function:
            if not call.primitive:
                return index
    return -1


def precondition_indices(sequence: Vfcs, oracle: OracleSpec, probe: int) -> tuple[int, ...]:
    return tuple(
        index
        for index, call in enumerate(sequence.calls)
        if call.role in oracle.precondition_roles and index < probe
    )


def evaluate(
    sequence: Vfcs, oracle: OracleSpec, observation: OracleObservation
) -> PropertyEvaluation:
    """Judge one oracle from a structured observation. Deterministic; nothing verified."""
    if not oracle.executable:
        return PropertyEvaluation(
            PropertyVerdict.EXECUTION_ONLY_OBSERVATION, "no executable oracle for this sequence"
        )
    probe = probe_index(sequence, oracle)
    if probe < 0 or probe >= len(observation.call_ok):
        return PropertyEvaluation(
            PropertyVerdict.NOT_EVALUATED, "the oracle's probe call is not in this sequence"
        )
    required = precondition_indices(sequence, oracle, probe)
    if any(
        role not in {sequence.calls[i].role for i in required} for role in oracle.precondition_roles
    ):
        return PropertyEvaluation(
            PropertyVerdict.NOT_EVALUATED,
            "a precondition call the oracle needs does not precede the probe",
            probe,
            False,
        )
    if any(observation.call_ok[i] != 1 for i in required):
        return PropertyEvaluation(
            PropertyVerdict.NOT_EVALUATED,
            "a precondition call reverted, so the property could not be judged",
            probe,
            False,
        )
    probe_ok = observation.call_ok[probe]
    if probe_ok not in {0, 1}:
        return PropertyEvaluation(
            PropertyVerdict.NOT_EVALUATED, "the probe call was not reached", probe
        )
    kind = oracle.kind
    if kind is OracleKind.MECHANISM_PROBE:
        if probe_ok == 1:
            return PropertyEvaluation(
                PropertyVerdict.MECHANISM_OBSERVED,
                (
                    f"the callee reverted and {oracle.probe_function} returned normally. "
                    "Application-level impact and intentional design remain unproven. "
                    "This is a mechanism observation, not a validated security-property violation."
                ),
                probe,
                True,
                True,
            )
        return PropertyEvaluation(
            PropertyVerdict.FAILURE_PROPAGATED,
            (
                "the outer call also reverted, so the swallowed-call mechanism was not "
                "reproduced. This is not a security-property result."
            ),
            probe,
            True,
            True,
        )
    if kind in {OracleKind.CALL_MUST_FAIL, OracleKind.SECOND_CALL_MUST_FAIL}:
        if oracle.getter:
            if not observation.readable or not observation.before or not observation.after:
                return PropertyEvaluation(
                    PropertyVerdict.NOT_EVALUATED,
                    f"{oracle.getter} could not be read, so the probe result is not a verdict",
                    probe,
                )
            if observation.before != observation.after:
                return PropertyEvaluation(
                    PropertyVerdict.PROPERTY_VIOLATED,
                    f"{oracle.getter} changed across the probe ({observation.before} -> "
                    f"{observation.after})",
                    probe,
                    True,
                    True,
                )
        if probe_ok == 1:
            return PropertyEvaluation(
                PropertyVerdict.PROPERTY_VIOLATED,
                f"the probe call {oracle.probe_function} succeeded: {oracle.expression}",
                probe,
                True,
                True,
            )
        return PropertyEvaluation(
            PropertyVerdict.PROPERTY_HELD, "the probe call reverted as required", probe, True, True
        )
    if kind is OracleKind.STATE_UNCHANGED:
        if not observation.readable or not observation.before or not observation.after:
            return PropertyEvaluation(
                PropertyVerdict.NOT_EVALUATED, f"{oracle.getter} could not be read", probe
            )
        if observation.before != observation.after:
            return PropertyEvaluation(
                PropertyVerdict.PROPERTY_VIOLATED,
                f"{oracle.getter} changed across the probe ({observation.before} -> "
                f"{observation.after})",
                probe,
                True,
                True,
            )
        return PropertyEvaluation(
            PropertyVerdict.PROPERTY_HELD, f"{oracle.getter} did not change", probe, True, True
        )
    if kind in {OracleKind.CREDIT_LE_RECEIVED, OracleKind.BALANCE_NOT_INCREASED}:
        if not observation.readable or observation.credited is None or observation.received is None:
            return PropertyEvaluation(
                PropertyVerdict.NOT_EVALUATED, "credit or receipt could not be read", probe
            )
        if probe_ok != 1:
            return PropertyEvaluation(
                PropertyVerdict.NOT_EVALUATED, "the credit call reverted; nothing credited", probe
            )
        if observation.credited > observation.received:
            return PropertyEvaluation(
                PropertyVerdict.PROPERTY_VIOLATED,
                f"credited {observation.credited} exceeds received {observation.received}",
                probe,
                True,
                True,
            )
        return PropertyEvaluation(
            PropertyVerdict.PROPERTY_HELD,
            f"credited {observation.credited} <= received {observation.received}",
            probe,
            True,
            True,
        )
    if kind is OracleKind.CREDIT_LE_HOLDINGS:
        if (
            not observation.readable
            or observation.credit_after is None
            or observation.holdings_after is None
        ):
            return PropertyEvaluation(
                PropertyVerdict.NOT_EVALUATED, "credit or holdings could not be read", probe
            )
        if probe_ok != 1:
            return PropertyEvaluation(
                PropertyVerdict.NOT_EVALUATED, "the credit call reverted; nothing credited", probe
            )
        if observation.credit_after > observation.holdings_after:
            return PropertyEvaluation(
                PropertyVerdict.PROPERTY_VIOLATED,
                f"credit {observation.credit_after} exceeds holdings {observation.holdings_after}",
                probe,
                True,
                True,
            )
        return PropertyEvaluation(
            PropertyVerdict.PROPERTY_HELD,
            f"credit {observation.credit_after} <= holdings {observation.holdings_after}",
            probe,
            True,
            True,
        )
    return PropertyEvaluation(PropertyVerdict.NOT_EVALUATED, f"unknown oracle {kind.value}")


def property_index(specs: list[PropertySpec]) -> dict[str, Any]:
    """Counts by declaration and family (for coverage reporting)."""
    by_declaration: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for spec in specs:
        by_declaration[spec.declaration.value] = by_declaration.get(spec.declaration.value, 0) + 1
        by_family[spec.family.value] = by_family.get(spec.family.value, 0) + 1
    return {"by_declaration": by_declaration, "by_family": by_family, "total": len(specs)}


__all__ = [
    "EXECUTABLE_ORACLES",
    "OracleKind",
    "OracleObservation",
    "OracleSpec",
    "PropertyDeclaration",
    "PropertyEvaluation",
    "PropertyFamily",
    "PropertyProvenance",
    "PropertySpec",
    "PropertyVerdict",
    "build_property",
    "evaluate",
    "family_for",
    "probe_index",
    "property_index",
    "public_getter",
]

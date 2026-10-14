"""Bounded state transitions, candidate invariants, and exploit-path search.

A transition is an abstract step, not a symbolic execution. A candidate path
is not an exploit. A generated invariant is not proved. This module never
emits ``verified`` or ``reproduced``.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.parsing.comments import strip_comments
from app.parsing.solidity_cfg import node_at, node_ranges
from app.parsing.solidity_cross_dataflow import (
    analyze_operation_authorization,
    reentrancy_dependencies,
)
from app.parsing.solidity_dataflow import (
    DataflowModel,
    DependencyEdge,
    FunctionSummary,
    analyze_dataflow,
)
from app.parsing.solidity_defi import _token_calls
from app.parsing.solidity_ir import (
    SemanticFunction,
    SemanticProgram,
    StateAccess,
    StateDeclaration,
)
from app.parsing.solidity_value_flow import CallResolution, _path_in, resolve_call

MAX_TRANSITIONS = 64
MAX_PATHS = 16
MAX_DEPTH = 4
MAX_INVARIANTS = 32
_FORBIDDEN = frozenset({"verified", "reproduced", "safe"})
_ROLES = {
    "totalsupply": "supply",
    "_totalsupply": "supply",
    "totalassets": "assets",
    "_totalassets": "assets",
    "balances": "balance",
    "balance": "balance",
    "allowance": "allowance",
    "allowances": "allowance",
    "debt": "debt",
    "debts": "debt",
    "reserve": "reserve",
    "reserves": "reserve",
    "shares": "shares",
    "implementation": "implementation",
    "_implementation": "implementation",
    "paused": "paused",
    "_paused": "paused",
    "nonce": "nonce",
    "nonces": "nonce",
    "owner": "authority",
    "_owner": "authority",
}
_ASSET_CALLS = frozenset({"transfer", "transferFrom", "safeTransfer", "safeTransferFrom", "send"})
_INFLOW = frozenset({"transferFrom", "safeTransferFrom"})


@dataclass(frozen=True)
class TransitionFact:
    fact_id: str
    kind: str
    relation: str
    subject: str
    value: str
    provenance: str
    operation_ids: tuple[str, ...] = ()
    call_ids: tuple[str, ...] = ()
    span: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class StateTransition:
    transition_id: str
    contract: str
    function_id: str
    pre_state: tuple[str, ...]
    inputs: tuple[str, ...]
    conditions: tuple[str, ...]
    computations: tuple[str, ...]
    external_calls: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    assets: tuple[str, ...]
    authorization: tuple[str, ...]
    post_state: tuple[str, ...]
    assumptions: tuple[str, ...]
    provenance: str
    spans: tuple[tuple[int, int, int, int], ...]
    status: str
    incomplete_reason: str = ""
    call_ids: tuple[str, ...] = ()
    resolutions: tuple[str, ...] = ()
    facts: tuple[TransitionFact, ...] = ()
    source_file: str = ""
    source_identity: str = ""


@dataclass(frozen=True)
class PathStep:
    step_id: str
    transition_id: str
    function_id: str
    contract_id: str
    relation: str
    source_id: str
    sink_id: str
    call_ids: tuple[str, ...] = ()
    operation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidatePath:
    path_id: str
    status: str
    depth: int
    steps: tuple[PathStep, ...]
    transition_ids: tuple[str, ...]
    function_ids: tuple[str, ...]
    contract_ids: tuple[str, ...]
    call_ids: tuple[str, ...]
    operation_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    assumptions: tuple[str, ...]
    bound: str
    completeness: str
    incomplete_reason: str = ""
    reason: str = ""
    invariant_id: str = ""
    relationship: str = ""


@dataclass(frozen=True)
class AbstractPredicate:
    """Machine-readable candidate predicate. Not a proved formula."""

    predicate_type: str
    lhs: str = ""
    rhs: str = ""
    subject: str = ""
    operator: str = ""
    previous: str = ""
    next: str = ""
    required_guard: str = ""
    representation_status: str = "unknown"
    unsupported: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateInvariant:
    invariant_id: str
    category: str
    relation: str
    operations: tuple[str, ...]
    state_variables: tuple[str, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    source: str
    status: str
    confidence: str
    assumptions: tuple[str, ...]
    declarations: tuple[str, ...] = ()
    predicate: AbstractPredicate | None = None
    unsupported: tuple[str, ...] = ()
    representation_status: str = "unknown"
    contract: str = ""
    source_file: str = ""
    source_identity: str = ""


@dataclass(frozen=True)
class InvariantCheck:
    invariant_id: str
    transition_id: str
    status: str
    reason: str


@dataclass(frozen=True)
class AssetMovement:
    """One observed movement. Unknown is not an inflow."""

    kind: str
    detail: str
    provenance: str


@dataclass(frozen=True)
class AccountingAssessment:
    status: str
    reason: str
    relation: str = ""
    operation_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


@dataclass
class TransitionModel:
    status: str
    transitions: tuple[StateTransition, ...] = ()
    paths: tuple[CandidatePath, ...] = ()
    invariants: tuple[CandidateInvariant, ...] = ()
    checks: tuple[InvariantCheck, ...] = ()
    incomplete_reason: str = ""
    bounds: dict[str, int] = field(default_factory=dict)
    compiler_version: str = ""
    compiler_ir_status: str = ""
    origin: str = "parser"


def analyze_state_transitions(
    program: SemanticProgram,
    *,
    also: Sequence[SemanticProgram] = (),
    limits: dict[str, int] | None = None,
) -> TransitionModel:
    programs = (program, *tuple(also))
    functions = [item for item in _functions(programs)]
    if not functions:
        return TransitionModel(
            "unavailable", incomplete_reason="no functions", bounds=_bounds(limits)
        )
    caps = _bounds(limits)
    collided = _identity_collisions(programs)
    flow_by_source: dict[str, DataflowModel] = {}
    source_of: dict[int, str] = {}
    for item in programs:
        identity = _source_identity(item)
        source_of[id(item)] = identity
        flow_by_source[identity] = analyze_dataflow(item)
    reason = program.incomplete_reason
    if collided:
        reason = _join_reason(
            reason, "function identity collision across sources; transitions were namespaced"
        )
    if _compiler_version(programs) == "mixed":
        reason = _join_reason(reason, "compiler versions differ")
    selected = _function_pairs(programs)
    if len(selected) > caps["transitions"]:
        selected = selected[: caps["transitions"]]
        reason = _join_reason(reason, "transition limit reached")
    transitions = tuple(
        _transition(
            programs,
            flow_by_source[source_of[id(home)]],
            item,
            caps,
            home,
            source_of[id(home)],
            namespace=item.identity in collided,
        )
        for home, item in selected
    )
    invariants = _invariants(programs, caps, source_of)
    checks = _checks(programs, transitions, invariants, flow_by_source, caps)
    paths, path_reason = _paths(programs, flow_by_source, transitions, checks, invariants, caps)
    reason = _join_reason(reason, path_reason)
    status = "partial" if reason else "available"
    model = TransitionModel(
        status,
        transitions,
        paths,
        invariants,
        checks,
        reason,
        {**caps, "transitions_emitted": len(transitions), "paths_emitted": len(paths)},
        _compiler_version(programs),
        program.compiler_ir_status,
        "parser",
    )
    _reject_forbidden(model)
    return model


def find_candidate_exploit_paths(
    program: SemanticProgram,
    *,
    also: Sequence[SemanticProgram] = (),
    limits: dict[str, int] | None = None,
) -> tuple[CandidatePath, ...]:
    return analyze_state_transitions(program, also=also, limits=limits).paths


def analyze_accounting_transition(
    program: SemanticProgram, function_id: str, flow: DataflowModel | None = None
) -> AccountingAssessment:
    """Classify an accounting relation. An asset transfer alone is observed."""
    if not program.functions:
        return AccountingAssessment("unavailable", "no functions")
    flow = flow or analyze_dataflow(program)
    summary = flow._summary(function_id)
    function = _function(program, function_id)
    if summary is None or function is None:
        return AccountingAssessment("unknown", "function was not resolved")
    if summary.incomplete:
        return AccountingAssessment(
            "incomplete",
            summary.incomplete_reason or "function summary is incomplete",
        )
    return _accounting(program, function, summary)


def _bounds(limits: dict[str, int] | None) -> dict[str, int]:
    defaults = {
        "transitions": MAX_TRANSITIONS,
        "paths": MAX_PATHS,
        "depth": MAX_DEPTH,
        "invariants": MAX_INVARIANTS,
    }
    if not limits:
        return defaults
    bounded = {}
    for key, default in defaults.items():
        requested = int(limits.get(key, default))
        bounded[key] = max(1, min(requested, default))
    return bounded


def _functions(programs: Sequence[SemanticProgram]) -> list[SemanticFunction]:
    return [function for _program, function in _function_pairs(programs)]


def _function_pairs(
    programs: Sequence[SemanticProgram],
) -> list[tuple[SemanticProgram, SemanticFunction]]:
    found = [(program, function) for program in programs for function in program.functions]
    found.sort(
        key=lambda item: (
            item[1].contract,
            item[1].line,
            item[1].name,
            item[1].identity,
            item[0].file,
        )
    )
    return found


def _transition(
    programs: Sequence[SemanticProgram],
    flow: DataflowModel,
    function: SemanticFunction,
    caps: dict[str, int],
    home: SemanticProgram,
    source_identity: str,
    *,
    namespace: bool,
) -> StateTransition:
    del caps
    summary = flow._summary(function.identity)
    reads = tuple(item.operation_id for item in function.access_sites if item.kind != "write")
    writes = tuple(item.operation_id for item in function.access_sites if item.kind != "read")
    external = tuple(site.call_id for site in function.call_sites if site.external)
    assumptions: list[str] = []
    resolutions: list[str] = []
    for site in function.call_sites:
        resolution, _callee_source = _resolve_in_program(programs, home, function, site)
        resolutions.append(f"{site.call_id}:{resolution.status}:{resolution.reason}")
        if resolution.status != "resolved":
            assumptions.append(f"{site.call_id} {resolution.reason}".strip())
    if summary is not None and summary.placement_gaps:
        assumptions.append("CFG placement incomplete")
    conditions, condition_gap = _conditions(function)
    if condition_gap:
        assumptions.append(condition_gap)
    authorization = tuple(
        f"{item.operation_id}:{item.guard_status}"
        for item in function.access_sites
        if item.kind != "read"
    )
    assets = tuple(
        f"{item.source}->{item.destination}:{item.amount}"
        for item in (summary.asset_flows if summary else ())
    )
    computations = tuple(
        value.identity
        for value in (summary.values if summary else ())
        if value.kind in {"local", "join", "internal-return", "external-return"}
    )
    incomplete = bool(summary and summary.incomplete)
    reason = summary.incomplete_reason if summary and summary.incomplete else ""
    status = (
        "incomplete" if incomplete else "observed" if (reads or writes or external) else "unknown"
    )
    facts = _facts(function, summary)
    transition_id = f"{source_identity}:{function.identity}" if namespace else function.identity
    return StateTransition(
        transition_id,
        function.contract,
        function.identity,
        reads,
        function.parameters,
        conditions,
        computations,
        external,
        reads,
        writes,
        assets,
        authorization,
        writes,
        tuple(dict.fromkeys(assumptions)),
        "parser",
        tuple(_span(item) for item in function.access_sites),
        status,
        reason,
        tuple(site.call_id for site in function.call_sites),
        tuple(resolutions),
        tuple(facts),
        home.file,
        source_identity,
    )


def _facts(function: SemanticFunction, summary: FunctionSummary | None) -> list[TransitionFact]:
    found: list[TransitionFact] = []
    for item in function.access_sites:
        found.append(
            TransitionFact(
                item.operation_id,
                "read" if item.kind == "read" else "write",
                item.kind,
                item.path,
                item.symbol,
                "state",
                (item.operation_id,),
                span=_span(item),
            )
        )
    for site in function.call_sites:
        if not site.external:
            continue
        found.append(
            TransitionFact(
                site.call_id,
                "external",
                site.call_type or "external",
                site.target,
                site.callee,
                "external",
                call_ids=(site.call_id,),
                span=(
                    site.span.start_byte,
                    site.span.end_byte,
                    site.span.start_line,
                    site.span.end_line,
                ),
            )
        )
    if summary is not None:
        for flow_item in summary.asset_flows:
            found.append(
                TransitionFact(
                    f"asset:{flow_item.function}:{flow_item.amount[:40]}",
                    "asset",
                    "moves-assets",
                    flow_item.source,
                    flow_item.destination,
                    "observed",
                    call_ids=(),
                )
            )
    return found


def _conditions(function: SemanticFunction) -> tuple[tuple[str, ...], str]:
    source = function.source or ""
    cfg, _ranges = node_ranges(source)
    if not cfg.known:
        return (), "control flow is unknown"
    texts = tuple(
        node.text.strip()[:180]
        for node in cfg.nodes
        if node.kind in {"require", "if"} and node.text.strip()
    )
    return texts, ""


def _accounting(
    program: SemanticProgram, function: SemanticFunction, summary: FunctionSummary
) -> AccountingAssessment:
    found = _relation_findings(program, function, summary)
    for item in found:
        if item.status == "potential":
            return item
    for item in found:
        if item.status == "observed":
            return item
    if found:
        return found[-1]
    return AccountingAssessment("unknown", "no accounting relation was established")


def _relation_findings(
    program: SemanticProgram, function: SemanticFunction, summary: FunctionSummary
) -> tuple[AccountingAssessment, ...]:
    """Every accounting relation this function supports. The first is not the only one."""
    roles = _declared_roles(program, function.contract)
    assumption = ("variable role is a name heuristic, not a type proof",)
    writes = [item for item in function.access_sites if item.kind != "read"]
    found: list[AccountingAssessment] = []
    supply_writes = [item for item in writes if _role(item.symbol) == "supply"]
    share_writes = [item for item in writes if _role(item.symbol) == "shares"]
    asset_symbols = {name for name, role in roles.items() if role == "assets"}
    balance_symbols = {name for name, role in roles.items() if role == "balance"}
    assets_written = any(_role(item.symbol) == "assets" for item in writes)
    balances_written = any(_role(item.symbol) == "balance" for item in writes)
    influenced = {edge.sink for edge in summary.edges if edge.kind == "state-read-influences-write"}
    increased = [
        item
        for item in (*supply_writes, *share_writes)
        if item.operation_id in influenced and _direction(function, item) == "increase"
    ]
    if increased and asset_symbols and not assets_written and not _has_inflow(function):
        found.append(
            AccountingAssessment(
                "potential",
                "supply or shares increase without a modeled asset inflow",
                "supply-increased-without-asset-inflow",
                tuple(item.operation_id for item in increased),
                assumption,
            )
        )
    supply_increased = [
        item
        for item in supply_writes
        if item.operation_id in influenced and _direction(function, item) == "increase"
    ]
    if supply_increased and balance_symbols and not balances_written and not _has_inflow(function):
        found.append(
            AccountingAssessment(
                "potential",
                "total supply increases without a modeled balance update",
                "supply-increased-without-balance-update",
                tuple(item.operation_id for item in supply_increased),
                assumption,
            )
        )
    debt_writes = [
        item
        for item in writes
        if _role(item.symbol) == "debt"
        and item.operation_id in influenced
        and _direction(function, item) == "decrease"
        and not _repayment_reaches(summary, item)
    ]
    if debt_writes:
        found.append(
            AccountingAssessment(
                "potential",
                "debt decreases without a modeled repayment",
                "debt-reduced-without-repayment",
                tuple(item.operation_id for item in debt_writes),
                assumption,
            )
        )
    allowance_hits = _allowance_unconsumed(function, summary)
    if allowance_hits:
        found.append(
            AccountingAssessment(
                "potential",
                "an allowance value reaches transferFrom and allowance is not written",
                "allowance-not-consumed",
                tuple(item.operation_id for item in allowance_hits),
                assumption,
            )
        )
    if summary.asset_flows and not any(
        _role(item.symbol) in {"supply", "assets", "shares"} for item in writes
    ):
        found.append(
            AccountingAssessment(
                "observed",
                "asset transfer observed",
                "asset-transfer",
                assumptions=assumption,
            )
        )
    if balances_written:
        found.append(
            AccountingAssessment(
                "observed",
                "balance movement observed",
                "balance-movement",
                tuple(item.operation_id for item in writes if _role(item.symbol) == "balance"),
                assumption,
            )
        )
    if supply_writes and assets_written:
        found.append(
            AccountingAssessment(
                "observed",
                "supply and assets are both written; conservation is not proved",
                "supply-and-assets-updated",
                tuple(
                    item.operation_id
                    for item in (
                        *supply_writes,
                        *[item for item in writes if _role(item.symbol) == "assets"],
                    )
                ),
                assumption,
            )
        )
    if not found:
        found.append(
            AccountingAssessment(
                "unknown", "no accounting relation was established", assumptions=assumption
            )
        )
    return tuple(found)


def _allowance_unconsumed(
    function: SemanticFunction, summary: FunctionSummary
) -> tuple[StateAccess, ...]:
    spends = [site for site in function.call_sites if site.callee in _INFLOW]
    reads = [
        item
        for item in function.access_sites
        if _role(item.symbol) == "allowance" and item.kind != "write"
    ]
    writes = [
        item
        for item in function.access_sites
        if _role(item.symbol) == "allowance" and item.kind != "read"
    ]
    if not spends or not reads:
        return ()
    hits: list[StateAccess] = []
    for site in spends:
        reaching = [
            item for item in reads if _read_reaches_arguments(summary, item, site.arguments)
        ]
        if not reaching:
            continue
        if any(_write_consumes_call(item, site, writes) for item in reaching):
            continue
        hits.extend(reaching)
    return tuple(dict.fromkeys(hits))


def _write_consumes_call(read: StateAccess, site: object, writes: list[StateAccess]) -> bool:
    """A later write of the same declaration covers only calls that precede it."""
    call_end = int(getattr(getattr(site, "span", None), "end_byte", 0) or 0)
    return any(
        item.declaration_id == read.declaration_id and item.span.start_byte > call_end
        for item in writes
    )


def _read_reaches_arguments(summary: FunctionSummary, access: StateAccess, arguments: str) -> bool:
    if _path_in(access.path, arguments):
        return True
    seen: set[str] = set()
    stack = [access.operation_id]
    reached: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for edge in summary.edges:
            if edge.source != current:
                continue
            reached.add(edge.sink)
            stack.append(edge.sink)
    for value in summary.values:
        if value.identity not in reached:
            continue
        slug = value.identity.rsplit(":", 1)[-1]
        if slug and _name_in(slug, arguments):
            return True
    return False


def _name_in(name: str, text: str) -> bool:
    for match in re.finditer(rf"\b{re.escape(name)}\b", text):
        del match
        return True
    return False


def _invariants(
    programs: Sequence[SemanticProgram],
    caps: dict[str, int],
    source_of: dict[int, str],
) -> tuple[CandidateInvariant, ...]:
    """One candidate per declaring contract. Matching names do not merge contracts."""
    grouped: dict[tuple[int, str, str], list[StateDeclaration]] = defaultdict(list)
    for program in programs:
        for decl in program.declarations:
            if _role(decl.symbol):
                grouped[(id(program), program.file, decl.contract)].append(decl)
    found: list[CandidateInvariant] = []
    for program_id, source_file, contract in sorted(grouped, key=lambda item: (item[1], item[2])):
        decls = grouped[(program_id, source_file, contract)]
        by_role: dict[str, list[StateDeclaration]] = defaultdict(list)
        for decl in sorted(decls, key=lambda item: item.declaration_id):
            by_role[_role(decl.symbol)].append(decl)
        built = _contract_invariants(
            programs,
            source_file,
            contract,
            by_role,
            source_of.get(program_id, ""),
        )
        for item in built:
            if len(found) >= caps["invariants"]:
                return tuple(found)
            found.append(item)
    return tuple(found)


def _contract_invariants(
    programs: Sequence[SemanticProgram],
    source_file: str,
    contract: str,
    by_role: dict[str, list[StateDeclaration]],
    source_identity: str,
) -> list[CandidateInvariant]:
    specs: list[tuple[str, str, str, AbstractPredicate, tuple[StateDeclaration, ...]]] = []
    if by_role["supply"] and by_role["balance"]:
        supply, balance = by_role["supply"][0], by_role["balance"][0]
        mapping = "mapping" in balance.type_name
        specs.append(
            (
                "supply-balance",
                "equality",
                f"{supply.symbol} == sum({balance.symbol})"
                if mapping
                else f"{supply.symbol} == {balance.symbol}",
                AbstractPredicate(
                    "equality",
                    lhs=supply.symbol,
                    rhs=f"sum({balance.symbol})" if mapping else balance.symbol,
                    representation_status="partial" if mapping else "exact",
                    unsupported=("mapping aggregation is not encoded",) if mapping else (),
                ),
                tuple(by_role["supply"] + by_role["balance"]),
            )
        )
    if by_role["assets"] and (by_role["supply"] or by_role["shares"]):
        assets = by_role["assets"][0]
        other = (by_role["shares"] or by_role["supply"])[0]
        involved = tuple(by_role["assets"] + by_role["supply"] + by_role["shares"])
        specs.append(
            (
                "asset-share",
                "asset-share",
                f"{other.symbol} corresponds to {assets.symbol}",
                AbstractPredicate(
                    "asset-share",
                    lhs=other.symbol,
                    rhs=assets.symbol,
                    representation_status="partial",
                    unsupported=("exchange-rate formula is not encoded",),
                ),
                involved,
            )
        )
    if by_role["authority"]:
        subject = by_role["authority"][0]
        specs.append(
            (
                "authorization",
                "authorization",
                f"authorized actor required to modify {subject.symbol}",
                AbstractPredicate(
                    "authorization",
                    subject=subject.symbol,
                    required_guard="authorized_actor",
                    representation_status="partial",
                    unsupported=("guard predicate is not a complete authorization formula",),
                ),
                tuple(by_role["authority"]),
            )
        )
    if by_role["nonce"]:
        subject = by_role["nonce"][0]
        specs.append(
            (
                "nonce",
                "monotonicity",
                f"{subject.symbol}_after > {subject.symbol}_before",
                AbstractPredicate(
                    "monotonic",
                    subject=subject.symbol,
                    operator=">",
                    previous=f"{subject.symbol}_before",
                    next=f"{subject.symbol}_after",
                    representation_status="partial",
                    unsupported=("before and after values are not tracked",),
                ),
                tuple(by_role["nonce"]),
            )
        )
    if by_role["implementation"]:
        subject = by_role["implementation"][0]
        specs.append(
            (
                "upgrade",
                "upgrade",
                f"authorized actor required to modify {subject.symbol}",
                AbstractPredicate(
                    "authorization",
                    subject=subject.symbol,
                    required_guard="authorized_actor",
                    representation_status="partial",
                    unsupported=("storage compatibility is not encoded",),
                ),
                tuple(by_role["implementation"]),
            )
        )
    if by_role["paused"]:
        subject = by_role["paused"][0]
        specs.append(
            (
                "pause",
                "pause",
                f"{subject.symbol} implies sensitive operations are unavailable",
                AbstractPredicate(
                    "pause",
                    subject=subject.symbol,
                    representation_status="partial",
                    unsupported=("sensitive-operation set is not encoded",),
                ),
                tuple(by_role["paused"]),
            )
        )
    if by_role["debt"]:
        subject = by_role["debt"][0]
        specs.append(
            (
                "debt",
                "debt",
                f"{subject.symbol} decreases only with modeled repayment",
                AbstractPredicate(
                    "debt",
                    subject=subject.symbol,
                    representation_status="partial",
                    unsupported=("repayment formula is not encoded",),
                ),
                tuple(by_role["debt"]),
            )
        )
    if by_role["reserve"]:
        subject = by_role["reserve"][0]
        specs.append(
            (
                "reserve",
                "reserve",
                f"{subject.symbol}_after = {subject.symbol}_before + inflow - outflow",
                AbstractPredicate(
                    "reserve",
                    subject=subject.symbol,
                    representation_status="partial",
                    unsupported=("inflow and outflow are not fully encoded",),
                ),
                tuple(by_role["reserve"]),
            )
        )
    if by_role["allowance"]:
        subject = by_role["allowance"][0]
        specs.append(
            (
                "allowance",
                "allowance",
                f"transferFrom consumes {subject.symbol}",
                AbstractPredicate(
                    "allowance",
                    subject=subject.symbol,
                    representation_status="partial",
                    unsupported=("allowance consumption formula is not encoded",),
                ),
                tuple(by_role["allowance"]),
            )
        )
    found: list[CandidateInvariant] = []
    for category, kind, relation, predicate, involved in specs:
        declarations = tuple(dict.fromkeys(item.declaration_id for item in involved))
        symbols = tuple(dict.fromkeys(item.symbol for item in involved))
        found.append(
            CandidateInvariant(
                f"inv:{category}:{contract}:{source_identity}:{','.join(symbols)}",
                kind,
                relation,
                _operations_for(programs, source_file, contract, set(declarations)),
                symbols,
                (),
                (relation,),
                "name-heuristic",
                "candidate",
                "low",
                (
                    "candidate invariant, not a proof",
                    "variable role is a name heuristic, not a type proof",
                    "scoped to the declaring contract",
                ),
                declarations,
                predicate,
                predicate.unsupported,
                predicate.representation_status,
                contract,
                source_file,
                source_identity,
            )
        )
    return found


def _checks(
    programs: Sequence[SemanticProgram],
    transitions: tuple[StateTransition, ...],
    invariants: tuple[CandidateInvariant, ...],
    flow_by_source: dict[str, DataflowModel],
    caps: dict[str, int],
) -> tuple[InvariantCheck, ...]:
    del caps
    found: list[InvariantCheck] = []
    for transition in transitions:
        flow = flow_by_source.get(transition.source_identity)
        summary = flow._summary(transition.function_id) if flow is not None else None
        function = _function_for_transition(programs, transition)
        findings: tuple[AccountingAssessment, ...] | None = None
        for invariant in invariants:
            if (
                invariant.contract
                and function is not None
                and invariant.contract != function.contract
            ):
                continue
            if invariant.source_identity and transition.source_identity:
                if invariant.source_identity != transition.source_identity:
                    continue
            elif invariant.source_file and transition.source_file:
                if invariant.source_file != transition.source_file:
                    continue
            if transition.status == "incomplete":
                reason = transition.incomplete_reason or "transition is incomplete"
                if function is not None and _loop_unmodeled(function):
                    reason = _join_reason(reason, "loop-carried state is not modeled")
                status, reason = "incomplete", reason
            elif summary is None or function is None:
                status, reason = "unknown", "function summary is unavailable"
            elif summary.incomplete:
                reason = summary.incomplete_reason or "function summary is incomplete"
                if function is not None and _loop_unmodeled(function):
                    reason = _join_reason(reason, "loop-carried state is not modeled")
                status, reason = "incomplete", reason
            else:
                if findings is None:
                    program = _program_for_transition(programs, transition)
                    findings = (
                        _relation_findings(program, function, summary)
                        if program is not None
                        else (AccountingAssessment("unknown", "function was not resolved"),)
                    )
                status, reason = _judge(
                    transition, invariant, findings, function, summary, programs
                )
            found.append(
                InvariantCheck(invariant.invariant_id, transition.transition_id, status, reason)
            )
    return tuple(found)


def _judge(
    transition: StateTransition,
    invariant: CandidateInvariant,
    findings: tuple[AccountingAssessment, ...],
    function: SemanticFunction,
    summary: FunctionSummary,
    programs: Sequence[SemanticProgram],
) -> tuple[str, str]:
    declared = set(invariant.declarations)
    touched = any(
        (item.declaration_id in declared if declared else item.symbol in invariant.state_variables)
        and item.kind != "read"
        for item in function.access_sites
    )
    external = bool(transition.external_calls)
    for item in findings:
        if item.status == "potential" and _relation_matches(item.relation, invariant.category):
            return "potential", item.reason
    if invariant.category in {"authorization", "upgrade"}:
        verdict = _guard_verdict(function, invariant)
        if verdict is not None:
            return verdict
    if not touched and not external and summary.placement_gaps == 0:
        if _internal_might_touch(programs, function, invariant, frozenset()):
            return "unknown", "an internal call may update the invariant variables"
        if _loop_unmodeled(function):
            return "incomplete", "loop-carried state is not modeled"
        return (
            "non_interference_observed",
            "no modeled write of these declarations was observed; this is not a proof",
        )
    if not touched and external:
        return "unknown", "an external call may update the invariant variables"
    return "unknown", "the transition touches the invariant but no violation was established"


def _guard_verdict(
    function: SemanticFunction, invariant: CandidateInvariant
) -> tuple[str, str] | None:
    relevant = [
        item
        for item in function.access_sites
        if item.kind != "read"
        and (
            item.declaration_id in invariant.declarations
            if invariant.declarations
            else item.symbol in invariant.state_variables
        )
    ]
    if not relevant:
        return None
    if any(item.guard_status == "unguarded" for item in relevant):
        if invariant.category == "upgrade":
            return "potential", "implementation is written without a dominating guard"
        return "potential", "authority state is written without a dominating guard"
    if any(item.guard_status != "guarded" for item in relevant):
        return "unknown", "a guard does not dominate every write of this variable"
    return None


def _internal_might_touch(
    programs: Sequence[SemanticProgram],
    function: SemanticFunction,
    invariant: CandidateInvariant,
    seen: frozenset[str],
) -> bool:
    if function.identity in seen or len(seen) > 4:
        return True
    for site in function.call_sites:
        if site.external:
            continue
        resolution = _resolve_site(programs, function, site)
        if resolution.status != "resolved" or not resolution.function_id:
            return True
        callee = _function_in(programs, resolution.function_id)
        if callee is None:
            return True
        if any(
            (
                item.declaration_id in invariant.declarations
                if invariant.declarations
                else item.symbol in invariant.state_variables
            )
            and item.kind != "read"
            for item in callee.access_sites
        ):
            return True
        if _internal_might_touch(programs, callee, invariant, seen | {function.identity}):
            return True
    return False


def _paths(
    programs: Sequence[SemanticProgram],
    flow_by_source: dict[str, DataflowModel],
    transitions: tuple[StateTransition, ...],
    checks: tuple[InvariantCheck, ...],
    invariants: tuple[CandidateInvariant, ...],
    caps: dict[str, int],
) -> tuple[tuple[CandidatePath, ...], str]:
    by_id = {item.transition_id: item for item in transitions}
    found: list[CandidatePath] = []
    reason = ""
    for transition in transitions:
        if len(found) >= caps["paths"]:
            reason = "path limit reached"
            break
        local_flow = flow_by_source.get(transition.source_identity)
        if local_flow is not None:
            _single_paths(programs, local_flow, transition, checks, invariants, found, caps)
    edges = _resolved_edges(programs, transitions)
    if len(found) < caps["paths"]:
        _multi_paths(by_id, edges, checks, invariants, found, caps)
    if len(found) > caps["paths"]:
        found = found[: caps["paths"]]
        reason = "path limit reached"
    for item in found:
        if item.status in _FORBIDDEN:
            raise RuntimeError(item.status)
    if any(item.completeness == "partial" for item in found):
        reason = _join_reason(reason, "depth bound reached")
    return tuple(found), reason


def _single_paths(
    programs: Sequence[SemanticProgram],
    flow: DataflowModel,
    transition: StateTransition,
    checks: tuple[InvariantCheck, ...],
    invariants: tuple[CandidateInvariant, ...],
    found: list[CandidatePath],
    caps: dict[str, int],
) -> None:
    if len(found) >= caps["paths"] or transition.status == "incomplete":
        return
    home = _program_for_transition(programs, transition) or programs[0]
    for call, read_id, write_id in reentrancy_dependencies(home, transition.function_id, flow):
        if len(found) >= caps["paths"]:
            return
        found.append(
            _path(
                f"reentrancy:{transition.function_id}:{call}:{write_id}",
                transition,
                (
                    _step(transition, "reads-from", read_id, call, (read_id,), (call,)),
                    _step(
                        transition, "depends-on", read_id, write_id, (read_id, write_id), (call,)
                    ),
                    _step(
                        transition, "externally-calls", call, write_id, (read_id, write_id), (call,)
                    ),
                    _step(transition, "writes-to", write_id, write_id, (write_id,), (call,)),
                ),
                "state read reaches a write after an external call",
                depth=1,
                cap=caps["depth"],
            )
        )
    summary = flow._summary(transition.function_id)
    function = _function_for_transition(programs, transition)
    program = home
    if summary is not None and function is not None and not summary.incomplete:
        for accounting in _relation_findings(program, function, summary):
            if accounting.status != "potential" or len(found) >= caps["paths"]:
                continue
            invariant = _matching_invariant(
                invariants, checks, transition.transition_id, accounting.relation
            )
            found.append(
                _path(
                    f"accounting:{transition.function_id}:{accounting.relation}",
                    transition,
                    (
                        _step(
                            transition,
                            "violates-candidate-invariant",
                            accounting.relation,
                            invariant or accounting.relation,
                            accounting.operation_ids,
                            (),
                        ),
                    ),
                    accounting.reason,
                    depth=1,
                    cap=caps["depth"],
                    invariant_id=invariant,
                )
            )
        _oracle_paths(transition, function, summary, found, caps)
        _delegate_paths(program, flow, transition, function, found, caps)
        _authority_paths(program, transition, function, found, caps)


def _oracle_paths(
    transition: StateTransition,
    function: SemanticFunction,
    summary: FunctionSummary,
    found: list[CandidatePath],
    caps: dict[str, int],
) -> None:
    sensitive_ids = {
        item.operation_id
        for item in function.access_sites
        if item.kind != "read"
        and _role(item.symbol) in {"assets", "supply", "debt", "balance", "shares", "reserve"}
    }
    ordered_writes = [
        item.operation_id for item in function.access_sites if item.operation_id in sensitive_ids
    ]
    for site in function.call_sites:
        if site.call_type != "oracle" or len(found) >= caps["paths"]:
            continue
        return_id = f"{function.identity}:return:{site.call_id}"
        reached = [
            write_id for write_id in ordered_writes if _forward(summary.edges, return_id, write_id)
        ]
        if not reached:
            continue
        write_id = reached[0]
        found.append(
            _path(
                f"oracle:{site.call_id}",
                transition,
                (
                    _step(
                        transition, "externally-calls", site.call_id, return_id, (), (site.call_id,)
                    ),
                    _step(
                        transition, "depends-on", return_id, write_id, (write_id,), (site.call_id,)
                    ),
                    _step(
                        transition, "writes-to", write_id, write_id, (write_id,), (site.call_id,)
                    ),
                ),
                "oracle return reaches a security-sensitive write",
                depth=1,
                cap=caps["depth"],
                assumptions=("no oracle validation was established",),
            )
        )


def _delegate_paths(
    program: SemanticProgram,
    flow: DataflowModel,
    transition: StateTransition,
    function: SemanticFunction,
    found: list[CandidatePath],
    caps: dict[str, int],
) -> None:
    from app.parsing.solidity_cross_dataflow import analyze_delegatecall

    for site in function.call_sites:
        if site.call_type != "delegatecall" or len(found) >= caps["paths"]:
            continue
        result = analyze_delegatecall(program, function.identity, site.call_id, flow)
        if result.status != "potential":
            continue
        found.append(
            _path(
                f"delegate:{site.call_id}",
                transition,
                (
                    _step(
                        transition,
                        "externally-calls",
                        site.target,
                        site.call_id,
                        (),
                        (site.call_id,),
                    ),
                ),
                result.summary,
                depth=1,
                cap=caps["depth"],
                assumptions=(f"target provenance is {result.provenance}",),
            )
        )


def _authority_paths(
    program: SemanticProgram,
    transition: StateTransition,
    function: SemanticFunction,
    found: list[CandidatePath],
    caps: dict[str, int],
) -> None:
    for item in function.access_sites:
        if item.kind == "read" or len(found) >= caps["paths"]:
            continue
        role = _role(item.symbol)
        if role not in {"authority", "implementation"}:
            continue
        result = analyze_operation_authorization(program, function.identity, item.operation_id)
        if result.status != "potential":
            continue
        relation = "changes-authority" if role == "authority" else "changes-implementation"
        found.append(
            _path(
                f"{relation}:{item.operation_id}",
                transition,
                (
                    _step(
                        transition, relation, item.operation_id, item.path, (item.operation_id,), ()
                    ),
                ),
                result.summary,
                depth=1,
                cap=caps["depth"],
            )
        )


def _multi_paths(
    by_id: dict[str, StateTransition],
    edges: list[dict[str, str]],
    checks: tuple[InvariantCheck, ...],
    invariants: tuple[CandidateInvariant, ...],
    found: list[CandidatePath],
    caps: dict[str, int],
) -> None:
    children: dict[str, list[dict[str, str]]] = defaultdict(list)
    for edge in edges:
        children[edge["caller"]].append(edge)
    for key in children:
        children[key].sort(key=lambda item: item["call_id"])
    property_ids = {check.transition_id for check in checks if check.status == "potential"}
    property_ids.update(
        item.transition_id
        for path in found
        if path.completeness == "complete" and path.depth == 1
        for item in (by_id.get(path.function_ids[0]),)
        if item is not None
    )

    def walk(chain: list[tuple[str, str]]) -> None:
        if len(found) >= caps["paths"]:
            return
        current = chain[-1][0]
        if len(chain) >= 2 and current in property_ids:
            _emit_chain(by_id, chain, checks, invariants, found, caps, complete=True, reason="")
        if len(chain) >= caps["depth"]:
            if children.get(current) and len(found) < caps["paths"]:
                _emit_chain(
                    by_id,
                    chain,
                    checks,
                    invariants,
                    found,
                    caps,
                    complete=False,
                    reason="depth bound reached",
                )
            return
        for edge in children.get(current, []):
            if edge["callee"] in {item[0] for item in chain}:
                continue
            walk([*chain, (edge["callee"], edge["call_id"])])

    for origin in sorted(by_id):
        if len(found) >= caps["paths"]:
            break
        if children.get(origin):
            walk([(origin, "")])


def _emit_chain(
    by_id: dict[str, StateTransition],
    chain: list[tuple[str, str]],
    checks: tuple[InvariantCheck, ...],
    invariants: tuple[CandidateInvariant, ...],
    found: list[CandidatePath],
    caps: dict[str, int],
    *,
    complete: bool,
    reason: str,
) -> None:
    if len(found) >= caps["paths"]:
        return
    steps: list[PathStep] = []
    for index, (function_id, call_id) in enumerate(chain[1:], start=1):
        previous = chain[index - 1][0]
        transition = by_id.get(previous)
        if transition is None:
            return
        steps.append(
            _step(
                transition,
                "invokes",
                previous,
                function_id,
                (),
                (call_id,) if call_id else (),
            )
        )
    last = by_id.get(chain[-1][0])
    first = by_id.get(chain[0][0])
    if last is None or first is None:
        return
    relationship = "reachable"
    status = "incomplete"
    path_reason = f"truncated exploration: {reason}" if reason else "truncated exploration"
    path_id = "explore:" + ">".join(item[0] for item in chain)
    invariant_id = ""
    if complete:
        shared = _shared_declarations(chain, by_id, checks, invariants)
        violated = [
            check.invariant_id
            for check in checks
            if check.transition_id == last.transition_id and check.status == "potential"
        ]
        path_id = "chain:" + ">".join(item[0] for item in chain)
        if shared:
            relationship = "candidate_security_path"
            status = "candidate"
            path_reason = "shared declarations connect the chain to the candidate property"
            invariant_id = violated[0] if len(violated) == 1 else ""
            steps.append(_step(last, "depends-on", shared[0], last.function_id, last.writes, ()))
            if len(violated) == 1:
                steps.append(
                    _step(
                        last,
                        "violates-candidate-invariant",
                        violated[0],
                        last.function_id,
                        last.writes,
                        (),
                    )
                )
        elif _step_dependence(chain, by_id):
            relationship = "semantically_related"
            status = "reachable"
            path_reason = "semantically related state is not the candidate property"
        else:
            relationship = "reachable"
            status = "reachable"
            path_reason = (
                "reachable only: resolved calls reach a candidate but no shared state was shown"
            )
    if any(item.path_id == path_id for item in found):
        return
    found.append(
        CandidatePath(
            path_id,
            status,
            len(chain),
            tuple(steps),
            tuple(item[0] for item in chain),
            tuple(by_id[item[0]].function_id for item in chain if item[0] in by_id),
            tuple(by_id[item[0]].contract for item in chain if item[0] in by_id),
            tuple(item[1] for item in chain if item[1]),
            last.writes,
            first.conditions,
            last.assumptions,
            f"depth={len(chain)}; cap<={caps['depth']}",
            "complete" if complete else "partial",
            "" if complete else path_reason,
            path_reason,
            invariant_id,
            relationship,
        )
    )


def _resolved_edges(
    programs: Sequence[SemanticProgram], transitions: tuple[StateTransition, ...]
) -> list[dict[str, str]]:
    """Caller and callee are source-aware transition ids, not bare function identities."""
    edges: list[dict[str, str]] = []
    for program in programs:
        source = _source_identity(program)
        for function in program.functions:
            caller = _transition_key(transitions, source, function.identity)
            if not caller:
                continue
            for site in sorted(function.call_sites, key=lambda item: item.span.start_byte):
                resolution, callee_source = _resolve_in_program(programs, program, function, site)
                if resolution.status != "resolved" or not resolution.function_id:
                    continue
                callee = _transition_key(transitions, callee_source, resolution.function_id)
                if not callee:
                    continue
                edges.append(
                    {
                        "caller": caller,
                        "callee": callee,
                        "call_id": site.call_id,
                        "status": resolution.status,
                    }
                )
    edges.sort(key=lambda item: (item["caller"], item["call_id"], item["callee"]))
    return edges


def _transition_key(
    transitions: Sequence[StateTransition], source_identity: str, function_identity: str
) -> str:
    scoped = [
        item.transition_id
        for item in transitions
        if item.function_id == function_identity and item.source_identity == source_identity
    ]
    if len(scoped) == 1:
        return scoped[0]
    unique = [item.transition_id for item in transitions if item.function_id == function_identity]
    if len(unique) == 1 and (not source_identity or unique):
        only = next(item for item in transitions if item.transition_id == unique[0])
        if not source_identity or only.source_identity == source_identity:
            return unique[0]
    return ""


def _resolve_in_program(
    programs: Sequence[SemanticProgram],
    home: SemanticProgram,
    function: SemanticFunction,
    site: object,
) -> tuple[CallResolution, str]:
    """Resolve a call from the caller's program. A colliding name in another source loses."""
    home_source = _source_identity(home)
    resolution = resolve_call(home, function, site)
    if resolution.status == "resolved" and resolution.function_id:
        if any(item.identity == resolution.function_id for item in home.functions):
            return resolution, home_source
    target = str(getattr(site, "target", "") or "").strip()
    callee = str(getattr(site, "callee", "") or "")
    if not target:
        return resolution, ""
    facts = [fact for program in programs for fact in program.contracts if fact.name == target]
    if any(fact.kind == "interface" for fact in facts):
        return CallResolution("unknown", reason="interface call is not an implementation"), ""
    named = [
        (program, item)
        for program in programs
        for item in program.functions
        if item.contract == target and item.name == callee
    ]
    if len(named) == 1:
        program, item = named[0]
        return (
            CallResolution("resolved", item.identity, "contract-typed"),
            _source_identity(program),
        )
    if len(named) > 1:
        return CallResolution("unknown", reason="external callee is ambiguous"), ""
    return resolution, ""


def _resolve_site(
    programs: Sequence[SemanticProgram], function: SemanticFunction, site: object
) -> CallResolution:
    home = _program_containing(programs, function.identity)
    resolution = (
        resolve_call(home, function, site)
        if home is not None
        else CallResolution("unknown", reason="function program was not found")
    )
    if resolution.status == "resolved":
        return resolution
    target = str(getattr(site, "target", "") or "").strip()
    callee = str(getattr(site, "callee", "") or "")
    if not target:
        return resolution
    facts = [fact for program in programs for fact in program.contracts if fact.name == target]
    if any(fact.kind == "interface" for fact in facts):
        return CallResolution("unknown", reason="interface call is not an implementation")
    named = [
        item
        for program in programs
        for item in program.functions
        if item.contract == target and item.name == callee
    ]
    if len(named) == 1:
        return CallResolution("resolved", named[0].identity, "contract-typed")
    if len(named) > 1:
        return CallResolution("unknown", reason="external callee is ambiguous")
    return resolution


def _path(
    path_id: str,
    transition: StateTransition,
    steps: tuple[PathStep, ...],
    reason: str,
    *,
    depth: int,
    invariant_id: str = "",
    assumptions: tuple[str, ...] = (),
    cap: int = MAX_DEPTH,
) -> CandidatePath:
    return CandidatePath(
        path_id,
        "candidate",
        depth,
        steps,
        (transition.transition_id,),
        (transition.function_id,),
        (transition.contract,),
        tuple(dict.fromkeys(call for step in steps for call in step.call_ids)),
        tuple(dict.fromkeys(op for step in steps for op in step.operation_ids)),
        transition.conditions,
        tuple(dict.fromkeys((*transition.assumptions, *assumptions))),
        f"depth=1; cap<={cap}",
        "complete",
        "",
        reason,
        invariant_id,
        "intra-function",
    )


def _step(
    transition: StateTransition,
    relation: str,
    source: str,
    sink: str,
    operations: tuple[str, ...],
    calls: tuple[str, ...],
) -> PathStep:
    return PathStep(
        f"{transition.transition_id}:{relation}:{source}:{sink}",
        transition.transition_id,
        transition.function_id,
        transition.contract,
        relation,
        source,
        sink,
        calls,
        operations,
    )


def _direction(function: SemanticFunction, access: StateAccess) -> str:
    text = _statement_for(function, access.span.start_byte)
    path = access.path
    increased = (
        f"{path} +=",
        f"{path}=",
    )
    if "+=" in text and path in text.split("+=", 1)[0]:
        return "increase"
    if "-=" in text and path in text.split("-=", 1)[0]:
        return "decrease"
    if f"{path} = {path} +" in text or f"{path}={path}+" in text.replace(" ", ""):
        return "increase"
    if f"{path} = {path} -" in text or f"{path}={path}-" in text.replace(" ", ""):
        return "decrease"
    del increased
    return "unknown"


def _statement_for(function: SemanticFunction, start_byte: int) -> str:
    source = function.source or ""
    cfg, ranges = node_ranges(source)
    base = function.span[0] if function.span else 0
    node_id = node_at(ranges, start_byte - base)
    if node_id is None:
        return ""
    for node in cfg.nodes:
        if node.node_id == node_id:
            return node.text
    return ""


def _identity_collisions(programs: Sequence[SemanticProgram]) -> set[str]:
    """The same contract/function/line in two program objects is not one function."""
    owners: dict[str, int] = {}
    collided: set[str] = set()
    for program in programs:
        marker = id(program)
        for function in program.functions:
            previous = owners.get(function.identity)
            if previous is None:
                owners[function.identity] = marker
                continue
            if previous != marker:
                collided.add(function.identity)
    return collided


def _operations_for(
    programs: Sequence[SemanticProgram],
    source_file: str,
    contract: str,
    declarations: set[str],
) -> tuple[str, ...]:
    found: list[str] = []
    for program in programs:
        if source_file and program.file != source_file:
            continue
        for function in program.functions:
            if function.contract != contract:
                continue
            for access in function.access_sites:
                if access.declaration_id in declarations:
                    found.append(access.operation_id)
    return tuple(dict.fromkeys(found))


def _step_dependence(chain: list[tuple[str, str]], by_id: dict[str, StateTransition]) -> bool:
    """True when an earlier step's declarations meet a later step. A call edge is not enough."""
    seen: set[str] = set()
    related = False
    for function_id, _call in chain:
        transition = by_id.get(function_id)
        if transition is None:
            continue
        declarations = {
            _operation_declaration(item) for item in (*transition.reads, *transition.writes)
        }
        declarations.discard("")
        if seen & declarations:
            related = True
        seen.update(declarations)
    return related


def _shared_declarations(
    chain: list[tuple[str, str]],
    by_id: dict[str, StateTransition],
    checks: tuple[InvariantCheck, ...],
    invariants: tuple[CandidateInvariant, ...],
) -> tuple[str, ...]:
    last = by_id.get(chain[-1][0])
    if last is None:
        return ()
    wanted = {
        check.invariant_id
        for check in checks
        if check.transition_id == last.transition_id and check.status == "potential"
    }
    property_decls: set[str] = set()
    for invariant in invariants:
        if invariant.invariant_id in wanted:
            property_decls.update(invariant.declarations)
    if not property_decls:
        return ()
    earlier: set[str] = set()
    for function_id, _call in chain[:-1]:
        transition = by_id.get(function_id)
        if transition is None:
            continue
        earlier.update(
            _operation_declaration(item) for item in (*transition.reads, *transition.writes)
        )
    earlier.discard("")
    return tuple(sorted(earlier & property_decls))


def _operation_declaration(operation_id: str) -> str:
    parts = operation_id.rsplit(":", 2)
    if len(parts) < 3:
        return ""
    return parts[0]


def _repayment_reaches(summary: FunctionSummary, write: StateAccess) -> bool:
    value_ids = [
        value.identity
        for value in summary.values
        if value.provenance == "msg.value" or value.kind == "msg.value"
    ]
    return any(_forward(summary.edges, value_id, write.operation_id) for value_id in value_ids)


def _loop_unmodeled(function: SemanticFunction) -> bool:
    return bool(re.search(r"\b(for|while|do)\b", function.source or ""))


def asset_movements(function: SemanticFunction) -> tuple[AssetMovement, ...]:
    """Direction relative to the executing contract. An outbound value call is not an inflow."""
    source = strip_comments(
        function.source or "",
        line_comment="//",
        block_comment=("/*", "*/"),
    )
    found: list[AssetMovement] = []
    received = re.sub(r"\{[^{}]*\bvalue\s*:[^{}]*\}", " ", source)
    if "msg.value" in received:
        found.append(AssetMovement("inbound-eth", "msg.value", "call-value"))
    for site in function.call_sites:
        if site.value or site.call_type in {"send", "transfer"}:
            found.append(
                AssetMovement(
                    "outbound-eth",
                    site.value or site.call_type,
                    "call-site",
                )
            )
    for call in _token_calls(source, function.name, {}, {}, {}, set()):
        if call.method not in {"transfer", "transferFrom", "safeTransfer", "safeTransferFrom"}:
            continue
        if call.direction == "in":
            found.append(AssetMovement("inbound-token", call.text, "phase31-token-call"))
        elif call.direction == "out":
            found.append(AssetMovement("outbound-token", call.text, "phase31-token-call"))
        else:
            found.append(AssetMovement("unknown", call.text, "token-direction-unknown"))
    classified = {site.call_id for site in function.call_sites if site.call_type == "token"}
    for site in function.call_sites:
        if site.call_id in classified or site.value or site.call_type in {"send", "transfer"}:
            continue
        if not site.external or site.call_type in {"internal", "this", "inherited"}:
            continue
        found.append(AssetMovement("unknown", site.call_id or site.callee, "external-call"))
    return tuple(found)


def _has_inflow(function: SemanticFunction) -> bool:
    return any(item.kind in {"inbound-eth", "inbound-token"} for item in asset_movements(function))


def _declared_roles(program: SemanticProgram, contract: str = "") -> dict[str, str]:
    return {
        item.symbol: role
        for item in program.declarations
        if (not contract or item.contract == contract) and (role := _role(item.symbol))
    }


def _role(symbol: str) -> str:
    return _ROLES.get(symbol.lower(), "")


def _relation_matches(relation: str, category: str) -> bool:
    pairs = {
        "supply-increased-without-asset-inflow": {"asset-share"},
        "supply-increased-without-balance-update": {"equality"},
        "debt-reduced-without-repayment": {"debt"},
        "allowance-not-consumed": {"allowance"},
    }
    return category in pairs.get(relation, set())


def _matching_invariant(
    invariants: tuple[CandidateInvariant, ...],
    checks: tuple[InvariantCheck, ...],
    transition_id: str,
    relation: str,
) -> str:
    by_id = {item.invariant_id: item for item in invariants}
    found: list[str] = []
    for check in checks:
        invariant = by_id.get(check.invariant_id)
        if invariant is None or check.transition_id != transition_id:
            continue
        if check.status == "potential" and _relation_matches(relation, invariant.category):
            found.append(check.invariant_id)
    if len(found) == 1:
        return found[0]
    return ""


def _forward(edges: tuple[DependencyEdge, ...], source: str, sink: str) -> bool:
    seen: set[str] = set()
    stack = [source]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == sink and current != source:
            return True
        stack.extend(edge.sink for edge in edges if edge.source == current)
    return False


def _span(item: StateAccess) -> tuple[int, int, int, int]:
    return (item.span.start_byte, item.span.end_byte, item.span.start_line, item.span.end_line)


def _function(program: SemanticProgram, function_id: str) -> SemanticFunction | None:
    exact = [item for item in program.functions if item.identity == function_id]
    if len(exact) == 1:
        return exact[0]
    named = [item for item in program.functions if item.name == function_id]
    if len(named) == 1:
        return named[0]
    return None


def _source_identity(program: SemanticProgram) -> str:
    path = Path(program.file)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        text = "\n".join(function.source for function in program.functions)
    return hashlib.sha256(f"{program.file}\n{text}".encode()).hexdigest()[:16]


def _function_for_transition(
    programs: Sequence[SemanticProgram], transition: StateTransition
) -> SemanticFunction | None:
    found = [
        function
        for program in programs
        if _source_identity(program) == transition.source_identity
        for function in program.functions
        if function.identity == transition.function_id
    ]
    if len(found) == 1:
        return found[0]
    return None


def _program_for_transition(
    programs: Sequence[SemanticProgram], transition: StateTransition
) -> SemanticProgram | None:
    found = [
        program
        for program in programs
        if _source_identity(program) == transition.source_identity
        and any(function.identity == transition.function_id for function in program.functions)
    ]
    if len(found) == 1:
        return found[0]
    return None


def _function_in(programs: Sequence[SemanticProgram], function_id: str) -> SemanticFunction | None:
    exact = [
        item for program in programs for item in program.functions if item.identity == function_id
    ]
    if len(exact) == 1:
        return exact[0]
    if len(programs) == 1:
        return _function(programs[0], function_id)
    return None


def _program_containing(
    programs: Sequence[SemanticProgram], identity: str
) -> SemanticProgram | None:
    matches = [
        program
        for program in programs
        if any(item.identity == identity for item in program.functions)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _compiler_version(programs: Sequence[SemanticProgram]) -> str:
    versions = [item.compiler_version for item in programs if item.compiler_version]
    unique = tuple(dict.fromkeys(versions))
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _join_reason(left: str, right: str) -> str:
    parts = [part for part in (left, right) if part]
    return "; ".join(dict.fromkeys(parts))


def _reject_forbidden(model: TransitionModel) -> None:
    for path in model.paths:
        if path.status in _FORBIDDEN:
            raise RuntimeError(path.status)
    for check in model.checks:
        if check.status in _FORBIDDEN:
            raise RuntimeError(check.status)
    for transition in model.transitions:
        if transition.status in _FORBIDDEN:
            raise RuntimeError(transition.status)

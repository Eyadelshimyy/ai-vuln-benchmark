"""Bounded Solidity dataflow over the semantic IR.

Unknown provenance propagates. An incomplete model is not a proof of safety.
This is deterministic program analysis, not a planner and not verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsing.solidity_ir import SemanticCall, SemanticFunction, SemanticProgram

MAX_FUNCTIONS = 200
MAX_BLOCKS = 80
MAX_DEPTH = 4
MAX_ITERATIONS = 3
MAX_VALUES = 400
MAX_EDGES = 800


@dataclass(frozen=True)
class FlowValue:
    identity: str
    kind: str
    provenance: str
    dependencies: tuple[str, ...]
    type_name: str
    origin: str
    span: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    sink: str
    kind: str


@dataclass(frozen=True)
class AssetFlow:
    function: str
    source: str
    destination: str
    amount: str
    dependencies: tuple[str, ...]


@dataclass
class FunctionSummary:
    identity: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    calls: tuple[str, ...]
    values: tuple[FlowValue, ...]
    edges: tuple[DependencyEdge, ...]
    authorization: str
    asset_flows: tuple[AssetFlow, ...]
    cyclic: bool = False
    incomplete: bool = False
    incomplete_reason: str = ""
    ordered_reads: tuple[tuple[int, str], ...] = ()
    ordered_writes: tuple[tuple[int, str], ...] = ()
    ordered_calls: tuple[tuple[int, str, str], ...] = ()
    may_read_before: tuple[tuple[str, str], ...] = ()
    may_write_after: tuple[tuple[str, str], ...] = ()
    may_read_ops: tuple[tuple[str, str], ...] = ()
    may_write_ops: tuple[tuple[str, str], ...] = ()
    operations: tuple[tuple[str, str, str, int], ...] = ()
    placement_gaps: int = 0
    cfg_iterations: int = 0


@dataclass
class DataflowModel:
    status: str
    summaries: dict[str, FunctionSummary] = field(default_factory=dict)
    values: dict[str, FlowValue] = field(default_factory=dict)
    edges: tuple[DependencyEdge, ...] = ()
    incomplete_reason: str = ""
    profile: str = "normal"
    iterations: int = 0

    def origin(self, value_id: str) -> str:
        value = self.values.get(value_id)
        return value.provenance if value else "unknown"

    def dependencies(self, value_id: str) -> tuple[str, ...]:
        value = self.values.get(value_id)
        return value.dependencies if value else ()

    def value_reaches(self, value_id: str, sink: str) -> bool:
        if value_id == sink:
            return True
        seen: set[str] = set()
        stack = [value_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current == sink:
                return True
            stack.extend(edge.sink for edge in self.edges if edge.source == current)
        return False

    def reads_before(self, function: str, call: str) -> tuple[str, ...]:
        return self.reads_before_call(function, call)

    def writes_after(self, function: str, call: str) -> tuple[str, ...]:
        return self.writes_after_call(function, call)

    def reads_before_call(self, function: str, call_id: str) -> tuple[str, ...]:
        summary = self._summary(function)
        if summary is None or not call_id:
            return ()
        return tuple(path for call, path in summary.may_read_before if call == call_id)

    def writes_after_call(self, function: str, call_id: str) -> tuple[str, ...]:
        summary = self._summary(function)
        if summary is None or not call_id:
            return ()
        return tuple(path for call, path in summary.may_write_after if call == call_id)

    def target_provenance(self, function: str, call_id: str = "") -> str:
        summary = self._summary(function)
        if summary is None:
            return "unknown"
        matches = [
            value
            for value in summary.values
            if value.kind == "delegatecall-target"
            and (not call_id or value.identity.endswith(":" + call_id))
        ]
        if len(matches) == 1:
            return matches[0].provenance
        return "unknown"

    def _call_order(self, summary: FunctionSummary | None, call_id: str) -> int | None:
        if summary is None or not call_id:
            return None
        found = [
            index for index, callee, ident in summary.ordered_calls if call_id in {callee, ident}
        ]
        if len(found) != 1:
            return None
        return found[0]

    def authorization_path(self, function: str) -> str:
        summary = self._summary(function)
        return summary.authorization if summary else "unknown"

    def asset_flow(self, function: str) -> tuple[AssetFlow, ...]:
        summary = self._summary(function)
        return summary.asset_flows if summary else ()

    def state_dependency(self, function: str) -> tuple[DependencyEdge, ...]:
        summary = self._summary(function)
        if summary is None:
            return ()
        return tuple(
            edge for edge in summary.edges if "state" in edge.kind or "calldata" in edge.kind
        )

    def reachable_functions(self, function: str) -> tuple[str, ...]:
        summary = self._summary(function)
        if summary is None:
            return ()
        return summary.calls

    def possible_callbacks(self, function: str) -> tuple[str, ...]:
        summary = self._summary(function)
        if summary is None:
            return ()
        return tuple(
            value.identity
            for value in summary.values
            if value.kind in {"external-return", "callback"}
        )

    def _summary(self, function: str) -> FunctionSummary | None:
        if function in self.summaries:
            return self.summaries[function]
        exact = [item for item in self.summaries.values() if item.identity == function]
        if len(exact) == 1:
            return exact[0]
        named = [
            item
            for item in self.summaries.values()
            if item.identity.split(":")[0].rsplit(".", 1)[-1] == function
        ]
        if len(named) == 1:
            return named[0]
        return None


def analyze_dataflow(program: SemanticProgram, *, profile: str = "normal") -> DataflowModel:
    limits = _limits(profile)
    model = DataflowModel(status="partial", profile=profile)
    if program.status == "unavailable" and not program.functions:
        model.status = "unavailable"
        model.incomplete_reason = "no semantic functions"
        return model
    reasons: list[str] = []
    if program.incomplete_reason:
        reasons.append(program.incomplete_reason)
    stack: list[str] = []
    for function in program.functions[: limits["functions"]]:
        _summarize(program, function, model, stack, limits, 0)
    if len(program.functions) > limits["functions"]:
        reasons.append("dataflow function limit reached")
    if len(model.values) >= limits["values"]:
        reasons.append("semantic node limit reached")
    if len(model.edges) >= limits["edges"]:
        reasons.append("dependency edge limit reached")
    model.incomplete_reason = "; ".join(dict.fromkeys(reasons))
    model.status = "partial" if model.incomplete_reason else "available"
    return model


def _limits(profile: str) -> dict[str, int]:
    if profile == "fast":
        return {
            "functions": 40,
            "blocks": 20,
            "depth": 2,
            "iterations": 1,
            "values": 120,
            "edges": 200,
        }
    if profile == "deep":
        return {
            "functions": MAX_FUNCTIONS,
            "blocks": MAX_BLOCKS * 2,
            "depth": MAX_DEPTH + 2,
            "iterations": MAX_ITERATIONS + 1,
            "values": MAX_VALUES * 2,
            "edges": MAX_EDGES * 2,
        }
    return {
        "functions": MAX_FUNCTIONS,
        "blocks": MAX_BLOCKS,
        "depth": MAX_DEPTH,
        "iterations": MAX_ITERATIONS,
        "values": MAX_VALUES,
        "edges": MAX_EDGES,
    }


def _summarize(
    program: SemanticProgram,
    function: SemanticFunction,
    model: DataflowModel,
    stack: list[str],
    limits: dict[str, int],
    depth: int,
) -> FunctionSummary:
    from app.parsing.solidity_value_flow import (
        external_return,
        intraprocedural,
        resolve_call,
        returned_reads,
    )

    if function.identity in model.summaries and function.identity not in stack:
        return model.summaries[function.identity]
    cyclic = function.identity in stack
    incomplete = depth > limits["depth"] or cyclic
    reason = ""
    if cyclic:
        reason = "cyclic call"
    elif depth > limits["depth"]:
        reason = "call depth limit reached"
    values: list[FlowValue] = []
    edges: list[DependencyEdge] = []
    cfg_iterations = 0
    may_read: list[tuple[str, str]] = []
    may_write: list[tuple[str, str]] = []
    read_ops: list[tuple[str, str]] = []
    write_ops: list[tuple[str, str]] = []
    gaps = 0
    if not incomplete:
        (
            values,
            edges,
            may_read_before,
            may_write_after,
            may_read_ops,
            may_write_ops,
            cfg_iterations,
            gaps,
        ) = intraprocedural(program, function, limits)
        may_read = list(may_read_before)
        may_write = list(may_write_after)
        read_ops = list(may_read_ops)
        write_ops = list(may_write_ops)
        if cfg_iterations < 0:
            incomplete = True
            reason = reason or "CFG block limit reached"
        elif cfg_iterations >= limits["iterations"] and _has_loop(function):
            incomplete = True
            reason = reason or "fixed-point iteration limit reached"
    ordered_reads = tuple(
        (item.span.start_byte, item.path) for item in function.access_sites if item.kind != "write"
    )
    ordered_writes = tuple(
        (item.span.start_byte, item.path) for item in function.access_sites if item.kind != "read"
    )
    ordered_calls = tuple(
        (site.span.start_byte, site.callee, site.call_id) for site in function.call_sites
    )
    for site in function.call_sites:
        if site.external:
            returned = external_return(function, site)
            if returned.identity not in {item.identity for item in values}:
                values.append(returned)
            if site.callback_potential:
                callback_id = f"{function.identity}:callback:{site.call_id}"
                if callback_id not in {item.identity for item in values}:
                    values.append(
                        FlowValue(
                            callback_id,
                            "callback",
                            "external",
                            (returned.identity,),
                            "",
                            function.identity,
                            returned.span,
                        )
                    )
            if site.call_type == "delegatecall":
                provenance = _delegate_provenance(program, function, site.target)
                values.append(
                    FlowValue(
                        f"{function.identity}:delegate-target:{site.call_id}",
                        "delegatecall-target",
                        provenance,
                        (site.target,),
                        "",
                        function.identity,
                        returned.span,
                    )
                )
        elif depth < limits["depth"]:
            resolution = resolve_call(program, function, site)
            callee = (
                _function_by_id(program, resolution.function_id)
                if resolution.status == "resolved" and resolution.function_id
                else None
            )
            if callee is None:
                continue
            stack.append(function.identity)
            child = _summarize(program, callee, model, stack, limits, depth + 1)
            stack.pop()
            _anchor(values, function)
            _anchor(values, callee)
            edges.append(DependencyEdge(function.identity, child.identity, "internal-call"))
            ret_id = f"{function.identity}:internal-return:{site.call_id}"
            if any(item.identity == ret_id for item in values):
                for read_id in returned_reads(callee):
                    edges.append(DependencyEdge(read_id, ret_id, "return-value"))
                _inherit_reads(function, callee, site, read_ops, may_read)
            if child.cyclic or child.incomplete:
                incomplete = True
                reason = child.incomplete_reason or reason
    assets = _assets(function, values)
    for value in values:
        if len(model.values) >= limits["values"]:
            incomplete = True
            reason = reason or "semantic node limit reached"
            break
        model.values[value.identity] = value
    idents = {item.identity for item in values} | set(model.values)
    edges = [edge for edge in edges if edge.source in idents and edge.sink in idents]
    if len(edges) > limits["edges"]:
        edges = edges[: limits["edges"]]
        incomplete = True
        reason = reason or "dependency edge limit reached"
    summary = FunctionSummary(
        function.identity,
        function.reads,
        function.writes,
        tuple(site.callee for site in function.call_sites),
        tuple(values),
        tuple(edges),
        function.authorization,
        tuple(assets),
        cyclic=cyclic,
        incomplete=incomplete or bool(program.incomplete_reason),
        incomplete_reason=reason or program.incomplete_reason,
        ordered_reads=ordered_reads,
        ordered_writes=ordered_writes,
        ordered_calls=ordered_calls,
        may_read_before=tuple(may_read),
        may_write_after=tuple(may_write),
        may_read_ops=tuple(read_ops),
        may_write_ops=tuple(write_ops),
        operations=tuple(
            (item.operation_id, item.path, item.kind, item.span.start_byte)
            for item in function.access_sites
        ),
        placement_gaps=gaps,
        cfg_iterations=cfg_iterations,
    )
    model.summaries[function.identity] = summary
    model.edges = tuple([*model.edges, *edges])[: limits["edges"]]
    return summary


def _anchor(values: list[FlowValue], function: SemanticFunction) -> None:
    if any(item.identity == function.identity for item in values):
        return
    values.append(FlowValue(function.identity, "function", "unknown", (), "", function.identity))


def _function_by_id(program: SemanticProgram, identity: str) -> SemanticFunction | None:
    for item in program.functions:
        if item.identity == identity:
            return item
    return None


def _inherit_reads(
    function: SemanticFunction,
    callee: SemanticFunction,
    site: SemanticCall,
    read_ops: list[tuple[str, str]],
    may_read: list[tuple[str, str]],
) -> None:
    from app.parsing.solidity_cfg import node_at, node_ranges
    from app.parsing.solidity_value_flow import returned_reads

    cfg, ranges = node_ranges(function.source or "")
    if not cfg.known:
        return
    base = function.span[0] if function.span else 0
    internal_at = node_at(ranges, site.span.start_byte - base)
    if internal_at is None:
        return
    read_ids = returned_reads(callee)
    if not read_ids:
        return
    paths = {
        item.operation_id: item.path
        for item in callee.access_sites
        if item.operation_id in set(read_ids)
    }
    for other in function.call_sites:
        if not other.external or other.call_id == site.call_id:
            continue
        call_at = node_at(ranges, other.span.start_byte - base)
        if call_at is None or call_at == internal_at:
            continue
        if call_at not in cfg.reachable_from(internal_at):
            continue
        for read_id in read_ids:
            read_ops.append((other.call_id, read_id))
            may_read.append((other.call_id, paths.get(read_id, "")))


def _has_loop(function: SemanticFunction) -> bool:
    return bool(re.search(r"\b(for|while|do)\b", function.source))


def _delegate_provenance(program: SemanticProgram, function: SemanticFunction, target: str) -> str:
    base = re.match(r"([A-Za-z_]\w*)", target.strip())
    name = base.group(1) if base else ""
    if "?" in target:
        return "unknown"
    if target.strip().endswith(")"):
        callee = _resolve_internal(program, function, name) if name else None
        returned = re.search(r"return\s+([A-Za-z_]\w*)", callee.source) if callee else None
        if (
            callee is not None
            and returned
            and returned.group(1) in set(callee.reads) | set(callee.writes)
        ):
            return "state"
        return "derived" if callee else "unknown"
    if name in function.parameters or name in {"msg", "tx"}:
        return "attacker" if name != "msg" else "attacker"
    if name in function.reads or name in function.writes:
        return "state"
    if re.fullmatch(r"0x[0-9a-fA-F]+", target.strip()):
        return "trusted_constant"
    return "unknown"


def _resolve_internal(
    program: SemanticProgram, function: SemanticFunction, name: str
) -> SemanticFunction | None:
    matches = [
        item
        for item in program.functions
        if item.name == name
        and item.contract == function.contract
        and item.identity != function.identity
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _assets(function: SemanticFunction, values: list[FlowValue]) -> list[AssetFlow]:
    flows: list[AssetFlow] = []
    for site in function.call_sites:
        if site.call_type not in {"transfer", "token", "low-level-call", "send"} and not site.value:
            continue
        source = "contract"
        destination = "external" if site.external else "internal"
        if site.value:
            source = "contract"
            destination = site.target or "external"
        relevant = f"{site.target} {site.value}"
        flows.append(
            AssetFlow(
                function.identity,
                source,
                destination,
                site.value or site.arguments[:80],
                tuple(
                    value.identity
                    for value in values
                    if value.identity in relevant
                    or any(part and part in relevant for part in value.identity.split(":"))
                )[:8],
            )
        )
    return flows

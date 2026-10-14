"""Project-level security properties over semantic dataflow.

Results are static observations. They do not verify findings. Unknown and
incomplete stay unknown. A property that finds nothing is not proof of safety.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.parsing.solidity_dataflow import DataflowModel, analyze_dataflow
from app.parsing.solidity_ir import SemanticFunction, SemanticProgram


@dataclass(frozen=True)
class PropertyResult:
    status: str
    confidence: str
    summary: str
    provenance: str = "unknown"
    spans: tuple[tuple[int, int], ...] = ()
    dependencies: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    incomplete_reason: str = ""
    function_ids: tuple[str, ...] = ()
    call_ids: tuple[str, ...] = ()


def analyze_reentrancy(
    program: SemanticProgram, function_id: str, flow: DataflowModel | None = None
) -> PropertyResult:
    flow = flow or analyze_dataflow(program)
    summary = flow._summary(function_id)
    if summary is None:
        return PropertyResult(
            "unknown", "low", "function was not resolved", function_ids=(function_id,)
        )
    if summary.incomplete:
        return PropertyResult(
            "incomplete",
            "low",
            "reentrancy analysis is incomplete",
            incomplete_reason=summary.incomplete_reason,
            function_ids=(summary.identity,),
        )
    hits = _dependent_reentrancy(flow, summary)
    if not hits:
        if getattr(summary, "placement_gaps", 0):
            return PropertyResult(
                "incomplete",
                "low",
                "CFG placement could not locate every state operation",
                incomplete_reason="CFG placement incomplete",
                function_ids=(summary.identity,),
            )
        return PropertyResult(
            "unknown",
            "low",
            "no state read was shown to reach a later write through a call",
            function_ids=(summary.identity,),
        )
    call, read_id, write_id = hits[0]
    return PropertyResult(
        "potential",
        "medium",
        "a state read reaches a write after an external call",
        provenance="state",
        dependencies=(read_id, write_id),
        function_ids=(summary.identity,),
        call_ids=(call,),
    )


def reentrancy_dependencies(
    program: SemanticProgram, function_id: str, flow: DataflowModel | None = None
) -> tuple[tuple[str, str, str], ...]:
    """Call id, read operation, and the write operation that read reaches."""
    flow = flow or analyze_dataflow(program)
    summary = flow._summary(function_id)
    if summary is None or summary.incomplete:
        return ()
    return tuple(_dependent_reentrancy(flow, summary))


def analyze_authorization(program: SemanticProgram, function_id: str) -> PropertyResult:
    function = _one(program, function_id)
    if function is None:
        return PropertyResult("unknown", "low", "function was not resolved")
    writes = [item for item in function.access_sites if item.kind != "read"]
    if not writes:
        return PropertyResult(
            "unknown", "low", "no sensitive store was identified", function_ids=(function.identity,)
        )
    statuses = {item.guard_status for item in writes}
    if statuses == {"guarded"}:
        status = "satisfied"
    elif "unguarded" in statuses:
        status = "potential"
    else:
        status = "unknown"
    return PropertyResult(
        status,
        "medium" if status == "potential" else "low",
        f"authorization is {function.authorization}",
        provenance="authority",
        function_ids=(function.identity,),
    )


def analyze_delegatecall(
    program: SemanticProgram,
    function_id: str,
    call_id: str = "",
    flow: DataflowModel | None = None,
) -> PropertyResult:
    flow = flow or analyze_dataflow(program)
    summary = flow._summary(function_id)
    if summary is None:
        return PropertyResult("unknown", "low", "function was not resolved", provenance="unknown")
    prefix = f"{summary.identity}:delegate-target:"
    matches = [value for value in summary.values if value.identity.startswith(prefix)]
    if call_id:
        matches = [value for value in matches if value.identity[len(prefix) :] == call_id]
        if not matches:
            return PropertyResult(
                "unknown",
                "low",
                "delegatecall call was not found",
                provenance="unknown",
                function_ids=(summary.identity,),
                call_ids=(call_id,),
            )
    elif len(matches) > 1:
        return PropertyResult(
            "unknown",
            "low",
            "call id is required when several delegatecalls exist",
            provenance="unknown",
            unresolved=tuple(value.identity[len(prefix) :] for value in matches),
            function_ids=(summary.identity,),
            call_ids=tuple(value.identity[len(prefix) :] for value in matches),
        )
    if len(matches) != 1:
        return PropertyResult(
            "unknown",
            "low",
            "delegatecall target was not resolved",
            provenance="unknown",
            function_ids=(summary.identity,),
        )
    value = matches[0]
    exact_call = value.identity[len(prefix) :]
    provenance = value.provenance
    status = "potential" if provenance == "attacker" else "unknown"
    return PropertyResult(
        status,
        "medium" if status == "potential" else "low",
        f"delegatecall target provenance is {provenance}",
        provenance=provenance,
        spans=((value.span[0], value.span[1]),) if value.span[1] else (),
        dependencies=value.dependencies,
        function_ids=(summary.identity,),
        call_ids=(exact_call,),
    )


def analyze_operation_authorization(
    program: SemanticProgram, function_id: str, operation_id: str
) -> PropertyResult:
    """Authorization for one state write. A function-wide status is not used."""
    function = _one(program, function_id)
    if function is None:
        return PropertyResult("unknown", "low", "function was not resolved")
    matches = [item for item in function.access_sites if item.operation_id == operation_id]
    if len(matches) != 1:
        return PropertyResult(
            "unknown",
            "low",
            "state operation was not found",
            function_ids=(function.identity,),
        )
    item = matches[0]
    if item.kind == "read":
        return PropertyResult(
            "unknown",
            "low",
            "reads are not authorization subjects",
            function_ids=(function.identity,),
        )
    if item.guard_status == "guarded" and item.guard_dominates == "yes":
        status = "satisfied"
    elif item.guard_status == "unguarded":
        status = "potential"
    else:
        status = "unknown"
    return PropertyResult(
        status,
        "medium" if status == "potential" else "low",
        item.guard_predicate or f"operation authorization is {item.guard_status}",
        provenance="authority",
        spans=((item.span.start_byte, item.span.end_byte),),
        dependencies=(item.operation_id,),
        function_ids=(function.identity,),
    )


def analyze_asset_flow(
    program: SemanticProgram, function_id: str, flow: DataflowModel | None = None
) -> PropertyResult:
    flow = flow or analyze_dataflow(program)
    summary = flow._summary(function_id)
    if summary is None or not summary.asset_flows:
        return PropertyResult("unknown", "low", "no asset flow was resolved")
    flow_item = summary.asset_flows[0]
    return PropertyResult(
        "unknown",
        "low",
        f"asset flow observed from {flow_item.source} to {flow_item.destination}",
        dependencies=flow_item.dependencies,
        function_ids=(summary.identity,),
    )


def _dependent_reentrancy(flow: DataflowModel, summary: object) -> list[tuple[str, str, str]]:
    """Pair one read operation with the write operation it actually reaches."""
    reads = getattr(summary, "may_read_ops", ())
    writes = getattr(summary, "may_write_ops", ())
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for call, write_id in writes:
        for read_call, read_id in reads:
            if read_call != call or read_id == write_id:
                continue
            key = (call, read_id, write_id)
            if key in seen:
                continue
            if _reaches(flow, read_id, {write_id}):
                seen.add(key)
                found.append(key)
    return found


def _reaches(flow: DataflowModel, source: str, sinks: set[str]) -> bool:
    seen: set[str] = set()
    stack = [source]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in sinks and current != source:
            return True
        stack.extend(edge.sink for edge in flow.edges if edge.source == current)
    return False


def graph_edges_are_connected(flow: DataflowModel) -> bool:
    known = set(flow.values)
    return all(edge.source in known and edge.sink in known for edge in flow.edges)


def _one(program: SemanticProgram, function_id: str) -> SemanticFunction | None:
    exact = [item for item in program.functions if item.identity == function_id]
    if len(exact) == 1:
        return exact[0]
    named = [item for item in program.functions if item.name == function_id]
    if len(named) == 1:
        return named[0]
    return None


def _identity(flow: DataflowModel, function_id: str) -> str:
    summary = flow._summary(function_id)
    return summary.identity if summary else function_id

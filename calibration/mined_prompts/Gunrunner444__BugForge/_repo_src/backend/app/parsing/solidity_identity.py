"""Canonical Solidity function identity.

Protocol-wide identity comes from the parser function object: contract, source
file, name, signature, and declaration span. A shared name is not a shared
function. Overloads stay distinct. An unresolved id is unknown.
"""

from __future__ import annotations

import re

from app.parsing.solidity_ir import SemanticFunction, SemanticProgram


def resolve_function(program: SemanticProgram | None, function_id: str) -> SemanticFunction | None:
    """Match one parser function. Name splitting is not identity."""
    if program is None or not function_id:
        return None
    matches = [item for item in program.functions if item.identity == function_id]
    if len(matches) != 1:
        return None
    return matches[0]


def canonical_function_key(
    function: SemanticFunction,
    *,
    source_file: str,
    parameter_types: tuple[str, ...] = (),
    project: str = "",
    source_snapshot: str = "",
    compiler_configuration: str = "",
) -> str:
    """Identity that keeps overloads and source files apart."""
    signature = f"{function.name}({','.join(parameter_types)})"
    span = ",".join(str(part) for part in function.span)
    return "\n".join(
        (
            project,
            source_snapshot,
            compiler_configuration,
            source_file,
            function.contract,
            signature,
            str(function.line),
            span,
        )
    )


def declared_name(function_id: str) -> tuple[str, str] | None:
    """Read ``Contract.name:line`` without treating the name as unique."""
    if not function_id or ":" not in function_id:
        return None
    head = function_id.split(":", 1)[0]
    if "." not in head:
        return None
    contract, name = head.rsplit(".", 1)
    if not contract or not name:
        return None
    return contract, name


def unique_declared_name(source: str, contract: str, name: str) -> str | None:
    """Return ``name`` only when that contract declares it once."""
    if not source or not contract or not name:
        return None
    body = _contract_body(source, contract)
    if body is None:
        return None
    count = len(re.findall(rf"function\s+{re.escape(name)}\s*\(", body))
    if count != 1:
        return None
    return name


def function_name_for(
    function_id: str,
    *,
    program: SemanticProgram | None,
    source: str,
    contract: str,
) -> str | None:
    """Resolve a name from parser identity, or from one unambiguous declaration."""
    if program is not None:
        found = resolve_function(program, function_id)
        if found is None:
            return None
        if contract and found.contract != contract:
            return None
        return found.name
    parsed = declared_name(function_id)
    if parsed is None:
        return None
    declared_contract, name = parsed
    if contract and declared_contract != contract:
        return None
    return unique_declared_name(source, declared_contract, name)


def _contract_body(source: str, contract: str) -> str | None:
    match = re.search(rf"\b(?:contract|interface|library)\s+{re.escape(contract)}\b", source)
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

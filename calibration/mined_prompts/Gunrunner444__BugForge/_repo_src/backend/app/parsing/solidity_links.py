"""Unambiguous Solidity relationships across files.

Duplicate contract names are left unresolved. BugForge does not invent an edge
when more than one definition matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.keccak import function_selector
from app.parsing.model import SyntaxGraph
from app.parsing.solidity_types import canonical_with_aliases


@dataclass(frozen=True)
class SolidityLink:
    kind: str
    source_file: str
    source_name: str
    target_file: str
    target_name: str


def relate_solidity_files(graphs: dict[str, SyntaxGraph]) -> list[SolidityLink]:
    contracts: dict[str, list[tuple[str, str]]] = {}
    structs: dict[str, list[str]] = {}
    for path, graph in graphs.items():
        if graph.language != "solidity" or graph.parser_tier.value == "profile_fallback":
            continue
        for entity in graph.entities:
            if entity.entity_type in {"contract", "interface", "library"}:
                contracts.setdefault(entity.name, []).append((path, entity.entity_type))
            elif entity.entity_type == "struct":
                structs.setdefault(entity.name, []).append(path)
    links: list[SolidityLink] = []
    for path, graph in graphs.items():
        for event in graph.events:
            if event.kind != "sol_contract":
                continue
            fields = _fields(event.extra)
            for base in fields.get("bases", "").split(","):
                base_name = base.strip().split("(")[0].strip()
                if not base_name:
                    continue
                matches = contracts.get(base_name, [])
                if len(matches) != 1:
                    continue
                target_file, _kind = matches[0]
                if target_file == path:
                    continue
                links.append(
                    SolidityLink(
                        "inherits",
                        path,
                        fields.get("kind", "contract") + ":" + event.text[:80],
                        target_file,
                        base_name,
                    )
                )
        for imported in graph.imports:
            module = imported.module.rsplit("/", 1)[-1]
            for other, other_graph in graphs.items():
                if other == path:
                    continue
                other_name = other.rsplit("/", 1)[-1]
                if module and module == other_name:
                    links.append(SolidityLink("imports", path, module, other, other_name))
                for name, definitions in structs.items():
                    if len(definitions) != 1 or definitions[0] != other:
                        continue
                    if name and re.search(rf"\b{re.escape(name)}\b", graph.source):
                        links.append(SolidityLink("struct", path, name, other, name))
    return links


def unique_abi_aliases(graphs: dict[str, SyntaxGraph]) -> dict[str, str]:
    """ABI types defined exactly once across the supplied graphs.

    A repeated name is omitted. Callers must not guess which definition applies.
    """
    found: dict[str, list[str]] = {}
    for graph in graphs.values():
        if graph.language != "solidity" or graph.parser_tier.value == "profile_fallback":
            continue
        for event in graph.events:
            if event.kind != "sol_type_def":
                continue
            fields = _fields(event.extra)
            name = fields.get("name", "")
            abi = fields.get("abi", "")
            if name and abi:
                found.setdefault(name, []).append(abi)
    return {name: values[0] for name, values in found.items() if len(values) == 1}


def overload_selectors(graph: SyntaxGraph) -> list[dict[str, str]]:
    """One selector per canonical signature. Overloads are not collapsed."""
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in graph.entities:
        if entity.entity_type != "function" or entity.name in {
            "constructor",
            "fallback",
            "receive",
        }:
            continue
        canons: list[str] = []
        resolved = True
        for param in entity.parameters:
            canon = canonical_with_aliases(param.annotation or "", {})
            if not canon:
                resolved = False
                break
            canons.append(canon)
        if not resolved:
            continue
        signature = f"{entity.name}({','.join(canons)})"
        contract = entity.parent or ""
        if (contract, signature) in seen:
            continue
        seen.add((contract, signature))
        rows.append(
            {
                "contract": contract,
                "symbol": signature,
                "selector": function_selector(signature),
            }
        )
    return rows


def selector_for_function(
    graph: SyntaxGraph, function: str, aliases: dict[str, str]
) -> tuple[str, str]:
    """Return ``(canonical params, selector)`` or ``("", "")`` when unresolved."""
    matches = [
        entity
        for entity in graph.entities
        if entity.entity_type == "function" and entity.name == function
    ]
    if len(matches) != 1:
        return "", ""
    entity = matches[0]
    canons: list[str] = []
    for param in entity.parameters:
        canon = canonical_with_aliases(param.annotation or "", aliases)
        if not canon:
            return "", ""
        canons.append(canon)
    joined = ",".join(canons)
    return joined, function_selector(f"{function}({joined})")


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields

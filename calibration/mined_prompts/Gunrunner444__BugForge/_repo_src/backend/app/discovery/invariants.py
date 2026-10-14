"""Candidate invariants derived from Solidity relationships.

A name such as totalSupply is not enough. The suggestion stays a candidate
until some engine actually executes it.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.parsing.model import SyntaxEvent, SyntaxGraph


@dataclass(frozen=True)
class InvariantCandidate:
    name: str
    statement: str
    reason: str
    status: str = "candidate"
    valid: bool = False


def suggest_invariants(graph: SyntaxGraph) -> list[InvariantCandidate]:
    if graph.language != "solidity":
        return []
    state = _state(graph)
    functions = [event for event in graph.events if event.kind == "sol_function"]
    names = {event.extra for event in functions}
    texts = [event.text for event in functions]
    blob = "\n".join(texts)
    found: list[InvariantCandidate] = []
    balance_maps = [
        name for name, info in state.items() if "mapping" in info and "balance" in name.lower()
    ]
    writes_supply = "totalSupply" in blob and any(
        name in {"transfer", "mint", "burn"} or "totalSupply" in text
        for name, text in _named(functions)
    )
    if balance_maps and "totalSupply" in state and writes_supply:
        found.append(
            InvariantCandidate(
                name="supply_conservation",
                statement="sum of balance mapping entries stays consistent with totalSupply",
                reason="A balance mapping, totalSupply, and a supply-changing function are all present.",
            )
        )
    if balance_maps and any(name == "withdraw" for name, _text in _named(functions)):
        found.append(
            InvariantCandidate(
                name="withdrawal_bound",
                statement="a withdrawal cannot exceed the recorded balance",
                reason="withdraw coexists with a balance mapping.",
            )
        )
    if any(name.lower() in {"owner", "admin"} for name in state) and _has_auth_path(graph):
        found.append(
            InvariantCandidate(
                name="owner_authorized",
                statement="privileged configuration changes go through an authorization check",
                reason="An owner or admin variable has an authorization path.",
            )
        )
    if any("function=initialize" in extra or "function=reinitialize" in extra for extra in names):
        found.append(
            InvariantCandidate(
                name="initialize_once",
                statement="initialize cannot succeed twice",
                reason="An initialize or reinitialize function is present.",
            )
        )
    if "totalAssets" in state and "totalSupply" in state and "/" in blob:
        found.append(
            InvariantCandidate(
                name="share_asset_consistency",
                statement="share and asset conversion stays consistent",
                reason="totalAssets, totalSupply, and a division are present together.",
            )
        )
    return found


def _state(graph: SyntaxGraph) -> dict[str, str]:
    found: dict[str, str] = {}
    for event in graph.events:
        if event.kind != "sol_state" or "name=" not in event.extra:
            continue
        name = event.extra.split("name=", 1)[1].split("|", 1)[0]
        found[name] = event.extra
    return found


def _named(functions: list[SyntaxEvent]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for event in functions:
        match_name = ""
        for part in event.extra.split("|"):
            if part.startswith("function="):
                match_name = part.split("=", 1)[1]
        pairs.append((match_name, event.text))
    return pairs


def _has_auth_path(graph: SyntaxGraph) -> bool:
    for event in graph.events:
        if event.kind == "sol_auth_guard":
            return True
        if event.kind == "sol_function" and any(
            token in event.extra.lower() for token in ("onlyowner", "onlyrole", "onlyadmin")
        ):
            return True
    return False

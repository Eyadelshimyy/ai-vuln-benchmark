"""Conservative modifier lookup across contracts and files.

A modifier is usable only when one definition is reached through the contract
or its bases. Duplicate names are ambiguous and are not guessed.
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass

from app.parsing.model import SyntaxGraph

_INDEX: ContextVar[ModifierIndex | None] = ContextVar("bugforge_solidity_modifiers", default=None)


@dataclass(frozen=True)
class ModifierBody:
    file_path: str
    contract: str
    name: str
    body: str


@dataclass(frozen=True)
class ModifierResolution:
    status: str
    body: str
    contract: str
    file_path: str


class ModifierIndex:
    """Modifiers and inheritance edges collected from syntax graphs."""

    def __init__(self) -> None:
        self._modifiers: dict[tuple[str, str, str], list[ModifierBody]] = {}
        self._bases: dict[tuple[str, str], list[str]] = {}
        self._contracts: dict[str, list[tuple[str, str]]] = {}
        self._graphs: dict[str, SyntaxGraph] = {}

    @classmethod
    def from_graphs(cls, graphs: dict[str, SyntaxGraph]) -> ModifierIndex:
        index = cls()
        for path, graph in graphs.items():
            if graph.language != "solidity" or graph.parser_tier.value == "profile_fallback":
                continue
            index._absorb(path, graph)
        return index

    def resolve(self, graph: SyntaxGraph, contract: str, name: str) -> ModifierResolution:
        file_path = graph.file_path
        local = self._modifiers.get((file_path, contract, name), [])
        if len(local) > 1:
            return _ambiguous(contract, file_path)
        if len(local) == 1:
            return _finish(local[0])
        chain, ambiguous = self._base_chain(file_path, contract)
        found: list[ModifierBody] = []
        for base_file, base_contract in chain:
            matches = self._modifiers.get((base_file, base_contract, name), [])
            if len(matches) > 1:
                ambiguous = True
            elif len(matches) == 1:
                found.append(matches[0])
        distinct = {(item.file_path, item.contract, item.body) for item in found}
        if ambiguous or len(distinct) > 1:
            return _ambiguous(contract, file_path)
        if len(found) == 1:
            return _finish(found[0])
        return ModifierResolution("unresolved", "", contract, file_path)

    def graph_for(self, file_path: str) -> SyntaxGraph | None:
        return self._graphs.get(file_path)

    def _absorb(self, path: str, graph: SyntaxGraph) -> None:
        self._graphs[path] = graph
        for event in graph.events:
            if event.kind == "sol_contract":
                fields = _fields(event.extra)
                contract = event.text.strip()
                bases = [_base_name(item) for item in fields.get("bases", "").split(",")]
                bases = [item for item in bases if item]
                self._bases[(path, contract)] = bases
                self._contracts.setdefault(contract, []).append((path, contract))
            elif event.kind == "sol_modifier":
                fields = _fields(event.extra)
                contract = fields.get("contract", "")
                name = fields.get("name", "")
                if not name:
                    continue
                body = ModifierBody(path, contract, name, event.text)
                self._modifiers.setdefault((path, contract, name), []).append(body)

    def _base_chain(self, file_path: str, contract: str) -> tuple[list[tuple[str, str]], bool]:
        chain: list[tuple[str, str]] = []
        ambiguous = False
        seen: set[tuple[str, str]] = set()
        stack: list[tuple[str, str]] = [(file_path, contract)]
        while stack:
            current_file, current = stack.pop()
            for base in self._bases.get((current_file, current), []):
                homes = self._homes(current_file, base)
                if len(homes) > 1:
                    ambiguous = True
                    continue
                if len(homes) != 1:
                    continue
                nxt = homes[0]
                if nxt in seen:
                    continue
                seen.add(nxt)
                chain.append(nxt)
                stack.append(nxt)
        return chain, ambiguous

    def _homes(self, file_path: str, base: str) -> list[tuple[str, str]]:
        homes = self._contracts.get(base, [])
        same_file = [item for item in homes if item[0] == file_path]
        if same_file:
            return same_file
        return homes


def set_modifier_index(index: ModifierIndex) -> Token[ModifierIndex | None]:
    return _INDEX.set(index)


def reset_modifier_index(token: Token[ModifierIndex | None]) -> None:
    _INDEX.reset(token)


def current_modifier_index() -> ModifierIndex | None:
    return _INDEX.get()


def resolve_modifier(graph: SyntaxGraph, contract: str, name: str) -> ModifierResolution:
    index = current_modifier_index()
    if index is None:
        index = ModifierIndex.from_graphs({graph.file_path: graph})
    return index.resolve(graph, contract, name)


def modifier_home_graph(graph: SyntaxGraph, resolution: ModifierResolution) -> SyntaxGraph | None:
    """The syntax graph of the file that defines a resolved modifier, when known."""
    if resolution.file_path == graph.file_path:
        return graph
    index = current_modifier_index()
    if index is None:
        return None
    return index.graph_for(resolution.file_path)


def _finish(body: ModifierBody) -> ModifierResolution:
    if not re.search(r"_\s*;", body.body):
        return ModifierResolution("malformed", body.body, body.contract, body.file_path)
    return ModifierResolution("resolved", body.body, body.contract, body.file_path)


def _ambiguous(contract: str, file_path: str) -> ModifierResolution:
    return ModifierResolution("ambiguous", "", contract, file_path)


def _base_name(text: str) -> str:
    token = text.strip().split("(", 1)[0].strip()
    match = re.search(r"[A-Za-z_]\w*", token)
    return match.group(0) if match else ""


def _fields(extra: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in extra.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields

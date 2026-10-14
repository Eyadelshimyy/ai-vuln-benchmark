"""Candidate Solidity harnesses. They are not written into the analyzed repository."""

from __future__ import annotations

from pathlib import Path

from app.discovery.engine import AnalysisRequest
from app.parsing.model import SyntaxGraph
from app.parsing.solidity_graph import parse_solidity_source


def candidate_foundry_test(request: AnalysisRequest, graph: SyntaxGraph | None = None) -> str:
    function = request.function or "target"
    contract = request.contract or "Target"
    signature = request.extra.get("signature", "")
    payable = False
    unresolved = False
    if not signature and graph is not None:
        signature, payable = _signature_from_graph(graph, request.contract, function)
    if not signature:
        signature = f"{function}()"
        unresolved = "signature" not in request.extra and graph is None
    payable_text = " payable" if payable else ""
    note = ""
    if unresolved:
        note = "        // Signature unresolved: parsed parameters were not available.\n"
    return (
        "// UNTRUSTED CANDIDATE. Not a verified proof.\n"
        "pragma solidity ^0.8.20;\n"
        'import {Test} from "forge-std/Test.sol";\n'
        f"interface ITarget {{ function {signature} external{payable_text}; }}\n"
        f"contract {contract}Candidate is Test {{\n"
        f"    function test_{function.split('(')[0]}() public {{\n"
        f"{note}"
        "        // Candidate sequence generated from a static target.\n"
        "    }\n"
        "}\n"
    )


def validate_harness(source: str, path: Path) -> bool:
    graph = parse_solidity_source(path, source)
    return graph.parser_tier.value == "full_ast" and not graph.diagnostics.has_errors


def write_candidate(source: str, destination: Path, *, repo_root: Path) -> Path:
    resolved = destination.resolve()
    root = repo_root.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("candidate harnesses are not written into the analyzed repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return destination


def _signature_from_graph(graph: SyntaxGraph, contract: str, function: str) -> tuple[str, bool]:
    for entity in graph.entities:
        if entity.entity_type != "function" or entity.name != function:
            continue
        if contract and entity.parent not in {None, contract}:
            continue
        params = []
        resolved = True
        for param in entity.parameters:
            annotation = (param.annotation or "").strip()
            if not annotation:
                resolved = False
                break
            params.append(f"{annotation} {param.name}".strip())
        if not resolved:
            return "", False
        for event in graph.events:
            if event.kind == "sol_function" and f"function={function}" in event.extra:
                payable = "payable=true" in event.extra
                joined = ", ".join(params)
                return f"{function}({joined})", payable
        joined = ", ".join(params)
        return f"{function}({joined})", False
    return "", False

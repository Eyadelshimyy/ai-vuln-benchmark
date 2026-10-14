"""Optional Solidity compiler overlay.

solc and forge are not required to start BugForge or to build a syntax graph.
When a compiler is absent, this module reports UNAVAILABLE and does not invent
storage layout, types, selectors, or IR. A successful compiler response may add
facts beside the parser. A failed or unparsable response stays failed. Compiler
facts never replace the Tree-sitter graph and never verify a finding.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from app.parsing.solidity_version import SolidityLanguageFacts, solidity_language_facts

_MAX_IR_CHARS = 65_536
_MAX_COMPILER_OUTPUT = 1_000_000
_OUTPUT_CONFIG = "storageLayout,ir,evm.methodIdentifiers,ast"

_COMPILER_CACHE: ContextVar[dict[str, CompilerSemantics] | None] = ContextVar(
    "bugforge_compiler_cache", default=None
)
_TYPE_HINTS: ContextVar[dict[tuple[str, str], str] | None] = ContextVar(
    "bugforge_compiler_type_hints", default=None
)


@dataclass(frozen=True)
class SolidityCompilerStatus:
    status: str
    tool: str
    detail: str


@dataclass(frozen=True)
class SemanticDisagreement:
    """A parser fact and a compiler fact that were not forced to one winner."""

    category: str
    contract: str
    symbol: str
    parser_value: str
    compiler_value: str
    severity: str
    confidence: str


@dataclass
class CompilerSemantics:
    status: str
    tool: str
    detail: str
    storage: list[dict[str, str]] = field(default_factory=list)
    layouts: list[dict[str, str]] = field(default_factory=list)
    compiler_version: str = ""
    pragma_version: str = ""
    ir: str = ""
    ir_available: bool = False
    ir_truncated: bool = False
    ast_available: bool = False
    selectors: list[dict[str, str]] = field(default_factory=list)
    language: dict[str, str] = field(default_factory=dict)


def solidity_compiler_status() -> SolidityCompilerStatus:
    """Report whether a local compiler exists. Does not execute it."""
    for name in ("solc", "forge"):
        if shutil.which(name):
            return SolidityCompilerStatus(
                "AVAILABLE",
                name,
                f"{name} is installed. Tree-sitter remains the syntax graph. "
                "Compiler output is an optional overlay and is not a proof.",
            )
    return SolidityCompilerStatus(
        "UNAVAILABLE",
        "",
        "solc and forge are not installed. Type resolution, storage layout, "
        "and inheritance stay on the Tree-sitter parser. No compiler facts are fabricated.",
    )


def set_compiler_cache() -> Token[dict[str, CompilerSemantics] | None]:
    return _COMPILER_CACHE.set({})


def reset_compiler_cache(token: Token[dict[str, CompilerSemantics] | None]) -> None:
    _COMPILER_CACHE.reset(token)


def set_compiler_type_hints() -> Token[dict[tuple[str, str], str] | None]:
    return _TYPE_HINTS.set({})


def reset_compiler_type_hints(token: Token[dict[tuple[str, str], str] | None]) -> None:
    _TYPE_HINTS.reset(token)


def compiler_type_hint(contract: str, name: str) -> str:
    hints = _TYPE_HINTS.get()
    if not hints:
        return ""
    return hints.get((contract, name), "")


def record_compiler_type_hints(semantics: CompilerSemantics) -> None:
    """Remember compiler storage types for this scan. Unavailable adds nothing."""
    hints = _TYPE_HINTS.get()
    if hints is None or semantics.status != "AVAILABLE":
        return
    for item in semantics.layouts:
        contract = item.get("contract", "")
        label = item.get("label", "")
        type_name = item.get("type", "")
        if contract and label and type_name:
            hints[(contract, label)] = type_name


def compiler_cache_key(
    *, snapshot: str, compiler: str, version: str, config: str, sources: str
) -> str:
    raw = json.dumps(
        {
            "compiler": compiler,
            "config": config,
            "snapshot": snapshot,
            "sources": sources,
            "version": version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def compiler_semantics_for_scan(source: str, *, snapshot: str = "") -> CompilerSemantics:
    """Overlay helper used by a Solidity scan.

    A missing compiler stays UNAVAILABLE. ``solc`` may be invoked with
    standard JSON and a minimal environment. ``forge`` without a safe
    standard-JSON runner does not invent a layout. The scan cache is keyed
    by snapshot, compiler identity, configuration, and source bytes. A
    changed snapshot does not reuse an older result.
    """
    status = solidity_compiler_status()
    fingerprint = hashlib.sha256(source.encode()).hexdigest()
    key = compiler_cache_key(
        snapshot=snapshot or fingerprint,
        compiler=status.tool,
        version=status.detail,
        config=_OUTPUT_CONFIG,
        sources=fingerprint,
    )
    cache = _COMPILER_CACHE.get()
    if cache is not None and key in cache:
        return cache[key]
    if status.status != "AVAILABLE":
        result = CompilerSemantics("UNAVAILABLE", "", status.detail)
    elif status.tool != "solc":
        result = CompilerSemantics(
            "AVAILABLE",
            status.tool,
            "Compiler is present but BugForge has no safe standard-JSON runner "
            "for it. No storage layout was invented.",
        )
    elif not _host_compiler_enabled():
        result = CompilerSemantics(
            "UNAVAILABLE",
            status.tool,
            "Host compiler execution is disabled. No compiler facts were fabricated.",
        )
    else:
        result = compiler_semantics(source, runner=_run_solc_standard_json)
    result.pragma_version = pragma_solidity_version(source)
    result.language = _language_dict(solidity_language_facts(source, result.compiler_version))
    if cache is not None:
        cache[key] = result
    return result


def _host_compiler_enabled() -> bool:
    from app.core.config import get_settings

    return bool(get_settings().solidity_host_compiler)


def run_solc_standard_json(payload: str, *, binary: str = "solc") -> str:
    """Developer-only host solc. The default research path does not call this."""
    import os
    import subprocess

    from app.core.config import get_settings

    if not _host_compiler_enabled():
        raise OSError("host compiler execution is disabled")
    settings = get_settings()
    timeout = int(getattr(settings, "solidity_compiler_timeout_seconds", 20) or 20)
    completed = subprocess.run(
        [binary, "--standard-json"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={"PATH": os.environ.get("PATH", "")},
        preexec_fn=_limit_compiler_process,
    )
    stdout = completed.stdout or ""
    output_limit = int(
        getattr(settings, "solidity_compiler_max_output", _MAX_COMPILER_OUTPUT)
        or _MAX_COMPILER_OUTPUT
    )
    if len(stdout) > output_limit:
        raise ValueError("compiler output exceeded the bound")
    if not stdout:
        raise ValueError((completed.stderr or "solc produced no JSON")[:400])
    return stdout


def _limit_compiler_process() -> None:
    import resource

    from app.core.config import get_settings

    megabytes = int(getattr(get_settings(), "solidity_compiler_memory_mb", 512) or 512)
    limit = megabytes * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (OSError, ValueError):
        return


def _run_solc_standard_json(source: str) -> str:
    return run_solc_standard_json(json.dumps(_standard_json_input(source)))


def _standard_json_input(source: str) -> dict[str, object]:
    return {
        "language": "Solidity",
        "sources": {"BugForge.sol": {"content": source}},
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["storageLayout", "ir", "evm.methodIdentifiers"],
                    "": ["ast"],
                }
            }
        },
    }


def compiler_semantics(
    source: str, *, runner: Callable[[str], str] | None = None
) -> CompilerSemantics:
    """Return compiler facts when a real result is available.

    ``runner`` receives the source and returns a standard-JSON string. It is
    invoked only after a compiler binary is reported available. A missing
    compiler never produces a layout.
    """
    status = solidity_compiler_status()
    pragma = pragma_solidity_version(source)
    if status.status != "AVAILABLE":
        result = CompilerSemantics("UNAVAILABLE", "", status.detail, pragma_version=pragma)
        result.language = language_semantics(source, "")
        return result
    if runner is None:
        result = CompilerSemantics(
            "AVAILABLE",
            status.tool,
            "Compiler binary is present. BugForge does not replace the parser "
            "graph with compiler output unless a standard-JSON result is supplied.",
            pragma_version=pragma,
        )
        result.language = language_semantics(source, "")
        return result
    try:
        raw = runner(source)
    except (OSError, TimeoutError, ValueError) as exc:
        result = CompilerSemantics(
            "FAILED", status.tool, f"compiler invocation failed: {exc}", pragma_version=pragma
        )
        result.language = language_semantics(source, "")
        return result
    parsed = interpret_standard_json(raw, tool=status.tool)
    parsed.pragma_version = pragma
    version = parsed.compiler_version if parsed.status == "AVAILABLE" else ""
    parsed.language = language_semantics(source, version)
    return parsed


def interpret_standard_json(raw: str, *, tool: str = "solc") -> CompilerSemantics:
    """Parse standard JSON without requiring a compiler binary.

    Tests inject this path. Malformed JSON, compiler errors, and an object
    with none of the requested outputs stay failed and do not invent slots.
    """
    return _parse_standard_json(raw, tool)


def pragma_solidity_version(source: str) -> str:
    """Return a pragma solidity clause. Comments and strings are ignored."""
    stripped = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    stripped = re.sub(r"//.*?$", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r'"(?:\\.|[^"\\])*"', " ", stripped)
    stripped = re.sub(r"'(?:\\.|[^'\\])*'", " ", stripped)
    match = re.search(r"(?m)^[ \t]*pragma\s+solidity\s+([^;]+);", stripped)
    if match is None:
        return ""
    return " ".join(match.group(1).split())


def language_semantics(pragma: str, compiler_version: str) -> dict[str, str]:
    """Arithmetic and selfdestruct notes from the canonical version helper.

    A comment is not a version. A range is known only when every version it
    allows shares the same rule. Otherwise the result stays unknown.
    """
    return _language_dict(solidity_language_facts(pragma, compiler_version))


def _language_dict(facts: SolidityLanguageFacts) -> dict[str, str]:
    return {
        "arithmetic": facts.arithmetic,
        "selfdestruct": facts.selfdestruct,
        "source": facts.source,
    }


def reconcile_semantics(
    parser_facts: list[dict[str, str]], compiler_facts: list[dict[str, str]]
) -> list[SemanticDisagreement]:
    """Record slot, offset, type, source, and inheritance disagreements.

    Both values are kept. This function does not choose a layout.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for item in compiler_facts:
        symbol = item.get("label") or item.get("symbol") or ""
        grouped.setdefault((item.get("contract", ""), symbol), []).append(item)
    found: list[SemanticDisagreement] = []
    for fact in parser_facts:
        key = (fact.get("contract", ""), fact.get("symbol", ""))
        matches = grouped.get(key, [])
        if len(matches) != 1:
            continue
        compiler = matches[0]
        if fact.get("uncertain") == "true":
            slot = compiler.get("slot", "")
            if slot:
                found.append(
                    _disagreement(
                        "inherited_layout",
                        key[0],
                        key[1],
                        fact.get("slot") or "unknown",
                        slot,
                    )
                )
            continue
        for category, field_name in (
            ("storage_slot", "slot"),
            ("offset", "offset"),
            ("type", "type"),
            ("source_location", "source"),
        ):
            _compare(found, category, field_name, fact, compiler, key)
    return found


def reconcile_selectors(
    parser_selectors: list[dict[str, str]], compiler_selectors: list[dict[str, str]]
) -> list[SemanticDisagreement]:
    """Record selector disagreements. A missing selector is not filled in."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for item in compiler_selectors:
        symbol = item.get("name") or item.get("symbol") or ""
        selector = _normalize_selector(item.get("selector", ""))
        if symbol and selector:
            grouped.setdefault((item.get("contract", ""), symbol), []).append(selector)
    found: list[SemanticDisagreement] = []
    for item in parser_selectors:
        symbol = item.get("symbol", "")
        selector = _normalize_selector(item.get("selector", ""))
        if not symbol or not selector:
            continue
        matches = grouped.get((item.get("contract", ""), symbol), [])
        if len(matches) != 1 or matches[0] == selector:
            continue
        found.append(
            _disagreement("selector", item.get("contract", ""), symbol, selector, matches[0])
        )
    return found


def _parse_standard_json(raw: str, tool: str) -> CompilerSemantics:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return CompilerSemantics("FAILED", tool, "compiler output was not JSON")
    if not isinstance(payload, dict):
        return CompilerSemantics("FAILED", tool, "compiler output was not an object")
    errors = payload.get("errors")
    if isinstance(errors, list) and any(
        isinstance(item, dict) and item.get("severity") == "error" for item in errors
    ):
        message = next(
            (
                str(
                    item.get("formattedMessage")
                    or item.get("message")
                    or "compiler reported an error"
                )
                for item in errors
                if isinstance(item, dict) and item.get("severity") == "error"
            ),
            "compiler reported an error",
        )
        return CompilerSemantics("FAILED", tool, f"compiler reported an error: {message}"[:400])
    storage: list[dict[str, str]] = []
    layouts: list[dict[str, str]] = []
    selectors: list[dict[str, str]] = []
    ir_parts: list[str] = []
    ast_available = False
    contracts = payload.get("contracts")
    if isinstance(contracts, dict):
        for file_name, file_contracts in contracts.items():
            if not isinstance(file_contracts, dict):
                continue
            for contract_name, contract in file_contracts.items():
                if not isinstance(contract, dict):
                    continue
                _collect_ir(ir_parts, contract.get("ir"))
                _collect_ir(ir_parts, contract.get("irOptimized"))
                evm = contract.get("evm")
                if isinstance(evm, dict):
                    _collect_ir(ir_parts, evm.get("legacyAssembly"))
                    _collect_selectors(
                        selectors,
                        str(file_name),
                        str(contract_name),
                        evm.get("methodIdentifiers"),
                    )
                _collect_storage(
                    storage,
                    layouts,
                    str(file_name),
                    str(contract_name),
                    contract.get("storageLayout"),
                )
    sources = payload.get("sources")
    if isinstance(sources, dict):
        ast_available = any(
            isinstance(item, dict) and isinstance(item.get("ast"), dict)
            for item in sources.values()
        )
    version = payload.get("version") if isinstance(payload.get("version"), str) else ""
    ir = "\n".join(part for part in ir_parts if part)
    truncated = False
    if len(ir) > _MAX_IR_CHARS:
        ir = ir[:_MAX_IR_CHARS]
        truncated = True
    ir_available = bool(ir)
    if not storage and not version and not ast_available and not ir_available and not selectors:
        return CompilerSemantics(
            "FAILED", tool, "compiler JSON did not include the requested output"
        )
    return CompilerSemantics(
        "AVAILABLE",
        tool,
        "Compiler overlay parsed. These facts do not mark any finding verified.",
        storage=storage,
        layouts=layouts,
        compiler_version=str(version),
        ir=ir,
        ir_available=ir_available,
        ir_truncated=truncated,
        ast_available=ast_available,
        selectors=selectors,
    )


def _collect_storage(
    storage: list[dict[str, str]],
    layouts: list[dict[str, str]],
    file_name: str,
    contract_name: str,
    layout: object,
) -> None:
    if not isinstance(layout, dict):
        return
    slots = layout.get("storage")
    if not isinstance(slots, list):
        return
    raw_types = layout.get("types")
    types: dict[str, object] = raw_types if isinstance(raw_types, dict) else {}
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        label = str(slot.get("label") or "")
        index = str(slot.get("slot") or "")
        if not label or not index:
            continue
        storage.append({"label": label, "slot": index})
        rich = {
            "contract": contract_name,
            "label": label,
            "slot": index,
            "source": file_name,
        }
        if slot.get("offset") is not None:
            rich["offset"] = str(slot.get("offset"))
        type_name = slot.get("type")
        if isinstance(type_name, str) and type_name:
            described = types.get(type_name)
            if isinstance(described, dict) and described.get("label"):
                rich["type"] = str(described["label"])
            else:
                rich["type"] = type_name
        layouts.append(rich)


def _collect_ir(parts: list[str], value: object) -> None:
    if isinstance(value, str) and value.strip():
        parts.append(value.strip())
    elif isinstance(value, dict):
        parts.append(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _collect_selectors(
    selectors: list[dict[str, str]], source: str, contract: str, value: object
) -> None:
    if not isinstance(value, dict):
        return
    for name, selector in value.items():
        if not name or not isinstance(selector, str):
            continue
        selectors.append(
            {
                "contract": contract,
                "name": str(name),
                "selector": selector,
                "source": source,
            }
        )


def _compare(
    found: list[SemanticDisagreement],
    category: str,
    field_name: str,
    parser: dict[str, str],
    compiler: dict[str, str],
    key: tuple[str, str],
) -> None:
    left = parser.get(field_name, "")
    right = compiler.get(field_name, "")
    if not left or not right or left == right:
        return
    if field_name == "source" and left.replace("\\", "/") == right.replace("\\", "/"):
        return
    found.append(_disagreement(category, key[0], key[1], left, right))


def _disagreement(
    category: str, contract: str, symbol: str, parser_value: str, compiler_value: str
) -> SemanticDisagreement:
    return SemanticDisagreement(
        category, contract, symbol, parser_value, compiler_value, "disagreement", "unresolved"
    )


def _normalize_selector(value: str) -> str:
    return value.lower().removeprefix("0x")

"""Project-level Solidity compiler input.

The bundle is the repository snapshot already being analyzed. Paths stay
inside that root. BugForge does not execute Hardhat, Foundry scripts, or
package-manager commands. A missing compiler does not invent facts.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.core.paths import safe_join
from app.parsing.model import SyntaxGraph
from app.parsing.solidity_compiler import (
    CompilerSemantics,
    compiler_cache_key,
    interpret_standard_json,
)
from app.parsing.solidity_version import solidity_language_facts

_MAX_FILES = 40
_MAX_BYTES = 200_000
_EXTRA_SKIP = {"test", "tests", "script", "scripts", "mock", "mocks", "audits"}


def _max_files() -> int:
    from app.core.config import get_settings

    return int(getattr(get_settings(), "solidity_project_max_files", _MAX_FILES) or _MAX_FILES)


def _max_bytes() -> int:
    from app.core.config import get_settings

    return int(getattr(get_settings(), "solidity_project_max_bytes", _MAX_BYTES) or _MAX_BYTES)


def _include_dependencies() -> bool:
    from app.core.config import get_settings

    return bool(getattr(get_settings(), "solidity_project_include_dependencies", False))


_IMPORT_RE = re.compile(r"""import\s+(?:[^"']+\s+from\s+)?["']([^"']+)["']""")
_SKIP_DIRS = {
    ".git",
    "node_modules",
    "out",
    "cache",
    "lib",
    "broadcast",
    "bugforge-output",
    "bugforge-explore",
}
_PROJECT: ContextVar[CompilerProjectModel | None] = ContextVar(
    "bugforge_compiler_project", default=None
)
_PROJECT_CACHE: ContextVar[dict[str, CompilerProjectModel] | None] = ContextVar(
    "bugforge_compiler_project_cache", default=None
)


@dataclass(frozen=True)
class CompilerProjectModel:
    """One scan's compiler picture. Unknown stays unknown."""

    status: str
    tool: str = ""
    compiler_version: str = ""
    compiler_profiles: tuple[dict[str, str], ...] = ()
    source_identity: str = ""
    configuration_identity: str = ""
    diagnostics: tuple[str, ...] = ()
    contracts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    storage: tuple[dict[str, str], ...] = ()
    layouts: tuple[dict[str, str], ...] = ()
    selectors: tuple[dict[str, str], ...] = ()
    ast_available: bool = False
    ir_available: bool = False
    ir: str = ""
    ir_truncated: bool = False
    language: dict[str, str] = field(default_factory=dict)
    complete: bool = False
    unsupported: tuple[str, ...] = ()
    detail: str = ""


def set_compiler_project(
    model: CompilerProjectModel | None,
) -> Token[CompilerProjectModel | None]:
    return _PROJECT.set(model)


def reset_compiler_project(token: Token[CompilerProjectModel | None]) -> None:
    _PROJECT.reset(token)


def current_compiler_project() -> CompilerProjectModel | None:
    return _PROJECT.get()


def set_compiler_project_cache() -> Token[dict[str, CompilerProjectModel] | None]:
    return _PROJECT_CACHE.set({})


def reset_compiler_project_cache(token: Token[dict[str, CompilerProjectModel] | None]) -> None:
    _PROJECT_CACHE.reset(token)


def resolve_analysis_source(
    repo_root: Path, public_path: str, embedded: str
) -> tuple[str, str, str]:
    """Return ``(status, relative_path, text)``.

    ``status`` is ``ok``, ``missing``, or ``rejected``. An escaping path is
    rejected and is not read. Embedded parser bytes are used only after the
    public path resolves inside the repository.
    """
    if not public_path or "\x00" in public_path:
        return "rejected", "", ""
    root = repo_root.resolve()
    candidate = Path(public_path)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            return "rejected", "", ""
        if _contains_symlink(root, relative):
            return "rejected", "", ""
    else:
        if ".." in Path(public_path).parts or public_path.startswith(("/", "\\")):
            return "rejected", "", ""
        try:
            safe_join(root, public_path)
        except ValueError:
            return "rejected", "", ""
        relative = Path(public_path).as_posix()
        if _contains_symlink(root, relative):
            return "rejected", "", ""
    full = root / relative
    if embedded:
        return "ok", relative, embedded
    if not full.is_file():
        return "missing", relative, ""
    try:
        return "ok", relative, full.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "missing", relative, ""


def project_standard_json(
    sources: dict[str, str],
    remappings: list[str],
    build: dict[str, str] | None = None,
) -> dict[str, object]:
    """Standard JSON for the bounded source bundle. No model-supplied flags."""
    from app.core.config import get_settings

    selected = ["storageLayout", "evm.methodIdentifiers"]
    if bool(getattr(get_settings(), "solidity_compiler_emit_ir", True)):
        selected.insert(1, "ir")
    settings: dict[str, object] = {
        "remappings": [_solc_remapping(item) for item in remappings],
        "outputSelection": {
            "*": {
                "*": selected,
                "": ["ast"],
            }
        },
    }
    chosen = build or {}
    if chosen.get("optimizer") in {"true", "false"}:
        runs = int(chosen.get("optimizer_runs") or "200")
        settings["optimizer"] = {"enabled": chosen["optimizer"] == "true", "runs": runs}
    if chosen.get("via_ir") in {"true", "false"}:
        settings["viaIR"] = chosen["via_ir"] == "true"
    if chosen.get("evm_version"):
        settings["evmVersion"] = chosen["evm_version"]
    return {
        "language": "Solidity",
        "sources": {path: {"content": text} for path, text in sorted(sources.items())},
        "settings": settings,
    }


def build_compiler_project(
    repo_root: Path,
    graphs: dict[str, SyntaxGraph],
    *,
    runner: Callable[[str], str] | None = None,
) -> CompilerProjectModel:
    """Build a project compiler model from graphs already parsed for this scan."""
    sources, diagnostics, complete, unsupported, remappings = _bundle(repo_root, graphs)
    identity = _source_identity(sources)
    build_config = _static_build_config(repo_root)
    config_identity = hashlib.sha256(
        json.dumps(
            {"remappings": remappings, "unsupported": unsupported, "build": build_config},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    language = solidity_language_facts("\n".join(sources.values()))
    language_dict = {
        "arithmetic": language.arithmetic,
        "selfdestruct": language.selfdestruct,
        "source": language.source,
    }
    tool, host_enabled = _compiler_tool()
    key = compiler_cache_key(
        snapshot=identity,
        compiler=tool,
        version="host" if host_enabled else "disabled",
        config=config_identity,
        sources=identity,
    )
    cache = _PROJECT_CACHE.get()
    if cache is not None and key in cache and runner is None:
        return cache[key]
    base = CompilerProjectModel(
        status="UNAVAILABLE",
        tool=tool,
        source_identity=identity,
        configuration_identity=config_identity,
        diagnostics=tuple(diagnostics),
        sources=tuple(sorted(sources)),
        language=language_dict,
        complete=False,
        unsupported=tuple(unsupported),
    )
    if not sources:
        model = _replace(base, detail="no readable Solidity sources were available for compilation")
        return _store(cache, key, model)
    if not complete:
        model = _replace(
            base,
            status="INCOMPLETE",
            detail="project source bundle is incomplete; compiler facts are not treated as proof",
        )
        return _store(cache, key, model)
    if runner is None and not host_enabled:
        model = _replace(
            base,
            detail=(
                "Host compiler execution is disabled. No compiler facts were fabricated. "
                "Parser results remain available."
            ),
        )
        return _store(cache, key, model)
    try:
        if runner is not None:
            parsed_groups = [
                interpret_standard_json(
                    runner(json.dumps(project_standard_json(sources, remappings, build_config))),
                    tool=tool or "solc",
                )
            ]
        else:
            parsed_groups = _compile_version_groups(
                sources, remappings, tool or "solc", repo_root, build_config
            )
    except ValueError as exc:
        model = _replace(base, status="INCOMPLETE", detail=str(exc)[:400], complete=False)
        return _store(cache, key, model)
    except (OSError, TimeoutError, subprocess.SubprocessError) as exc:
        model = _replace(
            base,
            status="FAILED",
            detail=f"compiler invocation failed: {exc}"[:400],
        )
        return _store(cache, key, model)
    if any(item.status != "AVAILABLE" for item in parsed_groups):
        failed = next(item for item in parsed_groups if item.status != "AVAILABLE")
        model = _replace(
            base,
            status="INCOMPLETE"
            if any(item.status == "AVAILABLE" for item in parsed_groups)
            else "FAILED",
            detail=failed.detail[:400],
            tool=failed.tool or tool,
        )
        return _store(cache, key, model)
    parsed = _merge_compiler_groups(parsed_groups)
    if parsed.status != "AVAILABLE":
        model = _replace(
            base, status="FAILED", detail=parsed.detail[:400], tool=parsed.tool or tool
        )
        return _store(cache, key, model)
    contracts = tuple(
        sorted(
            {
                item.get("contract", "")
                for item in [*parsed.layouts, *parsed.selectors]
                if item.get("contract")
            }
        )
    )
    compiled_language = solidity_language_facts(
        "\n".join(sources.values()), parsed.compiler_version
    )
    model = CompilerProjectModel(
        status="AVAILABLE",
        tool=parsed.tool or tool,
        compiler_version=parsed.compiler_version,
        source_identity=identity,
        configuration_identity=config_identity,
        diagnostics=tuple(diagnostics),
        contracts=contracts,
        sources=tuple(sorted(sources)),
        storage=tuple(parsed.storage),
        layouts=tuple(parsed.layouts),
        selectors=tuple(parsed.selectors),
        ast_available=parsed.ast_available,
        ir_available=parsed.ir_available,
        ir=parsed.ir,
        ir_truncated=parsed.ir_truncated,
        language={
            "arithmetic": compiled_language.arithmetic,
            "selfdestruct": compiled_language.selfdestruct,
            "source": compiled_language.source,
        },
        complete=True,
        unsupported=tuple(unsupported),
        detail=parsed.detail,
        compiler_profiles=tuple(
            {
                "version": item.compiler_version,
                "tool": item.tool,
                "status": item.status,
                "ir": "truncated"
                if item.ir_truncated
                else "available"
                if item.ir_available
                else "unavailable",
                "ast": "available" if item.ast_available else "unavailable",
                "sources": item.language.get("sources", ""),
            }
            for item in parsed_groups
        ),
    )
    return _store(cache, key, model)


def project_semantics(model: CompilerProjectModel) -> CompilerSemantics:
    """Compiler semantics for rules. Incomplete and unavailable invent no layout."""
    if model.status != "AVAILABLE" or not model.complete:
        return CompilerSemantics(model.status or "UNAVAILABLE", model.tool, model.detail)
    return CompilerSemantics(
        "AVAILABLE",
        model.tool,
        model.detail,
        storage=[dict(item) for item in model.storage],
        layouts=[dict(item) for item in model.layouts],
        compiler_version=model.compiler_version,
        ir=model.ir,
        ir_available=model.ir_available,
        ir_truncated=model.ir_truncated,
        ast_available=model.ast_available,
        selectors=[dict(item) for item in model.selectors],
        language=dict(model.language),
    )


def slice_semantics(semantics: CompilerSemantics, relative_path: str) -> CompilerSemantics:
    """Keep compiler facts whose source path is this file. Do not merge names."""
    if semantics.status != "AVAILABLE":
        return semantics
    wanted = _normalize(relative_path)
    layouts = [
        item for item in semantics.layouts if _normalize(str(item.get("source", ""))) == wanted
    ]
    selectors = [
        item for item in semantics.selectors if _normalize(str(item.get("source", ""))) == wanted
    ]
    storage = [
        {"label": item["label"], "slot": item["slot"]}
        for item in layouts
        if item.get("label") and item.get("slot")
    ]
    return CompilerSemantics(
        semantics.status,
        semantics.tool,
        semantics.detail,
        storage=storage,
        layouts=layouts,
        compiler_version=semantics.compiler_version,
        pragma_version=semantics.pragma_version,
        ir=semantics.ir,
        ir_available=semantics.ir_available,
        ir_truncated=semantics.ir_truncated,
        ast_available=semantics.ast_available,
        selectors=selectors,
        language=dict(semantics.language),
    )


def _bundle(
    repo_root: Path, graphs: dict[str, SyntaxGraph]
) -> tuple[dict[str, str], list[str], bool, list[str], list[str]]:
    sources: dict[str, str] = {}
    diagnostics: list[str] = []
    complete = True
    unsupported = _static_limits(repo_root)
    remappings, remap_notes = _read_remappings(repo_root)
    unsupported.extend(remap_notes)
    for public, graph in graphs.items():
        if graph.language != "solidity":
            continue
        status, relative, text = resolve_analysis_source(repo_root, public, graph.source)
        if status != "ok":
            complete = False
            diagnostics.append(f"{status} source: {public}")
            continue
        if text == "":
            complete = False
            diagnostics.append(f"empty source: {relative or public}")
            continue
        sources[relative] = text
    file_limit = _max_files()
    byte_limit = _max_bytes()
    total = sum(len(item.encode("utf-8")) for item in sources.values())
    if len(sources) > file_limit or total > byte_limit:
        complete = False
        diagnostics.append("required source bundle limit reached")
        return sources, diagnostics, complete, unsupported, remappings
    root = repo_root.resolve()
    omitted = 0
    # Import closure is the relevant dependency set. A repository-wide search
    # would spend the same budget on files the production sources never import.
    if _include_dependencies() and root.is_dir():
        sources, total, omitted, complete = _close_imports(
            root,
            sources,
            remappings,
            file_limit,
            byte_limit,
            total,
            omitted,
            diagnostics,
            complete,
        )
    elif root.is_dir():
        for path in sorted(root.rglob("*.sol")):
            if any(part in _SKIP_DIRS or part in _EXTRA_SKIP for part in path.parts):
                continue
            if path.is_symlink():
                diagnostics.append(f"skipped symlink: {path.name}")
                continue
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                diagnostics.append(f"rejected source: {path.name}")
                complete = False
                continue
            if relative in sources or _contains_symlink(root, relative):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                diagnostics.append(f"missing source: {relative}")
                complete = False
                continue
            size = len(text.encode("utf-8"))
            if len(sources) >= file_limit or total + size > byte_limit:
                omitted += 1
                continue
            sources[relative] = text
            total += size
    if omitted:
        diagnostics.append(f"omitted {omitted} optional sources outside the bounded bundle")
    return sources, diagnostics, complete, unsupported, remappings


def _close_imports(
    root: Path,
    sources: dict[str, str],
    remappings: list[str],
    file_limit: int,
    byte_limit: int,
    total: int,
    omitted: int,
    diagnostics: list[str],
    complete: bool,
) -> tuple[dict[str, str], int, int, bool]:
    """Add the import closure of the bundled sources, still inside the file and byte caps."""
    pairs: list[tuple[str, str]] = []
    for raw in remappings:
        if "=" not in raw:
            continue
        prefix, target = raw.split("=", 1)
        pairs.append((prefix, target))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    queue = list(sources)
    while queue:
        relative = queue.pop()
        origin = root / relative
        for spec in _IMPORT_RE.findall(sources.get(relative, "")):
            candidate = _resolve_import(root, origin, spec, pairs)
            if candidate is None or not candidate.is_file():
                diagnostics.append(f"unresolved import: {spec}")
                complete = False
                continue
            try:
                imported = candidate.resolve().relative_to(root).as_posix()
            except ValueError:
                diagnostics.append(f"rejected import: {spec}")
                complete = False
                continue
            if imported in sources or _contains_symlink(root, imported):
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                diagnostics.append(f"missing import: {imported}")
                complete = False
                continue
            size = len(text.encode("utf-8"))
            if len(sources) >= file_limit or total + size > byte_limit:
                omitted += 1
                complete = False
                diagnostics.append(f"omitted import over limit: {imported}")
                continue
            sources[imported] = text
            total += size
            queue.append(imported)
    return sources, total, omitted, complete


def _resolve_import(
    root: Path, origin: Path, spec: str, remappings: list[tuple[str, str]]
) -> Path | None:
    for prefix, target in remappings:
        if spec.startswith(prefix):
            return root / target / spec[len(prefix) :]
    if spec.startswith("."):
        return origin.parent / spec
    return root / spec


def _static_build_config(repo_root: Path) -> dict[str, str]:
    """Read foundry.toml as text. Do not execute it."""
    path = repo_root.resolve() / "foundry.toml"
    config = {"optimizer": "", "optimizer_runs": "", "via_ir": "", "evm_version": "", "solc": ""}
    if not path.is_file() or path.is_symlink():
        return config
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return config
    for key, pattern in (
        ("optimizer", r"(?m)^\s*optimizer\s*=\s*(\w+)"),
        ("optimizer_runs", r"(?m)^\s*optimizer_runs\s*=\s*(\d+)"),
        ("via_ir", r"(?m)^\s*via_ir\s*=\s*(\w+)"),
        ("evm_version", r'(?m)^\s*evm_version\s*=\s*"([^"]+)"'),
        ("solc", r'(?m)^\s*solc\s*=\s*"([^"]+)"'),
    ):
        match = re.search(pattern, text)
        if match:
            config[key] = match.group(1)
    return config


def _static_limits(repo_root: Path) -> list[str]:
    notes: list[str] = []
    root = repo_root.resolve()
    if not root.is_dir():
        return notes
    for name in ("hardhat.config.js", "hardhat.config.ts", "foundry.toml"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            notes.append(f"{name} was not executed")
    lib = root / "lib"
    if lib.is_dir() and not lib.is_symlink():
        notes.append("package directory lib was not executed or compiled")
    return notes


def _read_remappings(repo_root: Path) -> tuple[list[str], list[str]]:
    path = repo_root.resolve() / "remappings.txt"
    if not path.is_file() or path.is_symlink():
        return [], []
    found: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], ["remappings.txt could not be read"]
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if any(char in line for char in "$`;&|<>"):
            return [], ["remappings.txt contains unsupported syntax and was not used"]
        if line.count("=") != 1:
            return [], ["remappings.txt is not a static prefix map and was not used"]
        found.append(line)
        if len(found) > 64:
            return found[:64], ["remappings.txt was truncated at 64 entries"]
    return found, []


def _compiler_tool() -> tuple[str, bool]:
    import shutil

    from app.core.config import get_settings

    enabled = bool(get_settings().solidity_host_compiler)
    if shutil.which("solc"):
        return "solc", enabled
    if enabled and _solc_binary("0.8.34"):
        return "solc", True
    if shutil.which("forge"):
        return "forge", False
    return "", False


def _host_runner(payload: str) -> str:
    from app.parsing.solidity_compiler import run_solc_standard_json

    return run_solc_standard_json(payload)


def _pinned_version(text: str) -> str:
    match = re.search(r"(?m)^[ \t]*pragma[ \t]+solidity[ \t]*=[ \t]*(0\.\d+\.\d+)", text)
    if match:
        return match.group(1)
    return "0.8.34"


def _solc_binary(version: str) -> str | None:
    named = shutil.which(f"solc-{version}")
    if named:
        return named
    candidates = (
        Path.home() / "Library" / "Application Support" / "svm" / version / f"solc-{version}",
        Path.home() / ".svm" / version / f"solc-{version}",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("solc")


def _solc_remapping(raw: str) -> str:
    """Keep a directory remapping from concatenating onto the next path segment."""
    if "=" not in raw:
        return raw
    prefix, target = raw.split("=", 1)
    if prefix.endswith("/") and target and not target.endswith("/"):
        target = target + "/"
    return f"{prefix}={target}"


def _version_groups(
    sources: dict[str, str],
    remappings: list[str],
    root: Path,
    build: dict[str, str] | None = None,
) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    """Group sources by the single exact compiler their import closure allows."""
    pairs: list[tuple[str, str]] = []
    for raw in remappings:
        if "=" not in raw:
            continue
        prefix, target = raw.split("=", 1)
        pairs.append((prefix, target))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    pin_re = re.compile(r"(?m)^[ \t]*pragma[ \t]+solidity[ \t]*=[ \t]*(0\.\d+\.\d+)")
    pins = {
        path: (match.group(1) if (match := pin_re.search(text)) else "")
        for path, text in sources.items()
    }
    imported: dict[str, list[str]] = {path: [] for path in sources}
    root = root.resolve()
    for path, text in sources.items():
        origin = root / path
        for spec in _IMPORT_RE.findall(text):
            candidate = _resolve_import(root, origin, spec, pairs)
            if candidate is None:
                continue
            try:
                relative = candidate.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if relative in sources:
                imported[path].append(relative)
    memo: dict[str, set[str]] = {}

    def closure(path: str, stack: set[str]) -> set[str]:
        if path in memo:
            return memo[path]
        found: set[str] = set()
        if pins.get(path):
            found.add(pins[path])
        stack.add(path)
        for nxt in imported.get(path, []):
            if nxt not in stack:
                found |= closure(nxt, stack)
        stack.remove(path)
        memo[path] = found
        return found

    assigned: dict[str, str] = {}
    omitted: list[str] = []
    for path in sources:
        found = closure(path, set())
        if len(found) == 1:
            assigned[path] = next(iter(found))
        else:
            omitted.append(path)
    groups: dict[str, dict[str, str]] = {}
    seen: set[tuple[str, str]] = set()

    def add(path: str, version: str) -> None:
        key = (path, version)
        if key in seen or path not in sources:
            return
        seen.add(key)
        groups.setdefault(version, {})[path] = sources[path]
        for nxt in imported.get(path, []):
            other = assigned.get(nxt)
            if other == version or (other == "0.8.34" and not pins.get(nxt)):
                add(nxt, version)

    pinned_solc = (build or {}).get("solc", "")
    if pinned_solc:
        for path in list(omitted):
            if not pins.get(path):
                assigned[path] = pinned_solc
                omitted.remove(path)
    changed = True
    while changed:
        changed = False
        for path, version in list(assigned.items()):
            for nxt in imported.get(path, []):
                if nxt in assigned:
                    continue
                if pins.get(nxt) and pins[nxt] != version:
                    continue
                assigned[nxt] = version
                if nxt in omitted:
                    omitted.remove(nxt)
                changed = True
    for path, version in assigned.items():
        add(path, version)
    return groups, tuple(dict.fromkeys(omitted))


def _compile_version_groups(
    sources: dict[str, str],
    remappings: list[str],
    tool: str,
    root: Path,
    build: dict[str, str] | None = None,
) -> list[CompilerSemantics]:
    groups, omitted = _version_groups(sources, remappings, root, build)
    if omitted:
        raise ValueError("compiler groups omitted unresolved sources: " + ", ".join(omitted[:12]))
    parsed: list[CompilerSemantics] = []
    for version, group in sorted(groups.items()):
        if not group:
            continue
        binary = _solc_binary(version)
        if not binary:
            raise OSError(f"solc {version} is unavailable")
        raw = _host_runner_binary(
            json.dumps(project_standard_json(group, remappings, build)), binary
        )
        parsed_group = interpret_standard_json(raw, tool=tool)
        parsed_group.language["sources"] = ",".join(sorted(group))
        if not parsed_group.compiler_version:
            parsed_group = replace(parsed_group, compiler_version=version)
        if parsed_group.status != "AVAILABLE":
            parsed_group = replace(
                parsed_group, detail=f"solc {version}: {parsed_group.detail}"[:400]
            )
        parsed.append(parsed_group)
    return parsed


def _host_runner_binary(payload: str, binary: str) -> str:
    from app.parsing.solidity_compiler import run_solc_standard_json

    return run_solc_standard_json(payload, binary=binary)


def _merge_compiler_groups(groups: list[CompilerSemantics]) -> CompilerSemantics:
    first = groups[0]
    if len(groups) == 1:
        return first
    layouts: list[dict[str, str]] = []
    selectors: list[dict[str, str]] = []
    storage: list[dict[str, str]] = []
    ir_parts: list[str] = []
    for item in groups:
        layouts.extend(item.layouts)
        selectors.extend(item.selectors)
        storage.extend(item.storage)
        if item.ir:
            ir_parts.append(item.ir)
    ir = "\n".join(ir_parts)
    truncated = any(item.ir_truncated for item in groups)
    return CompilerSemantics(
        "AVAILABLE",
        first.tool,
        "compiled pinned source groups separately",
        storage=storage,
        layouts=layouts,
        compiler_version=""
        if len({item.compiler_version for item in groups}) > 1
        else first.compiler_version,
        ir=ir,
        ir_available=any(item.ir_available for item in groups),
        ir_truncated=truncated,
        ast_available=any(item.ast_available for item in groups),
        selectors=selectors,
        language=dict(first.language),
    )


def _source_identity(sources: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(sources[path].encode("utf-8")).digest())
    return digest.hexdigest()


def _store(
    cache: dict[str, CompilerProjectModel] | None, key: str, model: CompilerProjectModel
) -> CompilerProjectModel:
    if cache is not None:
        cache[key] = model
    return model


def _replace(model: CompilerProjectModel, **changes: object) -> CompilerProjectModel:
    data = {
        "status": model.status,
        "tool": model.tool,
        "compiler_version": model.compiler_version,
        "source_identity": model.source_identity,
        "configuration_identity": model.configuration_identity,
        "diagnostics": model.diagnostics,
        "contracts": model.contracts,
        "sources": model.sources,
        "storage": model.storage,
        "layouts": model.layouts,
        "selectors": model.selectors,
        "ast_available": model.ast_available,
        "ir_available": model.ir_available,
        "ir": model.ir,
        "ir_truncated": model.ir_truncated,
        "language": model.language,
        "complete": model.complete,
        "unsupported": model.unsupported,
        "detail": model.detail,
    }
    data.update(changes)
    return CompilerProjectModel(**data)  # type: ignore[arg-type]


def _contains_symlink(root: Path, relative: str) -> bool:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _normalize(path: str) -> str:
    text = path.replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text

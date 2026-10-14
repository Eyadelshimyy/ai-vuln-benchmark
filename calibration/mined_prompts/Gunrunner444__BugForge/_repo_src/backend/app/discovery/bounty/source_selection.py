"""Truncation-aware source selection (Phase 51, owner Phase 4).

When a repository holds more Solidity than can be analysed under the file budget,
the files that are kept must be the ones that matter: in-scope and deployed assets
first, then proxy/implementation code, then files imported by those, then
security-sensitive code, then the rest. Truncation is always reported explicitly
with ``truncated=true``; a file that was dropped is *not* implied to be clean, safe,
or out of scope. It was simply not read under the budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from app.discovery.bounty.campaign import BountyManifest

MAX_CANDIDATE_FILES = 2000
DEFAULT_LIMIT = 64
MAX_FILE_BYTES = 400_000
MAX_DEPENDENCY_DEPTH = 4
MAX_DEPENDENCY_FILES = 32
# Bounded follow-up for high-priority files that did not fit the primary budget.
MAX_FOLLOWUP_BATCHES = 2
FOLLOWUP_PRIORITY = 4  # Priority.SECURITY_SENSITIVE and better
_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "lib",
        "out",
        "cache",
        "artifacts",
        "broadcast",
        "script",
        "test",
        "tests",
    }
)
_IMPORT = re.compile(r"""import\s+(?:[^\"';]*\s+from\s+)?["']([^"']+)["']""")
_CONTRACT_DECL = re.compile(r"\b(?:contract|library|interface)\s+([A-Za-z_]\w*)")
_BASES = re.compile(r"\b(?:abstract\s+)?contract\s+[A-Za-z_]\w*(?:\s+is\s+([^{;]+))?\s*\{")
_SENSITIVE = (
    re.compile(r"\.delegatecall\s*\("),
    re.compile(r"\bselfdestruct\s*\(|\bsuicide\s*\("),
    re.compile(r"\bassembly\b"),
    re.compile(r"\.call\s*\{\s*value"),
    re.compile(r"\becrecover\s*\("),
    re.compile(r"\btransferFrom\s*\(|\bsafeTransferFrom\s*\("),
    re.compile(r"\b_mint\b|\b_burn\b"),
    re.compile(r"\bonlyOwner\b|\bAccessControl\b|\bhasRole\b"),
    re.compile(r"\b_authorizeUpgrade\b|\bupgradeTo\b"),
)


class Priority(IntEnum):
    SCOPE = 0
    DEPLOYED = 1
    PROXY = 2
    IMPORTED_BY_SCOPE = 3
    SECURITY_SENSITIVE = 4
    OTHER = 5


@dataclass(frozen=True)
class RankedFile:
    path: str
    priority: Priority
    reason: str
    role: str = "unclassified"  # target | compile_dependency | analysis_context | excluded


@dataclass(frozen=True)
class FileRead:
    """One bounded file-read outcome. Empty text is never a stand-in for a failure."""

    path: str
    status: str  # read | skipped
    reason: str  # read | size_limit | filesystem_error | invalid_encoding | disappeared | path_rejected | excluded

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class SourceSelection:
    selected: tuple[str, ...]
    ranked: tuple[RankedFile, ...]
    considered: int
    limit: int
    truncated: bool
    dropped: tuple[str, ...]
    # contract name -> every file that declares it, when more than one does. Such a
    # name is never silently bound to the first file in alphabetical order.
    ambiguous_contracts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    mode: str = "ranked"  # ranked | explicit | unassessed
    # Imports and bases that could not be tied to one file inside the repository.
    unresolved_imports: tuple[tuple[str, str], ...] = ()
    ambiguous_imports: tuple[tuple[str, tuple[str, ...]], ...] = ()
    escaped_imports: tuple[tuple[str, str], ...] = ()
    dependency_notes: tuple[str, ...] = ()
    reads: tuple[FileRead, ...] = ()
    repository_candidates: int = 0
    excluded_count: int = 0
    enumeration_capped: bool = False
    # True when every traced dependency of the target was kept in ``selected``.
    closure_complete: bool = True
    scope_excluded: tuple[str, ...] = ()

    @staticmethod
    def explicit(files: tuple[str, ...]) -> SourceSelection:
        """The operator named the files; nothing was ranked or dropped."""
        ranked = tuple(
            RankedFile(path, Priority.SCOPE, "named by the operator", "target") for path in files
        )
        return SourceSelection(
            selected=tuple(files),
            ranked=ranked,
            considered=len(files),
            limit=len(files),
            truncated=False,
            dropped=(),
            mode="explicit",
        )

    @property
    def dropped_high_priority(self) -> tuple[str, ...]:
        """Dropped files that matter: scope, deployed, proxy, imported, sensitive."""
        priority = {item.path: item.priority for item in self.ranked}
        return tuple(
            path
            for path in self.dropped
            if priority.get(path, Priority.OTHER).value <= FOLLOWUP_PRIORITY
        )

    def followup_batches(
        self, max_batches: int = MAX_FOLLOWUP_BATCHES
    ) -> tuple[tuple[str, ...], ...]:
        """Bounded follow-up batches over the dropped high-priority files, best first."""
        pending = list(self.dropped_high_priority)
        size = max(1, self.limit)
        batches: list[tuple[str, ...]] = []
        while pending and len(batches) < max(0, max_batches):
            batches.append(tuple(pending[:size]))
            pending = pending[size:]
        return tuple(batches)

    @property
    def files_read(self) -> int:
        return sum(1 for item in self.reads if item.status == "read")

    @property
    def files_not_read(self) -> int:
        return sum(1 for item in self.reads if item.status != "read")

    def role_of(self, path: str) -> str:
        for item in self.ranked:
            if item.path == path:
                return item.role
        return ""

    def dependency_status(self) -> str:
        """traced, partial, bounded, or unknown.

        traced: every examined file was read and its imports and bases resolved.
        partial: an import or base is unresolved, ambiguous, or outside the
        repository, or a file that was supposed to be read was not.
        bounded: a selection, enumeration, or dependency-traversal limit stopped
        the trace and nothing above is unresolved.
        unknown: the repository was not assessed, or a named set was never read.
        """
        if self.mode == "unassessed":
            return "unknown"
        unread = any(item.status != "read" and item.reason != "excluded" for item in self.reads)
        if unread or self.unresolved_imports or self.ambiguous_imports or self.escaped_imports:
            return "partial"
        if not self.closure_complete or self.dependency_notes or self.enumeration_capped:
            return "bounded"
        if self.mode == "explicit" and not self.reads:
            return "unknown"
        return "traced"

    def request_extra(self) -> dict[str, str]:
        """Selection facts that travel with the analysis request and its evidence."""
        extra = {
            "source_selection_mode": self.mode,
            "source_selection_truncated": str(self.truncated).lower(),
            "source_selection_considered": str(self.considered),
            "source_selection_selected": str(len(self.selected)),
        }
        if self.truncated:
            extra["source_selection_dropped"] = str(len(self.dropped))
            extra["source_selection_dropped_high_priority"] = str(len(self.dropped_high_priority))
        if self.ambiguous_contracts:
            extra["source_selection_ambiguous"] = ",".join(
                name for name, _ in self.ambiguous_contracts
            )[:300]
        extra["source_dependency_status"] = self.dependency_status()
        extra["source_files_read"] = str(self.files_read)
        extra["source_files_not_read"] = str(self.files_not_read)
        return extra

    def as_dict(self) -> dict[str, Any]:
        batches = self.followup_batches()
        followed = {path for batch in batches for path in batch}
        return {
            "mode": self.mode,
            "selected": list(self.selected),
            "considered": self.considered,
            "limit": self.limit,
            "truncated": self.truncated,
            "dropped_count": len(self.dropped),
            "dropped": list(self.dropped[:200]),
            "dropped_high_priority": list(self.dropped_high_priority[:200]),
            "followup_batches": [list(batch) for batch in batches],
            "never_read": [p for p in self.dropped_high_priority if p not in followed][:200],
            "ambiguous_contracts": {name: list(files) for name, files in self.ambiguous_contracts},
            "unresolved_imports": [
                {"file": src, "import": target} for src, target in self.unresolved_imports
            ],
            "ambiguous_imports": [
                {"import": target, "files": list(files)} for target, files in self.ambiguous_imports
            ],
            "escaped_imports": [
                {"file": src, "import": target} for src, target in self.escaped_imports
            ],
            "dependency_notes": list(self.dependency_notes),
            "dependency_status": self.dependency_status(),
            "files_read": self.files_read,
            "files_not_read": self.files_not_read,
            "file_reads": [item.as_dict() for item in self.reads if item.status != "read"][:200],
            "repository_candidates": self.repository_candidates,
            "excluded_count": self.excluded_count,
            "enumeration_capped": self.enumeration_capped,
            "closure_complete": self.closure_complete,
            "scope_excluded_count": len(self.scope_excluded),
            "scope_excluded": list(self.scope_excluded[:200]),
            "role_counts": _role_counts(self.ranked),
            "note": (
                "truncated=true means files were not read under the budget; a dropped file "
                "is not implied to be clean, safe, or out of scope"
            )
            if self.truncated
            else "all candidate files fit within the budget",
            "ranking": [
                {
                    "path": item.path,
                    "priority": item.priority.name.lower(),
                    "reason": item.reason,
                    "role": item.role,
                }
                for item in self.ranked
            ],
        }


def _enumerate(repo_root: Path) -> tuple[list[Path], int, bool]:
    """Solidity files eligible for analysis, how many directory policy excluded, and
    whether the candidate cap stopped the walk."""
    files: list[Path] = []
    excluded = 0
    capped = False
    for path in sorted(repo_root.rglob("*.sol")):
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        if set(rel.parts) & _SKIP_DIRS:
            excluded += 1
            continue
        if len(files) >= MAX_CANDIDATE_FILES:
            capped = True
            break
        files.append(path)
    return files, excluded, capped


def _relative_name(name: str) -> str:
    text = str(name).replace("\\", "/").strip()
    if text.startswith("./"):
        text = text[2:]
    pure_parts = tuple(part for part in text.split("/") if part not in {"", "."})
    if not text or text.startswith("/") or ".." in pure_parts:
        return ""
    return "/".join(pure_parts)


def inspect_named(root: Path, name: str) -> tuple[FileRead, str]:
    """Read one repository-relative Solidity file.

    The text is returned only when the file was actually read. A skipped file
    yields an empty string and a reason; that empty string is not source text.
    """
    rel = _relative_name(name)
    if not rel or not rel.endswith(".sol"):
        return FileRead(str(name), "skipped", "path_rejected"), ""
    path = (root / rel).resolve()
    try:
        inside = path == root or root in path.parents
    except ValueError:
        inside = False
    if not inside:
        return FileRead(rel, "skipped", "path_rejected"), ""
    return _inspect_path(root, path, rel)


def _inspect_path(root: Path, path: Path, rel: str) -> tuple[FileRead, str]:
    try:
        if path.is_symlink():
            target = path.resolve()
            if root not in target.parents:
                return FileRead(rel, "skipped", "path_rejected"), ""
        if not path.is_file():
            return FileRead(rel, "skipped", "disappeared"), ""
        if path.stat().st_size > MAX_FILE_BYTES:
            return FileRead(rel, "skipped", "size_limit"), ""
        return FileRead(rel, "read", "read"), path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return FileRead(rel, "skipped", "invalid_encoding"), ""
    except FileNotFoundError:
        return FileRead(rel, "skipped", "disappeared"), ""
    except (OSError, ValueError):
        return FileRead(rel, "skipped", "filesystem_error"), ""


def read_source_files(
    repo_root: Path, names: tuple[str, ...]
) -> tuple[dict[str, str], tuple[FileRead, ...]]:
    """Successfully read sources, plus a record for every requested path."""
    root = repo_root.resolve()
    sources: dict[str, str] = {}
    reads: list[FileRead] = []
    for name in names:
        record, text = inspect_named(root, name)
        reads.append(record)
        if record.status == "read":
            sources[record.path] = text
    return sources, tuple(reads)


def select_sources(
    repo_root: Path,
    *,
    manifest: BountyManifest | None = None,
    focus_contract: str = "",
    focus_file: str = "",
    limit: int = DEFAULT_LIMIT,
) -> SourceSelection:
    """Rank and select source files, keeping the most relevant within ``limit``."""

    root = repo_root.resolve()
    if not root.is_dir():
        return SourceSelection((), (), 0, limit, False, (), mode="unassessed")
    paths, excluded, capped = _enumerate(root)
    rel_paths = [p.relative_to(root).as_posix() for p in paths]
    reads: list[FileRead] = []
    texts: dict[str, str] = {}
    for rel, path in zip(rel_paths, paths, strict=True):
        record, text = _inspect_path(root, path, rel)
        reads.append(record)
        if record.status == "read":
            texts[rel] = text

    scope_contracts, scope_paths = _scope_targets(manifest, focus_contract, focus_file)
    deployed_contracts = (
        {d.contract_name for d in manifest.deployments if d.contract_name} if manifest else set()
    )
    contract_to_files = _contract_index(texts)
    scope_files = _files_for(scope_contracts, scope_paths, contract_to_files, focus_file)
    dependencies, unresolved, ambiguous_imports, escaped, dependency_notes = _trace_dependencies(
        scope_files, texts, root, contract_to_files
    )
    notes = list(dependency_notes)
    for record in reads:
        if record.status != "read":
            notes.append(
                f"{record.path} was not read ({record.reason}); "
                "its declarations and imports were not traced"
            )
    if capped:
        notes.append(f"repository enumeration stopped at {MAX_CANDIDATE_FILES} Solidity files")
    ambiguous = tuple(
        (name, tuple(files))
        for name, files in sorted(contract_to_files.items())
        if len(files) > 1 and (name in scope_contracts or name in deployed_contracts)
    )
    ambiguous_files = {path: name for name, files in ambiguous for path in files}

    ranked: list[RankedFile] = []
    for rel in rel_paths:
        text = texts.get(rel, "")
        priority, reason = _rank_one(rel, text, scope_files, deployed_contracts, dependencies)
        if rel in ambiguous_files and priority.value <= Priority.DEPLOYED.value:
            reason = f"{reason}; contract name {ambiguous_files[rel]} is declared in several files"
        ranked.append(RankedFile(rel, priority, reason))

    ranked = _assign_roles(ranked, scope_files, dependencies, _excluded_prefixes(manifest))
    selectable = [item for item in ranked if item.role != "excluded"]
    selectable.sort(
        key=lambda item: (_ROLE_ORDER.get(item.role, 9), item.priority.value, item.path)
    )
    selected_list = [item.path for item in selectable[:limit]]
    selected = tuple(selected_list)
    dropped = tuple(item.path for item in selectable[limit:])
    missing_dependencies = sorted(path for path in dependencies if path not in set(selected))
    if missing_dependencies:
        notes.append(
            "required dependencies were outside the selection budget: "
            + ", ".join(missing_dependencies[:8])
        )
    limited = any(
        "dependency trace stopped" in note or "dependency depth limit" in note for note in notes
    )
    closure_complete = not missing_dependencies and not limited and not capped
    scope_excluded = tuple(item.path for item in ranked if item.role == "excluded")
    return SourceSelection(
        selected=selected,
        ranked=tuple(ranked),
        considered=len(ranked),
        limit=limit,
        truncated=len(selectable) > limit,
        dropped=dropped,
        ambiguous_contracts=ambiguous,
        unresolved_imports=tuple(unresolved),
        ambiguous_imports=tuple(ambiguous_imports),
        escaped_imports=tuple(escaped),
        dependency_notes=tuple(notes),
        reads=tuple(reads),
        repository_candidates=len(paths),
        excluded_count=excluded,
        enumeration_capped=capped,
        closure_complete=closure_complete,
        scope_excluded=scope_excluded,
    )


def select_named(repo_root: Path, files: tuple[str, ...]) -> SourceSelection:
    """The operator named the files. Imports are traced inside that set only.

    A named file that cannot be read is recorded and is not replaced with empty
    source. An import that is not part of the named set stays unresolved.
    """
    root = repo_root.resolve()
    if not root.is_dir():
        return SourceSelection((), (), 0, len(files), False, (), mode="unassessed")
    candidates, excluded, capped = _enumerate(root)
    sources, reads = read_source_files(root, files)
    contract_to_files = _contract_index(sources)
    _resolved, unresolved, ambiguous_imports, escaped, dependency_notes = _trace_dependencies(
        set(sources), sources, root, contract_to_files
    )
    notes = list(dependency_notes)
    for record in reads:
        if record.status != "read":
            notes.append(
                f"{record.path} was not read ({record.reason}); "
                "its declarations and imports were not traced"
            )
    if capped:
        notes.append(f"repository enumeration stopped at {MAX_CANDIDATE_FILES} Solidity files")
    ranked = tuple(
        RankedFile(path, Priority.SCOPE, "named by the operator", "target") for path in files
    )
    return SourceSelection(
        selected=tuple(files),
        ranked=ranked,
        considered=len(files),
        limit=len(files),
        truncated=False,
        dropped=(),
        mode="explicit",
        unresolved_imports=tuple(unresolved),
        ambiguous_imports=tuple(ambiguous_imports),
        escaped_imports=tuple(escaped),
        dependency_notes=tuple(notes),
        reads=reads,
        repository_candidates=len(candidates),
        excluded_count=excluded,
        enumeration_capped=capped,
    )


def _scope_targets(
    manifest: BountyManifest | None, focus_contract: str, focus_file: str
) -> tuple[set[str], set[str]]:
    contracts: set[str] = set()
    paths: set[str] = set()
    if focus_contract:
        contracts.add(focus_contract)
    if focus_file:
        paths.add(focus_file)
    if manifest is not None:
        for asset in manifest.in_scope:
            if asset.kind == "contract":
                contracts.add(asset.identifier)
            elif asset.kind == "path":
                paths.add(asset.identifier)
        contracts.update(name for name in manifest.contract_names if name)
    return contracts, paths


def _contract_index(texts: dict[str, str]) -> dict[str, list[str]]:
    """Every file that declares each contract name (not only the first one)."""
    index: dict[str, list[str]] = {}
    for rel, text in texts.items():
        for name in _CONTRACT_DECL.findall(text):
            files = index.setdefault(name, [])
            if rel not in files:
                files.append(rel)
    return index


def _files_for(
    contracts: set[str],
    paths: set[str],
    contract_to_files: dict[str, list[str]],
    focus_file: str = "",
) -> set[str]:
    """Files for the scoped contracts. A name declared in several files keeps all of
    them unless the operator's focus file disambiguates it."""
    files: set[str] = set()
    for name in contracts:
        declaring = contract_to_files.get(name, [])
        if focus_file and focus_file in declaring:
            files.add(focus_file)
        else:
            files.update(declaring)
    files.update(paths)
    return files


def _base_names(text: str) -> list[str]:
    names: list[str] = []
    for match in _BASES.finditer(text):
        clause = match.group(1) or ""
        for part in clause.split(","):
            found = re.match(r"\s*([A-Za-z_][\w.]*)", part)
            if found:
                names.append(found.group(1).split(".")[-1])
    return names


def _resolve_import(
    target: str, rel: str, texts: dict[str, str], root: Path
) -> tuple[str, str | tuple[str, ...]]:
    """Resolve one import to a single repository path, or report why it cannot.

    A non-relative import matches the full suffix first. A bare basename is used
    only when exactly one file has that name. Several matches stay ambiguous.
    """
    if target.startswith("."):
        resolved = ((root / rel).parent / target).resolve()
        try:
            posix = resolved.relative_to(root).as_posix()
        except ValueError:
            return "escaped", target
        if posix in texts:
            return "resolved", posix
        return "unresolved", target
    suffix = target.lstrip("/")
    matches = [path for path in texts if path == suffix or path.endswith("/" + suffix)]
    if len(matches) == 1:
        return "resolved", matches[0]
    if len(matches) > 1:
        return "ambiguous", tuple(sorted(matches))
    base = suffix.split("/")[-1]
    if base and base != suffix:
        base_matches = [path for path in texts if path == base or path.endswith("/" + base)]
        if len(base_matches) == 1:
            return "resolved", base_matches[0]
        if len(base_matches) > 1:
            return "ambiguous", tuple(sorted(base_matches))
    base_matches = [path for path in texts if path == base or path.endswith("/" + base)]
    if len(base_matches) == 1:
        return "resolved", base_matches[0]
    if len(base_matches) > 1:
        return "ambiguous", tuple(sorted(base_matches))
    return "unresolved", target


def _trace_dependencies(
    scope_files: set[str],
    texts: dict[str, str],
    root: Path,
    contract_to_files: dict[str, list[str]],
) -> tuple[
    set[str],
    list[tuple[str, str]],
    list[tuple[str, tuple[str, ...]]],
    list[tuple[str, str]],
    list[str],
]:
    """Bounded imports and unique bases of the in-scope files. Uncertainty stays visible."""
    resolved: set[str] = set()
    unresolved: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, tuple[str, ...]]] = []
    escaped: list[tuple[str, str]] = []
    notes: list[str] = []
    seen_imports: set[tuple[str, str]] = set()
    queue = [(path, 0) for path in sorted(scope_files) if path in texts]
    visited: set[str] = set()
    while queue:
        rel, depth = queue.pop(0)
        if rel in visited:
            continue
        visited.add(rel)
        if depth > 0 and len(resolved) >= MAX_DEPENDENCY_FILES:
            notes.append(
                f"dependency trace stopped after {MAX_DEPENDENCY_FILES} files; "
                "further imports were not followed"
            )
            break
        if depth >= MAX_DEPENDENCY_DEPTH:
            notes.append(f"dependency depth limit {MAX_DEPENDENCY_DEPTH} reached at {rel}")
            continue
        text = texts.get(rel, "")
        for target in _IMPORT.findall(text):
            key = (rel, target)
            if key in seen_imports:
                continue
            seen_imports.add(key)
            status, payload = _resolve_import(target, rel, texts, root)
            if status == "resolved" and isinstance(payload, str):
                if payload not in scope_files:
                    resolved.add(payload)
                if payload not in visited:
                    queue.append((payload, depth + 1))
            elif status == "ambiguous" and isinstance(payload, tuple):
                ambiguous.append((target, payload))
            elif status == "escaped":
                escaped.append((rel, target))
            else:
                unresolved.append((rel, target))
        for base in _base_names(text):
            declaring = contract_to_files.get(base, [])
            if len(declaring) == 1:
                path = declaring[0]
                if path not in scope_files:
                    resolved.add(path)
                if path not in visited:
                    queue.append((path, depth + 1))
            elif len(declaring) > 1:
                ambiguous.append((base, tuple(declaring)))
            else:
                unresolved.append((rel, base))
    return resolved, unresolved, ambiguous, escaped, notes


_ROLE_ORDER = {
    "target": 0,
    "compile_dependency": 1,
    "analysis_context": 2,
    "excluded": 3,
}


def _role_counts(ranked: tuple[RankedFile, ...] | list[RankedFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in ranked:
        counts[item.role] = counts.get(item.role, 0) + 1
    return dict(sorted(counts.items()))


def _excluded_prefixes(manifest: BountyManifest | None) -> tuple[str, ...]:
    if manifest is None:
        return ()
    prefixes: list[str] = []
    for asset in manifest.out_of_scope:
        if asset.kind != "path":
            continue
        text = asset.identifier.replace("\\", "/").strip().strip("/")
        if text and ".." not in text.split("/"):
            prefixes.append(text)
    return tuple(prefixes)


def _path_excluded(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)


def _assign_roles(
    ranked: list[RankedFile],
    scope_files: set[str],
    dependencies: set[str],
    prefixes: tuple[str, ...],
) -> list[RankedFile]:
    """Target and its compile closure outrank extra files. Excluded paths stay out
    of the budget unless the target actually imports them."""
    assigned: list[RankedFile] = []
    for item in ranked:
        if item.path in scope_files:
            role = "target"
        elif item.path in dependencies:
            role = "compile_dependency"
        elif _path_excluded(item.path, prefixes):
            role = "excluded"
        else:
            role = "analysis_context"
        assigned.append(RankedFile(item.path, item.priority, item.reason, role))
    assigned.sort(key=lambda item: (item.priority.value, item.path))
    return assigned


def _rank_one(
    rel: str,
    text: str,
    scope_files: set[str],
    deployed_contracts: set[str],
    imported: set[str],
) -> tuple[Priority, str]:
    if rel in scope_files or any(rel.endswith("/" + f) for f in scope_files):
        return Priority.SCOPE, "in-scope asset"
    declared = set(_CONTRACT_DECL.findall(text))
    if declared & deployed_contracts:
        return Priority.DEPLOYED, "declares a deployed contract"
    if _is_proxy_like(text):
        return Priority.PROXY, "proxy / implementation code (delegatecall + slot/base)"
    if rel in imported:
        return Priority.IMPORTED_BY_SCOPE, "import or base required by an in-scope contract"
    if any(pattern.search(text) for pattern in _SENSITIVE):
        return Priority.SECURITY_SENSITIVE, "security-sensitive construct present"
    return Priority.OTHER, "not prioritized"


def _is_proxy_like(text: str) -> bool:
    has_delegate = "delegatecall(" in text
    erc1967 = "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc" in text.lower()
    bases = any(
        token in text
        for token in ("UUPSUpgradeable", "TransparentUpgradeableProxy", "BeaconProxy", "diamondCut")
    )
    return bool(bases or (has_delegate and ("fallback" in text or erc1967)))

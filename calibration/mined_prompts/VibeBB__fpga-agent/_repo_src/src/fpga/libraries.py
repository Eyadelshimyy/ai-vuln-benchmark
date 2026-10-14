"""External HDL library file resolution.

A contract library declares files explicitly (``files``) and/or names
design units (``components``): VHDL entities/packages or Verilog/SV
modules. Components are resolved by scanning the checkout at ``root``:

- every ``entity``/``package``/``package body`` (VHDL) and
  ``module``/``package`` (Verilog) declaration is indexed to its file,
- ``use <lib>.<unit>``, ``entity <lib>.<unit>`` and ``use work.<unit>``
  references become dependency edges (``work`` and the library's own
  name mean "inside this library"),
- `` `include`` directives and Verilog ``<name>.f`` file lists are
  followed,
- ``import <pkg>::`` references become edges for SystemVerilog.

The dependency closure is returned in compile order (dependencies
first). References to a *different* VHDL library name are validated
against the contract's declared libraries, which must appear earlier.
The special component ``all`` resolves to every design unit found under
``root`` in dependency order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contract import FpgaContract, Library

BUILTIN_VHDL_LIBS = frozenset({"ieee", "ieee_proposed", "std", "work"})

# Directories that never hold synthesizable components.
_SKIP_DIRS = frozenset(
    {
        ".git",
        "sim",
        "test",
        "tests",
        "testbench",
        "tb",
        "doc",
        "docs",
        "example",
        "examples",
        "formal",
        "fv",
        "syn",
        "uvvm",
        "verification",
        "ucf",
        "prj",
        "tools",
        "scripts",
        "lint",
        "regression",
    }
)

_VHDL_DECL = re.compile(
    r"^\s*(entity|package)\s+(body\s+)?([a-zA-Z][a-zA-Z0-9_]*)\s+is\b",
    re.IGNORECASE | re.MULTILINE,
)
_VHDL_USE = re.compile(
    r"\b(?:use|entity|configuration)\s+([a-zA-Z][a-zA-Z0-9_]*)\.([a-zA-Z][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)
_VLOG_DECL = re.compile(
    r"^\s*(module|macromodule|package)\s+([a-zA-Z_][a-zA-Z0-9_$]*)", re.MULTILINE
)
_VLOG_INCLUDE = re.compile(r'`include\s+"([^"]+)"')
_VLOG_IMPORT = re.compile(r"\bimport\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*::")
# Candidate direct instantiations: ``name inst(`` or ``name #(...) inst(``.
# Matched names are kept only when they resolve to an indexed provider, so
# keywords/primitives never become phantom dependencies.
_VLOG_INST = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_$]*)\s*(?:#\s*\((?:[^()]|\([^()]*\))*\)\s*)?"
    r"[a-zA-Z_][a-zA-Z0-9_$]*\s*\("
)
_COMMENT_VHDL = re.compile(r"--.*$", re.MULTILINE)
_COMMENT_VLOG_LINE = re.compile(r"//.*$", re.MULTILINE)
_COMMENT_VLOG_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)

_HDL_SUFFIXES = {
    "vhdl": {".vhd", ".vhdl"},
    "verilog": {".v", ".vh", ".sv", ".svh"},
    "systemverilog": {".v", ".vh", ".sv", ".svh"},
}


@dataclass(frozen=True)
class Ref:
    """A dependency edge made by a scanned file."""

    library: str  # referenced VHDL library, "" for Verilog-style refs
    unit: str


class Index:
    """Unit declaration index for one library checkout."""

    def __init__(self) -> None:
        self.providers: dict[str, Path] = {}
        self.ambiguous: dict[str, list[Path]] = {}
        self.bodies: dict[str, Path] = {}  # VHDL package bodies
        self.refs: dict[Path, set[Ref]] = {}
        self.filelists: dict[str, Path] = {}  # "<stem>.f" lists
        self.kinds: dict[str, str] = {}  # unit -> entity/module/package/...
        self.pending_insts: dict[Path, set[str]] = {}  # instantiation candidates
        self.cond_includes: dict[Path, set[str]] = {}  # `ifdef`-guarded `include`s
        self.root: Path | None = None  # include search fallback

    @property
    def units(self) -> list[str]:
        return sorted(self.providers)


def _hdl_files(root: Path, language: str, exclude: list[str] | None = None) -> list[Path]:
    suffixes = _HDL_SUFFIXES[language]
    return sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in suffixes
        and not ({part.lower() for part in path.parts} & _SKIP_DIRS)
        and not any(path.relative_to(root).match(pattern) for pattern in (exclude or []))
    )


def _strip_comments(text: str, language: str) -> str:
    if language == "vhdl":
        return _COMMENT_VHDL.sub("", text)
    text = _COMMENT_VLOG_BLOCK.sub("", text)
    return _COMMENT_VLOG_LINE.sub("", text)


def scan_library(root: Path, language: str, exclude: list[str] | None = None) -> Index:
    """Index every design unit declared under ``root``."""
    index = Index()
    index.root = root
    for path in _hdl_files(root, language, exclude):
        try:
            text = _strip_comments(path.read_text(encoding="utf-8", errors="replace"), language)
        except OSError:
            continue
        refs: set[Ref] = set()
        if language == "vhdl":
            for kind, body, name in _VHDL_DECL.findall(text):
                key = name.lower()
                if body:
                    index.bodies.setdefault(key, path)
                    index.kinds.setdefault(key, f"{kind.lower()}_body")
                elif key in index.providers and index.providers[key] != path:
                    index.ambiguous.setdefault(key, []).append(path)
                else:
                    index.providers[key] = path
                    index.kinds.setdefault(key, kind.lower())
            for lib, unit in _VHDL_USE.findall(text):
                refs.add(Ref(lib.lower(), unit.lower()))
        else:
            for kind, name in _VLOG_DECL.findall(text):
                key = name.lower()
                if key in index.providers and index.providers[key] != path:
                    index.ambiguous.setdefault(key, []).append(path)
                else:
                    index.providers[key] = path
                    index.kinds.setdefault(
                        key, "module" if kind.lower() == "macromodule" else kind.lower()
                    )
            depth = 0
            cond: set[str] = set()
            for line in text.splitlines():
                stripped = line.strip()
                if re.match(r"`ifn?def\b", stripped):
                    depth += 1
                elif stripped.startswith("`endif"):
                    depth = max(0, depth - 1)
                    continue
                match = _VLOG_INCLUDE.search(stripped)
                if match:
                    if depth:
                        cond.add(match.group(1))
                    else:
                        refs.add(Ref("", match.group(1)))
            if cond:
                index.cond_includes[path] = cond
            for pkg in _VLOG_IMPORT.findall(text):
                refs.add(Ref("", pkg.lower()))
        index.refs[path] = refs
        if language != "vhdl":
            index.pending_insts[path] = {name.lower() for name in _VLOG_INST.findall(text)}
    for path in root.rglob("*.f"):
        if not ({part.lower() for part in path.parts} & _SKIP_DIRS) and not any(
            path.relative_to(root).match(pattern) for pattern in (exclude or [])
        ):
            index.filelists[path.stem.lower()] = path
    if language != "vhdl":
        known = set(index.providers) | set(index.ambiguous)
        for path, candidates in index.pending_insts.items():
            index.refs[path] |= {Ref("", name) for name in candidates & known}
        index.pending_insts.clear()
    return index


def _filelist_entries(path: Path, root: Path | None = None) -> list[Path]:
    entries: list[Path] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        # Some lists carry per-entry keywords (e.g. hazard3's ``file <path>``)
        # or env-style root prefixes (e.g. neorv32's ``$NEORV32_HOME/...``).
        tokens = line.split()
        entry = tokens[-1]
        if entry.startswith("$") and root is not None:
            entry = str(root / entry.split("/", 1)[-1])
        else:
            entry = str(path.parent / entry)
        entries.append(Path(entry).resolve())
    return entries


class Resolved:
    """Resolution result for one contract library."""

    def __init__(self) -> None:
        self.files: list[Path] = []
        self.includes: set[Path] = set()  # dirs of `include`d header files
        self.problems: list[str] = []
        self.external: set[str] = set()  # earlier libraries it needs


def _include_file(index: Index, path: Path) -> None:
    """Register ``path`` and files it `` `include``s in the ref graph."""
    queue = [path]
    seen: set[Path] = set()
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        refs = index.refs.setdefault(current, set())
        for ref in list(refs):
            if ref.library == "" and ref.unit.endswith((".v", ".vh", ".sv", ".svh")):
                target = (current.parent / ref.unit).resolve()
                if not target.is_file() and index.root is not None:
                    target = (index.root / ref.unit).resolve()
                if target.is_file() and target not in index.refs:
                    queue.append(target)


def resolve_library(contract_path: Path, lib: Library, declared: list[str]) -> Resolved:
    """Resolve ``lib``'s files/components to an ordered file list.

    ``declared`` lists every library name in contract declaration order;
    references to other VHDL libraries must point to earlier entries.
    """
    from .contract import resolve

    result = Resolved()
    root = resolve(contract_path, lib.root)
    if not root.is_dir():
        result.problems.append(f"{lib.name}: root {lib.root} does not exist")
        return result

    index = scan_library(root, lib.language, lib.exclude)
    seeds: list[Path] = []
    for component in lib.components:
        filelist = component.lower().removesuffix(".f")
        key = component.lower()
        if component == "all":
            seeds += list(index.refs)
        elif filelist in index.filelists:
            seeds += _filelist_entries(index.filelists[filelist], index.root)
        elif key in index.ambiguous:
            choices = ", ".join(p.name for p in index.ambiguous[key])
            result.problems.append(
                f"{lib.name}: component {component!r} is ambiguous ({choices}); use files"
            )
        elif key in index.providers:
            seeds.append(index.providers[key])
        elif (root / component).is_dir():
            seeds += [
                path
                for path in index.refs
                if (root / component) in path.parents or path.parent == root / component
            ]
        else:
            names = ", ".join(index.units[:10])
            result.problems.append(
                f"{lib.name}: component {component!r} not found "
                f"(units include: {names}{', ...' if len(index.units) > 10 else ''})"
            )

    ordered: list[Path] = []
    state: dict[Path, str] = {}  # path -> "visiting" | "done"
    position = {name: i for i, name in enumerate(declared)}

    def visit(path: Path, via: str) -> None:
        if state.get(path) == "done":
            return
        if state.get(path) == "visiting":
            result.problems.append(f"{lib.name}: dependency cycle through {path.name}")
            return
        state[path] = "visiting"
        for ref in sorted(index.refs.get(path, ()), key=lambda r: (r.library, r.unit)):
            if ref.library in BUILTIN_VHDL_LIBS and ref.library != "work":
                continue
            if ref.library == "" and ref.unit.endswith((".v", ".vh", ".sv", ".svh")):
                target = (path.parent / ref.unit).resolve()
                if not target.is_file() and index.root is not None:
                    target = (index.root / ref.unit).resolve()
                if not target.is_file():
                    if ref.unit not in index.cond_includes.get(path, set()):
                        result.problems.append(
                            f"{lib.name}: {path.name} includes missing {ref.unit}"
                        )
                    continue
                result.includes.add(target.parent)
                _include_file(index, target)
                visit(target, path.name)
                continue
            target_lib = lib.name if ref.library in ("work", lib.name, "") else ref.library
            if target_lib == lib.name:
                provider = index.providers.get(ref.unit)
                if provider is None:
                    if ref.unit not in {"all", "work"}:
                        result.problems.append(
                            f"{lib.name}: {path.name} needs {ref.unit} "
                            f"(no declaration found under {lib.root})"
                        )
                    continue
                if provider != path:
                    visit(provider, path.name)
                body = index.bodies.get(ref.unit)
                if body is not None and body != provider:
                    visit(body, path.name)
            else:
                if ref.library not in position:
                    result.problems.append(
                        f"{lib.name}: {path.name} uses library {ref.library!r} "
                        f"which is not declared in libraries[]"
                    )
                elif position[ref.library] >= position[lib.name]:
                    result.problems.append(
                        f"{lib.name}: {path.name} uses library {ref.library!r} "
                        f"declared after {lib.name!r}; it must come earlier"
                    )
                else:
                    result.external.add(ref.library)
        state[path] = "done"
        if path not in ordered and path.suffix.lower() not in {".vh", ".svh"}:
            ordered.append(path)
        # A VHDL package body of a unit declared here compiles after the
        # spec — emit it now that the spec is ordered.
        for name, spec in index.providers.items():
            if spec == path and name in index.bodies and index.bodies[name] != path:
                visit(index.bodies[name], path.name)

    for seed in seeds:
        if not seed.is_file():
            result.problems.append(f"{lib.name}: {seed} does not exist")
            continue
        visit(seed, "")

    result.files = ordered
    for item in lib.files:
        path = (root / item).resolve()
        if path not in result.files:
            result.files.append(path)
    return result


def resolve_contract_libraries(contract: FpgaContract, contract_path: Path) -> dict[str, Resolved]:
    """Resolve every declared library in the contract."""
    declared = [lib.name for lib in contract.libraries]
    return {lib.name: resolve_library(contract_path, lib, declared) for lib in contract.libraries}

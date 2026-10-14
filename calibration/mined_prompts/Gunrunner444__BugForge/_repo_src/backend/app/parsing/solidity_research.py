"""Bounded Solidity research model shared by the Phase 50 analyzers.

The model is a deliberately small structural reader: contracts, inheritance,
state variables, modifiers, and functions with their bodies. It strips comments,
keeps strings, and never executes or compiles anything. Every walk is capped, and
a cap that was hit is reported as ``truncated`` instead of being hidden.

Facts here are static. Nothing in this module can mark a finding verified, and a
fact the source does not show stays absent or ``unknown``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

MAX_FILES = 64
MAX_FILE_BYTES = 400_000
MAX_CONTRACTS = 256
MAX_FUNCTIONS = 2000
MAX_BASE_DEPTH = 12
MAX_CALLS_PER_FUNCTION = 200

_KEYWORDS = frozenset(
    {
        "if", "for", "while", "do", "return", "returns", "require", "assert", "revert", "emit",
        "new", "delete", "unchecked", "try", "catch", "else", "function", "mapping", "type",
        "abi", "keccak256", "sha256", "ripemd160", "ecrecover", "addmod", "mulmod", "assembly",
        "uint", "int", "bool", "address", "bytes", "string", "payable", "this", "super",
    }
)  # fmt: skip
_VISIBILITY = ("external", "public", "internal", "private")
_MUTABILITY = ("payable", "view", "pure")
_HEADER_WORDS = frozenset(
    {*_VISIBILITY, *_MUTABILITY, "virtual", "override", "returns", "constant", "nonpayable"}
)
_CONTRACT_RE = re.compile(
    r"\b(?:abstract\s+)?(contract|interface|library)\s+([A-Za-z_]\w*)\b([^{;]*)\{"
)
_MEMBER_RE = re.compile(
    r"\b(function\s+([A-Za-z_]\w*)|constructor|receive|fallback|modifier\s+([A-Za-z_]\w*))\s*"
    r"(?=[({])"
)


@dataclass(frozen=True)
class Param:
    type_name: str
    name: str


@dataclass(frozen=True)
class RFunction:
    file: str
    contract: str
    name: str
    kind: str
    signature: str
    line: int
    visibility: str
    mutability: str
    modifiers: tuple[str, ...]
    params: tuple[Param, ...]
    returns: tuple[Param, ...]
    body: str
    has_body: bool

    @property
    def exposed(self) -> bool:
        return self.visibility in {"public", "external"} and self.kind in {
            "function",
            "receive",
            "fallback",
        }

    @property
    def identity(self) -> str:
        return f"{self.contract}.{self.signature}"

    @property
    def semantic_id(self) -> str:
        return f"{self.contract}.{self.name}:{self.line}"

    def param_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.params if item.name)


@dataclass(frozen=True)
class RModifier:
    contract: str
    name: str
    params: tuple[Param, ...]
    body: str
    line: int


@dataclass(frozen=True)
class RContract:
    name: str
    kind: str
    file: str
    line: int
    bases: tuple[str, ...]
    head: str
    state_vars: tuple[tuple[str, str], ...]
    functions: tuple[RFunction, ...]
    modifiers: tuple[RModifier, ...]
    body: str
    pragma: str = ""

    @property
    def is_interface(self) -> bool:
        return self.kind == "interface"


@dataclass
class ResearchModel:
    contracts: dict[str, RContract] = field(default_factory=dict)
    ambiguous: frozenset[str] = frozenset()
    truncated: bool = False
    notes: tuple[str, ...] = ()
    digest: str = ""
    # path -> sha256 of every file this model was built from, in canonical order.
    # Empty means the model has no source attestation. A later caller must not
    # treat that as a match for whatever source text is supplied.
    source_hashes: tuple[tuple[str, str], ...] = ()
    source_set_digest: str = ""

    def bases_of(self, contract: str) -> tuple[str, ...]:
        """Base contracts, nearest first, bounded and cycle-safe."""
        order: list[str] = []
        stack = [(contract, 0)]
        seen = {contract}
        while stack:
            name, depth = stack.pop(0)
            item = self.contracts.get(name)
            if item is None or depth >= MAX_BASE_DEPTH:
                continue
            for base in item.bases:
                if base not in seen:
                    seen.add(base)
                    order.append(base)
                    stack.append((base, depth + 1))
        return tuple(order)

    def lineage(self, contract: str) -> tuple[str, ...]:
        return (contract, *self.bases_of(contract))

    def functions_of(self, contract: str, *, inherited: bool = True) -> tuple[RFunction, ...]:
        names = self.lineage(contract) if inherited else (contract,)
        seen: set[str] = set()
        found: list[RFunction] = []
        for owner in names:
            item = self.contracts.get(owner)
            if item is None:
                continue
            for function in item.functions:
                key = function.signature if function.kind == "function" else function.kind
                if key in seen:
                    continue
                seen.add(key)
                found.append(function)
        return tuple(found)

    def function(self, contract: str, name: str) -> RFunction | None:
        for function in self.functions_of(contract):
            if function.kind == "function" and function.name == name:
                return function
        return None

    def functions_named(self, contract: str, name: str) -> tuple[RFunction, ...]:
        return tuple(
            item
            for item in self.functions_of(contract)
            if item.kind == "function" and item.name == name
        )

    def modifier(self, contract: str, name: str) -> RModifier | None:
        for owner in self.lineage(contract):
            item = self.contracts.get(owner)
            if item is None:
                continue
            for modifier in item.modifiers:
                if modifier.name == name:
                    return modifier
        return None

    def state_vars(self, contract: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for owner in reversed(self.lineage(contract)):
            item = self.contracts.get(owner)
            if item is not None:
                found.update(dict(item.state_vars))
        return found

    def contract_for_type(self, type_name: str) -> str:
        """The contract a declared type names, or empty when it is not in the model."""
        base = re.sub(r"\[.*\]", "", type_name).strip()
        base = base.replace("payable", "").strip()
        return base if base in self.contracts and base not in self.ambiguous else ""

    def all_functions(self) -> tuple[RFunction, ...]:
        return tuple(
            function
            for name in sorted(self.contracts)
            for function in self.contracts[name].functions
        )

    def implementers(self, function: RFunction) -> tuple[RFunction, ...]:
        """Concrete bodies in other contracts with the same signature."""
        return tuple(
            item
            for item in self.all_functions()
            if item.signature == function.signature
            and item.has_body
            and item.contract != function.contract
        )


# ---- text helpers ---------------------------------------------------------------------------


def strip_comments(text: str) -> str:
    """Blank comments, keep every offset and newline."""
    out = list(text)
    i = 0
    size = len(text)
    quote = ""
    while i < size:
        char = text[i]
        if quote:
            if char == "\\":
                i += 2
                continue
            if char == quote:
                quote = ""
            i += 1
            continue
        if char in "\"'":
            quote = char
            i += 1
            continue
        if text.startswith("//", i):
            while i < size and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = size if end == -1 else end + 2
            for j in range(i, end):
                if out[j] != "\n":
                    out[j] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def skeleton(clean: str) -> str:
    """Blank string interiors so brackets inside strings never confuse a scan."""
    out = list(clean)
    i = 0
    size = len(clean)
    while i < size:
        char = clean[i]
        if char in "\"'":
            quote = char
            i += 1
            while i < size and clean[i] != quote:
                if clean[i] == "\\":
                    if out[i] != "\n":
                        out[i] = " "
                    i += 1
                if i < size and out[i] != "\n":
                    out[i] = " "
                i += 1
            i += 1
            continue
        i += 1
    return "".join(out)


_PAIRS = {"(": ")", "{": "}", "[": "]"}


def match_close(skel: str, open_index: int) -> int:
    """Index of the bracket closing ``skel[open_index]``, or -1."""
    opener = skel[open_index]
    closer = _PAIRS.get(opener)
    if closer is None:
        return -1
    depth = 0
    for i in range(open_index, len(skel)):
        char = skel[i]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top(text: str, separator: str = ",") -> list[str]:
    """Split on top-level separators in text that is already a skeleton or has no strings."""
    parts: list[str] = []
    depth = 0
    start = 0
    quote = ""
    for i, char in enumerate(text):
        if quote:
            if char == quote and text[i - 1] != "\\":
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail or parts:
        parts.append(tail)
    return parts


def normalize_type(raw: str) -> str:
    text = " ".join(raw.split())
    text = re.sub(r"\b(memory|calldata|storage|payable)\b", "", text).strip()
    text = re.sub(r"\buint\b", "uint256", text)
    text = re.sub(r"\bint\b", "int256", text)
    return re.sub(r"\s+", "", text)


def parse_params(text: str) -> tuple[Param, ...]:
    found: list[Param] = []
    for part in split_top(text):
        if not part:
            continue
        words = part.replace("\n", " ").split()
        cleaned = [w for w in words if w not in {"memory", "calldata", "storage", "payable"}]
        if not cleaned:
            continue
        if len(cleaned) >= 2 and re.fullmatch(r"[A-Za-z_]\w*", cleaned[-1]):
            found.append(Param(normalize_type(" ".join(cleaned[:-1])), cleaned[-1]))
        else:
            found.append(Param(normalize_type(" ".join(cleaned)), ""))
    return tuple(found)


def identifiers(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*", text))


def mentions(text: str, name: str) -> bool:
    return bool(name) and re.search(rf"(?<![\w.]){re.escape(name)}\b", text) is not None


# ---- model construction ---------------------------------------------------------------------


def build_research_model(
    sources: Mapping[str, str],
    *,
    max_files: int = MAX_FILES,
    max_contracts: int = MAX_CONTRACTS,
    max_functions: int = MAX_FUNCTIONS,
) -> ResearchModel:
    model = ResearchModel()
    notes: list[str] = []
    contracts: dict[str, RContract] = {}
    ambiguous: set[str] = set()
    function_count = 0
    ordered = sorted(sources)
    if len(ordered) > max_files:
        model.truncated = True
        notes.append(f"only {max_files} of {len(ordered)} files were read")
    hasher = hashlib.sha256()
    considered = ordered[:max_files]
    file_hashes: list[tuple[str, str]] = []
    for path in considered:
        text = sources[path]
        file_hashes.append((path, hashlib.sha256(text.encode("utf-8")).hexdigest()))
        hasher.update(path.encode())
        hasher.update(hashlib.sha256(text.encode()).digest())
        if len(text) > MAX_FILE_BYTES:
            model.truncated = True
            notes.append(f"{path} exceeds {MAX_FILE_BYTES} bytes and was skipped")
            continue
        for contract in _contracts_in_file(path, text):
            if len(contracts) >= max_contracts or function_count >= max_functions:
                model.truncated = True
                notes.append("contract or function limit reached")
                break
            if contract.name in contracts:
                ambiguous.add(contract.name)
                continue
            function_count += len(contract.functions)
            contracts[contract.name] = contract
    model.contracts = contracts
    model.ambiguous = frozenset(ambiguous)
    model.notes = tuple(notes)
    model.digest = hasher.hexdigest()[:16]
    model.source_hashes = tuple(file_hashes)
    model.source_set_digest = canonical_source_digest(dict(file_hashes))
    return model


def _posix_repo_path(path: str) -> str:
    """A relative POSIX path. Absolute paths and ``..`` aliases are rejected."""
    pure = PurePosixPath(str(path).replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        return ""
    text = pure.as_posix()
    return text[2:] if text.startswith("./") else text


def canonical_source_digest(hashes: Mapping[str, str]) -> str:
    """Deterministic digest of a path→sha256 map. Path order cannot change it."""
    payload = json.dumps(sorted(hashes.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_source_problems(model: ResearchModel, sources: Mapping[str, str]) -> tuple[str, ...]:
    """Why ``sources`` is not the text this model was parsed from.

    A model parsed from source A must not be trusted with a mapping of source B,
    even when the contract name and function signature are unchanged. Missing
    attestation is not treated as a match.
    """
    problems: list[str] = []
    normalized: dict[str, str] = {}
    for path, text in sources.items():
        key = _posix_repo_path(path)
        if not key:
            problems.append(f"{path} is not a normalized relative path")
            continue
        if key in normalized:
            problems.append(f"{path} duplicates the source path {key}")
            continue
        normalized[key] = text
    recorded = dict(model.source_hashes)
    if not recorded:
        if model.contracts or normalized:
            problems.append("the parsed model has no source attestation")
        return tuple(problems[:12])
    current = {
        path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in normalized.items()
    }
    for path, digest_value in current.items():
        expected = recorded.get(path)
        if expected is None:
            problems.append(f"{path} is not part of the model's parsed source")
        elif expected != digest_value:
            problems.append(f"{path} does not match the model's parsed source")
    for path in recorded:
        if path not in current:
            problems.append(f"{path} from the model's parsed source is missing")
    if (
        model.source_set_digest
        and canonical_source_digest(recorded) != canonical_source_digest(current)
        and not problems
    ):
        problems.append("the model source-set digest does not match the supplied sources")
    return tuple(problems[:12])


def _contracts_in_file(path: str, text: str) -> Iterator[RContract]:
    clean = strip_comments(text)
    skel = skeleton(clean)
    pragma_match = re.search(r"pragma\s+solidity\s+([^;]+);", clean)
    pragma = pragma_match.group(1).strip() if pragma_match else ""
    newlines = [i for i, c in enumerate(clean) if c == "\n"]
    cursor = 0
    while True:
        match = _CONTRACT_RE.search(skel, cursor)
        if match is None:
            return
        open_index = match.end() - 1
        close = match_close(skel, open_index)
        if close == -1:
            return
        kind, name, head = match.group(1), match.group(2), match.group(3)
        yield _build_contract(
            path, clean, skel, kind, name, head, open_index + 1, close, newlines, pragma
        )
        cursor = close + 1


def _line(newlines: list[int], offset: int) -> int:
    lo, hi = 0, len(newlines)
    while lo < hi:
        mid = (lo + hi) // 2
        if newlines[mid] < offset:
            lo = mid + 1
        else:
            hi = mid
    return lo + 1


def _bases(head: str) -> tuple[str, ...]:
    match = re.search(r"\bis\b(.*)$", head, re.S)
    if not match:
        return ()
    bases = []
    for part in split_top(match.group(1)):
        found = re.match(r"\s*([A-Za-z_][\w.]*)", part)
        if found:
            bases.append(found.group(1).split(".")[-1])
    return tuple(bases)


def _build_contract(
    path: str,
    clean: str,
    skel: str,
    kind: str,
    name: str,
    head: str,
    start: int,
    end: int,
    newlines: list[int],
    pragma: str,
) -> RContract:
    body_skel = skel[start:end]
    body_clean = clean[start:end]
    depth = _depth_map(body_skel)
    functions: list[RFunction] = []
    modifiers: list[RModifier] = []
    cursor = 0
    for member in _MEMBER_RE.finditer(body_skel):
        if member.start() < cursor or depth[member.start()] != 0:
            continue
        built = _member(path, name, kind, body_clean, body_skel, member, start, newlines)
        if built is None:
            continue
        item, stop = built
        cursor = stop
        if isinstance(item, RModifier):
            modifiers.append(item)
        else:
            functions.append(item)
    state = _state_vars(body_skel, body_clean, depth)
    return RContract(
        name=name,
        kind=kind,
        file=path,
        line=_line(newlines, start),
        bases=_bases(head),
        head=" ".join(head.split())[:300],
        state_vars=state,
        functions=tuple(functions),
        modifiers=tuple(modifiers),
        body=body_clean,
        pragma=pragma,
    )


def _depth_map(skel: str) -> list[int]:
    depth = 0
    out: list[int] = []
    for char in skel:
        if char == "}":
            depth -= 1
        out.append(max(depth, 0))
        if char == "{":
            depth += 1
    return out


def _member(
    path: str,
    contract: str,
    contract_kind: str,
    clean: str,
    skel: str,
    match: re.Match[str],
    base_offset: int,
    newlines: list[int],
) -> tuple[RFunction | RModifier, int] | None:
    word = match.group(1).split()[0]
    name = match.group(2) or match.group(3) or word
    index = match.end()
    params_text = ""
    if skel[index] == "(":
        close = match_close(skel, index)
        if close == -1:
            return None
        params_text = clean[index + 1 : close]
        index = close + 1
    header_start = index
    while index < len(skel) and skel[index] not in "{;":
        if skel[index] == "(":
            close = match_close(skel, index)
            if close == -1:
                return None
            index = close + 1
        else:
            index += 1
    if index >= len(skel):
        return None
    header = clean[header_start:index]
    has_body = skel[index] == "{"
    body = ""
    stop = index + 1
    if has_body:
        close = match_close(skel, index)
        if close == -1:
            return None
        body = clean[index + 1 : close]
        stop = close + 1
    line = _line(newlines, base_offset + match.start())
    params = parse_params(params_text)
    if word == "modifier":
        return RModifier(contract, name, params, body, line), stop
    returns_match = re.search(r"\breturns\s*\(", header)
    returns: tuple[Param, ...] = ()
    if returns_match:
        open_at = header.index("(", returns_match.start())
        close_at = match_close(skeleton(header), open_at)
        if close_at != -1:
            returns = parse_params(header[open_at + 1 : close_at])
    visibility, mutability, modifier_names = _header_facts(header, contract_kind)
    kind = "function" if word == "function" else word
    signature = f"{name}({','.join(item.type_name for item in params)})"
    if kind != "function":
        signature = f"{kind}()"
    return (
        RFunction(
            file=path,
            contract=contract,
            name=name if kind == "function" else kind,
            kind=kind,
            signature=signature,
            line=line,
            visibility=visibility,
            mutability=mutability,
            modifiers=modifier_names,
            params=params,
            returns=returns,
            body=body,
            has_body=has_body,
        ),
        stop,
    )


def _header_facts(header: str, contract_kind: str) -> tuple[str, str, tuple[str, ...]]:
    text = re.sub(r"\breturns\s*\(", "returns (", header)
    skel = skeleton(text)
    stripped = list(text)
    # Remove balanced groups after override and returns so their words are not modifiers.
    for keyword in ("returns", "override"):
        for found in re.finditer(rf"\b{keyword}\b\s*\(", skel):
            open_at = skel.index("(", found.start())
            close_at = match_close(skel, open_at)
            if close_at != -1:
                for i in range(found.start(), close_at + 1):
                    stripped[i] = " "
    flat = "".join(stripped)
    visibility = next(
        (v for v in _VISIBILITY if re.search(rf"\b{v}\b", flat)),
        "external" if contract_kind == "interface" else "public",
    )
    mutability = next((m for m in _MUTABILITY if re.search(rf"\b{m}\b", flat)), "nonpayable")
    names: list[str] = []
    for part in re.finditer(r"\b([A-Za-z_]\w*)\b(\s*\([^)]*\))?", flat):
        word = part.group(1)
        if word in _HEADER_WORDS:
            continue
        names.append(f"{word}{part.group(2) or ''}".replace(" ", "")[:80])
    return visibility, mutability, tuple(names)


_NON_VARIABLE = (
    "function", "event", "error", "using", "modifier", "constructor", "struct", "enum",
    "receive", "fallback", "pragma", "import", "emit", "type",
)  # fmt: skip


def _state_vars(skel: str, clean: str, depth: list[int]) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    start = 0
    for i, char in enumerate(skel):
        if depth[i] != 0:
            continue
        if char == "}":
            start = i + 1
            continue
        if char != ";":
            continue
        statement = clean[start:i].strip()
        start = i + 1
        if not statement or statement.split()[0] in _NON_VARIABLE:
            continue
        decl = re.split(r"(?<![=!<>])=(?![=>])", statement, maxsplit=1)[0].strip()
        if decl.startswith("mapping"):
            close = match_close(skeleton(decl), decl.index("(")) if "(" in decl else -1
            if close == -1:
                continue
            type_text = decl[: close + 1]
            rest = decl[close + 1 :].split()
        else:
            words = decl.split()
            if len(words) < 2:
                continue
            type_text, rest = words[0], words[1:]
        names = [w for w in rest if re.fullmatch(r"[A-Za-z_]\w*", w)]
        flags = {
            "public", "private", "internal", "constant", "immutable", "override", "transient",
        }  # fmt: skip
        names = [w for w in names if w not in flags]
        if names:
            found.append((names[-1], normalize_type(type_text)))
    return tuple(found)


# ---- expression helpers ---------------------------------------------------------------------


@dataclass(frozen=True)
class MemberCall:
    receiver: str
    name: str
    options: str
    arguments: tuple[str, ...]
    start: int
    end: int
    text: str


def member_calls(body: str) -> list[MemberCall]:
    """``receiver.name{options}(args)`` calls in source order, capped."""
    skel = skeleton(body)
    found: list[MemberCall] = []
    for match in re.finditer(r"\.\s*([A-Za-z_]\w*)\s*(?=[({])", skel):
        index = match.end()
        options = ""
        if skel[index] == "{":
            close = match_close(skel, index)
            if close == -1:
                continue
            options = body[index + 1 : close]
            index = close + 1
            while index < len(skel) and skel[index].isspace():
                index += 1
        if index >= len(skel) or skel[index] != "(":
            continue
        close = match_close(skel, index)
        if close == -1:
            continue
        receiver = _receiver(skel, body, match.start())
        arguments = tuple(split_top(body[index + 1 : close]))
        found.append(
            MemberCall(
                receiver=receiver,
                name=match.group(1),
                options=options,
                arguments=arguments if arguments != ("",) else (),
                start=match.start() - len(receiver),
                end=close + 1,
                text=body[match.start() - len(receiver) : close + 1],
            )
        )
        if len(found) >= MAX_CALLS_PER_FUNCTION:
            break
    return found


def _receiver(skel: str, body: str, dot: int) -> str:
    i = dot
    while i > 0:
        char = skel[i - 1]
        if char in ")]":
            opener = "(" if char == ")" else "["
            depth = 0
            j = i - 1
            while j >= 0:
                if skel[j] == char:
                    depth += 1
                elif skel[j] == opener:
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j < 0:
                break
            i = j
        elif char.isalnum() or char in "_.":
            i -= 1
        else:
            break
    return body[i:dot].strip()


def plain_calls(body: str) -> list[tuple[str, tuple[str, ...], int]]:
    """``name(args)`` calls that are not member calls and not language keywords."""
    skel = skeleton(body)
    found: list[tuple[str, tuple[str, ...], int]] = []
    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", skel):
        name = match.group(1)
        if name in _KEYWORDS:
            continue
        open_at = match.end() - 1
        close = match_close(skel, open_at)
        if close == -1:
            continue
        prefix = skel[: match.start()].rstrip()
        if prefix.endswith(("function", "modifier", "event", "error", "returns", "new")):
            continue
        args = tuple(split_top(body[open_at + 1 : close]))
        found.append((name, args if args != ("",) else (), match.start()))
        if len(found) >= MAX_CALLS_PER_FUNCTION:
            break
    return found


def line_of_text(function: RFunction, needle: str) -> int:
    """Source line of the first occurrence of ``needle`` in the function body."""
    index = function.body.find(needle)
    if index < 0:
        return function.line
    return function.line + function.body[:index].count("\n")


# ---- shared candidate type ------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticCandidate:
    """A static candidate. It is never verification and never a confirmed vulnerability."""

    detector: str
    family: str
    title: str
    summary: str
    file: str
    line: int
    contract: str
    function: str
    path: tuple[str, ...] = ()
    facts: tuple[tuple[str, str], ...] = ()
    observed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    confidence: str = "medium"
    impact_tags: tuple[str, ...] = ()
    status: str = "candidate"
    verified: bool = False

    @property
    def identity_key(self) -> str:
        return f"{self.contract}.{self.function}" if self.contract and self.function else ""

    def fact(self, name: str, default: str = "") -> str:
        for key, value in self.facts:
            if key == name:
                return value
        return default


MAX_CANDIDATES_PER_FAMILY = 64


def cap(candidates: list[SemanticCandidate]) -> list[SemanticCandidate]:
    """Deterministic order and an explicit cap. Duplicates by detector and location collapse."""
    unique: dict[tuple[str, str, str, int], SemanticCandidate] = {}
    for item in candidates:
        unique.setdefault((item.detector, item.contract, item.function, item.line), item)
    ordered = sorted(
        unique.values(), key=lambda c: (c.file, c.line, c.contract, c.function, c.detector)
    )
    return ordered[:MAX_CANDIDATES_PER_FAMILY]

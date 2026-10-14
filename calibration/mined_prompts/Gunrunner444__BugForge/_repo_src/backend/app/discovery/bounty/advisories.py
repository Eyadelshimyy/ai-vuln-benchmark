"""Solidity compiler advisories matched against actual compiler evidence.

The corpus is a versioned local file copied from the Solidity project's official bug
list, with its provenance. Nothing here downloads anything. An advisory matches only
when the exact compiler version is known and every pipeline condition it names
(IR pipeline, optimizer, EVM version) is established; a condition that is not known
leaves the result unknown instead of quietly meaning "affected" or "safe". A match is
a candidate for a differential or source review and never a vulnerability by itself.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.discovery.bounty.campaign import UNKNOWN, CompilerConfiguration
from app.parsing.solidity_research import (
    ResearchModel,
    build_research_model,
    plain_calls,
    strip_comments,
)

CORPUS_PATH = Path(__file__).parent / "data" / "solidity_advisories.json"
SUPPORTED_SCHEMA = 1
MAX_MATCHES = 16
MAX_SOURCES = 64

_EVM_ORDER = (
    "homestead",
    "tangerineWhistle",
    "spuriousDragon",
    "byzantium",
    "constantinople",
    "petersburg",
    "istanbul",
    "berlin",
    "london",
    "paris",
    "shanghai",
    "cancun",
    "prague",
    "osaka",
)
_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")

Version = tuple[int, int, int]


class AdvisoryCorpusError(ValueError):
    """The local corpus is missing, malformed, or an unsupported schema."""


@dataclass(frozen=True)
class Advisory:
    uid: str
    name: str
    summary: str
    severity: str
    introduced: str
    fixed: str
    link: str
    conditions: tuple[tuple[str, str], ...]
    regex_check: str


@dataclass(frozen=True)
class Corpus:
    schema_version: int
    provenance: tuple[tuple[str, str], ...]
    advisories: tuple[Advisory, ...]

    def provenance_dict(self) -> dict[str, str]:
        return dict(self.provenance)


@dataclass(frozen=True)
class AdvisoryMatch:
    uid: str
    name: str
    severity: str
    status: str  # applicable_candidate | version_match_no_trigger | version_match_unassessed
    # | unknown | no_match
    reasons: tuple[str, ...]
    trigger: str  # present | absent | not_assessed
    trigger_locations: tuple[str, ...]
    link: str
    verified: bool = False


@dataclass(frozen=True)
class AdvisoryReport:
    compiler: CompilerConfiguration
    matches: tuple[AdvisoryMatch, ...]
    considered: int
    corpus_provenance: tuple[tuple[str, str], ...]
    truncated: bool
    notes: tuple[str, ...]
    unknown: int = 0


def parse_version(text: str) -> Version | None:
    match = _VERSION.match(text.strip()) if text else None
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


@lru_cache(maxsize=2)
def load_corpus(path: Path = CORPUS_PATH) -> Corpus:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdvisoryCorpusError(f"the advisory corpus cannot be read: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != SUPPORTED_SCHEMA:
        raise AdvisoryCorpusError("unsupported advisory corpus schema")
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source"):
        raise AdvisoryCorpusError("the advisory corpus carries no provenance")
    entries: list[Advisory] = []
    for item in raw.get("advisories", []):
        conditions = item.get("conditions") or {}
        check = item.get("check") or {}
        entries.append(
            Advisory(
                uid=str(item["uid"]),
                name=str(item["name"]),
                summary=str(item.get("summary", ""))[:400],
                severity=str(item.get("severity", UNKNOWN)),
                introduced=str(item.get("introduced", "")),
                fixed=str(item["fixed"]),
                link=str(item.get("link", "")),
                conditions=tuple(sorted((str(k), json.dumps(v)) for k, v in conditions.items())),
                regex_check=str(check.get("regex-source", "")),
            )
        )
    return Corpus(
        schema_version=SUPPORTED_SCHEMA,
        provenance=tuple(sorted((str(k), str(v)) for k, v in provenance.items())),
        advisories=tuple(entries),
    )


# ---- matching -------------------------------------------------------------------------------


def match_advisories(
    compiler: CompilerConfiguration,
    sources: Mapping[str, str] | None = None,
    *,
    corpus: Corpus | None = None,
) -> AdvisoryReport:
    corpus = corpus or load_corpus()
    version = parse_version(compiler.version) if compiler.version != UNKNOWN else None
    cleaned = {
        path: strip_comments(text) for path, text in sorted((sources or {}).items())[:MAX_SOURCES]
    }
    model = build_research_model(dict(sources or {})) if sources else None
    matches: list[AdvisoryMatch] = []
    notes: list[str] = []
    if version is None:
        notes.append("the exact compiler version is unknown, so no advisory can be ruled in or out")
        return AdvisoryReport(
            compiler=compiler,
            matches=(),
            considered=len(corpus.advisories),
            corpus_provenance=corpus.provenance,
            truncated=False,
            notes=tuple(notes),
            unknown=len(corpus.advisories),
        )
    for advisory in corpus.advisories:
        matches.append(_match_one(advisory, version, compiler, cleaned, model))
    ordered = sorted(matches, key=lambda m: (_RANK[m.status], m.uid))
    relevant = [m for m in ordered if m.status != "no_match"]
    return AdvisoryReport(
        compiler=compiler,
        matches=tuple(relevant[:MAX_MATCHES]),
        considered=len(corpus.advisories),
        corpus_provenance=corpus.provenance,
        truncated=len(relevant) > MAX_MATCHES,
        notes=tuple(notes),
        unknown=sum(1 for m in matches if m.status == "unknown"),
    )


_RANK = {
    "applicable_candidate": 0,
    "version_match_unassessed": 1,
    "unknown": 2,
    "version_match_no_trigger": 3,
    "no_match": 4,
}


def _match_one(
    advisory: Advisory,
    version: Version | None,
    compiler: CompilerConfiguration,
    cleaned: Mapping[str, str],
    model: ResearchModel | None,
) -> AdvisoryMatch:
    def result(
        status: str,
        reasons: list[str],
        trigger: str = "not_assessed",
        locations: tuple[str, ...] = (),
    ) -> AdvisoryMatch:
        return AdvisoryMatch(
            uid=advisory.uid,
            name=advisory.name,
            severity=advisory.severity,
            status=status,
            reasons=tuple(reasons),
            trigger=trigger,
            trigger_locations=locations,
            link=advisory.link,
        )

    if version is None:
        return result("unknown", ["compiler version unknown"])
    introduced = parse_version(advisory.introduced) if advisory.introduced else (0, 0, 0)
    fixed = parse_version(advisory.fixed)
    if introduced is None or fixed is None:
        return result("unknown", ["the advisory's version range could not be read"])
    if not introduced <= version < fixed:
        return result(
            "no_match",
            [f"version {_fmt(version)} is outside {advisory.introduced or '0'}..{advisory.fixed}"],
        )

    reasons = [
        f"version {_fmt(version)} is within [{advisory.introduced or '0'}, {advisory.fixed})"
    ]
    undecided: list[str] = []
    for name, encoded in advisory.conditions:
        wanted = json.loads(encoded)
        state = _condition(name, wanted, compiler, cleaned)
        if state == "no":
            return result("no_match", [*reasons, f"condition {name}={wanted} is not satisfied"])
        if state == "unknown":
            undecided.append(f"{name}={wanted}")
        else:
            reasons.append(f"condition {name}={wanted} holds")
    if undecided:
        return result(
            "unknown", [*reasons, f"pipeline evidence missing for {', '.join(undecided)}"]
        )

    detector = _TRIGGERS.get(advisory.uid)
    if detector is None and advisory.regex_check:
        detector = _regex_trigger(advisory.regex_check)
    if detector is None:
        return result(
            "version_match_unassessed", [*reasons, "no source trigger is modeled for this advisory"]
        )
    if not cleaned or model is None:
        return result(
            "version_match_unassessed", [*reasons, "no source was supplied to assess the trigger"]
        )
    locations = detector(cleaned, model)
    if locations:
        return result(
            "applicable_candidate",
            [*reasons, "a source construct the advisory names is present"],
            "present",
            locations,
        )
    return result(
        "version_match_no_trigger",
        [*reasons, "no construct the advisory names was found"],
        "absent",
    )


def _fmt(version: Version) -> str:
    return ".".join(str(part) for part in version)


def _condition(
    name: str, wanted: object, compiler: CompilerConfiguration, cleaned: Mapping[str, str]
) -> str:
    """yes, no, or unknown. Missing evidence is never read as satisfied or refuted."""
    if name == "viaIR":
        return _tri(compiler.via_ir, wanted)
    if name == "optimizer":
        return _tri(
            {"enabled": "true", "disabled": "false"}.get(compiler.optimizer, UNKNOWN), wanted
        )
    if name == "yulOptimizer":
        if compiler.optimizer == "disabled":
            return "no" if wanted else "yes"
        return "unknown"
    if name == "evmVersion":
        return _evm(compiler.evm_version, str(wanted))
    if name == "ABIEncoderV2":
        return _abi_v2(compiler, cleaned, bool(wanted))
    return "unknown"


def _tri(value: str, wanted: object) -> str:
    if value not in {"true", "false"}:
        return "unknown"
    return "yes" if (value == "true") == bool(wanted) else "no"


def _evm(actual: str, wanted: str) -> str:
    if actual == UNKNOWN or actual not in _EVM_ORDER:
        return "unknown"
    match = re.match(r"^(>=|<=|>|<|=)?\s*(\w+)$", wanted)
    if match is None or match.group(2) not in _EVM_ORDER:
        return "unknown"
    operator = match.group(1) or "="
    left, right = _EVM_ORDER.index(actual), _EVM_ORDER.index(match.group(2))
    ok = {
        ">=": left >= right,
        "<=": left <= right,
        ">": left > right,
        "<": left < right,
        "=": left == right,
    }[operator]
    return "yes" if ok else "no"


def _abi_v2(compiler: CompilerConfiguration, cleaned: Mapping[str, str], wanted: bool) -> str:
    version = parse_version(compiler.version)
    if version is not None and version >= (0, 8, 0):
        return "yes" if wanted else "no"
    if not cleaned:
        return "unknown"
    enabled = any(
        re.search(r"pragma\s+(?:experimental\s+ABIEncoderV2|abicoder\s+v2)", text)
        for text in cleaned.values()
    )
    return "yes" if enabled == wanted else "no"


# ---- source triggers ------------------------------------------------------------------------

Trigger = Callable[[Mapping[str, str], ResearchModel], tuple[str, ...]]


def _regex_trigger(pattern: str) -> Trigger:
    def run(cleaned: Mapping[str, str], _model: ResearchModel) -> tuple[str, ...]:
        try:
            compiled = re.compile(pattern)
        except re.error:
            return ()
        return tuple(path for path, text in cleaned.items() if compiled.search(text))[:4]

    return run


def _transient_clear(cleaned: Mapping[str, str], model: ResearchModel) -> tuple[str, ...]:
    found: list[str] = []
    for contract in model.contracts.values():
        transient = set(re.findall(r"\btransient\b(?:\s+\w+)*?\s+(\w+)\s*(?:=|;)", contract.body))
        if not transient:
            continue
        cleared = set(re.findall(r"\bdelete\s+(\w+)", contract.body))
        storage = {name for name, _type in contract.state_vars} - transient
        if cleared & transient and cleared & storage:
            found.append(f"{contract.file}:{contract.line} {contract.name}")
    return tuple(found[:4])


def _mutual_recursion(cleaned: Mapping[str, str], model: ResearchModel) -> tuple[str, ...]:
    found: list[str] = []
    for contract in model.contracts.values():
        graph: dict[str, set[str]] = {}
        names = {f.name for f in contract.functions if f.kind == "function"}
        for function in contract.functions:
            if function.kind != "function":
                continue
            calls = {name for name, _args, _off in plain_calls(function.body) if name in names}
            graph.setdefault(function.name, set()).update(calls)
        for start in sorted(graph):
            if _reaches_back(graph, start):
                found.append(f"{contract.file}:{contract.line} {contract.name}.{start}")
                break
    return tuple(found[:4])


def _reaches_back(graph: Mapping[str, set[str]], start: str) -> bool:
    for neighbor in sorted(graph.get(start, ())):
        if neighbor == start:
            continue
        seen = {neighbor}
        stack = [neighbor]
        while stack:
            current = stack.pop()
            for nxt in graph.get(current, ()):
                if nxt == start:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return False


def _layout_with_bases(cleaned: Mapping[str, str], model: ResearchModel) -> tuple[str, ...]:
    return tuple(
        f"{c.file}:{c.line} {c.name}"
        for c in model.contracts.values()
        if re.search(r"\blayout\s+at\b", c.head) and len(c.bases) >= 2
    )[:4]


def _layout_with_arrays(cleaned: Mapping[str, str], model: ResearchModel) -> tuple[str, ...]:
    return tuple(
        f"{c.file}:{c.line} {c.name}"
        for c in model.contracts.values()
        if re.search(r"\blayout\s+at\b", c.head)
        and any("[" in kind for _name, kind in c.state_vars)
    )[:4]


def _memory_bytes_delete(cleaned: Mapping[str, str], model: ResearchModel) -> tuple[str, ...]:
    found: list[str] = []
    for function in model.all_functions():
        text = cleaned.get(function.file, "")
        declared = set(re.findall(r"\bbytes\s+memory\s+(\w+)", text))
        local = {item.name for item in function.params if item.type_name == "bytes"} | set(
            re.findall(r"\bbytes\s+memory\s+(\w+)", function.body)
        )
        for name in re.findall(r"\bdelete\s+(\w+)\s*\[", function.body):
            if name in local and name in declared:
                found.append(f"{function.file}:{function.line} {function.identity}")
                break
    return tuple(found[:4])


def _named_require_error(cleaned: Mapping[str, str], model: ResearchModel) -> tuple[str, ...]:
    pattern = re.compile(r"\brequire\s*\([^;]*?\b[A-Za-z_]\w*\s*\(\s*\{")
    return tuple(path for path, text in cleaned.items() if pattern.search(text))[:4]


def _delegatecall_use(cleaned: Mapping[str, str], _model: ResearchModel) -> tuple[str, ...]:
    pattern = re.compile(r"\.delegatecall\s*\(")
    return tuple(path for path, text in cleaned.items() if pattern.search(text))[:4]


def _raw_send_use(cleaned: Mapping[str, str], _model: ResearchModel) -> tuple[str, ...]:
    pattern = re.compile(r"\.send\s*\(")
    return tuple(path for path, text in cleaned.items() if pattern.search(text))[:4]


def _ecrecover_use(cleaned: Mapping[str, str], _model: ResearchModel) -> tuple[str, ...]:
    pattern = re.compile(r"\becrecover\s*\(")
    return tuple(path for path, text in cleaned.items() if pattern.search(text))[:4]


def _inline_assembly(cleaned: Mapping[str, str], _model: ResearchModel) -> tuple[str, ...]:
    pattern = re.compile(r"\bassembly\b[^{]*\{")
    return tuple(path for path, text in cleaned.items() if pattern.search(text))[:4]


_TRIGGERS: dict[str, Trigger] = {
    "SOL-2026-1": _transient_clear,
    "SOL-2026-2": _mutual_recursion,
    "SOL-2026-4": _mutual_recursion,
    "SOL-2026-3": _layout_with_bases,
    "SOL-2025-1": _layout_with_arrays,
    "SOL-2026-5": _memory_bytes_delete,
    "SOL-2026-6": _named_require_error,
    "SOL-2017-4": _delegatecall_use,
    "SOL-2016-6": _raw_send_use,
    "SOL-2017-3": _ecrecover_use,
    "SOL-2022-4": _inline_assembly,
}


def advisory_dict(report: AdvisoryReport) -> dict[str, Any]:
    return {
        "compiler": report.compiler.fingerprint() or UNKNOWN,
        "considered": report.considered,
        "provenance": dict(report.corpus_provenance),
        "truncated": report.truncated,
        "unknown": report.unknown,
        "notes": list(report.notes),
        "matches": [
            {
                "uid": m.uid,
                "name": m.name,
                "severity": m.severity,
                "status": m.status,
                "trigger": m.trigger,
                "locations": list(m.trigger_locations),
                "reasons": list(m.reasons),
            }
            for m in report.matches
        ],
    }


# ---- precondition graphs (Phase 51) ---------------------------------------------------------
#
# A precondition graph makes an advisory's reasoning explicit: it is an ordered chain of
# preconditions (compiler version in range -> each pipeline condition -> a source trigger)
# where every node carries a status that is one of "yes", "no", "unknown", or "not_assessed".
# The graph never upgrades an advisory to a vulnerability; it only shows which preconditions
# are established, which are refuted, and which cannot be decided from the evidence present.


@dataclass(frozen=True)
class PreconditionNode:
    key: str  # version_range | condition:<name> | source_trigger
    label: str
    status: str  # yes | no | unknown | not_assessed
    detail: str = ""
    locations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreconditionGraph:
    uid: str
    name: str
    severity: str
    link: str
    nodes: tuple[PreconditionNode, ...]
    outcome: str  # applicable_candidate | refuted | undecided | not_in_range

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "severity": self.severity,
            "link": self.link,
            "outcome": self.outcome,
            "nodes": [
                {
                    "key": node.key,
                    "label": node.label,
                    "status": node.status,
                    "detail": node.detail,
                    "locations": list(node.locations),
                }
                for node in self.nodes
            ],
        }


def precondition_graph(
    advisory: Advisory,
    compiler: CompilerConfiguration,
    cleaned: Mapping[str, str],
    model: ResearchModel | None,
) -> PreconditionGraph:
    """Build the ordered precondition graph for one advisory.

    The traversal stops marking later nodes decidable only when an earlier node is
    refuted; a refuted node short-circuits to ``refuted``. A node whose evidence is
    missing is ``unknown`` and never silently read as satisfied.
    """

    nodes: list[PreconditionNode] = []
    version = parse_version(compiler.version) if compiler.version != UNKNOWN else None
    introduced = parse_version(advisory.introduced) if advisory.introduced else (0, 0, 0)
    fixed = parse_version(advisory.fixed)

    # 1. version-range node
    if version is None:
        nodes.append(
            PreconditionNode(
                "version_range",
                f"compiler version in [{advisory.introduced or '0'}, {advisory.fixed})",
                "unknown",
                "the exact compiler version is not known",
            )
        )
        return PreconditionGraph(
            advisory.uid, advisory.name, advisory.severity, advisory.link, tuple(nodes), "undecided"
        )
    if introduced is None or fixed is None:
        nodes.append(
            PreconditionNode(
                "version_range", "compiler version in range", "unknown", "range unreadable"
            )
        )
        return PreconditionGraph(
            advisory.uid, advisory.name, advisory.severity, advisory.link, tuple(nodes), "undecided"
        )
    in_range = introduced <= version < fixed
    nodes.append(
        PreconditionNode(
            "version_range",
            f"compiler version in [{advisory.introduced or '0'}, {advisory.fixed})",
            "yes" if in_range else "no",
            f"version {_fmt(version)}",
        )
    )
    if not in_range:
        return PreconditionGraph(
            advisory.uid,
            advisory.name,
            advisory.severity,
            advisory.link,
            tuple(nodes),
            "not_in_range",
        )

    # 2. one node per pipeline condition
    undecided = False
    refuted = False
    for name, encoded in advisory.conditions:
        wanted = json.loads(encoded)
        state = _condition(name, wanted, compiler, cleaned)
        status = {"yes": "yes", "no": "no", "unknown": "unknown"}[state]
        nodes.append(
            PreconditionNode(
                f"condition:{name}",
                f"{name}={wanted}",
                status,
                "" if state != "unknown" else "pipeline evidence missing",
            )
        )
        if state == "no":
            refuted = True
        elif state == "unknown":
            undecided = True

    # 3. source-trigger node
    detector = _TRIGGERS.get(advisory.uid)
    if detector is None and advisory.regex_check:
        detector = _regex_trigger(advisory.regex_check)
    if detector is None:
        nodes.append(
            PreconditionNode(
                "source_trigger",
                "a source construct the advisory names",
                "not_assessed",
                "no source trigger is modeled for this advisory",
            )
        )
        trigger_status = "not_assessed"
        locations: tuple[str, ...] = ()
    elif not cleaned or model is None:
        nodes.append(
            PreconditionNode(
                "source_trigger",
                "a source construct the advisory names",
                "not_assessed",
                "no source was supplied to assess the trigger",
            )
        )
        trigger_status = "not_assessed"
        locations = ()
    else:
        locations = detector(cleaned, model)
        trigger_status = "yes" if locations else "no"
        nodes.append(
            PreconditionNode(
                "source_trigger",
                "a source construct the advisory names",
                trigger_status,
                "present" if locations else "no matching construct found",
                locations,
            )
        )

    if refuted:
        outcome = "refuted"
    elif undecided or trigger_status == "not_assessed":
        outcome = "undecided"
    elif trigger_status == "yes":
        outcome = "applicable_candidate"
    else:
        outcome = "refuted"
    return PreconditionGraph(
        advisory.uid, advisory.name, advisory.severity, advisory.link, tuple(nodes), outcome
    )


def precondition_graphs(
    compiler: CompilerConfiguration,
    sources: Mapping[str, str] | None = None,
    *,
    corpus: Corpus | None = None,
    only: tuple[str, ...] = (),
) -> tuple[PreconditionGraph, ...]:
    """Precondition graphs for the corpus (or a chosen subset of advisory uids).

    Graphs whose version range does not contain the compiler version are omitted so
    the result stays focused on advisories that could still apply.
    """

    corpus = corpus or load_corpus()
    cleaned = {
        path: strip_comments(text) for path, text in sorted((sources or {}).items())[:MAX_SOURCES]
    }
    model = build_research_model(dict(sources or {})) if sources else None
    wanted = set(only)
    graphs = [
        precondition_graph(advisory, compiler, cleaned, model)
        for advisory in corpus.advisories
        if not wanted or advisory.uid in wanted
    ]
    kept = [g for g in graphs if g.outcome != "not_in_range"]
    kept.sort(key=lambda g: (_GRAPH_RANK.get(g.outcome, 9), g.uid))
    return tuple(kept)


_GRAPH_RANK = {"applicable_candidate": 0, "undecided": 1, "refuted": 2}

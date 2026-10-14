"""Conservative intra-procedural def-use for one Solidity function.

Definitions come from the CFG statement nodes. A value reaches a sink only when
every reaching definition of the used expression includes that value. Unknown
control flow or an unparsed definition yields None, which is not evidence that
the value is absent or present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsing.solidity_cfg import (
    FunctionCfg,
    _consume_parens,
    _skip_string_or_comment,
    build_function_cfg,
)

_NOISE = frozenset(
    {
        "abi",
        "assert",
        "block",
        "bool",
        "bytes",
        "calldata",
        "ecrecover",
        "encode",
        "encodePacked",
        "false",
        "keccak256",
        "memory",
        "msg",
        "require",
        "sender",
        "sha256",
        "storage",
        "string",
        "this",
        "true",
        "uint",
        "address",
    }
)
_DOMAIN_MARKERS = (
    "DOMAIN_SEPARATOR",
    "domainSeparator",
    "chainid",
    "chainId",
    "address(this)",
    "typeHash",
    "TYPEHASH",
    "typehash",
)
_ORACLE_LABELS = ("roundId", "answer", "startedAt", "updatedAt", "answeredInRound")


@dataclass(frozen=True)
class FlowEdge:
    source: str
    target: str
    kind: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Def:
    expr: str
    node: int


@dataclass
class FunctionFlow:
    known: bool
    edges: list[FlowEdge] = field(default_factory=list)
    incoming: dict[int, dict[str, frozenset[_Def]]] = field(default_factory=dict)
    cfg: FunctionCfg | None = None


def analyze_flow(function_text: str) -> FunctionFlow:
    cfg = build_function_cfg(function_text)
    flow = FunctionFlow(known=cfg.known, cfg=cfg)
    if not cfg.known:
        return flow
    gens: dict[int, dict[str, _Def]] = {node.node_id: {} for node in cfg.nodes}
    for name in _parameters(function_text):
        gens[cfg.entry][name] = _Def("<param>", cfg.entry)
        _remember(flow, function_text, name, "<param>", "param")
    for node in cfg.nodes:
        if node.kind in {
            "entry",
            "exit",
            "join",
            "if",
            "loop",
            "unchecked",
            "try",
            "break",
            "continue",
        }:
            continue
        for name, expr in _assignments(node.text):
            gens[node.node_id][name] = _Def(expr, node.node_id)
            kind = "hash" if "keccak256" in expr or "sha256" in expr else "assign"
            _remember(flow, function_text, expr, name, kind)
            _remember_indexes(flow, function_text, node.text)
    if not _solve(cfg, gens, flow):
        flow.known = False
    return flow


def digest_contains(function_text: str, marker: str) -> bool | None:
    """Whether every ecrecover digest in the function includes ``marker``."""
    flow = analyze_flow(function_text)
    if not flow.known or flow.cfg is None:
        return None
    uses = [
        node
        for node in flow.cfg.nodes
        if "ecrecover" in node.text and node.kind in {"stmt", "return", "require"}
    ]
    if not uses:
        return None
    results = [_arg_contains(flow, node, marker) for node in uses]
    if any(item is None for item in results):
        return None
    return all(item is True for item in results)


def digest_contains_any(
    function_text: str, markers: tuple[str, ...] = _DOMAIN_MARKERS
) -> bool | None:
    results = [digest_contains(function_text, marker) for marker in markers]
    if any(item is True for item in results):
        return True
    if any(item is None for item in results):
        return None
    return False


def signature_replay_gap(function_text: str) -> str | None:
    """A reason ecrecover may be replayable, or None when a nonce is signed and consumed.

    Consumption means the same nonce that reaches the digest is incremented or
    decremented on a path shared with ``ecrecover``. Assigning a constant, or
    writing a different nonce index, does not consume the signed value. The
    result is potential evidence, not a proof of signature security.
    """
    if "ecrecover" not in function_text:
        return None
    unknown = False
    for name in ("nonce", "nonces"):
        included = digest_contains(function_text, name)
        if included is None:
            unknown = True
            continue
        if included is True:
            if _nonce_consumed_for_digest(function_text, name):
                return None
            return f"{name} reaches the ecrecover digest but is not consumed"
    if unknown:
        return "nonce flow into ecrecover could not be established"
    return "ecrecover digest does not include a nonce from this function"


def signature_nonce_order(function_text: str) -> str:
    """Order of nonce consumption, digest construction, and ecrecover.

    ``digest-verify-consume`` increments after verification.
    ``consume-digest-verify`` increments first and signs the updated nonce.
    ``consume-unrelated-verify`` increments a nonce that never reaches ecrecover.
    ``digest-not-consumed`` signs a nonce that is not incremented.
    """
    if "ecrecover" not in function_text:
        return "absent"
    flow = analyze_flow(function_text)
    if not flow.known or flow.cfg is None:
        return "unknown"
    live = flow.cfg.reachable_from(flow.cfg.entry)
    consume_at: int | None = None
    digest_at: int | None = None
    verify_at: int | None = None
    for node in flow.cfg.nodes:
        if node.node_id not in live:
            continue
        if consume_at is None and _statement_consumes_nonce(node.text):
            consume_at = node.node_id
        if digest_at is None and _statement_builds_nonce_digest(node.text):
            digest_at = node.node_id
        if (
            verify_at is None
            and "ecrecover" in node.text
            and node.kind in {"stmt", "return", "require"}
        ):
            verify_at = node.node_id
    included_nonce = digest_contains(function_text, "nonce")
    included_nonces = digest_contains(function_text, "nonces")
    if included_nonce is True or included_nonces is True:
        included: bool | None = True
    elif included_nonce is None or included_nonces is None:
        included = None
    else:
        included = False
    if (
        included is True
        and consume_at is not None
        and digest_at is not None
        and verify_at is not None
    ):
        if consume_at < digest_at <= verify_at:
            return "consume-digest-verify"
        if digest_at <= verify_at < consume_at:
            return "digest-verify-consume"
    if included is True:
        return "digest-not-consumed"
    if included is False and consume_at is not None and verify_at is not None:
        return "consume-unrelated-verify"
    if included is False:
        return "not-in-digest"
    return "unknown"


def signature_domain_gap(function_text: str) -> str | None:
    if "ecrecover" not in function_text:
        return None
    included = digest_contains_any(function_text)
    if included is True:
        return None
    if included is None:
        return "domain binding of the ecrecover digest could not be established"
    return "ecrecover digest is not bound to a domain, chain, or this contract"


def oracle_freshness_protects(function_text: str) -> bool | None:
    """True when freshness from the same oracle observation dominates its answer.

    Each ``latestRoundData`` or ``latestAnswer`` call is one observation.
    ``updatedAt`` from call 1 does not protect ``answer`` from call 2.
    """
    flow = analyze_flow(function_text)
    if not flow.known or flow.cfg is None:
        return None
    if "latestRoundData" not in function_text and "latestAnswer" not in function_text:
        return None
    tagged = _tagged_oracle_assignments(flow.cfg)
    observations: dict[str, dict[str, set[str]]] = {}
    defining: dict[str, set[int]] = {}
    for node_id, name, expr in tagged:
        parsed = _oracle_expr(expr)
        if parsed is None:
            continue
        observation, component = parsed
        slot = observations.setdefault(observation, {"answers": set(), "fresh": set()})
        if component == "answer":
            slot["answers"].add(name)
            defining.setdefault(name, set()).add(node_id)
        if component in {"updatedAt", "answeredInRound"}:
            slot["fresh"].add(name)
    if not observations:
        return False
    saw_use = False
    for slot in observations.values():
        uses = _oracle_uses(flow, slot["answers"], defining)
        if not uses:
            continue
        saw_use = True
        if not slot["fresh"]:
            return False
        checks = [
            node.node_id
            for node in flow.cfg.nodes
            if node.kind in {"require", "if"} and _is_real_freshness(node.text, slot["fresh"])
        ]
        if not checks:
            return False
        if not all(any(flow.cfg.dominates(check, use) for check in checks) for use in uses):
            return False
    if not saw_use:
        return any(bool(slot["answers"]) for slot in observations.values())
    return True


def _arg_contains(flow: FunctionFlow, node: object, marker: str) -> bool | None:
    arg = _call_arg(getattr(node, "text", ""), "ecrecover", 0)
    if arg is None or flow.cfg is None:
        return None
    return _expr_contains(arg, marker, flow, int(getattr(node, "node_id")), set())


def _expr_contains(
    expr: str, marker: str, flow: FunctionFlow, node_id: int, seen: set[str]
) -> bool | None:
    if _direct(expr, marker):
        return True
    env = flow.incoming.get(node_id, {})
    idents = [ident for ident in _idents(expr) if ident not in _NOISE and not _is_type_name(ident)]
    any_true = False
    unknown = False
    for ident in idents:
        if ident in seen:
            continue
        defs = env.get(ident)
        if not defs:
            continue
        results: list[bool | None] = []
        for definition in defs:
            if definition.expr == "<unknown>":
                results.append(None)
            else:
                results.append(
                    _expr_contains(definition.expr, marker, flow, definition.node, seen | {ident})
                )
        if any(item is True for item in results) and any(item is False for item in results):
            return False
        if any(item is None for item in results):
            unknown = True
            continue
        if results and all(item is True for item in results):
            any_true = True
    if any_true and unknown:
        return None
    if any_true:
        return True
    if unknown:
        return None
    return False


def _solve(cfg: FunctionCfg, gens: dict[int, dict[str, _Def]], flow: FunctionFlow) -> bool:
    preds: dict[int, list[int]] = {node.node_id: [] for node in cfg.nodes}
    for src, dst, _label in cfg.edges:
        if dst in preds:
            preds[dst].append(src)
    incoming: dict[int, dict[str, frozenset[_Def]]] = {node.node_id: {} for node in cfg.nodes}
    outgoing: dict[int, dict[str, frozenset[_Def]]] = {node.node_id: {} for node in cfg.nodes}
    limit = max(4, len(cfg.nodes) * len(cfg.nodes) + 2)
    for _ in range(limit):
        changed = False
        for node in cfg.nodes:
            merged: dict[str, set[_Def]] = {}
            for pred in preds[node.node_id]:
                for name, defs in outgoing[pred].items():
                    merged.setdefault(name, set()).update(defs)
            frozen = {name: frozenset(defs) for name, defs in merged.items()}
            if frozen != incoming[node.node_id]:
                incoming[node.node_id] = frozen
                changed = True
            out = dict(frozen)
            for name, definition in gens[node.node_id].items():
                out[name] = frozenset((definition,))
            if out != outgoing[node.node_id]:
                outgoing[node.node_id] = out
                changed = True
        if not changed:
            flow.incoming = incoming
            return True
    flow.incoming = incoming
    return False


def _assignments(text: str) -> list[tuple[str, str]]:
    raw = text.strip().rstrip(";").strip()
    if not raw or re.match(
        r"(if|for|while|do|return|require|assert|revert|break|continue|unchecked|try)\b", raw
    ):
        return []
    tuple_match = re.match(r"\((.*)\)\s*=\s*(.+)\Z", raw, re.S)
    if tuple_match:
        rhs = tuple_match.group(2).strip()
        parts = _split_top(tuple_match.group(1))
        oracle = "latestRoundData" in rhs
        found: list[tuple[str, str]] = []
        for index, part in enumerate(parts):
            name = _binding_name(part)
            if not name:
                continue
            if oracle and index < len(_ORACLE_LABELS):
                found.append((name, f"latestRoundData.{_ORACLE_LABELS[index]}"))
            else:
                found.append((name, rhs))
        return found
    split = _split_assign(raw)
    if split is None:
        return []
    lhs, rhs = split
    name = _binding_name(lhs)
    if not name:
        return []
    return [(name, rhs.strip())]


def _split_assign(text: str) -> tuple[str, str] | None:
    depth = 0
    index = 0
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif (
            depth == 0
            and char == "="
            and not text.startswith("==", index)
            and not text.startswith("=>", index)
        ):
            if index > 0 and text[index - 1] in "=!<>":
                index += 1
                continue
            op = index - 1 if index > 0 and text[index - 1] in "+-*/%" else index
            lhs = text[:op].strip()
            rhs = text[index + 1 :].strip()
            return lhs, rhs
        index += 1
    return None


def _parameters(function_text: str) -> list[str]:
    match = re.search(r"\b(function|constructor|fallback|receive)\b", function_text)
    start = match.end() if match else 0
    open_at = function_text.find("(", start)
    if open_at < 0:
        return []
    try:
        end = _consume_parens(function_text, open_at)
    except ValueError:
        return []
    names: list[str] = []
    for part in _split_top(function_text[open_at + 1 : end - 1]):
        name = _binding_name(part)
        if name:
            names.append(name)
    return names


def _binding_name(part: str) -> str:
    idents = re.findall(r"[A-Za-z_]\w*", part)
    for ident in reversed(idents):
        if ident in _NOISE or _is_type_name(ident) or ident in {"indexed", "payable"}:
            continue
        return str(ident)
    return ""


def _is_type_name(ident: str) -> bool:
    return bool(re.fullmatch(r"u?int\d*|bytes\d*|bytes|string|address|bool", ident))


def _idents(expr: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*", expr)


def _direct(expr: str, marker: str) -> bool:
    if marker == "address(this)":
        return "address(this)" in re.sub(r"\s+", "", expr)
    if marker.lower() == "chainid":
        compact = re.sub(r"\s+", "", expr).lower()
        return "chainid" in compact or "block.chainid" in compact
    return bool(re.search(rf"\b{re.escape(marker)}\b", expr))


def _call_arg(text: str, name: str, index: int) -> str | None:
    match = re.search(rf"\b{name}\s*\(", text)
    if not match:
        return None
    open_at = text.find("(", match.start())
    try:
        end = _consume_parens(text, open_at)
    except ValueError:
        return None
    args = _split_top(text[open_at + 1 : end - 1])
    if len(args) <= index:
        return None
    return args[index].strip()


def _split_top(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return parts


def _nonce_consumed_for_digest(function_text: str, name: str) -> bool:
    """True when every live ecrecover is covered by the nonce it actually signs.

    Coverage means the same nonce (and the same mapping index, when there is
    one) dominates that verification, or is incremented on every path from
    that verification to the exit. An increment on another branch, another
    index, or a path that never reaches ecrecover does not count.
    """
    flow = analyze_flow(function_text)
    if not flow.known or flow.cfg is None:
        return False
    live = flow.cfg.reachable_from(flow.cfg.entry)
    verifies = [
        node
        for node in flow.cfg.nodes
        if node.node_id in live
        and "ecrecover" in node.text
        and node.kind in {"stmt", "return", "require"}
    ]
    if not verifies:
        return False
    for verify in verifies:
        indexes = _digest_nonce_indexes(flow, verify.node_id, verify.text, name)
        if indexes is None:
            return False
        writes = [
            node
            for node in flow.cfg.nodes
            if node.node_id in live
            and node.kind == "stmt"
            and _is_nonce_consume(node.text, name)
            and _nonce_indexes_match(node.text, name, indexes)
        ]
        if not any(_nonce_covers(flow.cfg, write.node_id, verify.node_id) for write in writes):
            return False
    return True


def _nonce_covers(cfg: FunctionCfg, write: int, verify: int) -> bool:
    if write == verify or cfg.dominates(write, verify):
        return True
    if write not in cfg.reachable_from(verify):
        return False
    seen: set[int] = set()
    stack = [verify]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current == write and current != verify:
            continue
        node = cfg.nodes[current]
        if node.kind == "exit" and current != verify:
            return False
        for nxt in cfg.successors(current):
            if nxt != write:
                stack.append(nxt)
    return True


def _digest_nonce_indexes(
    flow: FunctionFlow, node_id: int, text: str, name: str
) -> set[str] | None:
    arg = _call_arg(text, "ecrecover", 0)
    if arg is None or flow.cfg is None:
        return None
    found: set[str] = set()
    if _collect_nonce_indexes(arg, flow, node_id, name, found, set()) is None:
        return None
    return found


def _collect_nonce_indexes(
    expr: str,
    flow: FunctionFlow,
    node_id: int,
    name: str,
    found: set[str],
    seen: set[str],
) -> bool | None:
    found.update(_nonce_indexes(expr, name))
    env = flow.incoming.get(node_id, {})
    unknown = False
    for ident in _idents(expr):
        if ident in _NOISE or _is_type_name(ident) or ident in seen:
            continue
        defs = env.get(ident)
        if not defs:
            continue
        for definition in defs:
            if definition.expr == "<unknown>":
                unknown = True
                continue
            nested = _collect_nonce_indexes(
                definition.expr, flow, definition.node, name, found, seen | {ident}
            )
            if nested is None:
                unknown = True
    if unknown and not found and not _direct(expr, name):
        return None
    return True


def _nonce_indexes_match(write_text: str, name: str, digest_indexes: set[str]) -> bool:
    write_indexes = _nonce_indexes(write_text, name)
    if not write_indexes and not digest_indexes:
        return True
    if not write_indexes or not digest_indexes:
        return False
    combined = write_indexes | digest_indexes
    if not all(re.fullmatch(r"[A-Za-z_]\w*", item) for item in combined):
        return False
    return not write_indexes.isdisjoint(digest_indexes)


def _statement_consumes_nonce(text: str) -> bool:
    return _is_nonce_consume(text, "nonce") or _is_nonce_consume(text, "nonces")


def _statement_builds_nonce_digest(text: str) -> bool:
    if "keccak256" not in text and "sha256" not in text:
        return False
    return bool(re.search(r"\bnonce\b|\bnonces\b", text))


def _is_nonce_consume(text: str, name: str) -> bool:
    indexed = rf"\b{name}\b\s*(?:\[[^\]]+\])?"
    if re.search(rf"{indexed}\s*(\+\+|--)", text):
        return True
    if re.search(rf"(\+\+|--)\s*{name}\b", text):
        return True
    if re.search(rf"{indexed}\s*(\+=|-=)", text):
        return True
    return bool(re.search(rf"{indexed}\s*=\s*{name}\b[^;]*[+\-]", text))


def _nonce_indexes(text: str, name: str) -> set[str]:
    return {item.strip() for item in re.findall(rf"\b{name}\s*\[([^\]]+)\]", text)}


def _is_real_freshness(text: str, names: set[str]) -> bool:
    """A timestamp exists only when the check bounds its age or its round.

    ``updatedAt != 0`` shows the feed returned a timestamp. It does not limit
    how old that timestamp is.
    """
    compact = re.sub(r"\s+", "", text)
    for name in names:
        escaped = re.escape(name)
        if re.search(rf"block\.timestamp-{escaped}", compact) and re.search(r"[<>]", compact):
            return True
        if re.search(rf"{escaped}(?:>|>=)block\.timestamp", compact):
            return True
        if re.search(r"answeredInRound", name) and re.search(
            rf"\b{escaped}\b\s*(?:>=|==)\s*[A-Za-z_]\w*", text
        ):
            return True
        if re.search(rf"\b{escaped}\b\s*(?:>=|==)\s*\w*roundId\b", text):
            return True
    return False


_ORACLE_COMPONENT = re.compile(
    r"(oracle\d+)\.(roundId|answer|startedAt|updatedAt|answeredInRound)\Z"
)


def _oracle_expr(expr: str) -> tuple[str, str] | None:
    match = _ORACLE_COMPONENT.fullmatch(expr.strip())
    if match is None:
        return None
    return match.group(1), match.group(2)


def _tagged_oracle_assignments(cfg: FunctionCfg) -> list[tuple[int, str, str]]:
    counter = 0
    found: list[tuple[int, str, str]] = []
    for node in cfg.nodes:
        raw = _assignments(node.text)
        if not any(
            expr.startswith("latestRoundData.") or "latestAnswer" in expr for _, expr in raw
        ):
            continue
        counter += 1
        observation = f"oracle{counter}"
        for name, expr in raw:
            if expr.startswith("latestRoundData."):
                tagged = observation + expr[len("latestRoundData") :]
            elif "latestAnswer" in expr:
                tagged = f"{observation}.answer"
            else:
                continue
            found.append((node.node_id, name, tagged))
    return found


def _oracle_uses(flow: FunctionFlow, names: set[str], defining: dict[str, set[int]]) -> list[int]:
    if flow.cfg is None or not names:
        return []
    defined_at = {node_id for name in names for node_id in defining.get(name, set())}
    uses: list[int] = []
    for node in flow.cfg.nodes:
        if node.kind not in {"stmt", "return", "require", "if"}:
            continue
        if node.node_id in defined_at:
            continue
        if any(re.search(rf"\b{re.escape(name)}\b", node.text) for name in names):
            uses.append(node.node_id)
    return uses


def _remember(flow: FunctionFlow, function_text: str, source: str, target: str, kind: str) -> None:
    start = function_text.find(source) if source != "<param>" else 0
    if start < 0:
        start = 0
    end = start + (0 if source == "<param>" else len(source))
    flow.edges.append(FlowEdge(source[:80], target[:80], kind, start, end, source[:120]))
    for ident in _idents(source):
        if ident in _NOISE or ident == target or _is_type_name(ident):
            continue
        origin = function_text.find(ident)
        flow.edges.append(
            FlowEdge(
                ident, target[:80], kind, max(origin, 0), max(origin, 0) + len(ident), source[:120]
            )
        )


def _remember_indexes(flow: FunctionFlow, function_text: str, text: str) -> None:
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]", text):
        index = match.group(2).strip()
        start = function_text.find(match.group(0))
        if start < 0:
            start = 0
        flow.edges.append(
            FlowEdge(
                index[:80],
                match.group(1),
                "index",
                start,
                start + len(match.group(0)),
                match.group(0),
            )
        )

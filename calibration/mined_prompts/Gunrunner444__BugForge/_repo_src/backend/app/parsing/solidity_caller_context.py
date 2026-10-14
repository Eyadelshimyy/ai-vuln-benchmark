"""Caller-context and authorization-flow analysis.

The model tracks which identity ``msg.sender`` carries across each call boundary
and which identity an authorization check evaluates. Every conclusion is built
from the source of the target: function names such as ``multicall`` or ``router``
carry no meaning here. A candidate needs an authorization-sensitive operation, a
call path that can reach it, an identity change across the boundary, and a check
that evaluates the wrong identity. Anything else stays unreported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.solidity_call_outcome import swallowed_call_failures
from app.parsing.solidity_research import (
    MemberCall,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    cap,
    member_calls,
    mentions,
)

MAX_CONTEXT_EDGES = 256
MAX_PATH_DEPTH = 4
FAMILY = "caller_context"

_SENDER = r"(?:msg\.sender|_msgSender\(\))"
_ROLE_WORDS = ("hasRole", "_checkRole", "onlyRole")
_SELF_EXPRESSIONS = frozenset({"address(this)", "this", "address(this).balance"})
_HELPER_DEPTH = 2
_SKIP_CALLS = frozenset(
    {
        "if",
        "for",
        "while",
        "do",
        "return",
        "require",
        "assert",
        "revert",
        "emit",
        "new",
        "delete",
        "unchecked",
        "try",
        "catch",
        "else",
        "abi",
        "keccak256",
        "sha256",
        "ecrecover",
        "addmod",
        "mulmod",
        "super",
        "_",
    }
)
_AUTH_HINT = re.compile(r"check|only|auth|owner|role|guard", re.I)
_INTERNAL_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")


@dataclass(frozen=True)
class AuthCheck:
    """One place a function decides who may proceed, and whom it evaluates."""

    kind: str  # self | state_account | claimed_param | forwarded_sender | role | mapping | unknown
    expected: str
    text: str
    origin: str
    negated: bool = False


@dataclass(frozen=True)
class ContextEdge:
    """How ``msg.sender`` changes when ``caller`` reaches ``callee``."""

    caller: str
    callee: str
    mechanism: str  # internal | self_call | self_delegatecall | external | low_level_call | ...
    sender_after: str  # preserved | this_contract | unknown
    line: int


def sender_checks(model: ResearchModel, function: RFunction) -> tuple[list[AuthCheck], bool]:
    """Authorization checks on a function and whether any modifier could not be resolved.

    An inherited modifier is visible through ``model.modifier``. A check that lives
    in an internal helper called from that modifier or from the function body is
    inlined for a bounded number of hops. An auth-shaped helper that cannot be
    resolved stays unresolved, so a protected function is not reported as unprotected.
    """
    checks: list[AuthCheck] = []
    unresolved = False
    state = model.state_vars(function.contract)
    params = set(function.param_names())
    seen: set[tuple[str, str]] = set()
    regions: list[tuple[str, str]] = []
    for raw in function.modifiers:
        name = raw.split("(", 1)[0]
        modifier = model.modifier(function.contract, name)
        if modifier is None:
            unresolved = True
            continue
        regions.append((modifier.body, f"modifier:{name}"))
    regions.append((function.body, "body"))
    for text, origin in regions:
        checks.extend(_checks_in(text, state, params, origin))
        extra, missing = _helper_checks(
            model, function.contract, text, state, params, seen, _HELPER_DEPTH
        )
        checks.extend(extra)
        unresolved = unresolved or missing
    return checks, unresolved


def _helper_checks(
    model: ResearchModel,
    contract: str,
    text: str,
    state: dict[str, str],
    params: set[str],
    seen: set[tuple[str, str]],
    depth: int,
) -> tuple[list[AuthCheck], bool]:
    """Checks inside internal helpers, and whether an auth-shaped helper was missing."""
    if depth <= 0:
        return [], False
    found: list[AuthCheck] = []
    unresolved = False
    for match in _INTERNAL_CALL.finditer(text):
        name = match.group(1)
        if name in _SKIP_CALLS or name.startswith("abi"):
            continue
        key = (contract, name)
        if key in seen:
            continue
        seen.add(key)
        helper = model.function(contract, name)
        if helper is None or not helper.has_body or helper.kind != "function":
            if name.startswith("_") and _AUTH_HINT.search(name):
                unresolved = True
            continue
        if helper.visibility in {"public", "external"} and not name.startswith("_"):
            continue
        found.extend(_checks_in(helper.body, state, params, f"helper:{name}"))
        nested, missing = _helper_checks(
            model, helper.contract, helper.body, state, params, seen, depth - 1
        )
        found.extend(nested)
        unresolved = unresolved or missing
    return found, unresolved


def _checks_in(text: str, state: dict[str, str], params: set[str], origin: str) -> list[AuthCheck]:
    found: list[AuthCheck] = []
    for match in re.finditer(
        rf"{_SENDER}\s*(==|!=)\s*([A-Za-z_][\w.]*(?:\(\s*[\w.]*\s*\))?)", text
    ):
        found.append(_classify(match.group(2), match.group(0), origin, state, params))
    for match in re.finditer(
        rf"([A-Za-z_][\w.]*(?:\(\s*[\w.]*\s*\))?)\s*(==|!=)\s*{_SENDER}", text
    ):
        found.append(_classify(match.group(1), match.group(0), origin, state, params))
    for word in _ROLE_WORDS:
        for match in re.finditer(rf"\b{word}\s*\(([^;]*)", text):
            found.append(AuthCheck("role", match.group(1)[:60], match.group(0)[:80], origin))
    for match in re.finditer(rf"\b(\w+)\s*\[\s*{_SENDER}\s*\]", text):
        context = text[max(0, match.start() - 40) : match.start()]
        if re.search(r"(require|if|assert)\s*\(\s*!?\s*$", context):
            found.append(AuthCheck("mapping", match.group(1), match.group(0), origin))
    if re.search(r"\b_checkOwner\s*\(\s*\)|\b_onlyOwner\s*\(\s*\)", text):
        found.append(AuthCheck("role", "owner", "_checkOwner()", origin))
    return found


def _classify(
    expected: str, text: str, origin: str, state: dict[str, str], params: set[str]
) -> AuthCheck:
    expr = expected.strip()
    negated = "!=" in text
    if expr in _SELF_EXPRESSIONS:
        return AuthCheck("self", expr, text, origin, negated)
    if expr in params:
        return AuthCheck("claimed_param", expr, text, origin, negated)
    if expr in state:
        return AuthCheck("state_account", expr, text, origin, negated)
    if expr.startswith("_msgSender"):
        return AuthCheck("forwarded_sender", expr, text, origin, negated)
    return AuthCheck("unknown", expr, text, origin, negated)


def has_sender_authorization(model: ResearchModel, function: RFunction) -> bool | None:
    """True when a sender check exists, False when none can exist, None when unknowable."""
    checks, unresolved = sender_checks(model, function)
    if any(check.kind != "unknown" or check.expected for check in checks):
        return True
    return None if unresolved else False


def context_edges(model: ResearchModel) -> list[ContextEdge]:
    """Bounded identity edges for every call in the model."""
    edges: list[ContextEdge] = []
    for function in model.all_functions():
        if not function.has_body:
            continue
        for call in member_calls(function.body):
            edge = _edge(function, call.receiver, call.name, call.start, function.body)
            if edge is not None:
                edges.append(edge)
                if len(edges) >= MAX_CONTEXT_EDGES:
                    return edges
    return edges


def _edge(
    function: RFunction, receiver: str, name: str, offset: int, body: str
) -> ContextEdge | None:
    line = function.line + body[:offset].count("\n")
    target = receiver.strip()
    if target in {"address(this)", "this"}:
        mechanism = {
            "call": "self_call",
            "delegatecall": "self_delegatecall",
            "staticcall": "self_staticcall",
        }.get(name, "self_call")
        sender = "preserved" if name == "delegatecall" else "this_contract"
        return ContextEdge(function.identity, f"self.{name}", mechanism, sender, line)
    if name in {"call", "staticcall"}:
        return ContextEdge(
            function.identity, f"{target}.{name}", "low_level_call", "this_contract", line
        )
    if name == "delegatecall":
        return ContextEdge(
            function.identity, f"{target}.delegatecall", "delegatecall", "preserved", line
        )
    if name in {"transfer", "send", "balanceOf", "encode", "encodePacked", "length"}:
        return None
    return ContextEdge(function.identity, f"{target}.{name}", "external", "this_contract", line)


def context_summary(model: ResearchModel) -> dict[str, bool]:
    """Which context mechanisms the target actually contains."""
    text = "\n".join(item.body for item in model.contracts.values())
    return {
        "entrypoint": bool(re.search(r"\bvalidateUserOp\b|\bPackedUserOperation\b", text)),
        "forwarder": bool(re.search(r"\bisTrustedForwarder\b|\b_msgSender\s*\(", text)),
        "self_call": bool(re.search(r"address\(this\)\s*\.\s*call\b|\bthis\.\w+\(", text)),
        "delegatecall": bool(re.search(r"\.\s*delegatecall\b", text)),
        "eip7702": bool(
            re.search(r"\b0xef0100\b|\bauthorization_list\b|\bEIP7702|\bIAuthorized", text)
        ),
    }


def _tainted(function: RFunction) -> set[str]:
    names = set(function.param_names())
    changed = True
    rounds = 0
    while changed and rounds < 4:
        changed = False
        rounds += 1
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*=\s*([^;=][^;]*);", function.body):
            left, right = match.group(1), match.group(2)
            if left not in names and any(mentions(right, name) for name in names):
                names.add(left)
                changed = True
    return names


def _controlled(expression: str, tainted: set[str]) -> bool:
    return any(mentions(expression, name) for name in tainted)


def analyze_caller_context(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    found.extend(_self_call_elevation(model))
    found.extend(_unrestricted_dispatch(model))
    found.extend(_trusted_intermediary(model))
    found.extend(_forwarded_sender_spoof(model))
    found.extend(swallowed_call_failures(model))
    return cap(found)


def _self_trusted_targets(model: ResearchModel, contract: str) -> list[tuple[RFunction, AuthCheck]]:
    targets: list[tuple[RFunction, AuthCheck]] = []
    lineage_text = "\n".join(
        model.contracts[name].body for name in model.lineage(contract) if name in model.contracts
    )
    for function in model.functions_of(contract):
        if not function.exposed or not function.has_body:
            continue
        checks, _ = sender_checks(model, function)
        for check in checks:
            if check.negated:
                continue
            if check.kind == "self":
                targets.append((function, check))
                break
            if check.kind == "state_account" and re.search(
                rf"\b{re.escape(check.expected)}\s*=\s*address\(this\)", lineage_text
            ):
                targets.append((function, check))
                break
            if check.kind in {"role", "mapping"} and re.search(
                r"(grantRole|_setupRole|_grantRole|\b" + re.escape(check.expected) + r"\s*\[)"
                r"[^;]*address\(this\)",
                lineage_text,
            ):
                targets.append((function, check))
                break
    return targets


def _dispatcher_unauthenticated(model: ResearchModel, function: RFunction) -> bool:
    state = has_sender_authorization(model, function)
    return state is False


def _self_call_elevation(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        if contract.is_interface:
            continue
        trusted = _self_trusted_targets(model, name)
        if not trusted:
            continue
        for function in model.functions_of(name):
            if not function.exposed or not function.has_body:
                continue
            if not _dispatcher_unauthenticated(model, function):
                continue
            tainted = _tainted(function)
            for call in member_calls(function.body):
                if call.receiver.strip() not in {"address(this)", "this"}:
                    continue
                if call.name not in {"call", "delegatecall"} or call.name == "delegatecall":
                    continue
                if not call.arguments or not _controlled(call.arguments[0], tainted):
                    continue
                for target, check in trusted:
                    found.append(_self_candidate(function, call.text, call.start, target, check))
    return found


def _self_candidate(
    function: RFunction, call_text: str, offset: int, target: RFunction, check: AuthCheck
) -> SemanticCandidate:
    line = function.line + function.body[:offset].count("\n")
    path = (
        f"{function.contract}.{function.signature} @{function.file}:{function.line}",
        f"{call_text[:60]}  (msg.sender becomes {function.contract})",
        f"{target.contract}.{target.signature} @{target.file}:{target.line}",
    )
    return SemanticCandidate(
        detector="caller_context.self_call_elevation",
        family=FAMILY,
        title="Caller-context confusion: nested self-call satisfies a self-trust check",
        summary=(
            f"{function.contract}.{function.name} lets any caller choose calldata that is "
            f"dispatched through address(this).call. In the nested frame msg.sender is the "
            f"contract itself, and {target.contract}.{target.name} authorizes "
            f"`{check.text.strip()}`, so the check evaluates the contract instead of the "
            f"original actor."
        ),
        file=function.file,
        line=line,
        contract=function.contract,
        function=function.signature,
        path=path,
        facts=(
            ("original_actor", "any external account calling the dispatching function"),
            ("intermediate_callers", function.contract),
            ("eventual_callee", target.identity),
            ("authorization_check", check.text.strip()[:120]),
            (
                "identity_difference",
                "msg.sender changes from the original actor to address(this) at the call boundary",
            ),
            ("identity_evaluated_by_check", "address(this)"),
        ),
        observed=(
            "dispatching function has no sender authorization",
            "calldata for the nested call derives from a caller-supplied parameter",
            "callee authorizes the contract's own address",
        ),
        missing=("a check that the original actor is allowed to reach the callee",),
        confidence="medium",
        impact_tags=("privileged_operations", "arbitrary_calls", "router_dispatch"),
    )


_RESTRICTION = re.compile(
    r"(allow|whitelist|approved|supported|valid|trusted|permitted|registry)", re.I
)


def _unrestricted_dispatch(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        if contract.is_interface:
            continue
        pulls = _approval_spender_evidence(model, name)
        if not pulls:
            continue
        for function in model.functions_of(name):
            if not function.exposed or not function.has_body:
                continue
            if not _dispatcher_unauthenticated(model, function):
                continue
            tainted = _tainted(function)
            for call in member_calls(function.body):
                if call.name != "call" or call.receiver.strip() in {"address(this)", "this"}:
                    continue
                if not call.arguments or not _controlled(call.arguments[0], tainted):
                    continue
                destination = _destination_param(call.receiver, tainted)
                if not destination or _restricted(function, call.receiver, destination):
                    continue
                found.append(_dispatch_candidate(function, call.text, call.start, pulls[0]))
    return found


def _approval_spender_evidence(model: ResearchModel, contract: str) -> list[str]:
    """Places where users must have approved this contract as an ERC20 spender."""
    found: list[str] = []
    for function in model.functions_of(contract):
        for call in member_calls(function.body):
            if call.name in {"transferFrom", "safeTransferFrom"} and len(call.arguments) >= 3:
                if call.arguments[1].strip() in {"address(this)", "this"} or (
                    call.arguments[1].strip() == "address(this)"
                ):
                    found.append(f"{function.contract}.{function.name}: {call.text[:80]}")
    return found


def _destination_param(receiver: str, tainted: set[str]) -> str:
    for name in sorted(tainted):
        if mentions(receiver, name):
            return name
    return ""


def _restricted(function: RFunction, receiver: str, destination: str) -> bool:
    text = function.body
    for match in re.finditer(r"\b(require|if|assert)\s*\(([^;]*)", text):
        condition = match.group(2)
        if mentions(condition, destination) and re.search(
            rf"\b\w+\s*\[\s*{re.escape(destination)}\s*\]", condition
        ):
            return True
        if mentions(condition, destination) and re.search(r"(==|!=)", condition):
            return True
    return any(
        mentions(item, destination) and bool(_RESTRICTION.search(item))
        for item in re.findall(r"[^;]*;", text)
        if re.search(r"\b(require|if)\b", item)
    ) or bool(re.search(rf"\b\w+\s*\[\s*{re.escape(destination)}\s*\]\s*\)", text))


def _dispatch_candidate(
    function: RFunction, call_text: str, offset: int, evidence: str
) -> SemanticCandidate:
    line = function.line + function.body[:offset].count("\n")
    return SemanticCandidate(
        detector="caller_context.unrestricted_dispatch_with_approval_authority",
        family=FAMILY,
        title="Caller-context confusion: caller-chosen call executes with the contract's allowances",
        summary=(
            f"{function.contract}.{function.name} performs a low-level call to a "
            f"caller-supplied destination with caller-supplied data and no sender "
            f"authorization or destination restriction. The contract also pulls tokens from "
            f"users ({evidence}), so users may have approved it. A token contract authorizes "
            f"the spender it sees, which is this contract, not the original actor."
        ),
        file=function.file,
        line=line,
        contract=function.contract,
        function=function.signature,
        path=(
            f"{function.contract}.{function.signature} @{function.file}:{function.line}",
            f"{call_text[:60]}  (msg.sender becomes {function.contract})",
            "destination contract authorizes the spender it observes",
        ),
        facts=(
            ("original_actor", "any external account calling the dispatching function"),
            ("intermediate_callers", function.contract),
            ("eventual_callee", "caller-supplied destination"),
            ("authorization_check", "destination evaluates msg.sender == this contract"),
            (
                "identity_difference",
                "the allowance or role is held by the dispatching contract, not by the actor",
            ),
            ("approval_evidence", evidence[:120]),
        ),
        observed=(
            "dispatching function has no sender authorization",
            "destination and calldata derive from caller-supplied parameters",
            "contract pulls tokens from users with transferFrom",
        ),
        missing=("destination or selector restriction",),
        confidence="medium",
        impact_tags=("token_approvals", "arbitrary_calls", "router_dispatch", "value_transfer"),
    )


def _trusted_intermediary(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for router in model.all_functions():
        if not router.exposed or not router.has_body:
            continue
        if has_sender_authorization(model, router) is not False:
            continue
        state = model.state_vars(router.contract)
        for call in member_calls(router.body):
            declared = _callee_contract(model, router.contract, state, call.receiver)
            if not declared:
                continue
            for callee_contract in _concrete_targets(model, declared, call.name):
                for callee in model.functions_named(callee_contract, call.name):
                    if not callee.has_body or not callee.exposed:
                        continue
                    if len(callee.params) != len(call.arguments):
                        continue
                    candidate = _intermediary_candidate(model, router, call, callee)
                    if candidate is not None:
                        found.append(candidate)
    return found


MAX_INTERFACE_TARGETS = 4


def _concrete_targets(model: ResearchModel, declared: str, name: str) -> tuple[str, ...]:
    """The contract itself, or its concrete implementations when the type is an interface.

    An interface-typed receiver says nothing about which code runs, so every
    implementation with the called function is considered, bounded.
    """
    item = model.contracts.get(declared)
    if item is None or not item.is_interface:
        return (declared,)
    found = [
        other
        for other in sorted(model.contracts)
        if other != declared
        and not model.contracts[other].is_interface
        and (declared in model.bases_of(other) or model.functions_named(other, name))
        and any(f.has_body for f in model.functions_named(other, name))
    ]
    return tuple(found[:MAX_INTERFACE_TARGETS])


def _callee_contract(
    model: ResearchModel, contract: str, state: dict[str, str], receiver: str
) -> str:
    receiver = receiver.strip()
    direct = re.fullmatch(r"([A-Za-z_]\w*)\s*\([^)]*\)", receiver)
    if direct and direct.group(1) in model.contracts:
        return direct.group(1)
    if receiver in state:
        return model.contract_for_type(state[receiver])
    return ""


def _intermediary_candidate(
    model: ResearchModel, router: RFunction, call: MemberCall, callee: RFunction
) -> SemanticCandidate | None:
    checks, _ = sender_checks(model, callee)
    trusted = [c for c in checks if c.kind == "state_account" and not c.negated]
    if not trusted:
        return None
    router_params = set(router.param_names())
    for index, parameter in enumerate(callee.params):
        argument = call.arguments[index].strip()
        if parameter.type_name != "address" or argument not in router_params:
            continue
        if not _actor_subject(callee, parameter.name):
            continue
        if _binds_sender(router, argument):
            continue
        check = trusted[0]
        line = router.line + router.body[: call.start].count("\n")
        return SemanticCandidate(
            detector="caller_context.trusted_intermediary_actor_parameter",
            family=FAMILY,
            title="Caller-context confusion: trusted intermediary forwards an unauthenticated actor",
            summary=(
                f"{callee.contract}.{callee.name} authorizes only `{check.text.strip()}` and then "
                f"acts for the address parameter `{parameter.name}`. "
                f"{router.contract}.{router.name} passes its own parameter `{argument}` to it "
                f"without binding it to msg.sender, so the callee evaluates the intermediary "
                f"instead of the original actor."
            ),
            file=router.file,
            line=line,
            contract=router.contract,
            function=router.signature,
            path=(
                f"{router.contract}.{router.signature} @{router.file}:{router.line}",
                f"{call.text[:70]}  (msg.sender becomes {router.contract})",
                f"{callee.contract}.{callee.signature} @{callee.file}:{callee.line}",
            ),
            facts=(
                ("original_actor", "any external account calling the intermediary"),
                ("intermediate_callers", router.contract),
                ("eventual_callee", callee.identity),
                ("authorization_check", check.text.strip()[:120]),
                (
                    "identity_difference",
                    f"callee trusts the intermediary while the acted-for account `{parameter.name}` "
                    "is supplied by the original actor",
                ),
                ("acted_for_parameter", parameter.name),
            ),
            observed=(
                "callee authorizes a stored account equal to the calling intermediary",
                "callee uses an address parameter as the account it acts for",
                "intermediary has no sender authorization for that parameter",
            ),
            missing=(f"binding of `{argument}` to msg.sender or a signature in the intermediary",),
            confidence="medium",
            impact_tags=("permissions_roles", "value_transfer", "router_dispatch"),
        )
    return None


def _actor_subject(callee: RFunction, name: str) -> bool:
    """The parameter selects whose assets or permissions the callee changes."""
    if not name:
        return False
    body = callee.body
    if re.search(rf"\b\w+\s*\[\s*{re.escape(name)}\s*\]\s*(-=|\+=|=[^=])", body):
        return True
    for call in member_calls(body):
        if call.name in {"transferFrom", "safeTransferFrom", "burn", "burnFrom", "_burn"}:
            if call.arguments and call.arguments[0].strip() == name:
                return True
    return bool(re.search(rf"\b_?burn\s*\(\s*{re.escape(name)}\b", body))


def _binds_sender(function: RFunction, name: str) -> bool:
    text = function.body
    if re.search(
        rf"{_SENDER}\s*(==|!=)\s*{re.escape(name)}\b|\b{re.escape(name)}\s*(==|!=)\s*{_SENDER}",
        text,
    ):
        return True
    if re.search(r"\b(ecrecover|isValidSignature|SignatureChecker|recover)\b", text):
        return True
    if re.search(rf"isApprovedForAll\s*\(\s*{re.escape(name)}\s*,", text):
        return True
    return bool(re.search(rf"\[\s*{re.escape(name)}\s*\]\s*\[\s*{_SENDER}\s*\]", text))


def _forwarded_sender_spoof(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        if model.contracts[name].is_interface:
            continue
        sender_fn = next(
            (
                f
                for f in model.functions_named(name, "_msgSender")
                if f.has_body and "msg.data" in f.body
            ),
            None,
        )
        if sender_fn is None:
            continue
        uses = [
            f
            for f in model.functions_of(name)
            if f.has_body and f.exposed and "_msgSender()" in (f.body + _modifier_text(model, f))
        ]
        if not uses:
            continue
        for function in model.functions_of(name):
            if not function.exposed or not function.has_body:
                continue
            tainted = _tainted(function)
            for call in member_calls(function.body):
                if call.name != "delegatecall" or call.receiver.strip() not in {
                    "address(this)",
                    "this",
                }:
                    continue
                if not call.arguments or not _controlled(call.arguments[0], tainted):
                    continue
                if re.search(
                    r"_contextSuffixLength|abi\.encodePacked\s*\([^;]*_msgSender", function.body
                ):
                    continue
                found.append(_spoof_candidate(function, sender_fn, uses[0], call.text, call.start))
    return found


def _modifier_text(model: ResearchModel, function: RFunction) -> str:
    parts = []
    for raw in function.modifiers:
        modifier = model.modifier(function.contract, raw.split("(", 1)[0])
        if modifier is not None:
            parts.append(modifier.body)
    return "\n".join(parts)


def _spoof_candidate(
    function: RFunction, sender_fn: RFunction, user: RFunction, call_text: str, offset: int
) -> SemanticCandidate:
    line = function.line + function.body[:offset].count("\n")
    return SemanticCandidate(
        detector="caller_context.forwarded_sender_delegatecall",
        family=FAMILY,
        title="Caller-context confusion: delegatecall preserves msg.sender but not the forwarded sender",
        summary=(
            f"{function.contract}.{function.name} delegatecalls itself with caller-supplied data. "
            f"delegatecall preserves msg.sender but {sender_fn.contract}._msgSender() derives the "
            f"sender from a calldata suffix, which the nested data controls. "
            f"{user.contract}.{user.name} authorizes through _msgSender(), so a trusted forwarder "
            f"call can be made to evaluate an attacker-chosen identity."
        ),
        file=function.file,
        line=line,
        contract=function.contract,
        function=function.signature,
        path=(
            f"{function.contract}.{function.signature} @{function.file}:{function.line}",
            f"{call_text[:60]}  (msg.sender preserved, calldata suffix replaced)",
            f"{user.contract}.{user.signature} @{user.file}:{user.line}",
        ),
        facts=(
            ("original_actor", "account relaying through a trusted forwarder"),
            ("intermediate_callers", "trusted forwarder; this contract via delegatecall"),
            ("eventual_callee", user.identity),
            ("authorization_check", "_msgSender() reads the sender from msg.data"),
            (
                "identity_difference",
                "the nested call keeps msg.sender but its calldata suffix is caller-supplied",
            ),
        ),
        observed=(
            "_msgSender() is derived from msg.data",
            "a function delegatecalls address(this) with caller-supplied data",
            "an external function authorizes through _msgSender()",
        ),
        missing=("the forwarded sender is not re-appended to each nested call",),
        confidence="medium",
        impact_tags=("permissions_roles", "router_dispatch", "privileged_operations"),
    )


__all__ = [
    "MAX_CONTEXT_EDGES",
    "AuthCheck",
    "ContextEdge",
    "analyze_caller_context",
    "context_edges",
    "context_summary",
    "has_sender_authorization",
    "sender_checks",
]

"""Proof, message, and signature binding analysis.

A security-critical digest or proof is only meaningful if it covers the fields that
control the decision it authorizes. This module resolves the digest a verification
reads, resolves what the function does after verification, and reports exactly
which decision-controlling fields the digest omits. A field that the function never
uses to decide anything is not reported, and a missing field is never called
exploitable by itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.parsing.solidity_research import (
    MemberCall,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    cap,
    member_calls,
    mentions,
    plain_calls,
)

FAMILY = "message_binding"
MAX_DIGEST_DEPTH = 4
MAX_FIELDS = 24
MAX_VERIFICATIONS_PER_FUNCTION = 4

_CRYPTO_SKIP = frozenset({"keccak256", "abi", "sha256", "ecrecover", "type", "bytes32", "uint256"})
_DOMAIN_MARKERS = (
    "DOMAIN_SEPARATOR",
    "_domainSeparatorV4",
    "_hashTypedDataV4",
    "EIP712Domain",
    "domainSeparator",
)
_CHAIN_RE = re.compile(r"block\.chainid|chain_?id|chainId|CHAIN_ID|chainid", re.I)
_CONSUMES = re.compile(
    r"\b\w+\s*(?:\[[^\]]*\]\s*){1,2}(?:=\s*(?:true|1)\b|\+\+|\+=\s*1\b)"
    r"|(?:\+\+)\s*\w+\s*\["
    r"|\b_?(?:use|consume|mark|set|invalidate)\w*(?:Nonce|Used|Claimed|Processed)\w*\s*\(",
    re.I,
)
_COUNTER = re.compile(r"\b(\w+)\s*\[([^\]]*)\]\s*(?:\+\+|\+=\s*1\b)|\+\+\s*(\w+)\s*\[([^\]]*)\]")


@dataclass(frozen=True)
class Verification:
    kind: str  # ecdsa | erc1271 | merkle
    call: MemberCall | None
    expression: str
    offset: int
    end: int


@dataclass(frozen=True)
class Decision:
    parameter: str
    role: str
    usage: str


def analyze_message_binding(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for function in model.all_functions():
        if not function.has_body or function.contract not in model.contracts:
            continue
        if _entry_point_validated(function):
            continue
        for verification in _verifications(function)[:MAX_VERIFICATIONS_PER_FUNCTION]:
            found.extend(_binding(model, function, verification))
    found.extend(_typehash_mismatch(model))
    found.extend(_cached_domain(model))
    return cap(found)


def _entry_point_validated(function: RFunction) -> bool:
    """ERC-4337 validation signs a hash the entry point computed over the whole operation.

    The account abstraction analyzer owns that family, so the generic field-binding
    heuristics do not apply to a function that receives a user operation.
    """
    return any("UserOperation" in item.type_name for item in function.params)


# ---- verification sites ---------------------------------------------------------------------


def _verifications(function: RFunction) -> list[Verification]:
    body = function.body
    found: list[Verification] = []
    for match in re.finditer(r"\becrecover\s*\(", body):
        args = plain_calls(body[match.start() :])
        if args and args[0][0] == "ecrecover" and args[0][1]:
            found.append(
                Verification("ecdsa", None, args[0][1][0], match.start(), match.start() + 10)
            )
    for call in member_calls(body):
        name = call.name
        if name in {"recover", "tryRecover"} and call.arguments:
            found.append(Verification("ecdsa", call, call.arguments[0], call.start, call.end))
        elif name in {"isValidSignatureNow", "isValidSignature"} and call.arguments:
            index = 1 if name == "isValidSignatureNow" and len(call.arguments) > 2 else 0
            found.append(Verification("erc1271", call, call.arguments[index], call.start, call.end))
        elif name in {"verify", "verifyCalldata", "processProof"} and len(call.arguments) >= 3:
            found.append(Verification("merkle", call, call.arguments[2], call.start, call.end))
    for plain_name, plain_args, offset in plain_calls(body):
        if (
            plain_name in {"verify", "_verify", "verifyProof"}
            and len(plain_args) >= 3
            and "proof" in body.lower()
        ):
            found.append(
                Verification("merkle", None, plain_args[-1], offset, offset + len(plain_name))
            )
    return sorted(found, key=lambda item: item.offset)


# ---- digest resolution ----------------------------------------------------------------------


def resolve_digest(
    model: ResearchModel, function: RFunction, expression: str, depth: int = 0
) -> str:
    """Expand a digest expression through local variables and in-model helpers."""
    text = expression.strip()
    if depth > MAX_DIGEST_DEPTH:
        return text
    if re.fullmatch(r"[A-Za-z_]\w*", text):
        assignment = re.search(rf"\b{re.escape(text)}\s*=\s*([^;]+);", function.body)
        if assignment:
            return f"{text} := {resolve_digest(model, function, assignment.group(1), depth + 1)}"
        return text
    expanded = text
    for name, args, _ in plain_calls(text):
        if name in _CRYPTO_SKIP:
            continue
        helper = model.function(function.contract, name)
        if helper is None or not helper.has_body or helper is function:
            continue
        returned = re.search(r"\breturn\s+([^;]+);", helper.body)
        if not returned:
            continue
        inner = resolve_digest(model, helper, returned.group(1), depth + 1)
        for parameter, argument in zip(helper.params, args, strict=False):
            if parameter.name:
                inner = re.sub(rf"\b{re.escape(parameter.name)}\b", argument.strip(), inner)
        expanded += f" || {inner}"
    for inner_match in re.finditer(r"\b([A-Za-z_]\w*)\b(?!\s*[(\[.])", text):
        name = inner_match.group(1)
        assignment = re.search(rf"\b{re.escape(name)}\s*=\s*([^;]+);", function.body)
        if assignment and name not in _CRYPTO_SKIP and depth < MAX_DIGEST_DEPTH:
            expanded += (
                f" || {name} := {resolve_digest(model, function, assignment.group(1), depth + 1)}"
            )
    return expanded


def _bindings(digest_text: str) -> dict[str, bool]:
    return {
        "chain": bool(_CHAIN_RE.search(digest_text))
        or any(m in digest_text for m in _DOMAIN_MARKERS),
        "contract": "address(this)" in digest_text
        or any(m in digest_text for m in _DOMAIN_MARKERS),
        "domain": any(m in digest_text for m in _DOMAIN_MARKERS),
    }


# ---- decisions ------------------------------------------------------------------------------


_PROOF_TYPES = {"bytes", "bytes32[]", "bytes[]", "uint8", "bytes32"}


def _decisions(function: RFunction, after: int) -> list[Decision]:
    tail = function.body[after:]
    statements = [s.strip() for s in re.findall(r"[^;{}]+[;{}]", tail)]
    effects = [s for s in statements if s and not re.match(r"(require|assert|if|emit|revert)\b", s)]
    decisions: list[Decision] = []
    for parameter in function.params:
        name = parameter.name
        if not name or parameter.type_name in _PROOF_TYPES or name in {"v", "r", "s"}:
            continue
        usage = next((s for s in effects if mentions(s, name)), "")
        if not usage:
            continue
        decisions.append(Decision(name, _role(function, name, usage), usage[:100]))
    return decisions[:MAX_FIELDS]


def _role(function: RFunction, name: str, usage: str) -> str:
    n = re.escape(name)
    parameter = next((p for p in function.params if p.name == name), None)
    for call in member_calls(usage):
        arguments = [a.strip() for a in call.arguments]
        if call.name in {"transfer", "safeTransfer", "send"} and arguments:
            if arguments[0] == name:
                return "recipient"
            if len(arguments) > 1 and arguments[1] == name:
                return "amount"
        if call.name in {"transferFrom", "safeTransferFrom"} and len(arguments) >= 3:
            if arguments[1] == name:
                return "recipient"
            if arguments[0] == name:
                return "sender"
            if arguments[2] == name:
                return "amount_or_token_id"
        if call.receiver.strip() == name or re.search(rf"\(\s*{n}\s*\)$", call.receiver):
            return "token"
        if call.name == "call" and re.search(rf"value\s*:\s*{n}\b", call.options):
            return "amount"
    for name_call, args, _ in plain_calls(usage):
        arguments = [a.strip() for a in args]
        if name_call in {"_mint", "_safeMint", "mint"} and arguments:
            if arguments[0] == name:
                return "recipient"
            if len(arguments) > 1 and arguments[1] == name:
                return "amount_or_token_id"
    if re.search(rf"\[\s*{n}\s*\]", usage):
        return "state_key"
    if parameter is not None and parameter.type_name == "address":
        return "address_argument"
    return "state_or_call_value"


# ---- candidates -----------------------------------------------------------------------------


def _binding(
    model: ResearchModel, function: RFunction, verification: Verification
) -> list[SemanticCandidate]:
    digest_text = resolve_digest(model, function, verification.expression)
    decisions = _decisions(function, verification.end)
    if not decisions and not _has_effect(function, verification.end):
        return []
    covered = {d.parameter for d in decisions if mentions(digest_text, d.parameter)}
    omitted = [d for d in decisions if d.parameter not in covered]
    bound = _bindings(digest_text)
    line = function.line + function.body[: verification.offset].count("\n")
    found: list[SemanticCandidate] = []
    base_path = (
        f"{function.contract}.{function.signature} @{function.file}:{function.line}",
        f"verification: {verification.kind} over `{verification.expression.strip()[:50]}`",
    )

    def make(
        detector: str,
        title: str,
        summary: str,
        missing: tuple[str, ...],
        tags: tuple[str, ...],
        extra: tuple[tuple[str, str], ...] = (),
        confidence: str = "medium",
    ) -> SemanticCandidate:
        return SemanticCandidate(
            detector=f"message.{detector}",
            family=FAMILY,
            title=title,
            summary=summary,
            file=function.file,
            line=line,
            contract=function.contract,
            function=function.signature,
            path=base_path,
            facts=(
                ("verification", verification.kind),
                ("bound_fields", ",".join(sorted(covered)) or "none"),
                ("omitted_fields", ",".join(d.parameter for d in omitted) or "none"),
                ("chain_bound", str(bound["chain"]).lower()),
                ("contract_bound", str(bound["contract"]).lower()),
                *extra,
            ),
            observed=(f"{verification.kind} verification precedes a state or value effect",),
            missing=missing,
            confidence=confidence,
            impact_tags=tags,
        )

    if omitted:
        names = ", ".join(f"{d.parameter} ({d.role})" for d in omitted)
        found.append(
            make(
                "field_not_bound",
                "Signed message or proof does not bind a field that controls the decision",
                f"{function.contract}.{function.name} verifies a {verification.kind} message and "
                f"then uses {names} in its effect, but the verified digest does not include "
                f"{'it' if len(omitted) == 1 else 'them'}. A holder of one valid message can "
                f"substitute the omitted field.",
                tuple(f"{d.parameter} in the verified digest" for d in omitted),
                ("signature_authorization", "bridge_message_verification", "value_transfer"),
                (("omitted_roles", ",".join(sorted({d.role for d in omitted}))),),
            )
        )
    consumed = bool(_CONSUMES.search(function.body[verification.end :]))
    counter = _COUNTER.search(function.body[verification.end :])
    nonce_in_digest = bool(re.search(r"nonce", digest_text, re.I))
    if _has_effect(function, verification.end) and not consumed and not nonce_in_digest:
        found.append(
            make(
                "replay_no_consumption",
                "Verified message can be replayed",
                f"{function.contract}.{function.name} acts on a verified {verification.kind} "
                f"message but neither includes a nonce in the digest nor records the message as "
                f"used, so the same message verifies again.",
                ("nonce in the digest", "used or claimed record"),
                ("signature_authorization", "value_transfer"),
            )
        )
    elif counter and not nonce_in_digest and verification.kind != "merkle":
        found.append(
            make(
                "nonce_not_in_digest",
                "Nonce is incremented but is not part of the signed digest",
                f"{function.contract}.{function.name} increments a counter after verifying a "
                f"message, but the digest does not contain the counter value, so the counter "
                f"does not stop the same signature from verifying again.",
                ("nonce value in the verified digest",),
                ("signature_authorization",),
            )
        )
    if (
        _has_effect(function, verification.end)
        and verification.kind != "merkle"
        and not (bound["chain"] and bound["contract"])
    ):
        absent = [
            n
            for n, ok in (("chain id", bound["chain"]), ("verifying contract", bound["contract"]))
            if not ok
        ]
        found.append(
            make(
                "missing_domain_separation",
                "Signed message lacks domain separation",
                f"{function.contract}.{function.name} verifies a signature over a digest with no "
                f"{' or '.join(absent)}, so a signature for another contract or chain verifies "
                f"here.",
                tuple(f"{n} in the verified digest" for n in absent),
                ("signature_authorization", "bridge_message_verification"),
                confidence="low",
            )
        )
    if (
        _has_effect(function, verification.end)
        and verification.kind != "merkle"
        and "block.timestamp" not in function.body
        and not re.search(r"deadline|expir|validUntil", digest_text, re.I)
    ):
        found.append(
            make(
                "missing_expiry",
                "Signed message has no expiry",
                f"{function.contract}.{function.name} accepts a verified message that carries no "
                "deadline and performs no time check.",
                ("deadline or expiry in the verified digest",),
                ("signature_authorization",),
                confidence="low",
            )
        )
    return found


def _has_effect(function: RFunction, after: int) -> bool:
    tail = function.body[after:]
    if re.search(r"\.\s*(transfer|safeTransfer|safeTransferFrom|transferFrom|send)\s*\(", tail):
        return True
    if re.search(r"\bcall\s*\{\s*value", tail) or re.search(r"\b_?(mint|safeMint|burn)\s*\(", tail):
        return True
    return bool(re.search(r"\b\w+\s*(\[[^\]]*\]\s*)+(=|\+=|-=)[^=]", tail)) or bool(
        re.search(r"(^|\s)(owner|admin|root|implementation|signer)\w*\s*=[^=]", tail)
    )


# ---- signature composition ------------------------------------------------------------------


def _typehash_mismatch(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        typehashes: dict[str, tuple[str, int]] = {}
        for match in re.finditer(
            r"\b([A-Za-z_]\w*)\s*=\s*keccak256\s*\(\s*\"([A-Za-z_]\w*\(([^\"]*)\))\"\s*\)",
            contract.body,
        ):
            fields = [f for f in match.group(3).split(",") if f.strip()]
            typehashes[match.group(1)] = (match.group(2), len(fields))
        if not typehashes:
            continue
        for function in contract.functions:
            for call in member_calls(function.body):
                if call.name != "encode" or call.receiver.strip() != "abi" or not call.arguments:
                    continue
                if call.arguments[0].strip() not in typehashes:
                    continue
                struct, declared = typehashes[call.arguments[0].strip()]
                encoded = len(call.arguments) - 1
                if encoded != declared:
                    found.append(
                        _typehash_candidate(function, struct, declared, encoded, call.start)
                    )
    return found


def _typehash_candidate(
    function: RFunction, struct: str, declared: int, encoded: int, offset: int
) -> SemanticCandidate:
    return SemanticCandidate(
        detector="signature.typehash_arity_mismatch",
        family=FAMILY,
        title="Typed-data struct and encoded values disagree",
        summary=(
            f"{function.contract}.{function.name} hashes `{struct}` which declares {declared} "
            f"fields but encodes {encoded} values, so the signature authenticates a different "
            f"operation than the typed data describes."
        ),
        file=function.file,
        line=function.line + function.body[:offset].count("\n"),
        contract=function.contract,
        function=function.signature,
        path=(f"{function.contract}.{function.signature} @{function.file}:{function.line}",),
        facts=(
            ("struct", struct[:100]),
            ("declared_fields", str(declared)),
            ("encoded_values", str(encoded)),
        ),
        observed=("typehash string and abi.encode arity are both present",),
        missing=("one encoded value for every declared field",),
        confidence="high",
        impact_tags=("signature_authorization",),
    )


def _cached_domain(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        text = contract.body
        cached = re.search(
            r"\b(DOMAIN_SEPARATOR|_?domainSeparator)\w*\s*=\s*keccak256\s*\([^;]*(block\.chainid|chainid)[^;]*;",
            text,
        )
        if not cached:
            continue
        variable = cached.group(1)
        immutable_or_state = any(
            var == variable or var.startswith(variable) for var, _ in contract.state_vars
        )
        if not immutable_or_state:
            continue
        refreshed = re.search(
            r"block\.chainid\s*(==|!=)|(==|!=)\s*block\.chainid|_cachedChainId|CACHED_CHAIN", text
        )
        used = any(
            variable in function.body
            and ("ecrecover" in function.body or ".recover" in function.body)
            for function in contract.functions
        )
        if refreshed or not used:
            continue
        found.append(
            SemanticCandidate(
                detector="signature.domain_not_refreshed_on_fork",
                family=FAMILY,
                title="Domain separator is cached and never refreshed for a chain-id change",
                summary=(
                    f"{contract.name} computes {variable} once with the chain id and never "
                    "compares block.chainid with the cached value, so signatures stay valid on a "
                    "chain fork."
                ),
                file=contract.file,
                line=contract.line,
                contract=contract.name,
                function="",
                facts=(("domain_variable", variable),),
                observed=("domain separator is stored once",),
                missing=("block.chainid comparison or recomputation",),
                confidence="low",
                impact_tags=("signature_authorization",),
            )
        )
    return found


__all__ = ["Decision", "Verification", "analyze_message_binding", "resolve_digest"]

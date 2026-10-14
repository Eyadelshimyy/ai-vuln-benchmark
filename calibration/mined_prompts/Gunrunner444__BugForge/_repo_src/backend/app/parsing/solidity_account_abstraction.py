"""ERC-4337 account-abstraction checks. They run only when the target contains them.

Nothing here assumes an account exists: the analysis activates when a source
declares ``validateUserOp``, ``validatePaymasterUserOp``, or a user-operation type.
Where an inherited base is not part of the analyzed sources, a binding that the
base might supply stays unknown and is not reported.
"""

from __future__ import annotations

import re

from app.parsing.solidity_caller_context import has_sender_authorization
from app.parsing.solidity_message_binding import _bindings, resolve_digest
from app.parsing.solidity_research import (
    ResearchModel,
    RFunction,
    SemanticCandidate,
    cap,
    member_calls,
    mentions,
)

FAMILY = "account_abstraction"
_ENTRYPOINT = re.compile(r"entry_?point", re.I)
_VERIFIERS = re.compile(r"\becrecover\b|\.recover\b|\.tryRecover\b|isValidSignature")


def account_abstraction_present(model: ResearchModel) -> bool:
    for function in model.all_functions():
        if function.name in {"validateUserOp", "validatePaymasterUserOp"}:
            return True
        if any(re.search(r"(Packed)?UserOperation", p.type_name) for p in function.params):
            return True
    return False


def analyze_account_abstraction(model: ResearchModel) -> list[SemanticCandidate]:
    if not account_abstraction_present(model):
        return []
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        if contract.is_interface:
            continue
        functions = model.functions_of(name)
        has_account = any(f.name == "validateUserOp" and f.has_body for f in functions)
        for function in functions:
            if not function.has_body or function.contract != name:
                continue
            if function.name == "validateUserOp":
                found.extend(_validate_user_op(model, function))
            elif function.name == "validatePaymasterUserOp":
                found.extend(_validate_paymaster(model, function))
            elif function.name == "postOp":
                found.extend(_post_op(model, function))
        if has_account:
            found.extend(_unauthenticated_execution(model, name, functions))
        found.extend(_factory(model, name))
        found.extend(_initializer(model, name, functions))
    return cap(found)


def _entrypoint_known(model: ResearchModel, function: RFunction) -> bool | None:
    """True when an entry-point binding exists, False when none can exist, None when unknowable."""
    text = function.body
    for raw in function.modifiers:
        modifier = model.modifier(function.contract, raw.split("(", 1)[0])
        if modifier is None:
            if _ENTRYPOINT.search(raw):
                return True
            continue
        text += "\n" + modifier.body
    for call in re.finditer(r"\b(_?require\w*|_check\w*)\s*\(", function.body):
        if _ENTRYPOINT.search(call.group(1)):
            return True
        helper = model.function(function.contract, call.group(1))
        if helper is not None and helper.has_body:
            text += "\n" + helper.body
    if re.search(
        r"msg\.sender[^;]*" + _ENTRYPOINT.pattern + "|" + _ENTRYPOINT.pattern + r"[^;]*msg\.sender",
        text,
        re.I,
    ):
        return True
    unresolved = any(
        base not in model.contracts for base in _declared_bases(model, function.contract)
    )
    return None if unresolved else False


def _declared_bases(model: ResearchModel, contract: str) -> list[str]:
    bases: list[str] = []
    for name in model.lineage(contract):
        item = model.contracts.get(name)
        if item is not None:
            bases.extend(item.bases)
    return bases


def _candidate(
    function: RFunction,
    detector: str,
    title: str,
    summary: str,
    missing: tuple[str, ...],
    observed: tuple[str, ...],
    tags: tuple[str, ...],
    facts: tuple[tuple[str, str], ...] = (),
    confidence: str = "medium",
    line: int | None = None,
) -> SemanticCandidate:
    return SemanticCandidate(
        detector=f"aa.{detector}",
        family=FAMILY,
        title=title,
        summary=summary,
        file=function.file,
        line=line or function.line,
        contract=function.contract,
        function=function.signature,
        path=(f"{function.contract}.{function.signature} @{function.file}:{function.line}",),
        facts=facts,
        observed=observed,
        missing=missing,
        confidence=confidence,
        impact_tags=tags,
    )


def _validate_user_op(model: ResearchModel, function: RFunction) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    tags = ("account_abstraction_validation", "signature_authorization", "privileged_operations")
    if _entrypoint_known(model, function) is False and function.exposed:
        found.append(
            _candidate(
                function,
                "validate_without_entrypoint_binding",
                "validateUserOp does not bind its caller to the EntryPoint",
                f"{function.contract}.validateUserOp is externally callable and checks no "
                "EntryPoint caller, so anyone can trigger validation side effects such as "
                "nonce or prefund handling.",
                ("msg.sender equals the EntryPoint",),
                ("validateUserOp is exposed",),
                tags,
            )
        )
    verification = _verification_in(model, function)
    hash_param = next((p.name for p in function.params if p.type_name == "bytes32" and p.name), "")
    if verification is None:
        found.append(
            _candidate(
                function,
                "no_signature_validation",
                "validateUserOp performs no signature validation",
                f"{function.contract}.validateUserOp neither verifies a signature nor calls a "
                "helper that does, so every user operation passes validation.",
                ("signature verification of the user operation",),
                ("validateUserOp has a body",),
                tags,
            )
        )
        return found
    where, expression, recovered = verification
    digest_text = resolve_digest(model, where, expression)
    bindings = _bindings(digest_text)
    if hash_param and mentions(digest_text, hash_param):
        pass
    elif not (bindings["chain"] and bindings["contract"]):
        found.append(
            _candidate(
                function,
                "signature_not_bound_to_userophash",
                "Account signature is not bound to the user operation hash",
                f"{function.contract}.validateUserOp verifies a signature over a digest that "
                "neither is the EntryPoint's userOpHash nor binds a chain id and the EntryPoint, "
                "so a signature can replay across chains or EntryPoints.",
                ("userOpHash, or chain id and EntryPoint, in the verified digest",),
                ("signature is verified over a custom digest",),
                tags,
                (("digest_chain_bound", str(bindings["chain"]).lower()),),
            )
        )
    if recovered and not re.search(
        rf"\b{re.escape(recovered)}\b\s*(==|!=)|(==|!=)\s*{re.escape(recovered)}\b", where.body
    ):
        found.append(
            _candidate(
                function,
                "signature_result_ignored",
                "Recovered signer is never compared with the account owner",
                f"{function.contract}.validateUserOp recovers `{recovered}` but never compares "
                "it, so validation succeeds for any signature.",
                ("comparison of the recovered signer with the authorized owner",),
                ("a signer is recovered",),
                tags,
            )
        )
    return found


def _verification_in(
    model: ResearchModel, function: RFunction
) -> tuple[RFunction, str, str] | None:
    candidates = [function]
    for call in re.finditer(r"\b(_[A-Za-z]\w*)\s*\(", function.body):
        helper = model.function(function.contract, call.group(1))
        if helper is not None and helper.has_body and helper is not function:
            candidates.append(helper)
    for item in candidates:
        for match in re.finditer(
            r"\s*(?:(?:address\s+(\w+)\s*=\s*)|(?:\breturn\s+))?([^;]*)", item.body
        ):
            statement = match.group(0)
            if not _VERIFIERS.search(statement):
                continue
            calls = member_calls(statement)
            expression = ""
            for verifier in calls:
                if verifier.name in {"recover", "tryRecover"} and verifier.arguments:
                    expression = verifier.arguments[0]
                elif verifier.name == "isValidSignature" and verifier.arguments:
                    expression = verifier.arguments[0]
            ec = re.search(r"\becrecover\s*\(\s*([^,]+),", statement)
            if ec:
                expression = ec.group(1)
            if expression:
                return item, expression, match.group(1) or ""
    return None


def _validate_paymaster(model: ResearchModel, function: RFunction) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    tags = ("account_abstraction_validation", "signature_authorization", "value_transfer")
    if _entrypoint_known(model, function) is False and function.exposed:
        found.append(
            _candidate(
                function,
                "paymaster_validate_without_entrypoint_binding",
                "validatePaymasterUserOp does not bind its caller to the EntryPoint",
                f"{function.contract}.validatePaymasterUserOp is externally callable and checks "
                "no EntryPoint caller.",
                ("msg.sender equals the EntryPoint",),
                ("validatePaymasterUserOp is exposed",),
                tags,
            )
        )
    verification = _verification_in(model, function)
    if verification is None:
        return found
    where, expression, _ = verification
    digest_text = resolve_digest(model, where, expression)
    bindings = _bindings(digest_text)
    absent = [
        label
        for label, ok in (
            ("chain id", bindings["chain"]),
            ("paymaster address", bindings["contract"]),
        )
        if not ok
    ]
    if absent:
        found.append(
            _candidate(
                function,
                "paymaster_authorization_not_bound",
                "Paymaster authorization is not bound to its domain",
                f"{function.contract}.validatePaymasterUserOp verifies a paymaster signature "
                f"over a digest without {' or '.join(absent)}, so an approval for one paymaster "
                "or chain is accepted by another.",
                tuple(f"{label} in the verified digest" for label in absent),
                ("paymaster signature is verified",),
                tags,
            )
        )
    return found


def _post_op(model: ResearchModel, function: RFunction) -> list[SemanticCandidate]:
    if not function.exposed or _entrypoint_known(model, function) is not False:
        return []
    return [
        _candidate(
            function,
            "postop_without_entrypoint_binding",
            "postOp can be called by anyone",
            f"{function.contract}.postOp adjusts accounting and checks no EntryPoint caller, "
            "so post-operation charges can be forged.",
            ("msg.sender equals the EntryPoint",),
            ("postOp is exposed",),
            ("account_abstraction_validation", "reserve_accounting", "value_transfer"),
        )
    ]


def _unauthenticated_execution(
    model: ResearchModel, contract: str, functions: tuple[RFunction, ...]
) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for function in functions:
        if not function.exposed or function.name in {"validateUserOp", "postOp"}:
            continue
        executes = any(
            call.name == "call"
            and re.search(r"\bvalue\s*:", call.options or "")
            or (call.name == "call" and call.arguments)
            for call in member_calls(function.body)
        )
        if not executes or has_sender_authorization(model, function) is not False:
            continue
        if _entrypoint_known(model, function) is not False:
            continue
        found.append(
            _candidate(
                function,
                "unauthenticated_account_execution",
                "Account execution is not tied to validation",
                f"{contract}.{function.name} performs an arbitrary call from the account but "
                "checks neither the EntryPoint nor an owner, so validateUserOp's authorization "
                "is bypassed by calling it directly.",
                ("EntryPoint or owner check on the execution path",),
                ("the account validates user operations", "the function performs a low-level call"),
                ("account_abstraction_validation", "arbitrary_calls", "value_transfer"),
            )
        )
    return found


def _factory(model: ResearchModel, contract: str) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for function in model.functions_of(contract):
        if not function.exposed or not function.has_body or function.contract != contract:
            continue
        if not re.search(r"\bcreate2\b|\bnew\s+\w+\s*\{\s*salt\s*:", function.body):
            continue
        owner_params = [p.name for p in function.params if p.type_name == "address" and p.name]
        salt_params = [
            p.name for p in function.params if p.type_name in {"bytes32", "uint256"} and p.name
        ]
        if not owner_params or not salt_params:
            continue
        salt_use = re.search(r"salt\s*:\s*([^}]+)\}|create2\s*\([^;]*", function.body)
        if salt_use is None:
            continue
        derived = any(mentions(salt_use.group(0), owner) for owner in owner_params)
        if not derived:
            for name in set(re.findall(r"[A-Za-z_]\w*", salt_use.group(0))):
                assigned = re.search(rf"\b{re.escape(name)}\s*=\s*([^;]+);", function.body)
                if assigned and any(mentions(assigned.group(1), owner) for owner in owner_params):
                    derived = True
        if not derived:
            found.append(
                _candidate(
                    function,
                    "factory_salt_not_bound_to_owner",
                    "Account factory salt is independent of the owner",
                    f"{contract}.{function.name} deploys an account with a caller-chosen salt that "
                    "is not derived from the owner, so an attacker can occupy the address a "
                    "victim expects.",
                    ("owner in the CREATE2 salt or initializer data",),
                    ("a deterministic deployment takes a salt and an owner",),
                    ("account_abstraction_validation", "permissions_roles"),
                )
            )
    return found


def _initializer(
    model: ResearchModel, contract: str, functions: tuple[RFunction, ...]
) -> list[SemanticCandidate]:
    if not any(f.name == "validateUserOp" for f in functions):
        return []
    found: list[SemanticCandidate] = []
    for function in functions:
        if function.contract != contract or not function.exposed or not function.has_body:
            continue
        if function.name not in {"initialize", "init"}:
            continue
        guarded = any("initializer" in m for m in function.modifiers) or re.search(
            r"require\s*\([^;]*(initialized|owner\s*==\s*address\(0\)|!\s*init)", function.body
        )
        if guarded or not re.search(r"\bowner\w*\s*=\s*\w+", function.body):
            continue
        found.append(
            _candidate(
                function,
                "unprotected_account_initializer",
                "Smart account initializer can be called again by anyone",
                f"{contract}.{function.name} assigns the account owner without an initializer "
                "guard, so the first or any later caller can take control of the account.",
                ("one-time initialization guard",),
                ("an exposed initializer assigns the owner",),
                ("account_abstraction_validation", "permissions_roles", "privileged_operations"),
            )
        )
    return found

"""Token balance-delta accounting analysis.

The dangerous assumption is that a contract's balance after an operation equals what
the operation received. It fails for fee-on-transfer tokens, rebasing tokens, tokens
that return false, and balances changed by an external transfer. Each candidate
names the credited expression and the missing balance-delta control. The token's
behavior is reported as ``unknown`` unless the source establishes it.
"""

from __future__ import annotations

import re

from app.parsing.solidity_research import (
    MemberCall,
    ResearchModel,
    RFunction,
    SemanticCandidate,
    cap,
    member_calls,
    mentions,
)

FAMILY = "balance_delta"
_BALANCE = r"balanceOf\s*\(\s*address\(this\)\s*\)"
_CREDIT = re.compile(
    r"(?:\b\w+\s*(?:\[[^\]]*\]\s*)*(?:\+=|=)\s*[^;=][^;]*;)|(?:\b_?(?:mint|safeMint)\s*\([^;]*;)"
)


def analyze_balance_delta(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for function in model.all_functions():
        if not function.has_body or function.contract not in model.contracts:
            continue
        if model.contracts[function.contract].is_interface:
            continue
        found.extend(_inflow(model, function))
        found.extend(_pull_all(model, function))
        found.extend(_balance_as_deposit(model, function))
        found.extend(_unchecked_return(function))
    found.extend(_cached_balance(model))
    return cap(found)


def _token_origin(model: ResearchModel, function: RFunction, receiver: str) -> tuple[str, str]:
    """(origin, confidence): is the token any caller-chosen address or a fixed asset?"""
    name = re.sub(r"^\w+\((.*)\)$", r"\1", receiver.strip())
    if any(p.name == name for p in function.params):
        return "caller-chosen token parameter", "medium"
    if name in model.state_vars(function.contract):
        return "configured token variable", "low"
    return "token expression", "low"


def _inflow(model: ResearchModel, function: RFunction) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    body = function.body
    for call in member_calls(body):
        if call.name not in {"transferFrom", "safeTransferFrom"} or len(call.arguments) < 3:
            continue
        if call.arguments[1].strip() not in {"address(this)", "this"}:
            continue
        amount = call.arguments[2].strip()
        tail = body[call.end :]
        before = body[: call.start]
        measured = len(re.findall(_BALANCE, before)) >= 1 and len(re.findall(_BALANCE, tail)) >= 1
        credit = _credited(tail, amount)
        if measured or not credit:
            continue
        origin, confidence = _token_origin(model, function, call.receiver)
        line = function.line + body[: call.start].count("\n")
        found.append(
            SemanticCandidate(
                detector="accounting.fee_on_transfer_mismatch",
                family=FAMILY,
                title="Deposit is credited with the requested amount, not the received amount",
                summary=(
                    f"{function.contract}.{function.name} pulls `{amount}` with "
                    f"`{call.name}` and credits `{credit}` without measuring the balance change. "
                    f"For a fee-on-transfer, rebasing, or non-reverting token the credit exceeds "
                    f"what the contract received. Token behavior: unknown ({origin})."
                ),
                file=function.file,
                line=line,
                contract=function.contract,
                function=function.signature,
                path=(
                    f"{function.contract}.{function.signature} @{function.file}:{function.line}",
                    f"pull: {call.text[:70]}",
                    f"credit: {credit[:70]}",
                ),
                facts=(
                    ("token_behavior", "unknown"),
                    ("token_origin", origin),
                    ("credited_expression", credit[:100]),
                    ("requested_amount", amount[:60]),
                ),
                observed=("tokens are pulled into the contract", "state is credited afterwards"),
                missing=("post-balance minus pre-balance accounting",),
                confidence=confidence,
                impact_tags=("custody", "vault_accounting", "value_transfer"),
            )
        )
    return found


def _credited(tail: str, amount: str) -> str:
    for match in _CREDIT.finditer(tail):
        statement = match.group(0)
        if re.match(r"\s*(require|assert|if|emit|return)\b", statement):
            continue
        right = statement.split("=", 1)[1] if "=" in statement else statement
        if mentions(right, amount) and not re.search(_BALANCE, statement):
            return statement.strip().rstrip(";")
    return ""


def _pull_all(model: ResearchModel, function: RFunction) -> list[SemanticCandidate]:
    if not function.exposed:
        return []
    state = model.contracts[function.contract].state_vars
    tracked = [
        name for name, kind in state if kind.startswith("mapping") or kind.startswith("uint")
    ]
    if not tracked:
        return []
    from app.parsing.solidity_caller_context import has_sender_authorization

    if has_sender_authorization(model, function) is not False:
        return []
    found: list[SemanticCandidate] = []
    for call in member_calls(function.body):
        if call.name not in {"transfer", "safeTransfer"} or len(call.arguments) < 2:
            continue
        amount = call.arguments[1]
        if not re.search(_BALANCE, amount) or re.search(r"[-]", amount):
            continue
        found.append(
            SemanticCandidate(
                detector="accounting.pull_all_balance",
                family=FAMILY,
                title="Function sends the contract's entire token balance",
                summary=(
                    f"{function.contract}.{function.name} transfers the whole absolute balance "
                    f"(`{amount.strip()[:60]}`) while the contract also tracks per-account "
                    f"state, so donated funds and other accounts' funds leave with it."
                ),
                file=function.file,
                line=function.line + function.body[: call.start].count("\n"),
                contract=function.contract,
                function=function.signature,
                path=(
                    f"{function.contract}.{function.signature} @{function.file}:{function.line}",
                ),
                facts=(
                    ("sent_expression", amount.strip()[:80]),
                    ("tracked_state", ",".join(tracked[:4])),
                ),
                observed=("absolute balance is sent", "the contract keeps its own accounting"),
                missing=("a recorded entitlement for the amount sent",),
                confidence="medium",
                impact_tags=("custody", "value_transfer", "vault_accounting"),
            )
        )
    return found


def _balance_as_deposit(model: ResearchModel, function: RFunction) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    body = function.body
    for match in re.finditer(rf"\b(\w+)\s*=\s*([^;]*{_BALANCE}[^;]*);", body):
        variable, expression = match.group(1), match.group(2)
        if re.search(r"[-]\s*\w", expression):
            continue
        tail = body[match.end() :]
        if re.search(rf"/\s*\(?\s*{re.escape(variable)}\b", tail):
            found.extend(_share_price(function, variable, match, tail))
            continue
        credit = _credited(tail, variable)
        if not credit or re.search(
            rf"\b{_BALANCE}", tail[: tail.find(variable) if variable in tail else 0]
        ):
            continue
        before = body[: match.start()]
        if re.search(_BALANCE, before):
            continue
        found.append(
            SemanticCandidate(
                detector="accounting.balance_as_deposit",
                family=FAMILY,
                title="Absolute token balance is credited as one account's deposit",
                summary=(
                    f"{function.contract}.{function.name} credits `{credit[:60]}` where "
                    f"`{variable}` is the contract's whole balance. Funds that were already "
                    f"there, including a donation, are credited to the caller."
                ),
                file=function.file,
                line=function.line + body[: match.start()].count("\n"),
                contract=function.contract,
                function=function.signature,
                path=(
                    f"{function.contract}.{function.signature} @{function.file}:{function.line}",
                ),
                facts=(
                    ("credited_expression", credit[:100]),
                    ("balance_expression", expression.strip()[:80]),
                ),
                observed=("balanceOf(address(this)) is credited",),
                missing=("subtraction of the previously recorded balance",),
                confidence="medium",
                impact_tags=("custody", "vault_accounting", "value_transfer"),
            )
        )
    return found


def _share_price(
    function: RFunction, variable: str, match: re.Match[str], tail: str
) -> list[SemanticCandidate]:
    if re.search(r"\+\s*1\b|MINIMUM_LIQUIDITY|address\(0\)", function.body):
        return []
    use = re.search(rf"[^;]*/\s*\(?\s*{re.escape(variable)}\b[^;]*;", tail)
    if use is None:
        return []
    return [
        SemanticCandidate(
            detector="accounting.donation_share_price",
            family=FAMILY,
            title="Share price depends on the absolute token balance",
            summary=(
                f"{function.contract}.{function.name} divides by `{variable}`, the contract's "
                f"whole token balance, to price new shares. A direct token transfer raises the "
                f"balance without minting shares, so the next depositor receives fewer shares "
                f"than their deposit is worth."
            ),
            file=function.file,
            line=function.line + function.body[: match.start()].count("\n"),
            contract=function.contract,
            function=function.signature,
            path=(f"{function.contract}.{function.signature} @{function.file}:{function.line}",),
            facts=(("denominator", variable), ("conversion", use.group(0).strip()[:100])),
            observed=("balanceOf(address(this)) is the share-price denominator",),
            missing=("internally tracked assets or virtual shares",),
            confidence="medium",
            impact_tags=("vault_accounting", "custody", "value_transfer"),
        )
    ]


def _unchecked_return(function: RFunction) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    body = function.body
    for call in member_calls(body):
        if call.name not in {"transfer", "transferFrom"}:
            continue
        if call.name == "transfer" and len(call.arguments) != 2:
            continue
        if call.name == "transferFrom" and len(call.arguments) != 3:
            continue
        before = body[: call.start]
        line_start = before.rfind(";") + 1
        lead = body[line_start : call.start]
        if (
            re.search(r"(require|assert|if|bool|=|return|&&|\|\|)\s*\(?\s*$", lead)
            or "(" in lead.strip()
        ):
            continue
        if call.receiver.strip() in {"payable", ""} or call.receiver.strip().startswith("payable"):
            continue
        tail = body[call.end :]
        amount = call.arguments[-1].strip()
        credit = _credited(tail, amount) if call.name == "transferFrom" else ""
        before_credit = _credited(before, amount) if call.name == "transfer" else ""
        effect = credit or before_credit
        if not effect:
            continue
        found.append(
            SemanticCandidate(
                detector="accounting.unchecked_token_return",
                family=FAMILY,
                title="Token transfer result is ignored while accounting changes",
                summary=(
                    f"{function.contract}.{function.name} ignores the boolean result of "
                    f"`{call.name}` and updates `{effect[:60]}`. A token that returns false "
                    f"instead of reverting leaves accounting changed without a transfer."
                ),
                file=function.file,
                line=function.line + body[: call.start].count("\n"),
                contract=function.contract,
                function=function.signature,
                path=(
                    f"{function.contract}.{function.signature} @{function.file}:{function.line}",
                ),
                facts=(("ignored_call", call.text[:80]), ("accounting_effect", effect[:80])),
                observed=("transfer result is not checked",),
                missing=("require on the result, or SafeERC20",),
                confidence="medium",
                impact_tags=("custody", "vault_accounting"),
            )
        )
    return found


def _cached_balance(model: ResearchModel) -> list[SemanticCandidate]:
    found: list[SemanticCandidate] = []
    for name in sorted(model.contracts):
        contract = model.contracts[name]
        state = dict(contract.state_vars)
        for function in contract.functions:
            if not function.has_body:
                continue
            for match in re.finditer(rf"\b(\w+)\s*=\s*[^;]*{_BALANCE}[^;]*;", function.body):
                variable = match.group(1)
                if variable not in state or "mapping" in state[variable]:
                    continue
                spenders = [
                    other
                    for other in contract.functions
                    if other is not function
                    and other.has_body
                    and re.search(rf"\.(transfer|safeTransfer)\s*\([^;]*\b{variable}\b", other.body)
                ]
                if not spenders:
                    continue
                found.append(
                    SemanticCandidate(
                        detector="accounting.cached_balance",
                        family=FAMILY,
                        title="Token balance is cached and later paid out",
                        summary=(
                            f"{name}.{function.name} stores the contract's token balance in "
                            f"`{variable}` and {name}.{spenders[0].name} later pays out that stored "
                            f"value. A rebasing token or an external transfer changes the real "
                            f"balance without updating the cache. Token behavior: unknown."
                        ),
                        file=function.file,
                        line=function.line + function.body[: match.start()].count("\n"),
                        contract=name,
                        function=function.signature,
                        path=(
                            f"{name}.{function.signature} @{function.file}:{function.line}",
                            f"{name}.{spenders[0].signature} @{spenders[0].file}:{spenders[0].line}",
                        ),
                        facts=(("cached_variable", variable), ("token_behavior", "unknown")),
                        observed=("balance is cached in storage", "the cache is paid out later"),
                        missing=("re-measuring the balance before paying out",),
                        confidence="low",
                        impact_tags=("custody", "vault_accounting"),
                    )
                )
    return found


__all__ = ["MemberCall", "analyze_balance_delta"]

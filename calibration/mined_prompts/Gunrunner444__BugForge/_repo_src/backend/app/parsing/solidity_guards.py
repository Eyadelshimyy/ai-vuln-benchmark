"""Semantic checks for initializer and reentrancy modifiers.

A modifier name is not evidence. The body has to show the check, the state
write, and, for a reentrancy lock, the clear after the placeholder.
"""

from __future__ import annotations

import re

from app.parsing.solidity_cfg import (
    FunctionCfg,
    _consume_parens,
    _skip_string_or_comment,
    build_function_cfg,
)

_INIT_FLAG = r"(?:_?initialized|version)"
_LOCK_NAME = (
    r"(?:locked|_locked|_status|status|_entered|entered|"
    r"_reentrancyStatus|_reentrancyGuardEntered)"
)


def initializer_is_protected(body: str, helpers: dict[str, str] | None = None) -> bool:
    """True when prior init state is checked and then written before ``_;``.

    ``initialized = true; _;`` and ``version = 1; _;`` are not protection.
    A versioned guard counts only when a comparison establishes monotonicity.
    """
    if _direct_initializer(body):
        return True
    if not helpers or not re.search(r"_\s*;", body):
        return False
    before = body.split("_;", 1)[0]
    for name, helper in helpers.items():
        if not re.search(rf"\b{re.escape(name)}\s*\(", before):
            continue
        if _direct_initializer(_helper_as_modifier(helper)):
            return True
    return False


def reentrancy_guard_holds(body: str) -> bool:
    """True for check → entered → body → clear.

    ``locked = true; _;``, an owner check, and ``locked = computeLock(); _;``
    do not establish that sequence. Unknown assignments are not treated as safe.
    """
    if not re.search(r"_\s*;", body):
        return False
    cfg_text = _as_function(_replace_placeholder(body, "BODY_PLACEHOLDER = 1;"))
    cfg = build_function_cfg(cfg_text)
    if not cfg.known:
        return False
    placeholders = [node for node in cfg.nodes if "BODY_PLACEHOLDER" in node.text]
    if len(placeholders) != 1:
        return False
    placeholder = placeholders[0].node_id
    for variable in _lock_variables(body):
        if not _lock_check_dominates(cfg, variable, placeholder):
            continue
        if not _lock_set_dominates(cfg, variable, placeholder):
            continue
        if _lock_cleared_after(cfg, variable, placeholder):
            return True
    return False


def _direct_initializer(body: str) -> bool:
    if not re.search(r"_\s*;", body):
        return False
    cfg = build_function_cfg(_as_function(_replace_placeholder(body, "INIT_PLACEHOLDER = 1;")))
    if not cfg.known:
        return False
    placeholders = [node.node_id for node in cfg.nodes if "INIT_PLACEHOLDER" in node.text]
    checks = [
        (node.node_id, variable, kind, bound)
        for node in cfg.nodes
        for variable, kind, bound in _init_checks(node.text)
    ]
    writes = [
        (node.node_id, variable, rhs)
        for node in cfg.nodes
        for variable, rhs in _init_assignments(node.text)
    ]
    if not placeholders or not checks or not writes:
        return False
    for placeholder in placeholders:
        if not any(
            cfg.dominates(check_id, placeholder)
            and cfg.dominates(write_id, placeholder)
            and cfg.dominates(check_id, write_id)
            and variable == written
            and _init_write_matches(kind, bound, rhs)
            for check_id, variable, kind, bound in checks
            for write_id, written, rhs in writes
        ):
            return False
    return True


def _helper_as_modifier(helper: str) -> str:
    if re.search(r"_\s*;", helper):
        return helper
    start = helper.find("{")
    end = helper.rfind("}")
    if start < 0 or end <= start:
        return helper + "\n_;"
    return "{\n" + helper[start + 1 : end] + "\n_;\n}"


def _as_function(body: str) -> str:
    inner = body.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    return "function _mod() {\n" + inner + "\n}"


def _replace_placeholder(body: str, replacement: str) -> str:
    return re.sub(r"_\s*;", replacement, body, count=1)


def _strip(text: str) -> str:
    chars: list[str] = []
    index = 0
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            chars.append(" " * (skipped - index))
            index = skipped
            continue
        chars.append(text[index])
        index += 1
    return "".join(chars)


def _init_checks(text: str) -> list[tuple[str, str, str]]:
    """Prior-state checks that can actually stop a repeat.

    A comparison that merely mentions ``version`` or ``<`` is not enough.
    ``||`` bypasses the check, so the whole condition is rejected.
    """
    if "INIT_PLACEHOLDER" in text or "BODY_PLACEHOLDER" in text:
        return []
    compact = re.sub(r"\s+", " ", _strip(text))
    condition = _paren_condition(compact)
    if condition is None or "||" in condition:
        return []
    found: list[tuple[str, str, str]] = []
    for match in re.finditer(rf"!\s*({_INIT_FLAG})\b", condition):
        found.append((match.group(1), "zero", ""))
    for match in re.finditer(rf"\b({_INIT_FLAG})\b\s*==\s*(?:false|0)\b", condition):
        found.append((match.group(1), "zero", ""))
    if re.search(r"\brevert\b", compact):
        bare = re.fullmatch(rf"\s*({_INIT_FLAG})\s*", condition)
        if bare:
            found.append((bare.group(1), "zero", ""))
        for match in re.finditer(
            rf"\b({_INIT_FLAG})\b\s*(?:!=\s*(?:0|false)\b|==\s*true\b)",
            condition,
        ):
            found.append((match.group(1), "zero", ""))
        for match in re.finditer(rf"\b({_INIT_FLAG})\b\s*>=\s*([A-Za-z_]\w*)\b", condition):
            bound = match.group(2)
            if bound not in {match.group(1), "true", "false"}:
                found.append((match.group(1), "below", bound))
    for match in re.finditer(rf"\b({_INIT_FLAG})\b\s*<\s*([A-Za-z_]\w*)\b", condition):
        bound = match.group(2)
        if bound != match.group(1):
            found.append((match.group(1), "below", bound))
    return found


def _init_assignments(text: str) -> list[tuple[str, str]]:
    if "INIT_PLACEHOLDER" in text or "BODY_PLACEHOLDER" in text:
        return []
    compact = re.sub(r"\s+", " ", _strip(text))
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(rf"\b({_INIT_FLAG})\b\s*=(?!=)\s*([^;]+)", compact)
    ]


def _init_write_matches(kind: str, bound: str, rhs: str) -> bool:
    value = rhs.strip()
    if "(" in value or not re.fullmatch(r"true|false|\d+|[A-Za-z_]\w*", value):
        return False
    if kind == "below":
        return value == bound
    if kind == "zero":
        return value not in {"false", "0"}
    return False


def _paren_condition(compact: str) -> str | None:
    match = re.search(r"\b(?:require|if)\s*\(", compact)
    if match is None:
        return None
    open_at = compact.find("(", match.start())
    try:
        end = _consume_parens(compact, open_at)
    except ValueError:
        return None
    return compact[open_at + 1 : end - 1]


def _lock_variables(body: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(rf"\b({_LOCK_NAME})\b", body):
        name = match.group(1)
        if name not in found:
            found.append(name)
    return found


def _lock_check_dominates(cfg: FunctionCfg, variable: str, placeholder: int) -> bool:
    for node in cfg.nodes:
        if not _is_lock_check(node.text, variable):
            continue
        if cfg.dominates(node.node_id, placeholder):
            return True
    return False


def _lock_set_dominates(cfg: FunctionCfg, variable: str, placeholder: int) -> bool:
    for node in cfg.nodes:
        if node.kind != "stmt" or not _is_entered_assignment(node.text, variable):
            continue
        if not cfg.dominates(node.node_id, placeholder):
            continue
        if any(
            _is_lock_check(check.text, variable) and cfg.dominates(check.node_id, node.node_id)
            for check in cfg.nodes
        ):
            return True
    return False


def _lock_cleared_after(cfg: FunctionCfg, variable: str, placeholder: int) -> bool:
    """True when every path from the body to a normal exit clears ``variable``.

    A clear that is only reachable on some branch does not hold. Reaching the
    exit through ``revert`` without a clear is also not a guard: the lock would
    stay set only when that branch reverts, and the other branch can re-enter.
    """
    clears = {
        node.node_id
        for node in cfg.nodes
        if node.kind == "stmt"
        and node.node_id != placeholder
        and _is_clear_assignment(node.text, variable)
    }
    if not any(node_id in cfg.reachable_from(placeholder) for node_id in clears):
        return False
    seen: set[int] = set()
    stack = [placeholder]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in clears:
            continue
        node = cfg.nodes[current]
        if node.kind == "exit" and current != placeholder:
            return False
        for nxt in cfg.successors(current):
            if nxt not in clears:
                stack.append(nxt)
    return True


def _is_lock_check(text: str, variable: str) -> bool:
    name = re.escape(variable)
    compact = re.sub(r"\s+", " ", _strip(text))
    condition = _paren_condition(compact)
    if condition is not None and "||" in condition:
        return False
    examined = condition if condition is not None else compact
    if re.search(rf"!\s*{name}\b", examined) and re.search(r"\brequire\b|\brevert\b", compact):
        return True
    if re.search(
        rf"\b{name}\b\s*(?:==\s*false|!=\s*_?ENTERED|==\s*_?NOT_ENTERED|==\s*1\b|!=\s*2\b)",
        examined,
    ) and re.search(r"\brequire\b|\brevert\b", compact):
        return True
    if re.fullmatch(rf"\s*{name}\s*", examined) and "revert" in compact:
        return True
    return bool(
        re.search(rf"\b{name}\b\s*(?:==\s*_?ENTERED|!=\s*_?NOT_ENTERED)", examined)
        and "revert" in compact
    )


def _is_entered_assignment(text: str, variable: str) -> bool:
    match = re.search(rf"\b{re.escape(variable)}\b\s*=\s*([^;]+)", _strip(text))
    if match is None:
        return False
    rhs = match.group(1).strip()
    if "(" in rhs:
        return False
    return bool(re.fullmatch(r"true|2|_ENTERED|ENTERED|_entered", rhs))


def _is_clear_assignment(text: str, variable: str) -> bool:
    match = re.search(rf"\b{re.escape(variable)}\b\s*=\s*([^;]+)", _strip(text))
    if match is None:
        return False
    rhs = match.group(1).strip()
    if "(" in rhs:
        return False
    return bool(re.fullmatch(r"false|0|1|_NOT_ENTERED|NOT_ENTERED|_notEntered", rhs))

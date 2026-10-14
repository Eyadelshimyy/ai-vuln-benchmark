"""Conservative intra-procedural control-flow for one Solidity function.

This is not a compiler CFG. Brace structure that cannot be split stays unknown,
and security rules must not treat an unknown graph as proof that a path is safe.
A guard dominates a later operation only when every entry path to that operation
has already passed the guard. A check on one branch does not protect another.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

_CONTROL = ("if", "for", "while", "do", "unchecked", "try")
_TYPE_WORD = (
    r"(?:u?int\d*|address|bool|string|bytes\d*|mapping|function|struct|"
    r"[A-Za-z_]\w*)"
)


@dataclass
class CfgNode:
    node_id: int
    kind: str
    text: str


@dataclass
class FunctionCfg:
    nodes: list[CfgNode] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    entry: int = 0
    known: bool = True
    then_of: dict[int, set[int]] = field(default_factory=dict)
    else_of: dict[int, set[int]] = field(default_factory=dict)
    join_of: dict[int, int] = field(default_factory=dict)

    def successors(self, node_id: int) -> list[int]:
        return [dst for src, dst, _label in self.edges if src == node_id]

    def reachable_from(self, start: int) -> set[int]:
        seen: set[int] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen or current < 0 or current >= len(self.nodes):
                continue
            seen.add(current)
            stack.extend(self.successors(current))
        return seen

    def dominates(self, guard: int, target: int) -> bool:
        """Every entry-to-target path passes through ``guard``."""
        if not self.known:
            return False
        if guard == target:
            return True
        if target not in self.reachable_from(self.entry):
            return False
        seen: set[int] = set()
        stack = [self.entry]
        while stack:
            current = stack.pop()
            if current == guard or current in seen:
                continue
            if current == target:
                return False
            seen.add(current)
            for nxt in self.successors(current):
                if nxt != guard:
                    stack.append(nxt)
        return True


def extract_body(function_text: str) -> str | None:
    """Return the function body, or None when the outer braces do not match."""
    start = function_text.find("{")
    if start < 0:
        return ""
    try:
        _body, end = _matching_brace(function_text, start)
    except ValueError:
        return None
    return function_text[start + 1 : end - 1]


def build_function_cfg(function_text: str) -> FunctionCfg:
    body = extract_body(function_text)
    if body is None:
        failed = FunctionCfg(known=False)
        failed.entry = _add(failed, "unknown", function_text[:180])
        return failed
    if not body.strip():
        cfg = FunctionCfg()
        entry = _add(cfg, "entry", "")
        cfg.entry = entry
        return cfg
    cfg = FunctionCfg()
    try:
        statements = split_top_statements(body)
        entry = _add(cfg, "entry", "")
        cfg.entry = entry
        exit_id = _add(cfg, "exit", "")
        terminals = _link_sequence(cfg, statements, entry, exit_id)
        for terminal in terminals:
            if terminal != exit_id:
                cfg.edges.append((terminal, exit_id, "next"))
    except ValueError:
        failed = FunctionCfg(known=False)
        failed.entry = _add(failed, "unknown", body[:180])
        return failed
    return cfg


def auth_dominates_sensitive(function_text: str) -> bool:
    """True when an authorization check protects every sensitive operation.

    ``owner = msg.sender`` and a ternary that mentions ``msg.sender`` are not
    checks. A check that appears only after the operation, or only on another
    branch, does not protect it.
    """
    cfg = build_function_cfg(function_text)
    if not cfg.known:
        return False
    guards = _tight_nodes(cfg, _is_auth_guard)
    sensitive = _tight_nodes(cfg, _is_sensitive)
    if not guards or not sensitive:
        return False
    return all(_op_protected(cfg, op, guards) for op in sensitive)


def operation_guarded(function_text: str, operation: str) -> bool | None:
    """Whether ``operation`` is dominated by a resolved authorization check.

    None means the body could not be split, or the operation text was not found.
    Callers must not treat None as safe. This helper is path-wide. Prefer
    :func:`operation_guard_at` when several occurrences share one path.
    """
    cfg = build_function_cfg(function_text)
    if not cfg.known or not operation:
        return None
    ops = _tight_ids(cfg, lambda text: operation in text)
    if not ops:
        return None
    guards = _tight_nodes(cfg, _is_auth_guard)
    if not guards:
        return False
    return all(_op_protected(cfg, op, guards) for op in ops)


def node_ranges(
    function_text: str, cfg: FunctionCfg | None = None
) -> tuple[FunctionCfg, dict[int, tuple[int, int]]]:
    """Map CFG nodes to offsets in ``function_text``.

    A wrapping control node starts the next search inside its body so a nested
    statement keeps its own range. Duplicate statement text is found in order.
    Nodes that cannot be located are omitted. Callers must not guess a home.
    """
    graph = cfg if cfg is not None else build_function_cfg(function_text)
    ranges: dict[int, tuple[int, int]] = {}
    cursor = 0
    for node in graph.nodes:
        text = node.text.strip()
        if not text:
            continue
        at = function_text.find(text, cursor)
        if at < 0:
            continue
        ranges[int(node.node_id)] = (at, at + len(text))
        brace = text.find("{")
        if brace >= 0:
            cursor = at + brace + 1
        elif node.kind in {"if", "loop", "try"}:
            # The braceless body is a later node still inside this statement.
            cursor = at
        else:
            cursor = at + len(text)
    return graph, ranges


def node_at(ranges: dict[int, tuple[int, int]], offset: int) -> int | None:
    """Return the tightest node whose range contains ``offset``."""
    matches = [
        (node_id, end - start) for node_id, (start, end) in ranges.items() if start <= offset < end
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item[1], item[0]))
    return matches[0][0]


def operation_guard_at(function_text: str, local_offset: int) -> tuple[str, str]:
    """Guard status and predicate for the statement at ``local_offset``.

    Status is ``guarded``, ``unguarded``, or ``unknown``. Unknown means the
    CFG or the statement placement could not be established. It is not safe.
    """
    cfg, ranges = node_ranges(function_text)
    if not cfg.known:
        return "unknown", ""
    node_id = node_at(ranges, local_offset)
    if node_id is None:
        return "unknown", ""
    guards = _tight_nodes(cfg, _is_auth_guard)
    protecting = [guard for guard in guards if _guard_protects(cfg, guard, node_id)]
    if not protecting:
        return "unguarded", ""
    return "guarded", cfg.nodes[protecting[0]].text.strip()


def placeholder_is_guarded(body: str) -> bool:
    """True when checks in a modifier body dominate its ``_;`` placeholder."""
    if not re.search(r"_\s*;", body):
        return False
    inner = body.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    replaced = re.sub(r"_\s*;", "owner = next;", inner, count=1)
    return auth_dominates_sensitive("function _mod() {\n" + replaced + "\n}")


def write_reachable_after(function_text: str, earlier: str, later: str) -> bool | None:
    """Whether ``later`` can run after ``earlier`` on one path.

    None means the function body could not be split confidently.
    """
    cfg = build_function_cfg(function_text)
    if not cfg.known or not earlier or not later:
        return None
    starts = _tight_ids(cfg, lambda text: earlier in text)
    ends = _tight_ids(cfg, lambda text: later in text)
    if not starts or not ends:
        return None
    reachable: set[int] = set()
    for start in starts:
        reachable |= cfg.reachable_from(start)
    for end in ends:
        if end in reachable and end not in starts:
            return True
    return False


def split_loop(statement: str) -> tuple[str, str, str]:
    """Return ``(header, body, style)`` for a for, while, or do-while statement."""
    stripped = statement.strip()
    if re.match(r"do\b", stripped):
        brace = stripped.find("{")
        if brace < 0:
            raise ValueError("do without block")
        body, end = _matching_brace(stripped, brace)
        header = stripped[end:].strip() or "while"
        return header, body, "do"
    match = re.match(r"(for|while)\b", stripped)
    if not match:
        raise ValueError("not a loop")
    cursor = _consume_parens(stripped, match.end())
    header = stripped[:cursor].strip()
    rest = stripped[cursor:].strip()
    if rest.startswith("{"):
        body, _end = _matching_brace(rest, 0)
        return header, body, match.group(1)
    return header, rest, match.group(1)


def _op_protected(cfg: FunctionCfg, op: int, guards: list[int]) -> bool:
    for guard in guards:
        if _guard_protects(cfg, guard, op):
            return True
    return False


def _guard_protects(cfg: FunctionCfg, guard: int, op: int) -> bool:
    node = cfg.nodes[guard]
    if not cfg.dominates(guard, op):
        return False
    if node.kind == "require":
        return True
    if node.kind != "if":
        return False
    polarity = _auth_polarity(_condition(node.text))
    if polarity is None:
        return False
    then_ids = cfg.then_of.get(guard, set())
    else_ids = cfg.else_of.get(guard, set())
    if polarity == "positive":
        if op in else_ids:
            return False
        if op in then_ids:
            return True
        return _else_terminates(cfg, guard)
    if op in then_ids:
        return False
    return _region_terminates(cfg, then_ids, cfg.join_of.get(guard))


def _region_terminates(cfg: FunctionCfg, region: set[int], join: int | None) -> bool:
    if not region or join is None:
        return False
    return not any(dst == join and src in region for src, dst, _label in cfg.edges)


def _else_terminates(cfg: FunctionCfg, guard: int) -> bool:
    """True when the unauthorized else branch cannot continue after the if."""
    else_ids = cfg.else_of.get(guard, set())
    join = cfg.join_of.get(guard)
    if join is None:
        return False
    if not else_ids:
        return not any(
            src == guard and dst == join and label == "else" for src, dst, label in cfg.edges
        )
    return _region_terminates(cfg, else_ids, join)


def _tight_nodes(cfg: FunctionCfg, predicate: Callable[[str], bool]) -> list[int]:
    matched = [
        node
        for node in cfg.nodes
        if node.kind not in {"entry", "exit", "join"} and predicate(node.text)
    ]
    return _drop_wrappers(matched)


def _tight_ids(cfg: FunctionCfg, predicate: Callable[[str], bool]) -> list[int]:
    matched = [
        node
        for node in cfg.nodes
        if node.kind not in {"entry", "exit", "join"} and predicate(node.text)
    ]
    return _drop_wrappers(matched)


def _drop_wrappers(nodes: list[CfgNode]) -> list[int]:
    kept: list[int] = []
    for node in nodes:
        wrapped = any(
            other is not node
            and other.text
            and other.text in node.text
            and len(other.text) < len(node.text)
            for other in nodes
        )
        if not wrapped:
            kept.append(node.node_id)
    return kept


def split_top_statements(body: str) -> list[str]:
    parts: list[str] = []
    index = 0
    length = len(body)
    while index < length:
        while index < length and body[index].isspace():
            index += 1
        if index >= length:
            break
        skipped = _skip_string_or_comment(body, index)
        if skipped is not None:
            index = skipped
            continue
        start = index
        if _keyword_at(body, index):
            index = _consume_control(body, index)
        else:
            index = _consume_simple(body, index)
        piece = body[start:index].strip()
        if piece:
            parts.append(piece)
    return parts


def nested_statements(text: str) -> list[tuple[int, str]]:
    """Leaf statements and control headers, with offsets into ``text``.

    An assignment inside a branch is its own statement. Classifying the whole
    ``if`` as one expression would mark that assignment as a read.
    """
    return _nested_at(text, 0)


def _nested_at(text: str, base: int) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    cursor = 0
    for statement in split_top_statements(text):
        at = text.find(statement, cursor)
        if at < 0:
            continue
        cursor = at + len(statement)
        found.extend(_expand_statement(statement, base + at))
    return found


def _expand_statement(statement: str, abs_at: int) -> list[tuple[int, str]]:
    kind = _statement_kind(statement)
    if kind not in {"if", "loop", "unchecked", "try"}:
        return [(abs_at, statement)]
    if "{" not in statement:
        return _expand_braceless(statement, abs_at, kind)
    found: list[tuple[int, str]] = []
    for part_kind, offset, fragment in _split_top_braces(statement):
        if part_kind == "body":
            found.extend(_nested_at(fragment, abs_at + offset))
            continue
        stripped = fragment.strip()
        if not stripped:
            continue
        found.append((abs_at + offset + fragment.find(stripped), stripped))
    return found


def _expand_braceless(statement: str, abs_at: int, kind: str) -> list[tuple[int, str]]:
    if kind == "loop":
        try:
            header, body, _style = split_loop(statement)
        except ValueError:
            return [(abs_at, statement)]
        found: list[tuple[int, str]] = []
        if header:
            found.append((abs_at + max(statement.find(header), 0), header))
        if body.strip():
            body_at = statement.find(body)
            if body_at >= 0:
                found.extend(_nested_at(body, abs_at + body_at))
        return found or [(abs_at, statement)]
    if kind != "if":
        return [(abs_at, statement)]
    match = re.match(r"if\b", statement)
    if match is None:
        return [(abs_at, statement)]
    try:
        end_paren = _consume_parens(statement, match.end())
    except ValueError:
        return [(abs_at, statement)]
    header = statement[:end_paren].strip()
    found = [(abs_at, header)] if header else []
    then_body, else_body = _if_branches(statement)
    search = end_paren
    for body in (then_body, else_body or ""):
        if not body or not body.strip() or body.strip() == statement:
            continue
        body_at = statement.find(body, search)
        if body_at < 0:
            body_at = statement.find(body)
        if body_at < 0:
            continue
        found.extend(_expand_statement(body.strip(), abs_at + body_at))
        search = body_at + len(body)
    return found


def _split_top_braces(statement: str) -> list[tuple[str, int, str]]:
    parts: list[tuple[str, int, str]] = []
    depth = 0
    start = 0
    body_start = 0
    index = 0
    while index < len(statement):
        skipped = _skip_string_or_comment(statement, index)
        if skipped is not None:
            index = skipped
            continue
        char = statement[index]
        if char == "{":
            if depth == 0:
                if index > start:
                    parts.append(("code", start, statement[start:index]))
                body_start = index + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                parts.append(("body", body_start, statement[body_start:index]))
                start = index + 1
        index += 1
    if depth == 0 and start < len(statement):
        parts.append(("code", start, statement[start:]))
    return parts


def _link_sequence(
    cfg: FunctionCfg,
    statements: list[str],
    entry: int,
    exit_id: int,
    *,
    loop_header: int | None = None,
    loop_exit: int | None = None,
    entry_label: str = "next",
) -> list[int]:
    previous = [entry]
    label = entry_label
    for statement in statements:
        kind = _statement_kind(statement)
        if kind == "loop":
            previous = _link_loop(cfg, statement, previous, exit_id, loop_header, loop_exit, label)
            label = "next"
            continue
        if kind == "unchecked":
            previous = _link_unchecked(
                cfg, statement, previous, exit_id, loop_header, loop_exit, label
            )
            label = "next"
            continue
        if kind == "try":
            previous = _link_try(cfg, statement, previous, exit_id, loop_header, loop_exit, label)
            label = "next"
            continue
        node = _add(cfg, kind, statement.strip())
        _connect(cfg, previous, node, label, entry)
        label = "next"
        if kind in {"return", "revert"}:
            cfg.edges.append((node, exit_id, "exit"))
            previous = []
            continue
        if kind == "break":
            if loop_exit is None:
                cfg.known = False
                cfg.edges.append((node, exit_id, "exit"))
            else:
                cfg.edges.append((node, loop_exit, "break"))
            previous = []
            continue
        if kind == "continue":
            if loop_header is None:
                cfg.known = False
            else:
                cfg.edges.append((node, loop_header, "continue"))
            previous = []
            continue
        if kind == "require":
            cfg.edges.append((node, exit_id, "fail"))
            previous = [node]
            continue
        if kind == "if":
            previous = _link_if(cfg, statement, node, exit_id, loop_header, loop_exit)
            continue
        previous = [node]
    return previous


def _connect(cfg: FunctionCfg, sources: list[int], dest: int, label: str, entry: int) -> None:
    for src in sources:
        cfg.edges.append((src, dest, label if src == entry else "next"))


def _link_if(
    cfg: FunctionCfg,
    statement: str,
    node: int,
    exit_id: int,
    loop_header: int | None,
    loop_exit: int | None,
) -> list[int]:
    then_body, else_body = _if_branches(statement)
    join = _add(cfg, "join", "")
    then_start = len(cfg.nodes)
    then_end = _link_sequence(
        cfg,
        split_top_statements(then_body),
        node,
        exit_id,
        loop_header=loop_header,
        loop_exit=loop_exit,
        entry_label="then",
    )
    then_ids = set(range(then_start, len(cfg.nodes)))
    for src in then_end:
        cfg.edges.append((src, join, "then"))
    if else_body is None:
        cfg.edges.append((node, join, "else"))
        else_ids: set[int] = set()
    else:
        else_start = len(cfg.nodes)
        else_end = _link_sequence(
            cfg,
            split_top_statements(else_body),
            node,
            exit_id,
            loop_header=loop_header,
            loop_exit=loop_exit,
            entry_label="else",
        )
        else_ids = set(range(else_start, len(cfg.nodes)))
        for src in else_end:
            cfg.edges.append((src, join, "else"))
    cfg.then_of[node] = then_ids
    cfg.else_of[node] = else_ids
    cfg.join_of[node] = join
    return [join]


def _link_loop(
    cfg: FunctionCfg,
    statement: str,
    previous: list[int],
    exit_id: int,
    outer_header: int | None,
    outer_exit: int | None,
    label: str,
) -> list[int]:
    del outer_header, outer_exit
    header_text, body, style = split_loop(statement)
    after = _add(cfg, "join", "")
    if style == "do":
        gate = _add(cfg, "join", "")
        header = _add(cfg, "loop", header_text)
        _connect(cfg, previous, gate, label, previous[0] if previous else -1)
        terminals = _link_sequence(
            cfg,
            split_top_statements(body),
            gate,
            exit_id,
            loop_header=header,
            loop_exit=after,
        )
        for src in terminals:
            cfg.edges.append((src, header, "next"))
        succs = cfg.successors(gate)
        back_target = succs[0] if succs else gate
        cfg.edges.append((header, back_target, "back"))
        cfg.edges.append((header, after, "exit"))
        return [after]
    header = _add(cfg, "loop", header_text)
    _connect(cfg, previous, header, label, previous[0] if previous else -1)
    cfg.edges.append((header, after, "exit"))
    terminals = _link_sequence(
        cfg,
        split_top_statements(body),
        header,
        exit_id,
        loop_header=header,
        loop_exit=after,
    )
    for src in terminals:
        cfg.edges.append((src, header, "back"))
    return [after]


def _link_unchecked(
    cfg: FunctionCfg,
    statement: str,
    previous: list[int],
    exit_id: int,
    loop_header: int | None,
    loop_exit: int | None,
    label: str,
) -> list[int]:
    node = _add(cfg, "unchecked", "unchecked")
    _connect(cfg, previous, node, label, previous[0] if previous else -1)
    brace = statement.find("{")
    if brace < 0:
        return [node]
    body, _end = _matching_brace(statement, brace)
    return _link_sequence(
        cfg,
        split_top_statements(body),
        node,
        exit_id,
        loop_header=loop_header,
        loop_exit=loop_exit,
    )


def _link_try(
    cfg: FunctionCfg,
    statement: str,
    previous: list[int],
    exit_id: int,
    loop_header: int | None,
    loop_exit: int | None,
    label: str,
) -> list[int]:
    node = _add(cfg, "try", "try")
    _connect(cfg, previous, node, label, previous[0] if previous else -1)
    join = _add(cfg, "join", "")
    bodies = _brace_bodies(statement)
    if len(bodies) < 2:
        cfg.known = False
    if not bodies:
        cfg.edges.append((node, join, "next"))
        return [join]
    for body in bodies:
        terminals = _link_sequence(
            cfg,
            split_top_statements(body),
            node,
            exit_id,
            loop_header=loop_header,
            loop_exit=loop_exit,
        )
        for src in terminals:
            cfg.edges.append((src, join, "next"))
    return [join]


def _brace_bodies(statement: str) -> list[str]:
    bodies: list[str] = []
    index = 0
    while index < len(statement):
        skipped = _skip_string_or_comment(statement, index)
        if skipped is not None:
            index = skipped
            continue
        if statement[index] == "{":
            body, end = _matching_brace(statement, index)
            bodies.append(body)
            index = end
            continue
        index += 1
    return bodies


def _if_branches(statement: str) -> tuple[str, str | None]:
    stripped = statement.strip()
    match = re.match(r"if\b", stripped)
    if not match:
        return "", None
    try:
        end_paren = _consume_parens(stripped, match.end())
    except ValueError:
        return "", None
    rest = stripped[end_paren:].strip()
    if rest.startswith("{"):
        then_body, then_end = _matching_brace(
            stripped, end_paren + (len(stripped[end_paren:]) - len(rest))
        )
        tail = stripped[then_end:].strip()
    else:
        then_body, tail = _split_braceless(rest)
    if tail.startswith("else") and _boundary(tail, 4):
        else_rest = tail[4:].strip()
        if else_rest.startswith("{"):
            else_body, _end = _matching_brace(else_rest, 0)
            return then_body, else_body
        return then_body, else_rest
    return then_body, None


def _split_braceless(rest: str) -> tuple[str, str]:
    if not rest:
        return "", ""
    if _keyword_at(rest, 0):
        end = _consume_control(rest, 0)
    else:
        end = _consume_simple(rest, 0)
    return rest[:end].strip(), rest[end:].strip()


def _matching_brace(text: str, open_at: int) -> tuple[str, int]:
    end = _consume_delimited(text, open_at, "{", "}")
    return text[open_at + 1 : end - 1], end


def _add(cfg: FunctionCfg, kind: str, text: str) -> int:
    node_id = len(cfg.nodes)
    cfg.nodes.append(CfgNode(node_id, kind, text[:4000]))
    return node_id


def _keyword_at(text: str, index: int) -> bool:
    for word in _CONTROL:
        if text.startswith(word, index) and _boundary(text, index + len(word)):
            return True
    return False


def _boundary(text: str, index: int) -> bool:
    return index >= len(text) or not (text[index].isalnum() or text[index] == "_")


def _consume_simple(text: str, index: int) -> int:
    depth = 0
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        char = text[index]
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        elif char == ";" and depth == 0:
            return index + 1
        index += 1
    return index


def _consume_control(text: str, index: int) -> int:
    keyword = _word(text, index)
    cursor = index + len(keyword)
    if keyword in {"if", "for", "while"}:
        cursor = _consume_parens(text, cursor)
    if keyword == "do":
        cursor = _consume_block_or_stmt(text, cursor)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if text.startswith("while", cursor):
            cursor = _consume_parens(text, cursor + len("while"))
            if cursor < len(text) and text[cursor] == ";":
                cursor += 1
        return cursor
    cursor = _consume_block_or_stmt(text, cursor)
    if keyword == "if":
        probe = cursor
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if text.startswith("else", probe) and _boundary(text, probe + 4):
            cursor = probe + 4
            cursor = _consume_block_or_stmt(text, cursor)
    if keyword == "try":
        while True:
            probe = cursor
            while probe < len(text) and text[probe].isspace():
                probe += 1
            if text.startswith("catch", probe) and _boundary(text, probe + 5):
                cursor = _consume_block_or_stmt(text, probe + 5)
                continue
            break
    return cursor


def _consume_parens(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "(":
        return index
    return _consume_delimited(text, index, "(", ")")


def _consume_delimited(text: str, index: int, open_ch: str, close_ch: str) -> int:
    if index >= len(text) or text[index] != open_ch:
        raise ValueError("missing delimiter")
    depth = 0
    while index < len(text):
        skipped = _skip_string_or_comment(text, index)
        if skipped is not None:
            index = skipped
            continue
        if text[index] == open_ch:
            depth += 1
        elif text[index] == close_ch:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError("unbalanced delimiter")


def _consume_block_or_stmt(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "{":
        return _consume_delimited(text, index, "{", "}")
    if index < len(text) and _keyword_at(text, index):
        return _consume_control(text, index)
    return _consume_simple(text, index)


def _word(text: str, index: int) -> str:
    end = index
    while end < len(text) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return text[index:end]


def _statement_kind(statement: str) -> str:
    stripped = statement.lstrip()
    if re.match(r"if\b", stripped):
        return "if"
    if re.match(r"(for|while|do)\b", stripped):
        return "loop"
    if re.match(r"unchecked\b", stripped):
        return "unchecked"
    if re.match(r"try\b", stripped):
        return "try"
    if re.match(r"return\b", stripped):
        return "return"
    if re.match(r"revert\b", stripped):
        return "revert"
    if re.match(r"break\b", stripped):
        return "break"
    if re.match(r"continue\b", stripped):
        return "continue"
    if re.match(r"require\s*\(", stripped) or re.match(r"assert\s*\(", stripped):
        return "require"
    return "stmt"


def _condition(statement: str) -> str:
    stripped = statement.strip()
    if re.match(r"(require|assert)\b", stripped):
        return stripped
    open_at = stripped.find("(")
    if open_at < 0:
        return stripped
    try:
        end = _consume_parens(stripped, open_at)
    except ValueError:
        return stripped
    return stripped[open_at:end]


# ``_msgSender()`` is OpenZeppelin's Context accessor for the caller (Ownable v4/v5).
_CALLER = r"(?:msg\.sender|\b_msgSender\s*\(\s*\))"


def _auth_polarity(text: str) -> str | None:
    if re.search(rf"{_CALLER}\s*!=|!=\s*{_CALLER}|!\s*hasRole\s*\(", text):
        return "negative"
    if re.search(rf"{_CALLER}\s*==|==\s*{_CALLER}|\bhasRole\s*\(", text):
        return "positive"
    return None


def _is_auth_guard(text: str) -> bool:
    if not text:
        return False
    head = text.lstrip()
    if re.match(r"(require|assert)\s*\(", head):
        return _auth_polarity(head) is not None
    if re.match(r"if\b", head):
        return _auth_polarity(_condition(head)) is not None
    return False


def _is_sensitive(text: str) -> bool:
    if not text or _is_auth_guard(text):
        return False
    if re.search(r"\.(call|delegatecall|staticcall|send|transfer|safeTransferFrom)\s*[\(\{]", text):
        return True
    if re.match(rf"{_TYPE_WORD}\b", text.lstrip()) and re.search(
        r"\b(memory|calldata|storage)\b", text
    ):
        return False
    if re.match(
        r"(uint\d*|int\d*|address|bool|string|bytes\d*|mapping)\b",
        text.lstrip(),
    ):
        return False
    return bool(re.search(r"\b[A-Za-z_]\w*\b\s*(\[[^\]]+\])?\s*=[^=]", text))


def _skip_string_or_comment(text: str, index: int) -> int | None:
    if text.startswith("//", index):
        newline = text.find("\n", index)
        return len(text) if newline < 0 else newline + 1
    if text.startswith("/*", index):
        end = text.find("*/", index + 2)
        return len(text) if end < 0 else end + 2
    if index >= len(text) or text[index] not in "\"'":
        return None
    quote = text[index]
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == quote:
            return cursor + 1
        cursor += 1
    return len(text)

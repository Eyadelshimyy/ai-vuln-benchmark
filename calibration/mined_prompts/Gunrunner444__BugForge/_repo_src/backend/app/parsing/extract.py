"""Profile-driven extraction of imports, entities, calls, and bindings.

The scanner is language-neutral: profiles supply declaration regexes and
comment markers. It is still a conservative approximation, not a full AST.

Known limitations (kept explicit rather than hidden):
- No preprocessor / macro expansion.
- Nested template/generic syntax may attach extra tokens to a name.
- Regex literals that look like calls after a division operator can still
  be missed; slash-started identifiers are not treated as names.
- Cross-function taint is not reconstructed here (see security.taint).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.domain.language import ParserTier
from app.domain.source import ParsedEntity, ParsedImport
from app.parsing.comments import strip_comments
from app.parsing.model import Binding, CallSite, LanguageProfile, ParserDiagnostics, SyntaxGraph

_IDENT = re.compile(r"[A-Za-z_$\\][\w$\\]*")
_DYNAMIC_MARKERS = ("+", "${", 'f"', "f'", "%s", "%d", "{}", ".format(", "`", "#{")
_DECL_KEYWORDS = frozenset(
    {
        "function",
        "func",
        "fn",
        "fun",
        "def",
        "proc",
        "sub",
        "macro",
        "classmethod",
        "staticmethod",
    }
)
_NON_CALL_KEYWORDS = frozenset(
    {
        "if",
        "elif",
        "else",
        "for",
        "foreach",
        "while",
        "switch",
        "case",
        "catch",
        "except",
        "with",
        "return",
        "throw",
        "typeof",
        "instanceof",
        "void",
        "sizeof",
        "function",
        "func",
        "fn",
        "fun",
        "def",
        "class",
        "struct",
        "enum",
        "interface",
        "impl",
        "match",
        "select",
        "when",
        "unless",
        "until",
        "loop",
        "try",
        "finally",
        "synchronized",
        "lock",
        "using",
        "unsafe",
        "await",
        "yield",
        "delete",
        "package",
        "let",
        "const",
        "var",
        "val",
        "pub",
        "public",
        "private",
        "protected",
        "static",
        "async",
        "export",
        "from",
        "type",
        "namespace",
        "module",
        "object",
        "actor",
        "protocol",
        "extension",
        "where",
        "in",
        "of",
        "as",
        "is",
        "not",
        "and",
        "or",
        "do",
        "then",
        "end",
        "begin",
        "rescue",
        "ensure",
        "elif",
        "elseif",
        "fi",
        "done",
        "goto",
        "break",
        "continue",
        "pass",
        "lambda",
        "new",
        "delete",
        "sizeof",
        "alignof",
        "decltype",
        "static_cast",
        "dynamic_cast",
        "reinterpret_cast",
        "const_cast",
        "template",
        "typename",
        "requires",
        "concept",
        "co_await",
        "co_yield",
        "co_return",
    }
)
_TYPE_KEYWORDS = (
    ("interface", "interface"),
    ("protocol", "protocol"),
    ("namespace", "namespace"),
    ("extension", "extension"),
    ("module", "module"),
    ("object", "object"),
    ("actor", "actor"),
    ("enum", "enum"),
    ("struct", "struct"),
    ("trait", "trait"),
    ("impl", "impl"),
    ("type", "type"),
    ("class", "class"),
)


def parse_with_profile(profile: LanguageProfile, file_path: Path, source: str) -> SyntaxGraph:
    cleaned = strip_comments(
        source,
        line_comment=profile.line_comment,
        block_comment=profile.block_comment,
    )
    lines = tuple(cleaned.splitlines() or [""])
    logical = _logical_lines(lines)
    imports = _imports(profile, logical)
    entities = _entities(profile, logical)
    bindings = _bindings(profile, logical)
    calls = _calls(cleaned, bare_keywords=profile.bare_call_keywords)
    bindings.extend(_decorator_param_bindings(cleaned))
    bindings = [_with_literal_flags(b) for b in bindings]
    return SyntaxGraph(
        language=profile.language_id,
        file_path=str(file_path),
        source=source,
        lines=lines,
        imports=tuple(imports),
        entities=tuple(entities),
        calls=tuple(calls),
        bindings=tuple(bindings),
        parser_backend="profile",
        parser_tier=ParserTier.PROFILE_FALLBACK,
        diagnostics=ParserDiagnostics(
            message="profile fallback parser",
            native_available=False,
            status="profile_fallback_used",
            fallback_reason="profile fallback used",
        ),
    )


def _looks_quoted(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 2 and stripped[0] in {'"', "'", "`"} and stripped[-1] == stripped[0]


def _with_literal_flags(binding: Binding) -> Binding:
    if not _looks_quoted(binding.rhs):
        return binding
    return Binding(
        name=binding.name,
        line=binding.line,
        rhs=binding.rhs,
        scope_id=binding.scope_id,
        kind=binding.kind,
        span=binding.span,
        node_id=binding.node_id,
        rhs_is_literal=True,
        rhs_callees=(),
        rhs_accesses=(),
        rhs_idents=(),
    )


def _decorator_param_bindings(source: str) -> list[Binding]:
    """Bind identifiers that follow `@Decorator(...)` (NestJS/Java annotations)."""
    found: list[Binding] = []
    i = 0
    n = len(source)
    line = 1
    in_str: str | None = None
    escape = False
    while i < n:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            if ch == "\n":
                line += 1
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
            i += 1
            continue
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch == "@":
            match = _IDENT.match(source, i + 1)
            if not match:
                i += 1
                continue
            deco = match.group(0)
            j = _skip_ws(source, match.end())
            args = deco
            if j < n and source[j] == "(":
                text, end = _balanced_parens(source, j)
                args = f"{deco}({text})"
                j = end
            j = _skip_ws(source, j)
            while j < n and source[j] in "@":  # stacked decorators
                break
            nxt = _IDENT.match(source, j)
            if nxt and nxt.group(0).lower() not in _NON_CALL_KEYWORDS:
                found.append(Binding(name=nxt.group(0), line=line, rhs=f"@{args}"))
            i = match.end()
            continue
        i += 1
    return found


def _logical_lines(lines: tuple[str, ...]) -> list[tuple[int, str]]:
    """Join continued statements so multiline imports/decls are visible."""
    spans: list[tuple[int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        start = i + 1
        text = lines[i]
        i += 1
        while i < n and _should_join(text, lines[i]):
            text = f"{text}\n{lines[i]}"
            i += 1
        spans.append((start, text))
    return spans


def _should_join(current: str, nxt: str) -> bool:
    stripped_next = nxt.strip()
    if not stripped_next:
        return _unmatched(current) > 0
    if _unmatched(current) > 0:
        return True
    trimmed = current.rstrip()
    if trimmed.endswith((".", "?.", "::", "->", "=>", "=", ",", "\\", "+", "(", "{", "[")):
        return True
    if stripped_next.startswith(("{", ".", "?.", "->", "::")):
        return True
    return False


def _unmatched(text: str) -> int:
    depth = 0
    in_str: str | None = None
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
    return depth


def _imports(profile: LanguageProfile, spans: list[tuple[int, str]]) -> list[ParsedImport]:
    found: list[ParsedImport] = []
    compiled = [re.compile(p, re.MULTILINE) for p in profile.import_patterns]
    seen: set[tuple[int, str]] = set()
    for line_no, text in spans:
        for rx in compiled:
            for match in rx.finditer(text):
                module = match.group(1)
                key = (line_no, module)
                if key in seen:
                    continue
                seen.add(key)
                rel_line = line_no + text[: match.start()].count("\n")
                found.append(
                    ParsedImport(
                        module=module,
                        line_number=rel_line,
                        is_from_import="from" in text or ("import " in text and "{" in text),
                        import_type="unknown",
                    )
                )
    return found


def _entities(profile: LanguageProfile, spans: list[tuple[int, str]]) -> list[ParsedEntity]:
    found: list[ParsedEntity] = []
    func_rx = [re.compile(p, re.MULTILINE) for p in profile.function_patterns]
    class_rx = [re.compile(p, re.MULTILINE) for p in profile.class_patterns]
    seen: set[tuple[str, int, str]] = set()
    for line_no, text in spans:
        for rx in class_rx:
            for match in rx.finditer(text):
                name = match.group(1)
                if not name or name.lower() in _NON_CALL_KEYWORDS:
                    continue
                start = line_no + text[: match.start()].count("\n")
                key = ("type", start, name)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    ParsedEntity(
                        entity_type=_entity_kind(text, match.start()),
                        name=name,
                        qualified_name=name,
                        start_line=start,
                        end_line=start,
                    )
                )
        for rx in func_rx:
            for match in rx.finditer(text):
                name = match.group(1)
                if not name or name.lower() in _NON_CALL_KEYWORDS:
                    continue
                rest = text[match.end() :]
                if rest.lstrip().startswith(";"):
                    continue
                start = line_no + text[: match.start()].count("\n")
                key = ("fn", start, name)
                if key in seen:
                    continue
                seen.add(key)
                async_prefix = bool(
                    re.search(r"\basync\b", text[max(0, match.start() - 12) : match.start()])
                )
                found.append(
                    ParsedEntity(
                        entity_type="async_function" if async_prefix else "function",
                        name=name,
                        qualified_name=name,
                        start_line=start,
                        end_line=start,
                    )
                )
    return found


def _entity_kind(text: str, at: int) -> str:
    window = text[max(0, at - 24) : at + 24].lower()
    for keyword, kind in _TYPE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", window):
            return kind
    return "class"


def _bindings(profile: LanguageProfile, spans: list[tuple[int, str]]) -> list[Binding]:
    found: list[Binding] = []
    compiled = [re.compile(p, re.MULTILINE) for p in profile.assignment_patterns]
    for line_no, text in spans:
        for raw_line in text.splitlines() or [text]:
            for statement in _split_top_level(raw_line, ";"):
                stripped = statement.strip()
                if not stripped:
                    continue
                if stripped.startswith(("if ", "for ", "while ", "return ", "elif ", "elseif ")):
                    continue
                for rx in compiled:
                    match = rx.search(stripped)
                    if not match or not match.lastindex or match.lastindex < 2:
                        continue
                    lhs, rhs = match.group(1), match.group(2).strip()
                    if not lhs or not rhs:
                        continue
                    eq_at = match.start(2) - 1
                    if not _is_true_assignment(stripped, eq_at):
                        continue
                    found.extend(_expand_binding(lhs, rhs.rstrip(";"), line_no))
                    break
    return found


def _is_true_assignment(text: str, eq_index: int) -> bool:
    if eq_index < 0 or eq_index >= len(text) or text[eq_index] != "=":
        # Patterns may capture rhs after `=`; search backwards from rhs.
        return True
    prev = text[eq_index - 1] if eq_index else ""
    nxt = text[eq_index + 1] if eq_index + 1 < len(text) else ""
    if prev in "=!<>:" and prev:
        if prev == ":" and nxt != "=":
            return True  # Go `:=` / typed default handled by the profile
        if prev in "=!<>":
            return False
    if nxt in "=>":
        return False
    return True


def _expand_binding(lhs: str, rhs: str, line: int) -> list[Binding]:
    name = lhs.strip()
    if name.startswith("{") and name.endswith("}"):
        return _destructure_object(name[1:-1], rhs, line)
    if name.startswith("[") and name.endswith("]"):
        return _destructure_array(name[1:-1], rhs, line)
    ident = name.lstrip("$")
    if not ident or not _IDENT.fullmatch(ident.split(":", 1)[0].strip()):
        # `const x: Type =` — keep the identifier before a type annotation.
        ident = ident.split(":", 1)[0].strip().lstrip("$")
        if not ident or not _IDENT.fullmatch(ident):
            return []
    return [Binding(name=ident, line=line, rhs=rhs)]


def _destructure_object(inner: str, rhs: str, line: int) -> list[Binding]:
    found: list[Binding] = []
    for part in _split_top_level(inner):
        token = part.strip()
        if not token:
            continue
        if token.startswith("..."):
            rest = token[3:].strip()
            if rest:
                found.append(Binding(name=rest, line=line, rhs=rhs))
            continue
        if ":" in token:
            src, dest = token.split(":", 1)
            src, dest = src.strip(), dest.strip().split("=", 1)[0].strip()
            dest = dest.split(":", 1)[0].strip()
            if dest and _IDENT.fullmatch(dest):
                found.append(Binding(name=dest, line=line, rhs=f"{rhs}.{src}"))
            continue
        name = token.split("=", 1)[0].strip()
        if name and _IDENT.fullmatch(name):
            found.append(Binding(name=name, line=line, rhs=f"{rhs}.{name}"))
    return found


def _destructure_array(inner: str, rhs: str, line: int) -> list[Binding]:
    found: list[Binding] = []
    for index, part in enumerate(_split_top_level(inner)):
        token = part.strip()
        if not token:
            continue
        if token.startswith("..."):
            rest = token[3:].strip()
            if rest:
                found.append(Binding(name=rest, line=line, rhs=rhs))
            continue
        name = token.split("=", 1)[0].strip()
        if name and _IDENT.fullmatch(name):
            found.append(Binding(name=name, line=line, rhs=f"{rhs}[{index}]"))
    return found


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str: str | None = None
    escape = False
    for ch in text:
        if in_str:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _calls(source: str, *, bare_keywords: tuple[str, ...] = ()) -> list[CallSite]:
    found: list[CallSite] = []
    n = len(source)
    i = 0
    line = 1
    in_str: str | None = None
    escape = False
    bare = {kw.lower() for kw in bare_keywords}

    while i < n:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
                i += 1
                continue
            if ch == "\\":
                escape = True
                i += 1
                continue
            if in_str == "`" and ch == "$" and i + 1 < n and source[i + 1] == "{":
                i += 2
                line += _scan_interpolation_calls(source, i, line, found, bare)
                # Advance to the matching `}` that closed the interpolation.
                i = _skip_balanced_brace(source, i)
                continue
            if ch == "\n":
                line += 1
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
            i += 1
            continue
        if ch == "\n":
            line += 1
            i += 1
            continue
        match = _IDENT.match(source, i)
        if not match:
            i += 1
            continue
        ident = match.group(0)
        ident_start = match.start()
        j = match.end()

        if ident == "new":
            ctor = _constructor_call(source, j, ident_start, line)
            if ctor is not None:
                site, end, extra_lines = ctor
                found.append(site)
                line += extra_lines
                i = end
                continue

        qualified, ident, j = _read_member_chain(source, ident, j)
        j = _skip_ws_nl(source, j)

        if j < n and source[j] == "(":
            is_member = any(sep in qualified for sep in (".", "::", "->"))
            if (ident.lower() in _NON_CALL_KEYWORDS and not is_member) or _is_declaration(
                source, ident_start
            ):
                i = match.end()
                continue
            args, end = _balanced_parens(source, j)
            site = CallSite(
                name=ident,
                qualified=qualified,
                line=line,
                argument_text=args,
                dynamic=_is_dynamic(args),
            )
            found.append(site)
            nested_line = line
            line += source[j:end].count("\n")
            _calls_from_args(args, nested_line, found, bare)
            i, line = _consume_call_chain(source, end, line, site, found)
            continue

        if j < n and source.startswith("={", j):
            args, end = _balanced_braces(source, j + 1)
            found.append(
                CallSite(
                    name=ident,
                    qualified=qualified,
                    line=line,
                    argument_text=args,
                    dynamic=_is_dynamic(args),
                )
            )
            line += source[j:end].count("\n")
            i = end
            continue

        if ident.lower() in bare:
            args, end = _bare_arguments(source, j)
            if args.strip():
                found.append(
                    CallSite(
                        name=ident,
                        qualified=qualified,
                        line=line,
                        argument_text=args,
                        dynamic=_is_dynamic(args),
                    )
                )
                line += source[j:end].count("\n")
                i = end
                continue

        if j < n and source[j] == "=" and not _is_comparison_or_arrow(source, j):
            if "." in qualified or "::" in qualified:
                rhs, end = _read_rhs(source, j + 1)
                found.append(
                    CallSite(
                        name=ident,
                        qualified=qualified,
                        line=line,
                        argument_text=rhs.strip(),
                        dynamic=_is_dynamic(rhs),
                    )
                )
                line += source[j:end].count("\n")
                i = end
                continue

        i = match.end()
    return found


def _calls_from_args(args: str, line: int, found: list[CallSite], bare: set[str]) -> None:
    if not args or not any(ch == "(" or ch == "=" for ch in args):
        return
    for call in _calls(args, bare_keywords=tuple(bare)):
        found.append(
            CallSite(
                name=call.name,
                qualified=call.qualified,
                line=line + call.line - 1,
                argument_text=call.argument_text,
                dynamic=call.dynamic,
            )
        )


def _scan_interpolation_calls(
    source: str,
    start: int,
    line: int,
    found: list[CallSite],
    bare: set[str],
) -> int:
    """Parse `${...}` as code. Returns extra newlines consumed inside it."""
    end = _skip_balanced_brace(source, start)
    inner = source[start : end - 1] if end > start else ""
    for call in _calls(inner, bare_keywords=tuple(bare)):
        found.append(
            CallSite(
                name=call.name,
                qualified=call.qualified,
                line=line + call.line - 1,
                argument_text=call.argument_text,
                dynamic=call.dynamic,
            )
        )
    return source[start:end].count("\n")


def _consume_call_chain(
    source: str,
    index: int,
    line: int,
    previous: CallSite,
    found: list[CallSite],
) -> tuple[int, int]:
    """Fold `.next(` / `?.next(` after a call into a qualified chain."""
    n = len(source)
    i = index
    last = previous
    while True:
        j = _skip_ws(source, i)
        if j < n and source[j] == "\n":
            # Allow `.method(` on the next line of a chain.
            peek = _skip_ws(source, j + 1)
            if peek < n and source.startswith((".", "?."), peek):
                line += 1
                j = peek
            else:
                break
        if j < n and source.startswith("?.", j):
            j += 2
        elif j < n and source[j] == ".":
            j += 1
        else:
            break
        j = _skip_ws(source, j)
        nxt = _IDENT.match(source, j)
        if not nxt:
            break
        ident = nxt.group(0)
        j = _skip_ws(source, nxt.end())
        if j >= n or source[j] != "(":
            break
        args, end = _balanced_parens(source, j)
        last = CallSite(
            name=ident,
            qualified=f"{last.qualified}().{ident}",
            line=line,
            argument_text=args,
            dynamic=_is_dynamic(args),
        )
        found.append(last)
        _calls_from_args(args, line, found, set())
        line += source[j:end].count("\n")
        i = end
    return i, line


def _balanced_braces(source: str, start: int) -> tuple[str, int]:
    if start >= len(source) or source[start] != "{":
        return "", start
    end = _skip_balanced_brace(source, start + 1)
    return source[start + 1 : end - 1], end


def _skip_balanced_brace(source: str, start: int) -> int:
    depth = 1
    i = start
    n = len(source)
    in_str: str | None = None
    escape = False
    while i < n and depth:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_str == "`" and ch == "$" and i + 1 < n and source[i + 1] == "{":
                i += 2
                i = _skip_balanced_brace(source, i)
                continue
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _constructor_call(
    source: str, j: int, _new_start: int, line: int
) -> tuple[CallSite, int, int] | None:
    j = _skip_ws(source, j)
    match = _IDENT.match(source, j)
    if not match:
        return None
    ident = match.group(0)
    qualified, ident, j = _read_member_chain(source, ident, match.end())
    j = _skip_ws(source, j)
    if j >= len(source) or source[j] != "(":
        return None
    args, end = _balanced_parens(source, j)
    site = CallSite(
        name=ident,
        qualified=f"new {qualified}",
        line=line,
        argument_text=args,
        dynamic=_is_dynamic(args),
    )
    return site, end, source[j:end].count("\n")


def _read_member_chain(source: str, ident: str, j: int) -> tuple[str, str, int]:
    qualified = ident
    n = len(source)
    while True:
        j = _skip_ws_nl(source, j)
        if j < n and source.startswith("?.", j):
            sep, j = ".", j + 2
        elif j < n and source.startswith("::", j):
            sep, j = "::", j + 2
        elif j < n and source.startswith("->", j):
            sep, j = "->", j + 2
        elif j < n and source[j] == ".":
            sep, j = ".", j + 1
        else:
            break
        j = _skip_ws_nl(source, j)
        nxt = _IDENT.match(source, j)
        if not nxt:
            break
        ident = nxt.group(0)
        qualified = f"{qualified}{sep}{ident}"
        j = nxt.end()
    return qualified, ident, j


def _skip_ws_nl(source: str, index: int) -> int:
    n = len(source)
    while index < n and source[index] in " \t\n":
        index += 1
    return index


def _is_declaration(source: str, ident_start: int) -> bool:
    k = ident_start
    while k > 0 and source[k - 1] in " \t":
        k -= 1
    prev = []
    while k > 0 and (source[k - 1].isalnum() or source[k - 1] in "_$"):
        k -= 1
        prev.append(source[k])
    token = "".join(reversed(prev)).lower()
    return token in _DECL_KEYWORDS


def _is_comparison_or_arrow(source: str, eq_at: int) -> bool:
    nxt = source[eq_at + 1] if eq_at + 1 < len(source) else ""
    prev = source[eq_at - 1] if eq_at else ""
    return nxt in "=>" or prev in "=!<>"


def _bare_arguments(source: str, start: int) -> tuple[str, int]:
    i = start
    n = len(source)
    while i < n and source[i] in " \t":
        i += 1
    begin = i
    in_str: str | None = None
    escape = False
    depth = 0
    while i < n:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif ch in ";\n" and depth == 0:
            break
        i += 1
    return source[begin:i].strip(), i


def _read_rhs(source: str, start: int) -> tuple[str, int]:
    i = start
    n = len(source)
    in_str: str | None = None
    escape = False
    depth = 0
    while i < n:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif ch in ";\n" and depth == 0:
            break
        i += 1
    return source[start:i], i


def _skip_ws(source: str, index: int) -> int:
    n = len(source)
    while index < n and source[index] in " \t":
        index += 1
    return index


def _balanced_parens(source: str, start: int) -> tuple[str, int]:
    depth = 0
    i = start
    n = len(source)
    in_str: str | None = None
    escape = False
    while i < n:
        ch = source[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_str == "`" and ch == "$" and i + 1 < n and source[i + 1] == "{":
                i += 2
                i = _skip_balanced_brace(source, i)
                continue
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return source[start + 1 : i], i + 1
        i += 1
    return source[start + 1 :], n


def _is_dynamic(argument_text: str) -> bool:
    return any(marker in _mask_strings(argument_text) for marker in _DYNAMIC_MARKERS)


def _mask_strings(text: str) -> str:
    """Replace quoted text with spaces so markers inside literals are ignored."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str: str | None = None
    escape = False
    while i < n:
        ch = text[i]
        if in_str:
            if escape:
                escape = False
                out.append(" ")
            elif ch == "\\":
                escape = True
                out.append(" ")
            elif in_str == "`" and ch == "$" and i + 1 < n and text[i + 1] == "{":
                out.append("${")
                i += 2
                continue
            elif ch == in_str:
                in_str = None
                out.append(" ")
            else:
                out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch in {'"', "'", "`"}:
            in_str = ch
            out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)

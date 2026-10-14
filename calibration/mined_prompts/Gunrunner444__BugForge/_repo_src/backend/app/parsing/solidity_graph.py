"""Solidity syntax graph.

Tree-sitter is the graph builder. A regex profile is used only when that
parser is unavailable, and that graph is labeled PROFILE_FALLBACK. solc is
not required. When solc happens to be installed, compile success is recorded
as a diagnostic and does not replace this graph.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.domain.language import ParserTier
from app.domain.source import ParsedEntity, ParsedImport, ParsedParameter
from app.parsing.extract import parse_with_profile
from app.parsing.keccak import function_selector
from app.parsing.model import (
    CallKind,
    CallSite,
    LanguageProfile,
    ParserDiagnostics,
    ParserStatus,
    Scope,
    ScopeKind,
    SemanticKind,
    SemanticNode,
    Symbol,
    SymbolKind,
    SyntaxEvent,
    SyntaxGraph,
)
from app.parsing.profiles import profile_for
from app.parsing.solidity_types import canonical_solidity_type, is_dynamic_type
from app.parsing.span import span_from_ts_node

_LOW_LEVEL = frozenset({"call", "delegatecall", "staticcall", "transfer", "send"})


def parse_solidity_source(file_path: Path, source: str) -> SyntaxGraph:
    graph = _from_treesitter(file_path, source)
    if graph is not None:
        return graph
    return _profile_fallback(file_path, source, reason="native parser unavailable")


def _from_treesitter(file_path: Path, source: str) -> SyntaxGraph | None:
    try:
        import tree_sitter_language_pack as pack
        from tree_sitter import Parser
    except ImportError:
        return None
    try:
        language = pack.get_language("solidity")
    except (LookupError, OSError, ValueError):
        return None
    except Exception:
        return None
    parser = Parser(language)
    raw = source.encode("utf-8", errors="replace")
    tree = parser.parse(raw)
    builder = _Builder(file_path, source, raw)
    builder.consume(tree.root_node)
    has_error = bool(getattr(tree.root_node, "has_error", False))
    builder.graph.diagnostics = ParserDiagnostics(
        has_errors=has_error,
        error_count=1 if has_error else 0,
        recoverable=True,
        native_available=True,
        status=ParserStatus.NATIVE_AVAILABLE,
        message="tree-sitter solidity",
    )
    builder.graph.parser_backend = "tree_sitter"
    builder.graph.parser_tier = ParserTier.FULL_AST
    return builder.graph


def _profile_fallback(file_path: Path, source: str, *, reason: str) -> SyntaxGraph:
    profile = profile_for("solidity")
    if profile is None:
        profile = LanguageProfile(
            language_id="solidity", display_name="Solidity", extensions=frozenset({".sol"})
        )
    graph = parse_with_profile(profile, file_path, source)
    graph.parser_backend = "profile"
    graph.parser_tier = ParserTier.PROFILE_FALLBACK
    graph.diagnostics = ParserDiagnostics(
        native_available=False,
        status=ParserStatus.PROFILE_FALLBACK,
        fallback_reason=reason,
        message=reason,
    )
    return graph


class _Builder:
    def __init__(self, file_path: Path, source: str, raw: bytes) -> None:
        self.raw = raw
        self.graph = SyntaxGraph(
            language="solidity",
            file_path=str(file_path),
            source=source,
            lines=tuple(source.splitlines()),
            scopes=(Scope("module", ScopeKind.MODULE, "module"),),
        )
        self._imports: list[ParsedImport] = []
        self._entities: list[ParsedEntity] = []
        self._calls: list[CallSite] = []
        self._nodes: list[SemanticNode] = []
        self._symbols: list[Symbol] = []
        self._events: list[SyntaxEvent] = []
        self._scopes: list[Scope] = [self.graph.scopes[0]]
        self._node_index = 0
        self._structs: dict[str, str] = {}
        self._aliases: dict[str, str] = {}

    def consume(self, root: object) -> None:
        self._index_types(root)
        for child in _children(root):
            kind = _type(child)
            if kind == "pragma_directive":
                self._pragma(child)
            elif kind == "import_directive":
                self._import(child)
            elif kind in {"contract_declaration", "interface_declaration", "library_declaration"}:
                self._contract(child, kind)
            elif kind in {"struct_declaration", "enum_declaration"}:
                self._named(child, kind.removesuffix("_declaration"))
        from app.parsing.solidity_version import solidity_language_facts

        facts = solidity_language_facts(self.graph.source)
        context = ["solidity"]
        if facts.floor:
            context.append(f"compiler_floor={facts.floor}")
            context.append("compiler_floor_is_minimum=true")
        else:
            context.append("compiler_floor=unknown")
        if facts.arithmetic == "checked":
            context.append("checked_arithmetic=true")
        elif facts.arithmetic == "wrapping":
            context.append("checked_arithmetic=false")
        else:
            context.append("checked_arithmetic=unknown")
        if facts.constraints:
            context.append("pragma_constraints=" + facts.constraints[:300])
        if facts.selfdestruct != "unknown":
            context.append(f"selfdestruct={facts.selfdestruct}")
        self.graph.semantic_context = tuple(context)
        self.graph.imports = tuple(self._imports)
        self.graph.entities = tuple(self._entities)
        self.graph.calls = tuple(self._calls)
        self.graph.nodes = tuple(self._nodes)
        self.graph.symbols = tuple(self._symbols)
        self.graph.events = tuple(self._events)
        self.graph.scopes = tuple(self._scopes)

    def _pragma(self, node: object) -> None:
        text = self._text(node)
        self._event("sol_pragma", node, text)
        if ">=" in text and "<" not in text:
            self._event("sol_pragma_unbounded", node, text)

    def _import(self, node: object) -> None:
        text = self._text(node)
        module = ""
        for child in _descendants(node):
            if _type(child) == "string":
                module = self._text(child).strip().strip('"').strip("'")
        span = span_from_ts_node(node)
        self._imports.append(
            ParsedImport(
                module=module or text,
                line_number=span.start_line,
                is_from_import=" from " in text,
                import_type="relative" if module.startswith(".") else "local",
                column=span.start_column,
                start_byte=span.start_byte,
                end_byte=span.end_byte,
                syntax_kind="import",
            )
        )
        self._event("sol_import", node, module or text)

    def _contract(self, node: object, kind: str) -> None:
        name = _child_text(self, node, "identifier") or "contract"
        entity_kind = {
            "contract_declaration": "contract",
            "interface_declaration": "interface",
            "library_declaration": "library",
        }[kind]
        span = span_from_ts_node(node)
        scope_id = f"module::{name}"
        self._scopes.append(Scope(scope_id, ScopeKind.CLASS, name, parent_id="module", span=span))
        bases = [
            self._text(child)
            for child in _descendants(node)
            if _type(child) == "inheritance_specifier"
        ]
        state: dict[str, str] = {}
        body = _first(node, "contract_body")
        if body is not None:
            for child in _children(body):
                if _type(child) == "state_variable_declaration":
                    var_name, mutability = self._state_var(child, name, scope_id)
                    if var_name:
                        state[var_name] = mutability
                elif _type(child) == "using_directive":
                    self._event("sol_using", child, self._text(child), extra=f"contract={name}")
                elif _type(child) == "modifier_definition":
                    self._named(child, "modifier", parent=name)
                elif _type(child) in {
                    "event_definition",
                    "error_declaration",
                    "struct_declaration",
                    "enum_declaration",
                }:
                    self._named(child, _type(child).split("_", 1)[0], parent=name)
        self._entities.append(
            ParsedEntity(
                entity_type=entity_kind,
                name=name,
                qualified_name=name,
                start_line=span.start_line,
                end_line=span.end_line,
                parent=None,
                start_column=span.start_column,
                end_column=span.end_column,
                start_byte=span.start_byte,
                end_byte=span.end_byte,
                node_id=self._add_node(entity_kind, name, node, extra=",".join(bases)),
            )
        )
        self._event(
            "sol_contract",
            node,
            name,
            extra=f"kind={entity_kind}|bases={','.join(bases)}",
        )
        if body is None:
            return
        for child in _children(body):
            if _type(child) in {
                "function_definition",
                "constructor_definition",
                "fallback_receive_definition",
                "modifier_definition",
            }:
                if _type(child) == "modifier_definition":
                    continue
                self._function(child, name, scope_id, state)

    def _function(
        self, node: object, contract: str, parent_scope: str, state: dict[str, str]
    ) -> None:
        name = _child_text(self, node, "identifier")
        label = _type(node)
        if label == "constructor_definition":
            name = "constructor"
        if label == "fallback_receive_definition":
            name = "receive" if "receive" in self._text(node).lstrip()[:12] else "fallback"
        name = name or "function"
        visibility = ""
        mutability = ""
        modifiers: list[str] = []
        override = False
        params: list[ParsedParameter] = []
        for child in _children(node):
            ctype = _type(child)
            if ctype == "visibility":
                visibility = self._text(child).strip()
            elif ctype == "state_mutability":
                mutability = self._text(child).strip()
            elif ctype == "modifier_invocation":
                modifiers.append(self._text(child).strip())
            elif ctype == "override_specifier":
                override = True
            elif ctype == "parameter" and _parent_is_return(child, node) is False:
                pname = _child_text(self, child, "identifier")
                ptype = _child_text(self, child, "type_name") or _child_text(
                    self, child, "primitive_type"
                )
                if pname:
                    params.append(ParsedParameter(name=pname, annotation=ptype))
        body = _first(node, "function_body")
        span = span_from_ts_node(node)
        scope_id = f"{parent_scope}::{name}:{span.start_line}"
        self._scopes.append(
            Scope(scope_id, ScopeKind.FUNCTION, name, parent_id=parent_scope, span=span)
        )
        types = [self._type_of(param.annotation or "") for param in params]
        selector = ""
        selector_status = "not_applicable"
        errored = bool(getattr(node, "has_error", False))
        if name not in {"constructor", "fallback", "receive"}:
            if not errored and all(types):
                selector = function_selector(f"{name}({','.join(types)})")
                selector_status = "canonical"
            else:
                selector_status = "unresolved"
        self._entities.append(
            ParsedEntity(
                entity_type="function",
                name=name,
                qualified_name=f"{contract}.{name}",
                start_line=span.start_line,
                end_line=span.end_line,
                decorators=list(modifiers),
                parameters=params,
                parent=contract,
                start_column=span.start_column,
                end_column=span.end_column,
                start_byte=span.start_byte,
                end_byte=span.end_byte,
                node_id=self._add_node(
                    SemanticKind.FUNCTION.value,
                    name,
                    node,
                    extra=selector,
                ),
            )
        )
        extra = "|".join(
            [
                f"contract={contract}",
                f"function={name}",
                f"visibility={visibility}",
                f"mutability={mutability}",
                f"modifiers={','.join(modifiers)}",
                f"override={str(override).lower()}",
                f"selector={selector}",
                f"selector_status={selector_status}",
                f"params={','.join(types)}",
                f"payable={str(mutability == 'payable').lower()}",
            ]
        )
        param_types = {param.name: self._type_of(param.annotation or "") for param in params}
        function_text = self._text(node)
        self._event("sol_function", node, function_text, extra=extra, limit=12000)
        self._record_cfg(node, function_text, contract, name)
        if override:
            self._event("sol_override", node, name, extra=f"contract={contract}")
        inner = "" if body is None else self._text(body).strip().strip("{}").strip()
        if not inner:
            if name in {"fallback", "receive"}:
                self._event("sol_empty_handler", node, name, extra=f"contract={contract}")
            return
        statements = [child for child in _descendants(body) if _type(child) == "statement"]
        # Direct statements only, so following-statement checks stay local.
        direct = [child for child in _children(body) if _type(child) == "statement"]
        for index, statement in enumerate(direct or statements):
            self._statement(
                statement,
                contract,
                name,
                scope_id,
                state,
                direct[index + 1 : index + 3] if direct else [],
                param_types,
            )

    def _statement(
        self,
        statement: object,
        contract: str,
        function: str,
        scope_id: str,
        state: dict[str, str],
        following: list[object],
        param_types: dict[str, str],
    ) -> None:
        text = self._text(statement)
        prefix = f"contract={contract}|function={function}"
        for node in _descendants(statement):
            kind = _type(node)
            if kind == "call_expression":
                self._call(node, function, scope_id, prefix, text, following, param_types)
            elif kind in {"assignment_expression", "augmented_assignment_expression"}:
                self._write(node, state, prefix)
            elif kind == "unchecked":
                if re.search(r"[+\-*/]", text):
                    self._event("sol_unchecked", statement, text, extra=prefix)
                else:
                    self._event("sol_unchecked_empty", statement, text, extra=prefix)
            elif kind == "type_cast_expression":
                cast = self._text(node)
                if re.match(r"u?int(8|16|32|64|128)\s*\(", cast):
                    self._event("sol_downcast", node, cast, extra=prefix + _cast_guard(text))
                if re.match(r"uint(256)?\s*\(", cast) and _signed_argument(cast, param_types):
                    self._event("sol_sign_cast", node, cast, extra=prefix)
            elif kind in {"for_statement", "while_statement", "do_while_statement"}:
                self._event(
                    "sol_loop", node, self._text(node), extra=prefix + f"|style={kind}", limit=4000
                )
            elif kind == "yul_evm_builtin":
                opcode = self._text(node)
                self._event("sol_yul", node, opcode, extra=prefix + f"|op={opcode}")
            elif kind == "assembly_statement":
                self._event("sol_assembly", node, self._text(node)[:180], extra=prefix)
            elif kind == "identifier" and self._text(node) == "ecrecover":
                self._event("sol_ecrecover", node, "ecrecover", extra=prefix)
            elif kind == "member_expression" and self._text(node) == "tx.origin":
                self._event("sol_tx_origin", node, "tx.origin", extra=prefix)
        for name in state:
            if _reads_state(text, name):
                self._event("sol_state_read", statement, text, extra=f"{prefix}|name={name}")
        if re.search(r"\brequire\s*\(|\bif\s*\(", text) and re.search(
            r"msg\.sender\s*==|==\s*msg\.sender|\bhasRole\s*\(", text
        ):
            self._event("sol_auth_guard", statement, text, extra=prefix)
        if re.search(r"\brequire\s*\(", text):
            self._event("sol_require", statement, text, extra=prefix)
        if re.search(r"\bassert\s*\(", text):
            self._event("sol_assert", statement, text, extra=prefix)
        if re.search(r"\brevert\b", text):
            self._event("sol_revert", statement, text, extra=prefix)
        if re.search(r"\w+\s*/\s*\w+\s*\*\s*\w+", text):
            self._event("sol_div_mul", statement, text, extra=prefix)
        if re.search(r"\b(throw|suicide)\b", text):
            self._event("sol_deprecated", statement, text, extra=prefix)
        if re.search(r"\b(selfdestruct|suicide)\s*\(", text):
            self._event("sol_selfdestruct", statement, text, extra=prefix)
        if "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc" in text.lower():
            self._event("sol_eip1967", statement, text, extra=prefix)
        if re.search(r"block\.(timestamp|prevrandao)|blockhash\s*\(", text):
            deadline = bool(re.search(r"\b(require|if)\b", text) and re.search(r"[<>]", text))
            seed = (
                "keccak" in text
                or "%" in text
                or bool(re.search(r"\b(winner|raffle|lottery)\b", text))
            )
            if seed and not (deadline and "keccak" not in text and "%" not in text):
                self._event("sol_randomness", statement, text, extra=prefix)

    def _call(
        self,
        node: object,
        function: str,
        scope_id: str,
        prefix: str,
        statement: str,
        following: list[object],
        param_types: dict[str, str],
    ) -> None:
        text = self._text(node)
        head = re.split(r"[\(\{]", text, maxsplit=1)[0].strip()
        member = head.split(".")[-1].strip() if "." in head else ""
        target = head[: -len(member)].rstrip(".") if member else head
        following_text = [self._text(item) for item in following]
        checked, success = _call_is_checked(statement, text, following_text)
        span = span_from_ts_node(node)
        self._calls.append(
            CallSite(
                name=member or target,
                qualified=text,
                line=span.start_line,
                argument_text=text,
                kind=CallKind.MEMBER if member else CallKind.DIRECT,
                span=span,
                scope_id=scope_id,
            )
        )
        extra = (
            f"{prefix}|member={member}|target={target}|checked={str(checked).lower()}"
            f"|success={success}"
        )
        if member in _LOW_LEVEL or member == "safeTransferFrom":
            self._event("sol_external_call", node, text, extra=extra + f"|kind={member}")
        if member in {"transfer", "transferFrom"} and "," in text:
            self._event("sol_erc20", node, text, extra=extra)
        if member == "delegatecall":
            self._event("sol_delegatecall", node, text, extra=extra)
        if member == "safeTransferFrom":
            self._event("sol_token_callback", node, text, extra=extra)
        if member in {"latestRoundData", "latestAnswer"}:
            self._event("sol_oracle_call", node, text, extra=f"{prefix}|member={member}")
        if member == "encodePacked":
            packed = _packed_args(text, param_types)
            if "keccak" in statement or "ecrecover" in statement:
                self._event("sol_encode_packed", node, text, extra=f"{prefix}|{packed}")
        if not member:
            self._event("sol_direct_call", node, text, extra=f"{prefix}|name={target}")

    def _write(self, node: object, state: dict[str, str], prefix: str) -> None:
        text = self._text(node)
        lhs = re.split(r"\s*(?:=|\+=|-=|\*=|/=)\s*", text, maxsplit=1)[0]
        for name, mutability in state.items():
            if re.search(rf"\b{re.escape(name)}\b", lhs):
                self._event(
                    "sol_state_write",
                    node,
                    text,
                    extra=f"{prefix}|name={name}|mutability={mutability}",
                )

    def _state_var(self, node: object, contract: str, scope_id: str) -> tuple[str, str]:
        name = ""
        for child in _children(node):
            if _type(child) == "identifier":
                name = self._text(child)
        text = self._text(node)
        mutability = (
            "immutable" if "immutable" in text else "constant" if "constant" in text else "storage"
        )
        type_text = text
        for word in (
            "public",
            "private",
            "internal",
            "external",
            "immutable",
            "constant",
            "override",
        ):
            type_text = re.sub(rf"\b{word}\b", "", type_text)
        if name:
            type_text = re.sub(rf"\b{re.escape(name)}\b", "", type_text, count=1)
        type_text = type_text.replace(";", "").strip()
        if name:
            self._symbols.append(
                Symbol(
                    Symbol.make_id(scope_id, name),
                    name,
                    scope_id,
                    SymbolKind.FIELD,
                    span_from_ts_node(node),
                )
            )
            self._event(
                "sol_state",
                node,
                text,
                extra=(f"contract={contract}|name={name}|mutability={mutability}|type={type_text}"),
            )
        return name, mutability

    def _named(self, node: object, kind: str, parent: str | None = None) -> None:
        name = _child_text(self, node, "identifier") or kind
        span = span_from_ts_node(node)
        self._entities.append(
            ParsedEntity(
                entity_type=kind,
                name=name,
                qualified_name=f"{parent}.{name}" if parent else name,
                start_line=span.start_line,
                end_line=span.end_line,
                parent=parent,
                start_column=span.start_column,
                end_column=span.end_column,
                start_byte=span.start_byte,
                end_byte=span.end_byte,
                node_id=self._add_node(kind, name, node),
            )
        )
        if kind == "struct":
            signature = self._struct_signature(node)
            if signature:
                self._structs[name] = signature
        if kind == "modifier":
            body = _first(node, "function_body")
            body_text = self._text(body) if body is not None else ""
            self._event(
                "sol_modifier",
                node,
                body_text,
                extra=f"contract={parent or ''}|name={name}",
                limit=4000,
            )
            return
        self._event(f"sol_{kind}", node, name, extra=f"contract={parent or ''}")

    def _type_of(self, annotation: str) -> str:
        compact = re.sub(r"\s+", "", annotation)
        compact = re.sub(r"\b(memory|calldata|storage|indexed|payable)\b", "", compact)
        array = re.fullmatch(r"(.+)\[(\d*)\]", compact)
        if array:
            base = self._type_of(array.group(1))
            if not base:
                return ""
            return f"{base}[{array.group(2)}]"
        if compact.startswith("(") and compact.endswith(")"):
            parts = _split_args(compact[1:-1])
            if not parts and compact != "()":
                return ""
            canon_parts = [self._type_of(part) for part in parts]
            if not all(canon_parts):
                return ""
            return "(" + ",".join(canon_parts) + ")"
        canon = canonical_solidity_type(annotation)
        if canon:
            return canon
        if compact in self._structs:
            return self._structs[compact]
        return self._aliases.get(compact, "")

    def _index_types(self, root: object) -> None:
        for node in _preorder(root):
            kind = _type(node)
            if kind == "enum_declaration":
                name = _direct_identifier(self, node)
                if name:
                    self._aliases[name] = "uint8"
                    self._event(
                        "sol_type_def",
                        node,
                        name,
                        extra=f"name={name}|abi=uint8|kind=enum",
                    )
            elif kind in {"contract_declaration", "interface_declaration"}:
                name = _direct_identifier(self, node)
                if name and name not in self._structs:
                    self._aliases[name] = "address"
                    label = "interface" if kind.startswith("interface") else "contract"
                    self._event(
                        "sol_type_def",
                        node,
                        name,
                        extra=f"name={name}|abi=address|kind={label}",
                    )
        pending = [
            node for node in _preorder(root) if _type(node) == "user_defined_type_definition"
        ]
        for _ in range(len(pending) + 1):
            unresolved: list[object] = []
            for node in pending:
                match = re.search(r"\btype\s+([A-Za-z_]\w*)\s+is\s+([^;]+)", self._text(node))
                if not match:
                    continue
                name = match.group(1)
                canon = self._type_of(match.group(2).strip())
                if not canon:
                    unresolved.append(node)
                    continue
                self._aliases[name] = canon
                self._event(
                    "sol_type_def",
                    node,
                    name,
                    extra=f"name={name}|abi={canon}|kind=udvt",
                )
            if len(unresolved) == len(pending):
                break
            pending = unresolved
        self._index_structs(root)
        for node in _preorder(root):
            if _type(node) != "struct_declaration":
                continue
            name = _direct_identifier(self, node)
            signature = self._structs.get(name, "")
            if name and signature:
                self._event(
                    "sol_type_def",
                    node,
                    name,
                    extra=f"name={name}|abi={signature}|kind=struct",
                )

    def _index_structs(self, root: object) -> None:
        nodes = [node for node in _preorder(root) if _type(node) == "struct_declaration"]
        pending = nodes
        for _ in range(len(nodes) + 1):
            unresolved: list[object] = []
            for node in pending:
                name = _child_text(self, node, "identifier")
                signature = self._struct_signature(node)
                if name and signature:
                    self._structs[name] = signature
                else:
                    unresolved.append(node)
            if len(unresolved) == len(pending):
                break
            pending = unresolved

    def _record_cfg(self, node: object, function_text: str, contract: str, name: str) -> None:
        from app.parsing.solidity_cfg import build_function_cfg

        cfg = build_function_cfg(function_text)
        self._event(
            "sol_cfg",
            node,
            name,
            extra=(
                f"contract={contract}|function={name}|nodes={len(cfg.nodes)}"
                f"|known={str(cfg.known).lower()}"
            ),
        )

    def _struct_signature(self, node: object) -> str:
        members: list[str] = []
        for child in _descendants(node):
            if _type(child) != "struct_member":
                continue
            type_name = _child_text(self, child, "type_name")
            canon = self._type_of(type_name)
            if not canon:
                return ""
            members.append(canon)
        if not members:
            return ""
        return "(" + ",".join(members) + ")"

    def _event(self, kind: str, node: object, text: str, extra: str = "", limit: int = 400) -> None:
        span = span_from_ts_node(node)
        self._events.append(
            SyntaxEvent(
                kind=kind, line=span.start_line, text=text[:limit], span=span, extra=extra[:800]
            )
        )

    def _add_node(self, kind: str, name: str, node: object, extra: str = "") -> str:
        self._node_index += 1
        node_id = f"sol{self._node_index}"
        try:
            semantic = SemanticKind(kind)
        except ValueError:
            semantic = SemanticKind.TYPE
        self._nodes.append(
            SemanticNode(
                node_id=node_id,
                kind=semantic,
                name=name,
                span=span_from_ts_node(node),
                language_type=kind,
                extra=extra,
            )
        )
        return node_id

    def _text(self, node: object) -> str:
        return self.raw[int(getattr(node, "start_byte")) : int(getattr(node, "end_byte"))].decode(
            "utf-8", "replace"
        )


def _children(node: object) -> tuple[object, ...]:
    return tuple(getattr(node, "children", ()) or ())


def _type(node: object) -> str:
    return str(getattr(node, "type", ""))


def _descendants(node: object) -> list[object]:
    found: list[object] = []
    stack = list(_children(node))
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(reversed(_children(current)))
    return found


def _first(node: object, kind: str) -> object | None:
    for child in _children(node):
        if _type(child) == kind:
            return child
    return None


def _direct_identifier(builder: _Builder, node: object) -> str:
    for child in _children(node):
        if _type(child) == "identifier":
            return builder._text(child)
    return ""


def _child_text(builder: _Builder, node: object, kind: str) -> str:
    for child in _preorder(node):
        if child is not node and _type(child) == kind:
            return builder._text(child)
    return ""


def _preorder(node: object) -> list[object]:
    found: list[object] = []
    stack = [node]
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(reversed(_children(current)))
    return found


def _parent_is_return(param: object, function: object) -> bool:
    """Parameters under return_type_definition are not inputs."""
    return_node = _first(function, "return_type_definition")
    if return_node is None:
        return False
    return param in _descendants(return_node) or param is return_node


def _canonical_type(annotation: str) -> str:
    return canonical_solidity_type(annotation)


def _reads_state(text: str, name: str) -> bool:
    if not re.search(rf"\b{re.escape(name)}\b", text):
        return False
    if re.search(rf"\b{re.escape(name)}\b\s*(\[[^\]]*\])?\s*(\+=|-=|\*=|/=)", text):
        return True
    parts = re.split(r"(?<![<>=!+\-*/])=(?!=)", text, maxsplit=1)
    if len(parts) == 1:
        return True
    return bool(re.search(rf"\b{re.escape(name)}\b", parts[1]))


def _call_is_checked(statement: str, call_text: str, following: list[str]) -> tuple[bool, str]:
    """A call is checked only when its success value is actually tested."""
    require_at = statement.find("require")
    call_at = statement.find(call_text)
    if require_at != -1 and call_at != -1 and require_at < call_at:
        return True, ""
    match = re.search(r"\(\s*bool\s+([A-Za-z_]\w*)|bool\s+([A-Za-z_]\w*)\s*=", statement)
    name = ""
    if match:
        name = match.group(1) or match.group(2) or ""
    if not name:
        return False, ""
    for text in following:
        if text.strip().startswith("emit"):
            continue
        if re.search(r"\b(require|assert|if)\b", text) and re.search(rf"\b{name}\b", text):
            return True, name
    return False, name


def _packed_args(call_text: str, param_types: dict[str, str]) -> str:
    inner = call_text[call_text.find("(") + 1 : call_text.rfind(")")]
    args = [part.strip() for part in _split_args(inner)]
    dynamic = 0
    static = 0
    unknown = 0
    for arg in args:
        ident = arg.split(".")[-1].strip()
        known = param_types.get(ident, "")
        if known:
            if is_dynamic_type(known):
                dynamic += 1
            else:
                static += 1
            continue
        compact = re.sub(r"\s+", "", arg)
        cast = re.fullmatch(
            r"(u?int\d*|bytes\d+|address|bool|string|bytes)\((.*)\)",
            compact,
        )
        if re.fullmatch(r"0x[0-9a-fA-F]+|\d+|true|false|\"[^\"]*\"", arg):
            static += 1
        elif cast and cast.group(1) in {"string", "bytes"}:
            dynamic += 1
        elif cast:
            static += 1
        elif arg.startswith("string(") or arg.startswith("bytes(") or compact.endswith("[]"):
            dynamic += 1
        else:
            unknown += 1
    return f"args={len(args)}|dynamic={dynamic}|static={static}|unknown={unknown}"


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    if text[start:].strip():
        parts.append(text[start:])
    return parts


def _signed_argument(cast: str, param_types: dict[str, str]) -> bool:
    inner = cast[cast.find("(") + 1 : cast.rfind(")")]
    ident = inner.strip().split(".")[-1]
    typed = param_types.get(ident, "")
    if typed.startswith("int"):
        return True
    return bool(re.match(r"-\d", inner.strip()))


def _cast_guard(statement: str) -> str:
    guarded = "type(" in statement and ".max" in statement
    return "|guarded=" + ("true" if guarded else "false")

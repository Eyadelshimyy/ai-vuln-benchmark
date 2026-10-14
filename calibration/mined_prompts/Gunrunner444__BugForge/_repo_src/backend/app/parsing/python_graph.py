"""Python AST → SyntaxGraph. Primary Python backend (CPython)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from app.domain.language import ParserTier
from app.domain.source import ParsedEntity, ParsedImport, ParsedParameter
from app.parsing.model import (
    Binding,
    CallArgument,
    CallKind,
    CallSite,
    ParserDiagnostics,
    ReturnSite,
    RouteEndpoint,
    Scope,
    ScopeKind,
    SemanticKind,
    SemanticNode,
    Symbol,
    SymbolKind,
    SyntaxEvent,
    SyntaxGraph,
)
from app.parsing.routes import (
    constructed_route_receiver,
    is_route_method,
    looks_like_route_path,
    path_parameters,
)
from app.parsing.span import SourceSpan, span_from_lineno

_STDLIB = frozenset(sys.stdlib_module_names)


def parse_python_graph(file_path: Path, source: str) -> SyntaxGraph:
    lines = tuple(source.splitlines() or [""])
    errors: list[str] = []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
        span = span_from_lineno(
            source, exc.lineno or 1, exc.lineno or 1, start_column=(exc.offset or 1)
        )
        return SyntaxGraph(
            language="python",
            file_path=str(file_path),
            source=source,
            lines=lines,
            errors=tuple(errors),
            parser_backend="cpython_ast",
            parser_tier=ParserTier.FULL_AST,
            diagnostics=ParserDiagnostics(
                has_errors=True,
                error_count=1,
                error_spans=(span,),
                recoverable=False,
                message=errors[0],
            ),
        )

    builder = _PythonGraphVisitor(source)
    builder.visit(tree)
    return SyntaxGraph(
        language="python",
        file_path=str(file_path),
        source=source,
        lines=lines,
        imports=tuple(builder.imports),
        entities=tuple(builder.entities),
        calls=tuple(builder.calls),
        bindings=tuple(builder.bindings),
        errors=(),
        parser_backend="cpython_ast",
        parser_tier=ParserTier.FULL_AST,
        diagnostics=ParserDiagnostics(
            native_available=True,
            status="native_parser_available",
        ),
        scopes=tuple(builder.scopes),
        symbols=tuple(builder.symbols),
        nodes=tuple(builder.nodes),
        returns=tuple(builder.returns),
        events=tuple(builder.events),
        file_context=_file_context(str(file_path)),
        routes=tuple(builder.routes),
    )


class _PythonGraphVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.imports: list[ParsedImport] = []
        self.entities: list[ParsedEntity] = []
        self.calls: list[CallSite] = []
        self.bindings: list[Binding] = []
        self.returns: list[ReturnSite] = []
        self.events: list[SyntaxEvent] = []
        self.routes: list[RouteEndpoint] = []
        self.scopes: list[Scope] = [Scope(scope_id="module", kind=ScopeKind.MODULE, name="module")]
        self.symbols: list[Symbol] = []
        self.nodes: list[SemanticNode] = []
        self._scope_stack = list(self.scopes)
        self._class: str | None = None
        self._conditional_depth = 0
        self._def_index: dict[str, int] = {}

    @property
    def scope_id(self) -> str:
        return self._scope_stack[-1].scope_id

    def visit_Import(self, node: ast.Import) -> None:
        span = _span(self.source, node)
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    module=alias.name,
                    alias=alias.asname,
                    line_number=node.lineno,
                    is_from_import=False,
                    import_type=_classify_module(alias.name, relative=False),
                    column=node.col_offset + 1,
                    start_byte=span.start_byte,
                    end_byte=span.end_byte,
                    syntax_kind="import",
                )
            )
            bound = alias.asname or alias.name.split(".", 1)[0]
            self._bind(bound, node.lineno, alias.name, SymbolKind.LOCAL, span)
        self._node(SemanticKind.IMPORT, alias.name if node.names else "import", span, "Import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        relative = (node.level or 0) > 0
        import_type = "relative" if relative else _classify_module(module, relative=False)
        span = _span(self.source, node)
        for alias in node.names:
            self.imports.append(
                ParsedImport(
                    module=module,
                    name=alias.name,
                    alias=alias.asname,
                    line_number=node.lineno,
                    is_from_import=True,
                    import_type=import_type,
                    column=node.col_offset + 1,
                    start_byte=span.start_byte,
                    end_byte=span.end_byte,
                    syntax_kind="from_import",
                    relative_level=node.level or 0,
                )
            )
            bound = alias.asname or alias.name
            if bound and bound != "*":
                rhs = f"{module}.{alias.name}" if module else alias.name
                self._bind(bound, node.lineno, rhs, SymbolKind.LOCAL, span)
        self._node(SemanticKind.IMPORT, module, span, "ImportFrom")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        span = _span(self.source, node)
        scope_id = f"{self.scope_id}/class:{node.name}"
        scope = Scope(
            scope_id=scope_id,
            kind=ScopeKind.CLASS,
            name=node.name,
            parent_id=self.scope_id,
            span=span,
        )
        self.entities.append(
            ParsedEntity(
                entity_type="class",
                name=node.name,
                qualified_name=node.name,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                start_column=node.col_offset + 1,
                end_column=(node.end_col_offset or 0) + 1,
                start_byte=span.start_byte,
                end_byte=span.end_byte,
                docstring=ast.get_docstring(node),
                parent=self._class,
                decorators=[_expr(d) for d in node.decorator_list],
                node_id=f"python:class:{span.start_byte}:{node.name}",
            )
        )
        self._node(SemanticKind.CLASS, node.name, span, "ClassDef")
        prev = self._class
        self._class = node.name
        self._scope_stack.append(scope)
        self.scopes.append(scope)
        self.generic_visit(node)
        self._scope_stack.pop()
        self._class = prev

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        entity_type = (
            "async_method"
            if self._class and is_async
            else ("method" if self._class else ("async_function" if is_async else "function"))
        )
        qn = f"{self._class}.{node.name}" if self._class else node.name
        span = _span(self.source, node)
        kind = ScopeKind.METHOD if self._class else ScopeKind.FUNCTION
        scope_id = f"{self.scope_id}/{kind.value}:{node.name}"
        scope = Scope(
            scope_id=scope_id, kind=kind, name=node.name, parent_id=self.scope_id, span=span
        )
        params = [
            ParsedParameter(name=arg.arg)
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        ]
        self.entities.append(
            ParsedEntity(
                entity_type=entity_type,
                name=node.name,
                qualified_name=qn,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                start_column=node.col_offset + 1,
                end_column=(node.end_col_offset or 0) + 1,
                start_byte=span.start_byte,
                end_byte=span.end_byte,
                docstring=ast.get_docstring(node),
                parameters=params,
                parent=self._class,
                decorators=[_expr(d) for d in node.decorator_list],
                node_id=f"python:function:{span.start_byte}:{qn}",
            )
        )
        self._node(
            SemanticKind.METHOD if self._class else SemanticKind.FUNCTION,
            node.name,
            span,
            type(node).__name__,
        )
        self._scope_stack.append(scope)
        self.scopes.append(scope)
        for param in params:
            self._bind(param.name, node.lineno, "", SymbolKind.PARAMETER, span, rhs_is_literal=True)
        self._record_routes(node, qn, scope_id, params)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def _record_routes(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        function: str,
        scope_id: str,
        params: list[ParsedParameter],
    ) -> None:
        names = {param.name for param in params}
        for dec in node.decorator_list:
            recorded = self._decorator_route(dec)
            if recorded is None:
                continue
            method, path = recorded
            param_ids = tuple(
                Symbol.make_id(scope_id, name) for name in path_parameters(path) if name in names
            )
            self.routes.append(
                RouteEndpoint(
                    method=method,
                    path=path,
                    function=function,
                    scope_id=scope_id,
                    line=node.lineno,
                    parameter_ids=param_ids,
                )
            )

    def _decorator_route(self, node: ast.AST) -> tuple[str, str] | None:
        """``@app.get("/item/{id}")`` only when ``app`` is a constructed framework app."""
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        receiver = node.func.value
        if not isinstance(receiver, ast.Name):
            return None
        enclosing = self._scope_stack[-1].parent_id or "module"
        at_byte = _span(self.source, node).start_byte
        if not constructed_route_receiver(
            self.bindings,
            language="python",
            name=receiver.id,
            at_byte=at_byte,
            scope_id=enclosing,
            imports=self.imports,
            entities=self.entities,
        ):
            return None
        if not is_route_method(node.func.attr) or not node.args:
            return None
        path_node = node.args[0]
        if not isinstance(path_node, ast.Constant) or not isinstance(path_node.value, str):
            return None
        path = path_node.value
        if not looks_like_route_path(path):
            return None
        method = node.func.attr.lower()
        if method in {"route", "api_route"}:
            return _explicit_methods(node) or "ROUTE", path
        return method.upper(), path

    def visit_If(self, node: ast.If) -> None:
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    visit_AsyncFor = visit_For  # noqa: N815

    def visit_While(self, node: ast.While) -> None:
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.body:
            self.visit(stmt)
        self._conditional_depth += 1
        for handler in node.handlers:
            self.visit(handler)
        for item in node.orelse:
            self.visit(item)
        for item in node.finalbody:
            self.visit(item)
        self._conditional_depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        span = _span(self.source, node)
        scope = Scope(
            scope_id=f"{self.scope_id}/function:lambda",
            kind=ScopeKind.FUNCTION,
            name="lambda",
            parent_id=self.scope_id,
            span=span,
        )
        self._scope_stack.append(scope)
        self.scopes.append(scope)
        for arg in node.args.args:
            self._bind(arg.arg, node.lineno, "", SymbolKind.PARAMETER, span, rhs_is_literal=True)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        span = _span(self.source, node)
        self._block_seq = getattr(self, "_block_seq", 0) + 1
        scope = Scope(
            scope_id=f"{self.scope_id}/block:comp:{self._block_seq}",
            kind=ScopeKind.BLOCK,
            name=f"comp:{self._block_seq}",
            parent_id=self.scope_id,
            span=span,
        )
        self._scope_stack.append(scope)
        self.scopes.append(scope)
        self.generic_visit(node)
        self._scope_stack.pop()

    visit_ListComp = _visit_comprehension  # noqa: N815
    visit_SetComp = _visit_comprehension  # noqa: N815
    visit_DictComp = _visit_comprehension  # noqa: N815
    visit_GeneratorExp = _visit_comprehension  # noqa: N815

    def visit_Assign(self, node: ast.Assign) -> None:
        meta = _expr_meta(node.value)
        rhs = _expr(node.value)
        span = _span(self.source, node)
        for target in node.targets:
            self._bind_target(target, node.lineno, rhs, meta, span)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            meta = _expr_meta(node.value)
            self._bind_target(
                node.target, node.lineno, _expr(node.value), meta, _span(self.source, node)
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        qualified = _expr(node.func)
        name = qualified.rsplit(".", 1)[-1]
        args = ", ".join(_expr(a) for a in node.args)
        meta = _call_arg_meta(node)
        span = _span(self.source, node)
        kind = (
            CallKind.CONSTRUCTOR
            if isinstance(node.func, ast.Name) and node.func.id[:1].isupper()
            else CallKind.DIRECT
        )
        if isinstance(node.func, ast.Attribute):
            kind = CallKind.METHOD
        arguments_list = [
            CallArgument(
                index=i,
                text=_expr(arg),
                is_literal=_expr_meta(arg).is_literal,
                idents=_expr_meta(arg).idents,
                accesses=_expr_meta(arg).accesses,
                callees=_expr_meta(arg).callees,
                dynamic=_expr_meta(arg).dynamic,
            )
            for i, arg in enumerate(node.args)
        ]
        for keyword in node.keywords:
            meta = _expr_meta(keyword.value)
            arguments_list.append(
                CallArgument(
                    index=-1,
                    text=_expr(keyword.value),
                    is_literal=meta.is_literal,
                    idents=meta.idents,
                    accesses=meta.accesses,
                    callees=meta.callees,
                    dynamic=meta.dynamic,
                    keyword=keyword.arg or "**",
                )
            )
        arguments = tuple(arguments_list)
        self.calls.append(
            CallSite(
                name=name,
                qualified=qualified,
                line=node.lineno,
                argument_text=args,
                dynamic=meta.dynamic,
                kind=kind,
                span=span,
                node_id=f"python:call:{span.start_byte}:{qualified}",
                scope_id=self.scope_id,
                argument_is_literal=meta.is_literal,
                argument_idents=meta.idents,
                argument_accesses=meta.accesses + meta.callees,
                arguments=arguments,
                callee_identity=qn_if_method(self._class, name),
            )
        )
        self._node(SemanticKind.CALL, qualified, span, "Call")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        meta = _expr_meta(node.value) if node.value is not None else _Meta()
        span = _span(self.source, node)
        self.returns.append(
            ReturnSite(
                scope_id=self.scope_id,
                line=node.lineno,
                text=_expr(node.value) if node.value is not None else "",
                idents=meta.idents,
                accesses=meta.accesses + meta.callees,
                span=span,
            )
        )
        self._node(SemanticKind.RETURN, "return", span, "Return")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if isinstance(op, (ast.Eq, ast.NotEq)):
                right = node.comparators[0] if node.comparators else None
                extra = ""
                if isinstance(right, ast.Constant) and right.value is None:
                    extra = "none"
                self.events.append(
                    SyntaxEvent(
                        kind="loose_eq",
                        line=node.lineno,
                        text=_expr(node),
                        span=_span(self.source, node),
                        extra=extra,
                    )
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        empty = not node.body or all(isinstance(s, ast.Pass) for s in node.body)
        self.events.append(
            SyntaxEvent(
                kind="empty_catch" if empty else "catch",
                line=node.lineno,
                text="except",
                span=_span(self.source, node),
            )
        )
        self.generic_visit(node)

    def _bind_target(
        self,
        target: ast.AST,
        line: int,
        rhs: str,
        meta: _Meta,
        span: SourceSpan,
    ) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, line, rhs, SymbolKind.LOCAL, span, meta)
            return
        if isinstance(target, ast.Subscript):
            qualified = _constant_subscript(target)
            if qualified is None:
                return
            self._bind(qualified, line, rhs, SymbolKind.FIELD, span, meta)
            return
        if isinstance(target, ast.Attribute):
            qualified = _expr(target)
            self.calls.append(
                CallSite(
                    name=target.attr,
                    qualified=qualified,
                    line=line,
                    argument_text=rhs,
                    dynamic=meta.dynamic,
                    kind=CallKind.MEMBER_WRITE,
                    span=span,
                    node_id=f"python:write:{span.start_byte}:{qualified}",
                    scope_id=self.scope_id,
                    argument_is_literal=meta.is_literal,
                    argument_idents=meta.idents,
                    argument_accesses=meta.accesses + meta.callees,
                )
            )
            self._bind(qualified, line, rhs, SymbolKind.FIELD, span, meta)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt, line, rhs, meta, span)

    def _bind(
        self,
        name: str,
        line: int,
        rhs: str,
        kind: SymbolKind,
        span: SourceSpan,
        meta: _Meta | None = None,
        *,
        rhs_is_literal: bool = False,
    ) -> None:
        meta = meta or _Meta(is_literal=rhs_is_literal)
        symbol_id = Symbol.make_id(self.scope_id, name)
        nxt = self._def_index.get(symbol_id, 0) + 1
        self._def_index[symbol_id] = nxt
        binding = Binding(
            name=name,
            line=line,
            rhs=rhs,
            scope_id=self.scope_id,
            kind=kind,
            span=span,
            node_id=f"python:bind:{span.start_byte}:{name}",
            rhs_is_literal=meta.is_literal,
            rhs_callees=meta.callees,
            rhs_accesses=meta.accesses,
            rhs_idents=meta.idents,
            definition_index=nxt,
            is_declaration=kind is SymbolKind.PARAMETER or nxt == 1,
            is_conditional=self._conditional_depth > 0 and kind is not SymbolKind.PARAMETER,
            declarator="param" if kind is SymbolKind.PARAMETER else "assign",
        )
        self.bindings.append(binding)
        self.symbols.append(
            Symbol(
                symbol_id=binding.symbol_id,
                name=name,
                scope_id=self.scope_id,
                kind=kind,
                span=span,
                node_id=binding.node_id,
            )
        )

    def _node(self, kind: SemanticKind, name: str, span: SourceSpan, language_type: str) -> None:
        self.nodes.append(
            SemanticNode(
                node_id=f"python:{kind.value}:{span.start_byte}:{name}",
                kind=kind,
                name=name,
                span=span,
                parent_id=self.scope_id,
                language_type=language_type,
            )
        )


class _Meta:
    def __init__(
        self,
        callees: tuple[str, ...] = (),
        accesses: tuple[str, ...] = (),
        idents: tuple[str, ...] = (),
        dynamic: bool = False,
        is_literal: bool = False,
    ) -> None:
        self.callees = callees
        self.accesses = accesses
        self.idents = idents
        self.dynamic = dynamic
        self.is_literal = is_literal


def _explicit_methods(node: ast.Call) -> str:
    for keyword in node.keywords:
        if keyword.arg != "methods" or not isinstance(keyword.value, (ast.List, ast.Tuple)):
            continue
        names: list[str] = []
        for elt in keyword.value.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                return ""
            names.append(elt.value.upper())
        return ",".join(names)
    return ""


def qn_if_method(class_name: str | None, name: str) -> str:
    return f"{class_name}.{name}" if class_name else name


def _constant_subscript(node: ast.Subscript) -> str | None:
    """Static field path for ``obj["key"]`` and ``items[0]``. Dynamic keys are empty."""
    base = _subscript_base(node.value)
    key = _constant_key(node.slice)
    if not base or key is None:
        return None
    return f"{base}{key}"


def _subscript_base(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _expr(node)
    if isinstance(node, ast.Subscript):
        return _constant_subscript(node) or ""
    return ""


def _constant_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value and all(ch.isalnum() or ch == "_" for ch in node.value):
            return f'["{node.value}"]'
        return None
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return f"[{node.value}]"
    return None


def _expr_meta(node: ast.AST | None) -> _Meta:
    if node is None:
        return _Meta(is_literal=True)
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, bytes, int, float, type(None), bool)
    ):
        return _Meta(is_literal=True)
    callees: list[str] = []
    accesses: list[str] = []
    idents: list[str] = []
    dynamic = isinstance(node, (ast.BinOp, ast.JoinedStr, ast.FormattedValue))
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            callees.append(_expr(child.func))
        elif isinstance(child, ast.Attribute):
            accesses.append(_expr(child))
        elif isinstance(child, ast.Subscript):
            path = _constant_subscript(child)
            if path:
                accesses.append(path)
        elif isinstance(child, ast.Name):
            idents.append(child.id)
        elif isinstance(child, (ast.BinOp, ast.JoinedStr)):
            dynamic = True
    return _Meta(
        callees=tuple(callees),
        accesses=tuple(accesses),
        idents=tuple(dict.fromkeys(idents)),
        dynamic=dynamic,
        is_literal=False,
    )


def _call_arg_meta(node: ast.Call) -> _Meta:
    if not node.args:
        return _Meta(is_literal=True)
    callees: list[str] = []
    accesses: list[str] = []
    idents: list[str] = []
    dynamic = False
    literal = True
    for arg in node.args:
        meta = _expr_meta(arg)
        callees.extend(meta.callees)
        accesses.extend(meta.accesses)
        idents.extend(meta.idents)
        dynamic = dynamic or meta.dynamic
        literal = literal and meta.is_literal
    return _Meta(tuple(callees), tuple(accesses), tuple(dict.fromkeys(idents)), dynamic, literal)


def _expr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _span(source: str, node: ast.AST) -> SourceSpan:
    lineno = getattr(node, "lineno", 1) or 1
    end = getattr(node, "end_lineno", None) or lineno
    return span_from_lineno(
        source,
        lineno,
        end,
        start_column=(getattr(node, "col_offset", 0) or 0) + 1,
        end_column=(getattr(node, "end_col_offset", 0) or 0) + 1,
    )


def _classify_module(module: str, *, relative: bool) -> str:
    if relative:
        return "relative"
    root = (module or "").split(".", 1)[0]
    if root in _STDLIB:
        return "stdlib"
    if root:
        return "third_party"
    return "module"


def _file_context(path: str) -> str:
    lowered = path.replace("\\", "/").lower()
    if "/test" in lowered or lowered.startswith("test"):
        return "test"
    return "unknown"

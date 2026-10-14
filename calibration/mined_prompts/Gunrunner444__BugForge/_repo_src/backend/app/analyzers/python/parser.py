from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from app.domain.source import LanguageParseResult, ParsedEntity, ParsedImport, ParsedParameter

logger = logging.getLogger(__name__)

_STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)

# Backwards-compatible aliases for the language-neutral parse contract.
ParameterInfo = ParsedParameter
EntityInfo = ParsedEntity
ImportInfo = ParsedImport


@dataclass
class ParseResult(LanguageParseResult):
    """Python AST parse result. Satisfies :class:`LanguageParseResult`."""

    language: str = "python"


class PythonParser:
    """Parses Python source files using the built-in ast module."""

    def parse_file(self, file_path: Path) -> ParseResult:
        result = ParseResult(file_path=str(file_path))
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            result.errors.append(f"Cannot read file: {exc}")
            return result

        result.line_count = len(source.splitlines()) or 1

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            result.errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
            return result

        self._extract_imports(tree, result)
        self._extract_entities(tree, result, parent=None)
        return result

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _extract_imports(self, tree: ast.AST, result: ParseResult) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(
                        ImportInfo(
                            module=alias.name,
                            name=None,
                            alias=alias.asname,
                            line_number=node.lineno,
                            is_from_import=False,
                            import_type=self._classify_module(alias.name, relative=False),
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                relative = (node.level or 0) > 0
                import_type = (
                    "relative" if relative else self._classify_module(module, relative=False)
                )
                for alias in node.names:
                    result.imports.append(
                        ImportInfo(
                            module=module,
                            name=alias.name,
                            alias=alias.asname,
                            line_number=node.lineno,
                            is_from_import=True,
                            import_type=import_type,
                        )
                    )

    @staticmethod
    def _classify_module(module: str, *, relative: bool) -> str:
        if relative or not module:
            return "relative"
        top = module.split(".")[0]
        return "stdlib" if top in _STDLIB_MODULES else "third_party"

    # ------------------------------------------------------------------
    # Code entities
    # ------------------------------------------------------------------

    def _extract_entities(self, tree: ast.AST, result: ParseResult, parent: str | None) -> None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                result.entities.append(self._build_class_entity(node, parent))
                self._extract_entities(node, result, parent=node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.entities.append(self._build_function_entity(node, parent))

    def _build_class_entity(self, node: ast.ClassDef, parent: str | None) -> EntityInfo:
        qualified = f"{parent}.{node.name}" if parent else node.name
        return EntityInfo(
            entity_type="class",
            name=node.name,
            qualified_name=qualified,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            decorators=self._decorator_names(node),
            parameters=[],
            return_annotation=None,
            parent=parent,
        )

    def _build_function_entity(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent: str | None,
    ) -> EntityInfo:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        if parent:
            entity_type = "async_method" if is_async else "method"
        else:
            entity_type = "async_function" if is_async else "function"
        qualified = f"{parent}.{node.name}" if parent else node.name
        return EntityInfo(
            entity_type=entity_type,
            name=node.name,
            qualified_name=qualified,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            decorators=self._decorator_names(node),
            parameters=self._parameters(node),
            return_annotation=self._expr_str(node.returns),
            parent=parent,
        )

    def _decorator_names(
        self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        names: list[str] = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                names.append(f"{self._expr_str(dec.value)}.{dec.attr}")
            elif isinstance(dec, ast.Call) and isinstance(dec.func, (ast.Name, ast.Attribute)):
                result = self._expr_str(dec.func)
                if result is not None:
                    names.append(result)
        return names

    def _parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ParameterInfo]:
        params: list[ParameterInfo] = []
        args = node.args
        num_args = len(args.args)
        num_defaults = len(args.defaults)

        for i, arg in enumerate(args.args):
            default_idx = i - (num_args - num_defaults)
            default_val: str | None = None
            if 0 <= default_idx < num_defaults:
                default_val = self._expr_str(args.defaults[default_idx])
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=self._expr_str(arg.annotation),
                    default=default_val,
                    kind="positional",
                )
            )

        if args.vararg:
            params.append(
                ParameterInfo(
                    name=args.vararg.arg,
                    annotation=self._expr_str(args.vararg.annotation),
                    default=None,
                    kind="var_positional",
                )
            )

        for i, arg in enumerate(args.kwonlyargs):
            kw_default = args.kw_defaults[i] if i < len(args.kw_defaults) else None
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=self._expr_str(arg.annotation),
                    default=self._expr_str(kw_default) if kw_default else None,
                    kind="keyword",
                )
            )

        if args.kwarg:
            params.append(
                ParameterInfo(
                    name=args.kwarg.arg,
                    annotation=self._expr_str(args.kwarg.annotation),
                    default=None,
                    kind="var_keyword",
                )
            )

        return params

    @staticmethod
    def _expr_str(expr: ast.expr | None) -> str | None:
        if expr is None:
            return None
        try:
            return ast.unparse(expr)
        except Exception:
            return None

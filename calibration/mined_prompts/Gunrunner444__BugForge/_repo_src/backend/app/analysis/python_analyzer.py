"""Python static-analysis rules using the built-in ast module."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app.analysis.base import AnalyzerRule
from app.analysis.catalog import FindingCatalog
from app.analysis.finding import Finding

# ---------------------------------------------------------------------------
# Rule: mutable_default_argument
# ---------------------------------------------------------------------------


class MutableDefaultArgumentRule(AnalyzerRule):
    """Detect mutable objects (list, dict, set) as function-argument defaults."""

    RULE_ID = "mutable_default_argument"
    SEVERITY = "medium"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for default in node.args.defaults + node.args.kw_defaults:
                if default is None:
                    continue
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    type_name = type(default).__name__.replace("ast.", "").lower()
                    findings.append(
                        Finding(
                            category=self.RULE_ID,
                            severity=self.SEVERITY,
                            confidence=self.CONFIDENCE,
                            file_path=str(file_path),
                            line=node.lineno,
                            end_line=node.lineno,
                            column=node.col_offset,
                            message=f"Mutable default argument ({type_name}) in `{node.name}()`",
                            explanation=(
                                "Mutable defaults are shared across all calls that use the default. "
                                "Use `None` as the default and create the mutable inside the function."
                            ),
                            analyzer=self.RULE_ID,
                            evidence=ast.unparse(default),
                            suggested_fix=f"Change default to `None` and add `if param is None: param = {type_name}()`",
                            catalog=FindingCatalog.CODE_QUALITY,
                            language="python",
                            parser_backend="cpython_ast",
                        )
                    )
        return findings


# ---------------------------------------------------------------------------
# Rule: bare_except
# ---------------------------------------------------------------------------


class BareExceptRule(AnalyzerRule):
    """Detect bare `except:` clauses that catch everything including SystemExit."""

    RULE_ID = "bare_except"
    SEVERITY = "medium"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None:
                findings.append(
                    Finding(
                        category=self.RULE_ID,
                        severity=self.SEVERITY,
                        confidence=self.CONFIDENCE,
                        file_path=str(file_path),
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        message="Bare `except:` catches all exceptions including SystemExit and KeyboardInterrupt",
                        explanation=(
                            "Bare `except:` silences all exceptions, including system signals. "
                            "Use `except Exception:` at minimum, or catch specific exception types."
                        ),
                        analyzer=self.RULE_ID,
                        suggested_fix="Replace `except:` with `except Exception:`",
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# Rule: broad_exception_catch
# ---------------------------------------------------------------------------


class BroadExceptionCatchRule(AnalyzerRule):
    """Detect `except Exception:` handlers that re-raise nothing and do little."""

    RULE_ID = "broad_exception_catch"
    SEVERITY = "low"
    CONFIDENCE = "medium"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.type is None:
                continue
            # Check if the handler catches a very broad exception type
            exc_type = ast.unparse(node.type) if node.type else ""
            if exc_type not in ("Exception", "BaseException"):
                continue
            # If the body only passes or logs without re-raising, flag it
            body_has_raise = any(
                isinstance(n, ast.Raise)
                for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            )
            if not body_has_raise:
                findings.append(
                    Finding(
                        category=self.RULE_ID,
                        severity=self.SEVERITY,
                        confidence=self.CONFIDENCE,
                        file_path=str(file_path),
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        message=f"Broad `except {exc_type}:` without re-raising may hide bugs",
                        explanation=(
                            "Catching broad exception types without re-raising swallows unexpected "
                            "errors and makes debugging harder. Catch specific exceptions where possible."
                        ),
                        analyzer=self.RULE_ID,
                        evidence=exc_type,
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# Rule: comparison_to_none
# ---------------------------------------------------------------------------


class ComparisonToNoneRule(AnalyzerRule):
    """Detect `== None` or `!= None` instead of `is None` / `is not None`."""

    RULE_ID = "comparison_to_none"
    SEVERITY = "low"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        findings: list[Finding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for op, comparator in zip(node.ops, node.comparators):
                if not isinstance(comparator, ast.Constant) or comparator.value is not None:
                    continue
                if isinstance(op, ast.Eq):
                    findings.append(
                        Finding(
                            category=self.RULE_ID,
                            severity=self.SEVERITY,
                            confidence=self.CONFIDENCE,
                            file_path=str(file_path),
                            line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            message="Use `is None` instead of `== None`",
                            explanation="Identity comparison `is None` is correct and more explicit than equality comparison.",
                            analyzer=self.RULE_ID,
                            evidence=ast.unparse(node),
                            suggested_fix="Replace `== None` with `is None`",
                        )
                    )
                elif isinstance(op, ast.NotEq):
                    findings.append(
                        Finding(
                            category=self.RULE_ID,
                            severity=self.SEVERITY,
                            confidence=self.CONFIDENCE,
                            file_path=str(file_path),
                            line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            message="Use `is not None` instead of `!= None`",
                            explanation="Identity comparison `is not None` is correct and more explicit.",
                            analyzer=self.RULE_ID,
                            evidence=ast.unparse(node),
                            suggested_fix="Replace `!= None` with `is not None`",
                        )
                    )
        return findings


# ---------------------------------------------------------------------------
# Rule: hardcoded_credential_pattern
# ---------------------------------------------------------------------------

_CRED_PATTERN = re.compile(
    r"""(?i)(password|passwd|secret|api_key|apikey|token|auth_token|access_key)\s*=\s*['"][^'"]{4,}['"]"""
)


class HardcodedCredentialRule(AnalyzerRule):
    """Detect suspicious credential-like string assignments."""

    RULE_ID = "hardcoded_credential"
    SEVERITY = "high"
    CONFIDENCE = "medium"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        findings: list[Finding] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _CRED_PATTERN.search(line):
                # Skip obvious placeholders
                lower = line.lower()
                if any(
                    p in lower
                    for p in ("example", "your_", "placeholder", "change_me", "xxxx", "todo")
                ):
                    continue
                findings.append(
                    Finding(
                        category=self.RULE_ID,
                        severity=self.SEVERITY,
                        confidence=self.CONFIDENCE,
                        file_path=str(file_path),
                        line=lineno,
                        end_line=lineno,
                        message="Possible hardcoded credential in source code",
                        explanation=(
                            "Credentials should not be stored in source code. "
                            "Use environment variables or a secrets manager."
                        ),
                        analyzer=self.RULE_ID,
                        evidence=line.strip()[:120],
                        suggested_fix="Move the value to an environment variable and read it with os.environ.get()",
                    )
                )
        return findings


# ---------------------------------------------------------------------------
# Rule: unreachable_code_after_return
# ---------------------------------------------------------------------------


class UnreachableCodeRule(AnalyzerRule):
    """Detect statements that follow a `return` or `raise` in the same block."""

    RULE_ID = "unreachable_code"
    SEVERITY = "low"
    CONFIDENCE = "high"

    def check_file(self, file_path: Path, source: str) -> list[Finding]:
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        findings: list[Finding] = []
        for node in ast.walk(tree):
            body: list[ast.stmt] | None = None
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For, ast.While, ast.With)
            ):
                body = getattr(node, "body", None)
            if body is None:
                continue
            for i, stmt in enumerate(body[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    next_stmt = body[i + 1]
                    findings.append(
                        Finding(
                            category=self.RULE_ID,
                            severity=self.SEVERITY,
                            confidence=self.CONFIDENCE,
                            file_path=str(file_path),
                            line=next_stmt.lineno,
                            end_line=next_stmt.end_lineno or next_stmt.lineno,
                            message="Unreachable code after `return` or `raise`",
                            explanation="This statement will never execute because a return or raise precedes it in the same block.",
                            analyzer=self.RULE_ID,
                        )
                    )
                    break  # only report first unreachable per block
        return findings


# All rules exported from this module
ALL_PYTHON_RULES: list[AnalyzerRule] = [
    MutableDefaultArgumentRule(),
    BareExceptRule(),
    BroadExceptionCatchRule(),
    ComparisonToNoneRule(),
    HardcodedCredentialRule(),
    UnreachableCodeRule(),
]

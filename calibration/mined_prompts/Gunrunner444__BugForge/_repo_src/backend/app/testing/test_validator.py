"""
Test code validator — ensures generated tests are safe and syntactically valid
before execution in the sandbox.

Generated tests are UNTRUSTED AI output. Never execute without validation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# Direct call names that indicate dangerous operations
_DANGEROUS_NAMES = frozenset(
    {
        "exec",
        "eval",
        "__import__",
        "compile",
        "open",
        "os",
        "subprocess",
        "sys",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "input",
        "help",
        "memoryview",
        "classmethod",
    }
)

_DANGEROUS_ATTRS = frozenset(
    {
        "system",
        "popen",
        "spawn",
        "execl",
        "execv",
        "remove",
        "unlink",
        "rmdir",
        "makedirs",
        "walk",
        "getenv",
        "environ",
        "connect",
        "socket",
        "urlopen",
        "Process",
        "Thread",
        "Popen",
        "__import__",
        "__globals__",
        "__builtins__",
        "__dict__",
        "__class__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__code__",
        "__closure__",
        "__getattribute__",
        "__setattr__",
        "__reduce__",
        "__reduce_ex__",
    }
)

# Modules that must not be imported by generated/reproducer code.
# Docker isolation provides a second boundary, but we reject at validation time.
_DANGEROUS_MODULES = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "sys",
        "shutil",
        "pathlib",
        "tempfile",
        "pickle",
        "shelve",
        "marshal",
        "importlib",
        "ctypes",
        "signal",
        "gc",
        "resource",
        "platform",
        "select",
        "pty",
        "termios",
        "fcntl",
        "pwd",
        "grp",
        "mmap",
        "multiprocessing",
        "threading",
        "concurrent",
        "asyncio",
        "urllib",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "ssl",
        "builtins",
        "code",
        "codeop",
        "runpy",
        "webbrowser",
        "xmlrpc",
        "socketserver",
        "multiprocessing",
        "threading",
        "ctypes",
        "mmap",
        "pty",
        "fcntl",
        "sqlite3",
        "dbm",
        "nntplib",
        "poplib",
        "imaplib",
        "telnetlib",
        "paramiko",
    }
)


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None
    test_function_names: list[str] | None = None


def validate_test_code(code: str) -> ValidationResult:
    """Parse and validate AI-generated test code before sandbox execution.

    Checks:
      1. Valid Python syntax
      2. Contains at least one test function (name starting with test_)
      3. Does not import dangerous modules
      4. Does not call dangerous builtins directly or via aliases
    """
    if not code or not code.strip():
        return ValidationResult(valid=False, error="Generated code is empty")

    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(valid=False, error=f"Syntax error: {exc}")

    # 2. Find test functions
    test_fns: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") or node.name.startswith("Test"):
                test_fns.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            test_fns.append(node.name)

    if not test_fns:
        return ValidationResult(
            valid=False,
            error="No test functions or test classes found (expected names starting with test_ or Test)",
        )

    # 3. Reject imports of dangerous modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Check the top-level module name (e.g. "os.path" → "os")
                top = alias.name.split(".")[0]
                if top in _DANGEROUS_MODULES:
                    return ValidationResult(
                        valid=False,
                        error=f"Generated test imports dangerous module '{alias.name}'; rejecting for safety",
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _DANGEROUS_MODULES:
                return ValidationResult(
                    valid=False,
                    error=f"Generated test imports from dangerous module '{module}'; rejecting for safety",
                )
            # Also reject wildcard imports from any module
            for alias in node.names:
                if alias.name == "*":
                    return ValidationResult(
                        valid=False,
                        error=f"Generated test uses wildcard import from '{module}'; rejecting for safety",
                    )

    # 4. Reject dangerous builtins at call-expression level
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _DANGEROUS_NAMES or name in _DANGEROUS_ATTRS:
                return ValidationResult(
                    valid=False,
                    error=f"Generated test uses potentially unsafe name '{name}'; rejecting for safety",
                )
        if isinstance(node, ast.Attribute) and node.attr in _DANGEROUS_ATTRS:
            return ValidationResult(
                valid=False,
                error=f"Generated test uses potentially unsafe attribute '{node.attr}'; rejecting for safety",
            )
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_ATTRS:
            return ValidationResult(
                valid=False,
                error=f"Generated test uses potentially unsafe name '{node.id}'; rejecting for safety",
            )

    return ValidationResult(valid=True, test_function_names=test_fns)


def compute_quality_score(code: str, target_symbol: str) -> tuple[float, str]:
    """Return a quality score 0–1 and explanation for a generated test.

    Higher scores indicate more useful, meaningful tests.
    Deductions applied for: no assertions, trivial assertions, no target reference.
    """
    notes: list[str] = []
    score = 1.0

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0, "Unparseable code"

    # Check for assertions
    assertions = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    raises_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("raises", "warns")
    ]

    if not assertions and not raises_calls:
        score -= 0.4
        notes.append("No assertions or pytest.raises found")
    else:
        # Check for trivially-true assertions like `assert True`
        trivial = sum(
            1 for a in assertions if isinstance(a.test, ast.Constant) and a.test.value is True
        )
        if trivial and trivial == len(assertions):
            score -= 0.3
            notes.append("Only trivial assertions (assert True)")

    # Check that the target symbol is referenced in the code
    if target_symbol:
        short_name = target_symbol.split(".")[-1]
        if short_name not in code:
            score -= 0.2
            notes.append(f"Target symbol '{short_name}' not referenced in test")

    # Penalise very short tests (< 3 lines of non-blank code)
    non_blank = [line for line in code.splitlines() if line.strip()]
    if len(non_blank) < 3:
        score -= 0.1
        notes.append("Test body is very short")

    score = round(max(0.0, min(1.0, score)), 2)
    explanation = "; ".join(notes) if notes else "Looks reasonable"
    return score, explanation

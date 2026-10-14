#!/usr/bin/env python3
"""Check shared plugin hooks against their canonical normalized AST hashes."""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

_EXPECTED_ENSURE_LLM_PROFILES_NORMALIZED_AST_SHA256 = (
    "bc002d63c8c5c3cb6fae10845013da18c6e8758ff2d4828ea6a38990712c6c29"
)
_EXPECTED_ENSURE_AGENT_PROFILES_NORMALIZED_AST_SHA256 = (
    "81cf8a503e249c29e4a67901b58004bde0037e4e7d969b3287f5087ca7e25152"
)
_EXPECTED_PROVENANCE_NORMALIZED_AST_SHA256 = (
    "129bc2a98d85c300026ef50dabe90940c4b3c0e7054fb02d52d1d1dee3b18672"
)
_EXPECTED_SAFETY_RAIL_NORMALIZED_AST_SHA256 = (
    "1a9f3f72fec383f046db2ca8805a7190c33daa86daf42f06c3cd27e3f8be245b"
)
_EXPECTED_RECORDS_NORMALIZED_AST_SHA256 = (
    "0c04e50f44a7a54f28843d853a183fe77506ee8f64182d54d11bf878ac867413"
)
_EXPECTED_REQUIRE_RECORDS_NORMALIZED_AST_SHA256 = (
    "1867b1e0d391da292c17b3f2f0c2aad9bc5744da1fbaf0554d1189c046e43cff"
)
EXPECTED: dict[str, str] = {
    "ensure_llm_profiles.py": _EXPECTED_ENSURE_LLM_PROFILES_NORMALIZED_AST_SHA256,
    "ensure_agent_profiles.py": _EXPECTED_ENSURE_AGENT_PROFILES_NORMALIZED_AST_SHA256,
    "_provenance.py": _EXPECTED_PROVENANCE_NORMALIZED_AST_SHA256,
    "safety_rail.py": _EXPECTED_SAFETY_RAIL_NORMALIZED_AST_SHA256,
    "_records.py": _EXPECTED_RECORDS_NORMALIZED_AST_SHA256,
    "require_records.py": _EXPECTED_REQUIRE_RECORDS_NORMALIZED_AST_SHA256,
}
REQUIRED = frozenset(
    {
        "ensure_llm_profiles.py",
        "ensure_agent_profiles.py",
        "safety_rail.py",
        "_records.py",
        "require_records.py",
    }
)
_DOCSTRING_NODE_TYPES = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


def _strip_docstrings(node: ast.AST) -> None:
    if (
        isinstance(node, _DOCSTRING_NODE_TYPES)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        node.body = node.body[1:]
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)


def _digest(path: Path, plugin_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    source = re.sub(rf"\b{re.escape(plugin_name)}\b", "PLUGIN", source)
    if plugin_name == "ux":
        source = re.sub(r"\bux_creator\b", "PLUGIN", source)
    tree = ast.parse(source, filename=str(path))
    _strip_docstrings(tree)
    dump = cast(Callable[..., str], ast.dump)
    dump_options: dict[str, bool] = {"include_attributes": False}
    if sys.version_info >= (3, 13):
        dump_options["show_empty"] = True
    normalized = dump(tree, **dump_options).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def main() -> int:
    # ast.dump's serialization schema is interpreter-version-bound; the EXPECTED
    # digests are generated on the supported 3.12-3.14 series, so comparisons
    # only run on those interpreters (the required CI legs). Newer canary
    # interpreters report a skip instead of a false mismatch.
    if sys.version_info[:2] not in {(3, 12), (3, 13), (3, 14)}:
        print(
            "check_shared_hooks: normalized AST digests are version-bound; "
            f"skipping on Python {sys.version_info.major}.{sys.version_info.minor} "
            "(enforced on the 3.12-3.14 legs)",
            file=sys.stderr,
        )
        return 0
    root = Path(__file__).resolve().parents[1]
    hook_paths = (root / "plugins").glob("*/hooks/scripts")
    hook_dirs = sorted(path for path in hook_paths if path.is_dir())
    if len(hook_dirs) != 1:
        print("expected one plugins/<name>/hooks/scripts directory", file=sys.stderr)
        return 1

    hooks_dir = hook_dirs[0]
    plugin_name = hooks_dir.parent.parent.name
    failures: list[str] = []
    for filename, expected in EXPECTED.items():
        path = hooks_dir / filename
        if not path.is_file():
            if filename in REQUIRED:
                failures.append(f"missing required shared hook: {path}")
            continue
        try:
            actual = _digest(path, plugin_name)
        except (OSError, UnicodeError, SyntaxError) as exc:
            failures.append(f"could not normalize {path}: {exc}")
            continue
        if actual != expected:
            failures.append(
                f"{filename}: expected {expected}, got {actual}; compare with "
                "the wire canonical and update all copies and EXPECTED together"
            )

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("shared hooks match canonical AST hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

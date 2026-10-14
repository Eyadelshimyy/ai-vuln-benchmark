"""The evaluation suite and the product must not depend on each other."""

from __future__ import annotations

import ast
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EVAL_ROOT.parents[1] / "src"


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_evaluation_code_never_imports_the_product() -> None:
    for path in sorted(EVAL_ROOT.rglob("*.py")):
        assert "gvas" not in _imported_roots(path), f"{path} imports the product"


def test_product_code_never_imports_the_evaluation_suite() -> None:
    for path in sorted(SRC_ROOT.rglob("*.py")):
        assert "field_notes" not in _imported_roots(path), f"{path} imports the eval suite"


def test_only_the_transport_layer_may_speak_http() -> None:
    """Cases, scoring, adapters, and manifests stay network-free.

    Live endpoint access is confined to ``transports/``, which is constructed only
    by an explicit ``--live`` run, so loading a manifest or scoring a record can
    never reach an endpoint.
    """
    forbidden = {"httpx", "requests", "urllib", "socket", "openai", "anthropic", "boto3"}
    allowed_dirs = {EVAL_ROOT / "transports"}
    for path in sorted(EVAL_ROOT.rglob("*.py")):
        if any(directory in path.parents for directory in allowed_dirs):
            continue
        if path.name == "test_openai_compatible_transport.py":
            continue
        offending = forbidden & _imported_roots(path)
        assert not offending, f"{path} imports {sorted(offending)}"


def test_transport_layer_holds_no_provider_sdk_or_credential_default() -> None:
    forbidden = {"requests", "openai", "anthropic", "boto3"}
    for path in sorted((EVAL_ROOT / "transports").rglob("*.py")):
        offending = forbidden & _imported_roots(path)
        assert not offending, f"{path} imports {sorted(offending)}"
        assert "os.environ" not in path.read_text(encoding="utf-8"), (
            f"{path} reads the environment directly; keys come from the manifest's "
            "named variables via EndpointConfig"
        )

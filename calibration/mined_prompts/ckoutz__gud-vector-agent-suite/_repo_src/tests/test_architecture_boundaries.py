import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src" / "gvas"
FORBIDDEN = {
    "gvas.infrastructure",
    "gvas.interfaces",
    "sqlalchemy",
    "fastapi",
    "slack_sdk",
    "twilio",
    "openai",
    "boto3",
}
ALLOWED_EXTERNAL = {"pydantic"}


def find_violations(layer: str, path: Path, source: str) -> list[str]:
    violations: list[str] = []
    lowered = source.lower()
    for forbidden in ("slack", "twilio"):
        if forbidden in lowered:
            violations.append(f"{path}: forbidden transport substring {forbidden}")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        imported: str | None = None
        if isinstance(node, ast.Import):
            imported = node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            imported = node.module
        if imported is None:
            continue
        if any(imported == denied or imported.startswith(f"{denied}.") for denied in FORBIDDEN):
            violations.append(f"{path}: forbidden import {imported}")
        top_level = imported.split(".", 1)[0]
        is_stdlib = top_level in sys.stdlib_module_names
        is_allowed_domain = imported == "gvas.domain" or imported.startswith("gvas.domain.")
        if not (is_stdlib or top_level in ALLOWED_EXTERNAL or is_allowed_domain):
            violations.append(f"{path}: disallowed import {imported}")
        if layer == "application" and imported.startswith("gvas.") and not is_allowed_domain:
            violations.append(f"{path}: non-domain import {imported}")
    return violations


def test_domain_application_boundaries() -> None:
    for layer in ("domain", "application"):
        for path in (ROOT / layer).rglob("*.py"):
            assert not find_violations(layer, path, path.read_text())


VENDOR_SDKS = {"boto3", "botocore", "slack_sdk", "twilio", "openai", "mypy_boto3_s3"}


def vendor_imports(source: str, path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        imported: str | None = None
        if isinstance(node, ast.Import):
            imported = node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            imported = node.module
        if imported is None:
            continue
        if imported.split(".", 1)[0] in VENDOR_SDKS:
            found.append(f"{path}: vendor import {imported}")
    return found


def test_vendor_sdk_imports_stay_inside_infrastructure() -> None:
    for path in ROOT.rglob("*.py"):
        if path.is_relative_to(ROOT / "infrastructure"):
            continue
        assert not vendor_imports(path.read_text(), path)


def test_object_storage_sdk_is_confined_to_its_adapter() -> None:
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "infrastructure").rglob("*.py")
        if vendor_imports(path.read_text(), path)
    }
    assert importers == {"infrastructure/object_storage.py"}


PROVIDER_NAMES = ("openai", "resend", "whisper", "railway", "httpx", "uvicorn")


def test_provider_names_stay_out_of_domain_and_application() -> None:
    """A port that names its provider has stopped being a port."""

    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for layer in ("domain", "application")
        for path in (ROOT / layer).rglob("*.py")
        for name in PROVIDER_NAMES
        if name in path.read_text().lower()
    ]
    assert offenders == []


def test_the_http_client_is_confined_to_infrastructure() -> None:
    importers = {
        path.relative_to(ROOT).parts[0]
        for path in ROOT.rglob("*.py")
        if "import httpx" in path.read_text()
    }
    assert importers <= {"infrastructure", "composition"}


def test_boundary_checker_catches_violations(tmp_path: Path) -> None:
    path = tmp_path / "invalid.py"
    violations = find_violations("application", path, "import slack_sdk\nvalue = 'slack'\n")
    assert any("forbidden transport substring slack" in violation for violation in violations)
    assert any("forbidden import slack_sdk" in violation for violation in violations)

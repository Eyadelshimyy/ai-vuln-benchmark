import ast
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "gvas"
# The production composition root chooses the channel, so it is the one module
# outside the adapter that is allowed to name Slack. Everything else stays
# channel-neutral.
SLACK_MODULES = frozenset(
    {
        Path("infrastructure/slack"),
        Path("interfaces/http/slack.py"),
        Path("composition/production.py"),
    }
)
WORKFLOW_MODULES = (
    "gvas.application.processing",
    "gvas.application.intents",
    "gvas.domain.intents",
    "gvas.domain.workflows",
)
INTENT_PREFIXES = ("quote:", "field notes:")


def _slack_sources() -> list[Path]:
    return sorted((SRC / "infrastructure" / "slack").rglob("*.py")) + [
        SRC / "interfaces" / "http" / "slack.py"
    ]


def _is_slack_module(path: Path) -> bool:
    relative = path.relative_to(SRC)
    return any(relative == module or module in relative.parents for module in SLACK_MODULES)


def test_slack_specifics_live_only_in_slack_modules() -> None:
    offenders = [
        path.relative_to(SRC)
        for path in SRC.rglob("*.py")
        if "slack" in path.read_text().lower() and not _is_slack_module(path)
    ]
    assert offenders == []


def test_slack_adapter_does_not_import_workflow_or_intent_modules() -> None:
    violations: list[str] = []
    for path in _slack_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.Import):
                imported = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                imported = node.module
            if imported is None:
                continue
            if any(
                imported == module or imported.startswith(f"{module}.")
                for module in WORKFLOW_MODULES
            ):
                violations.append(f"{path}: {imported}")
    assert violations == []


def test_slack_adapter_does_not_route_intent_prefixes() -> None:
    offenders = [
        path.name
        for path in _slack_sources()
        if any(prefix in path.read_text().lower() for prefix in INTENT_PREFIXES)
    ]
    assert offenders == []

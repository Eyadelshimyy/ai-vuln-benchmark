import os
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent


def fixture_text(*parts: str) -> str:
    return (FIXTURES.joinpath(*parts)).read_text(encoding="utf-8")


def mlx_integration_enabled() -> bool:
    return bool(os.environ.get("BUGFORGE_MLX_INTEGRATION"))

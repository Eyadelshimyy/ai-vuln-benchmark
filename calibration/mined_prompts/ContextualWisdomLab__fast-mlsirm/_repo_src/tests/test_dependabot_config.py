"""Supply-chain metadata contracts for Dependabot update policy."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_dependabot_ecosystem_has_explicit_cooldown() -> None:
    text = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    ecosystems = re.findall(r'^\s+- package-ecosystem:', text, flags=re.MULTILINE)

    assert len(ecosystems) == 5
    assert text.count("cooldown:\n      default-days: 7") == len(ecosystems)

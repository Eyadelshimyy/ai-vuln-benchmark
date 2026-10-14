"""Local Compound source for Phase 58. The tree is never vendored into BugForge."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

COMET_PIN = "f766f51583c23acc33b2a7824654ef2029a96804"
COMET_FILE = "contracts/CometWithExtendedAssetList.sol"
CLOSURE = (
    "contracts/CometWithExtendedAssetList.sol",
    "contracts/CometConfiguration.sol",
    "contracts/CometCore.sol",
    "contracts/CometMainInterface.sol",
    "contracts/CometMath.sol",
    "contracts/CometStorage.sol",
    "contracts/interfaces/IAssetList.sol",
    "contracts/interfaces/IAssetListFactory.sol",
    "contracts/interfaces/IAssetListFactoryHolder.sol",
    "contracts/interfaces/IERC20NonStandard.sol",
    "contracts/interfaces/IPriceFeed.sol",
)


def comet_root() -> Path:
    """The pinned checkout, or skip. A wrong commit fails closed."""
    raw = os.environ.get("BUGFORGE_COMET_ROOT", "/tmp/comet")
    root = Path(raw)
    target = root / COMET_FILE
    if not target.is_file():
        if os.environ.get("BUGFORGE_REQUIRE_COMET") == "1":
            pytest.fail(f"{target} is not available")
        pytest.skip("pinned Comet source is not available; set BUGFORGE_COMET_ROOT")
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if sha != COMET_PIN:
        pytest.fail(f"Comet checkout is {sha}, expected {COMET_PIN}")
    return root


def comet_sources(root: Path) -> dict[str, str]:
    missing = [path for path in CLOSURE if not (root / path).is_file()]
    if missing:
        pytest.fail(f"Comet closure is incomplete: {missing}")
    return {path: (root / path).read_text(encoding="utf-8") for path in CLOSURE}

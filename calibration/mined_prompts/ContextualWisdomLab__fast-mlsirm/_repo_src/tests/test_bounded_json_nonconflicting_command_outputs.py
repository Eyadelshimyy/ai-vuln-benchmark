"""Regression for bounded JSON at an isolated command-output boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(filename: str) -> ModuleType:
    """Load one repository script under a unique import identity."""
    module_name = f"bounded_nonconflicting_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_ignored_rust_shard_rejects_overdeep_metadata() -> None:
    """Cargo metadata must be depth-bounded before workspace traversal."""
    module = _load_script("run_ignored_rust_shard.py")
    nested: object = 0
    for _ in range(129):
        nested = [nested]
    payload = json.dumps(
        {
            "packages": [
                {
                    "id": "path+file:///repo/crate#example@0.1.0",
                    "name": "example",
                    "targets": [],
                }
            ],
            "workspace_members": ["path+file:///repo/crate#example@0.1.0"],
            "irrelevant": nested,
        }
    )

    with pytest.raises(ValueError, match="Cargo metadata output is not valid JSON"):
        module.parse_workspace_targets(payload)

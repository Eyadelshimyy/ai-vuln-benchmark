"""Shared helpers for Phase 51 tests. Nothing here runs a tool or touches a network."""

from __future__ import annotations

import copy

from app.discovery.bounty.campaign import BountyManifest
from tests.test_phase50.phase50_support import FIXTURES, MANIFEST_DATA

CALLER = "caller_context_vulnerable.sol"
CONTRACT = "SelfTrustMulticall"
FUNCTION = "multicall(bytes[])"


def manifest(**overrides: object) -> BountyManifest:
    data = copy.deepcopy(MANIFEST_DATA)
    data.update(overrides)
    return BountyManifest.from_dict(data)


def create_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "manifest": copy.deepcopy(MANIFEST_DATA),
        "repo_root": str(FIXTURES),
        "target": f"{CONTRACT}.multicall",
        "contract": CONTRACT,
        "function": FUNCTION,
        "source_file": CALLER,
        "files": [CALLER],
        "max_rounds": 12,
    }
    body.update(overrides)
    return body


__all__ = ["CALLER", "CONTRACT", "FUNCTION", "FIXTURES", "create_body", "manifest"]

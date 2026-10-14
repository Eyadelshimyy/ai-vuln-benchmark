"""Fail-closed target manifest for authorized testing.

A company name, a public URL, or a web asset does not authorize smart-contract
testing. Live testing stays off unless the manifest explicitly allows that
mode for that asset. This module does not start a live test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetManifest:
    """Operator-supplied scope. BugForge does not infer eligibility."""

    program_id: str
    asset_id: str
    target_type: str
    language: str = ""
    source: str = ""
    commit: str = ""
    snapshot: str = ""
    allowed_modes: tuple[str, ...] = ()
    active_testing: bool = False
    excluded: tuple[str, ...] = ()
    operator_approved: bool = False
    provenance: str = ""
    scope_version: str = ""
    timestamp: str = ""


def eligible_for_mode(manifest: TargetManifest | None, mode: str) -> tuple[bool, str]:
    """Return whether ``mode`` is explicitly allowed. Missing data fails closed."""
    if manifest is None:
        return False, "fail closed: no target manifest"
    requested = mode.strip().lower()
    if not requested or requested not in {item.strip().lower() for item in manifest.allowed_modes}:
        return False, "fail closed: testing mode is not declared for this target"
    if requested in {"live", "active"} and not (
        manifest.operator_approved and manifest.active_testing
    ):
        return False, "fail closed: active testing is not approved"
    if manifest.asset_id and manifest.asset_id in manifest.excluded:
        return False, "fail closed: asset is excluded"
    return True, "target mode is explicitly allowed"


def solidity_live_testing_permitted(manifest: TargetManifest | None) -> tuple[bool, str]:
    """Smart-contract live testing requires an explicit in-scope contract asset.

    A web or API target is not converted into contract authorization.
    """
    allowed, reason = eligible_for_mode(manifest, "live")
    if not allowed or manifest is None:
        return False, reason
    if manifest.target_type.strip().lower() != "smart_contract":
        return False, "fail closed: target type is not an in-scope smart contract"
    if manifest.language.strip().lower() not in {"", "solidity"}:
        return False, "fail closed: language is not solidity"
    return True, "smart contract is explicitly in scope for live testing"

"""Bounty campaign manifest: identity, scope, secrets, and unknowns."""

from __future__ import annotations

import pytest

from app.discovery.bounty.campaign import (
    BountyManifest,
    ManifestError,
    ScopeStatus,
)
from app.discovery.orchestration.model import CampaignIdentity
from tests.test_phase50.phase50_support import MANIFEST_DATA, manifest


def test_round_trip_and_stable_digest() -> None:
    first = manifest()
    again = BountyManifest.from_dict(first.to_dict())
    assert again == first
    assert again.identity_digest() == first.identity_digest()
    assert first.identity_digest().startswith("pc_")


def test_any_program_field_changes_the_context_identity() -> None:
    base = manifest().identity_digest()
    assert manifest(rules_version="2026-02").identity_digest() != base
    assert manifest(source_commit="def5678").identity_digest() != base


def test_unknown_stays_unknown() -> None:
    sparse = BountyManifest.from_dict({"platform": "hackerone", "program_id": "p"})
    gaps = sparse.gaps()
    for name in (
        "rules_version",
        "in_scope",
        "poc_requirement",
        "testing_mode",
        "compiler.version",
    ):
        assert name in gaps
    assert sparse.compiler.fingerprint() == ""
    assert sparse.fork_reference() == ""
    assert sparse.scope_of(contract="Vault").status is ScopeStatus.UNKNOWN


def test_existing_in_repo_does_not_make_a_contract_in_scope() -> None:
    mani = manifest()
    assert mani.scope_of(contract="SelfTrustMulticall").status is ScopeStatus.IN_SCOPE
    assert mani.scope_of(contract="SomeOtherContract").status is ScopeStatus.UNKNOWN
    assert mani.scope_of(file="caller_context_vulnerable.sol").status is ScopeStatus.IN_SCOPE


def test_explicit_exclusion_wins_over_inclusion() -> None:
    mani = manifest(
        in_scope=[{"kind": "path", "identifier": "src"}],
        out_of_scope=[{"kind": "contract", "identifier": "Vault"}],
    )
    assert mani.scope_of(contract="Vault", file="src/Vault.sol").status is ScopeStatus.OUT_OF_SCOPE


def test_address_scope_respects_the_chain() -> None:
    mani = manifest()
    address = "0x" + "11" * 20
    assert mani.scope_of(address=address, chain_id="1").status is ScopeStatus.IN_SCOPE
    assert mani.scope_of(address=address, chain_id="10").status is ScopeStatus.UNKNOWN


def test_campaign_identity_carries_the_program_context() -> None:
    mani = manifest()
    identity = mani.to_campaign_identity(contract="SelfTrustMulticall")
    assert identity.program_context == mani.identity_digest()
    assert identity.source_snapshot == "abc1234"
    assert identity.compiler_configuration == mani.compiler.fingerprint()
    assert identity.fork_reference == "1:19000000:snap-19000000"
    assert identity.deployment == "1:0x" + "11" * 20


def test_request_extra_never_carries_scope_or_approval() -> None:
    extra = manifest().request_extra()
    assert "scope" not in extra
    assert not any("approv" in key for key in extra)
    assert extra["program_context"] == manifest().identity_digest()


def test_legacy_identity_digest_is_unchanged_without_a_program() -> None:
    plain = CampaignIdentity(campaign_id="x", project="p", source_snapshot="s")
    with_program = CampaignIdentity(
        campaign_id="x", project="p", source_snapshot="s", program_context="pc_1"
    )
    assert plain.target_digest() != with_program.target_digest()
    assert (
        plain.target_digest()
        == CampaignIdentity(campaign_id="y", project="p", source_snapshot="s").target_digest()
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"private_key": "x"},
        {"rpc_api_key": "x"},
        {"program_url": "https://user:pw@example.test/x"},
        {"program_url": "https://example.test/x?token=abc"},
        {"notes": "0x" + "ab" * 32},
    ],
)
def test_credentials_are_refused(bad: dict[str, str]) -> None:
    with pytest.raises(ManifestError):
        BountyManifest.from_dict({**MANIFEST_DATA, **bad})


def test_malformed_and_oversized_manifests_are_rejected() -> None:
    with pytest.raises(ManifestError):
        BountyManifest.from_dict({"platform": "", "program_id": ""})
    with pytest.raises(ManifestError):
        manifest(deployments=[{"address": "nope", "chain_id": "1"}])
    with pytest.raises(ManifestError):
        manifest(in_scope=[{"kind": "contract", "identifier": f"C{i}"} for i in range(300)])
    with pytest.raises(ManifestError):
        manifest(impact_categories=[{"name": "x", "severity": "high", "weight": 500, "tags": []}])

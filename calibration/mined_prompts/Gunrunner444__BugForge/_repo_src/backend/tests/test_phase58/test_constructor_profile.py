"""Trusted constructor profiles fail closed unless every value is typed and bound."""

from __future__ import annotations

from app.discovery.bounty.constructor_profile import (
    ProfileError,
    parse_profile,
    prepare_profile,
    sha256_text,
)
from app.discovery.bounty.properties import build_property
from app.discovery.bounty.stateful import build_harness
from app.discovery.bounty.vfcs import ATTACKER, SequenceIdentity, Vfcs, VfcsCall
from app.parsing.solidity_research import build_research_model

VAULT = """// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;

contract PauseVault {
    struct Configuration {
        address governor;
        address pauseGuardian;
    }

    address public governor;
    address public pauseGuardian;
    uint8 public pauseFlags;

    constructor(Configuration memory config) {
        governor = config.governor;
        pauseGuardian = config.pauseGuardian;
    }

    function pause(
        bool supplyPaused,
        bool transferPaused,
        bool withdrawPaused,
        bool absorbPaused,
        bool buyPaused
    ) external {
        if (msg.sender != governor && msg.sender != pauseGuardian) revert("unauthorized");
        pauseFlags =
            (supplyPaused ? 1 : 0) |
            (transferPaused ? 2 : 0) |
            (withdrawPaused ? 4 : 0) |
            (absorbPaused ? 8 : 0) |
            (buyPaused ? 16 : 0);
    }

    function isSupplyPaused() public view returns (bool) {
        return (pauseFlags & 1) != 0;
    }
}
"""

FILE = "PauseVault.sol"
STATEMENT = (
    "an account that is neither the configured governor nor the configured "
    "pause guardian must not change the protocol pause flags"
)


def _profile(**overrides: object) -> dict[str, object]:
    digest = sha256_text(VAULT)
    raw: dict[str, object] = {
        "profile_id": "pause-vault-authorization",
        "contract": "PauseVault",
        "source_file": FILE,
        "source_sha256": digest,
        "parameter": "config",
        "model_type": "Configuration",
        "solidity_type": "PauseVault.Configuration",
        "struct_file": FILE,
        "struct_name": "Configuration",
        "value": {
            "kind": "struct",
            "type_name": "PauseVault.Configuration",
            "fields": [
                {"name": "governor", "value": {"kind": "role", "value": "governor"}},
                {
                    "name": "pauseGuardian",
                    "value": {"kind": "role", "value": "pause_guardian"},
                },
            ],
        },
        "fixtures": [],
        "imports": [],
        "bound_files": [{"path": FILE, "sha256": digest}],
        "assumptions": ["governor and pauseGuardian are local harness addresses"],
        "property_function": "pause(bool,bool,bool,bool,bool)",
        "property_statement": STATEMENT,
        "property_getter": "isSupplyPaused()",
    }
    raw.update(overrides)
    return raw


def _sequence() -> Vfcs:
    signature = "pause(bool,bool,bool,bool,bool)"
    call = VfcsCall(
        "PauseVault",
        signature,
        "execute",
        ATTACKER,
        arguments=(
            ("supplyPaused", "true"),
            ("transferPaused", "false"),
            ("withdrawPaused", "false"),
            ("absorbPaused", "false"),
            ("buyPaused", "false"),
        ),
    )
    return Vfcs(
        sequence_id="vf_pause_vault",
        template="unauthorized-call",
        origin="constructor_profile:pause-vault-authorization",
        calls=(call,),
        property_under_test=STATEMENT,
        derived_from=f"profile.authorization@PauseVault.{signature}",
        identity=SequenceIdentity(campaign_id="phase58"),
    )


def test_profile_encodes_roles_and_rejects_a_missing_field() -> None:
    sources = {FILE: VAULT}
    prepared = prepare_profile(parse_profile(_profile()), sources)
    assert "governor: address(0xC0FE)" in prepared.expression
    assert "pauseGuardian: address(0x6A2D)" in prepared.expression
    assert "PauseVault.Configuration(" in prepared.expression
    assert prepared.deploy_lines == ()
    dropped = _profile()
    fields = dropped["value"]["fields"]  # type: ignore[index]
    dropped["value"]["fields"] = fields[:1]  # type: ignore[index]
    try:
        prepare_profile(parse_profile(dropped), sources)
    except ProfileError as exc:
        assert exc.code == "profile_struct_mismatch"
    else:
        raise AssertionError("a missing struct field was accepted")


def test_nested_struct_and_empty_array_bind_fixture_addresses() -> None:
    text = """// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;
contract PauseVault {
    struct Inner { address who; }
    struct Configuration {
        address governor;
        Inner extension;
        address baseToken;
        AssetConfig[] assetConfigs;
    }
    constructor(Configuration memory config) { governor = config.governor; }
    address public governor;
}
"""
    digest = sha256_text(text)
    raw = _profile(
        source_sha256=digest,
        bound_files=[{"path": FILE, "sha256": digest}],
        fixtures=[
            {"fixture_id": "baseToken", "kind": "decimals_token", "decimals": 6},
            {"fixture_id": "factory", "kind": "asset_list_factory"},
            {"fixture_id": "holder", "kind": "factory_holder", "depends_on": "factory"},
        ],
        imports=[
            "contracts/CometCore.sol",
            "contracts/interfaces/IAssetListFactory.sol",
            "contracts/interfaces/IAssetListFactoryHolder.sol",
        ],
        value={
            "kind": "struct",
            "type_name": "PauseVault.Configuration",
            "fields": [
                {"name": "governor", "value": {"kind": "role", "value": "governor"}},
                {
                    "name": "extension",
                    "value": {
                        "kind": "struct",
                        "type_name": "PauseVault.Inner",
                        "fields": [
                            {"name": "who", "value": {"kind": "fixture", "value": "holder"}},
                        ],
                    },
                },
                {"name": "baseToken", "value": {"kind": "fixture", "value": "baseToken"}},
                {
                    "name": "assetConfigs",
                    "value": {
                        "kind": "array",
                        "type_name": "AssetConfig",
                        "items": [],
                    },
                },
            ],
        },
    )
    sources = {
        FILE: text,
        "contracts/CometCore.sol": "contract CometCore {}",
        "contracts/interfaces/IAssetListFactory.sol": "interface IAssetListFactory {}",
        "contracts/interfaces/IAssetListFactoryHolder.sol": "interface IAssetListFactoryHolder {}",
    }
    prepared = prepare_profile(parse_profile(raw), sources)
    assert "governor: address(0xC0FE)" in prepared.expression
    assert "who: address(bf_holder)" in prepared.expression
    assert "baseToken: address(bf_baseToken)" in prepared.expression
    assert "assetConfigs: new AssetConfig[](0)" in prepared.expression
    assert "BugForgeHolder_holder bf_holder = new BugForgeHolder_holder(address(bf_factory));" in (
        "\n".join(prepared.deploy_lines)
    )
    assert "function decimals() external view returns (uint8)" in prepared.contracts
    assert "return 6" not in prepared.contracts
    assert "d = 6" in prepared.contracts


def test_malformed_profiles_and_executable_expressions_are_rejected() -> None:
    with_expression = _profile(expression="ffi()")
    try:
        parse_profile(with_expression)
    except ProfileError as exc:
        assert exc.code == "profile_malformed"
    else:
        raise AssertionError("an extra expression field was accepted")
    poisoned = _profile()
    poisoned["value"]["fields"][0]["value"] = {  # type: ignore[index]
        "kind": "uint",
        "value": "1; selfdestruct(address(0))",
    }
    try:
        parse_profile(poisoned)
    except ProfileError as exc:
        assert exc.code == "profile_malformed"
    else:
        raise AssertionError("a uint expression was accepted")
    try:
        parse_profile(_profile(solidity_type="PauseVault.Configuration; ffi()"))
    except ProfileError as exc:
        assert exc.code == "profile_malformed"
    else:
        raise AssertionError("a type expression was accepted")
    try:
        parse_profile(_profile(assumptions=["run ffi();"]))
    except ProfileError as exc:
        assert exc.code == "profile_malformed"
    else:
        raise AssertionError("code in an assumption was accepted")
    attacker = _profile()
    attacker["value"]["fields"][0]["value"] = {"kind": "role", "value": "attacker"}  # type: ignore[index]
    try:
        parse_profile(attacker)
    except ProfileError as exc:
        assert exc.code == "profile_role_conflict"
    else:
        raise AssertionError("the attacker was accepted as governor")
    filled = _profile(
        value={
            "kind": "array",
            "type_name": "PauseVault.Configuration",
            "items": [{"kind": "uint", "value": "1"}],
        }
    )
    try:
        parse_profile(filled)
    except ProfileError as exc:
        assert exc.code == "profile_malformed"
    else:
        raise AssertionError("a non-empty array was accepted")


def test_profile_identity_must_match_the_bound_source() -> None:
    sources = {FILE: VAULT}
    profile = parse_profile(_profile())
    changed = {FILE: VAULT + "\n"}
    try:
        prepare_profile(profile, changed)
    except ProfileError as exc:
        assert exc.code == "profile_source_mismatch"
    else:
        raise AssertionError("a changed snapshot reused the profile")
    other = _profile(contract="OtherVault")
    # The bound file still hashes PauseVault. The profile names a different contract.
    parsed = parse_profile(other)
    assert parsed.contract == "OtherVault"
    model = build_research_model(sources)
    sequence = _sequence()
    harness = build_harness(
        sequence,
        model,
        oracle=build_property(sequence, model).oracle,
        sources=sources,
        profile=parsed,
    )
    assert harness.reason_code == "unknown_constructor_argument"
    try:
        parse_profile(_profile(source_sha256="0" * 64))
    except ProfileError as exc:
        assert exc.code == "profile_source_mismatch"
    else:
        raise AssertionError("a profile whose hash is not bound was accepted")


def test_unknown_struct_stays_rejected_without_a_profile() -> None:
    sources = {FILE: VAULT}
    model = build_research_model(sources)
    sequence = _sequence()
    bare = build_harness(
        sequence, model, oracle=build_property(sequence, model).oracle, sources=sources
    )
    assert bare.reason_code == "unknown_constructor_argument"
    approved = build_harness(
        sequence,
        model,
        oracle=build_property(sequence, model).oracle,
        sources=sources,
        profile=parse_profile(_profile()),
    )
    assert approved.buildable, approved.reason
    assert "new PauseVault(PauseVault.Configuration(" in approved.source
    assert "address(0xC0FE)" in approved.source
    assert "vm.prank(address(0xA11CE))" in approved.source
    assert "true" in approved.source
    assert "ffi(" not in approved.source


def test_ambiguous_dependency_is_not_resolved_by_a_profile() -> None:
    text = VAULT.replace(
        "constructor(Configuration memory config) {\n"
        "        governor = config.governor;\n"
        "        pauseGuardian = config.pauseGuardian;\n"
        "    }",
        "IReg public reg;\n"
        "    constructor(Configuration memory config, IReg registry) {\n"
        "        governor = config.governor;\n"
        "        pauseGuardian = config.pauseGuardian;\n"
        "        reg = registry;\n"
        "    }",
    )
    text += (
        "\ninterface IReg { function ok(address who) external view returns (bool); }\n"
        "contract OpenReg is IReg {\n"
        "    function ok(address) external pure returns (bool) { return true; }\n"
        "}\n"
        "contract ClosedReg is IReg {\n"
        "    function ok(address) external pure returns (bool) { return false; }\n"
        "}\n"
    )
    sources = {FILE: text}
    model = build_research_model(sources)
    sequence = _sequence()
    digest = sha256_text(text)
    profile = parse_profile(
        _profile(source_sha256=digest, bound_files=[{"path": FILE, "sha256": digest}])
    )
    harness = build_harness(
        sequence,
        model,
        oracle=build_property(sequence, model).oracle,
        sources=sources,
        profile=profile,
    )
    assert harness.reason_code == "ambiguous_dependency"


def test_profile_serialization_is_stable() -> None:
    first = parse_profile(_profile())
    second = parse_profile(first.to_dict())
    assert first.to_dict() == second.to_dict()
    sources = {FILE: VAULT}
    assert prepare_profile(first, sources).expression == prepare_profile(second, sources).expression
    assert first.public_dict()["local_environment"] is True
    assert first.public_dict()["establishes_deployed_state"] is False
    assert first.public_dict()["validation"] == "accepted"

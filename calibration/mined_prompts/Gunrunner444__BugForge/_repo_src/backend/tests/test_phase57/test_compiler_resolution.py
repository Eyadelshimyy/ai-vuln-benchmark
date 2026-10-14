"""Pinned Solidity compilers are selected by version, never by source text."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.discovery.bounty.solc_resolve import (
    distribution_platforms,
    exact_version,
    reported_version,
    resolve_solc,
    verify_sha256,
)
from app.discovery.bounty.stateful import ProjectSettings, StatefulExecutor, foundry_config
from tests.test_phase52.phase52_support import requires_forge


def test_exact_version_rejects_paths_and_ranges() -> None:
    assert exact_version("0.8.15") == "0.8.15"
    assert exact_version(" 0.8.15 ") == "0.8.15"
    assert exact_version("^0.8.15") == ""
    assert exact_version("0.8.15; rm -rf /") == ""
    assert exact_version("../0.8.15") == ""
    assert exact_version("/usr/bin/solc") == ""
    assert exact_version("unknown") == ""


def test_platform_mapping_does_not_use_a_linux_binary_on_macos() -> None:
    assert distribution_platforms("Darwin", "arm64") == ("macosx-aarch64", "macosx-amd64")
    assert distribution_platforms("Linux", "x86_64") == ("linux-amd64",)
    assert distribution_platforms("Linux", "aarch64") == ("linux-aarch64",)
    assert "linux-amd64" not in distribution_platforms("Darwin", "arm64")


def test_checksum_helper_accepts_only_the_published_digest() -> None:
    data = b"solc-bytes"
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    assert verify_sha256(data, digest)
    assert verify_sha256(data, "0x" + digest)
    assert not verify_sha256(data, "0" * 64)
    assert not verify_sha256(data, "not-a-checksum")


def test_a_missing_version_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BUGFORGE_SOLC_ROOT", str(tmp_path))
    # 0.8.99 is not installed. The ambient 0.8.28 compiler must not be returned.
    assert resolve_solc("0.8.99") is None
    assert resolve_solc("../0.8.28") is None


def test_versioned_binary_is_selected_and_a_symlink_escape_is_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "compilers"
    binary = root / "9.9.9" / "solc"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'Version: 9.9.9'\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("BUGFORGE_SOLC_ROOT", str(root))
    resolved = resolve_solc("9.9.9")
    assert resolved is not None
    assert resolved.reported_version == "9.9.9"
    assert Path(resolved.path) == binary.resolve()
    assert reported_version(binary) == "9.9.9"

    outside = tmp_path / "outside" / "solc"
    outside.parent.mkdir()
    outside.write_text("#!/bin/sh\necho 'Version: 9.9.8'\n", encoding="utf-8")
    outside.chmod(0o755)
    link_dir = root / "9.9.8"
    link_dir.mkdir()
    (link_dir / "solc").symlink_to(outside)
    assert resolve_solc("9.9.8") is None


def test_project_settings_reach_the_harness_config_and_cannot_inject() -> None:
    config = foundry_config(
        "default",
        "/opt/solc-0.8.15",
        ProjectSettings(
            optimizer=True,
            optimizer_runs=1,
            via_ir=True,
            evm_version="london",
            evm_exact=True,
        ),
    )
    assert 'solc = "/opt/solc-0.8.15"' in config
    assert "optimizer = true" in config
    assert "optimizer_runs = 1" in config
    assert "via_ir = true" in config
    assert 'evm_version = "london"' in config
    assert "ffi = false" in config
    assert "offline = true" in config
    rejected = foundry_config(
        "default",
        "/opt/solc",
        ProjectSettings(evm_version='london"\nffi = true', evm_exact=True),
    )
    assert "ffi = true" not in rejected
    assert rejected.count("ffi =") == 1
    omitted = foundry_config(
        "default",
        "/opt/solc",
        ProjectSettings(evm_version="cancun", evm_exact=False, evm_requested="cancun"),
    )
    assert "cancun" not in omitted


@requires_forge
def test_pinned_missing_compiler_is_not_a_missing_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.discovery.bounty.service as service_module
    from app.core.config import get_settings
    from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
    from tests.test_phase51.phase51_support import manifest

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Acct.sol").write_text(
        """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract Acct {
    address public owner;
    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) { return 0; }
    function initialize(address newOwner) external { owner = newOwner; }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=manifest(
                compiler={
                    "version": "0.8.99",
                    "optimizer": "enabled",
                    "optimizer_runs": "1",
                    "via_ir": "true",
                    "evm_version": "london",
                },
                in_scope=[{"kind": "contract", "identifier": "Acct"}],
                out_of_scope=[],
                known_issues=[],
                source_commit="phase57-missing-solc",
            ),
            repo_root=repo,
            contract="Acct",
            source_file="Acct.sol",
            files=("Acct.sol",),
        )
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    summary = result["campaign_summary"]
    assert result["forge_runs"] == 0
    assert summary["foundry_invoked"] is False
    assert summary["execution_stages"]["compiler_unavailable"] >= 1
    assert summary["execution_stages"]["tool_unavailable"] == 0
    assert summary["property_evaluation_demonstrated"] is False
    assert summary["blocker"] == (
        "the required Solidity compiler is not installed, so the test was not run"
    )
    assert any("0.8.99" in str(item.get("reason") or "") for item in result["observations"])
    assert "unavailable" in result["outcomes"]


@requires_forge
def test_solc_0_8_15_is_the_compiler_that_evaluates_the_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.discovery.bounty.service as service_module
    from app.core.config import get_settings
    from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
    from tests.test_phase51.phase51_support import manifest

    if resolve_solc("0.8.15") is None:
        if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
            pytest.fail("the Foundry job requires solc 0.8.15")
        pytest.skip("solc 0.8.15 is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Acct.sol").write_text(
        """// SPDX-License-Identifier: MIT
pragma solidity 0.8.15;
contract Acct {
    address public owner;
    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) { return 0; }
    function initialize(address newOwner) external { owner = newOwner; }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=manifest(
                compiler={
                    "version": "0.8.15",
                    "optimizer": "enabled",
                    "optimizer_runs": "1",
                    "via_ir": "true",
                    "evm_version": "london",
                },
                in_scope=[{"kind": "contract", "identifier": "Acct"}],
                out_of_scope=[],
                known_issues=[],
                source_commit="phase57-solc-0.8.15",
            ),
            repo_root=repo,
            contract="Acct",
            source_file="Acct.sol",
            files=("Acct.sol",),
        )
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    summary = result["campaign_summary"]
    runs = svc.stateful_status(campaign.campaign_id)["runs"]
    assert runs
    tools = runs[-1]["tools"]
    assert tools["solc_selected_version"] == "0.8.15"
    assert tools["solc_version"] == "0.8.15"
    assert tools["compiler_required"] == "0.8.15"
    assert tools["optimizer"] == "true"
    assert tools["optimizer_runs"] == "1"
    assert tools["via_ir"] == "true"
    assert tools["evm_version"] == "london"
    assert tools["evm_exact"] == "true"
    assert Path(tools["solc_path"]).name == "solc"
    assert result["forge_runs"] > 0
    assert summary["foundry_invoked"] is True
    assert summary["property_evaluation_demonstrated"] is True
    assert summary["dynamic_foundry_execution_demonstrated"] is True
    assert summary["execution_stages"]["compiler_unavailable"] == 0
    assert summary["execution_stages"]["tool_unavailable"] == 0
    assert summary["verified"] is False
    assert result["outcomes"].get("property_violated", 0) >= 1


def test_cancun_is_not_claimed_when_solc_0_8_15_does_not_list_it() -> None:
    compiler = resolve_solc("0.8.15")
    if compiler is None:
        if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
            pytest.fail("the Foundry job requires solc 0.8.15")
        pytest.skip("solc 0.8.15 is not installed")
    executor = StatefulExecutor(
        compiler=compiler,
        compiler_required="0.8.15",
        project=ProjectSettings(
            optimizer=True,
            optimizer_runs=1,
            via_ir=True,
            evm_version="cancun",
            evm_exact=True,
            evm_requested="cancun",
        ),
    )
    assert executor.evm_exact is False
    assert executor.evm_requested == "cancun"
    assert executor.project is not None
    assert executor.project.evm_version == ""
    config = foundry_config("default", compiler.path, executor.project)
    assert "cancun" not in config
    assert "optimizer = true" in config
    assert "optimizer_runs = 1" in config
    assert "via_ir = true" in config
    assert f'solc = "{compiler.path}"' in config


@requires_forge
def test_a_pragma_mismatch_stops_before_forge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.discovery.bounty.service as service_module
    from app.core.config import get_settings
    from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
    from tests.test_phase51.phase51_support import manifest

    if resolve_solc("0.8.15") is None:
        if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
            pytest.fail("the Foundry job requires solc 0.8.15")
        pytest.skip("solc 0.8.15 is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Acct.sol").write_text(
        """// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;
contract Acct {
    address public owner;
    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) { return 0; }
    function initialize(address newOwner) external { owner = newOwner; }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=manifest(
                compiler={
                    "version": "0.8.15",
                    "optimizer": "enabled",
                    "optimizer_runs": "1",
                    "via_ir": "true",
                    "evm_version": "london",
                },
                in_scope=[{"kind": "contract", "identifier": "Acct"}],
                out_of_scope=[],
                known_issues=[],
                source_commit="phase57-pragma-mismatch",
            ),
            repo_root=repo,
            contract="Acct",
            source_file="Acct.sol",
            files=("Acct.sol",),
        )
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    summary = result["campaign_summary"]
    assert result["forge_runs"] == 0
    assert summary["foundry_invoked"] is False
    assert summary["property_evaluation_demonstrated"] is False
    assert summary["execution_stages"]["compiler_unavailable"] >= 1
    assert summary["execution_stages"]["tool_unavailable"] == 0
    assert "pragma" in summary["blocker"]
    assert "not installed" not in summary["blocker"]
    assert any(
        "0.8.15" in str(item.get("reason") or "") and "not installed" not in str(item.get("reason"))
        for item in result["observations"]
    )

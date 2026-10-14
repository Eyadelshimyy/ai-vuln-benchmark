"""Source reads and dependency status stay visible and do not over-claim completeness."""

from __future__ import annotations

from pathlib import Path

from app.discovery.bounty.research_ledger import campaign_coverage
from app.discovery.bounty.source_selection import (
    MAX_DEPENDENCY_DEPTH,
    MAX_DEPENDENCY_FILES,
    MAX_FILE_BYTES,
    select_named,
    select_sources,
)
from app.parsing.solidity_research import build_research_model


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_closed_selected_set_is_traced(tmp_path: Path) -> None:
    _write(tmp_path, "src/Token.sol", "contract Token { function ping() external {} }")
    selection = select_sources(tmp_path, focus_contract="Token")
    assert selection.dependency_status() == "traced"
    assert selection.files_read == 1
    assert selection.files_not_read == 0
    assert selection.reads[0].reason == "read"
    summary = campaign_coverage(
        files_discovered=selection.considered,
        files_selected=len(selection.selected),
        files_read=selection.files_read,
        dependency_status=selection.dependency_status(),
        repository_candidates=selection.repository_candidates,
        excluded_count=selection.excluded_count,
        selection_mode=selection.mode,
        candidates=1,
        sequences=1,
    )
    assert summary["analysis_complete"] is True
    assert summary["snapshot_fresh"] is True
    assert summary["repository_coverage"] == "complete"
    assert summary["dependency_status"] == "traced"


def test_a_missing_dependency_is_partial(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/Vault.sol",
        'import "./Missing.sol";\ncontract Vault { function f() external {} }\n',
    )
    selection = select_sources(tmp_path, focus_contract="Vault")
    assert selection.unresolved_imports
    assert selection.dependency_status() == "partial"
    summary = campaign_coverage(
        files_discovered=selection.considered,
        files_selected=len(selection.selected),
        files_read=selection.files_read,
        dependency_status=selection.dependency_status(),
        selection_mode=selection.mode,
        candidates=0,
        sequences=0,
    )
    assert summary["analysis_complete"] is False
    assert summary["repository_coverage"] != "complete"
    assert "dependencies" in summary["blocker"]


def test_an_ambiguous_import_is_partial(tmp_path: Path) -> None:
    _write(tmp_path, "src/a/Token.sol", "contract TokenA { uint x; }")
    _write(tmp_path, "src/b/Token.sol", "contract TokenB { uint y; }")
    _write(
        tmp_path,
        "src/Vault.sol",
        'import "Token.sol";\ncontract Vault { function f() external {} }\n',
    )
    selection = select_sources(tmp_path, focus_contract="Vault")
    assert selection.ambiguous_imports
    assert selection.dependency_status() == "partial"


def test_an_import_that_escapes_the_repository_is_partial(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/Vault.sol",
        'import "../../outside.sol";\ncontract Vault { function f() external {} }\n',
    )
    selection = select_sources(tmp_path, focus_contract="Vault")
    assert selection.escaped_imports
    assert selection.dependency_status() == "partial"
    assert all(
        item.path == "src/Vault.sol" or not item.path.startswith("..") for item in selection.ranked
    )


def test_a_dependency_traversal_limit_is_bounded(tmp_path: Path) -> None:
    previous = "src/L0.sol"
    _write(tmp_path, previous, "contract L0 { function f() external {} }")
    for index in range(1, MAX_DEPENDENCY_DEPTH + 3):
        name = f"src/L{index}.sol"
        _write(
            tmp_path,
            name,
            f'import "./L{index - 1}.sol";\ncontract L{index} {{ uint x; }}\n',
        )
        previous = name
    leaf = f"L{MAX_DEPENDENCY_DEPTH + 2}"
    selection = select_sources(tmp_path, focus_contract=leaf)
    assert selection.dependency_notes
    assert selection.dependency_status() == "bounded"
    summary = campaign_coverage(
        files_discovered=selection.considered,
        files_selected=len(selection.selected),
        files_read=selection.files_read,
        truncated=selection.truncated,
        dependency_status=selection.dependency_status(),
        candidates=1,
        sequences=0,
    )
    assert summary["analysis_complete"] is False
    assert summary["repository_coverage"] == "bounded"


def test_a_dependency_file_cap_is_bounded(tmp_path: Path) -> None:
    imports = "\n".join(f'import "./D{index}.sol";' for index in range(MAX_DEPENDENCY_FILES + 2))
    _write(tmp_path, "src/Root.sol", imports + "\ncontract Root { function f() external {} }\n")
    for index in range(MAX_DEPENDENCY_FILES + 2):
        _write(tmp_path, f"src/D{index}.sol", f"contract D{index} {{ uint x; }}")
    selection = select_sources(tmp_path, focus_contract="Root")
    assert any("stopped" in note for note in selection.dependency_notes)
    assert selection.dependency_status() == "bounded"


def test_a_parser_limit_is_not_a_complete_analysis() -> None:
    sources = {
        f"C{index}.sol": f"contract C{index} {{ function f() external {{}} }}" for index in range(3)
    }
    model = build_research_model(sources, max_files=1)
    assert model.truncated is True
    assert model.notes
    summary = campaign_coverage(
        files_discovered=3,
        files_selected=3,
        files_read=3,
        truncated=True,
        parser_limited=True,
        selection_truncated=False,
        truncation_notes=model.notes,
        dependency_status="traced",
        candidates=0,
        sequences=0,
    )
    assert summary["parser_limited"] is True
    assert summary["analysis_complete"] is False
    assert summary["repository_coverage"] == "bounded"
    assert "parser" in summary["blocker"]


def test_an_intentional_partial_analysis_is_not_repository_coverage(tmp_path: Path) -> None:
    _write(tmp_path, "Target.sol", "contract Target { function f() external {} }")
    _write(tmp_path, "Other.sol", "contract Other { uint x; }")
    selection = select_named(tmp_path, ("Target.sol",))
    assert selection.mode == "explicit"
    assert selection.dependency_status() == "traced"
    assert selection.repository_candidates == 2
    summary = campaign_coverage(
        files_discovered=selection.considered,
        files_selected=len(selection.selected),
        files_read=selection.files_read,
        dependency_status=selection.dependency_status(),
        selection_mode=selection.mode,
        repository_candidates=selection.repository_candidates,
        excluded_count=selection.excluded_count,
        candidates=1,
        sequences=1,
    )
    assert summary["analysis_complete"] is True
    assert summary["repository_coverage"] == "selected_subset"
    assert summary["snapshot_fresh"] is True


def test_oversized_unreadable_and_missing_files_keep_their_reasons(tmp_path: Path) -> None:
    _write(tmp_path, "Ok.sol", "contract Ok { function f() external {} }")
    big = tmp_path / "Big.sol"
    big.write_bytes(b"contract Big { uint x; }\n" + b" " * (MAX_FILE_BYTES + 8))
    (tmp_path / "Weird.sol").write_bytes(b"\xff\xfe not text")
    (tmp_path / "test").mkdir()
    _write(tmp_path, "test/Hidden.sol", "contract Hidden { function f() external {} }")
    outside = tmp_path.parent / "phase56-outside.sol"
    outside.write_text("contract Outside {}", encoding="utf-8")
    link = tmp_path / "Escape.sol"
    link.symlink_to(outside)
    selection = select_sources(tmp_path, focus_file="Ok.sol")
    reasons = {item.path: item.reason for item in selection.reads}
    assert reasons["Ok.sol"] == "read"
    assert reasons["Big.sol"] == "size_limit"
    assert reasons["Weird.sol"] == "invalid_encoding"
    assert reasons["Escape.sol"] == "path_rejected"
    assert "test/Hidden.sol" not in reasons
    assert selection.excluded_count >= 1
    assert selection.files_read == 1
    assert "Big.sol" not in {
        path for path, record_reason in reasons.items() if record_reason == "read"
    }
    named = select_named(tmp_path, ("Gone.sol", "../phase56-outside.sol", "notes.txt"))
    named_reasons = {item.path: item.reason for item in named.reads}
    assert named_reasons["Gone.sol"] == "disappeared"
    assert "path_rejected" in named_reasons.values()
    outside.unlink(missing_ok=True)
    summary = campaign_coverage(
        files_discovered=selection.considered,
        files_selected=len(selection.selected),
        files_read=1,
        dependency_status=selection.dependency_status(),
        unreadable_files=tuple(item.as_dict() for item in selection.reads if item.status != "read"),
        candidates=0,
        sequences=0,
    )
    assert summary["analysis_complete"] is False
    assert summary["dependency_status"] == "partial"
    assert any(item["reason"] == "size_limit" for item in summary["unreadable_files"])
    assert summary["dynamic_foundry_execution_demonstrated"] is False


def test_foundry_scripts_are_excluded_and_cannot_poison_selection(tmp_path: Path) -> None:
    _write(tmp_path, "src/Vault.sol", "contract Vault { function f() external {} }")
    _write(
        tmp_path,
        "forge/script/Deploy.sol",
        'import "forge-std/Script.sol";\ncontract Deploy is Script { function run() external { vm.startBroadcast(); } }\n',
    )
    selection = select_sources(tmp_path, focus_contract="Vault")
    assert "src/Vault.sol" in selection.selected
    assert all("script" not in path.split("/") for path in selection.selected)
    assert selection.excluded_count >= 1


def test_an_out_of_scope_path_is_excluded_before_execution(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.discovery.bounty.service as service_module
    from app.core.config import get_settings
    from app.discovery.bounty.campaign import ScopeStatus
    from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
    from tests.test_phase51.phase51_support import manifest

    account = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract {name} {{
    address public owner;
    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) {{ return 0; }}
    function initialize(address newOwner) external {{ owner = newOwner; }}
}}
"""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "vendor").mkdir()
    (repo / "src" / "Good.sol").write_text(account.format(name="Good"), encoding="utf-8")
    (repo / "vendor" / "Bad.sol").write_text(account.format(name="Bad"), encoding="utf-8")
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
                in_scope=[{"kind": "contract", "identifier": "Good"}],
                out_of_scope=[{"kind": "path", "identifier": "vendor"}],
                known_issues=[],
                source_commit="phase56-scope",
            ),
            repo_root=repo,
            contract="Good",
            source_file="src/Good.sol",
        )
    )
    assert "vendor/Bad.sol" not in campaign.selection.selected
    assert campaign.selection.role_of("vendor/Bad.sol") == "excluded"
    assert campaign.selection.role_of("src/Good.sol") == "target"
    analysis = svc._analysis(campaign)
    bad = [item for item in analysis.sequences if "Bad." in item.derived_from]
    good = [item for item in analysis.sequences if "Good." in item.derived_from]
    assert not bad
    assert good
    decision = campaign.manifest.scope_of(contract="Bad", file="vendor/Bad.sol")
    assert decision.status is ScopeStatus.OUT_OF_SCOPE


def test_permission_errors_are_recorded(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _write(tmp_path, "Secret.sol", "contract Secret { function f() external {} }")

    def denied(self: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)
    selection = select_sources(tmp_path, focus_file="Secret.sol")
    assert selection.reads
    assert all(item.reason == "filesystem_error" for item in selection.reads)
    assert selection.files_read == 0
    assert selection.dependency_status() == "partial"

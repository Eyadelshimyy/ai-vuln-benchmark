"""Import and inheritance selection stays inside the repo and reports uncertainty."""

from __future__ import annotations

from pathlib import Path

from app.discovery.bounty.research_ledger import campaign_coverage
from app.discovery.bounty.source_selection import Priority, select_sources


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_relative_imports_and_unique_bases_are_selected_with_their_paths(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/access/Guard.sol",
        "contract Guard { address public owner; modifier onlyOwner() { _; } }",
    )
    _write(
        tmp_path, "src/interfaces/IAccount.sol", "interface IAccount { function ping() external; }"
    )
    _write(
        tmp_path,
        "src/account/Account.sol",
        """
        import "../access/Guard.sol";
        import "../interfaces/IAccount.sol";
        contract Account is Guard, IAccount {
            function ping() external {}
        }
        """,
    )
    _write(tmp_path, "src/other/Boring.sol", "contract Boring { uint x; }")
    selection = select_sources(tmp_path, focus_contract="Account")
    assert "src/account/Account.sol" in selection.selected
    assert "src/access/Guard.sol" in selection.selected
    assert "src/interfaces/IAccount.sol" in selection.selected
    reasons = {item.path: item.reason for item in selection.ranked}
    assert "import or base" in reasons["src/access/Guard.sol"]
    assert selection.as_dict()["dependency_status"] == "traced"
    assert selection.escaped_imports == ()


def test_duplicate_basenames_are_ambiguous_instead_of_a_basename_match(tmp_path: Path) -> None:
    _write(tmp_path, "src/a/Token.sol", "contract TokenA { uint x; }")
    _write(tmp_path, "src/b/Token.sol", "contract TokenB { uint y; }")
    _write(
        tmp_path,
        "src/Vault.sol",
        'import "Token.sol";\ncontract Vault { function f() external {} }\n',
    )
    selection = select_sources(tmp_path, focus_contract="Vault")
    assert selection.ambiguous_imports
    matched = {item.path for item in selection.ranked if "import or base" in item.reason}
    assert "src/a/Token.sol" not in matched
    assert "src/b/Token.sol" not in matched
    assert selection.as_dict()["dependency_status"] == "partial"


def test_imports_that_escape_the_repository_are_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/Vault.sol",
        'import "../../outside.sol";\ncontract Vault { function f() external {} }\n',
    )
    outside = tmp_path.parent / "outside.sol"
    outside.write_text("contract Outside {}", encoding="utf-8")
    selection = select_sources(tmp_path, focus_contract="Vault")
    assert selection.escaped_imports
    assert all(not item.path.startswith("..") for item in selection.ranked)
    outside.unlink(missing_ok=True)


def test_a_tight_limit_reports_truncation_and_the_dropped_dependency(tmp_path: Path) -> None:
    _write(tmp_path, "src/Base.sol", "contract Base { address public owner; }")
    _write(
        tmp_path,
        "src/Child.sol",
        'import "./Base.sol";\ncontract Child is Base { function f() external {} }\n',
    )
    for index in range(4):
        _write(tmp_path, f"src/Noise{index}.sol", f"contract Noise{index} {{ uint x; }}")
    selection = select_sources(tmp_path, focus_contract="Child", limit=1)
    assert selection.truncated
    assert selection.selected == ("src/Child.sol",)
    assert "src/Base.sol" in selection.dropped
    assert selection.as_dict()["dependency_status"] != "traced"
    summary = campaign_coverage(
        files_discovered=selection.considered,
        files_selected=len(selection.selected),
        files_read=1,
        truncated=True,
        candidates=0,
        sequences=0,
    )
    assert summary["analysis_complete"] is False
    assert summary["blocker"] == "analysis was truncated before the target was modeled"
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["verified"] is False


def test_an_empty_repository_explains_itself(tmp_path: Path) -> None:
    selection = select_sources(tmp_path)
    assert selection.selected == ()
    summary = campaign_coverage(files_discovered=0, files_selected=0, files_read=0)
    assert summary["blocker"] == "no files selected"
    assert summary["files_selected"] == 0


def test_priority_order_still_prefers_the_in_scope_file(tmp_path: Path) -> None:
    _write(tmp_path, "InScope.sol", "contract InScope { function f() external {} }")
    _write(tmp_path, "Boring.sol", "contract Boring { uint x; }")
    selection = select_sources(tmp_path, focus_contract="InScope", limit=1)
    assert selection.selected == ("InScope.sol",)
    assert selection.ranked[0].priority is Priority.SCOPE

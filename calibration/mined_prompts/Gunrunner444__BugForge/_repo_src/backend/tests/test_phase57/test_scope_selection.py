"""Target selection keeps the dependency closure and leaves excluded paths out of the budget."""

from __future__ import annotations

from pathlib import Path

from app.discovery.bounty.source_selection import select_sources
from tests.test_phase51.phase51_support import manifest


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_excluded_directories_do_not_displace_the_target_closure(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "contracts/Vault.sol",
        'import "./Math.sol";\ncontract Vault { function f() external {} }\n',
    )
    _write(
        tmp_path,
        "contracts/Math.sol",
        "contract Math { function add(uint a, uint b) external pure returns (uint) { return a + b; } }\n",
    )
    for index in range(6):
        _write(
            tmp_path,
            f"contracts/bridges/Bridge{index}.sol",
            "contract Bridge { function pull(address token) external { token.call(hex''); } }\n",
        )
        _write(
            tmp_path,
            f"contracts/liquidator/Liq{index}.sol",
            "contract Liq { function take() external { } function sweep(address t) external { t.call(hex''); } }\n",
        )
    selection = select_sources(
        tmp_path,
        manifest=manifest(
            in_scope=[{"kind": "path", "identifier": "contracts/Vault.sol"}],
            out_of_scope=[
                {"kind": "path", "identifier": "contracts/bridges"},
                {"kind": "path", "identifier": "contracts/liquidator"},
            ],
            known_issues=[],
        ),
        focus_file="contracts/Vault.sol",
        limit=4,
    )
    assert selection.selected[0] == "contracts/Vault.sol"
    assert "contracts/Math.sol" in selection.selected
    assert selection.role_of("contracts/Vault.sol") == "target"
    assert selection.role_of("contracts/Math.sol") == "compile_dependency"
    assert selection.closure_complete is True
    assert all("bridges" not in path and "liquidator" not in path for path in selection.selected)
    assert any(path.startswith("contracts/bridges/") for path in selection.scope_excluded)
    assert selection.dependency_status() == "traced"


def test_a_required_dependency_stays_even_when_its_directory_is_excluded(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "contracts/Vault.sol",
        'import "./vendor/Lib.sol";\ncontract Vault { function f() external {} }\n',
    )
    _write(
        tmp_path,
        "contracts/vendor/Lib.sol",
        "contract Lib { function id(uint x) external pure returns (uint) { return x; } }\n",
    )
    _write(tmp_path, "contracts/vendor/Other.sol", "contract Other { function g() external {} }\n")
    selection = select_sources(
        tmp_path,
        manifest=manifest(
            in_scope=[{"kind": "contract", "identifier": "Vault"}],
            out_of_scope=[{"kind": "path", "identifier": "contracts/vendor"}],
            known_issues=[],
        ),
        focus_contract="Vault",
        focus_file="contracts/Vault.sol",
    )
    assert "contracts/vendor/Lib.sol" in selection.selected
    assert selection.role_of("contracts/vendor/Lib.sol") == "compile_dependency"
    assert "contracts/vendor/Other.sol" not in selection.selected
    assert selection.role_of("contracts/vendor/Other.sol") == "excluded"


def test_a_dropped_dependency_is_not_described_as_traced(tmp_path: Path) -> None:
    _write(tmp_path, "src/Base.sol", "contract Base { uint x; }")
    _write(
        tmp_path,
        "src/Child.sol",
        'import "./Base.sol";\ncontract Child is Base { function f() external {} }\n',
    )
    selection = select_sources(tmp_path, focus_contract="Child", limit=1)
    assert selection.selected == ("src/Child.sol",)
    assert selection.closure_complete is False
    assert selection.dependency_status() != "traced"
    assert "src/Base.sol" in selection.dropped

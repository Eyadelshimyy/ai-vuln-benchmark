"""Phase 35 Yul success tracking and cross-contract taxonomy."""

from __future__ import annotations

from pathlib import Path

from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.parsing.solidity_yul import analyze_yul
from app.security.solidity.rules import CrossContractRule, YulStructureRule


def _graph(tmp_path: Path, source: str, name: str = "Y.sol"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return parse_source("solidity", path, source)


def _yul(body: str) -> str:
    return f"""
    pragma solidity ^0.8.20;
    contract A {{
        function run() external {{
            assembly {{
                {body}
            }}
        }}
    }}
    """


def test_checked_success_is_not_ignored(tmp_path: Path) -> None:
    model = analyze_yul(
        _graph(tmp_path, _yul("let ok := call(gas(), caller(), 0, 0, 0, 0, 0)\nif ok { stop() }"))
    )
    assert model.calls
    assert model.calls[0].success_ignored is False
    assert not any("ignored" in item for item in model.hostile)


def test_unchecked_success_is_ignored(tmp_path: Path) -> None:
    model = analyze_yul(_graph(tmp_path, _yul("let ok := call(gas(), caller(), 0, 0, 0, 0, 0)")))
    assert model.calls[0].success_ignored is True
    assert any(item == "ignored call success" for item in model.hostile)


def test_reassignment_shadowing_and_branch_do_not_count_as_checks(tmp_path: Path) -> None:
    reassigned = analyze_yul(
        _graph(
            tmp_path,
            _yul("let ok := staticcall(gas(), caller(), 0, 0, 0, 0)\nok := 1\nif ok { stop() }"),
            "re.sol",
        )
    )
    assert reassigned.calls[0].success_ignored is True
    shadowed = analyze_yul(
        _graph(
            tmp_path,
            _yul(
                "let ok := delegatecall(gas(), caller(), 0, 0, 0, 0)\n{ let ok := 1\nif ok { stop() } }"
            ),
            "shadow.sol",
        )
    )
    assert shadowed.calls[0].success_ignored is True
    unrelated = analyze_yul(
        _graph(
            tmp_path,
            _yul(
                'let ok := 1\nlet result := call(gas(), caller(), 0, 0, 0, 0, 0)\nif ok { stop() }\nlet note := "result"'
            ),
            "name.sol",
        )
    )
    assert unrelated.calls[0].success_ignored is True
    branched = analyze_yul(
        _graph(
            tmp_path,
            _yul(
                "let ok := call(gas(), caller(), 0, 0, 0, 0, 0)\nif iszero(0) { if ok { stop() } }"
            ),
            "branch.sol",
        )
    )
    assert branched.calls[0].success_ignored is True


def test_yul_rules_do_not_share_one_class(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        function run() external {
            assembly {
                let impl := sload(0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc)
                pop(delegatecall(gas(), caller(), 0, 0, 0, 0))
                sstore(0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc, 1)
            }
        }
    }
    """
    observations = YulStructureRule().check(_graph(tmp_path, source, "Proxy.sol"))
    classes = {item.vulnerability_class for item in observations}
    assert VulnerabilityClass.UNSAFE_PROXY in classes
    assert (
        VulnerabilityClass.UNSAFE_EXTERNAL_CALL in classes
        or VulnerabilityClass.REENTRANCY in classes
    )
    assert len(classes) > 1


def test_cross_contract_authorization_is_not_reentrancy(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A { modifier onlyOwner() { require(msg.sender == msg.sender); _; } }
    contract B { modifier onlyOwner() { require(tx.origin == msg.sender); _; } }
    contract Ambiguous is A, B {
        function set(uint256 value) external onlyOwner { }
    }
    """
    observations = CrossContractRule().check(_graph(tmp_path, source, "Auth.sol"))
    authorization = [item for item in observations if "authorization" in item.title.lower()]
    assert authorization
    assert all(
        item.vulnerability_class is VulnerabilityClass.AUTHORIZATION for item in authorization
    )
    assert all(
        item.vulnerability_class is not VulnerabilityClass.REENTRANCY for item in authorization
    )
    reentrancy = """
    pragma solidity ^0.8.20;
    contract VaultLate {
        mapping(address => uint256) balances;
        function withdraw() external {
            uint256 amount = balances[msg.sender];
            _send(amount);
        }
        function _send(uint256 amount) internal {
            msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
    found = CrossContractRule().check(_graph(tmp_path, reentrancy, "Vault.sol"))
    assert any(item.vulnerability_class is VulnerabilityClass.REENTRANCY for item in found)


def test_unbounded_calldataload_is_hostile_and_a_literal_offset_is_not(tmp_path: Path) -> None:
    unbounded = """
    pragma solidity ^0.8.20;
    contract BadDecode {
        function pull(bytes calldata data) external pure returns (address token, uint256 amount) {
            assembly {
                token := calldataload(data.offset)
                amount := calldataload(add(data.offset, 32))
            }
        }
    }
    """
    model = analyze_yul(_graph(tmp_path, unbounded, "Bad.sol"))
    assert "unbounded calldata load" in model.hostile
    observations = YulStructureRule().check(_graph(tmp_path, unbounded, "Bad.sol"))
    assert any(
        item.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION for item in observations
    )
    bounded = """
    pragma solidity ^0.8.20;
    contract SafeDecode {
        function pull(bytes calldata data) external pure returns (address token, uint256 amount) {
            require(data.length >= 64);
            (token, amount) = abi.decode(data, (address, uint256));
        }
    }
    """
    safe = analyze_yul(_graph(tmp_path, bounded, "Safe.sol"))
    assert "unbounded calldata load" not in safe.hostile
    literal = analyze_yul(_graph(tmp_path, _yul("let word := calldataload(0)"), "Lit.sol"))
    assert "unbounded calldata load" not in literal.hostile
    checked = analyze_yul(
        _graph(
            tmp_path,
            _yul("let size := calldatasize()\nlet word := calldataload(add(4, 32))"),
            "Checked.sol",
        )
    )
    assert "unbounded calldata load" not in checked.hostile

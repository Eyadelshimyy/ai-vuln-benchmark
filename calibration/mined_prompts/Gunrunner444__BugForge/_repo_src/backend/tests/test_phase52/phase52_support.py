"""Helpers for Phase 52 hardening tests. Forge-backed tests skip when forge/solc are absent."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.discovery.bounty.vfcs import SequenceIdentity, Vfcs, VfcsCall
from app.parsing.solidity_research import ResearchModel, build_research_model

FOUNDRY_BIN = Path("/home/box/.foundry/bin")
LOCAL_BIN = Path("/home/box/.local/bin")


def _has(tool: str) -> bool:
    return bool(shutil.which(tool))


requires_forge = pytest.mark.skipif(
    not (_has("forge") and _has("solc")), reason="forge and solc are not installed"
)

VULNERABLE_INIT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Acct {
    address public owner;

    function initialize(address newOwner) external {
        owner = newOwner;
    }
}
"""

SAFE_INIT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Acct {
    address public owner;
    bool public initialized;

    function initialize(address newOwner) external {
        require(!initialized, "initialized");
        initialized = true;
        owner = newOwner;
    }
}
"""

TOKEN_IFACE = """interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}
"""

RAW_VAULT = (
    "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n"
    + TOKEN_IFACE
    + """
contract Vault {
    mapping(address => uint256) public balances;

    function deposit(IERC20 token, uint256 amount) external {
        token.transferFrom(msg.sender, address(this), amount);
        balances[msg.sender] += amount;
    }
}
"""
)

DELTA_VAULT = (
    "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\n"
    + TOKEN_IFACE
    + """
contract Vault {
    mapping(address => uint256) public balances;

    function deposit(IERC20 token, uint256 amount) external {
        uint256 before = token.balanceOf(address(this));
        require(token.transferFrom(msg.sender, address(this), amount), "pull");
        balances[msg.sender] += token.balanceOf(address(this)) - before;
    }
}
"""
)


def model_of(sources: dict[str, str]) -> ResearchModel:
    return build_research_model(sources)


def init_sequence(contract: str = "Acct", identity: SequenceIdentity | None = None) -> Vfcs:
    sig = "initialize(address)"
    calls = (
        VfcsCall(contract, sig, "initialize", "attacker", (("newOwner", "attacker"),)),
        VfcsCall(contract, sig, "reinitialize", "victim", (("newOwner", "second_owner"),)),
    )
    return Vfcs(
        sequence_id="vf_init_test",
        template="initialize→reinitialize",
        origin="template:initialize→reinitialize",
        calls=calls,
        property_under_test="an account initializes exactly once",
        derived_from=f"aa.unprotected_account_initializer@{contract}.{sig}",
        identity=identity or SequenceIdentity(),
    )


def deposit_sequence(detector: str = "accounting.fee_on_transfer_mismatch") -> Vfcs:
    sig = "deposit(IERC20,uint256)"
    calls = (
        VfcsCall(
            "Vault",
            sig,
            "deposit",
            "attacker",
            (("token", "unconstrained"), ("amount", "amount_one")),
        ),
    )
    return Vfcs(
        sequence_id="vf_deposit_test",
        template="deposit",
        origin="template:deposit",
        calls=calls,
        property_under_test="credited balance must equal the amount received",
        derived_from=f"{detector}@Vault.{sig}",
        identity=SequenceIdentity(),
    )


HARNESS_ADDR = "0x7fa9385be102ac3eac297483dd6233d62b3e1496"
TOPIC = "0x05cd64bb6ffe3a3804e150ecc2c28424720f099e6672c02e96eb325669e2f411"


def event_log(kind: int, index: int, value: int, address: str = HARNESS_ADDR) -> dict[str, object]:
    data = "0x" + "".join(f"{word:064x}" for word in (kind, index, value))
    return {"address": address, "topics": [TOPIC], "data": data}


def forge_json(logs: list[dict[str, object]], status: str = "Success", reason: str = "") -> str:
    return json.dumps(
        {
            "test/VfcsHarness.t.sol:VfcsHarness": {
                "test_results": {
                    "test_vfcs_execute()": {
                        "status": status,
                        "reason": reason or None,
                        "logs": logs,
                        "decoded_logs": [],
                    }
                }
            }
        }
    )


def requires_engine(name: str) -> pytest.MarkDecorator:
    """Skip unless forge, solc, crytic-compile and the named engine are installed."""
    return pytest.mark.skipif(
        not all(_has(tool) for tool in ("forge", "solc", "crytic-compile", name)),
        reason=f"{name}, crytic-compile or solc is not installed",
    )


REGISTRY_FILE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IRegistry {
    function isAllowed(address who) external view returns (bool);
}

contract OpenRegistry is IRegistry {
    function isAllowed(address) external pure returns (bool) {
        return true;
    }
}
"""

MULTI_ACCT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./registry/Registry.sol";

contract Clock {
    uint256 public tick;
}

contract Acct {
    IRegistry public registry;
    Clock public clock;
    address public owner;

    constructor(IRegistry r, Clock c) {
        registry = r;
        clock = c;
    }

    function initialize(address newOwner) external {
        require(registry.isAllowed(newOwner), "denied");
        owner = newOwner;
    }
}
"""


def multi_sources() -> dict[str, str]:
    return {"registry/Registry.sol": REGISTRY_FILE, "Acct.sol": MULTI_ACCT}

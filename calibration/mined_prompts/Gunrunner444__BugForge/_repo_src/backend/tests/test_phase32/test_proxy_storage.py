"""Phase 32 storage layout, delegatecall, and upgradeability evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_compiler import (
    CompilerSemantics,
    compiler_semantics,
    compiler_semantics_for_scan,
    solidity_compiler_status,
)
from app.parsing.solidity_proxy import analyze_proxy, proxy_context_active
from app.parsing.solidity_storage import (
    analyze_storage,
    apply_compiler_layout,
    compare_storage_layouts,
    storage_context_active,
)
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine

_IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _ids(tmp_path: Path, source: str) -> set[str]:
    path = tmp_path / "Contract.sol"
    path.write_text(source, encoding="utf-8")
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    assert all(item.metadata.get("status") == "potential" for item in result.observations)
    assert all(item.metadata.get("verified") != "true" for item in result.observations)
    return {item.rule_id for item in result.observations}


def _graph(tmp_path: Path, source: str, name: str = "Contract.sol"):
    return parse_source("solidity", tmp_path / name, source)


def _by_name(model, contract: str):
    order = set(model.order.get(contract, ()))
    found = {}
    for item in model.variables:
        if item.origin in order or item.contract == contract:
            found[(item.origin, item.name)] = item
    return found


def test_elementary_packed_struct_array_and_mapping_layout(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    struct Point { uint8 x; uint8 y; }
    contract Store {
        uint256 a;
        uint8 b;
        uint8 c;
        Point p;
        uint256 afterPoint;
        mapping(address => uint256) balances;
        uint256[] items;
        uint8[2] packed;
        address tail;
    }
    """
    model = analyze_storage(_graph(tmp_path, source))
    laid = _by_name(model, "Store")
    assert laid[("Store", "a")].slot == 0
    assert laid[("Store", "b")].slot == 1 and laid[("Store", "b")].offset == 0
    assert laid[("Store", "c")].slot == 1 and laid[("Store", "c")].offset == 1
    assert laid[("Store", "p")].slot == 2
    assert laid[("Store", "p")].members == ("uint8", "uint8")
    assert laid[("Store", "afterPoint")].slot == 3
    assert laid[("Store", "balances")].slot == 4
    assert laid[("Store", "balances")].key_type == "address"
    assert laid[("Store", "items")].slot == 5
    assert laid[("Store", "packed")].slot == 6
    assert laid[("Store", "tail")].slot == 7
    assert "Store" not in model.uncertain


def test_inherited_layout_and_ambiguous_bases(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract A { uint8 a; }
    contract B is A { uint8 b; }
    contract Child is A, B { uint256 e; }
    """
    model = analyze_storage(_graph(tmp_path, source))
    laid = _by_name(model, "Child")
    assert model.order["Child"] == ("A", "B", "Child")
    assert laid[("A", "a")].slot == 0 and laid[("A", "a")].offset == 0
    assert laid[("B", "b")].slot == 0 and laid[("B", "b")].offset == 1
    assert laid[("Child", "e")].slot == 1
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    left = "pragma solidity ^0.8.20; contract Ownable { uint256 left; }"
    right = "pragma solidity ^0.8.20; contract Ownable { uint8 right; }"
    child = """
    pragma solidity ^0.8.20;
    contract Child is Ownable { uint256 extra; }
    """
    graphs = {
        str(first / "L.sol"): _graph(first, left, "L.sol"),
        str(second / "R.sol"): _graph(second, right, "R.sol"),
        str(tmp_path / "C.sol"): _graph(tmp_path, child, "C.sol"),
    }
    from app.parsing.solidity_storage import reset_storage_context, set_storage_context

    token = set_storage_context(graphs)
    try:
        ambiguous = analyze_storage(graphs[str(tmp_path / "C.sol")])
    finally:
        reset_storage_context(token)
    assert ambiguous.uncertain.get("Child")
    assert all(item.slot is None for item in ambiguous.variables if item.origin == "Child")
    assert storage_context_active() is False


def test_proxy_overlap_is_slot_based_not_name_based(tmp_path: Path) -> None:
    overlap = """
    pragma solidity ^0.8.20;
    contract Impl { uint256 value; }
    contract Proxy {
        address implementation;
        function use(Impl next) external { implementation = address(next); }
        fallback() external payable { implementation.delegatecall(msg.data); }
    }
    """
    business = """
    pragma solidity ^0.8.20;
    contract Shop {
        address implementation;
        function setClerk(address implementation_) external { implementation = implementation_; }
    }
    """
    assert "sol.storage_collision" in _ids(tmp_path, overlap)
    assert "sol.storage_collision" not in _ids(tmp_path, business)
    model = analyze_storage(_graph(tmp_path, overlap, "Overlap.sol"))
    assert model.overlaps
    assert "slot 0" in model.overlaps[0]


def test_compiler_overlay_keeps_both_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "app.parsing.solidity_compiler.shutil.which",
        lambda _name: None,
    )
    assert solidity_compiler_status().status == "UNAVAILABLE"
    absent = compiler_semantics("contract C { uint a; }")
    assert absent.status == "UNAVAILABLE"
    assert absent.storage == [] and absent.layouts == []
    monkeypatch.setattr(
        "app.parsing.solidity_compiler.shutil.which",
        lambda _name: "/usr/bin/solc",
    )
    payload = """
    {"contracts":{"C.sol":{
      "Proxy":{"storageLayout":{"storage":[{"label":"implementation","slot":"0","offset":0}]}},
      "Impl":{"storageLayout":{"storage":[{"label":"value","slot":"1","offset":0}]}}
    }},"version":"0.8.20"}
    """
    parsed = compiler_semantics("contract C {}", runner=lambda _source: payload)
    assert parsed.status == "AVAILABLE"
    assert {item["label"] for item in parsed.storage} == {"implementation", "value"}
    assert {item["contract"] for item in parsed.layouts} == {"Proxy", "Impl"}
    source = """
    pragma solidity ^0.8.20;
    contract Impl { uint256 value; }
    """
    model = analyze_storage(_graph(tmp_path, source))
    apply_compiler_layout(
        model,
        CompilerSemantics(
            "AVAILABLE",
            "solc",
            "test",
            layouts=[{"contract": "Impl", "label": "value", "slot": "4"}],
        ),
    )
    assert model.disagreements
    assert model.disagreements[0].parser_slot == "0"
    assert model.disagreements[0].compiler_slot == "4"
    assert model.variables[0].slot == 0


def test_delegatecall_provenance(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Kinds {
        address implementation;
        address immutable fixedImpl;
        address constant CONST = address(0);
        function fixedCall(bytes calldata data) external { fixedImpl.delegatecall(data); }
        function mutableCall(bytes calldata data) external { implementation.delegatecall(data); }
        function callerCall(address target) external { target.delegatecall(""); }
        function selfCall(bytes calldata data) external { address(this).delegatecall(data); }
        function asm(bytes calldata data) external {
            assembly { let ok := delegatecall(gas(), implementation, 0, 0, 0, 0) }
        }
        function unresolved() external {
            assembly { let ok := delegatecall(gas(), who, 0, 0, 0, 0) }
        }
    }
    """
    model = analyze_proxy(_graph(tmp_path, source))
    by_fn = {item.function: item for item in model.delegates}
    assert by_fn["fixedCall"].provenance == "immutable"
    assert by_fn["mutableCall"].provenance == "state"
    assert by_fn["callerCall"].caller_controlled is True
    assert by_fn["selfCall"].provenance == "self"
    assert by_fn["asm"].provenance == "state"
    assert by_fn["unresolved"].provenance == "unknown"
    ids = _ids(tmp_path, source)
    assert "sol.arbitrary_delegatecall" in ids
    assert "sol.assembly_sensitive" in ids


def test_upgrade_authorization_uses_the_check_not_the_name(tmp_path: Path) -> None:
    protected = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address owner;
        address implementation;
        modifier onlyOwner() { require(msg.sender == owner); _; }
        function upgradeTo(address next) external onlyOwner { implementation = next; }
        fallback() external payable { implementation.delegatecall(msg.data); }
    }
    """
    open_fn = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        function upgradeTo(address next) external { implementation = next; }
    }
    """
    fake = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        modifier onlyOwner() { _; }
        function setImplementation(address next) external onlyOwner { implementation = next; }
    }
    """
    role = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        modifier onlyRole(bytes32 role) { require(hasRole(role, msg.sender)); _; }
        function upgradeTo(address next) external onlyRole(0) { implementation = next; }
    }
    """
    inherited = """
    pragma solidity ^0.8.20;
    contract Ownable {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    contract Proxy is Ownable {
        address implementation;
        function upgradeTo(address next) external onlyOwner { implementation = next; }
    }
    """
    ambiguous = """
    pragma solidity ^0.8.20;
    contract A {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    contract B {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    contract Proxy is A, B {
        address owner;
        address implementation;
        function upgradeTo(address next) external onlyOwner { implementation = next; }
    }
    """
    assert "sol.upgrade_auth" not in _ids(tmp_path, protected)
    assert "sol.upgrade_auth" in _ids(tmp_path, open_fn)
    assert "sol.upgrade_auth" in _ids(tmp_path, fake)
    assert "sol.upgrade_auth" not in _ids(tmp_path, role)
    assert "sol.upgrade_auth" not in _ids(tmp_path, inherited)
    assert "sol.upgrade_auth" in _ids(tmp_path, ambiguous)


def test_uups_shape_and_initializer_interaction(tmp_path: Path) -> None:
    valid = """
    pragma solidity ^0.8.20;
    contract Impl {
        address owner;
        address implementation;
        bool initialized;
        modifier onlyOwner() { require(msg.sender == owner); _; }
        modifier initializer() { require(!initialized); initialized = true; _; }
        function proxiableUUID() external pure returns (bytes32) { return bytes32(0); }
        function upgradeTo(address next) external onlyOwner { implementation = next; }
        function initialize() external initializer { owner = msg.sender; }
    }
    """
    malformed = """
    pragma solidity ^0.8.20;
    contract Impl {
        address implementation;
        function proxiableUUID() external pure returns (bytes32) { return bytes32(0); }
        function upgradeToAndCall(address next, bytes calldata data) external {
            implementation = next;
            (bool ok,) = next.call(data);
            require(ok);
        }
        function initialize() external { implementation = msg.sender; }
    }
    """
    good = analyze_proxy(_graph(tmp_path, valid, "Valid.sol"))
    bad = analyze_proxy(_graph(tmp_path, malformed, "Bad.sol"))
    assert "uups-like" in good.patterns
    assert all(item.authorized for item in good.upgrades)
    assert "uups-like" in bad.patterns
    assert any(not item.authorized and item.invokes_call for item in bad.upgrades)
    assert "sol.upgrade_auth" in _ids(tmp_path, malformed)
    assert "sol.initializer" in _ids(tmp_path, malformed)
    assert "sol.initializer" not in _ids(tmp_path, valid)
    assert "sol.upgrade_auth" not in _ids(tmp_path, valid)


def test_yul_slots_are_linked_only_when_known(tmp_path: Path) -> None:
    source = f"""
    pragma solidity ^0.8.20;
    contract Proxy {{
        bytes32 constant SLOT = {_IMPL_SLOT};
        bytes32 constant CUSTOM = 0x1234;
        function read() external {{
            assembly {{
                let impl := sload(SLOT)
                sstore(CUSTOM, impl)
                let mystery := sload(add(SLOT, 1))
            }}
        }}
    }}
    """
    model = analyze_storage(_graph(tmp_path, source))
    by_expr = {item.expression: item for item in model.yul}
    assert by_expr["SLOT"].known is True
    assert by_expr["SLOT"].literal.lower() == _IMPL_SLOT
    assert by_expr["CUSTOM"].known is True
    assert by_expr["add(SLOT, 1)"].known is False
    notes = " ".join(analyze_proxy(_graph(tmp_path, source, "Yul.sol")).notes)
    assert "EIP-1967" in notes
    assert "not a literal" in notes


def test_adversarial_proxy_words_stay_quiet(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Noise {
        address owner;
        uint version;
        function report() external pure returns (string memory) {
            return "upgrade the implementation";
        }
        function balanceOf(uint reserve) external pure returns (uint) {
            return reserve / 2;
        }
    }
    """
    ids = _ids(tmp_path, source)
    for rule_id in (
        "sol.storage_collision",
        "sol.arbitrary_delegatecall",
        "sol.upgrade_auth",
        "sol.initializer",
        "sol.erc4626",
        "sol.lending",
    ):
        assert rule_id not in ids


def test_phase31_regressions_still_hold(tmp_path: Path) -> None:
    from app.parsing.solidity_defi import analyze_defi
    from app.parsing.solidity_flow import oracle_freshness_protects, signature_replay_gap
    from app.parsing.solidity_guards import initializer_is_protected, reentrancy_guard_holds

    assert (
        reentrancy_guard_holds(
            "{ require(!locked); locked = true; _; if (condition) { locked = false; } }"
        )
        is False
    )
    assert initializer_is_protected("{ require(version < 999); version = 1; _; }") is False
    gap = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonces[owner] += 1;
        h = keccak256(abi.encode(nonces[attacker]));
        return ecrecover(h, v, r, s);
    }
    """
    assert signature_replay_gap(gap) is not None
    weak = """
    function price() external view returns (int) {
        (, int answer,, uint updatedAt,) = feed.latestRoundData();
        require(updatedAt != 0);
        return answer;
    }
    """
    assert oracle_freshness_protects(weak) is False
    twins = """
    pragma solidity ^0.8.20;
    contract VaultA {
        uint totalSupply; uint totalAssets;
        function previewDeposit(uint assets) external view returns (uint) {
            return assets * totalSupply / totalAssets;
        }
        function deposit(uint assets, uint minShares) external view returns (uint) {
            uint minted = assets * totalSupply / totalAssets;
            require(minted >= minShares);
            return minted;
        }
    }
    contract VaultB {
        uint totalSupply; uint totalAssets;
        function previewDeposit(uint assets) external view returns (uint) {
            return assets * totalAssets / totalSupply;
        }
        function deposit(uint assets) external view returns (uint) {
            return assets * totalSupply / totalAssets;
        }
    }
    """
    issues = [
        item
        for item in analyze_defi(_graph(tmp_path, twins, "Twins.sol")).issues
        if item.rule_id == "sol.erc4626"
    ]
    assert issues and all(item.contract == "VaultB" for item in issues)
    outbound = """
    pragma solidity ^0.8.20;
    interface IERC20 { function transfer(address to, uint256 amount) external returns (bool); }
    contract Lend {
        mapping(address => uint) debt;
        function repay(IERC20 token, uint amount, address attacker) external {
            debt[msg.sender] -= amount;
            token.transfer(attacker, amount);
        }
    }
    """
    assert "sol.lending" in _ids(tmp_path, outbound)
    assert proxy_context_active() is False
    assert storage_context_active() is False


def test_compatible_proxy_layout_is_not_a_collision(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy {
        address implementation;
        uint256 adminValue;
        function use(Impl next) external { implementation = address(next); }
        fallback() external payable { implementation.delegatecall(msg.data); }
    }
    contract Impl {
        address implementation;
        uint256 adminValue;
    }
    """
    model = analyze_storage(_graph(tmp_path, source, "Compatible.sol"))
    assert model.comparisons
    assert model.comparisons[0].compatible is True
    assert model.overlaps == []
    assert "sol.storage_collision" not in _ids(tmp_path, source)


def test_version_layout_detects_shift_and_append(tmp_path: Path) -> None:
    first = """
    pragma solidity ^0.8.20;
    contract Token {
        uint256 value;
        uint256 amount;
    }
    """
    shifted = """
    pragma solidity ^0.8.20;
    contract Token {
        address owner;
        uint256 value;
        uint256 amount;
    }
    """
    appended = """
    pragma solidity ^0.8.20;
    contract Token {
        uint256 value;
        uint256 amount;
        uint256 extra;
    }
    """
    removed = """
    pragma solidity ^0.8.20;
    contract Token { uint256 value; }
    """
    reordered = """
    pragma solidity ^0.8.20;
    contract Token { uint256 amount; uint256 value; }
    """
    left = analyze_storage(_graph(tmp_path, first, "V1.sol"))
    right = analyze_storage(_graph(tmp_path, shifted, "V2.sol"))
    comparison = compare_storage_layouts(left, right, "Token", "Token", "implementation-version")
    assert comparison.compatible is False
    assert "added_before_existing" in {item.kind for item in comparison.changes}
    assert "slot_changed" in {item.kind for item in comparison.changes}
    grown = compare_storage_layouts(
        left, analyze_storage(_graph(tmp_path, appended, "V3.sol")), "Token", "Token", "version"
    )
    assert grown.compatible is True
    assert grown.appended == ("extra",)
    shorter = compare_storage_layouts(
        left, analyze_storage(_graph(tmp_path, removed, "V4.sol")), "Token", "Token", "version"
    )
    assert any(item.kind == "removed_existing" for item in shorter.changes)
    swapped = compare_storage_layouts(
        left, analyze_storage(_graph(tmp_path, reordered, "V5.sol")), "Token", "Token", "version"
    )
    assert any(item.kind == "reordered" for item in swapped.changes)


def test_layout_type_packing_mapping_and_struct_changes(tmp_path: Path) -> None:
    base = """
    pragma solidity ^0.8.20;
    contract Token {
        uint128 left;
        uint128 right;
        mapping(address => uint256) balances;
        struct Point { uint256 x; uint256 y; }
        Point point;
        uint256[2] items;
    }
    """
    changed = """
    pragma solidity ^0.8.20;
    contract Token {
        uint256 left;
        uint128 right;
        mapping(address => address) balances;
        struct Point { uint256 x; address y; }
        Point point;
        uint256[3] items;
    }
    """
    comparison = compare_storage_layouts(
        analyze_storage(_graph(tmp_path, base, "A.sol")),
        analyze_storage(_graph(tmp_path, changed, "B.sol")),
        "Token",
        "Token",
        "version",
    )
    kinds = {item.kind for item in comparison.changes}
    assert "type_changed" in kinds
    assert "mapping_changed" in kinds
    assert "struct_changed" in kinds
    assert comparison.compatible is False


def test_compiler_match_requires_contract_identity(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Proxy { address implementation; }
    contract Impl { uint256 value; }
    """
    model = analyze_storage(_graph(tmp_path, source, "Both.sol"))
    apply_compiler_layout(
        model,
        CompilerSemantics(
            "AVAILABLE",
            "solc",
            "test",
            layouts=[
                {"contract": "Proxy", "label": "value", "slot": "9", "source": "Other.sol"},
                {
                    "contract": "Impl",
                    "label": "value",
                    "slot": "3",
                    "offset": "0",
                    "type": "uint256",
                },
                {"label": "implementation", "slot": "1"},
            ],
        ),
        source_path=str(tmp_path / "Both.sol"),
    )
    assert model.ambiguities
    assert any(item.name == "value" and item.compiler_slot == "3" for item in model.disagreements)
    assert all(
        item.contract != "Proxy" or item.name != "implementation" for item in model.disagreements
    )
    assert all(item.slot == 0 for item in model.variables)


def test_scan_records_unavailable_compiler_without_inventing_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.parsing.solidity_compiler.shutil.which", lambda _name: None)
    source = "pragma solidity ^0.8.20; contract C { uint256 value; }"
    assert compiler_semantics_for_scan(source).status == "UNAVAILABLE"
    path = tmp_path / "C.sol"
    path.write_text(source, encoding="utf-8")
    SecurityAnalysisEngine().analyze_repository(tmp_path, [path])
    model = analyze_storage(_graph(tmp_path, source, "C.sol"))
    assert model.variables[0].slot == 0
    monkeypatch.setattr(
        "app.parsing.solidity_compiler.shutil.which",
        lambda name: "/usr/bin/forge" if name == "forge" else None,
    )
    present = compiler_semantics_for_scan(source)
    assert present.status == "AVAILABLE"
    assert present.storage == [] and present.layouts == []


def test_namespaced_storage_is_not_sequential(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Names {
        bytes32 constant KNOWN = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
        bytes32 constant LITERAL = 0x1234;
        bytes32 constant CUSTOM = keccak256("example.main");
        bytes32 constant ERC = keccak256(abi.encode(uint256(keccak256("eip7201:example.main") - 1))) & ~bytes32(uint256(0xff));
        bytes32 constant UNKNOWN = keccak256(abi.encodePacked(LITERAL));
        uint256 value;
        function hash(uint256 seed) external pure returns (bytes32) {
            return keccak256(abi.encodePacked(seed));
        }
    }
    """
    model = analyze_storage(_graph(tmp_path, source, "Ns.sol"))
    by_name = {item.name: item for item in model.namespaces}
    assert by_name["LITERAL"].known is True
    assert by_name["LITERAL"].resolved_slot == "0x1234"
    assert by_name["CUSTOM"].known is True
    assert by_name["CUSTOM"].resolved_slot.startswith("0x")
    assert by_name["ERC"].known is True
    assert by_name["ERC"].resolved_slot != by_name["CUSTOM"].resolved_slot
    assert by_name["UNKNOWN"].known is False
    assert "hash" not in by_name
    value = next(item for item in model.variables if item.name == "value")
    assert value.slot == 0
    assert value.mutability == "storage"


def test_yul_delegatecall_follows_sload(tmp_path: Path) -> None:
    source = f"""
    pragma solidity ^0.8.20;
    contract Proxy {{
        bytes32 constant SLOT = {_IMPL_SLOT};
        fallback() external payable {{
            assembly {{
                let impl := sload(SLOT)
                let ok := delegatecall(gas(), impl, 0, 0, 0, 0)
            }}
        }}
        function copy() external {{
            assembly {{
                let x := sload(SLOT)
                sstore(SLOT, x)
            }}
        }}
    }}
    """
    model = analyze_proxy(_graph(tmp_path, source, "Flow.sol"))
    delegate = next(item for item in model.delegates if item.function == "fallback")
    assert delegate.provenance == "storage_slot"
    assert "sload(SLOT)" in delegate.target
    stored = next(item for item in model.storage.yul if item.op == "sstore")
    assert stored.known is True
    assert stored.value_from.startswith("sload(")


def test_upgrade_name_without_implementation_is_not_an_upgrade(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Counter {
        uint256 counter;
        mapping(address => uint256) users;
        function upgradeCounter(uint256 x) external { counter = x; }
        function upgradeUserRecord(address who, uint256 x) external { users[who] = x; }
        function proxiableUUID() external pure returns (bytes32) { return bytes32(0); }
    }
    """
    model = analyze_proxy(_graph(tmp_path, source, "Name.sol"))
    assert model.upgrades == []
    assert "uups-like" not in model.patterns
    assert "sol.upgrade_auth" not in _ids(tmp_path, source)


def test_proxy_relationships_need_more_than_words(tmp_path: Path) -> None:
    words = """
    pragma solidity ^0.8.20;
    contract Talk {
        function note() external pure returns (string memory) {
            return "beacon diamond facet admin";
        }
    }
    """
    beacon = """
    pragma solidity ^0.8.20;
    interface IBeacon { function implementation() external view returns (address); }
    contract Proxy {
        address beacon;
        fallback() external payable {
            address impl = IBeacon(beacon).implementation();
            impl.delegatecall(msg.data);
        }
    }
    """
    talk = analyze_proxy(_graph(tmp_path, words, "Talk.sol"))
    assert "beacon-like" not in talk.patterns
    assert "diamond-like" not in talk.patterns
    assert "transparent-like" not in talk.patterns
    linked = analyze_proxy(_graph(tmp_path, beacon, "Beacon.sol"))
    assert "beacon-like" in linked.patterns
    assert linked.pattern_evidence[0].confidence == "structural"

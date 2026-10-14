"""Phase 31 Solidity DeFi semantics: corrections, positives, and adversarial cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.engine import parse_source, reset_syntax_registry
from app.parsing.solidity_defi import analyze_defi, defi_context_active
from app.parsing.solidity_flow import (
    oracle_freshness_protects,
    signature_nonce_order,
    signature_replay_gap,
)
from app.parsing.solidity_guards import initializer_is_protected, reentrancy_guard_holds
from app.parsing.solidity_modifiers import ModifierIndex
from app.plugins import reset_plugin_catalog
from app.security.engine import SecurityAnalysisEngine


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_syntax_registry()
    reset_plugin_catalog()


def _result(tmp_path: Path, sources: dict[str, str]) -> object:
    paths = []
    for name, source in sources.items():
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        paths.append(path)
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, paths)
    assert all(item.metadata.get("status") == "potential" for item in result.observations)
    assert all(item.metadata.get("verified") != "true" for item in result.observations)
    return result


def _ids(tmp_path: Path, source: str) -> set[str]:
    result = _result(tmp_path, {"Contract.sol": source})
    return {item.rule_id for item in result.observations}


def _summaries(tmp_path: Path, source: str, rule_id: str) -> list[str]:
    result = _result(tmp_path, {"Contract.sol": source})
    return [item.summary for item in result.observations if item.rule_id == rule_id]


def test_initializer_requires_a_prior_state_check() -> None:
    assert initializer_is_protected("{ require(!initialized); initialized = true; _; }") is True
    assert initializer_is_protected("{ if (initialized) revert(); initialized = true; _; }") is True
    assert initializer_is_protected("{ require(version == 0); version = 1; _; }") is True
    assert initializer_is_protected(
        "{ require(_initialized < version); _initialized = version; _; }"
    )
    assert initializer_is_protected("{ initialized = true; _; }") is False
    assert initializer_is_protected("{ version = 1; _; }") is False
    assert initializer_is_protected("{ _; require(!initialized); initialized = true; }") is False
    assert initializer_is_protected("{ _; }") is False
    nested = initializer_is_protected(
        "{ _init(); _; }",
        {"_init": "function _init() internal { require(!initialized); initialized = true; }"},
    )
    assert nested is True


def test_fake_initializers_are_findings(tmp_path: Path) -> None:
    assign = """
    pragma solidity ^0.8.20;
    contract Boot {
        bool initialized;
        modifier initializer() { initialized = true; _; }
        function initialize() external initializer {}
    }
    """
    version = """
    pragma solidity ^0.8.20;
    contract Boot {
        uint version;
        modifier initializer() { version = 1; _; }
        function initialize() external initializer {}
    }
    """
    real = """
    pragma solidity ^0.8.20;
    contract Boot {
        bool initialized;
        modifier initializer() { require(!initialized); initialized = true; _; }
        function initialize() external initializer {}
    }
    """
    assert "sol.initializer" in _ids(tmp_path, assign)
    assert "sol.initializer" in _ids(tmp_path, version)
    assert "sol.initializer" not in _ids(tmp_path, real)


def test_reentrancy_guard_needs_the_full_sequence() -> None:
    assert reentrancy_guard_holds("{ require(!locked); locked = true; _; locked = false; }")
    assert reentrancy_guard_holds(
        "{ require(_status != _ENTERED); _status = _ENTERED; _; _status = _NOT_ENTERED; }"
    )
    assert reentrancy_guard_holds("{ locked = true; _; }") is False
    assert reentrancy_guard_holds("{ require(msg.sender == owner); _; }") is False
    assert reentrancy_guard_holds("{ locked = computeLock(); _; }") is False
    assert reentrancy_guard_holds("{ _; }") is False


def test_incomplete_locks_do_not_suppress_reentrancy(tmp_path: Path) -> None:
    def vault(modifier: str) -> str:
        return f"""
        pragma solidity ^0.8.20;
        contract V {{
            bool locked;
            mapping(address => uint) balances;
            modifier nonReentrant() {{ {modifier} }}
            function withdraw() external nonReentrant {{
                uint amount = balances[msg.sender];
                msg.sender.call{{value: amount}}("");
                balances[msg.sender] = 0;
            }}
        }}
        """

    assert "sol.reentrancy" in _ids(tmp_path, vault("locked = true; _;"))
    assert "sol.reentrancy" in _ids(tmp_path, vault("require(msg.sender == owner); _;"))
    assert "sol.reentrancy" in _ids(tmp_path, vault("locked = computeLock(); _;"))
    assert "sol.reentrancy" not in _ids(
        tmp_path, vault("require(!locked); locked = true; _; locked = false;")
    )


def test_inherited_modifiers_are_not_guessed(tmp_path: Path) -> None:
    ownable = """
    pragma solidity ^0.8.20;
    contract Ownable {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    """
    child = """
    pragma solidity ^0.8.20;
    import "./Ownable.sol";
    contract Child is Ownable {
        address owner;
        function setOwner(address next) external onlyOwner { owner = next; }
    }
    """
    same_file = """
    pragma solidity ^0.8.20;
    contract Ownable {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    contract Child is Ownable {
        address owner;
        function setOwner(address next) external onlyOwner { owner = next; }
    }
    """
    ambiguous = """
    pragma solidity ^0.8.20;
    contract A {
        address owner;
        modifier onlyOwner() { require(msg.sender == owner); _; }
    }
    contract B {
        modifier onlyOwner() { _; }
    }
    contract Child is A, B {
        address owner;
        function setOwner(address next) external onlyOwner { owner = next; }
    }
    """
    other = """
    pragma solidity ^0.8.20;
    contract Ownable {
        modifier onlyOwner() { _; }
    }
    """
    inherited = _result(tmp_path, {"Ownable.sol": ownable, "Child.sol": child})
    assert "sol.missing_authorization" not in {item.rule_id for item in inherited.observations}
    assert "sol.missing_authorization" not in _ids(tmp_path, same_file)
    assert "sol.missing_authorization" in _ids(tmp_path, ambiguous)
    dup = _result(
        tmp_path,
        {
            "Ownable.sol": ownable,
            "Other.sol": other,
            "Child.sol": child,
        },
    )
    assert "sol.missing_authorization" in {item.rule_id for item in dup.observations}
    parsed = [
        parse_source("solidity", tmp_path / "Ownable.sol", ownable),
        parse_source("solidity", tmp_path / "Other.sol", other),
        parse_source("solidity", tmp_path / "Child.sol", child),
    ]
    graphs = {graph.file_path: graph for graph in parsed}
    child_graph = next(graph for graph in parsed if graph.file_path.endswith("Child.sol"))
    resolution = ModifierIndex.from_graphs(graphs).resolve(child_graph, "Child", "onlyOwner")
    assert resolution.status == "ambiguous"


def test_nonce_consumption_respects_order_and_identity() -> None:
    after = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        h = keccak256(abi.encode(nonce));
        address signer = ecrecover(h, v, r, s);
        nonce += 1;
        return signer;
    }
    """
    before = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonce += 1;
        h = keccak256(abi.encode(DOMAIN_SEPARATOR, address(this), nonce));
        return ecrecover(h, v, r, s);
    }
    """
    unrelated = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonce += 1;
        return ecrecover(h, v, r, s);
    }
    """
    constant = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonce = 1;
        h = keccak256(abi.encode(nonce));
        return ecrecover(h, v, r, s);
    }
    """
    dead = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        h = keccak256(abi.encode(nonce));
        return ecrecover(h, v, r, s);
        nonce += 1;
    }
    """
    wrong_index = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        h = keccak256(abi.encode(nonces[owner]));
        nonces[attacker] += 1;
        return ecrecover(h, v, r, s);
    }
    """
    assert signature_nonce_order(after) == "digest-verify-consume"
    assert signature_replay_gap(after) is None
    assert signature_nonce_order(before) == "consume-digest-verify"
    assert signature_replay_gap(before) is None
    assert signature_nonce_order(unrelated) == "consume-unrelated-verify"
    assert signature_replay_gap(unrelated) is not None
    assert signature_replay_gap(constant) is not None
    assert signature_replay_gap(dead) is not None
    assert signature_replay_gap(wrong_index) is not None


def test_oracle_freshness_stays_with_its_call() -> None:
    mixed = """
    function price() external view returns (int) {
        (, int answer1,, uint updatedAt1,) = feed.latestRoundData();
        (, int answer2,, uint updatedAt2,) = other.latestRoundData();
        require(updatedAt1 != 0);
        return answer2;
    }
    """
    paired = """
    function price() external view returns (int) {
        (, int answer1,, uint updatedAt1,) = feed.latestRoundData();
        (, int answer2,, uint updatedAt2,) = other.latestRoundData();
        require(block.timestamp - updatedAt1 < 1 hours);
        require(block.timestamp - updatedAt2 < 1 hours);
        return answer1 + answer2;
    }
    """
    nonzero = """
    function price() external view returns (int) {
        (, int answer1,, uint updatedAt1,) = feed.latestRoundData();
        (, int answer2,, uint updatedAt2,) = other.latestRoundData();
        require(updatedAt1 != 0);
        require(updatedAt2 != 0);
        return answer1 + answer2;
    }
    """
    latest = """
    function price() external view returns (int) {
        int answer = feed.latestAnswer();
        return answer;
    }
    """
    assert oracle_freshness_protects(mixed) is False
    assert oracle_freshness_protects(paired) is True
    assert oracle_freshness_protects(nonzero) is False
    assert oracle_freshness_protects(latest) is False


def test_erc20_classification_is_not_the_method_name(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    interface IERC20 {
        function transfer(address to, uint256 amount) external returns (bool);
        function transferFrom(address from, address to, uint256 amount) external returns (bool);
        function balanceOf(address account) external view returns (uint256);
        function totalSupply() external view returns (uint256);
    }
    contract FakeERC20 {
        function transfer(address a, uint b) external {}
    }
    contract User {
        function typed(IERC20 token, address to, uint amount) external {
            token.transfer(to, amount);
        }
        function unknown(address foo, address a, uint b) external {
            foo.transfer(a, b);
        }
        function fake(FakeERC20 foo) external {
            foo.transfer(address(1), 1);
        }
    }
    """
    graph = parse_source("solidity", tmp_path / "User.sol", source)
    model = analyze_defi(graph)
    by_function = {item.function: item for item in model.interactions}
    assert by_function["typed"].classification == "erc20"
    assert by_function["typed"].confidence == "strong"
    assert by_function["unknown"].classification == "unknown_external"
    assert by_function["fake"].classification == "unknown_external"
    assert "sol.fee_on_transfer" not in _ids(tmp_path, source)


def test_fee_on_transfer_follows_balance_deltas(tmp_path: Path) -> None:
    credited = """
    pragma solidity ^0.8.20;
    interface IERC20 {
        function transferFrom(address from, address to, uint256 amount) external returns (bool);
        function balanceOf(address account) external view returns (uint256);
    }
    contract Vault {
        mapping(address => uint) shares;
        function deposit(IERC20 token, uint amount) external {
            token.transferFrom(msg.sender, address(this), amount);
            shares[msg.sender] += amount;
        }
    }
    """
    measured = """
    pragma solidity ^0.8.20;
    interface IERC20 {
        function transferFrom(address from, address to, uint256 amount) external returns (bool);
        function balanceOf(address account) external view returns (uint256);
    }
    contract Vault {
        mapping(address => uint) shares;
        function deposit(IERC20 token, uint amount) external {
            uint beforeBal = token.balanceOf(address(this));
            token.transferFrom(msg.sender, address(this), amount);
            uint afterBal = token.balanceOf(address(this));
            shares[msg.sender] += afterBal - beforeBal;
        }
    }
    """
    summaries = _summaries(tmp_path, credited, "sol.fee_on_transfer")
    assert summaries
    assert "balance delta" in summaries[0]
    assert "sol.fee_on_transfer" not in _ids(tmp_path, measured)


def test_donation_follows_share_conversion(tmp_path: Path) -> None:
    empty = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint totalSupply;
        function deposit(address token, uint assets) external view returns (uint) {
            uint totalAssets = token.balanceOf(address(this));
            return assets * totalSupply / totalAssets;
        }
    }
    """
    virtual = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint totalSupply;
        function deposit(address token, uint assets) external view returns (uint) {
            uint totalAssets = token.balanceOf(address(this));
            return (assets + 1) * (totalSupply + 1e3) / (totalAssets + 1);
        }
    }
    """
    words = """
    pragma solidity ^0.8.20;
    contract Vault {
        function demo(uint totalSupply) external pure returns (uint) {
            string memory note = "totalSupply";
            return totalSupply / 2;
        }
    }
    """
    summaries = _summaries(tmp_path, empty, "sol.donation_inflation")
    assert summaries
    assert "donation" in summaries[0].lower() or "exchange rate" in summaries[0]
    assert "sol.donation_inflation" not in _ids(tmp_path, virtual)
    assert "sol.donation_inflation" not in _ids(tmp_path, words)


def test_erc4626_preview_and_zero_results(tmp_path: Path) -> None:
    mismatched = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint totalSupply;
        uint totalAssets;
        function previewDeposit(uint assets) external view returns (uint) {
            return assets * totalSupply / totalAssets;
        }
        function deposit(uint assets) external returns (uint) {
            return assets * totalAssets / totalSupply;
        }
    }
    """
    protected = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint totalSupply;
        uint totalAssets;
        mapping(address => uint) shares;
        function previewDeposit(uint assets) external view returns (uint) {
            return assets * (totalSupply + 1) / (totalAssets + 1);
        }
        function deposit(uint assets, uint minShares) external returns (uint) {
            uint minted = assets * (totalSupply + 1) / (totalAssets + 1);
            require(minted >= minShares);
            shares[msg.sender] += minted;
            return minted;
        }
    }
    """
    assert "sol.erc4626" in _ids(tmp_path, mismatched)
    zero = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint totalSupply;
        uint totalAssets;
        function deposit(uint assets) external returns (uint) {
            return assets * totalSupply / totalAssets;
        }
    }
    """
    assert "sol.erc4626" in _ids(tmp_path, zero)
    assert "sol.erc4626" not in _ids(tmp_path, protected)


def test_rounding_direction_is_economic(tmp_path: Path) -> None:
    attacker = """
    pragma solidity ^0.8.20;
    contract Pool {
        function withdraw(uint shares, uint supply, uint balance) external pure returns (uint) {
            return shares * balance / supply + 1;
        }
    }
    """
    victim = """
    pragma solidity ^0.8.20;
    contract Pool {
        function price(uint share, uint a, uint b, uint c) external pure returns (uint) {
            return a / b * c;
        }
    }
    """
    harmless = """
    pragma solidity ^0.8.20;
    contract Pool {
        function fee(uint amount) external pure returns (uint) {
            return amount * 30 / 10000;
        }
    }
    """
    looping = """
    pragma solidity ^0.8.20;
    contract Pool {
        function accrue(uint rounds, uint credit, uint rate) external pure returns (uint) {
            for (uint i; i < rounds; i++) { credit = credit * rate / 10000; }
            return credit;
        }
    }
    """
    safe = """
    pragma solidity ^0.8.20;
    contract Pool {
        function identity(uint amount) external pure returns (uint) { return amount; }
    }
    """
    assert "sol.rounding_direction" in _ids(tmp_path, attacker)
    assert "sol.rounding" in _ids(tmp_path, victim)
    assert "sol.rounding_direction" in _ids(tmp_path, victim)
    assert "sol.rounding_direction" not in _ids(tmp_path, harmless)
    assert "sol.rounding_direction" in _ids(tmp_path, looping)
    assert "sol.rounding_direction" not in _ids(tmp_path, safe)


def test_slippage_bounds_must_apply_to_the_output(tmp_path: Path) -> None:
    ignored = """
    pragma solidity ^0.8.20;
    contract Pool {
        function swap(uint amount, uint minAmountOut) external pure returns (uint) {
            return amount * 2;
        }
    }
    """
    wrong = """
    pragma solidity ^0.8.20;
    contract Pool {
        function swap(uint amount, uint minAmountOut) external pure returns (uint) {
            require(minAmountOut > 0);
            return amount * 2;
        }
    }
    """
    late = """
    pragma solidity ^0.8.20;
    contract Pool {
        mapping(address => uint) shares;
        function deposit(uint assets, uint minShares) external returns (uint) {
            uint minted = assets;
            shares[msg.sender] += minted;
            require(minted >= minShares);
            return minted;
        }
    }
    """
    checked = """
    pragma solidity ^0.8.20;
    contract Pool {
        mapping(address => uint) shares;
        function deposit(uint assets, uint minShares) external returns (uint) {
            uint minted = assets;
            require(minted >= minShares);
            shares[msg.sender] += minted;
            return minted;
        }
    }
    """
    bare = """
    pragma solidity ^0.8.20;
    contract Pool {
        function swap(uint amount) external pure returns (uint) { return amount; }
    }
    """
    assert "sol.slippage" in _ids(tmp_path, ignored)
    assert "sol.slippage" in _ids(tmp_path, wrong)
    assert "sol.slippage" in _ids(tmp_path, late)
    assert "sol.slippage" not in _ids(tmp_path, checked)
    assert "sol.slippage" not in _ids(tmp_path, bare)


def test_oracle_accounting_follows_the_economic_path(tmp_path: Path) -> None:
    stale = """
    pragma solidity ^0.8.20;
    contract Lend {
        function borrow(address feed, uint amount, uint debt) external view returns (uint) {
            (, int answer,, uint updatedAt1,) = feed.latestRoundData();
            (, int other,, uint updatedAt2,) = feed.latestRoundData();
            require(updatedAt1 != 0);
            return uint(other) * debt / 1e8;
        }
    }
    """
    positive = """
    pragma solidity ^0.8.20;
    contract Lend {
        function health(address feed, uint collateral) external view returns (uint) {
            (, int answer,, uint updatedAt,) = feed.latestRoundData();
            require(block.timestamp - updatedAt < 1 hours);
            require(answer > 0);
            return uint(answer) * collateral / 1e8;
        }
    }
    """
    scales = """
    pragma solidity ^0.8.20;
    contract Lend {
        function health(address feed, uint collateral) external view returns (uint) {
            (, int answer,, uint updatedAt,) = feed.latestRoundData();
            require(updatedAt != 0 && answer > 0);
            return uint(answer) * collateral / 1e8 * 1e18;
        }
    }
    """
    summaries = _summaries(tmp_path, stale, "sol.oracle_accounting")
    assert summaries
    assert "observation" in summaries[0]
    assert "sol.oracle_accounting" not in _ids(tmp_path, positive)
    assert "sol.oracle_accounting" in _ids(tmp_path, scales)


def test_lending_and_amm_relationships(tmp_path: Path) -> None:
    repay = """
    pragma solidity ^0.8.20;
    contract Lend {
        mapping(address => uint) debt;
        function repay(uint amount) external {
            debt[msg.sender] -= amount;
        }
    }
    """
    repaid = """
    pragma solidity ^0.8.20;
    interface IERC20 {
        function transferFrom(address from, address to, uint256 amount) external returns (bool);
        function balanceOf(address) external view returns (uint256);
    }
    contract Lend {
        mapping(address => uint) debt;
        function repay(IERC20 token, uint amount) external {
            token.transferFrom(msg.sender, address(this), amount);
            debt[msg.sender] -= amount;
        }
    }
    """
    health = """
    pragma solidity ^0.8.20;
    contract Lend {
        mapping(address => uint) debt;
        function borrow(uint amount, uint collateral, uint price) external {
            uint hf = collateral * price / debt[msg.sender];
            debt[msg.sender] += amount;
            require(hf > 1);
        }
    }
    """
    liquidate = """
    pragma solidity ^0.8.20;
    contract Lend {
        function liquidate(uint collateral, uint debt) external pure returns (uint) {
            return collateral * 1e18 / debt * 1e6;
        }
    }
    """
    stale_reserve = """
    pragma solidity ^0.8.20;
    contract Pool {
        uint reserve0;
        uint reserve1;
        function swap(address token, uint amount) external returns (uint amountOut) {
            amountOut = amount * reserve1 / reserve0;
            token.transfer(msg.sender, amountOut);
        }
    }
    """
    updated = """
    pragma solidity ^0.8.20;
    contract Pool {
        uint reserve0;
        uint reserve1;
        function quote(uint amount) external view returns (uint amountOut) {
            amountOut = amount * reserve1 / reserve0;
        }
    }
    """
    assert "sol.lending" in _ids(tmp_path, repay)
    assert "sol.lending" not in _ids(tmp_path, repaid)
    assert "sol.lending" in _ids(tmp_path, health)
    assert "sol.lending" in _ids(tmp_path, liquidate)
    assert "sol.amm" in _ids(tmp_path, stale_reserve)
    assert "sol.amm" not in _ids(tmp_path, updated)


def test_token_callback_reentrancy_and_cei(tmp_path: Path) -> None:
    unsafe = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint) shares;
        function deposit(address token, uint amount) external {
            token.transferFrom(msg.sender, address(this), amount);
            shares[msg.sender] += amount;
        }
    }
    """
    cei = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint) shares;
        function deposit(address token, uint amount) external {
            shares[msg.sender] += amount;
            token.transferFrom(msg.sender, address(this), amount);
        }
    }
    """
    guarded = """
    pragma solidity ^0.8.20;
    contract Vault {
        bool locked;
        mapping(address => uint) shares;
        modifier nonReentrant() { require(!locked); locked = true; _; locked = false; }
        function deposit(address token, uint amount) external nonReentrant {
            token.transferFrom(msg.sender, address(this), amount);
            shares[msg.sender] += amount;
        }
    }
    """
    assert "sol.defi_reentrancy" in _ids(tmp_path, unsafe)
    assert "sol.defi_reentrancy" not in _ids(tmp_path, cei)
    assert "sol.defi_reentrancy" not in _ids(tmp_path, guarded)


def test_permit_binding(tmp_path: Path) -> None:
    replay = """
    pragma solidity ^0.8.20;
    contract Token {
        mapping(address => mapping(address => uint)) allowance;
        uint nonce;
        function permit(address owner, address spender, uint value, uint8 v, bytes32 r, bytes32 s) external {
            bytes32 digest = keccak256(abi.encode(owner, spender, value, nonce));
            require(ecrecover(digest, v, r, s) == owner);
            allowance[owner][spender] = value;
        }
    }
    """
    wrong_spender = """
    pragma solidity ^0.8.20;
    contract Token {
        mapping(address => mapping(address => uint)) allowance;
        uint nonce;
        bytes32 DOMAIN_SEPARATOR;
        function permit(address owner, address spender, uint value, uint8 v, bytes32 r, bytes32 s) external {
            nonce += 1;
            bytes32 digest = keccak256(abi.encode(DOMAIN_SEPARATOR, address(this), owner, value, nonce));
            require(ecrecover(digest, v, r, s) == owner);
            allowance[owner][spender] = value;
        }
    }
    """
    bound = """
    pragma solidity ^0.8.20;
    contract Token {
        mapping(address => mapping(address => uint)) allowance;
        uint nonce;
        bytes32 DOMAIN_SEPARATOR;
        function permit(address owner, address spender, uint value, uint8 v, bytes32 r, bytes32 s) external {
            nonce += 1;
            bytes32 digest = keccak256(abi.encode(DOMAIN_SEPARATOR, address(this), owner, spender, value, nonce));
            require(ecrecover(digest, v, r, s) == owner);
            allowance[owner][spender] = value;
        }
    }
    """
    assert "sol.approval" in _ids(tmp_path, replay)
    assert "sol.approval" in _ids(tmp_path, wrong_spender)
    assert "sol.approval" not in _ids(tmp_path, bound)
    assert "sol.signature_replay" not in _ids(tmp_path, bound)


def test_economic_transitions_are_reusable(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract Vault {
        mapping(address => uint) shares;
        mapping(address => uint) debt;
        uint totalSupply;
        function deposit(uint amount) external { shares[msg.sender] += amount; totalSupply += amount; }
        function borrow(uint amount) external { debt[msg.sender] += amount; }
        function repay(uint amount) external { debt[msg.sender] -= amount; }
    }
    """
    graph = parse_source("solidity", tmp_path / "Vault.sol", source)
    model = analyze_defi(graph)
    actions = {item.function: item for item in model.transitions}
    assert actions["deposit"].action == "deposit"
    assert "shares:credit" in actions["deposit"].user
    assert "totalShares:increase" in actions["deposit"].protocol
    assert "debt:increase" in actions["borrow"].user
    assert "debt:decrease" in actions["repay"].user


def test_adversarial_names_do_not_create_defi_findings(tmp_path: Path) -> None:
    source = """
    pragma solidity ^0.8.20;
    contract FakeERC20 {
        function transfer(address a, uint b) external {}
    }
    contract Noise {
        address owner;
        modifier nonReentrant() { require(msg.sender == owner); _; }
        function unrelated() external {
            uint updatedAt = block.timestamp;
            string memory note = "totalSupply";
        }
        function demo(uint totalSupply) external pure returns (uint) {
            return totalSupply / 2;
        }
        function ping(FakeERC20 foo) external nonReentrant {
            foo.transfer(address(1), 1);
        }
    }
    """
    rules = _ids(tmp_path, source)
    for rule_id in (
        "sol.donation_inflation",
        "sol.fee_on_transfer",
        "sol.erc4626",
        "sol.rounding_direction",
        "sol.slippage",
        "sol.oracle_accounting",
        "sol.lending",
        "sol.amm",
        "sol.approval",
        "sol.defi_reentrancy",
        "sol.stale_oracle",
    ):
        assert rule_id not in rules


def test_reentrancy_clear_must_cover_every_normal_exit() -> None:
    assert (
        reentrancy_guard_holds(
            "{ require(!locked); locked = true; _; if (condition) { locked = false; } }"
        )
        is False
    )
    assert (
        reentrancy_guard_holds(
            "{ require(!locked); locked = true; _; if (condition) { locked = false; } else { revert(); } }"
        )
        is False
    )
    assert (
        reentrancy_guard_holds(
            "{ require(!locked || msg.sender == owner); locked = true; _; locked = false; }"
        )
        is False
    )
    assert reentrancy_guard_holds("{ require(!other); locked = true; _; locked = false; }") is False
    assert (
        reentrancy_guard_holds("{ require(!locked); locked = true; _; return; locked = false; }")
        is False
    )
    assert reentrancy_guard_holds(
        "{ require(!locked); locked = true; _; if (ok) { locked = false; } else { locked = false; } }"
    )


def test_initializer_rejects_replayable_version_checks() -> None:
    assert initializer_is_protected("{ require(version < 999); version = 1; _; }") is False
    assert (
        initializer_is_protected(
            "{ require(!initialized || msg.sender == owner); initialized = true; _; }"
        )
        is False
    )
    assert initializer_is_protected("{ require(version > 0); version = 1; _; }") is False
    assert initializer_is_protected("{ require(version == 0); version = 1; _; }") is True


def test_nonce_consumption_is_branch_and_index_specific() -> None:
    swapped = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        nonces[owner] += 1;
        h = keccak256(abi.encode(nonces[attacker]));
        return ecrecover(h, v, r, s);
    }
    """
    one_branch = """
    function recover(bool flag, bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        if (flag) { nonce += 1; }
        h = keccak256(abi.encode(nonce));
        return ecrecover(h, v, r, s);
    }
    """
    missed_path = """
    function recover(bool flag, bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        if (flag) { nonce += 1; return address(0); }
        h = keccak256(abi.encode(nonce));
        return ecrecover(h, v, r, s);
    }
    """
    split = """
    function recover(bool flag, bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        if (flag) {
            h = keccak256(abi.encode(nonce));
            return ecrecover(h, v, r, s);
        }
        nonce += 1;
        h = keccak256(abi.encode(nonce));
        return ecrecover(h, v, r, s);
    }
    """
    other_counter = """
    function recover(bytes32 h, uint8 v, bytes32 r, bytes32 s) external returns (address) {
        fees += 1;
        h = keccak256(abi.encode(nonce));
        return ecrecover(h, v, r, s);
    }
    """
    assert signature_replay_gap(swapped) is not None
    assert signature_replay_gap(one_branch) is not None
    assert signature_replay_gap(missed_path) is not None
    assert signature_replay_gap(split) is not None
    assert signature_replay_gap(other_counter) is not None


def test_direction_fee_vault_and_oracle_regressions(tmp_path: Path) -> None:
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
    inbound = """
    pragma solidity ^0.8.20;
    interface IERC20 {
        function transferFrom(address from, address to, uint256 amount) external returns (bool);
    }
    contract Lend {
        mapping(address => uint) debt;
        function repay(IERC20 token, uint amount) external {
            token.transferFrom(msg.sender, address(this), amount);
            debt[msg.sender] -= amount;
        }
    }
    """
    unrelated_math = """
    pragma solidity ^0.8.20;
    contract Vault {
        function deposit(uint assets) external pure returns (uint) {
            uint fee = assets / 100;
            return assets - fee;
        }
    }
    """
    twins = """
    pragma solidity ^0.8.20;
    contract VaultA {
        uint totalSupply;
        uint totalAssets;
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
        uint totalSupply;
        uint totalAssets;
        function previewDeposit(uint assets) external view returns (uint) {
            return assets * totalAssets / totalSupply;
        }
        function deposit(uint assets) external view returns (uint) {
            return assets * totalSupply / totalAssets;
        }
    }
    """
    negative = """
    pragma solidity ^0.8.20;
    contract Lend {
        function health(address feed, uint collateral) external view returns (uint) {
            (, int answer,, uint updatedAt,) = feed.latestRoundData();
            require(answer != 0);
            require(block.timestamp - updatedAt < 1 hours);
            return uint(answer) * collateral / 1e8;
        }
    }
    """
    weak_fresh = """
    pragma solidity ^0.8.20;
    contract Lend {
        function health(address feed, uint collateral) external view returns (uint) {
            (, int answer,, uint updatedAt,) = feed.latestRoundData();
            require(updatedAt != 0 && answer > 0);
            return uint(answer) * collateral / 1e8;
        }
    }
    """
    local_token = """
    pragma solidity ^0.8.20;
    interface IERC20 {
        function transferFrom(address from, address to, uint256 amount) external returns (bool);
        function balanceOf(address) external view returns (uint256);
    }
    contract Vault {
        mapping(address => uint) shares;
        function deposit(address rawToken, uint amount) external {
            IERC20 token = IERC20(rawToken);
            token.transferFrom(msg.sender, address(this), amount);
            shares[msg.sender] += amount;
        }
    }
    """
    deadline = """
    pragma solidity ^0.8.20;
    contract Pool {
        function swap(uint amount, uint deadline) external pure returns (uint) {
            require(deadline > 0);
            return amount;
        }
    }
    """
    assert "sol.lending" in _ids(tmp_path, outbound)
    assert "sol.lending" not in _ids(tmp_path, inbound)
    assert "sol.erc4626" not in _ids(tmp_path, unrelated_math)
    graph = parse_source("solidity", tmp_path / "Twins.sol", twins)
    model = analyze_defi(graph)
    vault_issues = [item for item in model.issues if item.rule_id == "sol.erc4626"]
    assert vault_issues
    assert all(item.contract == "VaultB" for item in vault_issues)
    assert "sol.oracle_accounting" in _ids(tmp_path, negative)
    assert "sol.oracle_accounting" in _ids(tmp_path, weak_fresh)
    local = analyze_defi(parse_source("solidity", tmp_path / "Local.sol", local_token))
    typed = next(item for item in local.interactions if item.function == "deposit")
    assert typed.classification == "erc20"
    assert typed.confidence == "strong"
    assert typed.direction == "in"
    assert "sol.slippage" not in _ids(tmp_path, deadline)
    assert defi_context_active() is False


def test_scans_do_not_share_defi_cache(tmp_path: Path) -> None:
    first = """
    pragma solidity ^0.8.20;
    contract Vault {
        uint totalSupply;
        uint totalAssets;
        function deposit(uint assets) external view returns (uint) {
            return assets * totalSupply / totalAssets;
        }
    }
    """
    second = """
    pragma solidity ^0.8.20;
    contract Quiet {
        function ping() external pure returns (uint) { return 1; }
    }
    """
    assert "sol.erc4626" in _ids(tmp_path, first)
    assert defi_context_active() is False
    other = tmp_path / "other"
    other.mkdir()
    assert "sol.erc4626" not in _ids(other, second)
    assert defi_context_active() is False

"""Boundary, decision-table and fail-closed tests for the deterministic gates.

Techniques follow docs/test-coverage.md: 3-value boundaries (below / on /
above) for required clock frequency and resource budgets, equivalence
classes for clock-net matching, and decision tables for pin rules per
device family.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga.contract import Budget, FpgaContract, load_contract
from fpga.devices import load_profile
from fpga.gates import check_pins, check_timing, check_utilization

from .conftest import read_json, write_json

ULX3S_NET = "$glbnet$clk_25mhz"


def _below(value: float) -> float:
    return math.nextafter(value, -math.inf)


def _above(value: float) -> float:
    return math.nextafter(value, math.inf)


def _timing(contract: FpgaContract, fmax: dict[str, dict[str, float]]) -> tuple[str, str]:
    result = check_timing(contract, {"fmax": fmax})
    return result.status, result.detail


# ----------------------------------------------------------------- timing


def test_required_clock_three_value_boundary(ulx3s: Path) -> None:
    contract = load_contract(ulx3s)
    statuses = [
        _timing(contract, {ULX3S_NET: {"achieved": achieved, "constraint": 25.0}})[0]
        for achieved in (_below(25.0), 25.0, _above(25.0))
    ]
    assert statuses == ["fail", "pass", "pass"]


def test_unconstrained_net_three_value_boundary(ulx3s: Path) -> None:
    contract = load_contract(ulx3s)
    statuses = [
        _timing(
            contract,
            {
                ULX3S_NET: {"achieved": 100.0, "constraint": 25.0},
                "pll_out": {"achieved": achieved, "constraint": 50.0},
            },
        )[0]
        for achieved in (_below(50.0), 50.0, _above(50.0))
    ]
    assert statuses == ["fail", "pass", "pass"]


@pytest.mark.parametrize(
    ("net", "matched"),
    [
        (ULX3S_NET, True),
        ("clk_25mhz", True),
        ("clk_25mhz$TRELLIS_IO_IN", True),
        ("u0.clk_25mhz", True),
        ("clk_25mhzA", False),
        ("xclk_25mhz", False),
    ],
)
def test_clock_net_matching_equivalence_classes(ulx3s: Path, net: str, matched: bool) -> None:
    status, detail = _timing(load_contract(ulx3s), {net: {"achieved": 100.0, "constraint": 1.0}})
    assert (status == "pass") == matched
    assert ("no timing data" in detail) == (not matched)


def test_missing_achieved_counts_as_zero(ulx3s: Path) -> None:
    status, detail = _timing(load_contract(ulx3s), {ULX3S_NET: {"constraint": 25.0}})
    assert status == "fail"
    assert "0.00 MHz" in detail


def test_no_clocks_is_not_applicable(ulx3s: Path) -> None:
    contract = load_contract(ulx3s).model_copy(update={"clocks": []})
    assert check_timing(contract, {"fmax": {}}).status == "not_applicable"


# ------------------------------------------------------------ utilization


def _util(contract: FpgaContract, cells: dict[str, dict[str, int]]) -> tuple[str, str]:
    profile = load_profile(contract.device.profile)
    result = check_utilization(contract, profile, {"utilization": cells})
    return result.status, result.detail


# logic_pct defaults to 80 %.
@pytest.mark.parametrize(("used", "status"), [(799, "pass"), (800, "pass"), (801, "fail")])
def test_logic_budget_three_value_boundary(ulx3s: Path, used: int, status: str) -> None:
    contract = load_contract(ulx3s)
    assert _util(contract, {"TRELLIS_COMB": {"used": used, "available": 1000}})[0] == status


@pytest.mark.parametrize(("used", "status"), [(499, "pass"), (500, "pass"), (501, "fail")])
def test_authored_budget_three_value_boundary(ulx3s: Path, used: int, status: str) -> None:
    data = read_json(ulx3s)
    data["build"]["budget"] = {"ram_pct": 50}
    write_json(ulx3s, data)
    contract = load_contract(ulx3s)
    assert _util(contract, {"DP16KD": {"used": used, "available": 1000}})[0] == status


# Decision table: resource class x availability x usage.
@pytest.mark.parametrize(
    ("cell", "used", "available", "status"),
    [
        ("TRELLIS_COMB", 0, 0, "pass"),
        ("MYSTERY_CELL", 0, 10, "pass"),
        ("MYSTERY_CELL", 1, 10, "fail"),
        ("TRELLIS_COMB", 1, 0, "fail"),
        ("TRELLIS_IO", 1, 0, "pass"),
        ("TRELLIS_IO", 10, 10, "pass"),
        ("JTAGG", 5, 1, "pass"),
        ("EHXPLLL", 2, 2, "pass"),
    ],
)
def test_utilization_decision_table(
    ulx3s: Path, cell: str, used: int, available: int, status: str
) -> None:
    contract = load_contract(ulx3s)
    assert _util(contract, {cell: {"used": used, "available": available}})[0] == status


def test_empty_utilization_fails_closed(ulx3s: Path) -> None:
    status, detail = _util(load_contract(ulx3s), {})
    assert status == "fail"
    assert "no utilization" in detail


@pytest.mark.parametrize(
    ("value", "ok"), [(0.0, False), (math.ulp(0.0), True), (100.0, True), (_above(100.0), False)]
)
def test_budget_percent_bounds(value: float, ok: bool) -> None:
    if ok:
        assert Budget.model_validate({"logic_pct": value}).limit("logic") == value
    else:
        with pytest.raises(ValidationError):
            Budget.model_validate({"logic_pct": value})


def test_budget_has_no_limit_for_other_resources() -> None:
    assert Budget().limit("other") is None


# ------------------------------------------------------------------- pins


def _pins(contract_path: Path, port: str, **fields: Any) -> tuple[str, str, list[str]]:
    data = read_json(contract_path)
    pin = next(p for p in data["pins"] if p["port"] == port)
    for key, value in fields.items():
        if value is None:
            pin.pop(key, None)
        else:
            pin[key] = value
    write_json(contract_path, data)
    contract = load_contract(contract_path)
    result = check_pins(contract, load_profile(contract.device.profile))
    return result.status, result.detail, result.evidence


# Decision table: family x io_standard x pull.
@pytest.mark.parametrize(
    ("fixture", "port", "fields", "needle"),
    [
        ("uart_echo", "tx", {"io_standard": "LVCMOS33"}, "cannot set an I/O standard"),
        ("uart_echo", "tx", {"pull": "down"}, "no pull-down"),
        ("uart_echo", "tx", {"pull": "up"}, None),
        ("ulx3s", "led[0]", {"io_standard": "HSTL18_I"}, "not in"),
        ("ulx3s", "led[0]", {"io_standard": "LVDS"}, None),
        ("ulx3s", "led[0]", {"io_standard": None}, None),
        ("ulx3s", "led[0]", {"pull": "down"}, None),
        ("ulx3s", "led[0]", {"package_pin": "Z99"}, "not a user I/O"),
        ("ulx3s", "led[0]", {"package_pin": "R2"}, "config pin"),
        (
            "ulx3s",
            "led[0]",
            {"package_pin": "R2", "acknowledge": ["config"], "rationale": "x"},
            None,
        ),
    ],
)
def test_pin_rules_decision_table(
    request: pytest.FixtureRequest,
    fixture: str,
    port: str,
    fields: dict[str, Any],
    needle: str | None,
) -> None:
    status, detail, _ = _pins(request.getfixturevalue(fixture), port, **fields)
    assert status == ("pass" if needle is None else "fail")
    if needle is not None:
        assert needle in detail


def test_clock_on_general_pin_is_evidence_not_failure(ulx3s: Path) -> None:
    status, _, evidence = _pins(ulx3s, "clk_25mhz", package_pin="A12")
    assert status == "pass"
    assert any("not a dedicated clock input" in item for item in evidence)

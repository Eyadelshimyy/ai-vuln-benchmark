from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga.contract import FpgaContract, PowerEstimate, Thermal, load_contract
from fpga.gates import check_power

# 1000 LUT x 0.5 uW/MHz x 25 MHz x 0.2 = 2500 uW; 200 FF x 0.25 x 25 x 0.2 = 250 uW
REPORT: dict[str, Any] = {
    "utilization": {
        "TRELLIS_COMB": {"used": 1000, "available": 24288},
        "TRELLIS_FF": {"used": 200, "available": 24288},
        "DP16KD": {"used": 0, "available": 56},
    }
}
ESTIMATE = PowerEstimate(
    static_w=0.05,
    toggle_rate=0.2,
    uw_per_mhz={"TRELLIS_COMB": 0.5, "TRELLIS_FF": 0.25},
    source="Lattice Power Calculator 2026.1, LFE5U-25F",
)


def _contract(
    ulx3s: Path, power_w: float, estimate: PowerEstimate | None = ESTIMATE
) -> FpgaContract:
    thermal = Thermal(
        ambient_c=40,
        power_w=power_w,
        tj_max_c=100,
        theta_ja_c_per_w=30,
        source="vendor estimate",
        estimate=estimate,
    )
    return load_contract(ulx3s).model_copy(update={"thermal": thermal})


@pytest.mark.parametrize(("power_w", "status"), [(0.05275, "pass"), (0.0527, "fail")])
def test_power_estimate_boundary(ulx3s: Path, power_w: float, status: str) -> None:
    check = check_power(_contract(ulx3s, power_w), REPORT)
    assert check is not None and check.status == status
    assert "estimate 0.052750 W" in " ".join(check.evidence)
    if status == "fail":
        assert "understates the estimate" in check.detail


def test_power_gate_absent_without_estimate(ulx3s: Path) -> None:
    assert check_power(_contract(ulx3s, 1, None), REPORT) is None
    assert check_power(load_contract(ulx3s), REPORT) is None


def test_power_fails_closed(ulx3s: Path) -> None:
    contract = _contract(ulx3s, 10)
    mystery = {"utilization": {**REPORT["utilization"], "MULT18X18D": {"used": 2}}}
    check = check_power(contract, mystery)
    assert check is not None and "MULT18X18D: 2 used but no uw_per_mhz" in check.detail
    empty = check_power(contract, {})
    assert empty is not None and "no utilization" in empty.detail
    clockless = check_power(contract.model_copy(update={"clocks": []}), REPORT)
    assert clockless is not None and "no clock" in clockless.detail


@pytest.mark.parametrize(
    "bad",
    [
        {"toggle_rate": 0},
        {"toggle_rate": 1.1},
        {"static_w": -0.1},
        {"uw_per_mhz": {}},
        {"uw_per_mhz": {"TRELLIS_COMB": 0}},
        {"source": ""},
    ],
)
def test_estimate_rejects_bad_coefficients(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PowerEstimate.model_validate({**ESTIMATE.model_dump(), **bad})

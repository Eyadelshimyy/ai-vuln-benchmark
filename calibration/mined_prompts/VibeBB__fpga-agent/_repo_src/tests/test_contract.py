from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga.contract import load_contract
from fpga.devices import bundled_ids, load_profile

from .conftest import EXAMPLES, read_json, write_json


@pytest.mark.parametrize("contract", sorted(EXAMPLES.glob("*/*.fpga.json")))
def test_examples_validate(contract: Path) -> None:
    loaded = load_contract(contract)
    assert load_profile(loaded.device.profile).id == loaded.device.profile


def test_bundled_profiles() -> None:
    assert bundled_ids() == [
        "ecp5-lfe5u-25f-cabga381",
        "ecp5-lfe5u-85f-cabga381",
        "gowin-gw1nr9c-qn88p",
        "gowin-gw2a18c-pbga256",
        "gowin-gw2ar18c-qn88p",
        "ice40-up5k-sg48",
    ]
    ice40 = load_profile("ice40-up5k-sg48")
    assert ice40.family == "ice40"
    clk = ice40.pin("35")
    assert clk is not None and clk.clock
    flash = ice40.pin("14")
    assert flash is not None and flash.caution == "config"
    primer = load_profile("gowin-gw2a18c-pbga256")
    assert primer.synth_args == ["-family", "gw2a"]
    gclk = primer.pin("H11")
    assert gclk is not None and gclk.clock
    ready = primer.pin("A13")
    assert ready is not None and ready.caution == "config"
    nano20k = load_profile("gowin-gw2ar18c-qn88p")
    tms = nano20k.pin("5")
    assert tms is not None and tms.caution == "jtag"
    with pytest.raises(ValueError):
        load_profile("no-such-part")


def test_custom_profile_next_to_contract_wins(ulx3s: Path) -> None:
    profile = load_profile("ecp5-lfe5u-85f-cabga381").model_dump(mode="json")
    profile["boards"] = ["Custom board"]
    write_json(ulx3s.parent / "ecp5-lfe5u-85f-cabga381.fpga-device.json", profile)
    loaded = load_profile("ecp5-lfe5u-85f-cabga381", [ulx3s.parent])
    assert loaded.boards == ["Custom board"]


Contract = dict[str, Any]


def _extra_key(c: Contract) -> None:
    c["extra"] = 1


def _absolute_source(c: Contract) -> None:
    c["sources"][0]["path"] = "/abs/blinky.v"


def _duplicate_pin(c: Contract) -> None:
    c["pins"].append(dict(c["pins"][1]))


def _mixed_language(c: Contract) -> None:
    c["sources"].append({"path": "x.vhdl", "language": "vhdl"})


def _wrong_runner(c: Contract) -> None:
    c["simulations"][0]["runner"] = "nvc"


def _zero_clock(c: Contract) -> None:
    c["clocks"][0]["frequency_mhz"] = 0


def _future_schema(c: Contract) -> None:
    c["schema_version"] = 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_extra_key, "extra"),
        (_absolute_source, "relative"),
        (_duplicate_pin, "duplicate"),
        (_mixed_language, "all VHDL"),
        (_wrong_runner, "run on iverilog"),
        (_zero_clock, "greater than"),
        (_future_schema, "schema_version"),
    ],
)
def test_contract_rejects(ulx3s: Path, mutate: Callable[[Contract], None], message: str) -> None:
    contract = read_json(ulx3s)
    mutate(contract)
    write_json(ulx3s, contract)
    with pytest.raises(ValidationError, match=message):
        load_contract(ulx3s)

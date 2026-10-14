"""Reading the line's sensors, and never reading one by name.



The contract's claim is that adding, moving or removing a camera changes
`configs/world/sorting_line.yml` and nothing else. That claim is only worth
anything if no module reaches for a camera by its id, which is what the last
test in this file enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from clave.tracker.evidence import Role
from clave.tracker.sensors import (
    SensorError,
    SensorSpec,
    load_sensors,
    of_role,
    require_role,
)
from clave.world.config import load, require

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "world" / "sorting_line.yml"


def shipped() -> tuple[SensorSpec, ...]:
    """The sensors the shipped line declares."""
    return load_sensors(load(CONFIG))


def test_every_declared_camera_becomes_a_sensor() -> None:
    """The list in configuration is the list in the tracker."""
    raw = load(CONFIG)
    assert len(shipped()) == len(require(raw, "cameras"))


def test_a_sensor_carries_the_optics_its_parts_imply() -> None:
    """A sensor carries the optics its parts imply.

    The adapter converts pixels to belt metres, so it needs the optics, and
    they are derived from the sensor and the lens rather than restated here.
    """
    wide = require_role(shipped(), Role.DETECTION)
    assert wide.optics.camera_height == pytest.approx(1.942)
    assert wide.optics.fovy_degrees == pytest.approx(55.65, abs=0.05)
    assert wide.pixels == (2448, 2048)


def test_sensors_are_found_by_role_and_the_lookup_takes_no_id() -> None:
    """Role is the only key a consumer may use."""
    sensors = shipped()
    assert len(of_role(sensors, Role.CODE)) == 3
    # Two detection cameras: one over the sensing gate and one over the part
    # of the belt the arm reaches, because a pick made on an estimate last
    # observed at the gate is a pick made on dead reckoning.
    assert len(of_role(sensors, Role.DETECTION)) == 2
    assert of_role(sensors, Role.SPECTRAL) == ()


def test_requiring_a_role_nothing_produces_names_it() -> None:
    """Requiring a role nothing produces names it.

    A tracker configured to fuse a role no camera produces would emit records
    silently missing that evidence, which is worse than refusing to start.
    """
    with pytest.raises(SensorError, match="spectral"):
        require_role(shipped(), Role.SPECTRAL)


def test_a_camera_of_an_unknown_role_is_refused_naming_it() -> None:
    """A role the union does not define cannot be fused."""
    raw = load(CONFIG)
    raw["cameras"] = [dict(raw["cameras"][0], id="probe", role="thermal")]
    with pytest.raises(SensorError, match="thermal"):
        load_sensors(raw)


def test_two_cameras_cannot_share_one_source_id() -> None:
    """Provenance stops meaning anything if it is ambiguous."""
    raw = load(CONFIG)
    raw["cameras"] = [raw["cameras"][0], dict(raw["cameras"][1], id="gate_wide")]
    with pytest.raises(SensorError, match="gate_wide"):
        load_sensors(raw)


def test_adding_a_camera_of_an_existing_role_changes_nothing_else() -> None:
    """Adding a camera of an existing role changes nothing else.

    The non-vacuous half: asserting the shape is unchanged proves little on its
    own, so this also asserts the new sensor actually arrived under its role.
    """
    raw = load(CONFIG)
    before = {s.role for s in load_sensors(raw)}
    raw["cameras"] = [
        *raw["cameras"],
        dict(
            next(camera for camera in raw["cameras"] if camera["role"] == "code"),
            id="gate_code_fourth",
            position_meters=[-1.0, 0.16, 1.63],
        ),
    ]
    after = load_sensors(raw)
    assert {s.role for s in after} == before
    assert "gate_code_fourth" in {s.source_id for s in of_role(after, Role.CODE)}
    assert len(of_role(after, Role.CODE)) == 4


def test_no_module_names_a_camera_the_configuration_declares() -> None:
    """No module names a camera the configuration declares.

    This is the criterion's only formulation that cannot be satisfied by prose.
    It failed on the tree that introduced it, at `clave.data.recorder` and
    `clave.runtime.loop`, both of which held `DETECTION_CAMERA = "gate_wide"`.

    A camera id in Python is the abstraction leaking: it means that moving the
    detection camera to a new mounting, or renaming it, edits source rather
    than configuration.
    """
    raw = load(CONFIG)
    declared = [str(entry["id"]) for entry in require(raw, "cameras")]
    package = ROOT / "src" / "clave"

    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text()
        for name in declared:
            if re.search(rf"""['"]{re.escape(name)}['"]""", text):
                offenders.append(f"{path.relative_to(ROOT)} names {name!r}")
    assert offenders == []

"""vehicle_state (the parked/driving flag published to Home Assistant via MQTT and to ABRP) must
come from the GEAR + SPEED, not signal 1941 — that signal is acAirVolume (AC fan speed), so a fan
level of 3 or 5 while the car was parked used to flip the published state to "driving". CI-safe."""
import client


def _sig(**kw):
    base = {"1010": 0, "1319": 0}      # gear P, speed 0 → parked
    base.update(kw)
    return base


def test_ac_fan_level_does_not_imply_driving():
    # the exact live case caught on the car: parked, AC fan at level 3 (1941=3) → still parked
    assert client._parse_signal("V", _sig(**{"1941": 3})).vehicle_state == "parked"
    assert client._parse_signal("V", _sig(**{"1941": 5})).vehicle_state == "parked"


def test_gear_or_speed_imply_driving():
    assert client._parse_signal("V", _sig(**{"1010": 3})).vehicle_state == "driving"   # gear D
    assert client._parse_signal("V", _sig(**{"1010": 2})).vehicle_state == "driving"   # gear N
    assert client._parse_signal("V", _sig(**{"1010": 1})).vehicle_state == "driving"   # gear R
    assert client._parse_signal("V", _sig(**{"1319": 50})).vehicle_state == "driving"  # moving in P-ish
    assert client._parse_signal("V", _sig()).vehicle_state == "parked"                 # gear P, stopped

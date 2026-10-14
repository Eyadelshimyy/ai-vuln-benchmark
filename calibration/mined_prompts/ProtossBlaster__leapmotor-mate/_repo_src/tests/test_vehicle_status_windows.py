"""_parse_vehicle_status window mapping (#62, T03 windows).

A window is open if its open/closed flag (1693-1696) OR — only where it's a trusted sensor for this
car — its position % (3727/3728/1879/1880) is non-zero. The two sources are model-specific: the B10
drives the flags and its position % is a DEAD/garbage sensor (marked `windows_pct` broken in its
capability profile), while the T03 leaves the flags at 0 and reports only the live position %.
Reading the flags alone showed the T03's open windows as always "closed"; blindly OR-ing the % made
the B10's dead %-sensor false-positive every window — so the % fallback is gated on the profile.

Values below are from gody01's real T03 captures (bundle-3 closed / bundle-4 opened ~20% via the
official app). Needs web.main (fastapi) → skipped in the minimal CI test env, like the other
main-based tests."""
import pytest

pytest.importorskip("fastapi", reason="web.main needs fastapi (absent in the minimal CI test env)")

import main


def _windows(sig, vin, monkeypatch, *, pct_trusted, cmd_pct=None):
    # windows_pct trusted (T03) vs broken/ignored (B10); other features default to shown.
    monkeypatch.setattr(main.capability_profile, "is_shown",
                        lambda v, feat, *a, **k: pct_trusted if feat == "windows_pct" else True)
    return main._parse_vehicle_status(sig, vin, cmd_pct)["windows"]


def test_t03_open_windows_read_from_percent(monkeypatch):
    # flags stay false but the position % is 20 → all four windows must read as OPEN, and the
    # per-window opening % is surfaced for the Vehicle page (the T03 reports it).
    sig = {"1693": False, "1694": False, "1695": False, "1696": False,
           "3727": 20, "3728": 20, "1879": 20, "1880": 20}
    w = _windows(sig, "T03VIN", monkeypatch, pct_trusted=True)
    assert (w["fl"], w["fr"], w["rl"], w["rr"]) == (True, True, True, True)
    assert (w["fl_pct"], w["fr_pct"], w["rl_pct"], w["rr_pct"]) == (20, 20, 20, 20)


def test_b10_per_window_percent_from_commanded(monkeypatch):
    # B10: % sensor dead → the per-window % falls back to the last commanded position, but only for
    # a window the flag confirms OPEN (a closed window shows no stale %). Open = any non-zero flag.
    sig = {"1693": 1, "1694": 0, "1695": 0, "1696": 0}   # only FL open
    w = _windows(sig, "B10VIN", monkeypatch, pct_trusted=False, cmd_pct=50)
    assert (w["fl"], w["fl_pct"]) == (True, 50)
    assert (w["fr"], w["fr_pct"]) == (False, None)


def test_b10_no_percent_without_a_commanded_value(monkeypatch):
    # nothing commanded yet → open/closed only, no number.
    w = _windows({"1693": 1}, "B10VIN", monkeypatch, pct_trusted=False, cmd_pct=None)
    assert w["fl"] is True and w["fl_pct"] is None


def test_b10_flag_2_reads_open(monkeypatch):
    # B10 ground truth (verified on-car against the official app): the status flag is 1693=2 when the
    # window is OPEN (0 = shut). The B10's position % sensor is dead (always 0 → untrusted, use_pct=False),
    # so the flag is the only truth there → a flag of 2 must read OPEN, and the open window shows the
    # last commanded %. (Reverts the #68 assumption that 2 = closed, which came from a stale cloud frame.)
    sig = {"1693": 2, "1694": 0, "1695": 0, "1696": 0}   # only FL open (flag 2)
    w = _windows(sig, "B10VIN", monkeypatch, pct_trusted=False, cmd_pct=50)
    assert (w["fl"], w["fr"], w["rl"], w["rr"]) == (True, False, False, False)
    assert (w["fl_pct"], w["fr_pct"]) == (50, None)


def test_t03_closed_windows(monkeypatch):
    sig = {"1693": False, "1694": False, "1695": False, "1696": False,
           "3727": 0, "3728": 0, "1879": 0, "1880": 0}
    w = _windows(sig, "T03VIN", monkeypatch, pct_trusted=True)
    assert (w["fl"], w["fr"], w["rl"], w["rr"]) == (False, False, False, False)


def test_b10_dead_percent_does_not_false_positive(monkeypatch):
    # The exact regression (#62): on the B10 `windows_pct` is broken, so the dead sensor's garbage
    # (here 50) MUST be ignored — with the flags at 0 the windows read CLOSED, not open.
    sig = {"1693": 0, "1694": 0, "1695": 0, "1696": 0,
           "3727": 50, "3728": 50, "1879": 50, "1880": 50}
    w = _windows(sig, "B10VIN", monkeypatch, pct_trusted=False)
    assert (w["fl"], w["fr"], w["rl"], w["rr"]) == (False, False, False, False)


def test_b10_open_window_read_from_flag(monkeypatch):
    # B10: the open/closed flag drives it (% ignored), per-window mapping respected (only FL).
    sig = {"1693": 1, "1694": 0, "1695": 0, "1696": 0}
    w = _windows(sig, "B10VIN", monkeypatch, pct_trusted=False)
    assert (w["fl"], w["fr"], w["rl"], w["rr"]) == (True, False, False, False)


def test_windows_none_when_unreported(monkeypatch):
    # car asleep / no window signals at all → None (unknown), not a false "closed".
    w = _windows({}, "VIN", monkeypatch, pct_trusted=True)
    assert w["fl"] is None and w["sunshade"] is None

"""REEV auto-detection + dedicated-view strings.

A range-extender car reports a fuel-tank level (signal 3235) that BEV cars never send
(kerniger/leapmotor-ha#46). _parse_signal flags that as is_reev so the poller can enable
the dedicated REEV page even for a car onboarded before the variant existed. BEV payloads
must stay is_reev=False so nothing REEV-related shows on the majority of cars.
"""
import client as C
import i18n


def test_parse_signal_flags_reev_when_fuel_level_present():
    vd = C._parse_signal("LVIN0000000000001", {"3235": "82.1", "100003": "60"})
    assert vd.is_reev is True


def test_parse_signal_flags_reev_even_at_empty_tank():
    # 0 % fuel is still a REEV — presence of the signal, not its value, is the marker.
    vd = C._parse_signal("LVIN0000000000001", {"3235": "0"})
    assert vd.is_reev is True


def test_bev_payload_is_not_flagged_reev():
    vd = C._parse_signal("LVIN0000000000001", {"100003": "55", "3260": "349"})
    assert vd.is_reev is False


def test_reev_strings_exist_in_shipped_languages():
    for lang in ("en", "it", "fr"):
        t = i18n.get_t(lang)
        for key in ("nav_reev", "reev_title", "reev_fuel_level", "reev_combined_range", "reev_beta_note"):
            assert t(key) and t(key) != key, f"{lang}:{key} missing"

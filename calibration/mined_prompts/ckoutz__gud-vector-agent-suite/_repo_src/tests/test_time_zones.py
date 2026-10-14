from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from gvas.domain.intake import format_slot_label
from gvas.domain.time_zones import business_zone, normalize_time_zone, zone_label

PACIFIC = ZoneInfo("America/Los_Angeles")


def test_a_stored_utc_time_reads_in_the_business_zone() -> None:
    stored = datetime(2026, 10, 6, 16, 0, tzinfo=UTC)

    assert format_slot_label(stored, PACIFIC) == "Tue Oct 6, 9:00 AM Pacific time"


def test_a_reloaded_fixed_offset_reads_as_the_zone_name_not_the_offset() -> None:
    reloaded = datetime(2026, 12, 1, 9, 0, tzinfo=timezone(timedelta(hours=-8)))

    label = format_slot_label(reloaded, PACIFIC)

    assert label == "Tue Dec 1, 9:00 AM Pacific time"
    assert "UTC" not in label


def test_zones_outside_the_us_keep_their_own_abbreviation() -> None:
    local = datetime(2026, 7, 1, 9, 0, tzinfo=ZoneInfo("Europe/London"))

    assert zone_label(local) == "BST"


def test_havana_reads_as_its_city_not_us_central_time() -> None:
    local = datetime(2026, 1, 6, 9, 0, tzinfo=ZoneInfo("America/Havana"))

    assert zone_label(local) == "Havana time"


def test_us_central_zones_still_read_as_central_time() -> None:
    local = datetime(2026, 1, 6, 9, 0, tzinfo=ZoneInfo("America/Chicago"))

    assert zone_label(local) == "Central time"


def test_unknown_zone_names_are_rejected_and_ignored() -> None:
    with pytest.raises(ValueError):
        normalize_time_zone("Pacific")
    assert normalize_time_zone(" America/Chicago ") == "America/Chicago"
    assert business_zone("Not/AZone") is None
    assert business_zone(None) is None


def test_the_cuba_alias_reads_as_its_name_not_a_us_abbreviation() -> None:
    local = datetime(2026, 1, 6, 9, 0, tzinfo=ZoneInfo("Cuba"))

    assert zone_label(local) == "Cuba time"

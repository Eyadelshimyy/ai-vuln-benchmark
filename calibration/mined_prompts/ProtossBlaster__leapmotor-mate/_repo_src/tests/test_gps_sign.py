"""GPS longitude sign — GitHub #30.

The cloud reports the coordinates in several signal pairs: 2/3 are SIGNED, while
3724/3725 (and 2190/2191) are unsigned absolute values. A west-of-Greenwich car
(BatterBits' Lichfield B10, real position ~52.744N -1.916W) reported
2=-1.915912 but 3724=+1.915912 — parsing 3724 plotted it in the North Sea.
Fixtures below are trimmed from the two real raw-signal dumps on issue #30.
"""
import client


# BatterBits (Lichfield, UK) — west of Greenwich: 3724 lost the sign, 2 kept it.
UK_WEST = {
    "1": 1781076766374, "sts": 1781076766616,
    "2": -1.915912, "3": 52.744391,
    "3724": 1.915912, "3725": 52.744391,
    "2190": 52.744391, "2191": 1.915913,
    "100003": 62.6, "1204": 63, "1318": 2008, "1319": 0.0, "1010": 0,
}

# Silvio (Milan) — east of Greenwich: signed and unsigned pairs coincide.
IT_EAST = {
    "1": 1781076585786, "sts": 1781076587170,
    "2": 9.124942, "3": 45.443407,
    "3724": 9.124942, "3725": 45.443407,
    "2190": 45.443456, "2191": 9.12487,
    "100003": 100.0, "1204": 100, "1318": 3406, "1319": 0.0, "1010": 0,
}


def test_west_of_greenwich_longitude_keeps_sign():
    data = client._parse_signal("VINUK", UK_WEST)
    assert data.longitude == -1.915912          # NOT +1.915912 (the North Sea)
    assert data.latitude == 52.744391


def test_east_of_greenwich_unchanged():
    data = client._parse_signal("VINIT", IT_EAST)
    assert data.longitude == 9.124942
    assert data.latitude == 45.443407


def test_fallback_to_unsigned_pair_when_signed_missing():
    # Fresh VIN, no sign learned yet → can't know the hemisphere, so the unsigned value is
    # used as-is (unchanged pre-#43 behaviour; east-of-Greenwich cars are never affected).
    sig = {k: v for k, v in UK_WEST.items() if k not in ("2", "3")}
    data = client._parse_signal("VINFRESH1", sig)
    assert data.longitude == 1.915912           # best available without signal 2 OR memory
    assert data.latitude == 52.744391


def test_fallback_chain_to_2190_2191():
    sig = {k: v for k, v in UK_WEST.items() if k not in ("2", "3", "3724", "3725")}
    data = client._parse_signal("VINFRESH2", sig)
    assert data.longitude == 1.915913
    assert data.latitude == 52.744391


# ── #43: sign memory keeps west/south cars on dry land when a poll drops signals 2/3 ──

def test_unsigned_poll_keeps_remembered_sign():
    # A signed poll teaches the hemisphere; a later unsigned-only poll must NOT flip east.
    client._parse_signal("VINMEM", UK_WEST)                       # learns lon sign = −
    sig = {k: v for k, v in UK_WEST.items() if k not in ("2", "3")}
    data = client._parse_signal("VINMEM", sig)
    assert data.longitude == -1.915912          # re-signed from memory, NOT the North Sea
    assert data.latitude == 52.744391


def test_seed_primes_sign_before_any_signed_poll():
    # The restart case (#43): seeded from the persisted setting, the very first poll —
    # unsigned-only because the car is parked — already lands on dry land.
    client.seed_coord_signs("VINSEED", lat_sign=1.0, lon_sign=-1.0)
    sig = {k: v for k, v in UK_WEST.items() if k not in ("2", "3")}
    data = client._parse_signal("VINSEED", sig)
    assert data.longitude == -1.915912
    assert data.latitude == 52.744391


def test_east_car_seed_stays_positive():
    client.seed_coord_signs("VINSEEDIT", lat_sign=1.0, lon_sign=1.0)
    sig = {k: v for k, v in IT_EAST.items() if k not in ("2", "3")}
    data = client._parse_signal("VINSEEDIT", sig)
    assert data.longitude == 9.124942
    assert data.latitude == 45.443407


def test_signed_read_updates_persisted_signs():
    client._parse_signal("VINSIGNS", UK_WEST)
    assert client.get_coord_signs("VINSIGNS") == {"lat": 1.0, "lon": -1.0}

"""Manual trip merge — reversible, non-destructive.

A journey split by a SHORT, NON-charging stop can be merged back into one trip. The guard:
(1) gap between the two trips < gap_min, AND (2) the second trip's start SoC is not higher than
the first's end SoC (a SoC rise = a charge in the gap → never mergeable). Merge only sets
merged_into_id; unmerge clears it → the originals come back untouched.
"""
import sqlite3

import db as poller_db
import db_reader


def _setup(tmp_path, monkeypatch):
    path = str(tmp_path / "t.db")
    poller_db.Database(path)                       # builds schema incl. merged_into_id
    con = sqlite3.connect(path)
    con.execute("INSERT INTO vehicles (id, vin) VALUES (1, 'V')")
    con.commit(); con.close()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    return path


def _trip(path, tid, start, end, ssoc, esoc, sodo, eodo, dist, dur, regen=0.0):
    con = sqlite3.connect(path)
    con.execute(
        "INSERT INTO trips (id,vehicle_id,started_at,ended_at,start_soc,end_soc,"
        "start_odometer_km,end_odometer_km,distance_km,duration_min,regen_kwh) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (tid, 1, start, end, ssoc, esoc, sodo, eodo, dist, dur, regen),
    )
    con.commit(); con.close()


def _pos(path, tid, t, lat, lon):
    con = sqlite3.connect(path)
    con.execute("INSERT INTO trip_positions (trip_id,recorded_at,latitude,longitude,speed_kmh,soc)"
                " VALUES (?,?,?,?,?,?)", (tid, t, lat, lon, 10, 70))
    con.commit(); con.close()


def _three(path):
    # A: coffee-stop then B (18 min gap, SoC flat → mergeable). C: after a charge (SoC rose).
    _trip(path, 1, "2026-06-09T08:00:00+00:00", "2026-06-09T08:40:00+00:00", 80, 74, 1000, 1035, 35, 40)
    _trip(path, 2, "2026-06-09T08:58:00+00:00", "2026-06-09T09:30:00+00:00", 74, 68, 1035, 1063, 28, 32)
    _trip(path, 3, "2026-06-09T10:00:00+00:00", "2026-06-09T10:30:00+00:00", 75, 70, 1063, 1090, 27, 30)


# ── eligibility ─────────────────────────────────────────────────────────────────
def test_eligible_pair_default_gap(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    pairs = db_reader.get_mergeable_pairs(30)
    assert [(x["a_id"], x["b_id"]) for x in pairs] == [(1, 2)]       # only A–B (18 min, SoC flat)
    assert pairs[0]["gap_min"] == 18


def test_gap_too_small_excludes(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    assert db_reader.get_mergeable_pairs(15) == []                   # 18 min > 15 → none


def test_soc_rise_blocks_even_within_gap(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    # widen the window so B–C is within gap (30 min) — but C.start_soc 75 > B.end_soc 68 (charged)
    pairs = db_reader.get_mergeable_pairs(60)
    assert (2, 3) not in [(x["a_id"], x["b_id"]) for x in pairs]
    assert [(x["a_id"], x["b_id"]) for x in pairs] == [(1, 2)]


# ── merge / group stats / unmerge ────────────────────────────────────────────────
def test_merge_produces_group_then_unmerge_restores(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    assert len(db_reader.get_trips()) == 3

    res = db_reader.merge_trips(1, 2, 30)
    assert res["ok"] and res["parent_id"] == 1

    trips = db_reader.get_trips()
    assert len(trips) == 2                                            # group + C
    grp = next(t for t in trips if t["id"] == 1)
    assert grp["is_merged"] and grp["merged_count"] == 2
    assert grp["distance_km"] == 63.0                                 # 1063 − 1000 (odometer span)
    assert grp["duration_min"] == 72.0                               # 40 + 32 (driving only)
    assert grp["start_soc"] == 80 and grp["end_soc"] == 68           # earliest start → latest end
    assert grp["ended_at"].startswith("2026-06-09T09:30")

    # unmerge → back to 3 untouched trips
    db_reader.unmerge_trip(1)
    back = db_reader.get_trips()
    assert len(back) == 3
    a = next(t for t in back if t["id"] == 1)
    assert a["distance_km"] == 35 and a["end_soc"] == 74 and not a["is_merged"]


def test_merge_rejects_soc_rise_and_big_gap(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    assert db_reader.merge_trips(2, 3, 60)["error"] == "soc_rose_charge_in_gap"   # charged between
    assert db_reader.merge_trips(1, 2, 10)["error"] == "gap_too_large"            # 18 > 10


def test_track_is_group_aware(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    _pos(p, 1, "2026-06-09T08:10:00+00:00", 45.0, 9.0)
    _pos(p, 2, "2026-06-09T09:10:00+00:00", 45.1, 9.1)
    assert len(db_reader.get_trip_track(1)) == 1                      # before merge: only A's point
    db_reader.merge_trips(1, 2, 30)
    track = db_reader.get_trip_track(1)
    assert len(track) == 2                                            # union of A + B, chronological
    assert track[0]["latitude"] == 45.0 and track[1]["latitude"] == 45.1


def test_delete_merged_group_cascades(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    db_reader.merge_trips(1, 2, 30)
    assert db_reader.delete_trip(1) is True
    remaining = db_reader.get_trips()
    assert [t["id"] for t in remaining] == [3]                        # group (1+2) gone, C stays


def test_detail_of_child_resolves_to_group(tmp_path, monkeypatch):
    p = _setup(tmp_path, monkeypatch); _three(p)
    db_reader.merge_trips(1, 2, 30)
    d = db_reader.get_trip_detail(2)                                  # child id → parent group
    assert d["id"] == 1 and d["is_merged"] and d["merged_count"] == 2
    assert d.get("stop_min") == 18                                   # 09:30−08:00 elapsed − 72 driving

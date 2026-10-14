"""Provisional-SoC marker on the trip detail (`ec_pending`): a getEC-candidate trip whose official
cloud value hasn't locked yet shows the SoC estimate, flagged so the UI can say "provisional —
waiting for cloud" instead of looking like a final, slightly-imprecise number. Only while the trip is
still inside the enrichment retry window (~6h); older un-enriched trips stay plain SoC (no claim).
"""
from datetime import datetime, timedelta, timezone

import db as D
import db_reader


def _mk(tmp_path, monkeypatch, *, ended_min_ago, ec_stable=0, enabled="1"):
    pdb = D.Database(str(tmp_path / "t.db"))
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    now = datetime.now(timezone.utc)
    ended = now - timedelta(minutes=ended_min_ago)
    started = ended - timedelta(minutes=20)
    pdb._conn.execute(
        "INSERT INTO trips (id, vehicle_id, started_at, ended_at, distance_km,"
        " efficiency_kwh_100km, start_soc, end_soc, ec_stable) VALUES (1,1,?,?,7.0,30.0,80,70,?)",
        (started.isoformat(), ended.isoformat(), ec_stable))
    pdb._conn.commit()
    db_reader.set_setting("ec_trip_energy_enabled", enabled)
    db_reader.set_setting("ec_trip_since", (started - timedelta(hours=1)).isoformat())
    return pdb


def test_pending_when_candidate_not_stable_and_recent(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch, ended_min_ago=20, ec_stable=0)
    assert db_reader.get_trip_detail(1)["ec_pending"] is True


def test_not_pending_once_stable(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch, ended_min_ago=20, ec_stable=1)
    assert db_reader.get_trip_detail(1)["ec_pending"] is False


def test_not_pending_past_retry_window(tmp_path, monkeypatch):
    """Cloud never delivered within ~6h → no 'waiting' claim, just plain SoC."""
    _mk(tmp_path, monkeypatch, ended_min_ago=7 * 60, ec_stable=0)
    assert db_reader.get_trip_detail(1)["ec_pending"] is False


def test_not_pending_before_cutoff(tmp_path, monkeypatch):
    """A trip that started before the feature cutoff is not a getEC candidate."""
    _mk(tmp_path, monkeypatch, ended_min_ago=20, ec_stable=0)
    db_reader.set_setting("ec_trip_since", (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    assert db_reader.get_trip_detail(1)["ec_pending"] is False


def test_not_pending_when_feature_off(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch, ended_min_ago=20, ec_stable=0, enabled="0")
    assert db_reader.get_trip_detail(1)["ec_pending"] is False


# ── same marker in the trips LIST (get_trips_grouped) ─────────────────────────

def _grouped_trip(grouped, tid):
    for y in grouped:
        for m in y["months"].values():
            for d in m["days"].values():
                for t in d["trips"]:
                    if t["id"] == tid:
                        return t
    return None


def test_list_marks_pending_trip(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch, ended_min_ago=20, ec_stable=0)
    t = _grouped_trip(db_reader.get_trips_grouped(), 1)
    assert t is not None and t["ec_pending"] is True


def test_list_no_mark_when_stable(tmp_path, monkeypatch):
    _mk(tmp_path, monkeypatch, ended_min_ago=20, ec_stable=1)
    t = _grouped_trip(db_reader.get_trips_grouped(), 1)
    assert t is not None and t["ec_pending"] is False

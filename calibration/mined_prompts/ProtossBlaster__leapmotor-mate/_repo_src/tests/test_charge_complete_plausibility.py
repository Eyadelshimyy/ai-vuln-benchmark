"""A REEV's "charge complete" flag is not repeated, because it does not mean that.

Signal 3736 was mapped as "chargeCompleted" with a note to validate it on a real charge. Nine
complete charges from a B10 REEV (beta #12, michapr) show it is the other way round: it turns ON
when a charge starts — cable in, current flowing, hours remaining — and OFF when the charge ends,
three times with the battery exactly at the configured limit. Mate was printing "Fully charged"
for the whole time the car was filling.

What this replaced was a tolerance — ignore the flag when the SoC is more than 15 points below the
limit — fitted to the first sighting at 23 %. It hid the lie early in a charge and let it through
after 75 %, which is why the fault looked rare and kept coming back. These tests hold the shape of
the new rule: on a REEV the flag is not evidence of anything. A BEV does not read it either
(test_a_charge_is_complete_at_the_cars_limit.py).
"""
import db as D
import db_reader


def _status(tmp_path, monkeypatch, soc, completed, *, limit=None, reev=True, name="cc"):
    path = str(tmp_path / f"{name}.db")
    db = D.Database(path)
    db._conn.execute("INSERT INTO vehicles (id, vin, car_type) VALUES (1,'VIN','B10')")
    db._conn.execute(
        "INSERT INTO positions (vehicle_id, recorded_at, soc, charge_completed, plug_connected,"
        " latitude, longitude) VALUES (1,'2026-07-24T14:08:00+00:00',?,?,1,45.0,9.0)",
        (soc, completed))
    db._conn.execute("INSERT OR REPLACE INTO settings (key,value) VALUES ('is_reev',?)",
                     ("1" if reev else "0",))
    if limit is not None:
        db._conn.execute(
            "INSERT OR REPLACE INTO settings (key,value) VALUES ('charge_limit_percent_vin',?)",
            (str(limit),))
    db._conn.commit()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    return db_reader.get_latest_status()


def test_the_flag_is_not_repeated_on_a_reev(tmp_path, monkeypatch):
    """michapr's first numbers: the car claimed complete at 23.3 % with the limit at 90 %."""
    s = _status(tmp_path, monkeypatch, soc=23.3, completed=1, limit=90)
    assert not s["charge_completed"]


def test_the_case_the_old_tolerance_let_through(tmp_path, monkeypatch):
    """76.1 % with the limit at 90 — seven points more than the old rule would forgive, so the
    claim reached the screen. This is the charge in the bundle that runs 76.1 % → 90 %, and it was
    still filling the whole way."""
    s = _status(tmp_path, monkeypatch, soc=76.1, completed=1, limit=90)
    assert not s["charge_completed"]


def test_it_applies_without_a_known_limit(tmp_path, monkeypatch):
    """The old rule needed the charge limit as a reference and stood down without one. This one
    needs no reference: the signal is misread whatever the limit is."""
    s = _status(tmp_path, monkeypatch, soc=23.3, completed=1, limit=None)
    assert not s["charge_completed"]


def test_what_the_car_said_survives_for_diagnostics(tmp_path, monkeypatch):
    """Dropped from the UI, kept in the data — a bundle must still show what the car reported,
    or the next person investigating this signal starts from nothing."""
    s = _status(tmp_path, monkeypatch, soc=23.3, completed=1, limit=90)
    assert s["charge_completed_raw"] == 1

    s = _status(tmp_path, monkeypatch, soc=23.3, completed=0, limit=90, name="cc2")
    assert s["charge_completed_raw"] == 0


def test_a_car_not_claiming_complete_is_still_not_complete(tmp_path, monkeypatch):
    s = _status(tmp_path, monkeypatch, soc=23.3, completed=0, limit=90)
    assert not s["charge_completed"]

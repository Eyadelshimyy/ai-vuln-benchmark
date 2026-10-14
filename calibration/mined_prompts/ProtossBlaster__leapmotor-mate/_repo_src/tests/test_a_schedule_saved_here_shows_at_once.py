"""A charge window saved on the Scheduling page shows on the Overview at once, for the car it was saved on.

Saving wrote the window into the shared keys (`charge_sched_*`, #173) so the Overview's chip would not
keep the old times for the 30 minutes until the poller reads the car again. Since #186 the chip reads
each car's own keys first (`charge_sched_*_<vin>`), which only the poller writes: once the poller had
cached a car's window, a save from Mate changed nothing on screen until its next read. On #395 the
schedule was switched off at 11:02 and on again at 14:27 and 14:28, every save confirmed by the car,
while the chip went on showing the window from before.
"""
import db as D
import db_reader
import pytest

A, B = "LFZT03AAAAAAAAAA1", "LFZC10BBBBBBBBBB2"


@pytest.fixture
def page(tmp_path, monkeypatch):
    pytest.importorskip("httpx", reason="Starlette's TestClient is built on httpx")
    path = str(tmp_path / "s.db")
    database = D.Database(path)
    database._conn.execute("INSERT INTO vehicles (id, vin, car_type) VALUES (1,?,'T03')", (A,))
    database._conn.execute("INSERT INTO vehicles (id, vin, car_type) VALUES (2,?,'C10')", (B,))
    database._conn.commit()
    monkeypatch.setattr(db_reader, "DB_PATH", path)
    db_reader.set_setting("setup_complete", "1")
    db_reader.set_setting("language", "en")
    # What the poller last read from each car.
    database.set_charge_schedule(A, enabled=True, start="23:35", end="08:00")
    database.set_charge_schedule(B, enabled=True, start="01:00", end="05:00")
    import main
    from starlette.testclient import TestClient
    sent = []

    def save(**plan):
        sent.append(plan)
        return answer[0]
    answer = [(True, "")]
    monkeypatch.setattr(main.command_client, "save_charge_schedule", save)
    db_reader.set_active_vehicle(A)
    return TestClient(main.app), answer


def _save(client, enabled, start, end="08:00"):
    form = {"soc_limit": "100", "start_time": start, "end_time": end, "days": [str(i) for i in range(7)]}
    if enabled:
        form["enabled"] = "on"
    return client.post("/api/charge-schedule", data=form)


def test_switching_the_schedule_off_shows_at_once(page):
    client, _ = page
    assert _save(client, False, "23:35").status_code == 200
    assert db_reader.get_charge_schedule_window()["enabled"] is False


def test_new_times_show_at_once(page):
    client, _ = page
    _save(client, True, "23:30", "07:30")
    assert db_reader.get_charge_schedule_window() == {"enabled": True, "start": "23:30", "end": "07:30"}


def test_the_other_car_keeps_its_own_window(page):
    client, _ = page
    _save(client, False, "23:30")
    db_reader.set_active_vehicle(B)
    assert db_reader.get_charge_schedule_window() == {"enabled": True, "start": "01:00", "end": "05:00"}


def test_a_save_the_car_did_not_take_changes_nothing(page):
    client, answer = page
    answer[0] = (False, "unconfirmed")
    _save(client, False, "23:30")
    assert db_reader.get_charge_schedule_window() == {"enabled": True, "start": "23:35", "end": "08:00"}


def test_the_poller_s_next_read_still_wins(page):
    """The car stays the source: what the poller reads afterwards replaces what the page wrote."""
    client, _ = page
    _save(client, False, "23:30")
    D.Database(db_reader.DB_PATH).set_charge_schedule(A, enabled=True, start="22:00", end="06:00")
    assert db_reader.get_charge_schedule_window() == {"enabled": True, "start": "22:00", "end": "06:00"}

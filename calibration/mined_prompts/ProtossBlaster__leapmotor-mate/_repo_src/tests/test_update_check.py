"""Self-update check (update_check): the version comparison + the available/not-available logic.
The network fetch is background + best-effort and isn't exercised here (we pin a recent
'checked_at' so no thread is spawned)."""
import update_check as U


def test_ver_tuple_parsing_and_order():
    assert U._ver_tuple("1.14.0") == (1, 14, 0)
    assert U._ver_tuple("v1.14.0") == (1, 14, 0)
    assert U._ver_tuple("2.0") == (2, 0, 0)
    assert U._ver_tuple("") == (0, 0, 0)
    assert U._ver_tuple("1.14.0") < U._ver_tuple("1.14.1")
    assert U._ver_tuple("1.14.0") < U._ver_tuple("1.15.0")
    assert U._ver_tuple("1.9.0") < U._ver_tuple("1.14.0")     # numeric, not string, compare
    assert U._ver_tuple("1.14.0") < U._ver_tuple("2.0.0")
    assert not (U._ver_tuple("1.14.0") > U._ver_tuple("1.14.0"))


def _pin(monkeypatch, latest):
    vals = {"update_latest": latest, "update_checked_at": "9999999999"}  # future → no bg refresh
    monkeypatch.setattr(U.db_reader, "get_setting", lambda k, d="": vals.get(k, d))


def test_available_when_newer(monkeypatch):
    _pin(monkeypatch, "1.15.0")
    s = U.get_update_status("1.14.0")
    assert s["available"] is True and s["latest"] == "1.15.0" and "releases" in s["url"]


def test_not_available_when_same_or_older(monkeypatch):
    _pin(monkeypatch, "1.14.0")
    assert U.get_update_status("1.14.0")["available"] is False
    _pin(monkeypatch, "1.13.0")
    assert U.get_update_status("1.14.0")["available"] is False


def test_no_crash_when_nothing_cached(monkeypatch):
    monkeypatch.setattr(U, "_maybe_refresh", lambda: None)            # stay offline
    monkeypatch.setattr(U.db_reader, "get_setting", lambda k, d="": d)
    s = U.get_update_status("1.14.0")
    assert s["available"] is False and s["latest"] is None

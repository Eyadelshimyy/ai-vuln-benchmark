"""MateBetaTesterOnly research mode: flag gating, GPS redaction, encrypted export envelope,
logbook round-trip, and the poller's delta signal logging + retention.
"""
import json

import db as D
import db_reader
import research


# ── flag gating ──────────────────────────────────────────────────────────────
def test_research_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MATE_RESEARCH", raising=False)
    assert research.research_enabled() is False


def test_research_enabled_with_flag(monkeypatch):
    for v in ("1", "true", "yes"):
        monkeypatch.setenv("MATE_RESEARCH", v)
        assert research.research_enabled() is True
    for v in ("0", "false", ""):
        monkeypatch.setenv("MATE_RESEARCH", v)
        assert research.research_enabled() is False


# ── GPS redaction (location never leaves the tester's machine) ─────────────────
def test_redact_strips_gps_keeps_fuel():
    rows = [(1, "3235", "82.1"), (1, "2", "45.1"), (1, "3", "9.2"),
            (1, "3724", "9.2"), (1, "3259", "665")]
    kept = research.redact_signal_rows(rows)
    keys = {r[1] for r in kept}
    assert {"3235", "3259"} <= keys
    assert keys.isdisjoint({"2", "3", "3724", "3725", "2190", "2191"})


# ── export crypto: produces a sealed envelope (decrypt verified separately on the Mac) ──
def test_encrypt_bundle_produces_sealed_envelope():
    env = json.loads(research.encrypt_bundle(b"hello bundle"))
    assert env["v"] == 1 and env["alg"].startswith("RSA-OAEP")
    assert env["key"] and env["data"]            # sealed key + encrypted body present
    assert b"hello bundle" not in research.encrypt_bundle(b"hello bundle")  # not plaintext


# ── logbook round-trip (web side) ──────────────────────────────────────────────
def test_logbook_roundtrip(tmp_path, monkeypatch):
    D.Database(str(tmp_path / "t.db"))
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    db_reader.add_logbook_note("engine started to charge while driving")
    db_reader.add_logbook_note("   ")          # blank ignored
    notes = db_reader.get_logbook()
    assert len(notes) == 1
    assert notes[0]["note"] == "engine started to charge while driving"


# ── poller delta logging + retention ───────────────────────────────────────────
def test_raw_signal_delta_insert_and_prune(tmp_path):
    db = D.Database(str(tmp_path / "t.db"))
    vid = db.ensure_vehicle("LVIN0000000000001", "B10", 2025)
    assert db.insert_raw_signal_changes(vid, 1000, {"3235": "82.1", "3259": "665"}) == 2
    assert db.insert_raw_signal_changes(vid, 2000, {}) == 0      # nothing changed → no rows
    assert db.prune_raw_signals(0) == 0                          # 0 days = keep forever
    assert db.prune_raw_signals(1) == 2                          # 1970 rows are older than 1 day

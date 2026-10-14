"""#67 (rossiadobe): the grouped Charges page must show the FULL history, not just the newest
50. He imported ~89 charges (back to April 2025); the page "stopped at October 2025" because
get_charges_grouped() called get_charges() with its default limit=50. The data was in the DB all
along (CSV export and the monthly report, which read unbounded, showed everything) — only the
grouped list was capped. get_charges_grouped must load every charge.

Runs on a tmp_path DB (poller schema) with db_reader pointed at it. CI-safe."""
import db as D
import db_reader


def _count_in_tree(grouped) -> int:
    n = 0
    for y in grouped:
        for m in y["months"].values():
            for d in m["days"].values():
                n += len(d["charges"])
    return n


def _seed(pdb, n):
    # n charges, one per week going back from a fixed date — spans several months so a 50-cap
    # would visibly truncate the oldest ones.
    for i in range(n):
        day = 1 + (i % 27)
        month = 1 + (i // 27)          # roll into earlier months as i grows
        started = f"2026-{month:02d}-{day:02d}T14:00:00+02:00"
        pdb._conn.execute(
            "INSERT INTO charges (vehicle_id, started_at, ended_at, start_soc, end_soc,"
            " energy_added_kwh, location_type) VALUES (1,?,?,30,80,10.0,'HOME')",
            (started, started))
    pdb._conn.commit()


def test_grouped_shows_all_charges_over_50(tmp_path, monkeypatch):
    pdb = D.Database(str(tmp_path / "t.db"))
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    _seed(pdb, 89)                                  # rossiadobe's real count
    grouped = db_reader.get_charges_grouped()
    assert _count_in_tree(grouped) == 89            # ALL of them, not capped at 50


def test_grouped_not_capped_at_default_limit(tmp_path, monkeypatch):
    """Guard the exact regression: >50 charges must not collapse to 50 in the grouped view."""
    pdb = D.Database(str(tmp_path / "t.db"))
    monkeypatch.setattr(db_reader, "DB_PATH", str(tmp_path / "t.db"))
    _seed(pdb, 60)
    assert _count_in_tree(db_reader.get_charges_grouped()) == 60
    # sanity: the raw list DOES cap at 50 by default — that's what fed the bug
    assert len(db_reader.get_charges(limit=50)) == 50

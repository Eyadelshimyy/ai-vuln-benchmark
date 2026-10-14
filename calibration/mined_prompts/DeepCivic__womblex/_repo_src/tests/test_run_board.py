"""The run board over a SQLite DBOS system database: enqueue, ownership, reads.

Nothing here runs a worker, so every workflow stays enqueued (``pending``); the
states a worker produces are exercised in ``test_workflows.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dbos")

from womblex.cloud import dbos_app
from womblex.cloud.jobs import JobSpec, RunBoard, RunOwnedError

ROOT = "s3://bucket/inbox"


@pytest.fixture()
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    dbos_app.ensure_schema(None)
    with RunBoard(None) as b:
        yield b


def spec(n: int, root: str = ROOT) -> JobSpec:
    return JobSpec(batch_num=n, input_keys=[f"{n}.pdf"], shard_prefix="runs/x/documents",
                   ingest_root=root)


def test_enqueue_is_idempotent(board):
    assert board.enqueue("run-a", [spec(1)]) == 1
    assert board.enqueue("run-a", [spec(1), spec(2)]) == 1  # only batch 2 is new
    assert board.stats("run-a") == {"pending": 2}


def test_a_run_prefix_does_not_match_a_longer_run_id(board):
    board.enqueue("run-a", [spec(1)])
    board.enqueue("run-ab", [spec(1)])
    assert board.stats("run-a") == {"pending": 1}


def test_batches_route_to_the_queue_of_their_ingest_root(board):
    board.enqueue("run-a", [spec(1, "s3://one/inbox"), spec(2, "s3://two/inbox")])
    assert dbos_app.extract_queue("s3://one/inbox") != dbos_app.extract_queue("s3://two/inbox")
    # A trailing slash is the same location.
    assert dbos_app.extract_queue("s3://one/inbox/") == dbos_app.extract_queue("s3://one/inbox")


def test_enqueue_stages_is_one_coordinator_in_pipeline_order(board):
    assert board.enqueue_stages("run-a", ["embed", "enrich", "chunk"], "runs/x/documents") == 3
    assert board.enqueue_stages("run-a", ["chunk", "enrich", "embed"], "runs/x/documents") == 0
    (row,) = board.list_jobs("run-a")
    assert row.kind == "downstream" and row.status == "pending"


def test_enqueue_refuses_a_run_owned_by_someone_else(board):
    assert board.enqueue("run-a", [spec(1)], owner="alice") == 1
    with pytest.raises(RunOwnedError):
        board.enqueue("run-a", [spec(2)], owner="bob")
    with pytest.raises(RunOwnedError):
        board.enqueue_stages("run-a", ["chunk"], "p", owner="bob")
    assert board.enqueue("run-a", [spec(2)], owner="alice") == 1  # the owner resumes freely


def test_an_unowned_run_is_refused_to_a_named_owner(board):
    board.enqueue("run-a", [spec(1)])
    with pytest.raises(RunOwnedError):
        board.enqueue("run-a", [spec(2)], owner="alice")


def test_an_unscoped_enqueue_inherits_the_runs_owner(board):
    board.enqueue("run-a", [spec(1)], owner="alice")
    board.enqueue_stages("run-a", ["chunk"], "p")  # admin dispatch, no owner named
    assert {r.run_id: r.owner for r in board.runs()}["run-a"] == "alice"
    assert board.stats("run-a", owner="alice") == {"pending": 2}


def test_views_are_scoped_by_owner(board):
    board.enqueue("run-a", [spec(1)], owner="alice")
    assert board.stats("run-a", owner="alice") == {"pending": 1}
    assert board.stats("run-a", owner="bob") == {}
    assert len(board.list_jobs("run-a", owner="alice")) == 1
    assert board.list_jobs("run-a", owner="bob") == []
    assert board.workers("run-a", owner="bob") == []
    assert board.stale_jobs(-1.0, "run-a", owner="bob") == []
    assert board.throughput("run-a", owner="bob").completed == 0


def test_runs_rolls_up_status_per_run(board):
    board.enqueue("run-a", [spec(1), spec(2)], owner="alice")
    board.enqueue("run-b", [spec(1)])
    (summary,) = board.runs(owner="alice")
    assert (summary.run_id, summary.owner, summary.counts, summary.total) == (
        "run-a", "alice", {"pending": 2}, 2,
    )
    assert {r.run_id for r in board.runs()} == {"run-a", "run-b"}
    assert [r.run_id for r in board.runs(run_id="run-b")] == ["run-b"]
    assert board.runs(owner="bob", run_id="run-a") == []


def test_list_jobs_reads_the_row_the_dashboard_shows(board):
    board.enqueue("run-a", [spec(1), spec(2)])
    rows = board.list_jobs("run-a")
    assert {r.batch_num for r in rows} == {1, 2}
    assert all(r.kind == "batch" and r.max_attempts == 3 and r.locked_by is None for r in rows)
    assert board.list_jobs("run-a", status="running") == []
    assert board.list_jobs("no-such-run") == []


def test_reads_on_a_database_no_writer_has_migrated_are_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "fresh.sqlite"))
    with RunBoard(None) as b:
        assert b.stats("run-a") == {}
        assert b.runs() == [] and b.list_jobs() == []

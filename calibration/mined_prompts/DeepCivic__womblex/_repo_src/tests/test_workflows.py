"""The DBOS workflows end to end: a worker on a SQLite system database, a local store."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("dbos")

from tests.test_cloud import _minimal_config
from womblex.cloud import dbos_app, workflows
from womblex.cloud.jobs import JobSpec, RunBoard
from womblex.cloud.worker import run_worker
from womblex.store.remote import RemoteStore


@pytest.fixture()
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A store holding one CSV, a SQLite system database, and fast polling."""
    monkeypatch.setenv("WOMBLEX_DBOS_PATH", str(tmp_path / "dbos.sqlite"))
    monkeypatch.setattr(workflows, "SETTLE_POLL_SECONDS", 0.2)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "people.csv").write_text("name,role\nAlice,Director\n")
    dbos_app.ensure_schema(None)
    return tmp_path / "store", str(inbox), tmp_path


def batch(ingest: str, n: int = 1) -> JobSpec:
    return JobSpec(batch_num=n, input_keys=["people.csv"], shard_prefix="runs/r1/documents",
                   ingest_root=ingest)


def work(fleet, **kw) -> int:
    store, ingest, tmp = fleet
    return run_worker(
        None, str(store), _minimal_config(tmp), ingest_uri=ingest, poll_interval=0.2, **kw,
    )


def test_a_worker_runs_a_batch_and_records_it_done(fleet):
    store, ingest, _ = fleet
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
        assert work(fleet, once=True) == 1
        assert board.stats("r1") == {"done": 1}
        (row,) = board.list_jobs("r1")
    assert row.status == "done" and row.locked_by is None
    shards = RemoteStore.from_uri(str(store))
    assert shards.list_files("runs/r1/documents", "*._manifest.parquet")
    assert shards.exists("runs/r1/logs/batch-0001.log")


def test_a_failed_batch_is_recorded_not_lost(fleet, monkeypatch):
    _, ingest, _ = fleet
    monkeypatch.setattr(workflows, "process_batch", lambda *a, **k: 1 / 0)
    with RunBoard(None) as board:
        board.enqueue("r1", [JobSpec(batch_num=1, input_keys=["people.csv"],
                                     shard_prefix="runs/r1/documents", ingest_root=ingest,
                                     max_attempts=1)])
        work(fleet, idle_timeout=1)
        assert board.stats("r1") == {"failed": 1}
        assert "division by zero" in (board.list_jobs("r1")[0].error or "")


def test_a_worker_wired_to_another_ingest_root_never_sees_the_batch(fleet):
    with RunBoard(None) as board:
        board.enqueue("r1", [batch("s3://elsewhere/inbox")])
        assert work(fleet, idle_timeout=1) == 0
        assert board.stats("r1") == {"pending": 1}


def test_downstream_stages_wait_for_extraction_then_run_in_order(fleet):
    store, ingest, _ = fleet
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
        board.enqueue_stages("r1", ["normalise"], "runs/r1/documents")
        work(fleet, idle_timeout=2)
        assert board.stats("r1") == {"done": 3}  # batch, coordinator, stage
        stages = [r for r in board.list_jobs("r1") if r.kind == "stage"]
    assert [r.stage for r in stages] == ["normalise"]
    out = RemoteStore.from_uri(str(store))
    assert out.list_files("runs/r1/documents", "*.normalised_text.parquet")
    assert out.exists("runs/r1/logs/stage-normalise-batch-0001.log")


def test_a_stage_with_nothing_to_run_over_fails_with_the_reason(fleet):
    with RunBoard(None) as board:
        board.enqueue_stages("r1", ["normalise"], "runs/r1/documents")
        work(fleet, idle_timeout=2)
        failed = [r for r in board.list_jobs("r1") if r.status == "failed"]
    assert failed and any("No batch bases" in (r.error or "") for r in failed)


# --- the extraction body a batch workflow runs, outside DBOS ------------------


def _extract(config, store, ingest, *, keys, run_id="r1", batch_num=1, ingest_root=""):
    """Run the extraction body a worker's DBOS workflow runs, outside DBOS."""
    from womblex.cloud import workflows

    workflows.set_context(workflows.WorkerContext(config, store, ingest, ingest_root))
    try:
        return workflows._extract(
            run_id, batch_num, keys, f"runs/{run_id}/documents", ingest_root,
        )
    finally:
        workflows.set_context(None)


def test_extract_downloads_from_a_second_ingest_store(tmp_path):
    """The gap merge 1 closes: inputs and outputs can be different stores."""
    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    _extract(_minimal_config(tmp_path), output_store, ingest_store, keys=["people.csv"])

    assert output_store.list_files("runs/r1/documents", "*._manifest.parquet")
    assert not output_store.exists("people.csv")  # never lands in the output tree
    assert ingest_store.exists("people.csv")       # the source document is untouched


def test_extract_extracts_both_same_named_documents(tmp_path):
    """A job's keys come from a recursive listing, so two prefixes routinely hold
    a `people.csv` apiece. Flat staging landed them on one local file — one
    document extracted twice, one not at all, and the recorded source path the
    survivor's. Both are extracted, each under its own relpath."""
    import pyarrow.parquet as pq

    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    for prefix, body in (
        ("2026-07", "name,role\nAlice,Director\n"),
        ("2026-08", "name,role\nBob,Assistant\n"),
    ):
        local = tmp_path / f"{prefix}.csv"
        local.write_text(body)
        ingest_store.upload_file(local, f"{prefix}/people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    _extract(
        _minimal_config(tmp_path), output_store, ingest_store,
        keys=["2026-07/people.csv", "2026-08/people.csv"], ingest_root="s3://bucket/inbox",
    )

    manifest = pq.read_table(
        str(tmp_path / "store" / "runs" / "r1" / "documents" / "batch-0001._manifest.parquet")
    ).to_pylist()
    assert len(manifest) == 2
    assert len({row["source_hash"] for row in manifest}) == 2  # distinct documents
    assert sorted(row["source_relpath"] for row in manifest) == [
        "2026-07/people.csv",
        "2026-08/people.csv",
    ]


# --- run logs ---------------------------------------------------------------


def test_capture_batch_log_attaches_and_detaches_cleanly(tmp_path):
    """The handler is added for the block and removed after — no leak that would
    tee the next batch's records into this file."""
    import logging

    from womblex.utils.run_log import capture_batch_log

    womblex_logger = logging.getLogger("womblex")
    before = list(womblex_logger.handlers)
    log_path = tmp_path / "batch.log"
    with capture_batch_log(log_path):
        assert len(womblex_logger.handlers) == len(before) + 1
        logging.getLogger("womblex.some.module").error("a captured line")
    assert womblex_logger.handlers == before  # detached
    assert "a captured line" in log_path.read_text()


def test_capture_batch_log_detaches_even_when_the_block_raises(tmp_path):
    import logging

    from womblex.utils.run_log import capture_batch_log

    womblex_logger = logging.getLogger("womblex")
    before = list(womblex_logger.handlers)
    log_path = tmp_path / "batch.log"
    with pytest.raises(RuntimeError), capture_batch_log(log_path):
        logging.getLogger("womblex").error("before the raise")
        raise RuntimeError("boom")
    assert womblex_logger.handlers == before
    # The file is complete and readable even though the block raised — the
    # failing case is the one the operator needs.
    assert "before the raise" in log_path.read_text()


def test_extract_publishes_the_batch_log_beside_the_shards(tmp_path):
    """A successful batch leaves `runs/<run_id>/logs/batch-NNNN.log` in the store."""
    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    _extract(
        _minimal_config(tmp_path), output_store, ingest_store, keys=["people.csv"], batch_num=3,
    )

    assert output_store.exists("runs/r1/logs/batch-0003.log")


def _publish_one_batch(tmp_path, run_id: str):
    """Run one batch through the real worker; return the published shard's stamp."""
    import pyarrow.parquet as pq

    from womblex.store.run_stamp import read_footer_stamp

    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")

    output_store = RemoteStore.from_uri(str(tmp_path / "store"))
    _extract(
        _minimal_config(tmp_path), output_store, ingest_store,
        keys=["people.csv"], run_id=run_id, batch_num=3,
    )

    shard = (
        tmp_path / "store" / "runs" / run_id / "documents" / "batch-0003.elements.parquet"
    )
    return csv, read_footer_stamp(pq.read_metadata(str(shard)).metadata)


def test_a_published_shard_stamps_the_same_run_a_local_batch_would(tmp_path):
    """Local and distributed agree, stage and write timestamp aside: the worker
    declares from the job row's run id and the same config, and where the
    documents were staged is not in the digest."""
    from womblex.batch import process_batch
    from womblex.store.run_stamp import RunStamp, read_footer_stamp

    csv, published = _publish_one_batch(tmp_path, "r1")

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    config = _minimal_config(tmp_path)
    process_batch(
        [csv], config, batch_num=3, shard_dir=local_dir,
        stamp=RunStamp.declare("r1", config, stage="extract"),
    )
    import pyarrow.parquet as pq
    local = read_footer_stamp(
        pq.read_metadata(str(local_dir / "batch-0003.elements.parquet")).metadata,
    )
    assert published == local
    assert published["run_id"] == "r1"


def test_a_row_naming_no_run_is_published_unstamped_not_failed(tmp_path):
    """A malformed row loses its stamp, not its extraction — the same terms an
    undeclared ingest root is already on."""
    _, stamp = _publish_one_batch(tmp_path, "")
    assert stamp == {}


def test_extract_publishes_the_log_even_when_the_batch_fails(tmp_path, monkeypatch):
    """The failing case is the one that matters: the log is uploaded outside the
    try, so a job that raises still leaves its `batch-NNNN.log` in the store, and
    the original error is what propagates."""
    ingest_store = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest_store.upload_file(csv, "people.csv")
    output_store = RemoteStore.from_uri(str(tmp_path / "store"))

    from womblex.cloud import workflows

    def _boom(*_a, **_kw):
        raise RuntimeError("processing exploded")

    monkeypatch.setattr(workflows, "process_batch", _boom)
    with pytest.raises(RuntimeError, match="processing exploded"):
        _extract(_minimal_config(tmp_path), output_store, ingest_store, keys=["people.csv"])

    assert output_store.exists("runs/r1/logs/batch-0001.log")


def test_a_stage_run_out_of_order_is_not_retried():
    assert not workflows._not_a_failure_to_retry(workflows.StageNotReady("chunk: missing"))
    assert workflows._not_a_failure_to_retry(OSError("transient"))


def test_finalize_records_the_checksum_of_every_published_file(fleet):
    """The publish step returns each key with its digest; finalize writes them into
    the run record, and they equal the SHA-256 of the bytes now in the store."""
    import argparse
    import hashlib

    from womblex.cli.cloud import cmd_finalize
    from womblex.store.run_manifest import read_run_record

    store_root, ingest, tmp = fleet
    with RunBoard(None) as board:
        board.enqueue("r1", [batch(ingest)])
        board.enqueue_stages("r1", ["normalise"], "runs/r1/documents")
        work(fleet, idle_timeout=2)
        result = board.file_checksums("r1")
    checksums = result.checksums

    assert result.undigested == []

    assert "runs/r1/documents/batch-0001.elements.parquet" in checksums
    assert "runs/r1/documents/batch-0001.normalised_text.parquet" in checksums
    for key, digest in checksums.items():
        assert hashlib.sha256((store_root / key).read_bytes()).hexdigest() == digest

    rc = cmd_finalize(argparse.Namespace(
        store=str(store_root), run_id="r1", output_prefix=None,
        dsn=f"sqlite:///{tmp / 'dbos.sqlite'}",
    ))
    assert rc == 0
    record = read_run_record(store_root / "runs" / "r1" / "manifest.parquet")
    assert {f["key"]: f["sha256"] for f in record["files"]} == {
        k.removeprefix("runs/r1/"): v for k, v in checksums.items()
    }
    assert not any("file checksums" in gap for gap in record["partial"])


class _Row:
    def __init__(self, wid, status, output, kind="batch", created_at=0):
        self.workflow_id, self.status, self.output = wid, status, output
        self.attributes, self.created_at = {"run_id": "r1", "kind": kind}, created_at


class _Client:
    def __init__(self, rows, steps):
        self.rows, self.steps = rows, steps

    def list_workflows(self, **_kw):
        return self.rows

    def list_workflow_steps(self, wid):
        return self.steps.get(wid, [])


def _board(rows, steps=None) -> RunBoard:
    board = RunBoard.__new__(RunBoard)
    board._client = _Client(rows, steps or {})
    return board


def test_a_failed_stage_keeps_its_finished_units_digests_and_is_named():
    """Units that finished before the stage failed published their files; their
    digests are read from the recorded steps, and the stage is named as a gap."""
    steps = {"r1:stage:chunk": [
        {"function_id": 1, "output": [["b1"], ["b2"]], "error": None},
        {"function_id": 2, "output": {"k/b1.chunks.parquet": "aa"}, "error": None},
        {"function_id": 3, "output": None, "error": RuntimeError("boom")},
    ]}
    result = _board([_Row("r1:stage:chunk", "ERROR", None, kind="stage")], steps).file_checksums("r1")
    assert result.checksums == {"k/b1.chunks.parquet": "aa"}
    assert result.undigested == ["r1:stage:chunk"]


def test_a_workflow_recorded_before_checksums_is_named_not_counted():
    rows = [
        _Row("r1:batch:0001", "SUCCESS", ["k/b1.elements.parquet"]),
        _Row("r1:stage:chunk", "SUCCESS", {"published": 2, "checksums": {"k/a": "aa"}},
             kind="stage", created_at=1),
        _Row("r1:downstream:x", "SUCCESS", [{"stage": "chunk"}], kind="downstream"),
    ]
    result = _board(rows).file_checksums("r1")
    assert result.checksums == {"k/a": "aa"}
    assert result.undigested == ["r1:batch:0001", "r1:stage:chunk"]


def test_a_step_recorded_as_a_key_list_resumes_without_digests():
    assert workflows._digests(["k/a", "k/b"]) == {}
    assert workflows._digests({"k/a": "aa"}) == {"k/a": "aa"}


def test_a_dispatched_workflow_runs_its_stages_under_its_own_config_and_digest(fleet):
    """The overlay, not the worker's config, sizes the chunks; its digest rides the files."""
    import pyarrow.parquet as pq

    store, ingest, _ = fleet
    wf = {"id": "wf", "digest": "sha256:feed", "preset": "recipe",
          "config": {"chunking": {"chunk_size": 8}}}
    with RunBoard(None) as board:
        for run in ("r1", "r2"):
            board.enqueue(run, [JobSpec(batch_num=1, input_keys=["people.csv"],
                                        shard_prefix=f"runs/{run}/documents", ingest_root=ingest)])
        board.enqueue_stages("r1", ["chunk"], "runs/r1/documents")
        board.enqueue_stages("r2", ["chunk"], "runs/r2/documents", workflow=wf)
        work(fleet, idle_timeout=3)
        coordinators = [r for r in board.list_jobs("r2") if r.kind == "downstream"]
    assert coordinators and all(r.status == "done" for r in coordinators)
    out = Path(str(store))

    def chunks(run: str):
        return pq.read_table(out / f"runs/{run}/documents/batch-0001.chunks.parquet")

    assert chunks("r2").num_rows > chunks("r1").num_rows  # chunk_size 8 splits what the default keeps
    meta = pq.read_metadata(str(out / "runs/r2/documents/batch-0001.chunks.parquet")).metadata
    assert meta[b"womblex.workflow_digest"] == b"sha256:feed"
    base = pq.read_metadata(str(out / "runs/r1/documents/batch-0001.chunks.parquet")).metadata
    assert b"womblex.workflow_digest" not in base

    # finalize observes the workflow in the record; the manifest is not a workflow stage's file
    import argparse

    from womblex.cli.cloud import cmd_finalize
    from womblex.store.run_manifest import read_run_record

    for run in ("r1", "r2"):
        assert cmd_finalize(argparse.Namespace(
            store=str(store), run_id=run, output_prefix=None,
            dsn=f"sqlite:///{fleet[2] / 'dbos.sqlite'}")) == 0
    manifest = out / "runs/r2/manifest.parquet"
    assert read_run_record(manifest)["workflows"] == [{"digest": "sha256:feed", "stages": ["chunk"]}]
    assert b"womblex.workflow_digest" not in pq.read_metadata(str(manifest)).metadata
    assert read_run_record(out / "runs/r1/manifest.parquet")["workflows"] == []


def test_a_workflow_unit_takes_none_of_the_workers_config_and_leaves_no_digest_behind(tmp_path):
    """Same thread, workflow unit then plain unit: the worker's `chunk_size` reaches only
    the plain one, and the workflow's digest only the workflow one."""
    import pyarrow.parquet as pq

    from womblex.config import ChunkingConfig

    store = RemoteStore.from_uri(str(tmp_path / "store"))
    ingest = RemoteStore.from_uri(str(tmp_path / "ingest"))
    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\n")
    ingest.upload_file(csv, "people.csv")
    config = _minimal_config(tmp_path).model_copy(
        update={"chunking": ChunkingConfig(chunk_size=8)})
    _extract(config, store, ingest, keys=["people.csv"])
    out = tmp_path / "store/runs/r1/documents/batch-0001.chunks.parquet"
    wf = {"id": "wf", "digest": "sha256:feed", "preset": "recipe", "config": {}}

    workflows.set_context(workflows.WorkerContext(config, store, ingest, ""))
    try:
        (unit,) = workflows._plan("chunk", "runs/r1/documents", wf)
        workflows._unit("chunk", "r1", "runs/r1/documents", unit, wf)
        under_workflow = pq.read_table(out).num_rows, pq.read_metadata(str(out)).metadata
        workflows._unit("chunk", "r1", "runs/r1/documents", unit)
        plain = pq.read_table(out).num_rows, pq.read_metadata(str(out)).metadata
    finally:
        workflows.set_context(None)

    assert under_workflow[0] < plain[0]  # the worker's chunk_size 8 split only the plain run
    assert under_workflow[1][b"womblex.workflow_digest"] == b"sha256:feed"
    assert b"womblex.workflow_digest" not in plain[1]


def test_workflows_recorded_before_the_workflow_argument_still_run(fleet):
    """A coordinator (five inputs) and a stage (four) enqueued by an older release."""
    store, ingest, _ = fleet
    with RunBoard(None) as board:
        for run in ("r1", "r2"):
            board.enqueue(run, [JobSpec(batch_num=1, input_keys=["people.csv"],
                                        shard_prefix=f"runs/{run}/documents", ingest_root=ingest)])
        work(fleet, idle_timeout=2)  # a bare stage does not wait for extraction
        board._client.enqueue({
            "workflow_name": dbos_app.DOWNSTREAM_WORKFLOW, "queue_name": dbos_app.DOWNSTREAM_QUEUE,
            "workflow_id": dbos_app.downstream_id("r1", ["normalise"]),
            "attributes": {"run_id": "r1", "kind": dbos_app.KIND_DOWNSTREAM},
        }, "r1", ["normalise"], "runs/r1/documents", 1, None)
        board._client.enqueue({
            "workflow_name": dbos_app.STAGE_WORKFLOW, "queue_name": dbos_app.stage_queue("normalise"),
            "workflow_id": dbos_app.stage_id("r2", "normalise"),
            "attributes": {"run_id": "r2", "kind": dbos_app.KIND_STAGE, "stage": "normalise"},
        }, "r2", "normalise", "runs/r2/documents", 1)
        work(fleet, idle_timeout=3)
        rows = board.list_jobs("r1") + board.list_jobs("r2")
    assert rows and all(r.status == "done" for r in rows), [(r.id, r.error) for r in rows]
    out = RemoteStore.from_uri(str(store))
    for run in ("r1", "r2"):
        assert out.list_files(f"runs/{run}/documents", "*.normalised_text.parquet")

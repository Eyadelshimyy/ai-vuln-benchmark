"""The workflow digest on every file a workflow run writes, and in the run record."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests._synthetic import REGISTER_CSV
from tests.test_workflow_runner import _resolve, _stored, shards  # noqa: F401
from womblex.batch import process_batch
from womblex.cloud.workflow_runner import run_workflow_local
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.store.run_manifest import read_run_record, write_run_manifest
from womblex.store.run_stamp import RunStamp, read_footer_stamp, sidecar_footer, workflow_context

_NEW = (".normalised_text.parquet", ".chunks.parquet")


def _digest(path: Path) -> str | None:
    return read_footer_stamp(pq.read_metadata(str(path)).metadata).get("workflow_digest")


def test_every_file_a_workflow_writes_carries_its_digest(shards: Path) -> None:  # noqa: F811
    stored = _stored(["normalise", "chunk"], config={"processing": {"text_source": "normalised"}})
    before = {p.name for p in shards.glob("*.parquet")}
    assert run_workflow_local(stored, shards, resolve_preset=_resolve).exit_code == 0
    written = [p for p in shards.glob("*.parquet") if p.name not in before]
    assert {"." + p.name.split(".", 1)[1] for p in written} >= set(_NEW)
    assert {_digest(p) for p in written} == {stored.digest}
    assert all(_digest(p) is None for p in shards.glob("*.elements.parquet"))  # extraction is not the workflow's
    assert sidecar_footer(shards / "batch-0001.parquet", "chunk") is None  # the context ended with the run


def test_run_record_names_the_workflow_and_its_stages(tmp_path: Path) -> None:
    cfg = WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
    )
    out = tmp_path / "shards"
    out.mkdir()
    stamp = RunStamp.declare("run-1", cfg, stage="extract")
    process_batch([REGISTER_CSV], cfg, batch_num=1, shard_dir=out, stamp=stamp)
    assert read_run_record(write_run_manifest(out))["workflows"] == []  # type: ignore[index]
    stored = _stored(["normalise", "chunk"])
    assert run_workflow_local(stored, out, resolve_preset=_resolve).exit_code == 0
    record = read_run_record(write_run_manifest(out))
    assert record is not None
    assert record["workflows"] == [{"digest": stored.digest, "stages": ["chunk", "normalise"]}]


def test_no_digest_outside_a_workflow_and_the_context_resets(tmp_path: Path) -> None:
    base = tmp_path / "batch-0001.parquet"
    assert sidecar_footer(base, "chunk") is None
    with pytest.raises(RuntimeError), workflow_context("sha256:abc"):
        # An unstamped shard still records which workflow wrote the file.
        assert sidecar_footer(base, "chunk") == {b"womblex.workflow_digest": b"sha256:abc"}
        raise RuntimeError
    assert sidecar_footer(base, "chunk") is None

"""``run-stage layout``: fingerprint-aware reruns over a local shard directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

import womblex.utils.model_registry as reg
from tests._pdf_builders import PdfBuilder
from womblex.batch import process_batch
from womblex.cloud.stage_contracts import STAGE_CONTRACTS
from womblex.cloud.stage_runner import run_stage_local, run_stage_remote
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.ingest.interfaces.protocols import LayoutRegionResult
from womblex.ingest.pdf.types import Rect
from womblex.process.layout_stage import layout_shards
from womblex.store.layout_output import (
    MATCH,
    MISMATCH,
    UNKNOWN,
    layout_fingerprint_status,
    layout_fingerprint_statuses,
    read_footer_layout_fingerprint,
    read_footer_redaction_consumed,
    read_layout_regions,
)
from womblex.store.remote import RemoteStore
from womblex.store.source_provenance import IngestProvenance


class _Analyzer:
    def __init__(self, label: str) -> None:
        self.label = label

    def analyze(self, img: np.ndarray, conf_threshold: float = 0.3) -> list[LayoutRegionResult]:
        h, w = img.shape[:2]
        return [LayoutRegionResult((0.0, 0.0, w / 2, h / 2), self.label, "paragraph", 0.9)]


@pytest.fixture(autouse=True)
def _fake_models():
    for name in ("stage-a", "stage-b"):
        reg.register(reg.SLOT_LAYOUT, name, lambda _n=name, **_: _Analyzer(_n), source="test-dist")


def _config(tmp_path: Path, model: str, *, redaction: bool = False) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="t"),
        paths=PathsConfig(input_root=tmp_path, output_root=tmp_path, checkpoint_dir=tmp_path),
        layout={"model": model},
        redaction={"enabled": redaction},
    )


@pytest.fixture
def run(tmp_path: Path):
    """A one-PDF, one-CSV batch extracted under ``stage-a``: ``(corpus, shards)``."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    img = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
    pdf = PdfBuilder(corpus / "scan.pdf").page().image(Rect(50, 50, 450, 450), img).save()
    (corpus / "t.csv").write_text("a,b\n1,2\n")
    shards = tmp_path / "shards"
    process_batch(
        [pdf, corpus / "t.csv"], _config(tmp_path, "stage-a"), batch_num=1, shard_dir=shards,
        provenance=IngestProvenance.declare(corpus, "t"),
    )
    return corpus, shards


def _label(shards: Path) -> set[str]:
    return {r["label"] for r in read_layout_regions(shards).to_pylist()}


def _sidecar(shards: Path) -> Path:
    return shards / "batch-0001.layout_regions.parquet"


def test_extraction_leaves_the_sidecar_matching_the_elements(run) -> None:
    _, shards = run
    assert layout_fingerprint_statuses(shards) == {"batch-0001": MATCH}


def test_the_same_fingerprint_is_skipped(tmp_path: Path, run) -> None:
    corpus, shards = run
    before = _sidecar(shards).stat().st_mtime_ns
    result = layout_shards(shards, _config(tmp_path, "stage-a"), source_root=corpus)
    assert (result.batches_skipped, result.batches_written) == (1, 0)
    assert _sidecar(shards).stat().st_mtime_ns == before


def test_a_new_model_reruns_and_the_two_footers_then_disagree(tmp_path: Path, run) -> None:
    corpus, shards = run
    assert _label(shards) == {"stage-a"}
    result = layout_shards(shards, _config(tmp_path, "stage-b"), source_root=corpus)
    assert (result.batches_written, result.documents) == (1, 1)  # the CSV has no layout
    assert _label(shards) == {"stage-b"}
    assert layout_fingerprint_status(shards / "batch-0001.parquet") == MISMATCH
    fp = read_footer_layout_fingerprint(pq.read_metadata(str(_sidecar(shards))).metadata)
    assert fp is not None and fp.model == "stage-b"


def test_turning_on_the_redaction_filter_reruns_and_never_claims_consumption(tmp_path, run):
    corpus, shards = run
    assert layout_shards(shards, _config(tmp_path, "stage-a", redaction=True), source_root=corpus).batches_written == 1
    assert read_footer_redaction_consumed(pq.read_metadata(str(_sidecar(shards))).metadata) is False


def test_force_reruns_an_unchanged_fingerprint(tmp_path: Path, run) -> None:
    corpus, shards = run
    result = layout_shards(shards, _config(tmp_path, "stage-a"), source_root=corpus, force=True)
    assert result.batches_written == 1


def test_an_unresolvable_source_leaves_the_batch_as_it_was(tmp_path: Path, run) -> None:
    _, shards = run
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    result = layout_shards(shards, _config(tmp_path, "stage-b"), source_root=empty)
    assert (result.batches_failed, result.batches_written) == (1, 0)
    assert _label(shards) == {"stage-a"}


def test_a_run_from_before_the_layout_stage_reads_as_unknown(run) -> None:
    _, shards = run
    _sidecar(shards).unlink()
    assert layout_fingerprint_status(shards / "batch-0001.parquet") == UNKNOWN


def test_the_contract_runs_locally(tmp_path: Path, run) -> None:
    from womblex.cloud.stage_contracts import RunContext

    corpus, shards = run
    run_stage_local(
        STAGE_CONTRACTS["layout"], shards, _config(tmp_path, "stage-b"),
        ctx=RunContext(source_root=str(corpus)),
    )
    assert _label(shards) == {"stage-b"}


class TestStoreRun:
    """The distributed path: shards in a store, documents fetched from an ingest store."""

    PREFIX = "runs/r1/documents"

    @pytest.fixture
    def stores(self, tmp_path: Path, run):
        import shutil

        corpus, shards = run
        root = tmp_path / "store"
        shutil.copytree(shards, root / self.PREFIX)
        return RemoteStore.from_uri(str(root)), RemoteStore.from_uri(str(corpus)), root, corpus

    def _go(self, tmp_path: Path, store, ingest, model: str = "stage-b"):
        return run_stage_remote(
            STAGE_CONTRACTS["layout"], store, self.PREFIX, _config(tmp_path, model), ingest=ingest,
        )

    def test_a_new_model_replaces_the_published_sidecar(self, tmp_path: Path, stores) -> None:
        store, ingest, root, _ = stores
        summary = self._go(tmp_path, store, ingest)
        assert (summary.processed, summary.failed, summary.exit_code) == (1, 0, 0)
        assert _label(root / self.PREFIX) == {"stage-b"}
        assert not list((root / self.PREFIX / ".staging").glob("*"))

    def test_an_unchanged_fingerprint_fetches_no_source(self, tmp_path: Path, stores) -> None:
        store, _, root, _ = stores

        class _Tripwire(RemoteStore):
            def download_file(self, rel, local_path):
                raise AssertionError("a skipped batch must not download sources")

        ingest = _Tripwire.from_uri(str(root))
        assert self._go(tmp_path, store, ingest, "stage-a").exit_code == 0
        assert _label(root / self.PREFIX) == {"stage-a"}

    def test_a_missing_source_fails_the_batch_and_keeps_the_sidecar(self, tmp_path: Path, stores) -> None:
        store, ingest, root, corpus = stores
        (corpus / "scan.pdf").unlink()
        summary = self._go(tmp_path, store, ingest)
        assert (summary.failed, summary.exit_code) == (1, 1)
        assert _label(root / self.PREFIX) == {"stage-a"}

    def test_a_source_whose_bytes_changed_is_refused(self, tmp_path: Path, stores) -> None:
        store, ingest, root, corpus = stores
        other = PdfBuilder(tmp_path / "other.pdf").page().text(72, 72, "different").save()
        (corpus / "scan.pdf").write_bytes(other.read_bytes())
        assert self._go(tmp_path, store, ingest).failed == 1
        assert _label(root / self.PREFIX) == {"stage-a"}

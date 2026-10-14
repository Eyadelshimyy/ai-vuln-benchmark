"""Contract-version + sensitivity footer keys on every pipeline Parquet."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.store.contract import (
    CONTRACT_VERSION,
    ROLE_SENSITIVITY,
    read_footer_contract,
    role_of,
    sensitivity_for,
)
from womblex.store.egress_output import write_source_index
from womblex.store.embed_output import write_embeddings
from womblex.store.enrichment_output import (
    write_enrichment_metadata,
    write_entity_mentions,
    write_graph_edges,
)
from womblex.store.output import _write_rows
from womblex.store.pii_output import write_clean_text, write_pii_spans
from womblex.store.provenance_output import write_corpus_manifest, write_provenance_shard
from womblex.store.run_stamp import RunStamp, read_footer_stamp

SRC = Path(__file__).resolve().parents[1] / "src" / "womblex"


def _contract(path: Path) -> dict[str, str]:
    return read_footer_contract(pq.read_schema(str(path)).metadata)


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("batch-0001.elements.parquet", "elements"),
        ("batch-0001._manifest.parquet", "_manifest"),
        ("batch-0001.clean_text.parquet", "clean_text"),
        ("manifest.parquet", "manifest"),
    ],
)
def test_role_is_the_segment_before_parquet(name: str, role: str) -> None:
    assert role_of(Path(name)) == role


def test_unknown_role_reads_as_raw() -> None:
    assert sensitivity_for("something_new") == "raw"


def test_only_clean_text_is_masked() -> None:
    assert [r for r, s in ROLE_SENSITIVITY.items() if s == "masked"] == ["clean_text"]


def test_every_sidecar_role_in_the_source_is_classified() -> None:
    """A new ``*.<role>.parquet`` must be classified here, not fall back to ``raw`` silently."""
    pattern = re.compile(r"\.([a-z_]+)\.parquet")
    roles = {m for f in SRC.rglob("*.py") for m in pattern.findall(f.read_text())}
    assert roles, "scan found no sidecar roles"
    assert roles <= ROLE_SENSITIVITY.keys(), sorted(roles - ROLE_SENSITIVITY.keys())


def test_write_rows_stamps_contract_alongside_run(tmp_path: Path) -> None:
    stamp = RunStamp.inherit("run-1", "sha256:abc", stage="extract")
    path = tmp_path / "batch-0001.chunks.parquet"
    _write_rows([], path, pa.schema([("source_hash", pa.string())]), metadata=stamp.footer_metadata())
    meta = pq.read_schema(str(path)).metadata
    assert read_footer_contract(meta) == {"contract_version": CONTRACT_VERSION, "sensitivity": "raw"}
    assert read_footer_stamp(meta)["run_id"] == "run-1"


@pytest.mark.parametrize(
    ("write", "sensitivity"),
    [
        (lambda base: write_clean_text([], base), "masked"),
        (lambda base: write_pii_spans([], base), "raw"),
        (lambda base: write_embeddings([], base), "none"),
        (lambda base: write_provenance_shard([], ["title"], base), "raw"),
    ],
)
def test_sidecar_writers_carry_contract(tmp_path: Path, write, sensitivity: str) -> None:
    written = write(tmp_path / "batch-0001.parquet")
    assert _contract(written) == {"contract_version": CONTRACT_VERSION, "sensitivity": sensitivity}


@pytest.mark.parametrize(
    ("write", "sensitivity"),
    [
        (write_entity_mentions, "raw"),
        (write_graph_edges, "raw"),
        (write_enrichment_metadata, "none"),
    ],
)
def test_enrichment_writers_classify_by_role_not_filename(tmp_path: Path, write, sensitivity: str) -> None:
    written = write([], tmp_path / "anything.parquet")
    assert _contract(written)["sensitivity"] == sensitivity


def test_corpus_manifest_is_labelled_as_provenance(tmp_path: Path) -> None:
    shards = tmp_path / "documents"
    write_provenance_shard([{"source_hash": "h", "doc_id": "d", "title": "t"}], ["title"], shards / "batch-0001.parquet")
    assert _contract(write_corpus_manifest(shards))["sensitivity"] == "raw"


def test_source_index_carries_contract_when_unstamped(tmp_path: Path) -> None:
    written = write_source_index([], tmp_path, stamp=None)
    assert _contract(written) == {"contract_version": CONTRACT_VERSION, "sensitivity": "none"}


def test_run_manifest_and_redactions_carry_contract(tmp_path: Path) -> None:
    from womblex.redact.batch import _write_redactions_parquet
    from womblex.store.run_manifest import write_run_manifest

    shards = tmp_path / "documents"
    shards.mkdir()
    assert _contract(write_run_manifest(shards))["sensitivity"] == "none"
    redactions = shards / "batch-0001.redactions.parquet"
    _write_redactions_parquet([("h", 0)], redactions)
    assert _contract(redactions)["sensitivity"] == "none"
    assert pq.read_table(str(redactions)).to_pylist() == [
        {"source_hash": "h", "elem_order": 0, "has_redaction": True}
    ]

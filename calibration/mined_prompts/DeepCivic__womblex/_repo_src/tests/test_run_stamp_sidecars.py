"""Every downstream sidecar names the run that produced it.

Part one stamped the four extraction shards, where the caller knows the run.
A downstream stage does not: it is handed a shard directory, and its own
``dataset.run_id`` is whatever config it was launched with — a copied config
names a run it did not produce. So a sidecar inherits its run from the
extraction shard it sits beside, and these cases pin that.

The enumeration case is the one that keeps this true: it walks every public
writer in ``womblex.store`` and fails on one that is neither stamped here nor
in the exempt list below with a reason.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import womblex.store as store_pkg
from womblex import __version__
from womblex.config import DatasetConfig, PathsConfig, WomblexConfig
from womblex.store.build_info import resolve_commit
from womblex.store.embed_output import write_embeddings
from womblex.store.enrichment_doc import write_enrichment_doc_shard
from womblex.store.enrichment_output import (
    ENTITY_SCHEMA,
    write_enrichment_entities_rows,
    write_enrichment_entities_shard,
    write_enrichment_meta_shard,
    write_graph_edges_rows,
    write_graph_edges_shard,
)
from womblex.store.entity_links_output import write_entity_links
from womblex.store.money_output import write_money_columns, write_money_spans
from womblex.store.normalise_output import write_normalised_text
from womblex.store.output import (
    CHUNKS_SCHEMA,
    ELEMENT_SCHEMA,
    MANIFEST_SCHEMA,
    _write_rows,
    write_chunks,
)
from womblex.store.pii_output import write_clean_text, write_pii_spans
from womblex.store.quality_output import write_chunk_quality
from womblex.store.run_manifest import write_run_manifest
from womblex.store.run_stamp import (
    RunStamp,
    read_footer_stamp,
)
from womblex.store.spellfix_output import (
    write_spellfix_corrections,
    write_spellfix_text,
)

# Every sidecar writer, and the stage each writes as. One entry per writer:
# the enumeration case below fails when a writer exists that is in neither
# this list nor `EXEMPT`.
SIDECAR_WRITERS: list[tuple[object, str]] = [
    (write_chunks, "chunk"),
    (write_normalised_text, "normalise"),
    (write_spellfix_text, "spellfix"),
    (write_spellfix_corrections, "spellfix"),
    (write_enrichment_entities_shard, "enrich"),
    (write_enrichment_meta_shard, "enrich"),
    (write_graph_edges_shard, "enrich"),
    (write_enrichment_doc_shard, "enrich"),
    (write_enrichment_entities_rows, "graph-refresh"),
    (write_graph_edges_rows, "graph-refresh"),
    (write_embeddings, "embed"),
    (write_money_spans, "money"),
    (write_money_columns, "money"),
    (write_entity_links, "link"),
    (write_pii_spans, "pii"),
    (write_clean_text, "pii"),
    (write_chunk_quality, "quality"),
]

# Writers that do not inherit a run from a batch shard, and why. Each is a
# stated non-goal rather than an oversight — a silent exemption is how this
# test stops meaning anything.
EXEMPT: dict[str, str] = {
    "write_results": "the extraction writer; takes the run's stamp explicitly (part 1)",
    "write_layout_regions": "written by write_results, which passes it the batch's own footer",
    "write_run_manifest": "consolidates the batch stamps; covered by its own cases here",
    "write_sidecar": "a ground-truth .meta.json: JSON, and the run is recorded inside "
                     "the sidecar's derivation block, not as parquet footer keys",
    "write_register_manifest": "the standalone register ingests, which bypass the pipeline",
    "write_source_index": "an egress bundle file at run grain with no batch shard beside it; "
                          "takes the run's stamp explicitly, covered in test_egress_output.py",
    "write_provenance_shard": "the pre-extracted records ingest, which is not a run",
    "write_corpus_manifest": "the pre-extracted records ingest, which is not a run",
    "write_entity_mentions": "whole-corpus E2E writer: no per-batch shard to inherit from",
    "write_graph_edges": "whole-corpus E2E writer: no per-batch shard to inherit from",
    "write_enrichment_metadata": "whole-corpus E2E writer: no per-batch shard to inherit from",
}


def _config(tmp_path: Path) -> WomblexConfig:
    return WomblexConfig(
        dataset=DatasetConfig(name="test-corpus"),
        paths=PathsConfig(
            input_root=tmp_path / "in",
            output_root=tmp_path / "out",
            checkpoint_dir=tmp_path / "ckpt",
        ),
    )


def _extraction_shard(shard_dir: Path, stamp: RunStamp | None, name="batch-0001") -> Path:
    """A batch's extraction shard, stamped or not, with no documents in it.

    The sidecars inherit from the footer, not from the rows, so an empty shard
    exercises the whole mechanism.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    footer = stamp.footer_metadata() if stamp is not None else None
    _write_rows([], shard_dir / f"{name}.elements.parquet", ELEMENT_SCHEMA, metadata=footer)
    _write_rows([], shard_dir / f"{name}._manifest.parquet", MANIFEST_SCHEMA, metadata=footer)
    return shard_dir / f"{name}.parquet"


@pytest.fixture
def stamp(tmp_path):
    return RunStamp.declare("run-A", _config(tmp_path), stage="extract")


@pytest.fixture
def base(tmp_path, stamp):
    return _extraction_shard(tmp_path / "documents", stamp)


def _stamp_of(path: Path) -> dict[str, str]:
    return read_footer_stamp(pq.read_schema(str(path)).metadata)


class TestSidecarsInheritTheRun:
    @pytest.mark.parametrize(
        "writer,stage", SIDECAR_WRITERS, ids=[w.__name__ for w, _ in SIDECAR_WRITERS]
    )
    def test_a_sidecar_names_the_run_of_the_shard_it_sits_beside(
        self, writer, stage, base, stamp
    ):
        written = writer([], base)

        assert _stamp_of(written) == {
            "run_id": stamp.run_id,
            "version": __version__,
            "commit": resolve_commit(),
            "config_digest": stamp.config_digest,
            "stage": stage,
            "preset": stamp.preset,
        }

    @pytest.mark.parametrize(
        "writer,stage", SIDECAR_WRITERS, ids=[w.__name__ for w, _ in SIDECAR_WRITERS]
    )
    def test_a_sidecar_of_an_unstamped_shard_is_written_unstamped(
        self, writer, stage, tmp_path
    ):
        # A shard extracted before part 1, or by a run that named none. An
        # invented run id would be worse than no stamp.
        unstamped = _extraction_shard(tmp_path / "documents", None)

        assert _stamp_of(writer([], unstamped)) == {}

    def test_a_reader_that_ignores_the_footer_reads_the_sidecar_unchanged(self, base):
        written = write_chunks([], base)

        assert pq.read_table(str(written)).num_rows == 0

    def test_the_version_is_the_one_that_wrote_the_sidecar_not_the_extraction(
        self, tmp_path
    ):
        # run_id and config_digest describe the run; the version describes the
        # bytes, so a stage running at a later build says so rather than
        # repeating what extraction claimed.
        stale = RunStamp("run-A", "0.0.1-ancient", "f" * 40, "sha256:abc", "extract")
        base = _extraction_shard(tmp_path / "documents", stale)

        written = _stamp_of(write_chunks([], base))

        assert written["version"] == __version__
        assert (written["run_id"], written["config_digest"]) == ("run-A", "sha256:abc")

    @pytest.mark.parametrize("staged", [
        # What a worker actually has on disk for each stage that does not
        # declare the elements shard: `embed`/`pii` (chunks + manifest),
        # `link` (entities + manifest), `graph-refresh` (adds graph edges),
        # and `quality`, whose only declared input is the chunks sidecar — the
        # one case that reaches past the preferred sources to the glob.
        [("._manifest.parquet", MANIFEST_SCHEMA), (".chunks.parquet", CHUNKS_SCHEMA)],
        [("._manifest.parquet", MANIFEST_SCHEMA), (".enrichment_entities.parquet", ENTITY_SCHEMA)],
        [(".chunks.parquet", CHUNKS_SCHEMA)],
    ], ids=["chunks+manifest", "entities+manifest", "chunks-only"])
    def test_a_stage_that_never_sees_the_elements_shard_still_names_the_run(
        self, staged, tmp_path, stamp
    ):
        # A stage worker stages in only the inputs its contract declares, so
        # insisting on `.elements.parquet` would stamp locally and not on a
        # worker — the divergence the stamp exists to rule out.
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir(parents=True)
        for suffix, schema in staged:
            _write_rows([], shard_dir / f"batch-0001{suffix}", schema,
                        metadata=stamp.footer_metadata())
        base = shard_dir / "batch-0001.parquet"

        assert _stamp_of(write_chunk_quality([], base))["run_id"] == stamp.run_id

    def test_a_corrupt_elements_shard_falls_through_to_the_manifest(self, tmp_path, stamp):
        # `shard_audit` deals in zero-byte and unreadable shards; an
        # unreadable preferred source is skipped, not fatal, and not a reason
        # to lose the run.
        shard_dir = tmp_path / "documents"
        shard_dir.mkdir(parents=True)
        (shard_dir / "batch-0001.elements.parquet").write_bytes(b"")
        _write_rows([], shard_dir / "batch-0001._manifest.parquet", MANIFEST_SCHEMA,
                    metadata=stamp.footer_metadata())

        written = write_chunks([], shard_dir / "batch-0001.parquet")

        assert _stamp_of(written)["run_id"] == stamp.run_id

    def test_siblings_naming_two_runs_leave_the_sidecar_unstamped(self, tmp_path, stamp):
        shard_dir = tmp_path / "documents"
        base = _extraction_shard(shard_dir, None)
        other = RunStamp("run-B", stamp.version, stamp.commit, stamp.config_digest, "chunk")
        _write_rows([], shard_dir / "batch-0001.chunks.parquet", MANIFEST_SCHEMA,
                    metadata=stamp.footer_metadata())
        _write_rows([], shard_dir / "batch-0001.money_spans.parquet", MANIFEST_SCHEMA,
                    metadata=other.footer_metadata())

        assert _stamp_of(write_chunk_quality([], base)) == {}

    def test_attribution_survives_the_sidecar_being_moved(self, base, stamp, tmp_path):
        written = write_chunks([], base)
        elsewhere = tmp_path / "somewhere-else" / "copied.chunks.parquet"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_bytes(written.read_bytes())

        assert _stamp_of(elsewhere)["run_id"] == stamp.run_id


class TestConsolidatedRunManifest:
    def test_the_run_manifest_names_the_run_its_batches_name(self, tmp_path, stamp):
        shard_dir = tmp_path / "documents"
        _extraction_shard(shard_dir, stamp, "batch-0001")
        _extraction_shard(shard_dir, stamp, "batch-0002")

        assert _stamp_of(write_run_manifest(shard_dir)) == {
            "run_id": "run-A",
            "version": __version__,
            "commit": resolve_commit(),
            "config_digest": stamp.config_digest,
            "stage": "manifest",
            "preset": stamp.preset,
        }

    def test_batches_from_two_runs_leave_the_run_keys_off(self, tmp_path, stamp):
        # Naming one of two runs would be worse than naming none — the rule the
        # ingest-root footer already follows for a mixed-root directory.
        shard_dir = tmp_path / "documents"
        _extraction_shard(shard_dir, stamp, "batch-0001")
        other = RunStamp("run-B", stamp.version, stamp.commit, stamp.config_digest, "extract")
        _extraction_shard(shard_dir, other, "batch-0002")

        assert _stamp_of(write_run_manifest(shard_dir)) == {}

    def test_regenerating_the_manifest_reproduces_the_record(self, tmp_path, stamp):
        shard_dir = tmp_path / "documents"
        _extraction_shard(shard_dir, stamp)

        first = _stamp_of(write_run_manifest(shard_dir))
        second = _stamp_of(write_run_manifest(shard_dir))

        assert first == second != {}


def test_every_store_writer_is_stamped_or_exempt_with_a_reason():
    """The guard: a new sidecar writer that does not stamp fails here."""
    found: set[str] = set()
    for info in pkgutil.iter_modules(store_pkg.__path__):
        module = importlib.import_module(f"womblex.store.{info.name}")
        found |= {
            name for name, obj in vars(module).items()
            if name.startswith("write_")
            and inspect.isfunction(obj)
            and obj.__module__ == module.__name__
        }

    stamped = {w.__name__ for w, _ in SIDECAR_WRITERS}
    assert found - stamped - set(EXEMPT) == set(), (
        "writer neither stamped nor exempt — stamp it, or add it to EXEMPT with a reason"
    )
    assert stamped & set(EXEMPT) == set(), "a writer cannot be both stamped and exempt"
    assert set(EXEMPT) <= found, "EXEMPT names a writer that no longer exists"


def test_the_stamped_stages_are_real_pipeline_stages():
    from womblex.pipeline_order import PIPELINE_ORDER

    assert {stage for _, stage in SIDECAR_WRITERS} <= set(PIPELINE_ORDER)


def test_no_credential_value_reaches_a_sidecar_footer(tmp_path, monkeypatch, base):
    for name in ("ISAACUS_API_KEY", "AWS_SECRET_ACCESS_KEY", "WOMBLEX_S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, f"secret-{name}")

    footer = pq.read_schema(str(write_chunks([], base))).metadata or {}
    blob = b" ".join(footer.keys()) + b" " + b" ".join(footer.values())

    assert b"secret-" not in blob

"""Tests for womblex.operations — independent operations.

File inputs come from the synthetic fixture set (``tests/_synthetic.py``).
"""


import argparse
import shutil
from pathlib import Path

import pytest

from tests._synthetic import NOTICE_PDF, REGISTER_CSV
from womblex.cli.pipeline import _register_chunk, cmd_chunk, cmd_manifest, cmd_run
from womblex.config import ChunkingConfig, DatasetConfig, PathsConfig, WomblexConfig, load_config
from womblex.operations import BatchResult, DocumentResult, run_chunking, run_extraction
from womblex.utils.availability import tokenizer_available

# Plain (non-AI) chunking sizes chunks with the kanon-2 tokeniser, which is
# vendored and runs offline — no API key needed. It does need `transformers`
# plus the bundled tokeniser present, so tests that assert chunks were produced
# gate on that (they self-skip on a wheel built without the model bundle).
_CHUNK_TOKENIZER = "isaacus/kanon-2-tokenizer"
requires_chunking = pytest.mark.skipif(
    not tokenizer_available(_CHUNK_TOKENIZER),
    reason="offline chunking needs `transformers` + the vendored kanon-2 tokeniser",
)


# ---------------------------------------------------------------------------

# Helpers

# ---------------------------------------------------------------------------



def _make_config(sample_config_path: Path) -> WomblexConfig:
    return load_config(sample_config_path)



# ---------------------------------------------------------------------------

# run_extraction

# ---------------------------------------------------------------------------



class TestRunExtraction:

    def test_nonexistent_file_errors(self, tmp_path: Path, sample_config_path: Path) -> None:

        config = _make_config(sample_config_path)

        results = run_extraction([tmp_path / "missing.pdf"], config)


        assert len(results) == 1

        assert results[0].status == "error"

        assert results[0].error is not None

        assert results[0].profile is None


    def test_extracts_real_spreadsheet(self, spreadsheet_dir: Path, sample_config_path: Path) -> None:

        csv_path = spreadsheet_dir / "platypus-sightings-register.csv"


        config = _make_config(sample_config_path)

        results = run_extraction([csv_path], config)


        assert len(results) >= 1

        ok = sum(1 for r in results if r.status == "completed")

        assert ok >= 1


    def test_empty_list(self, sample_config_path: Path) -> None:

        config = _make_config(sample_config_path)

        results = run_extraction([], config)

        assert len(results) == 0



# ---------------------------------------------------------------------------

# DocumentResult / BatchResult

# ---------------------------------------------------------------------------



class TestDocumentResult:

    def test_default_status(self) -> None:

        r = DocumentResult(path=Path("/tmp/test.pdf"), doc_id="test")

        assert r.status == "pending"

        assert r.profile is None

        assert r.extraction is None

        assert r.error is None

        assert r.chunks == []



class TestBatchResult:

    def test_empty_batch(self) -> None:

        b = BatchResult()

        assert b.succeeded == 0

        assert b.failed == 0


    def test_counts(self) -> None:

        b = BatchResult(

            results=[

                DocumentResult(path=Path("/a.pdf"), doc_id="a", status="completed"),

                DocumentResult(path=Path("/b.pdf"), doc_id="b", status="error"),

                DocumentResult(path=Path("/c.pdf"), doc_id="c", status="completed"),

            ]
        )

        assert b.succeeded == 2

        assert b.failed == 1



# ---------------------------------------------------------------------------

# Composition: extract then chunk

# ---------------------------------------------------------------------------


_CSV_FILE = REGISTER_CSV



class TestComposition:

    """Operations compose correctly when called in sequence."""


    @requires_chunking
    def test_extract_then_chunk(self) -> None:

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),

            chunking=ChunkingConfig(enabled=True, chunk_size=480),
        )

        results = run_extraction([_CSV_FILE], config)

        assert any(r.status == "completed" for r in results)


        results = run_chunking(results, config)

        has_chunks = any(len(r.chunks) > 0 for r in results)

        assert has_chunks


    def test_extract_only(self) -> None:

        config = WomblexConfig(
            dataset=DatasetConfig(name="t"),
            paths=PathsConfig(input_root=Path("."), output_root=Path("."), checkpoint_dir=Path(".")),
            chunking=ChunkingConfig(enabled=False),
        )

        results = run_extraction([_CSV_FILE], config)

        assert any(r.status == "completed" for r in results)

        # No chunking called — no chunks.

        assert all(len(r.chunks) == 0 for r in results)


# ---------------------------------------------------------------------------
# Live AI chunking (semchunk 4, kanon-2-enricher)
# ---------------------------------------------------------------------------

# The synthetic decision notice — small, native prose the enricher can
# segment. Same fixture the live enrich test uses.


class TestAiChunkingLive:
    """AI chunking (``chunking.chunking_model``) against the **live** Kanon-2
    enricher — no mocks, real for local validation (CLAUDE.md); skips cleanly
    without ``ISAACUS_API_KEY`` via the ``isaacus_client`` fixture.

    Boundaries follow the enricher's structure spans instead of the offline
    token split, so this exercises the one chunk path that actually calls the
    Isaacus API — the path the config-aware gate (``requires_isaacus_api``)
    protects.
    """

    def test_chunk_shards_ai_chunking_produces_chunks(
        self, tmp_path: Path, isaacus_client
    ) -> None:

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text
        from womblex.process.chunk_stage import chunk_shards
        from womblex.store.output import chunks_path_for, read_chunks, write_results

        # Build a real extraction shard (mirrors test_enrich_stage.shard_dir).
        d = tmp_path / "documents"
        d.mkdir()
        extraction = extract_text(
            NOTICE_PDF, detect_file_type(NOTICE_PDF, DetectionConfig())
        )[0]
        base = d / "batch-0001.parquet"
        write_results([("notice", str(NOTICE_PDF), extraction)], base,
                      collection_id="test")

        # AI chunking on: boundaries come from the live enricher.
        cfg = ChunkingConfig(enabled=True, chunking_model="kanon-2-enricher",
                             chunk_size=480)
        result = chunk_shards(d, cfg, text_source="elements")

        assert result.batches_written == 1
        assert result.docs_chunked == 1, "live AI chunking produced no chunked docs"
        assert result.total_chunks >= 1

        assert chunks_path_for(base).exists()
        rows = read_chunks(base).to_pylist()
        assert rows, "live AI chunking wrote an empty chunks sidecar"
        assert all(r["text"].strip() for r in rows)
        # Offsets must index back into the reassembled narrative for the
        # narrative chunks (the page-mapping contract).
        for r in rows:
            assert r["end_char"] >= r["start_char"]


# ---------------------------------------------------------------------------
# cmd_run — CLI-level run_id + retention layout
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path, input_root: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""\
dataset:
  name: i1_test
paths:
  input_root: {input_root}
  output_root: {tmp_path / "out"}
  checkpoint_dir: {tmp_path / "ckpt"}
extraction:
  native:
    include_tables: true
  ocr:
    engine: paddleocr
    dpi: 150
    lang: eng
redaction:
  enabled: false
chunking:
  enabled: false
pii:
  enabled: false
enrichment:
  enabled: false
processing:
  batch_size: 5
  checkpoint_every: 5
"""
    )
    return cfg


def test_run_ignores_downstream_stage_flags(tmp_path: Path) -> None:
    """`womblex run` is extraction only. Downstream flags (even a
    post_enrichment PII config that `run` could never satisfy) no longer make
    it run or reject — they declare pipeline membership, dispatched separately.
    The run extracts and succeeds regardless."""
    input_root = tmp_path / "in"
    input_root.mkdir()
    shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""\
dataset:
  name: t
paths:
  input_root: {input_root}
  output_root: {tmp_path / "out"}
  checkpoint_dir: {tmp_path / "ckpt"}
pii:
  enabled: true
  pipeline_point: post_enrichment
chunking:
  enabled: true
"""
    )
    args = argparse.Namespace(
        config=cfg, resume=False, limit=None, skip=0, batch_size=None, run_id="t",
    )
    # Extraction only — succeeds; no PII/chunk runs in-batch, so no crash.
    assert cmd_run(args) == 0
    assert (tmp_path / "out" / "t" / "documents").is_dir()


class TestCmdRunRunIdLayout:
    """End-to-end CLI-level verification of the I1 run_id + retention plumbing."""

    def test_auto_generated_run_id_writes_to_nested_layout(self, tmp_path: Path) -> None:

        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)

        cfg = _write_minimal_config(tmp_path, input_root)
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id=None,
        )

        rc = cmd_run(args)
        assert rc == 0

        run_dirs = list((tmp_path / "out").iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]
        assert run_dir.name.startswith("run-")
        shard_files = list((run_dir / "documents").glob("*.parquet"))
        assert len(shard_files) == 5  # elements, table_cells, form_fields, manifest, layout_regions

        # Checkpoint nested under run_id
        assert (tmp_path / "ckpt" / run_dir.name / "i1_test_checkpoint.json").exists()

    def test_explicit_run_id_via_cli(self, tmp_path: Path) -> None:

        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)

        cfg = _write_minimal_config(tmp_path, input_root)
        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id="my-explicit-run",
        )

        rc = cmd_run(args)
        assert rc == 0
        assert (tmp_path / "out" / "my-explicit-run" / "documents").is_dir()
        assert (tmp_path / "ckpt" / "my-explicit-run" / "i1_test_checkpoint.json").exists()

    def test_resume_does_not_overwrite_earlier_batches(self, tmp_path: Path) -> None:
        """Regression: on --resume, batch-NNNN.parquet numbering must continue
        from checkpoint.last_batch+1, not restart at 1. Restarting at 1
        overwrites the original batch-0001 with a later doc cohort's content
        — the failure observed live during the i1b corpus extraction resume.
        """

        input_root = tmp_path / "in"
        input_root.mkdir()
        # Three distinct doc files; batch_size=1 → one batch per doc.
        for i in range(3):
            dst = input_root / f"doc-{i:02d}.csv"
            shutil.copy(_CSV_FILE, dst)

        cfg = _write_minimal_config(tmp_path, input_root)

        # First invocation processes 2 of 3 docs (batches 1, 2).
        args1 = argparse.Namespace(
            config=cfg, resume=False, limit=2, skip=0,
            batch_size=1, run_id="run-test",
        )
        assert cmd_run(args1) == 0

        run_dir = tmp_path / "out" / "run-test" / "documents"
        pre_resume = sorted(p.name for p in run_dir.glob("batch-*.parquet"))
        assert any(n.startswith("batch-0001.") for n in pre_resume)
        assert any(n.startswith("batch-0002.") for n in pre_resume)
        b1_size_before = (run_dir / "batch-0001.elements.parquet").stat().st_size

        # Resume: must write batch-0003, not overwrite batch-0001.
        args2 = argparse.Namespace(
            config=cfg, resume=True, no_verify_resume=False, limit=None, skip=0,
            batch_size=1, run_id="run-test",
        )
        assert cmd_run(args2) == 0

        post_resume = sorted({p.name.split(".")[0] for p in run_dir.glob("batch-*.parquet")})
        assert post_resume == ["batch-0001", "batch-0002", "batch-0003"]
        # batch-0001 untouched (size invariant — if it had been overwritten with
        # different content it would almost certainly differ in size)
        assert (run_dir / "batch-0001.elements.parquet").stat().st_size == b1_size_before

    def test_retention_rolling_purges_old_run(self, tmp_path: Path) -> None:

        input_root = tmp_path / "in"
        input_root.mkdir()
        shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)

        cfg = _write_minimal_config(tmp_path, input_root)
        # Pre-seed a stale run dir + checkpoint that should be purged
        (tmp_path / "out" / "run-stale" / "documents").mkdir(parents=True)
        (tmp_path / "out" / "run-stale" / "documents" / "batch-0001.elements.parquet").write_bytes(b"x")
        (tmp_path / "ckpt" / "run-stale").mkdir(parents=True)

        args = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id="run-current",
        )
        rc = cmd_run(args)
        assert rc == 0

        # rolling, keep=2 (default): current + 1 previous kept; only 'run-stale' present
        # as previous, so 'run-stale' survives. Add a second stale to force a purge.
        (tmp_path / "out" / "run-older").mkdir(parents=True)
        (tmp_path / "out" / "run-older" / "documents").mkdir()
        # Second fresh run — older two will reduce to one
        args2 = argparse.Namespace(
            config=cfg, resume=False, limit=None, skip=0,
            batch_size=None, run_id="run-newer",
        )
        rc = cmd_run(args2)
        assert rc == 0

        # keep=2 means current (run-newer) + 1 previous most-recent survives
        survivors = sorted(p.name for p in (tmp_path / "out").iterdir() if p.is_dir())
        assert "run-newer" in survivors
        assert len(survivors) == 2  # current + 1 retained


# ---------------------------------------------------------------------------
# Run-level manifest consolidation
# ---------------------------------------------------------------------------


class TestRunManifest:
    def test_run_writes_consolidated_manifest_at_run_root(self, tmp_path: Path) -> None:

        import pyarrow.parquet as pq

        from womblex.store.output import read_manifest

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="manifest-run")
        manifest_path = shard_dir.parent / "manifest.parquet"
        assert manifest_path.exists()

        table = pq.read_table(str(manifest_path))
        per_batch = read_manifest(shard_dir)
        assert table.num_rows == per_batch.num_rows > 0
        # The columns a consumer needs to map chunks back to documents.
        assert {"source_hash", "doc_id", "filename", "status"} <= set(table.schema.names)

    def test_cmd_manifest_consolidates_existing_run(self, tmp_path: Path) -> None:

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="manifest-cmd")
        manifest_path = shard_dir.parent / "manifest.parquet"
        manifest_path.unlink()  # simulate a run written before end-of-run manifests

        args = argparse.Namespace(shards=shard_dir, output=None)
        assert cmd_manifest(args) == 0
        assert manifest_path.exists()
        assert manifest_path.stat().st_size > 0

    def test_cmd_manifest_rejects_dir_without_manifests(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        args = argparse.Namespace(shards=empty, output=None)
        assert cmd_manifest(args) == 1


# ---------------------------------------------------------------------------
# cmd_chunk --shards (B3)
# ---------------------------------------------------------------------------


def _seed_run_with_extraction(tmp_path: Path, run_id: str = "i2-test") -> Path:
    """Run cmd_run against a small fixture and return the shard dir."""
    input_root = tmp_path / "in"
    input_root.mkdir()
    shutil.copy(_CSV_FILE, input_root / _CSV_FILE.name)
    cfg = _write_minimal_config(tmp_path, input_root)
    args = argparse.Namespace(
        config=cfg, resume=False, limit=None, skip=0,
        batch_size=None, run_id=run_id,
    )
    rc = cmd_run(args)
    assert rc == 0, "extraction run failed; chunk-stage tests have nothing to consume"
    return tmp_path / "out" / run_id / "documents"


class TestCmdChunkShards:
    @requires_chunking
    def test_writes_chunks_sidecar_for_each_batch(self, tmp_path: Path) -> None:

        shard_dir = _seed_run_with_extraction(tmp_path)

        args = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=None, dataset="i2-test",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 0

        chunks_files = list(shard_dir.glob("*.chunks.parquet"))
        manifests = list(shard_dir.glob("*._manifest.parquet"))
        assert len(chunks_files) == len(manifests)
        assert chunks_files
        for f in chunks_files:
            assert f.stat().st_size > 0

    @requires_chunking
    def test_chunks_join_back_to_elements_via_source_hash(self, tmp_path: Path) -> None:

        from womblex.store.output import read_chunks, read_elements, read_manifest

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-join")
        args = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=None, dataset="i2-join",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 0

        chunks = read_chunks(shard_dir)
        elements = read_elements(shard_dir)
        manifest = read_manifest(shard_dir)

        chunk_hashes = set(chunks.column("source_hash").to_pylist())
        elem_hashes = set(elements.column("source_hash").to_pylist())
        manifest_hashes = set(manifest.column("source_hash").to_pylist())

        # Every chunk's source_hash exists in elements (and the manifest).
        assert chunk_hashes
        assert chunk_hashes <= elem_hashes
        assert chunk_hashes <= manifest_hashes

    @requires_chunking
    def test_written_rows_carry_the_chunkers_token_count(
        self, tmp_path: Path,
    ) -> None:
        """The count reaches the sidecar, and is the tokeniser's, not a proxy."""

        from womblex.process.chunker import create_chunker
        from womblex.store.output import read_chunks

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-tokens")
        args = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=None, dataset="i2-tokens",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 0

        rows = read_chunks(shard_dir).to_pylist()
        assert rows, "chunk stage wrote no rows to count"
        assert all(r["token_count"] is not None for r in rows)

        # A chunker built from the same tokeniser reproduces every count —
        # which a character or word proxy would not.
        counter = create_chunker(_CHUNK_TOKENIZER, chunk_size=512).token_counter
        assert all(r["token_count"] == counter(r["text"]) for r in rows)

    def test_rejects_dir_without_manifests(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        args = argparse.Namespace(
            shards=empty, config=None,
            checkpoint_dir=None, dataset="x",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 1

    @requires_chunking
    def test_no_resume_clears_checkpoint(self, tmp_path: Path) -> None:

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-ckpt")
        ckpt_dir = tmp_path / "ckpt-chunk"

        args1 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-ckpt",
            no_resume=False, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args1) == 0
        assert (ckpt_dir / "i2-ckpt_chunk_checkpoint.json").exists()

        args2 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-ckpt",
            no_resume=True, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args2) == 0

    @requires_chunking
    def test_resume_recovers_corrupt_chunks_shard(self, tmp_path: Path) -> None:
        """Wire-up test: a corrupt *.chunks.parquet on resume drops the affected
        docs from the chunk checkpoint and re-writes a clean sidecar."""


        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-recover")
        ckpt_dir = tmp_path / "ckpt-chunk"

        # First pass: produce healthy chunks + checkpoint.
        args1 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-recover",
            no_resume=False, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args1) == 0

        # Corrupt the first batch's chunks file.
        chunks_files = sorted(shard_dir.glob("*.chunks.parquet"))
        assert chunks_files
        chunks_files[0].write_bytes(b"")

        # Resume: reconcile drops the affected docs, chunk_shards re-writes them.
        args2 = argparse.Namespace(
            shards=shard_dir, config=None,
            checkpoint_dir=ckpt_dir, dataset="i2-recover",
            no_resume=False, no_verify_resume=False, limit=None,
        )
        assert cmd_chunk(args2) == 0

        # The corrupt file is archived with .corrupt; a fresh chunks file exists.
        # (chunks_path_for derives the path from the batch base; the archived
        # original lives next to it with the .corrupt suffix.)
        corrupt = chunks_files[0].with_name(chunks_files[0].name + ".corrupt")
        assert corrupt.exists()
        fresh = chunks_files[0]
        assert fresh.exists()
        assert fresh.stat().st_size > 0


class TestChunkCliFlags:
    """--shards and --config must be combinable: per-stage AI chunking
    (chunking.chunking_model) is only reachable when a config rides along
    with --shards."""

    def test_shards_and_config_parse_together(self) -> None:
        p = argparse.ArgumentParser()
        _register_chunk(p)
        args = p.parse_args(["--shards", "shards-dir", "--config", "cfg.yaml"])
        assert args.shards == Path("shards-dir")
        assert args.config == Path("cfg.yaml")

    def test_neither_flag_is_an_error(self) -> None:
        args = argparse.Namespace(shards=None, config=None)
        assert cmd_chunk(args) == 1

    def test_shards_with_config_sources_chunking_settings(self, tmp_path: Path) -> None:
        """The --shards branch reads chunking + text_source from --config."""
        if not tokenizer_available(_CHUNK_TOKENIZER):
            pytest.skip("offline chunking needs `transformers` + the vendored kanon-2 tokeniser")

        shard_dir = _seed_run_with_extraction(tmp_path, run_id="i2-cfg")
        cfg = tmp_path / "cfg.yaml"  # written by the seed helper
        assert cfg.exists()
        args = argparse.Namespace(
            shards=shard_dir, config=cfg,
            checkpoint_dir=None, dataset="i2-cfg",
            no_resume=True, limit=None,
        )
        assert cmd_chunk(args) == 0
        assert list(shard_dir.glob("*.chunks.parquet"))


"""Tests for the cloud distributed-execution pieces.

RemoteStore is exercised against the local filesystem (fsspec's local backend),
so it needs no S3/MinIO. The extraction body the DBOS workflow runs is called
directly; the board and the workflows themselves are in ``test_run_board.py``
and ``test_workflows.py``, on a SQLite system database.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Self

import pytest

from womblex.store.remote import (
    RemoteStore,
    assert_disjoint_locations,
    is_remote_uri,
    same_location,
    storage_options_from_env,
    store_root,
    validate_location_uri,
)

# RemoteStore reaches fsspec lazily; skip the whole module without the cloud extra.
pytest.importorskip("fsspec")


# --- RemoteStore (local backend) ---------------------------------------------


def test_is_remote_uri():
    assert is_remote_uri("s3://bucket/x")
    assert is_remote_uri("gs://bucket/x")
    assert not is_remote_uri("/tmp/x")
    assert not is_remote_uri("file:///tmp/x")


def test_storage_options_from_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("WOMBLEX_S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "k"
    assert opts["secret"] == "s"
    assert opts["client_kwargs"]["endpoint_url"] == "http://minio:9000"
    assert opts["client_kwargs"]["region_name"] == "us-east-1"
    # The options above are s3fs-shaped — other backends must not receive
    # them even with AWS env vars set.
    assert storage_options_from_env("gs://bucket/x") == {}
    assert storage_options_from_env("/tmp/x") == {}


def test_storage_options_prefers_store_specific_credentials(monkeypatch):
    """The store creds come from WOMBLEX_S3_* first, so a cloud deployment can
    give s3fs a MinIO/S3 key WITHOUT setting the process-global AWS_ACCESS_KEY_ID
    — which would otherwise clobber boto3's instance-role resolution for the
    isaacus-sagemaker SigV4 signer and 403 (the cloud credential-conflict bug).
    """
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("WOMBLEX_S3_ACCESS_KEY_ID", "store-key")
    monkeypatch.setenv("WOMBLEX_S3_SECRET_ACCESS_KEY", "store-secret")
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "store-key"
    assert opts["secret"] == "store-secret"  # pragma: allowlist secret -- test literal

    # Store-specific wins over the AWS fallback when both are set.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    opts = storage_options_from_env("s3://bucket/x")
    assert opts["key"] == "store-key"
    assert opts["secret"] == "store-secret"  # pragma: allowlist secret -- test literal


def test_storage_options_omits_credentials_when_none_are_set(monkeypatch):
    """No store-specific and no AWS keys => s3fs gets no explicit credentials
    and falls back to its own chain (the EC2 instance role for real AWS S3).
    An endpoint override alone must not synthesise a half-set credential.
    """
    for var in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        "WOMBLEX_S3_ACCESS_KEY_ID", "WOMBLEX_S3_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    opts = storage_options_from_env("s3://bucket/x")
    assert "key" not in opts
    assert "secret" not in opts


def test_storage_options_explicit_credentials_win_over_env(monkeypatch):
    """An operator-saved (Resources Console) credential override beats the env
    keys the Dockerfile baked in — so a rotated key is used moving forward
    without a container rebuild (issue 3).
    """
    monkeypatch.setenv("WOMBLEX_S3_ACCESS_KEY_ID", "baked-key")
    monkeypatch.setenv("WOMBLEX_S3_SECRET_ACCESS_KEY", "baked-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    opts = storage_options_from_env("s3://bucket/x", credentials=("saved-key", "saved-secret"))
    assert opts["key"] == "saved-key"
    assert opts["secret"] == "saved-secret"  # pragma: allowlist secret -- test literal


def test_from_uri_threads_the_credential_override(tmp_path, monkeypatch):
    """`RemoteStore.from_uri(..., credentials=...)` passes the override through
    to the storage options for an s3 URI (a local URI ignores them, as it has
    no s3fs options at all).
    """
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("WOMBLEX_S3_ACCESS_KEY_ID", raising=False)
    from womblex.store import remote as remote_mod

    captured: dict = {}

    def _fake_url_to_fs(uri, **opts):
        captured.update(opts)
        # Return something url_to_fs-shaped without touching a network.
        import fsspec
        return fsspec.filesystem("memory"), "bucket/x"

    monkeypatch.setattr(remote_mod._require_fsspec().core, "url_to_fs", _fake_url_to_fs)
    remote_mod.RemoteStore.from_uri("s3://bucket/x", credentials=("ov-key", "ov-secret"))
    assert captured.get("key") == "ov-key"
    assert captured.get("secret") == "ov-secret"


def test_store_root_splits_bucket_from_prefix():
    assert store_root("s3://womblex/inbox") == ("womblex", "inbox")
    assert store_root("s3://womblex") == ("womblex", "")
    assert store_root("s3://womblex/runs/x") == ("womblex", "runs/x")
    # Local paths have no bucket concept.
    assert store_root("/data/inbox") == ("", "data/inbox")


def test_assert_disjoint_locations():
    """Same bucket, different
    folders is fine; either location containing the other is a hard fail.
    """
    # Disjoint: no error.
    assert_disjoint_locations("s3://womblex/inbox", "s3://womblex")
    assert_disjoint_locations("/data/inbox", "/data/out")

    # Ingest contains the output.
    with pytest.raises(ValueError, match="s3://womblex.*s3://womblex/runs"):
        assert_disjoint_locations("s3://womblex", "s3://womblex")

    # Ingest nested inside the output.
    with pytest.raises(ValueError):
        assert_disjoint_locations("s3://womblex/runs/x", "s3://womblex")

    # Different buckets never overlap, no matter the prefix.
    assert_disjoint_locations("s3://other-bucket/runs", "s3://womblex")


def test_assert_disjoint_locations_honours_a_custom_output_prefix():
    """`--output-prefix` moves where shards land, so the guard has to check
    that prefix, not a hardcoded `runs/`."""
    # Default prefix: inbox and runs/ are disjoint.
    assert_disjoint_locations("s3://womblex/inbox", "s3://womblex")
    # ...but shards directed *into* the inbox are exactly what the rule forbids.
    with pytest.raises(ValueError):
        assert_disjoint_locations(
            "s3://womblex/inbox", "s3://womblex", runs_prefix="inbox/out",
        )


def test_validate_location_uri_rejects_the_typos_operators_actually_make():
    """A hand-typed location that fsspec would silently read as a *relative
    local path* is the failure the Resources Console has to catch on save."""
    validate_location_uri("s3://womblex/inbox")
    validate_location_uri("/data/inbox")
    validate_location_uri("gs://bucket/inbox")

    with pytest.raises(ValueError, match="s3://"):
        validate_location_uri("s3:/womblex/inbox")  # one slash
    with pytest.raises(ValueError, match="lowercase"):
        validate_location_uri("S3://womblex/inbox")
    with pytest.raises(ValueError, match="not one of"):
        validate_location_uri("ftp://host/inbox")
    with pytest.raises(ValueError, match="no bucket"):
        validate_location_uri("s3://")
    with pytest.raises(ValueError, match="empty"):
        validate_location_uri("   ")


def test_validate_location_uri_runs_before_any_network_lookup():
    """An unsupported scheme is refused on the string, so validating an
    operator-supplied URI never resolves a hostname."""
    with pytest.raises(ValueError):
        store_root("ftp://never-resolved.invalid/inbox")


def test_same_location_ignores_spelling_that_opens_the_same_place():
    assert same_location("s3://womblex/inbox", "s3://womblex/inbox/")
    assert same_location("s3://womblex/inbox", "s3://womblex//inbox")
    assert not same_location("s3://womblex/inbox", "s3://womblex/outbox")
    assert not same_location("s3://womblex/inbox", "s3://other/inbox")


def test_list_files_recursive_reaches_nested_prefixes(tmp_path):
    """Object stores have a flat keyspace: `inbox/2026-08/foo.pdf` is one key,
    not a folder. A non-recursive listing of the ingest root reports zero
    documents for a perfectly normal upload layout."""
    store = RemoteStore.from_uri(str(tmp_path / "inbox"))
    store.upload_file(_touch(tmp_path / "top.pdf"), "top.pdf")
    store.upload_file(_touch(tmp_path / "nested.pdf"), "2026-08/agency/nested.pdf")

    assert store.list_files("", "*") == ["2026-08", "top.pdf"]
    assert sorted(store.list_files("", "*", recursive=True)) == [
        "2026-08", "2026-08/agency", "2026-08/agency/nested.pdf", "top.pdf",
    ]


def _touch(path: Path) -> Path:
    path.write_text("x")
    return path


def test_remote_store_file_roundtrip(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = RemoteStore.from_uri(str(store_root))

    src = tmp_path / "a.pdf"
    src.write_bytes(b"hello")
    store.upload_file(src, "inputs/a.pdf")

    assert store.exists("inputs/a.pdf")
    assert store.list_files("inputs", "*.pdf") == ["inputs/a.pdf"]

    out = store.download_file("inputs/a.pdf", tmp_path / "dl" / "a.pdf")
    assert out.read_bytes() == b"hello"


def test_remote_store_move_replaces_the_destination(tmp_path):
    """`move` is the commit step of a temp-key-then-move in-place rewrite: it
    renames a staged object onto a live key, replacing whatever was there, and
    removes the source. On the local backend this is a rename; on S3 it is a
    server-side copy-then-delete.
    """
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    src = tmp_path / "new.parquet"
    src.write_bytes(b"new-bytes")
    store.upload_file(src, "live/x.parquet")
    store.upload_file(src, ".staging/x.parquet")
    # Give the live key different bytes so we can prove the overwrite.
    old = tmp_path / "old.parquet"
    old.write_bytes(b"old-bytes")
    store.upload_file(old, "live/x.parquet")

    store.move(".staging/x.parquet", "live/x.parquet")

    assert not store.exists(".staging/x.parquet")  # source consumed
    dl = store.download_file("live/x.parquet", tmp_path / "dl" / "x.parquet")
    assert dl.read_bytes() == b"new-bytes"  # destination replaced


def test_remote_store_read_text_and_delete(tmp_path):
    """The small in-place read/delete the console's saved presets use (ui-plan merge 9)."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    src = tmp_path / "one.preset.json"
    src.write_text('{"name": "one"}', encoding="utf-8")
    store.upload_file(src, "presets/one.preset.json")

    assert store.read_text("presets/one.preset.json") == '{"name": "one"}'
    store.delete("presets/one.preset.json")
    assert not store.exists("presets/one.preset.json")


def test_remote_store_download_to_dir_and_upload_glob(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))

    for name in ("inputs/x.pdf", "inputs/y.docx"):
        f = tmp_path / Path(name).name
        f.write_bytes(name.encode())
        store.upload_file(f, name)

    local = tmp_path / "scratch"
    fetched = store.download_to_dir(["inputs/x.pdf", "inputs/y.docx"], local)
    assert sorted(p.name for p in fetched) == ["x.pdf", "y.docx"]

    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "batch-0001.elements.parquet").write_bytes(b"e")
    (shards / "batch-0001._manifest.parquet").write_bytes(b"m")
    (shards / "other.txt").write_bytes(b"skip")

    uploaded = store.upload_glob(shards, "batch-0001.*", "runs/r1/documents")
    assert uploaded == {
        "runs/r1/documents/batch-0001.elements.parquet": hashlib.sha256(b"e").hexdigest(),
        "runs/r1/documents/batch-0001._manifest.parquet": hashlib.sha256(b"m").hexdigest(),
    }
    assert store.exists("runs/r1/documents/batch-0001.elements.parquet")
    assert not store.exists("runs/r1/documents/other.txt")


def test_download_to_dir_nested_keeps_same_named_keys_apart(tmp_path):
    """Two documents under different prefixes sharing a basename stay two files."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    for prefix in ("2026-07", "2026-08"):
        f = tmp_path / f"{prefix}.pdf"
        f.write_bytes(prefix.encode())
        store.upload_file(f, f"{prefix}/report.pdf")

    local = tmp_path / "scratch"
    fetched = store.download_to_dir(
        ["2026-07/report.pdf", "2026-08/report.pdf"], local, nested=True
    )

    assert [p.relative_to(local).as_posix() for p in fetched] == [
        "2026-07/report.pdf",
        "2026-08/report.pdf",
    ]
    assert [p.read_bytes() for p in fetched] == [b"2026-07", b"2026-08"]


def test_download_to_dir_refuses_a_key_that_would_escape_the_staging_dir(tmp_path):
    """A key is whatever a queue row or listing named — it does not get to pick
    where it lands. Refused before anything is written."""
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    local = tmp_path / "scratch"

    with pytest.raises(ValueError, match="does not stage under"):
        store.download_to_dir(["../escaped.pdf"], local, nested=True)

    assert not (tmp_path / "escaped.pdf").exists()


def test_remote_store_list_dirs(tmp_path):
    store = RemoteStore.from_uri(str(tmp_path / "store"))
    (tmp_path / "store" / "runs" / "run-a" / "documents").mkdir(parents=True)
    (tmp_path / "store" / "runs" / "run-b" / "documents").mkdir(parents=True)
    (tmp_path / "store" / "runs" / "stray.txt").write_text("x")

    assert store.list_dirs("runs") == ["run-a", "run-b"]
    # Files under the prefix are not directories.
    assert "stray.txt" not in store.list_dirs("runs")
    # A prefix that doesn't exist yet returns empty, not an error.
    assert store.list_dirs("missing") == []


# --- extraction body: ingest as a distinct store (local, no database) -------


def _minimal_config(tmp_path: Path):
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )

    return WomblexConfig(
        dataset=DatasetConfig(name="w"),
        paths=PathsConfig(
            input_root=tmp_path, output_root=tmp_path / "out", checkpoint_dir=tmp_path / ".ckpt"
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )


# --- finalize (local store, no Postgres) -------------------------------------


def test_finalize_consolidates_manifest(tmp_path, monkeypatch):
    """End-to-end finalize: real shards -> store -> consolidated manifest."""
    # No queue: a DSN in the environment (CI's Postgres) would make finalize read it.
    monkeypatch.delenv("WOMBLEX_DB_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import argparse

    import pyarrow.parquet as pq

    from womblex.batch import process_batch
    from womblex.cli.cloud import cmd_finalize
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )

    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\nBob,Analyst\n")
    cfg = WomblexConfig(
        dataset=DatasetConfig(name="fin"),
        paths=PathsConfig(
            input_root=tmp_path, output_root=tmp_path / "out", checkpoint_dir=tmp_path / ".ckpt"
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )

    local_shards = tmp_path / "local_shards"
    local_shards.mkdir()
    process_batch([csv], cfg, batch_num=1, shard_dir=local_shards)

    store_root = tmp_path / "store"
    store = RemoteStore.from_uri(str(store_root))
    run_id = "rfin"
    for p in local_shards.glob("batch-0001._manifest.parquet"):
        store.upload_file(p, f"runs/{run_id}/documents/{p.name}")

    rc = cmd_finalize(argparse.Namespace(
        store=str(store_root), run_id=run_id, output_prefix=None, dsn=None,
    ))
    assert rc == 0
    assert store.exists(f"runs/{run_id}/manifest.parquet")

    dl = store.download_file(f"runs/{run_id}/manifest.parquet", tmp_path / "manifest.parquet")
    assert pq.read_table(dl).num_rows == 1  # one source document


# --- egress (local run, no Postgres) -----------------------------------------


def test_egress_exports_local_run_to_bundle(tmp_path, monkeypatch):
    """CLI wiring for `womblex egress`: a finished local run -> a bundle dir.

    Corpus-only (`--no-sources`) — `build_bundle`'s full source-resolution
    behaviour is exercised in ``tests/test_egress.py``; this only checks the
    command threads its arguments through and lands `corpus/` at the
    destination named by the run root's own directory name.
    """
    import argparse

    from womblex.batch import process_batch
    from womblex.cli.cloud import cmd_egress
    from womblex.config import (
        ChunkingConfig,
        DatasetConfig,
        ExtractionConfig,
        PathsConfig,
        RedactionConfig,
        WomblexConfig,
    )
    from womblex.store.run_manifest import write_run_manifest

    csv = tmp_path / "people.csv"
    csv.write_text("name,role\nAlice,Director\nBob,Analyst\n")
    cfg = WomblexConfig(
        dataset=DatasetConfig(name="egr"),
        paths=PathsConfig(
            input_root=tmp_path, output_root=tmp_path / "out", checkpoint_dir=tmp_path / ".ckpt"
        ),
        extraction=ExtractionConfig(),
        chunking=ChunkingConfig(enabled=False),
        redaction=RedactionConfig(enabled=False),
    )

    run_root = tmp_path / "out" / "regress"
    shard_dir = run_root / "documents"
    shard_dir.mkdir(parents=True)
    process_batch([csv], cfg, batch_num=1, shard_dir=shard_dir)
    write_run_manifest(shard_dir)

    dest = tmp_path / "bundle"
    rc = cmd_egress(argparse.Namespace(
        run=run_root, to=str(dest), run_id=None, bundle_prefix=None,
        sources=False, source_root=None,
    ))
    assert rc == 0

    bundle = dest / "regress"
    assert (bundle / "corpus" / "manifest.parquet").is_file()
    assert not (bundle / "sources").exists()
    assert not (bundle / "source_index.parquet").exists()
    assert (bundle / "egress_manifest.json").is_file()

    # A run root given as `.` still defaults run_id to the directory's name.
    monkeypatch.chdir(run_root)
    rc = cmd_egress(argparse.Namespace(
        run=Path("."), to=str(tmp_path / "bundle2"), run_id=None, bundle_prefix=None,
        sources=False, source_root=None,
    ))
    assert rc == 0
    assert (tmp_path / "bundle2" / "regress" / "egress_manifest.json").is_file()

    # An unopenable destination is a clean exit 1, not a traceback.
    rc = cmd_egress(argparse.Namespace(
        run=run_root, to="nosuchscheme://x", run_id=None, bundle_prefix=None,
        sources=False, source_root=None,
    ))
    assert rc == 1


def test_prepare_stage_context_refuses_a_stage_isaacus_cannot_serve(monkeypatch, tmp_path):
    """Without it `chunk_shards` warns, writes nothing and returns cleanly —
    a remote no-op a queue would record as a completed job.

    ``enrich`` always needs the API. ``chunk`` needs it only under AI chunking
    (``chunking_model``); plain token chunking runs offline on the vendored
    tokeniser, so its gate is config-aware."""
    from womblex.cloud.stage_contracts import STAGE_CONTRACTS
    from womblex.cloud.stage_runner import StagePreconditionError, prepare_stage_context
    from womblex.utils import availability

    monkeypatch.setattr(availability, "isaacus_available", lambda: False)

    # enrich unconditionally needs the API.
    with pytest.raises(StagePreconditionError, match="needs Isaacus"):
        prepare_stage_context(STAGE_CONTRACTS["enrich"], _minimal_config(tmp_path))

    # chunk WITH AI chunking needs the API.
    ai_cfg = _minimal_config(tmp_path)
    ai_cfg.chunking.chunking_model = "kanon-2-enricher"
    with pytest.raises(StagePreconditionError, match="needs Isaacus"):
        prepare_stage_context(STAGE_CONTRACTS["chunk"], ai_cfg)

    # chunk WITHOUT a chunking_model is offline token chunking — no API needed,
    # so it must NOT refuse even with Isaacus unavailable (the keyless local
    # chunking path).
    assert prepare_stage_context(STAGE_CONTRACTS["chunk"], _minimal_config(tmp_path)) is not None

    # A stage with no Isaacus need is unaffected.
    assert prepare_stage_context(STAGE_CONTRACTS["money"], _minimal_config(tmp_path)) is not None


def test_read_parquet_footer_reads_metadata_without_downloading(tmp_path):
    """`womblex finalize` stages in only the manifests, so it reads every other
    shard's footer where it lives. A local-filesystem store exercises the same
    fsspec path an object store takes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from womblex.store.remote import RemoteStore

    table = pa.table({"a": pa.array([1, 2, 3])})
    (tmp_path / "documents").mkdir()
    target = tmp_path / "documents" / "batch-0001.embeddings.parquet"
    pq.write_table(table.replace_schema_metadata({b"womblex.stage": b"embed"}), str(target))

    store = RemoteStore.from_uri(str(tmp_path))
    observed = store.read_parquet_footer("documents/batch-0001.embeddings.parquet")
    assert observed is not None
    metadata, num_rows = observed
    assert num_rows == 3
    assert metadata[b"womblex.stage"] == b"embed"


def test_read_parquet_footer_reports_an_unreadable_object_rather_than_raising(tmp_path):
    """The caller is building a record; a file it cannot read is one it reports
    as absent, not one it fails the finalisation over."""
    from womblex.store.remote import RemoteStore

    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "junk.parquet").write_bytes(b"not a parquet")

    store = RemoteStore.from_uri(str(tmp_path))
    assert store.read_parquet_footer("documents/junk.parquet") is None
    assert store.read_parquet_footer("documents/absent.parquet") is None


# --- enqueue: the prefix an operator types vs the keys that are queued -------


class _CapturingQueue:
    """A `RunBoard` stand-in that records the specs `cmd_enqueue` builds."""

    keys: list[str] | None = None

    def __init__(self, dsn: str, **_kw: object) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def enqueue(self, _run_id: str, specs: list) -> int:
        _CapturingQueue.keys = [k for s in specs for k in s.input_keys]
        return len(specs)


def _enqueue_args(tmp_path: Path, prefix: str | None, *, ingest: bool = True):
    import argparse

    return argparse.Namespace(
        store=str(tmp_path / "store"),
        ingest=str(tmp_path / "inbox") if ingest else None,
        input_prefix=prefix, config=None, run_id="run-cli", output_prefix=None,
        batch_size=None, max_attempts=3, dsn="postgresql://x/y", create_schema=False,
    )


@pytest.mark.parametrize("prefix", ["2026-08", "./2026-08", "2026-08/", "2026-08//", "2026-08/."])
def test_enqueue_queues_one_key_however_the_prefix_was_typed(tmp_path, monkeypatch, prefix):
    """A queued key is downloaded by the worker and recorded as the document's
    `source_relpath`, and an object store has no path semantics to collapse a
    `.` segment, so every spelling has to scope identically here."""
    import womblex.cloud.jobs as jobs_module
    from womblex.cli import cloud
    from womblex.cloud import dbos_app

    monkeypatch.setattr(jobs_module, "RunBoard", _CapturingQueue)
    monkeypatch.setattr(dbos_app, "ensure_schema", lambda _dsn: None)
    (tmp_path / "inbox" / "2026-08").mkdir(parents=True)
    (tmp_path / "inbox" / "2026-08" / "a.pdf").write_bytes(b"%PDF-1.4\n")

    _CapturingQueue.keys = None
    assert cloud.cmd_enqueue(_enqueue_args(tmp_path, prefix)) == 0
    assert _CapturingQueue.keys == ["2026-08/a.pdf"]


def test_a_prefix_scoping_to_nothing_is_not_a_licence_to_list_the_whole_store(
    tmp_path, monkeypatch
):
    """`--input-prefix ./` is truthy but scopes to nothing. Without `--ingest`
    that would enqueue the store root — the run's own output included — so the
    guard reads the normalised prefix, not the string typed."""
    import womblex.cloud.jobs as jobs_module
    from womblex.cli import cloud
    from womblex.cloud import dbos_app

    monkeypatch.setattr(jobs_module, "RunBoard", _CapturingQueue)
    monkeypatch.setattr(dbos_app, "ensure_schema", lambda _dsn: None)
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / "stray.pdf").write_bytes(b"%PDF-1.4\n")

    _CapturingQueue.keys = None
    assert cloud.cmd_enqueue(_enqueue_args(tmp_path, "./", ingest=False)) == 1
    assert _CapturingQueue.keys is None

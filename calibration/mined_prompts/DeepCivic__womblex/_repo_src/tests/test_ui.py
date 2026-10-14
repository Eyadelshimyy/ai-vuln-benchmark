"""Tests for the console read API skeleton (womblex.ui).

Runs are plain directories with real, schema-correct manifest shards (no
fixture PDFs needed). The ``api_client`` fixture parametrizes every API test
over local (``output_root``) and store-backed (``RemoteStore``'s local
fsspec backend, as test_cloud.py uses — no S3/MinIO needed) settings, so
each test exercises both read paths through one body.
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from womblex.cli import ALL_COMMANDS
from womblex.store.enrichment_output import (
    ENRICHMENT_ENTITIES_SUFFIX,
    ENTITY_SCHEMA,
    LEGACY_HASH_COLUMN,
)
from womblex.store.money_output import MONEY_SPANS_SCHEMA, MONEY_SPANS_SUFFIX
from womblex.store.output import CHUNKS_SCHEMA, CHUNKS_SUFFIX, ELEMENTS_SUFFIX, MANIFEST_SCHEMA
from womblex.store.pii_output import PII_SPANS_SCHEMA, PII_SPANS_SUFFIX
from womblex.store.quality_output import CHUNK_QUALITY_SCHEMA, CHUNK_QUALITY_SUFFIX
from womblex.ui import dashboard, readers
from womblex.ui.app import create_app
from womblex.ui.deps import UISettings, resolve_settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def _compose() -> dict:
    return yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())


def _compose_service(name: str) -> dict:
    return _compose()["services"][name]

_OK_ROW = {
    "source_hash": "hash-a", "collection_id": "test", "doc_id": "doc-a",
    "filename": "doc-a.pdf", "ext": ".pdf", "extraction_method": "native",
    "elements_count": 3, "table_cells_count": 0, "form_fields_count": 0,
    "status": "ok", "error": "", "extracted_at_iso": "2026-08-16T00:00:00Z",
    "parser_version": "2.0",
}
# A failed document, which the Corpus Inspector's failure filter needs to see:
# the manifest carries it as a row, not as an omission.
_FAILED_ROW = {
    **_OK_ROW,
    "source_hash": "hash-b", "doc_id": "doc-b", "filename": "doc-b.pdf",
    "status": "error", "error": "boom",
}
_ROWS = [_OK_ROW, _FAILED_ROW]


def _write_manifest_shard(shard_dir: Path, rows: list[dict]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA),
        str(shard_dir / "batch-0001._manifest.parquet"),
    )
    (shard_dir / f"batch-0001{ELEMENTS_SUFFIX}").write_bytes(b"stub")


def _write_consolidated_manifest(run_dir: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), str(run_dir / "manifest.parquet"))


class TestUISettings:
    def test_needs_exactly_one_source(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            UISettings(output_root=None, store_uri=None)
        with pytest.raises(ValueError, match="exactly one"):
            UISettings(output_root=tmp_path, store_uri="s3://bucket")

    def test_is_remote(self, tmp_path: Path) -> None:
        assert UISettings(output_root=tmp_path, store_uri=None).is_remote is False
        assert UISettings(output_root=None, store_uri="s3://bucket").is_remote is True

    def test_ingest_and_store_must_be_disjoint(self) -> None:
        UISettings(output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex/inbox")
        with pytest.raises(ValueError, match="not disjoint"):
            UISettings(output_root=None, store_uri="s3://womblex", ingest_uri="s3://womblex")


class TestResolveSettings:
    def test_explicit_args_win(self, tmp_path: Path) -> None:
        assert resolve_settings(tmp_path, None).output_root == tmp_path
        assert resolve_settings(None, "s3://bucket").store_uri == "s3://bucket"

    def test_env_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WOMBLEX_UI_OUTPUT_ROOT", str(tmp_path))
        assert resolve_settings(None, None).output_root == tmp_path

    def test_neither_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WOMBLEX_UI_OUTPUT_ROOT", raising=False)
        monkeypatch.delenv("WOMBLEX_STORE_URI", raising=False)
        with pytest.raises(ValueError, match="no run source"):
            resolve_settings(None, None)

    def test_both_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="only one"):
            resolve_settings(tmp_path, "s3://bucket")

    def test_presets_dir_explicit_arg_wins(self, tmp_path: Path) -> None:
        presets_dir = tmp_path / "presets"
        assert resolve_settings(tmp_path, None, presets_dir=presets_dir).presets_dir == presets_dir

    def test_presets_dir_env_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        presets_dir = tmp_path / "presets"
        monkeypatch.setenv("WOMBLEX_UI_PRESETS_DIR", str(presets_dir))
        assert resolve_settings(tmp_path, None).presets_dir == presets_dir

    def test_presets_dir_defaults_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WOMBLEX_UI_PRESETS_DIR", raising=False)
        assert resolve_settings(tmp_path, None).presets_dir is None

    def test_presets_writable_needs_a_dir_locally_but_not_remotely(self, tmp_path: Path) -> None:
        """Local mode needs a writable presets dir; remote always writes to the store."""
        assert resolve_settings(tmp_path, None).presets_writable is False
        assert resolve_settings(tmp_path, None, presets_dir=tmp_path / "p").presets_writable is True
        # Remote mode saves to the store's own presets/ prefix, so it needs no dir.
        assert resolve_settings(None, "s3://bucket").presets_writable is True

    def test_ingest_uri_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WOMBLEX_INGEST_URI", "s3://womblex/inbox")
        assert resolve_settings(None, "s3://womblex").ingest_uri == "s3://womblex/inbox"


@pytest.fixture(params=["local", "remote"])
def api_client(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[TestClient, Path]:
    """(client, run_root) — write ``<run_root>/<run_id>/documents/`` to seed a run."""
    if request.param == "remote":
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        return TestClient(create_app(store_uri=str(store_root))), store_root / "runs"
    return TestClient(create_app(output_root=tmp_path)), tmp_path


class TestRunsApi:
    def test_health_and_empty_list(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/runs").json() == {"runs": []}

    def test_lists_runs_newest_first(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-20260101T000000Z" / "documents", _ROWS[:1])
        _write_manifest_shard(run_root / "run-20260201T000000Z" / "documents", _ROWS)
        runs = client.get("/api/runs").json()["runs"]
        assert [r["run_id"] for r in runs] == ["run-20260201T000000Z", "run-20260101T000000Z"]
        assert runs[0]["document_count"] == 2
        assert runs[0]["stages"] == ["extract"]

    def test_get_manifest_from_shards(self, api_client: tuple[TestClient, Path]) -> None:
        """Also covers a distributed run before `finalize` has published a consolidated manifest."""
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        body = client.get("/api/runs/run-a/manifest").json()
        assert body["run_id"] == "run-a"
        error_row = next(d for d in body["documents"] if d["doc_id"] == "doc-b")
        assert error_row["status"] == "error"
        assert error_row["error"] == "boom"

    def test_get_manifest_prefers_consolidated(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        run_dir = run_root / "run-a"
        _write_manifest_shard(run_dir / "documents", _ROWS[:1])
        _write_consolidated_manifest(run_dir, _ROWS)
        resp = client.get("/api/runs/run-a/manifest")
        assert len(resp.json()["documents"]) == 2

    def test_get_manifest_404(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/manifest").status_code == 404

    def test_stage_presence(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        chunks_path = shard_dir / "batch-0001.chunks.parquet"
        pq.write_table(pa.table({"source_hash": ["hash-a"]}), str(chunks_path))
        body = client.get("/api/runs/run-a/stage-presence/chunk").json()
        assert body == {"run_id": "run-a", "stage": "chunk", "source_hashes": ["hash-a"]}
        # A stage with no sidecar in this run reports present, not missing.
        empty = client.get("/api/runs/run-a/stage-presence/embed").json()
        assert empty["source_hashes"] == []

    def test_stage_presence_reads_the_enrich_sidecar_hash_column(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """The enrich sidecar joins on `source_hash` like every other one."""
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        _write_shard(shard_dir, ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA, [_ENTITY_ROW])
        body = client.get("/api/runs/run-a/stage-presence/enrich").json()
        assert body == {"run_id": "run-a", "stage": "enrich", "source_hashes": ["hash-a"]}

    def test_stage_presence_reads_a_pre_rename_enrich_sidecar(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """A sidecar written before 0.6.0 carries the hash in the legacy column.

        The console reads parquet directly so the predicate pushes down, which
        bypasses the store reader's shim — without the same fallback here,
        `enrich` reports empty on a completed pre-rename run.
        """
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        _write_shard(
            shard_dir, ENRICHMENT_ENTITIES_SUFFIX, _legacy_entity_schema(),
            [_legacy_entity_row(_ENTITY_ROW)],
        )
        body = client.get("/api/runs/run-a/stage-presence/enrich").json()
        assert body == {"run_id": "run-a", "stage": "enrich", "source_hashes": ["hash-a"]}

    def test_stage_presence_skips_an_unreadable_sidecar(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """One corrupt batch must not blank the other batches' answer."""
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        pq.write_table(
            pa.table({"source_hash": ["hash-a"]}), str(shard_dir / "batch-0001.chunks.parquet")
        )
        (shard_dir / "batch-0002.chunks.parquet").write_bytes(b"not parquet")
        body = client.get("/api/runs/run-a/stage-presence/chunk").json()
        assert body["source_hashes"] == ["hash-a"]

    def test_stage_presence_unknown_stage(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        assert client.get("/api/runs/run-a/stage-presence/nope").status_code == 400

    def test_stage_presence_404(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/stage-presence/chunk").status_code == 404

    def test_audit(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        report = client.get("/api/runs/run-a/audit").json()
        assert report["manifest_row_count"] == 0  # the stub elements file makes the batch unreadable
        assert len(report["corrupted_batches"]) == 1

    def test_audit_404(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/audit").status_code == 404


def _write_log(run_root: Path, run_id: str, name: str, body: str) -> None:
    logs_dir = run_root / run_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / name).write_text(body, encoding="utf-8")


class TestRunLogsApi:
    """Batch logs readable and downloadable from the console."""

    def test_lists_logs_newest_first(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "first")
        _write_log(run_root, "run-a", "batch-0002.log", "second")
        body = client.get("/api/runs/run-a/logs").json()
        assert body["run_id"] == "run-a"
        assert [entry["name"] for entry in body["logs"]] == [
            "batch-0002.log", "batch-0001.log",
        ]

    def test_list_404_when_run_missing(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/logs").status_code == 404

    def test_run_with_no_logs_lists_empty_not_404(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """A run created before this change exists but has no `logs/` prefix; it
        lists empty (an explained empty state) rather than 404-ing."""
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-old" / "documents", _ROWS)
        resp = client.get("/api/runs/run-old/logs")
        assert resp.status_code == 200
        assert resp.json()["logs"] == []

    def test_reads_one_log_as_plain_text(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "Extraction failed: doc=bad error=boom")
        resp = client.get("/api/runs/run-a/logs/batch-0001.log")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "Extraction failed: doc=bad" in resp.text
        assert "content-disposition" not in resp.headers

    def test_download_sets_content_disposition(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "log body")
        resp = client.get("/api/runs/run-a/logs/batch-0001.log?download=1")
        assert resp.status_code == 200
        assert 'attachment; filename="batch-0001.log"' in resp.headers["content-disposition"]

    def test_absent_log_404s_with_the_available_list(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "present")
        resp = client.get("/api/runs/run-a/logs/batch-9999.log")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert [entry["name"] for entry in detail["available"]] == ["batch-0001.log"]

    @pytest.mark.parametrize(
        "bad_name",
        ["batch-1.log", "batch-0001.txt", "passwd", "batch-0001.log.bak"],
    )
    def test_a_malformed_name_404s_the_same_way_never_touching_the_store(
        self, api_client: tuple[TestClient, Path], bad_name: str
    ) -> None:
        """A rejected name and an absent name return the identical 404 + available
        list, so the endpoint cannot be used to probe outside `logs/`."""
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        _write_log(run_root, "run-a", "batch-0001.log", "present")
        resp = client.get(f"/api/runs/run-a/logs/{bad_name}")
        assert resp.status_code == 404
        assert [e["name"] for e in resp.json()["detail"]["available"]] == ["batch-0001.log"]

    def test_a_traversal_never_reaches_the_filesystem(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS)
        # Encoded so the router passes it through as one path segment rather
        # than resolving `..` itself.
        resp = client.get("/api/runs/run-a/logs/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code == 404

    def test_read_404_when_run_missing(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/logs/batch-0001.log").status_code == 404

    def test_is_safe_log_name(self) -> None:
        assert readers.is_safe_log_name("batch-0001.log")
        assert not readers.is_safe_log_name("batch-1.log")
        assert not readers.is_safe_log_name("../batch-0001.log")
        assert not readers.is_safe_log_name("batch-0001.log.bak")


_CHUNK_ROW = {
    "source_hash": "hash-a", "chunk_index": 1, "text": "second chunk",
    "start_char": 20, "end_char": 32, "content_type": "narrative",
    "has_redaction": False, "page_start": 1, "page_end": 1, "elem_order": 3,
    "token_count": 2,
}
_CHUNK_ROW_0 = {**_CHUNK_ROW, "chunk_index": 0, "text": "first chunk", "start_char": 0, "end_char": 19}
_OTHER_DOC_CHUNK = {**_CHUNK_ROW, "source_hash": "hash-b"}

_ENTITY_ROW = {
    "source_hash": "hash-a", "entity_id": "PR-1", "entity_label": "person",
    "name": "Jane Citizen", "entity_type": "natural", "role": "other",
    "mention_start": 0, "mention_end": 12, "chunk_index": 0,
}

_PII_ROW = {
    "source_hash": "hash-a", "chunk_index": 0, "content_type": "narrative",
    "start": 0, "end": 12, "text": "Jane Citizen", "entity_type": "PERSON",
    "entity_id": "PR-1", "detector": "enrichment", "score": 0.9, "replacement": "<PERSON_1>",
}

_MONEY_ROW = {
    "source_hash": "hash-a", "locus": "narrative", "text_source": "elements",
    "start_char": 5, "end_char": 10, "page": 1, "elem_order": 3, "parent_elem_order": None,
    "sheet": None, "row": None, "col": None, "text": "$100", "value": Decimal("100.0000"),
    "currency": "AUD", "currency_source": "symbol", "evidence": "p1", "modifier": None,
    "multiplier": None, "negative": False, "confidence": 0.9, "range_group": None,
    "range_role": None, "column_id": None, "context": "",
}
_MONEY_ROW_TABLE_CELL = {**_MONEY_ROW, "locus": "table_cell", "start_char": None, "end_char": None}

_QUALITY_ROW = {
    "source_hash": "hash-a", "chunk_index": 0, "content_type": "narrative",
    "char_len": 19, "alpha_frac": 0.9, "is_short": False, "boilerplate_flag": False,
    "exact_dup_id": None, "near_dup_id": None,
}


def _legacy_entity_schema() -> pa.Schema:
    """ENTITY_SCHEMA as it was before 0.6.0 — hash under the legacy column."""
    return pa.schema([
        pa.field(LEGACY_HASH_COLUMN, f.type) if f.name == "source_hash" else f
        for f in ENTITY_SCHEMA
    ])


def _legacy_entity_row(row: dict) -> dict:
    return {LEGACY_HASH_COLUMN if k == "source_hash" else k: v for k, v in row.items()}


def _write_shard(
    shard_dir: Path, suffix: str, schema: pa.Schema, rows: list[dict], *, batch: str = "batch-0001",
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(shard_dir / f"{batch}{suffix}"))


class TestChunkDetailApi:
    def test_404_when_run_missing(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/runs/nope/chunks/hash-a").status_code == 404

    def test_empty_when_chunk_stage_not_run(self, api_client: tuple[TestClient, Path]) -> None:
        client, run_root = api_client
        _write_manifest_shard(run_root / "run-a" / "documents", _ROWS[:1])
        body = client.get("/api/runs/run-a/chunks/hash-a").json()
        assert body == {
            "run_id": "run-a", "source_hash": "hash-a",
            "chunks": [], "entities": [], "pii_spans": [], "money_spans": [], "quality": [],
        }

    def test_returns_chunks_ordered_and_scoped_overlays(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS)
        _write_shard(shard_dir, CHUNKS_SUFFIX, CHUNKS_SCHEMA, [_CHUNK_ROW, _CHUNK_ROW_0, _OTHER_DOC_CHUNK])
        _write_shard(shard_dir, ENRICHMENT_ENTITIES_SUFFIX, ENTITY_SCHEMA, [_ENTITY_ROW])
        _write_shard(shard_dir, PII_SPANS_SUFFIX, PII_SPANS_SCHEMA, [_PII_ROW])
        _write_shard(
            shard_dir, MONEY_SPANS_SUFFIX, MONEY_SPANS_SCHEMA, [_MONEY_ROW, _MONEY_ROW_TABLE_CELL],
        )
        _write_shard(shard_dir, CHUNK_QUALITY_SUFFIX, CHUNK_QUALITY_SCHEMA, [_QUALITY_ROW])

        body = client.get("/api/runs/run-a/chunks/hash-a").json()
        assert [c["chunk_index"] for c in body["chunks"]] == [0, 1]
        assert all(c["source_hash"] == "hash-a" for c in body["chunks"])

        assert len(body["entities"]) == 1
        assert body["entities"][0]["source_hash"] == "hash-a"
        assert LEGACY_HASH_COLUMN not in body["entities"][0]

        assert [p["entity_type"] for p in body["pii_spans"]] == ["PERSON"]

        # Only the narrative-locus span is returned — table_cell spans anchor
        # to table_cells.parquet, not chunk text, so they don't overlay here.
        assert len(body["money_spans"]) == 1
        assert body["money_spans"][0]["locus"] == "narrative"
        assert body["money_spans"][0]["value"] == "100.0000"

        assert len(body["quality"]) == 1
        assert body["quality"][0]["char_len"] == 19

    def test_drifted_shard_rows_keep_the_canonical_key_set(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        """A run spanning a schema bump must not hand the frontend ragged rows.

        `elem_order` post-dates parser 2.0 (`store.output._CHUNKS_BACKFILL`),
        so a long-lived run can hold batches on either side of it.
        """
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS[:1])
        old_schema = pa.schema([f for f in CHUNKS_SCHEMA if f.name != "elem_order"])
        pre_bump = {k: v for k, v in _CHUNK_ROW_0.items() if k != "elem_order"}
        pq.write_table(
            pa.Table.from_pylist([pre_bump], schema=old_schema),
            str(shard_dir / f"batch-0001{CHUNKS_SUFFIX}"),
        )
        _write_shard(shard_dir, CHUNKS_SUFFIX, CHUNKS_SCHEMA, [_CHUNK_ROW], batch="batch-0002")

        chunks = client.get("/api/runs/run-a/chunks/hash-a").json()["chunks"]
        assert len(chunks) == 2
        assert {frozenset(c) for c in chunks} == {frozenset(CHUNKS_SCHEMA.names)}
        assert chunks[0]["elem_order"] is None  # back-filled, not absent
        assert chunks[1]["elem_order"] == 3

    def test_a_corrupt_sidecar_narrows_the_answer_rather_than_blanking_it(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        shard_dir = run_root / "run-a" / "documents"
        _write_manifest_shard(shard_dir, _ROWS[:1])
        _write_shard(shard_dir, CHUNKS_SUFFIX, CHUNKS_SCHEMA, [_CHUNK_ROW_0])
        (shard_dir / f"batch-0002{CHUNKS_SUFFIX}").write_bytes(b"not parquet")
        body = client.get("/api/runs/run-a/chunks/hash-a").json()
        assert [c["chunk_index"] for c in body["chunks"]] == [0]


def _write_checkpoint(run_dir: Path, stage: str, **counters: int) -> None:
    """Write a stage checkpoint where the pipeline itself writes one.

    Goes through ``CheckpointManager`` rather than hand-rolling the JSON so
    the dashboard is tested against the real writer's filename and shape.
    """
    from womblex.store.checkpoint import CheckpointManager

    dirname = dashboard.CHECKPOINT_DIRNAMES[stage]
    mgr = CheckpointManager(run_dir / dirname, f"testset_{stage}")
    mgr.load()
    mgr.update(
        [f"doc-{i}" for i in range(counters.get("processed", 1))],
        succeeded=counters.get("succeeded", 1),
        failed=counters.get("failed", 0),
        batch_num=counters.get("batch_num", 1),
    )


def _missing_jobs_table_error() -> Exception:
    """The error a database raises when the DBOS system tables do not exist.

    Postgres names the ``workflow_status`` relation; the message is what
    ``dashboard._is_missing_jobs_table`` recognises, since the driver error
    arrives wrapped by SQLAlchemy.
    """
    return Exception('relation "dbos.workflow_status" does not exist')


class _MissingTableQueue:
    """A reachable ``RunBoard`` whose DBOS system tables have not been created.

    Connecting succeeds (a fresh Postgres before ``init``/enqueue); the first
    read raises ``UndefinedTable``. The dashboard must read this as an empty
    queue, not a fault, and must never create the table.
    """

    schema_ensured_count = 0

    def __init__(self, dsn: str, **_kw: object) -> None:
        self.dsn = dsn

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def stats(self, _run_id: str | None = None) -> dict:
        raise _missing_jobs_table_error()

    def list_jobs(self, *a: object, **k: object) -> list:
        raise _missing_jobs_table_error()

    def workers(self, *a: object, **k: object) -> list:
        raise _missing_jobs_table_error()

    def stale_jobs(self, *a: object, **k: object) -> list:
        raise _missing_jobs_table_error()

    def throughput(self, *a: object, **k: object) -> object:
        raise _missing_jobs_table_error()


class _BrokenQueue(_MissingTableQueue):
    """A reachable queue whose reads fail for a reason that is *not* a missing table.

    Proves the missing-table swallow is narrow: a real failure (here a
    permissions error) still surfaces as ``queue_error``, not a silent empty.
    """

    def stats(self, _run_id: str | None = None) -> dict:
        # A deliberately generic exception: the point is an *arbitrary* failure
        # that is not a missing table, so a custom type would defeat the test.
        raise Exception("permission denied for table womblex_jobs")  # noqa: TRY002


class TestDashboardApi:
    """The Dashboard reads the queue when there is one and the run's own
    per-stage checkpoints always (docs/ui-plan.md merge 8). These exercise
    the queue-less shape — a plain local deployment — and the checkpoint
    half, which is the same in both.
    """

    def test_checkpoint_dirnames_come_from_the_contracts(self) -> None:
        """The map is derived, not re-typed: a renamed dir cannot drift."""
        from womblex.cloud.stage_contracts import STAGE_CONTRACTS

        assert dashboard.CHECKPOINT_DIRNAMES["chunk"] == ".chunk-checkpoint"
        assert "quality" not in dashboard.CHECKPOINT_DIRNAMES  # whole-run scope, no checkpoint
        for stage, dirname in dashboard.CHECKPOINT_DIRNAMES.items():
            assert STAGE_CONTRACTS[stage].checkpoint_dirname == dirname

    def test_no_queue_is_not_an_error(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        body = client.get("/api/dashboard").json()
        assert body["queue"] is None
        assert body["queue_error"] is None  # absent by configuration, not by failure
        assert body["stages"] == []

    def test_unreachable_queue_reports_rather_than_raises(self, tmp_path: Path) -> None:
        """A dead DSN must not take the checkpoint half of the screen down with it."""
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://nope:1/nope"))
        _write_checkpoint(tmp_path / "run-a", "chunk", processed=2)
        body = client.get("/api/dashboard", params={"run_id": "run-a"}).json()
        assert body["queue"] is None
        assert body["queue_error"]
        assert [s["stage"] for s in body["stages"]] == ["chunk"]

    def test_a_reachable_queue_with_no_jobs_table_reads_as_empty_not_a_fault(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh Postgres before `init`/first-enqueue has no `womblex_jobs`.

        Reading that as "queue unreachable" was a reported symptom: the console
        gates its whole dashboard/execution surface on the queue, so a fresh
        cluster read as a fault (and the enqueue form vanished) until the table
        happened to appear. A reachable-but-schemaless queue is an *empty* queue
        — `queue_error` stays None and the tiles render as zero. The console must
        not create the table (it is read-only; `init`/enqueue own creation).
        """
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _MissingTableQueue)
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        body = client.get("/api/dashboard").json()
        assert body["queue_error"] is None
        assert body["queue"] == {
            "stats": {}, "total": 0, "jobs": [], "workers": [], "stale": [],
            "throughput": {
                "window_seconds": dashboard.DEFAULT_THROUGHPUT_WINDOW,
                "completed": 0, "per_minute": 0.0, "last_completed_at": None,
            },
        }

    def test_a_missing_table_never_creates_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The read path must not call ensure_schema — the console never mutates."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _MissingTableQueue)
        monkeypatch.setattr(
            "womblex.cloud.dbos_app.ensure_schema",
            lambda _dsn: setattr(
                _MissingTableQueue, "schema_ensured_count",
                _MissingTableQueue.schema_ensured_count + 1,
            ),
        )
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        client.get("/api/dashboard")
        assert _MissingTableQueue.schema_ensured_count == 0

    def test_a_real_query_error_still_reports_as_unreachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a missing table is swallowed; a genuine failure is not hidden."""
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _BrokenQueue)
        client = TestClient(create_app(output_root=tmp_path, db_dsn="postgresql://x/y"))
        body = client.get("/api/dashboard").json()
        assert body["queue"] is None
        assert "permission denied" in body["queue_error"]

    def test_stage_progress_from_checkpoints(
        self, api_client: tuple[TestClient, Path]
    ) -> None:
        client, run_root = api_client
        run_dir = run_root / "run-a"
        _write_manifest_shard(run_dir / "documents", _ROWS)
        _write_checkpoint(run_dir, "chunk", processed=4, succeeded=3, failed=1, batch_num=2)
        _write_checkpoint(run_dir, "pii", processed=1)
        body = client.get("/api/dashboard", params={"run_id": "run-a"}).json()
        by_stage = {s["stage"]: s for s in body["stages"]}
        assert set(by_stage) == {"chunk", "pii"}
        assert by_stage["chunk"]["processed"] == 4
        assert by_stage["chunk"]["succeeded"] == 3
        assert by_stage["chunk"]["failed"] == 1
        assert by_stage["chunk"]["last_batch"] == 2
        assert by_stage["chunk"]["name"] == "testset_chunk"

    def test_stages_empty_without_a_run_id(self, api_client: tuple[TestClient, Path]) -> None:
        """Checkpoints are per-run artefacts, so there is no cross-run answer."""
        client, run_root = api_client
        _write_checkpoint(run_root / "run-a", "chunk", processed=2)
        assert client.get("/api/dashboard").json()["stages"] == []

    def test_missing_run_is_empty_not_404(self, api_client: tuple[TestClient, Path]) -> None:
        """An enqueued run whose first batch hasn't landed has no directory yet."""
        client, _ = api_client
        resp = client.get("/api/dashboard", params={"run_id": "run-nope"})
        assert resp.status_code == 200
        assert resp.json()["stages"] == []

    def test_rejects_a_non_positive_window(self, api_client: tuple[TestClient, Path]) -> None:
        client, _ = api_client
        assert client.get("/api/dashboard", params={"stale_after": 0}).status_code == 422

    def test_unreadable_checkpoint_is_skipped(self, tmp_path: Path) -> None:
        """One corrupt checkpoint narrows the answer; it doesn't blank the screen."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "bad_checkpoint.json").write_text("{not json")
        (ckpt_dir / "good_checkpoint.json").write_text(
            json.dumps({"total_processed": 3, "started_at": "", "updated_at": ""})
        )
        progress = read_checkpoints(ckpt_dir)
        assert [p.name for p in progress] == ["good"]
        # Unusable timestamps mean no rate to report, rather than a fabricated one.
        assert progress[0].documents_per_minute is None

    def test_documents_per_minute_spans_the_checkpoint(self, tmp_path: Path) -> None:
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_checkpoint.json").write_text(
            json.dumps({
                "total_processed": 120,
                "started_at": "2026-08-16T00:00:00+00:00",
                "updated_at": "2026-08-16T00:02:00+00:00",
            })
        )
        assert read_checkpoints(ckpt_dir)[0].documents_per_minute == 60.0

    def test_run_id_cannot_escape_the_run_root(self, tmp_path: Path) -> None:
        """`run_id` is a query param, so nothing upstream rejects `../`.

        The remote branch is contained by its `list_dirs` check; the local
        join has to contain itself, or the two deployments disagree about
        what is reachable.
        """
        outside = tmp_path / "outside"
        _write_checkpoint(outside, "chunk", processed=9)
        run_root = tmp_path / "runs"
        run_root.mkdir()  # must exist, or the OS cannot resolve `..` through it
        client = TestClient(create_app(output_root=run_root))
        body = client.get("/api/dashboard", params={"run_id": "../outside"}).json()
        assert body["stages"] == []

    def test_malformed_checkpoints_are_skipped_not_raised(self, tmp_path: Path) -> None:
        """Well-formed JSON can still be unusable; the reader must survive it."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        # A null timestamp reaches `datetime.fromisoformat(None)`; a top-level
        # list reaches `.get` on a list. Both parse as JSON, neither is a
        # checkpoint, and neither may escape as a 500.
        (ckpt_dir / "null_checkpoint.json").write_text(
            json.dumps({"total_processed": 1, "started_at": None, "updated_at": None})
        )
        (ckpt_dir / "list_checkpoint.json").write_text("[1, 2, 3]")
        (ckpt_dir / "ok_checkpoint.json").write_text(
            json.dumps({"total_processed": 1, "started_at": "", "updated_at": ""})
        )
        progress = {p.name: p for p in read_checkpoints(ckpt_dir)}
        # The list is not a checkpoint at all, so it is dropped. The null
        # timestamps are: its counters are real progress and are kept — only
        # the rate they cannot support is withheld.
        assert set(progress) == {"null", "ok"}
        assert progress["null"].processed == 1
        assert progress["null"].documents_per_minute is None

    def test_no_rate_from_a_single_write(self, tmp_path: Path) -> None:
        """A just-started run's two timestamps are one write, not an interval."""
        from womblex.store.checkpoint import read_checkpoints

        ckpt_dir = tmp_path / ".chunk-checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_checkpoint.json").write_text(
            json.dumps({
                "total_processed": 3,
                "started_at": "2026-08-16T00:00:00.000100+00:00",
                "updated_at": "2026-08-16T00:00:00.000200+00:00",
            })
        )
        # Dividing by a 0.1ms span would report ~1.8M documents/minute.
        assert read_checkpoints(ckpt_dir)[0].documents_per_minute is None


_MINIMAL_CONFIG = {"dataset": {"name": "t"}}
_NO_DATASET: dict = {}


class TestComposerApi:
    """The Pipeline Composer reads no run — `config.py` and
    `cloud/stage_contracts.py` are static (docs/ui-plan.md merge 9), so
    `api_client`'s local/remote parametrisation is irrelevant here; a bare
    `TestClient` is enough.
    """

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        return TestClient(create_app(output_root=tmp_path))

    def test_graph_nodes_cover_extract_plus_every_stage(self, client: TestClient) -> None:
        from womblex.cloud.stage_contracts import STAGE_NAMES

        body = client.get("/api/composer/graph").json()
        assert [n["id"] for n in body["nodes"]] == ["extract", *STAGE_NAMES]

    def test_graph_edges_are_one_per_pair_and_resolve_their_producers(
        self, client: TestClient
    ) -> None:
        """chunk reading three extraction sidecars is one dependency, not three —
        otherwise every renderer dedupes to avoid parallel arrows."""
        body = client.get("/api/composer/graph").json()
        pairs = [(e["from"], e["to"]) for e in body["edges"]]
        assert len(pairs) == len(set(pairs))
        edges = {(e["from"], e["to"]): e["suffixes"] for e in body["edges"]}
        assert edges[("extract", "chunk")] == [
            ".elements.parquet", ".table_cells.parquet", "._manifest.parquet",
        ]
        # embed reads chunk's own output — a stage-to-stage edge, not extract.
        assert edges[("chunk", "embed")] == [".chunks.parquet"]

    def test_graph_is_acyclic_and_every_stage_descends_from_extract(
        self, client: TestClient
    ) -> None:
        from womblex.cloud.stage_contracts import STAGE_NAMES

        body = client.get("/api/composer/graph").json()
        adjacency: dict[str, set[str]] = {}
        for edge in body["edges"]:
            adjacency.setdefault(edge["from"], set()).add(edge["to"])
        reached: set[str] = set()

        def walk(node: str, seen: frozenset[str]) -> None:
            for nxt in adjacency.get(node, ()):
                assert nxt not in seen, f"cycle through {nxt}"
                reached.add(nxt)
                walk(nxt, seen | {nxt})

        walk("extract", frozenset({"extract"}))
        # A stage unreachable from extract would be one the composer cannot
        # order — the plan's guardrail is precisely this reachability.
        assert reached == set(STAGE_NAMES)

    def test_graph_node_config_sections_are_real_config_fields(
        self, client: TestClient
    ) -> None:
        """The composer's node → config-section map is hand-declared (nothing in
        `StageContract` names a config field), so a renamed section must fail
        here rather than silently leave a stage's toggle wired to nothing."""
        from womblex.config import WomblexConfig

        body = client.get("/api/composer/graph").json()
        sections = {n["config_section"] for n in body["nodes"]} - {None}
        assert sections and sections <= set(WomblexConfig.model_fields)
        assert {n["id"] for n in body["nodes"] if n["config_section"] is None} == {
            "extract",
            "graph-refresh",
            "layout",
        }

    def test_schema_is_the_womblex_config_schema_minus_paths(self, client: TestClient) -> None:
        """`paths` is the deployment's locations, not operator-retyped per run."""
        body = client.get("/api/composer/schema").json()
        assert set(body["properties"]) >= {"dataset", "chunking", "pii"}
        assert "paths" not in body["properties"]
        assert "paths" not in body.get("required", [])
        assert "dataset" in body.get("required", [])

    def test_presets_list_includes_default_isaacus(self, client: TestClient) -> None:
        body = client.get("/api/composer/presets").json()
        names = [p["name"] for p in body["presets"]]
        assert "DEFAULT-Isaacus" in names
        preset = next(p for p in body["presets"] if p["name"] == "DEFAULT-Isaacus")
        assert preset["formats"] == [".pdf", ".docx", ".md", ".markdown"]

    def test_default_isaacus_enables_the_full_reference_shape(
        self, client: TestClient
    ) -> None:
        """The preset is the reference extract → normalise → enrich → chunk →
        build_graph → embed → money → link pipeline: text cleaning (via
        processing.text_source=normalised, which is also what gates normalise),
        AI chunking, enrichment, embedding, money and linking all on; graph
        produced by enrich + the offline graph-refresh edge rebuild. Embed is
        part of the shape (the demo run carries it); a preset that omitted it
        disagreed with the sample corpus, which is the bug this pins.

        spellfix is NOT in the shape: dictionary-gated repair is a per-corpus
        call, so the reference default must not carry it."""
        preset = client.get("/api/composer/presets/DEFAULT-Isaacus").json()
        cfg = preset["config"]
        assert cfg["chunking"]["enabled"] is True
        assert cfg["chunking"]["chunking_model"] == "kanon-2-enricher"
        assert cfg["enrichment"]["enabled"] is True
        assert cfg["embedding"]["enabled"] is True
        assert cfg["money"]["enabled"] is True
        assert cfg["linking"]["enabled"] is True
        assert "spellfix" not in cfg
        # Text cleaning is selected once; this is also what gates normalise into
        # the downstream-stage dispatch and puts it before enrich.
        assert cfg["processing"]["text_source"] == "normalised"
        # No dataset/paths: the operator supplies the run's identity and paths.
        assert "dataset" not in cfg
        assert "paths" not in cfg

    def test_a_preset_overlaid_on_a_minimal_config_validates(
        self, client: TestClient
    ) -> None:
        """A preset is a partial config; merged onto dataset+paths it must be a
        config the CLI would load — that is the whole point of it being data
        the same `WomblexConfig(**raw)` construction checks."""
        preset = client.get("/api/composer/presets/DEFAULT-Isaacus").json()
        merged = {**_MINIMAL_CONFIG, **preset["config"]}
        resp = client.post("/api/composer/validate", json=merged)
        body = resp.json()
        assert body["valid"] is True

    def test_unknown_preset_404s(self, client: TestClient) -> None:
        assert client.get("/api/composer/presets/nope").status_code == 404

    def test_default_isaacus_config_file_ships_and_loads(self, client: TestClient) -> None:
        """The preset is CLI-runnable: `configs/default-isaacus.yaml` exists and
        `load_config` accepts it (a dev runs the per-stage sequence with it)."""
        from womblex.config import load_config

        cfg_path = REPO_ROOT / "configs" / "default-isaacus.yaml"
        assert cfg_path.is_file(), "configs/default-isaacus.yaml is the CLI source of truth"
        cfg = load_config(cfg_path)
        # The stages the shape names are on.
        assert cfg.chunking.enabled and cfg.chunking.chunking_model == "kanon-2-enricher"
        assert cfg.enrichment.enabled
        assert cfg.embedding.enabled
        assert cfg.money.enabled
        assert cfg.linking.enabled
        assert not cfg.spellfix.enabled, "spellfix is a per-corpus opt-in"
        # Text cleaning runs before enrich/chunk: the overlay both reassemble from.
        assert cfg.processing.text_source == "normalised"
        # AI chunking + enrich both on => reuse auto-wired, so no double enrich.
        assert cfg.enrichment.persist_document is True

    def test_config_file_and_ui_preset_agree(self, client: TestClient) -> None:
        """The console preset mirrors the shipped config file; they must not drift
        on the stage toggles/settings the preset carries."""
        from womblex.config import load_config

        cfg = load_config(REPO_ROOT / "configs" / "default-isaacus.yaml")
        overlay = client.get("/api/composer/presets/DEFAULT-Isaacus").json()["config"]
        assert overlay["chunking"]["chunk_size"] == cfg.chunking.chunk_size
        assert overlay["chunking"]["chunking_model"] == cfg.chunking.chunking_model
        assert overlay["chunking"]["overlap"] == cfg.chunking.overlap
        # The stage toggles the preset carries agree with the config file.
        assert overlay["enrichment"]["enabled"] == cfg.enrichment.enabled
        assert overlay["embedding"]["enabled"] == cfg.embedding.enabled
        assert overlay["money"]["enabled"] == cfg.money.enabled
        assert overlay["money"]["default_currency"] == cfg.money.default_currency
        assert overlay["linking"]["enabled"] == cfg.linking.enabled
        # Neither carries spellfix: absent from the overlay, off in the config.
        assert "spellfix" not in overlay and not cfg.spellfix.enabled
        # The text-cleaning selector (which also gates normalise into the
        # downstream dispatch, before enrich) agrees.
        assert overlay["processing"]["text_source"] == cfg.processing.text_source

    def test_validate_accepts_a_minimal_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/validate", json=_MINIMAL_CONFIG)
        assert resp.json() == {"valid": True, "errors": []}

    def test_validate_refuses_typos(self, client: TestClient) -> None:
        """The config models forbid extras, so a typo is an error naming the key."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "chunkng": {"chunk_size": 99},            # typo'd section
            "chunking": {"chnk_size": 99},            # typo'd field in a real section
        })
        body = resp.json()
        assert body["valid"] is False
        assert sorted(tuple(err["loc"]) for err in body["errors"]) == [
            ("chunking", "chnk_size"), ("chunkng",),
        ]

    def test_validate_does_not_mistake_free_form_dict_values_for_typos(
        self, client: TestClient
    ) -> None:
        """`normalise.substitutions` is an operator's own letterhead map — its keys
        are data, not config fields, and must not be walked into."""
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "normalise": {"substitutions": {"Depatment": "Department", "AB C": "ABC"}},
        })
        assert resp.json()["valid"] is True

    def test_validate_refuses_a_stray_key_in_an_optional_nested_model(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/composer/validate", json={
            **_MINIMAL_CONFIG,
            "linking": {"reference": {"path": "r.csv", "nonsense_key": 1}},
        })
        body = resp.json()
        assert body["valid"] is False
        assert any("nonsense_key" in err["loc"] for err in body["errors"])

    def test_validate_reports_pydantic_errors_for_a_missing_field(self, client: TestClient) -> None:
        resp = client.post("/api/composer/validate", json=_NO_DATASET)
        body = resp.json()
        assert body["valid"] is False
        assert any(err["loc"] == ["dataset"] for err in body["errors"])

    def test_deployment_paths_override_any_submitted_paths(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """A posted `paths` is replaced wholesale, never touched otherwise."""
        resp = client.post("/api/composer/validate", json={
            "dataset": {"name": "t"},
            "paths": {"input_root": "/nonexistent/xyz", "output_root": "/root/.ssh",
                      "checkpoint_dir": "/etc/shadow"},
        })
        assert resp.json()["valid"] is True
        yaml_resp = client.post("/api/composer/yaml", json={"dataset": {"name": "t"}})
        parsed = yaml.safe_load(yaml_resp.text)
        assert parsed["paths"]["output_root"] == str(tmp_path)
        assert parsed["paths"]["input_root"] != "/nonexistent/xyz"

    def test_yaml_download_roundtrips_a_valid_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json=_MINIMAL_CONFIG)
        assert resp.status_code == 200
        assert resp.headers["content-disposition"] == "attachment; filename=womblex.yaml"
        parsed = yaml.safe_load(resp.text)
        assert parsed["dataset"]["name"] == "t"
        # Pydantic-applied defaults are in the download, not just what was posted.
        assert "chunking" in parsed
        # The download's whole purpose: `load_config` must accept it back.
        from womblex.config import WomblexConfig

        assert WomblexConfig(**parsed).dataset.name == "t"
        assert not resp.text.startswith("#")

    def test_yaml_download_422s_on_a_stray_key(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json={
            **_MINIMAL_CONFIG, "chunkng": {"chunk_size": 99},
        })
        assert resp.status_code == 422
        assert any(err["loc"] == ["chunkng"] for err in resp.json()["detail"])

    def test_yaml_download_422s_on_an_invalid_config(self, client: TestClient) -> None:
        resp = client.post("/api/composer/yaml", json=_NO_DATASET)
        assert resp.status_code == 422
        assert any(err["loc"] == ["dataset"] for err in resp.json()["detail"])

    def test_built_in_presets_are_marked_as_such(self, client: TestClient) -> None:
        """`source` lets the screen offer delete only on saved presets; a built-in
        is code and cannot be removed."""
        body = client.get("/api/composer/presets").json()
        builtin = next(p for p in body["presets"] if p["name"] == "DEFAULT-Isaacus")
        assert builtin["source"] == "builtin"


class TestComposerSavePresets:
    """Saving a composed config as a named preset (docs/ui-plan.md merge 9).

    Parametrised over local (a writable `presets_dir`) and remote (the store's
    own `presets/` prefix, a sibling of `runs/`). A store-backed console therefore
    saves without any writable mount; a local console without a presets dir
    refuses with 409, the same shape the Execution Controls use.
    """

    def _saving_client(self, tmp_path: Path, mode: str) -> TestClient:
        """A client that *can* save, in the given mode."""
        if mode == "remote":
            pytest.importorskip("fsspec")
            return TestClient(create_app(store_uri=str(tmp_path / "store")))
        return TestClient(create_app(output_root=tmp_path, presets_dir=tmp_path / "presets"))

    def _client(self, tmp_path: Path, *, with_presets_dir: bool = True) -> TestClient:
        presets_dir = (tmp_path / "presets") if with_presets_dir else None
        return TestClient(create_app(output_root=tmp_path, presets_dir=presets_dir))

    @pytest.mark.parametrize("mode", ["local", "remote"])
    def test_save_then_list_and_fetch_round_trips(self, tmp_path: Path, mode: str) -> None:
        client = self._saving_client(tmp_path, mode)
        resp = client.post("/api/composer/presets", json={
            "name": "My-Run", "description": "chunk only", "formats": [".pdf"],
            "config": {"chunking": {"enabled": True, "chunk_size": 320}},
        })
        assert resp.status_code == 201
        assert resp.json()["source"] == "saved"

        names = {p["name"]: p for p in client.get("/api/composer/presets").json()["presets"]}
        assert "My-Run" in names and names["My-Run"]["source"] == "saved"
        # Built-ins still serve alongside the saved one.
        assert "DEFAULT-Isaacus" in names

        one = client.get("/api/composer/presets/My-Run").json()
        assert one["config"]["chunking"]["chunk_size"] == 320

    def test_remote_save_lands_under_the_store_presets_prefix(self, tmp_path: Path) -> None:
        """A store-backed console writes presets to `presets/`, a sibling of
        `runs/` — so the compose ui service needs no writable mount."""
        pytest.importorskip("fsspec")
        store_root = tmp_path / "store"
        client = TestClient(create_app(store_uri=str(store_root)))
        resp = client.post("/api/composer/presets", json={
            "name": "Remote-One", "config": {"chunking": {"chunk_size": 128}},
        })
        assert resp.status_code == 201
        # One object, under the store's presets/ prefix, and nowhere near runs/.
        saved = list((store_root / "presets").glob("*.preset.json"))
        assert [p.name for p in saved] == ["Remote-One.preset.json"]
        assert not (store_root / "runs").exists()

    def test_save_strips_dataset_and_paths(self, tmp_path: Path) -> None:
        """A preset is an overlay — the run's identity is never baked into it."""
        client = self._client(tmp_path)
        resp = client.post("/api/composer/presets", json={
            "name": "Overlay",
            "config": {**_MINIMAL_CONFIG, "money": {"enabled": True}},
        })
        cfg = resp.json()["config"]
        assert "dataset" not in cfg and "paths" not in cfg
        assert cfg["money"]["enabled"] is True

    def test_saved_preset_overlaid_on_a_minimal_config_still_validates(
        self, tmp_path: Path
    ) -> None:
        """The whole point of validating at save time: what is saved must load."""
        client = self._client(tmp_path)
        client.post("/api/composer/presets", json={
            "name": "Round", "config": {"chunking": {"chunk_size": 200}},
        })
        overlay = client.get("/api/composer/presets/Round").json()["config"]
        resp = client.post("/api/composer/validate", json={**_MINIMAL_CONFIG, **overlay})
        assert resp.json()["valid"] is True

    def test_save_rejects_a_config_that_would_not_load(self, tmp_path: Path) -> None:
        """A preset that fails to build a `WomblexConfig` is refused at save (400),
        not left to 500 whoever later picks it."""
        client = self._client(tmp_path)
        resp = client.post("/api/composer/presets", json={
            "name": "Bad", "config": {"chunking": {"chunk_size": "not-an-int"}},
        })
        assert resp.status_code == 400

    def test_save_rejects_an_unsafe_name(self, tmp_path: Path) -> None:
        client = self._client(tmp_path)
        for name in ["../escape", "a/b", "..", ".hidden"]:
            resp = client.post("/api/composer/presets", json={"name": name, "config": {}})
            assert resp.status_code == 400, name
        # Nothing was written anywhere — no file escaped the presets dir.
        assert list(tmp_path.rglob("*.preset.json")) == []

    def test_save_conflicts_without_a_presets_dir(self, tmp_path: Path) -> None:
        client = self._client(tmp_path, with_presets_dir=False)
        resp = client.post("/api/composer/presets", json={"name": "X", "config": {}})
        assert resp.status_code == 409

    @pytest.mark.parametrize("mode", ["local", "remote"])
    def test_delete_removes_a_saved_preset(self, tmp_path: Path, mode: str) -> None:
        client = self._saving_client(tmp_path, mode)
        client.post("/api/composer/presets", json={"name": "Temp", "config": {}})
        assert client.delete("/api/composer/presets/Temp").status_code == 200
        assert client.get("/api/composer/presets/Temp").status_code == 404

    def test_delete_refuses_to_remove_a_built_in(self, tmp_path: Path) -> None:
        """A built-in is code; deleting it is a 404 here, not a silent success."""
        client = self._client(tmp_path)
        assert client.delete("/api/composer/presets/DEFAULT-Isaacus").status_code == 404
        # And it is still there.
        assert client.get("/api/composer/presets/DEFAULT-Isaacus").status_code == 200

    def test_delete_conflicts_without_a_presets_dir(self, tmp_path: Path) -> None:
        client = self._client(tmp_path, with_presets_dir=False)
        assert client.delete("/api/composer/presets/anything").status_code == 409

    def test_a_saved_preset_shadows_a_built_in_of_the_same_name(self, tmp_path: Path) -> None:
        """The operator asked for that name; their save wins, still marked saved."""
        client = self._client(tmp_path)
        client.post("/api/composer/presets", json={
            "name": "DEFAULT-Isaacus", "config": {"chunking": {"chunk_size": 999}},
        })
        one = client.get("/api/composer/presets/DEFAULT-Isaacus").json()
        assert one["source"] == "saved"
        assert one["config"]["chunking"]["chunk_size"] == 999


class _FakeQueue:
    """A `RunBoard` stand-in — the execute happy path must not need Postgres.

    Records what `enqueue_extraction` hands it (schema call, run_id, specs) so
    a test can assert the batching without a live database, and returns a
    fixed `newly` count for `enqueue`.
    """

    last: _FakeQueue | None = None
    schema_ensured = False

    def __init__(self, dsn: str, **_kw: object) -> None:
        self.dsn = dsn
        self.enqueued: tuple[str, list] | None = None
        _FakeQueue.last = self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def enqueue(self, run_id: str, specs: list, *, owner: str | None = None) -> int:
        self.enqueued = (run_id, specs)
        return len(specs)


@pytest.fixture(autouse=True)
def _fake_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch provisions the DBOS schema through a module function; record, don't run it."""
    _FakeQueue.schema_ensured = False
    monkeypatch.setattr(
        "womblex.cloud.dispatch.ensure_schema",
        lambda _dsn: setattr(_FakeQueue, "schema_ensured", True),
    )


def _seed_store_inputs(store_root: Path, prefix: str, names: list[str]) -> None:
    """Write source documents under `<store_root>/<prefix>` for enqueue to list."""
    in_dir = store_root / prefix
    in_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (in_dir / name).write_bytes(b"%PDF-1.4 stub")


class TestExecuteApi:
    """Dispatch requires a store, an ingest location, and a DSN
    (docs/ui-plan.md merge 11)."""

    def test_status_reflects_the_flags(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        body = client.get("/api/execute/status").json()
        assert body["can_execute"] is False
        assert body == {
            "can_execute": False,
            "has_store": False, "has_ingest": False, "has_queue": False,
            "ingest_uri": None, "output_uri": str(tmp_path),
        }

    def test_status_advertises_no_stage_list(self, tmp_path: Path) -> None:
        """The pipeline's shape is the composer's graph, not a dispatch flag.

        `stages` here used to be `STAGE_NAMES` — read as "what this console
        dispatches" beside `can_execute`, while no dispatch path for a stage
        existed and no caller read the field.
        """
        client = TestClient(create_app(output_root=tmp_path))
        assert "stages" not in client.get("/api/execute/status").json()

    def test_status_can_execute_with_a_store_ingest_and_queue(self, tmp_path: Path) -> None:
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"),
            db_dsn="postgresql://x/y",
        ))
        body = client.get("/api/execute/status").json()
        assert body["can_execute"] is True
        assert body["ingest_uri"] == str(tmp_path / "inbox")
        assert body["output_uri"] == str(tmp_path / "store")

    def test_enqueue_conflict_without_a_store(self, tmp_path: Path) -> None:
        """A local output_root can configure and audit but not dispatch."""
        client = TestClient(create_app(
            output_root=tmp_path, ingest_uri=str(tmp_path / "inbox"), db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 409

    def test_enqueue_conflict_without_an_ingest_location(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 409

    def test_enqueue_conflict_without_a_queue(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"),
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 409

    def test_enqueue_400_when_no_documents_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ingest root is bad input (400), not a disabled console (403/409)."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["notes.txt"])  # unsupported ext
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.status_code == 400

    def test_enqueue_plans_batches_into_the_queue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List the whole ingest root (no prefix posted), batch, one idempotent
        row each, stamped with the ingest root."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", [f"doc-{i}.pdf" for i in range(5)])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post(
            "/api/execute/enqueue", json={"run_id": "run-exec", "batch_size": 2},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run-exec"
        assert body["document_count"] == 5
        assert body["batch_count"] == 3  # 2 + 2 + 1
        assert body["newly_enqueued"] == 3
        assert body["shard_prefix"] == "runs/run-exec/documents"

        queue = _FakeQueue.last
        assert queue is not None and queue.schema_ensured
        run_id, specs = queue.enqueued  # type: ignore[misc]
        assert run_id == "run-exec"
        assert [s.batch_num for s in specs] == [1, 2, 3]
        assert [len(s.input_keys) for s in specs] == [2, 2, 1]
        assert all(s.shard_prefix == "runs/run-exec/documents" for s in specs)
        assert all(s.ingest_root == str(ingest_root) for s in specs)

    def test_enqueue_refuses_a_nested_ingest_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P11: a run ingests what is directly under the location it is given.
        The console refuses the same layout the local CLI does, as bad input,
        and nothing reaches the queue."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        _FakeQueue.last = None
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(ingest_root, "2026-08", ["nested.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post(
            "/api/execute/enqueue", json={"run_id": "run-exec", "batch_size": 2},
        )
        assert resp.status_code == 400
        assert "2026-08/" in resp.json()["detail"]
        assert _FakeQueue.last is None

    def test_the_prefix_previewed_is_the_prefix_enqueued(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P12: preview and dispatch cannot disagree about which documents a
        press will run — the count the preflight reports for a prefix is the
        count that prefix enqueues, and the keys are scoped to it."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf", "b.pdf"])
        _seed_store_inputs(ingest_root, "2026-09", ["c.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        preview = client.get(
            "/api/execute/ingest", params={"input_prefix": "2026-08"},
        ).json()
        resp = client.post("/api/execute/enqueue", json={
            "run_id": "run-p", "input_prefix": preview["input_prefix"],
        })
        assert resp.status_code == 200
        assert resp.json()["document_count"] == preview["document_count"] == 2
        _, specs = _FakeQueue.last.enqueued  # type: ignore[union-attr, misc]
        assert sorted(specs[0].input_keys) == ["2026-08/a.pdf", "2026-08/b.pdf"]

    def test_the_keys_enqueued_carry_no_dot_segment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A queued key is downloaded by the worker and recorded as the
        document's `source_relpath`, so a prefix spelled `./2026-08` must not
        publish provenance naming a key that only a local path would resolve."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={
            "run_id": "run-dot", "input_prefix": "./2026-08/",
        })
        assert resp.status_code == 200
        _, specs = _FakeQueue.last.enqueued  # type: ignore[union-attr, misc]
        assert specs[0].input_keys == ["2026-08/a.pdf"]

    def test_enqueue_refuses_a_prefix_escaping_the_ingest_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bad input (400), before any listing, and nothing reaches the queue —
        the treatment an unsafe run id already gets."""
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        _FakeQueue.last = None
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post(
            "/api/execute/enqueue", json={"input_prefix": "../../etc"},
        )
        assert resp.status_code == 400
        assert _FakeQueue.last is None

    def test_enqueue_mints_a_run_id_when_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("fsspec")
        monkeypatch.setattr("womblex.cloud.jobs.RunBoard", _FakeQueue)
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["doc-0.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={})
        assert resp.json()["run_id"].startswith("run-")

    def test_enqueue_rejects_a_batch_size_below_one(self, tmp_path: Path) -> None:
        """Pydantic's `ge=1` refuses it at the boundary (422), before the guard runs."""
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(tmp_path / "inbox"),
            db_dsn="postgresql://x/y",
        ))
        resp = client.post("/api/execute/enqueue", json={"batch_size": 0})
        assert resp.status_code == 422


class TestIngestPreflight:
    """`GET /api/execute/ingest` — reachability + document count."""

    def test_no_ingest_configured(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path))
        body = client.get("/api/execute/ingest").json()
        assert body == {
            "uri": None, "input_prefix": "", "kind": None, "reachable": False,
            "document_count": 0, "sample": [], "subdirectories": [],
            "error": "no ingest location configured",
        }

    def test_nested_prefixes_report_the_refusal_not_a_count(self, tmp_path: Path) -> None:
        """P11: nested documents are refused, not ingested, so the preflight
        reports the refusal rather than a count the enqueue will not honour.

        Object stores have no folders — `inbox/2026-08/foo.pdf` is one key — so
        this layout is a normal upload and the operator needs to be told why it
        is rejected, not shown 0 documents with no reason.

        P12: the refusal also names the immediate subdirectories that hold
        documents, with a count each and the prefix that selects one — the
        recovery the message describes, made selectable from this screen
        instead of from the CLI."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(ingest_root, "2026-08/health", ["nested.pdf"])
        _seed_store_inputs(ingest_root, "scratch", ["a.pdf", "b.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get("/api/execute/ingest").json()
        assert body["reachable"] is True
        assert body["document_count"] == 0
        assert "2026-08/" in body["error"]
        assert body["subdirectories"] == [
            {"name": "2026-08", "input_prefix": "2026-08", "document_count": 1},
            {"name": "scratch", "input_prefix": "scratch", "document_count": 2},
        ]

    def test_a_refusal_under_a_prefix_names_prefixes_that_select_deeper(
        self, tmp_path: Path
    ) -> None:
        """The subdirectory prefixes are ingest-relative, so one press deeper
        selects the next level rather than restarting from the root."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08/health", ["nested.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get("/api/execute/ingest", params={"input_prefix": "2026-08"}).json()
        assert body["input_prefix"] == "2026-08"
        assert body["subdirectories"] == [
            {"name": "health", "input_prefix": "2026-08/health", "document_count": 1},
        ]

    def test_a_prefix_resolving_flat_reports_its_count(self, tmp_path: Path) -> None:
        """P12: the recovery the refusal names works, from the refusal alone —
        the prefix it hands back previews as a count, with no second tool and
        nothing for the operator to compose by hand."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf", "b.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        refused = client.get("/api/execute/ingest").json()
        chosen = refused["subdirectories"][0]["input_prefix"]
        body = client.get("/api/execute/ingest", params={"input_prefix": chosen}).json()
        assert body["error"] is None
        assert body["input_prefix"] == "2026-08"
        assert body["document_count"] == 2
        # The sample is the keys, prefix and all, as the enqueue would stage them.
        assert sorted(body["sample"]) == ["2026-08/a.pdf", "2026-08/b.pdf"]

    def test_a_prefix_is_echoed_normalised_so_the_composer_posts_one_scope(
        self, tmp_path: Path
    ) -> None:
        """Spelling-by-spelling normalisation is pinned on the shared rule in
        `test_discovery_parity`; what this holds is that the preflight echoes
        the normalised prefix, since that string is what the composer posts."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "2026-08", ["a.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get(
            "/api/execute/ingest", params={"input_prefix": "./2026-08/"},
        ).json()
        assert body["input_prefix"] == "2026-08"
        assert body["sample"] == ["2026-08/a.pdf"]

    def test_a_prefix_escaping_the_ingest_root_is_refused(self, tmp_path: Path) -> None:
        """Same footing as the unsafe-run-id check: 400, and no listing."""
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", ["top.pdf"])
        _seed_store_inputs(tmp_path / "elsewhere", "", ["secret.pdf"])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
            db_dsn="postgresql://x/y",
        ))
        resp = client.get("/api/execute/ingest", params={"input_prefix": "../elsewhere"})
        assert resp.status_code == 400
        assert "unsafe input_prefix" in resp.json()["detail"]

    def test_reachable_ingest_reports_count_and_sample(self, tmp_path: Path) -> None:
        pytest.importorskip("fsspec")
        ingest_root = tmp_path / "inbox"
        _seed_store_inputs(ingest_root, "", [f"doc-{i}.pdf" for i in range(3)])
        client = TestClient(create_app(
            store_uri=str(tmp_path / "store"), ingest_uri=str(ingest_root),
        ))
        body = client.get("/api/execute/ingest").json()
        assert body["kind"] == "local"
        assert body["reachable"] is True
        assert body["document_count"] == 3
        assert sorted(body["sample"]) == ["doc-0.pdf", "doc-1.pdf", "doc-2.pdf"]


class TestStoreUnreachable:
    """A store that cannot be opened degrades to a legible 503, not an opaque 500.

    The canonical cause in a cloud deployment is ``womblex[cloud]`` (and so
    ``s3fs``) not being installed: ``RemoteStore.from_uri('s3://…')`` raises
    ``ImportError: Install s3fs to access S3``. That is a deployment fault, not
    a bug in the request, so the read routes surface it as 503 carrying the
    underlying message — the same cause the Resources card's *Test connection*
    reports — rather than letting the raw exception become a 500 top-banner.
    """

    def _client(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        def _boom(_uri: str, _credentials: tuple[str, str] | None = None) -> object:
            raise readers.StoreUnreachable("Install s3fs to access S3")

        monkeypatch.setattr(readers, "_open_store", _boom)
        return TestClient(create_app(store_uri=str(tmp_path / "store")))

    def test_list_runs_returns_503_with_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        resp = client.get("/api/runs")
        assert resp.status_code == 503
        assert "s3fs" in resp.json()["detail"]

    def test_presets_returns_503_with_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        resp = client.get("/api/composer/presets")
        assert resp.status_code == 503
        assert "s3fs" in resp.json()["detail"]

    def test_manifest_and_audit_also_degrade_to_503(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        assert client.get("/api/runs/run-a/manifest").status_code == 503
        assert client.get("/api/runs/run-a/audit").status_code == 503
        assert client.get("/api/runs/run-a/stage-presence/chunk").status_code == 503
        assert client.get("/api/runs/run-a/chunks/hash-a").status_code == 503

    def test_open_store_wraps_a_backend_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wrap happens in `_open_store`, so every reader shares one mapping."""
        import womblex.store.remote as remote_mod

        def _raise(_uri: str, **_kw: object) -> object:
            raise ImportError("Install s3fs to access S3")

        monkeypatch.setattr(remote_mod.RemoteStore, "from_uri", staticmethod(_raise))
        with pytest.raises(readers.StoreUnreachable, match="s3fs"):
            readers._open_store("s3://redline")


class TestSpaMount:
    """The console serves the built SPA (docs/ui-plan.md merge 4) when one is
    present alongside it, and falls back to the API-only shape when not —
    the same image serves cloud and local deployments.
    """

    @pytest.fixture
    def spa_dir(self, tmp_path: Path) -> Path:
        build = tmp_path / "spa"
        (build / "_app").mkdir(parents=True)
        (build / "index.html").write_text("<html><body>console shell</body></html>")
        (build / "_app" / "app.js").write_text("console.log('spa')")
        return build

    def test_no_spa_dir_serves_api_only(self, tmp_path: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=tmp_path / "nope"))
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 404

    def test_serves_index_at_root(self, tmp_path: Path, spa_dir: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "console shell" in resp.text

    def test_serves_index_for_client_side_routes(self, tmp_path: Path, spa_dir: Path) -> None:
        """`/corpus` is a SvelteKit client route, not a file — the SPA router takes over."""
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/corpus")
        assert resp.status_code == 200
        assert "console shell" in resp.text

    def test_serves_real_asset_files(self, tmp_path: Path, spa_dir: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/_app/app.js")
        assert resp.status_code == 200
        assert "spa" in resp.text

    def test_api_routes_win_over_the_spa_fallback(self, tmp_path: Path, spa_dir: Path) -> None:
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        assert client.get("/api/runs").json() == {"runs": []}

    def test_unmatched_api_path_404s_instead_of_serving_the_shell(
        self, tmp_path: Path, spa_dir: Path
    ) -> None:
        """A wrong endpoint must not answer 200 + HTML — a JSON client would
        report that as a parse error rather than the 404 it is.
        """
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        resp = client.get("/api/bogus")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/json")

    def test_client_routes_still_serve_the_shell_when_api_404s(
        self, tmp_path: Path, spa_dir: Path
    ) -> None:
        """The /api guard must not catch routes that merely start with 'api'."""
        client = TestClient(create_app(output_root=tmp_path, spa_dir=spa_dir))
        assert "console shell" in client.get("/apiary").text

    def test_path_traversal_falls_back_to_index_instead_of_escaping(
        self, tmp_path: Path, spa_dir: Path
    ) -> None:
        """Unit-tested directly: an HTTP client may normalise `..` in the URL
        before it ever reaches the server, which would let this pass without
        exercising the containment check at all.
        """
        from womblex.ui.app import resolve_spa_path

        secret = tmp_path / "secret.txt"
        secret.write_text("outside the SPA root")
        resolved = resolve_spa_path(spa_dir, f"../{secret.name}")
        assert resolved == spa_dir / "index.html"


class TestSidecarImage:
    """The console container is only correct if it agrees with the CLI and the app.

    Nothing else in the suite reads the deployment files, so a renamed flag or
    a second run source would otherwise surface as a container that exits 1 in
    someone's compose stack.
    """

    def test_entrypoint_parses_against_the_real_cli(self) -> None:
        """`womblex ui`'s flags are the image's entrypoint — they must still exist."""
        match = re.search(
            r"^ENTRYPOINT (\[.*\])$", (REPO_ROOT / "Dockerfile.ui").read_text(), re.MULTILINE
        )
        assert match, "Dockerfile.ui has no JSON-form ENTRYPOINT"
        entrypoint = json.loads(match.group(1))
        assert entrypoint[0] == "womblex"

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        for cmd in ALL_COMMANDS:
            cmd.register(sub.add_parser(cmd.name))
        args = parser.parse_args(entrypoint[1:])  # SystemExit if a flag is gone
        assert args.command == "ui"
        # Loopback is the CLI default, and inside a container it would refuse
        # every request arriving via the published port.
        assert args.host == "0.0.0.0"

    def test_compose_ui_service_is_read_only_with_tmpfs(self) -> None:
        ui = _compose_service("ui")
        # The console image, named rather than built: the base file pins a
        # published image and the local override carries the build. Which
        # service runs which image is audited in docs/deployment-images.md.
        assert "womblex-console" in ui["image"]
        assert ui["read_only"] is True
        # A store-backed read stages the manifest through a temp dir.
        assert "/tmp" in ui["tmpfs"]

    def test_compose_ui_service_resolves_exactly_one_run_source(self) -> None:
        """Two sources make resolve_settings raise, so the container would exit 1."""
        env = _compose_service("ui")["environment"]
        assert "WOMBLEX_STORE_URI" in env
        assert "WOMBLEX_UI_OUTPUT_ROOT" not in env

    def test_compose_ui_service_waits_for_the_queue_and_store_it_advertises(self) -> None:
        """The ui service advertises WOMBLEX_DB_DSN and WOMBLEX_STORE_URI, so it must
        wait for postgres AND minio to be *healthy* — not just for the bucket to
        exist. Coming up before either is ready was the cause of the reported
        "queue unreachable" and the intermittent Composer 500s/failed-to-fetch:
        every screen's first reads raced the backends' startup.
        """
        deps = _compose_service("ui")["depends_on"]
        env = _compose_service("ui")["environment"]
        # It advertises both, so a consumer expects both to answer on boot.
        assert "WOMBLEX_DB_DSN" in env and "WOMBLEX_STORE_URI" in env
        assert deps["postgres"]["condition"] == "service_healthy"
        assert deps["minio"]["condition"] == "service_healthy"
        # The bucket must also exist before the store reads land.
        assert deps["createbuckets"]["condition"] == "service_completed_successfully"

    def test_bundled_backends_sit_behind_the_local_profile(self) -> None:
        """The self-contained stack (bundled Postgres/MinIO/bucket) is opt-in via the
        `local` profile, so a cloud deployment against external Postgres + S3 does
        not start them: `docker compose up worker` brings up only the profile-less
        services. Without this, cloud mode would spin up a Postgres/MinIO nobody
        uses (and, worse, `worker`'s deps would wait on them).
        """
        svcs = _compose()["services"]
        for name in ["postgres", "minio", "createbuckets"]:
            assert svcs[name]["profiles"] == ["local"], name
        # The consumers carry no profile — they are the default `up` target.
        for name in ["init", "worker", "ui"]:
            assert "profiles" not in svcs[name], name

    def test_cloud_services_do_not_hard_depend_on_the_bundled_backends(self) -> None:
        """Every dependency a cloud service declares on a `local`-profile backend is
        `required: false`, so an absent bundled Postgres/MinIO/bucket (the cloud
        case, external services instead) is a warning, not a missing-dependency
        error. This is the one thing that lets `docker compose up worker` run at
        all when the bundled backends were never started.
        """
        svcs = _compose()["services"]
        local_backends = {"postgres", "minio", "createbuckets", "init"}
        for name in ["init", "worker", "womblex", "seed-demo", "ui"]:
            for dep, spec in svcs[name].get("depends_on", {}).items():
                if dep in local_backends:
                    assert spec["required"] is False, f"{name} hard-depends on {dep}"

    def test_ui_health_gate_survives_into_cloud_mode(self) -> None:
        """The startup-race fix (ui waits for postgres/minio *healthy*) must still
        hold as `required: false` — the condition is unchanged, so a local stack
        still gates on health, and a cloud stack simply has no bundled backend to
        wait on (it reads the external DSN/S3 the same env names).
        """
        deps = _compose_service("ui")["depends_on"]
        assert deps["postgres"]["condition"] == "service_healthy"
        assert deps["postgres"]["required"] is False
        assert deps["minio"]["condition"] == "service_healthy"
        assert deps["minio"]["required"] is False

    def test_connection_env_is_overridable_with_bundled_local_defaults(self) -> None:
        """The whole local/cloud switch: each connection var reads from the
        environment with the bundled local stack as the `${VAR:-default}`
        fallback. Unset env == today's local stack (the defaults point at the
        bundled services); set env == external Postgres + S3. A hard-coded value
        here would silently ignore an operator's external endpoint.

        Store credentials live on the store-specific `WOMBLEX_S3_ACCESS_KEY_ID` /
        `WOMBLEX_S3_SECRET_ACCESS_KEY` (defaulting to MinIO's key locally), while
        the ambient `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` default to
        **empty** on purpose: setting them is process-global and would clobber
        the EC2 instance role the isaacus-sagemaker SigV4 signer resolves in a
        cloud deployment (the credential-conflict bug).
        """
        raw = (REPO_ROOT / "docker-compose.yml").read_text()
        # The connection vars + store creds are all `${VAR:-<bundled default>}`.
        for var, default in [
            # The bundled compose default, already baselined in docker-compose.yml.
            ("WOMBLEX_DB_DSN", "postgresql://womblex:womblex@postgres:5432/womblex"),  # pragma: allowlist secret
            ("WOMBLEX_STORE_URI", "s3://womblex"),
            ("WOMBLEX_S3_ENDPOINT", "http://minio:9000"),
            ("WOMBLEX_S3_ACCESS_KEY_ID", "minioadmin"),
            ("WOMBLEX_S3_SECRET_ACCESS_KEY", "minioadmin"),
            ("AWS_REGION", "us-east-1"),
        ]:
            assert f"${{{var}:-{default}}}" in raw, var
        # The AWS credential vars pass through only when the operator sets them,
        # defaulting to empty so the SageMaker signer can reach the instance role.
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            assert f"${{{var}:-}}" in raw, var
        # The anchor the services inherit carries the same overridable form —
        # `yaml.safe_load` does not run compose's substitution, so the literal
        # `${VAR:-default}` string is what a service's environment shows.
        env = _compose_service("worker")["environment"]
        assert env["WOMBLEX_DB_DSN"] == (
            "${WOMBLEX_DB_DSN:-postgresql://womblex:womblex@postgres:5432/womblex}"  # pragma: allowlist secret
        )
        assert env["WOMBLEX_STORE_URI"] == "${WOMBLEX_STORE_URI:-s3://womblex}"
        assert env["WOMBLEX_S3_ENDPOINT"] == "${WOMBLEX_S3_ENDPOINT:-http://minio:9000}"
        assert env["WOMBLEX_S3_ACCESS_KEY_ID"] == "${WOMBLEX_S3_ACCESS_KEY_ID:-minioadmin}"
        assert env["AWS_ACCESS_KEY_ID"] == "${AWS_ACCESS_KEY_ID:-}"

    def test_frontend_builder_stage_runs_a_real_package_script(self) -> None:
        """The builder stage's `npm run build` must name a script `ui/package.json` declares."""
        dockerfile = (REPO_ROOT / "Dockerfile.ui").read_text()
        match = re.search(r"^RUN npm run (\S+)$", dockerfile, re.MULTILINE)
        assert match, "Dockerfile.ui has no `npm run <script>` build step"
        scripts = json.loads((REPO_ROOT / "ui" / "package.json").read_text())["scripts"]
        assert match.group(1) in scripts

    def test_final_stage_copies_the_builder_stages_declared_output(self) -> None:
        """`svelte.config.js`'s adapter output dir must match what the Dockerfile COPYs out."""
        svelte_config = (REPO_ROOT / "ui" / "svelte.config.js").read_text()
        assert "pages: 'build'" in svelte_config
        dockerfile = (REPO_ROOT / "Dockerfile.ui").read_text()
        assert "COPY --from=frontend-builder /app/ui/build /app/ui/build" in dockerfile


class TestFrontendCi:
    """`ui/`'s CI job must call scripts that still exist — otherwise a rename
    here silently stops linting or building the SPA.
    """

    def test_frontend_job_runs_declared_package_scripts(self) -> None:
        workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
        frontend = workflow["jobs"]["frontend"]
        scripts = json.loads((REPO_ROOT / "ui" / "package.json").read_text())["scripts"]
        run_steps = [
            step["run"] for step in frontend["steps"]
            if step.get("working-directory") == "ui" and "run" in step
        ]
        assert run_steps, "no step in the frontend job runs anything in ui/"
        for run in run_steps:
            match = re.match(r"npm (run )?(\S+)$", run.strip())
            assert match, f"unrecognised frontend CI step: {run!r}"
            script = match.group(2)
            if script != "ci":  # `npm ci` installs; it isn't a package.json script
                assert script in scripts, f"ui/package.json has no {script!r} script"


class TestCliUi:
    """Only the paths that return before reaching uvicorn.run()."""

    def test_output_root_and_store_are_mutually_exclusive(self, tmp_path: Path) -> None:
        from womblex.cli import main as cli_main

        with pytest.raises(SystemExit):
            cli_main(["ui", "--output-root", str(tmp_path), "--store", "s3://bucket"])

    def test_no_run_source_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from womblex.cli import main as cli_main

        monkeypatch.delenv("WOMBLEX_UI_OUTPUT_ROOT", raising=False)
        monkeypatch.delenv("WOMBLEX_STORE_URI", raising=False)
        assert cli_main(["ui"]) == 1


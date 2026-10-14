"""Tests for womblex.score — label scoring against parquet output."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.score import (
    build_manifest_index,
    format_report_markdown,
    load_labels,
    reassemble_page_text,
    score_labels,
)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    table = pa.table({
        "source_hash": [r["source_hash"] for r in rows],
        "filename": [r["filename"] for r in rows],
    })
    pq.write_table(table, str(path))


def _write_elements(path: Path, rows: list[dict]) -> None:
    table = pa.table({
        "source_hash": [r["source_hash"] for r in rows],
        "elem_order": [r["elem_order"] for r in rows],
        "kind": [r["kind"] for r in rows],
        "page": [r["page"] for r in rows],
        "text": [r["text"] for r in rows],
    })
    pq.write_table(table, str(path))


def _write_label(labels: Path, stem: str, gt: str, meta: dict) -> None:
    (labels / f"{stem}.gt.md").write_text(gt, encoding="utf-8")
    (labels / f"{stem}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


class TestLoadLabels:
    def test_skips_when_meta_missing(self, tmp_path: Path) -> None:
        (tmp_path / "orphan.gt.md").write_text("hello", encoding="utf-8")
        assert load_labels(tmp_path) == []

    def test_page_from_meta_takes_precedence_over_stem_suffix(self, tmp_path: Path) -> None:
        _write_label(tmp_path, "doc_p7", "body", {"source_file": "doc.pdf", "page": 3})
        entries = load_labels(tmp_path)
        assert len(entries) == 1
        assert entries[0].page == 3

    def test_page_falls_back_to_stem_suffix(self, tmp_path: Path) -> None:
        _write_label(tmp_path, "doc_p5", "body", {"source_file": "doc.pdf"})
        entries = load_labels(tmp_path)
        assert entries[0].page == 5

    def test_source_pdf_alias_accepted(self, tmp_path: Path) -> None:
        _write_label(tmp_path, "doc_p0", "body", {"source_pdf": "legacy.pdf"})
        entries = load_labels(tmp_path)
        assert entries[0].source_file == "legacy.pdf"


class TestBuildManifestIndex:
    def test_missing_directory_raises_on_no_manifests(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_manifest_index(tmp_path)

    def test_indexes_filename_to_hash_and_path(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path / "batch-0001._manifest.parquet", [
            {"source_hash": "h-a", "filename": "a.pdf"},
        ])
        _write_elements(tmp_path / "batch-0001.elements.parquet", [
            {"source_hash": "h-a", "elem_order": 0, "kind": "paragraph",
             "page": 0, "text": "x"},
        ])
        idx = build_manifest_index(tmp_path)
        assert idx["a.pdf"][0] == "h-a"
        assert idx["a.pdf"][1].name == "batch-0001.elements.parquet"


class TestReassemblePageText:
    def test_orders_by_elem_order_and_joins_with_blank_line(self, tmp_path: Path) -> None:
        elements_path = tmp_path / "batch.elements.parquet"
        _write_elements(elements_path, [
            {"source_hash": "h", "elem_order": 2, "kind": "paragraph", "page": 0, "text": "second"},
            {"source_hash": "h", "elem_order": 0, "kind": "heading", "page": 0, "text": "Title"},
            {"source_hash": "h", "elem_order": 1, "kind": "paragraph", "page": 0, "text": "first"},
        ])
        text = reassemble_page_text(elements_path, "h", 0)
        assert text == "Title\n\nfirst\n\nsecond"

    def test_excludes_non_text_kinds_by_default(self, tmp_path: Path) -> None:
        # `table` and `form` carry content in sidecar parquets — their own
        # `text` column is not the source of truth. `sheet_meta` is a
        # spreadsheet marker, not page text.
        elements_path = tmp_path / "batch.elements.parquet"
        _write_elements(elements_path, [
            {"source_hash": "h", "elem_order": 0, "kind": "paragraph", "page": 0, "text": "keep"},
            {"source_hash": "h", "elem_order": 1, "kind": "table", "page": 0, "text": "drop-me"},
            {"source_hash": "h", "elem_order": 2, "kind": "sheet_meta", "page": 0, "text": "drop-too"},
        ])
        assert reassemble_page_text(elements_path, "h", 0) == "keep"

    def test_includes_figure_text_for_per_image_ocr(self, tmp_path: Path) -> None:
        # Hybrid PDFs route scanned cover pages through per-image OCR; the
        # extracted text lands on a `figure` element. Must be included
        # for page-text reassembly to match human GT on those pages.
        elements_path = tmp_path / "batch.elements.parquet"
        _write_elements(elements_path, [
            {"source_hash": "h", "elem_order": 0, "kind": "figure", "page": 0,
             "text": "OCR'd letter body from scanned page"},
        ])
        assert reassemble_page_text(elements_path, "h", 0) == "OCR'd letter body from scanned page"


class TestScoreLabelsIntegration:
    def test_end_to_end_with_grouping(self, tmp_path: Path) -> None:
        shards = tmp_path / "shards"
        labels = tmp_path / "labels"
        shards.mkdir()
        labels.mkdir()

        _write_manifest(shards / "batch-0001._manifest.parquet", [
            {"source_hash": "h-a", "filename": "a.pdf"},
            {"source_hash": "h-b", "filename": "b.pdf"},
        ])
        _write_elements(shards / "batch-0001.elements.parquet", [
            {"source_hash": "h-a", "elem_order": 0, "kind": "paragraph", "page": 0, "text": "hello world"},
            {"source_hash": "h-b", "elem_order": 0, "kind": "paragraph", "page": 0, "text": "completely different"},
        ])
        _write_label(labels, "a_p0", "hello world", {"source_file": "a.pdf", "strategy": "native"})
        _write_label(labels, "b_p0", "hello world", {"source_file": "b.pdf", "strategy": "ocr"})

        rows = score_labels(labels, shards, group_by="strategy")
        rows_by_stem = {r.stem: r for r in rows}
        # a_p0 matches perfectly; b_p0 doesn't match its GT at all.
        assert rows_by_stem["a_p0"].cer == 0.0
        assert rows_by_stem["a_p0"].group == "native"
        assert rows_by_stem["b_p0"].cer > 0.0
        assert rows_by_stem["b_p0"].group == "ocr"

        report = format_report_markdown(rows, group_label="strategy")
        assert "Per-strategy summary" in report
        assert "| native | 1 |" in report
        assert "| ocr | 1 |" in report

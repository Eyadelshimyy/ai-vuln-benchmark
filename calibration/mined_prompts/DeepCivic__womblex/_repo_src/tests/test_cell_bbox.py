"""Table cells carry a normalised bbox where the producer has geometry (contract 1.3)."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from tests._synthetic import AUDIT_PDF
from womblex.ingest.detect import detect_file_type
from womblex.ingest.elements import BBox, Cell, Element
from womblex.ingest.extract import extract_text
from womblex.ingest.views import ExtractionResult
from womblex.store.output import TABLE_CELLS_SCHEMA, _shard_paths, read_table_cells, write_results


def _result(cells: list[Cell]) -> ExtractionResult:
    table = Element(order=0, kind="table", extractor="t", page=1, cells=cells, header_rows=[0])
    return ExtractionResult(elements=[table])


def test_cell_bbox_round_trips(tmp_path):
    box = BBox(x=0.1, y=0.2, width=0.3, height=0.05)
    shard = tmp_path / "batch-0001.parquet"
    write_results(
        [("d", "d.pdf", _result([Cell(0, 0, "a", bbox=box), Cell(0, 1, "b")]))], shard,
    )
    rows = read_table_cells(shard).to_pylist()
    assert rows[0]["bbox"] == pytest.approx({"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.05})
    assert rows[1]["bbox"] is None


def test_pre_1_3_shard_reads_with_null_bbox(tmp_path):
    shard = tmp_path / "batch-0001.parquet"
    write_results([("d", "d.pdf", _result([Cell(0, 0, "a")]))], shard)
    path = _shard_paths(shard)["table_cells"]
    old = pq.read_table(path).drop_columns(["bbox"])
    pq.write_table(old, path)
    table = read_table_cells(shard)
    assert table.schema.equals(TABLE_CELLS_SCHEMA)
    assert table.column("bbox").null_count == table.num_rows


def test_native_pdf_table_cells_are_located():
    result = extract_text(AUDIT_PDF, detect_file_type(AUDIT_PDF))[0]
    cells = [c for e in result.elements if e.kind == "table" for c in e.cells or []]
    assert cells
    boxed = [c for c in cells if c.bbox is not None]
    assert boxed
    for c in boxed:
        b = c.bbox
        assert 0.0 <= b.x <= 1.0 and 0.0 <= b.y <= 1.0
        assert b.width > 0 and b.height > 0

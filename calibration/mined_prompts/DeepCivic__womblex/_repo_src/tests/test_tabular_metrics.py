"""Tests for tabular extraction accuracy metrics."""


from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests._synthetic import REGISTER_CSV
from womblex.utils.tabular_metrics import (
    data_integrity,
    key_column_preservation,
    schema_conformance,
    structural_fidelity,
)

# ── Structural Fidelity ────────────────────────────────────────────────────



class TestStructuralFidelity:

    def test_identical_frames(self) -> None:

        df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})

        result = structural_fidelity(df, df.copy())
        assert result.passed

        assert result.source_rows == 2

        assert result.source_cols == 2


    def test_row_count_mismatch(self) -> None:

        src = pd.DataFrame({"a": ["1", "2", "3"]})

        ext = pd.DataFrame({"a": ["1", "2"]})

        result = structural_fidelity(src, ext)
        assert not result.passed

        assert result.source_rows == 3

        assert result.extracted_rows == 2


    def test_missing_column(self) -> None:

        src = pd.DataFrame({"a": ["1"], "b": ["2"]})

        ext = pd.DataFrame({"a": ["1"]})

        result = structural_fidelity(src, ext)
        assert not result.passed

        assert result.missing_columns == ["b"]


    def test_extra_column(self) -> None:

        src = pd.DataFrame({"a": ["1"]})

        ext = pd.DataFrame({"a": ["1"], "c": ["3"]})

        result = structural_fidelity(src, ext)
        assert not result.passed

        assert result.extra_columns == ["c"]


    def test_empty_frames(self) -> None:

        df = pd.DataFrame()

        result = structural_fidelity(df, df.copy())
        assert result.passed



# ── Data Integrity ──────────────────────────────────────────────────────────



class TestDataIntegrity:

    def test_perfect_match(self) -> None:

        df = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})

        result = data_integrity(df, df.copy())
        assert result.passed

        assert result.score == 1.0

        assert result.mismatched_cells == 0


    def test_single_cell_mismatch(self) -> None:

        src = pd.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})

        ext = pd.DataFrame({"a": ["1", "2"], "b": ["x", "WRONG"]})

        result = data_integrity(src, ext)
        assert not result.passed

        assert result.mismatched_cells == 1

        assert result.score == pytest.approx(3 / 4)

        assert result.mismatches[0] == (1, "b", "y", "WRONG")


    def test_nan_normalised_to_empty(self) -> None:

        src = pd.DataFrame({"a": [None, "2"]})

        ext = pd.DataFrame({"a": ["", "2"]})

        result = data_integrity(src, ext)
        assert result.passed


    def test_no_shared_columns(self) -> None:

        src = pd.DataFrame({"a": ["1"]})

        ext = pd.DataFrame({"b": ["1"]})

        result = data_integrity(src, ext)
        assert not result.passed

        assert result.total_cells == 0


    def test_empty_frames(self) -> None:

        df = pd.DataFrame({"a": []})

        result = data_integrity(df, df.copy())
        assert result.passed

        assert result.total_cells == 0


    def test_max_mismatches_capped(self) -> None:

        src = pd.DataFrame({"a": [str(i) for i in range(50)]})

        ext = pd.DataFrame({"a": ["WRONG"] * 50})

        result = data_integrity(src, ext, max_mismatches=5)

        assert len(result.mismatches) == 5

        assert result.mismatched_cells == 50



# ── Key Column Preservation ─────────────────────────────────────────────────



class TestKeyColumnPreservation:

    def test_all_keys_preserved(self) -> None:

        df = pd.DataFrame({"id": ["A", "B", "C"], "val": ["1", "2", "3"]})

        result = key_column_preservation(df, df.copy(), "id")
        assert result.passed

        assert result.source_unique == 3

        assert result.extracted_unique == 3


    def test_missing_key(self) -> None:

        src = pd.DataFrame({"id": ["A", "B", "C"]})

        ext = pd.DataFrame({"id": ["A", "B"]})

        result = key_column_preservation(src, ext, "id")
        assert not result.passed

        assert result.missing_keys == ["C"]


    def test_duplicate_key(self) -> None:

        src = pd.DataFrame({"id": ["A", "B"]})

        ext = pd.DataFrame({"id": ["A", "B", "B"]})

        result = key_column_preservation(src, ext, "id")
        assert not result.passed

        assert result.duplicate_keys == ["B"]


    def test_whitespace_stripped(self) -> None:

        src = pd.DataFrame({"id": ["A ", " B"]})

        ext = pd.DataFrame({"id": ["A", "B"]})

        result = key_column_preservation(src, ext, "id")
        assert result.passed



# ── Schema Conformance ──────────────────────────────────────────────────────



class TestSchemaConformance:

    def test_matching_schema(self, tmp_path: Path) -> None:

        schema = pa.schema([("a", pa.string()), ("b", pa.int64())])

        table = pa.table({"a": ["x"], "b": [1]}, schema=schema)

        path = tmp_path / "test.parquet"

        pq.write_table(table, str(path))

        assert schema_conformance(path, schema)


    def test_mismatched_schema(self, tmp_path: Path) -> None:

        schema = pa.schema([("a", pa.string()), ("b", pa.int64())])

        wrong_schema = pa.schema([("a", pa.string()), ("c", pa.float64())])

        table = pa.table({"a": ["x"], "b": [1]}, schema=schema)

        path = tmp_path / "test.parquet"

        pq.write_table(table, str(path))

        assert not schema_conformance(path, wrong_schema)



# ── Integration: CSV Fixture ────────────────────────────────────────────────


_CSV_FILE = REGISTER_CSV



class TestCSVFixtureAccuracy:

    """Round-trip the synthetic register CSV through the spreadsheet extractor."""


    @pytest.fixture()

    def source_df(self) -> pd.DataFrame:

        return pd.read_csv(str(_CSV_FILE), dtype=str, keep_default_na=False)


    @pytest.fixture()

    def extracted_df(self, source_df: pd.DataFrame) -> pd.DataFrame:

        """Reconstruct a DataFrame from SpreadsheetExtractor's element stream.

        The new shape: one ExtractionResult per workbook, with
        ``kind='sheet_cell'`` elements giving (sheet, row, col, value).
        Row 0 is the header row; rows 1..n are data. We rebuild the
        DataFrame by grouping cells back into (row, col) form.
        """

        from womblex.ingest.detect import DetectionConfig, detect_file_type
        from womblex.ingest.extract import extract_text


        profile = detect_file_type(_CSV_FILE, DetectionConfig())

        results = extract_text(_CSV_FILE, profile)

        assert len(results) == 1, f"expected single result for CSV, got {len(results)}"
        result = results[0]
        assert result.error is None

        cells = [e for e in result.elements if e.kind == "sheet_cell"]
        assert cells, "no cell elements produced"

        max_col = max(c.col for c in cells)
        rows_by_idx: dict[int, list[str]] = {}
        for c in cells:
            row = rows_by_idx.setdefault(c.row, [""] * (max_col + 1))
            row[c.col] = c.value or ""

        sorted_rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
        headers = sorted_rows[0]
        data = sorted_rows[1:]
        return pd.DataFrame(data, columns=headers)


    def test_structural_fidelity(self, source_df: pd.DataFrame, extracted_df: pd.DataFrame) -> None:

        result = structural_fidelity(source_df, extracted_df)
        assert result.passed, (

            f"Structural mismatch: src={result.source_rows}x{result.source_cols} "

            f"ext={result.extracted_rows}x{result.extracted_cols} "

            f"missing={result.missing_columns} extra={result.extra_columns}"
        )


    def test_data_integrity(self, source_df: pd.DataFrame, extracted_df: pd.DataFrame) -> None:

        result = data_integrity(source_df, extracted_df)
        assert result.passed, (

            f"Data integrity: {result.mismatched_cells}/{result.total_cells} cells differ. "

            f"Score={result.score:.4f}. First mismatches: {result.mismatches[:5]}"
        )


    def test_key_column_preservation(self, source_df: pd.DataFrame, extracted_df: pd.DataFrame) -> None:

        result = key_column_preservation(source_df, extracted_df, "SightingID")
        assert result.passed, (

            f"Key column: {result.source_unique} source, {result.extracted_unique} extracted. "

            f"Missing: {result.missing_keys[:5]}. Duplicates: {result.duplicate_keys[:5]}"
        )


    def test_schema_conformance(self, tmp_path: Path) -> None:
        # The new output writes four sibling shards (elements + table_cells
        # + form_fields + manifest). Schema conformance checks each shard
        # matches its canonical schema.

        from womblex.config import ChunkingConfig, DatasetConfig, PathsConfig, WomblexConfig
        from womblex.operations import BatchResult, run_extraction, write_batch_parquet
        from womblex.store.output import (
            ELEMENT_SCHEMA,
            FORM_FIELDS_SCHEMA,
            MANIFEST_SCHEMA,
            TABLE_CELLS_SCHEMA,
            _shard_paths,
        )


        config = WomblexConfig(
            dataset=DatasetConfig(name="schema_test"),
            paths=PathsConfig(

                input_root=_CSV_FILE.parent,

                output_root=tmp_path / "out",

                checkpoint_dir=tmp_path / "ckpt",
            ),

            chunking=ChunkingConfig(enabled=False),
        )

        results = run_extraction([_CSV_FILE], config)

        batch = BatchResult(results=results)

        out = tmp_path / "docs.parquet"

        write_batch_parquet(batch, out)

        paths = _shard_paths(out)
        assert schema_conformance(paths["elements"], ELEMENT_SCHEMA)
        assert schema_conformance(paths["table_cells"], TABLE_CELLS_SCHEMA)
        assert schema_conformance(paths["form_fields"], FORM_FIELDS_SCHEMA)
        assert schema_conformance(paths["manifest"], MANIFEST_SCHEMA)


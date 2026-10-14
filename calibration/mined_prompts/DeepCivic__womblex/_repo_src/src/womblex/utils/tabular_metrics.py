"""Tabular extraction accuracy metrics.

Validates that spreadsheet extraction preserves source structure and data
with zero semantic mutation. All metrics compare a source DataFrame (ground
truth read directly from CSV/XLSX) against the extraction output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class StructuralFidelityResult:
    """Result of structural fidelity check."""

    passed: bool
    source_rows: int
    source_cols: int
    extracted_rows: int
    extracted_cols: int
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)


def structural_fidelity(
    source: pd.DataFrame,
    extracted: pd.DataFrame,
) -> StructuralFidelityResult:
    """Validate row/column count and column names match between source and extraction.

    Args:
        source: Ground truth DataFrame read directly from CSV/XLSX.
        extracted: DataFrame reconstructed from extraction output.

    Returns:
        StructuralFidelityResult with pass/fail and mismatch details.
    """
    src_cols = set(source.columns)
    ext_cols = set(extracted.columns)

    missing = sorted(src_cols - ext_cols)
    extra = sorted(ext_cols - src_cols)

    passed = (
        len(source) == len(extracted)
        and len(source.columns) == len(extracted.columns)
        and not missing
        and not extra
    )

    return StructuralFidelityResult(
        passed=passed,
        source_rows=len(source),
        source_cols=len(source.columns),
        extracted_rows=len(extracted),
        extracted_cols=len(extracted.columns),
        missing_columns=missing,
        extra_columns=extra,
    )


@dataclass
class DataIntegrityResult:
    """Result of cell-level data integrity check."""

    passed: bool
    total_cells: int
    mismatched_cells: int
    score: float  # 1.0 = perfect, 0.0 = all wrong
    mismatches: list[tuple[int, str, str, str]] = field(default_factory=list)
    """First N mismatches as (row_idx, column, source_val, extracted_val)."""


def data_integrity(
    source: pd.DataFrame,
    extracted: pd.DataFrame,
    *,
    max_mismatches: int = 20,
) -> DataIntegrityResult:
    """Cell-level exact match between source and extracted DataFrames.

    Both DataFrames are compared as strings. Empty strings and NaN are
    normalised to ``""`` before comparison.

    Args:
        source: Ground truth DataFrame.
        extracted: DataFrame from extraction output.
        max_mismatches: Max number of individual mismatches to record.

    Returns:
        DataIntegrityResult with score and mismatch details.
    """
    # Align columns — only compare shared columns.
    shared_cols = sorted(set(source.columns) & set(extracted.columns))
    if not shared_cols:
        return DataIntegrityResult(
            passed=False, total_cells=0, mismatched_cells=0,
            score=0.0, mismatches=[],
        )

    src = source[shared_cols].fillna("").astype(str)
    ext = extracted[shared_cols].fillna("").astype(str)

    # Truncate to shorter length if row counts differ.
    min_rows = min(len(src), len(ext))
    src = src.iloc[:min_rows]
    ext = ext.iloc[:min_rows]

    total = min_rows * len(shared_cols)
    if total == 0:
        return DataIntegrityResult(
            passed=True, total_cells=0, mismatched_cells=0,
            score=1.0, mismatches=[],
        )

    diff_mask = src.values != ext.values
    mismatched = int(diff_mask.sum())

    mismatches: list[tuple[int, str, str, str]] = []
    if mismatched > 0:
        rows, cols = diff_mask.nonzero()
        for i in range(min(len(rows), max_mismatches)):
            r, c = int(rows[i]), int(cols[i])
            mismatches.append((
                r,
                shared_cols[c],
                str(src.iloc[r, c]),
                str(ext.iloc[r, c]),
            ))

    score = 1.0 - (mismatched / total)

    return DataIntegrityResult(
        passed=mismatched == 0,
        total_cells=total,
        mismatched_cells=mismatched,
        score=score,
        mismatches=mismatches,
    )


@dataclass
class KeyColumnResult:
    """Result of key column preservation check."""

    passed: bool
    key_column: str
    source_unique: int
    extracted_unique: int
    missing_keys: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)


def key_column_preservation(
    source: pd.DataFrame,
    extracted: pd.DataFrame,
    key_column: str,
) -> KeyColumnResult:
    """Verify unique IDs are 100% preserved without duplication.

    Args:
        source: Ground truth DataFrame.
        extracted: DataFrame from extraction output.
        key_column: Column name containing unique identifiers.

    Returns:
        KeyColumnResult with missing/duplicate key details.
    """
    src_keys = set(source[key_column].astype(str).str.strip())
    ext_keys = extracted[key_column].astype(str).str.strip()

    ext_unique = set(ext_keys)
    missing = sorted(src_keys - ext_unique)

    # Duplicates: keys that appear more than once in extracted.
    counts = ext_keys.value_counts()
    duplicates = sorted(counts[counts > 1].index.tolist())

    return KeyColumnResult(
        passed=not missing and not duplicates,
        key_column=key_column,
        source_unique=len(src_keys),
        extracted_unique=len(ext_unique),
        missing_keys=missing[:20],
        duplicate_keys=duplicates[:20],
    )


def schema_conformance(
    parquet_path: Path,
    expected_schema: pa.Schema,
) -> bool:
    """Check that a Parquet file's schema matches the expected schema exactly.

    Args:
        parquet_path: Path to the Parquet file.
        expected_schema: Expected PyArrow schema.

    Returns:
        True if schemas match.
    """
    actual = pq.read_schema(str(parquet_path))
    return bool(actual.equals(expected_schema))

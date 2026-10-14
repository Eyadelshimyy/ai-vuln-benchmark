"""Sample-based column schema inference for CSV, Excel, Parquet, and NDJSON.

Returns per-column inferred type (integer / float / boolean / date / datetime /
string / empty), null counts, uniqueness, and value extremes. Reads up to a
sample cap so very large files stay cheap to profile.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_ROWS = 10_000

_INT_RE = re.compile(r"^-?\d+$")
_BOOL_TRUE = frozenset({"true", "yes", "y", "t"})
_BOOL_FALSE = frozenset({"false", "no", "n", "f"})
_BOOL_VALUES = _BOOL_TRUE | _BOOL_FALSE

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
)
_DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
)

# Per-column inference cost is O(N); cap the inspected rows for type detection.
_INFER_CAP = 1000


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str  # integer | float | boolean | date | datetime | string | empty
    null_count: int
    null_fraction: float
    unique_count: int
    is_unique: bool
    is_constant: bool
    min_value: str | None
    max_value: str | None
    max_length: int | None
    sample_values: list[str] = field(default_factory=list)


@dataclass
class TableProfile:
    source: str
    sheet_name: str | None
    row_count: int
    column_count: int
    sampled_rows: int
    columns: list[ColumnProfile]


def profile_file(
    path: Path | str,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
) -> list[TableProfile]:
    """Profile every table in *path*.

    Args:
        path: CSV, XLSX, XLS, Parquet, NDJSON, or JSONL file.
        sample_rows: Maximum rows to load for inference. Use ``0`` for all rows.

    Returns:
        One TableProfile per sheet (Excel) or one for single-table formats.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [_profile_csv(path, sample_rows)]
    if suffix in (".xlsx", ".xls"):
        return _profile_excel(path, sample_rows)
    if suffix == ".parquet":
        return [_profile_parquet(path, sample_rows)]
    if suffix in (".ndjson", ".jsonl"):
        return [_profile_ndjson(path, sample_rows)]
    raise ValueError(f"Unsupported file type for profiling: {path.suffix}")


def profile_dataframe(
    df: pd.DataFrame,
    *,
    source: str = "<dataframe>",
    sheet_name: str | None = None,
    total_rows: int | None = None,
) -> TableProfile:
    """Infer column-level schema for a pandas DataFrame.

    Treats empty strings as null (consistent with ``dtype=str, keep_default_na=False``).

    Args:
        df: DataFrame to profile.
        source: Origin label for the result (filename or descriptor).
        sheet_name: Sheet identifier for multi-sheet sources.
        total_rows: Authoritative row count if *df* is a sample; defaults to ``len(df)``.
    """
    cols = [_profile_column(name, df[name]) for name in df.columns]
    return TableProfile(
        source=source,
        sheet_name=sheet_name,
        row_count=total_rows if total_rows is not None else len(df),
        column_count=len(df.columns),
        sampled_rows=len(df),
        columns=cols,
    )


def _profile_csv(path: Path, sample_rows: int) -> TableProfile:
    import pandas as pd

    nrows = sample_rows if sample_rows > 0 else None
    df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=nrows)
    total = _count_csv_rows(path) if nrows is not None else len(df)
    return profile_dataframe(df, source=str(path), total_rows=total)


def _profile_excel(path: Path, sample_rows: int) -> list[TableProfile]:
    import pandas as pd

    xl = pd.ExcelFile(str(path))
    profiles: list[TableProfile] = []
    nrows = sample_rows if sample_rows > 0 else None
    for name in xl.sheet_names:
        df = xl.parse(name, dtype=str, keep_default_na=False, nrows=nrows)
        # pandas/openpyxl can't cheaply report total rows separately from a load;
        # if the sample filled, re-read the sheet to get the true count.
        if nrows is not None and len(df) >= nrows:
            full = xl.parse(name, dtype=str, keep_default_na=False)
            total = len(full)
        else:
            total = len(df)
        profiles.append(
            profile_dataframe(df, source=str(path), sheet_name=str(name), total_rows=total)
        )
    return profiles


def _profile_parquet(path: Path, sample_rows: int) -> TableProfile:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    total_rows = pf.metadata.num_rows
    batch_size = sample_rows if sample_rows > 0 else max(total_rows, 1)
    batch = next(pf.iter_batches(batch_size=batch_size))
    df = batch.to_pandas()

    # Stringify so the shared column profiler treats nulls + empties uniformly.
    str_df = df.astype("string").fillna("").astype(str)
    profile = profile_dataframe(
        str_df,
        source=str(path),
        total_rows=total_rows,
    )

    # Parquet has authoritative types; override the heuristic guess.
    type_map = {f.name: _arrow_to_type(f.type) for f in pf.schema_arrow}
    for col in profile.columns:
        native = type_map.get(col.name)
        if native is not None and col.inferred_type != "empty":
            col.inferred_type = native

    return profile


def _profile_ndjson(path: Path, sample_rows: int) -> TableProfile:
    import pandas as pd

    nrows = sample_rows if sample_rows > 0 else None
    df = pd.read_json(path, lines=True, nrows=nrows)
    str_df = df.astype("string").fillna("").astype(str)
    if nrows is not None and len(df) >= nrows:
        total = _count_csv_rows(path) + 1  # NDJSON has no header line
    else:
        total = len(df)
    return profile_dataframe(str_df, source=str(path), total_rows=total)


def _arrow_to_type(arrow_type: pa.DataType) -> str:
    import pyarrow as pa

    if pa.types.is_integer(arrow_type):
        return "integer"
    if pa.types.is_floating(arrow_type):
        return "float"
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "datetime"
    return "string"


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV (excluding the header) without loading values."""
    with path.open("rb") as f:
        total = sum(1 for _ in f)
    return max(total - 1, 0)


def _profile_column(name: str, series: pd.Series) -> ColumnProfile:
    str_series = series.astype(str)
    non_empty_mask = str_series.str.len() > 0
    non_empty = str_series[non_empty_mask]
    null_count = int(len(series) - len(non_empty))
    total = len(series)

    if len(non_empty) == 0:
        return ColumnProfile(
            name=str(name),
            inferred_type="empty",
            null_count=null_count,
            null_fraction=1.0 if total else 0.0,
            unique_count=0,
            is_unique=False,
            is_constant=True,
            min_value=None,
            max_value=None,
            max_length=None,
        )

    unique = non_empty.unique()
    unique_count = len(unique)
    inferred = _infer_type(non_empty)
    min_v, max_v = _min_max(non_empty, inferred)
    max_length = int(non_empty.str.len().max()) if inferred == "string" else None

    return ColumnProfile(
        name=str(name),
        inferred_type=inferred,
        null_count=null_count,
        null_fraction=null_count / total if total else 0.0,
        unique_count=unique_count,
        is_unique=unique_count == len(non_empty) and null_count == 0,
        is_constant=unique_count == 1,
        min_value=min_v,
        max_value=max_v,
        max_length=max_length,
        sample_values=[str(v) for v in unique[:5]],
    )


def _infer_type(values: pd.Series) -> str:
    sample = values.head(_INFER_CAP)

    if all(_INT_RE.match(v) for v in sample):
        return "integer"
    if all(_is_floatlike(v) for v in sample):
        return "float"
    if all(v.lower() in _BOOL_VALUES for v in sample):
        return "boolean"
    # Check date before datetime: ``datetime.fromisoformat`` accepts bare dates,
    # which would otherwise tag pure-date columns as datetimes.
    if all(_try_parse(v, _DATE_FORMATS, allow_iso=False) for v in sample):
        return "date"
    if all(_try_parse(v, _DATETIME_FORMATS, allow_iso=True) for v in sample):
        return "datetime"
    return "string"


def _is_floatlike(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    # Reject NaN/Infinity coming in as text — they're typically string sentinels.
    lowered = value.lower().lstrip("+-")
    return lowered not in {"nan", "inf", "infinity"}


def _try_parse(value: str, formats: tuple[str, ...], *, allow_iso: bool) -> bool:
    for fmt in formats:
        try:
            # Parsing is the *test* here — the datetime is discarded, so its
            # tz-awareness is irrelevant. Requiring %z would narrow which
            # formats are recognised, which is the opposite of the intent.
            datetime.strptime(value, fmt)  # noqa: DTZ007
            return True
        except ValueError:
            continue
    if allow_iso:
        try:
            datetime.fromisoformat(value)  # handles a trailing `Z`
            return True
        except ValueError:
            return False
    return False


def _min_max(values: pd.Series, inferred: str) -> tuple[str | None, str | None]:
    if inferred == "integer":
        nums = values.astype(int)
        return str(int(nums.min())), str(int(nums.max()))
    if inferred == "float":
        nums = values.astype(float)
        return f"{float(nums.min()):.6g}", f"{float(nums.max()):.6g}"
    if inferred == "boolean":
        return None, None
    # date / datetime / string: lexicographic min/max on the string form is meaningful
    # for ISO-style dates and stable for strings.
    return str(values.min()), str(values.max())

"""Business hash (CONTRACTS.md §9.1): python side over decoded source values, T-SQL side over target columns."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

from .convert import Timestamp12
from .typemap import ColumnSpec, Kind

EIGHT = Decimal("0.00000001")

HashValue = str | int | Decimal | bytes | Timestamp12 | datetime | date | None


def render(value: str | int | Decimal, kind: Kind, is_key: bool) -> str:
    """The contract's reference renderer (§9.1)."""
    if kind == "char":
        return value if is_key else value.rstrip(" ")  # type: ignore[union-attr]
    if kind == "int":
        return str(int(value))
    if kind == "decimal":
        d = Decimal(value).quantize(EIGHT)
        if d == 0:
            d = abs(d)
        return format(d, "f")
    return value  # type: ignore[return-value]


def business_hash(cols: Sequence[tuple[str | int | Decimal, Kind, bool]]) -> bytes:
    return hashlib.sha256("|".join(render(*c) for c in cols).encode("utf-8")).digest()


def _as_text(value: HashValue, spec: ColumnSpec) -> str | int | Decimal:
    if value is None:
        return ""
    if isinstance(value, Timestamp12):
        return value.text
    if isinstance(value, datetime):
        return Timestamp12.from_target(value.strftime("%Y-%m-%d %H:%M:%S.%f0"), 0).text
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, bytes):
        return value.hex().upper()
    return value


def _render_kind(spec: ColumnSpec) -> Kind:
    return spec.kind if spec.kind in ("char", "int", "decimal") else "char"


def render_source_column(value: HashValue, spec: ColumnSpec) -> str:
    """Rendering of one *source* value: CHAR keys keep padding, value_map applied, non-key CHAR trimmed."""
    text = _as_text(value, spec)
    if spec.kind == "char" and isinstance(text, str) and spec.value_map:
        key = text.rstrip(" ")
        if key in spec.value_map:
            text = spec.value_map[key].ljust(len(text)) if not spec.trim else spec.value_map[key]
    return render(text, _render_kind(spec), spec.is_key)


def render_target_column(row: dict[str, HashValue], spec: ColumnSpec) -> str:
    """Rendering of one *target* column as the T-SQL expression does it (row keyed by target column)."""
    v = row[spec.name]
    if spec.kind == "timestamp12" and isinstance(v, datetime):
        tail = row.get(f"{spec.name}_NANOS_TAIL")
        assert isinstance(tail, int)
        text: str | int | Decimal = Timestamp12.from_target(v.strftime("%Y-%m-%d %H:%M:%S.%f") + "0", tail).text
    else:
        text = _as_text(v, spec)
    return render(text, _render_kind(spec), spec.is_key)


def _join(parts: Sequence[str]) -> bytes:
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def source_hash(values: dict[str, HashValue], hash_columns: Sequence[str], specs: list[ColumnSpec]) -> bytes:
    by_name = {s.name: s for s in specs}
    return _join([render_source_column(values[n], by_name[n]) for n in hash_columns])


def target_hash(row: dict[str, HashValue], hash_columns: Sequence[str], specs: list[ColumnSpec]) -> bytes:
    """Reference implementation of what the T-SQL expression computes, over *target* column values."""
    by_name = {s.name: s for s in specs}
    return _join([render_target_column(row, by_name[n]) for n in hash_columns])


def hash_expression(provider: str, hash_columns: Sequence[str], specs: list[ColumnSpec]) -> str:
    """The target-side SQL expression for `provider` that reproduces `target_hash` over a stg.* row."""
    if provider == "azuresql":
        return tsql_hash_expression(hash_columns, specs)
    if provider == "postgresql":
        return pg_hash_expression(hash_columns, specs)
    raise ValueError(f"no hash expression for target provider {provider!r}")


def pg_hash_expression(hash_columns: Sequence[str], specs: list[ColumnSpec]) -> str:
    """PostgreSQL expression producing the same BYTEA(32) as `business_hash` for one target row.

    CHAR(n) keys are re-padded to n because a bpchar loses its padding when cast to text; a TIMESTAMP(12)
    is TIMESTAMP(6) plus the six digits of <col>_NANOS_TAIL (CONTRACTS.md §6.0).
    """
    by_name = {s.name: s for s in specs}
    parts: list[str] = []
    for name in hash_columns:
        spec = by_name[name]
        col = f'"{name}"'
        if spec.kind == "char":
            if spec.is_key and spec.target_length and not spec.trim:
                expr = f"COALESCE(RPAD({col}::text, {int(spec.target_length)}), '')"
            elif spec.is_key:
                expr = f"COALESCE({col}::text, '')"
            else:
                expr = f"COALESCE(RTRIM({col}::text), '')"
        elif spec.kind == "int":
            expr = f"COALESCE({col}::text, '')"
        elif spec.kind == "decimal":
            expr = f"COALESCE(({col}::numeric(38,8))::text, '')"
        elif spec.kind == "timestamp12":
            expr = (
                f"COALESCE(TO_CHAR({col}, 'YYYY-MM-DD-HH24.MI.SS.US') || LPAD(\"{name}_NANOS_TAIL\"::text, 6, '0'), '')"
            )
        elif spec.kind == "date8":
            expr = f"COALESCE(TO_CHAR({col}, 'YYYYMMDD'), '')"
        else:
            expr = f"COALESCE(UPPER(ENCODE({col}, 'hex')), '')"
        parts.append(expr)
    joined = " || '|' || ".join(parts) if len(parts) > 1 else parts[0]
    return f"sha256(convert_to({joined}, 'UTF8'))"


def tsql_hash_expression(hash_columns: Sequence[str], specs: list[ColumnSpec]) -> str:
    """T-SQL expression producing the same VARBINARY(32) as `business_hash` for one target row."""
    by_name = {s.name: s for s in specs}
    parts: list[str] = []
    for name in hash_columns:
        spec = by_name[name]
        col = f"[{name}]"
        if spec.kind == "char":
            expr = f"ISNULL({col}, N'')" if spec.is_key else f"ISNULL(RTRIM({col}), N'')"
            # RTRIM on NCHAR strips the padding; keys stay as stored (NCHAR keeps padding, NVARCHAR is trimmed)
        elif spec.kind == "int":
            expr = f"CONVERT(NVARCHAR(24), {col})"
        elif spec.kind == "decimal":
            expr = f"CONVERT(NVARCHAR(48), CAST({col} AS DECIMAL(38,8)))"
        elif spec.kind == "timestamp12":
            expr = (
                f"CONCAT(CONVERT(NCHAR(10), {col}, 23), N'-', "
                f"REPLACE(CONVERT(NCHAR(8), {col}, 108), N':', N'.'), N'.', "
                f"RIGHT(CONVERT(NVARCHAR(27), {col}, 121), 7), "
                f"RIGHT(CONCAT(N'00000', [{name}_NANOS_TAIL]), 5))"
            )
        elif spec.kind == "date8":
            expr = f"CONVERT(NVARCHAR(8), {col}, 112)"
        else:
            expr = f"UPPER(CONVERT(NVARCHAR(MAX), {col}, 2))"
        parts.append(expr)
    joined = ", N'|', ".join(parts) if len(parts) > 1 else parts[0]
    # UTF-8 bytes of the joined text (a BIN2 UTF-8 collation), exactly what the Python side hashes.
    return (
        "HASHBYTES('SHA2_256', CONVERT(VARBINARY(MAX), "
        f"CAST(CONCAT({joined}) AS VARCHAR(MAX)) COLLATE Latin1_General_100_BIN2_UTF8))"
    )

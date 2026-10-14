"""Align English Wyscout exports to the Portuguese column names used by the engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SERIE_A_TEMPLATE = ROOT / "Serie A 26.xlsx"


def portuguese_column_template() -> list[str]:
    xl = pd.ExcelFile(SERIE_A_TEMPLATE)
    sheet = "Search results (500)"
    if sheet not in xl.sheet_names:
        raise FileNotFoundError(f"Sheet {sheet!r} missing in {SERIE_A_TEMPLATE}")
    return list(pd.read_excel(SERIE_A_TEMPLATE, sheet_name=sheet, nrows=0).columns)


def rename_english_wyscout_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns when the export uses English headers (same order as PT template)."""
    pt_cols = portuguese_column_template()
    en_cols = list(df.columns)
    if en_cols == pt_cols:
        return df
    if len(en_cols) != len(pt_cols):
        raise ValueError(
            f"Wyscout column count mismatch: got {len(en_cols)}, expected {len(pt_cols)}"
        )
    if en_cols[0] in ("Player", "Jogador") and en_cols[0] != pt_cols[0]:
        rename = dict(zip(en_cols, pt_cols, strict=True))
        return df.rename(columns=rename)
    return df

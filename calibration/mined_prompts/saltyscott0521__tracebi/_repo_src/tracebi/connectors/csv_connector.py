"""CSV / Excel file connector."""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd

from tracebi.connectors.base import BaseConnector


class CSVConnector(BaseConnector):
    """
    Load CSV or Excel files from a directory.

    Usage:
        connector = CSVConnector("lookups", directory="data/")
        model.add_connector(connector)
        model.add_table("regions", connector="lookups", source="regions.csv")

    Args:
        name:      Logical name used to reference this connector in a DataModel.
        directory: Base directory containing the files. Defaults to current dir.
        encoding:  File encoding for CSV files (default ``utf-8``).
    """

    def __init__(
        self,
        name: str,
        directory: str = ".",
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(name)
        self.directory = directory
        self.encoding = encoding

    def describe(self) -> dict:
        return {**super().describe(), "directory": self.directory}

    def column_schema(self, source: str) -> list[dict[str, str]]:
        """Header and dtypes from a zero-row read. The body is not loaded."""
        path = os.path.join(self.directory, source)
        ext = os.path.splitext(source)[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(path, nrows=0)
        else:
            df = pd.read_csv(path, encoding=self.encoding, nrows=0)
        return [{"name": str(c), "dtype": str(df.dtypes[c])} for c in df.columns]

    def connect(self) -> None:
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(
                f"CSVConnector '{self.name}': directory not found: {self.directory}"
            )

    def load(
        self,
        source: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        path = os.path.join(self.directory, source)
        ext = os.path.splitext(source)[1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(
                path,
                encoding=self.encoding,
                usecols=columns if columns else None,
            )
        return self._apply_pandas_pushdown(df, filter, columns)

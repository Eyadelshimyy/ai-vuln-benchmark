"""In-memory connector for testing, demos, and notebook workflows."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from tracebi.connectors.base import BaseConnector


class MemoryConnector(BaseConnector):
    """
    In-memory connector backed by a dict of DataFrames.

    Useful for unit tests, Jupyter notebooks, and demos where
    you want lineage tracking without reading from disk or a database.

    Usage:
        import pandas as pd
        from tracebi import MemoryConnector, DataModel

        orders_df = pd.DataFrame({...})
        customers_df = pd.DataFrame({...})

        connector = MemoryConnector("mem", tables={
            "orders":    orders_df,
            "customers": customers_df,
        })

        model = DataModel("Demo")
        model.add_connector(connector)
        model.add_table("orders",    connector="mem", source="orders")
        model.add_table("customers", connector="mem", source="customers")

    Args:
        name:   Logical name used to reference this connector in a DataModel.
        tables: Dict mapping source names to DataFrames.
    """

    def __init__(self, name: str, tables: dict[str, pd.DataFrame]) -> None:
        super().__init__(name)
        self._tables: dict[str, pd.DataFrame] = {k: v.copy() for k, v in tables.items()}

    def connect(self) -> None:
        pass  # nothing to connect

    def describe(self) -> dict:
        return {**super().describe(), "tables": list(self._tables.keys())}

    def column_schema(self, source: str) -> list[dict[str, str]]:
        """Names and dtypes of the frame already held in memory."""
        if source not in self._tables:
            available = list(self._tables.keys())
            raise KeyError(
                f"MemoryConnector '{self.name}': source '{source}' not found. "
                f"Available: {available}"
            )
        df = self._tables[source]
        return [{"name": str(c), "dtype": str(df.dtypes[c])} for c in df.columns]

    def load(
        self,
        source: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if source not in self._tables:
            available = list(self._tables.keys())
            raise KeyError(
                f"MemoryConnector '{self.name}': source '{source}' not found. "
                f"Available: {available}"
            )
        df = self._tables[source].copy()
        return self._apply_pandas_pushdown(df, filter, columns)

    def write(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "replace",
    ) -> None:
        """Write *df* into the in-memory table store."""
        if if_exists == "fail" and table in self._tables:
            raise ValueError(f"Table '{table}' already exists in MemoryConnector '{self.name}'.")
        if if_exists == "append" and table in self._tables:
            self._tables[table] = pd.concat(
                [self._tables[table], df], ignore_index=True
            )
        else:
            self._tables[table] = df.copy()

    def add_table(self, name: str, df: pd.DataFrame) -> "MemoryConnector":
        """Register an additional DataFrame at runtime."""
        self._tables[name] = df.copy()
        return self

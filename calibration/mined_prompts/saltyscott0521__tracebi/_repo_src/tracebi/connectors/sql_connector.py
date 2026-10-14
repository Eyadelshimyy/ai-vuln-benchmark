"""SQLAlchemy-backed relational database connector."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from tracebi.connectors.base import BaseConnector


class SQLConnector(BaseConnector):
    """
    Load tables or run queries against any SQLAlchemy-supported database.

    Push-down: ``filter`` and ``columns`` are applied via SQL WHERE / SELECT
    clauses when *source* is a plain table name. When *source* is a raw
    SELECT query, push-down falls back to pandas after the query runs.

    Usage:
        connector = SQLConnector("sales_db", url="sqlite:///data/sales.db")
        model.add_connector(connector)
        model.add_table("orders", connector="sales_db", source="orders")

        # For Postgres / MySQL / etc., supply the URL via an env var rather
        # than hard-coding credentials:
        #   url=os.environ["TRACEBI_SALES_DB_URL"]

    Args:
        name:   Logical name used to reference this connector in a DataModel.
        url:    SQLAlchemy connection URL.
        **kwargs: Additional keyword arguments passed to ``create_engine``.
    """

    def __init__(self, name: str, url: str, **kwargs) -> None:
        super().__init__(name)
        self.url = url
        self._engine_kwargs = kwargs
        self._engine = None

    def supports_pushdown(self) -> bool:
        return True

    def describe(self) -> dict:
        return {**super().describe(), "url": self._redacted_url()}

    def list_tables(self) -> list[str]:
        """Table names from the dialect inspector. Not a scan."""
        if self._engine is None:
            self.connect()
        from sqlalchemy import inspect
        return [str(name) for name in inspect(self._engine).get_table_names()]

    def column_schema(self, source: str) -> Optional[list[dict[str, str]]]:
        """Column names and types from the dialect's inspector. Not a scan.

        A raw ``SELECT`` source has no catalog entry; return ``None``
        rather than executing it.
        """
        if source.strip().upper().startswith("SELECT"):
            return None
        if self._engine is None:
            self.connect()
        from sqlalchemy import inspect
        cols = inspect(self._engine).get_columns(source)
        return [{"name": str(c["name"]), "dtype": str(c["type"])} for c in cols]

    def _redacted_url(self) -> str:
        """Connection URL with any password replaced by ***."""
        try:
            from sqlalchemy.engine import make_url
            url = make_url(self.url)
            return url.render_as_string(hide_password=True)
        except Exception:
            return "<unparseable url>"

    def connect(self) -> None:
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "sqlalchemy is required for SQLConnector.\n"
                "Install with: pip install sqlalchemy"
            )
        self._engine = create_engine(self.url, **self._engine_kwargs)

    def load(
        self,
        source: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if self._engine is None:
            self.connect()
        is_query = source.strip().upper().startswith("SELECT")
        if is_query:
            # Raw query — fall back to pandas-side push-down
            df = pd.read_sql(source, con=self._engine)
            return self._apply_pandas_pushdown(df, filter, columns)

        # Table name — push columns + filter into SQL
        from sqlalchemy import text
        select_cols = ", ".join(self._quote_ident(c) for c in columns) if columns else "*"
        query = f"SELECT {select_cols} FROM {self._quote_ident(source)}"
        params: dict[str, Any] = {}
        if filter:
            clauses = []
            for i, (col, val) in enumerate(filter.items()):
                key = f"p{i}"
                clauses.append(f"{self._quote_ident(col)} = :{key}")
                params[key] = val
            query += " WHERE " + " AND ".join(clauses)
        return pd.read_sql(text(query), con=self._engine, params=params)

    @staticmethod
    def _quote_ident(name: str) -> str:
        # Conservative double-quote; most dialects accept this. Reject anything
        # with embedded quotes to avoid injection via a column name.
        if '"' in name:
            raise ValueError(f"Invalid identifier: {name!r}")
        return f'"{name}"'

    def write(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = "replace",
    ) -> None:
        """Write *df* to *table* using pandas ``to_sql``."""
        if self._engine is None:
            self.connect()
        df.to_sql(table, con=self._engine, if_exists=if_exists, index=False)

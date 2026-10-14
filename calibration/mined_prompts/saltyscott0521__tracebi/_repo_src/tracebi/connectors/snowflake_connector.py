"""Snowflake connector."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from tracebi.connectors.base import BaseConnector


class SnowflakeConnector(BaseConnector):
    """
    Load tables from Snowflake.

    Requires: pip install 'tracebi[snowflake]'

    Usage:
        connector = SnowflakeConnector(
            "sf",
            account="myaccount",
            user="myuser",
            password="mypassword",
            warehouse="COMPUTE_WH",
            database="SALES",
            schema="PUBLIC",
        )
        model.add_connector(connector)
        model.add_table("orders", connector="sf", source="ORDERS")

    Args:
        name:      Logical name used to reference this connector in a DataModel.
        account:   Snowflake account identifier.
        user:      Snowflake username.
        password:  Snowflake password.
        warehouse: Virtual warehouse name.
        database:  Database name.
        schema:    Schema name (default ``PUBLIC``).
        role:      Optional role. Omitted from the driver call when unset.
    """

    def __init__(
        self,
        name: str,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str = "PUBLIC",
        role: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.role = role
        self._conn = None

    def storage(self) -> dict:
        return {"kind": "cloud", "exists": None, "size": None,
                "where": f"Snowflake {self.account} · {self.database}.{self.schema} (warehouse {self.warehouse})"}

    def supports_pushdown(self) -> bool:
        return True

    def connect(self) -> None:
        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required for SnowflakeConnector.\n"
                "Install with: pip install 'tracebi[snowflake]'"
            )
        kwargs = {
            "account": self.account,
            "user": self.user,
            "password": self.password,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
        }
        if self.role:
            kwargs["role"] = self.role
        self._conn = snowflake.connector.connect(**kwargs)

    def load(
        self,
        source: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if self._conn is None:
            self.connect()
        cur = self._conn.cursor()
        is_query = source.strip().upper().startswith("SELECT")
        if is_query:
            cur.execute(source)
            df = cur.fetch_pandas_all()
            cur.close()
            return self._apply_pandas_pushdown(df, filter, columns)

        select_cols = ", ".join(self._quote_ident(c) for c in columns) if columns else "*"
        query = f"SELECT {select_cols} FROM {self._quote_ident(source)}"
        params: list[Any] = []
        if filter:
            clauses = []
            for col, val in filter.items():
                clauses.append(f"{self._quote_ident(col)} = %s")
                params.append(val)
            query += " WHERE " + " AND ".join(clauses)
        cur.execute(query, tuple(params))
        df = cur.fetch_pandas_all()
        cur.close()
        return df

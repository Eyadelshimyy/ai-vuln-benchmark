"""Abstract base class for all TraceBi connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class BaseConnector(ABC):
    """
    Abstract connector. Subclass this to add a new data source.

    Every connector has a ``name`` (used to reference it in a DataModel)
    and two abstract methods: ``connect()`` and ``load(source)``.

    Push-down: connectors may accept optional ``filter`` (dict of equality
    filters) and ``columns`` (list of column names to project). When the
    connector can apply these at source (SQL WHERE/SELECT, DuckDB query),
    only the filtered/projected data is returned. The default implementation
    applies them in pandas after loading.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def connect(self) -> None:
        """Open / validate the connection to the data source."""
        ...

    @abstractmethod
    def load(
        self,
        source: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load data from *source* and return a pandas DataFrame.

        Args:
            source:  Connector-specific identifier (file name, table name, query, …).
            filter:  Optional column-equality filter map applied at source where
                     possible, in pandas otherwise.
            columns: Optional list of columns to project.
        """
        ...

    def supports_pushdown(self) -> bool:
        """Whether this connector applies filter/columns at source (vs. in pandas)."""
        return False

    def describe(self) -> dict:
        """
        Metadata shown in registries/UIs. Subclasses may add fields
        (tables, directory, url, …) but must never include credentials.
        """
        return {"name": self.name, "type": type(self).__name__}

    def storage(self) -> dict:
        """
        Where this connector's data physically lives, for people.

        ``{"kind", "where", "exists", "size"}``: *kind* is ``file``,
        ``directory``, ``database``, ``cloud``, ``memory`` or ``unknown``;
        *where* is a path (relative to the working directory when inside it)
        or a redacted URL — never a credential; *exists* says whether the
        file or folder is there now (``None`` when it cannot be checked
        without connecting); *size* is a file's bytes. Built from
        :meth:`describe`, so a connector that describes itself gets this free.
        """
        d = self.describe()
        database = d.get("database")
        if database and database != ":memory:":
            return _on_disk("file", str(database))
        if d.get("directory"):
            return _on_disk("directory", str(d["directory"]))
        if d.get("url"):
            return {"kind": "database", "where": d["url"], "exists": None, "size": None}
        if database == ":memory:" or "tables" in d:
            return {"kind": "memory", "where": "in memory, rebuilt every run", "exists": None, "size": None}
        return {"kind": "unknown", "where": None, "exists": None, "size": None}

    def list_tables(self) -> Optional[list[str]]:
        """
        Table names this connector can see, from metadata.

        Return ``None`` when the connector cannot say (the default). A list
        (possibly empty) means the catalog was read. Implementations must
        not scan table rows.
        """
        return None

    def column_schema(self, source: str) -> Optional[list[dict[str, str]]]:
        """
        Column names and dtypes for *source*, from metadata.

        Return ``None`` when the connector cannot answer without reading
        the table's rows. A list (possibly empty) means the schema was
        read. Implementations must not ``SELECT *``.
        """
        return None

    @staticmethod
    def _quote_ident(name: str, quote: str = '"') -> str:
        """
        Quote a SQL identifier conservatively.

        Rejects identifiers containing the quote character itself so a
        column/table name can never break out of its quoting.
        """
        if quote in name:
            raise ValueError(f"Invalid identifier: {name!r}")
        return f"{quote}{name}{quote}"

    @staticmethod
    def _apply_pandas_pushdown(
        df: pd.DataFrame,
        filter: Optional[dict[str, Any]],
        columns: Optional[list[str]],
    ) -> pd.DataFrame:
        """Apply filter / column projection in pandas as a fallback."""
        if filter:
            for col, val in filter.items():
                if col in df.columns:
                    df = df[df[col] == val]
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return df

    def write(
        self,
        df: "pd.DataFrame",
        table: str,
        if_exists: str = "replace",
    ) -> None:
        """
        Write *df* to *table* in this connector's data store.

        Args:
            df:        DataFrame to write.
            table:     Destination table / key name.
            if_exists: What to do if the table already exists:
                       ``'replace'`` (default), ``'append'``, or ``'fail'``.

        Raises:
            NotImplementedError: If the connector does not support writes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support write(). "
            "Use SQLConnector or MemoryConnector for write operations."
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def _on_disk(kind: str, path: str) -> dict:
    """A file or folder: shown relative to the working directory when inside it."""
    import os

    shown = path
    try:
        rel = os.path.relpath(path)
        if not rel.startswith(".."):
            shown = rel
    except ValueError:      # another drive on Windows
        pass
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists and kind == "file" else None
    return {"kind": kind, "where": shown, "exists": exists, "size": size}

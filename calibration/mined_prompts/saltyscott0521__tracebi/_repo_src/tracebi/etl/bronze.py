"""
BronzeLayer / LandingLayer — raw ingest from a connector with zero transformations.

The new canonical name is ``LandingLayer`` — it matches TraceBi's positioning
as the *delivery and reporting* layer that connects to whatever upstream
table already exists, rather than owning a full ETL stack. ``BronzeLayer``
remains a fully supported alias for back-compat.

Stamps lineage with the layer's ``operation`` ("landing" or "bronze") and
an ingestion timestamp.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from tracebi.connectors.base import BaseConnector
from tracebi.model.dataset import DataSet, LineageNode


class BronzeLayer:
    """
    Raw ingest layer — load data as-is from any connector.

    Standalone usage::

        bronze = BronzeLayer(connector=csv_connector, source="orders.csv")
        ds = bronze.load(name="orders_raw")

    Pipeline usage (with sink)::

        bronze = BronzeLayer(
            connector=source_connector,
            source="orders",
            sink=db_connector,
            sink_table="orders_bronze",
        )
        ds = bronze.execute()   # loads + writes to sink
    """

    # Lineage operation tag — subclasses may override (LandingLayer uses "landing")
    operation: str = "bronze"
    layer_label: str = "bronze"

    def __init__(
        self,
        connector: BaseConnector,
        source: str,
        description: str = "",
        sink: Optional[BaseConnector] = None,
        sink_table: Optional[str] = None,
    ) -> None:
        self._connector = connector
        self._source = source
        self._description = (
            description
            or f"{self.layer_label.title()} ingest from '{connector.name}': {source}"
        )
        self._sink = sink
        self._sink_table = sink_table

    def load(self, name: Optional[str] = None) -> DataSet:
        """Load raw data and return a lineage-tracked DataSet."""
        ingestion_ts = datetime.now(timezone.utc).isoformat()
        df = self._connector.load(self._source)
        node = LineageNode(
            operation=self.operation,
            description=self._description,
            connector={
                "connector_name": self._connector.name,
                "connector_type": type(self._connector).__name__,
            },
            source=self._source,
            metadata={
                "layer":          self.layer_label,
                "ingestion_time": ingestion_ts,
                "rows_ingested":  len(df),
                "columns":        list(df.columns),
            },
        )
        return DataSet(df=df, name=name or self._source, lineage=[node])

    def execute(self, name: Optional[str] = None) -> DataSet:
        """Load raw data, write to the configured sink, and return the DataSet."""
        if self._sink is None or self._sink_table is None:
            raise RuntimeError(
                f"{type(self).__name__}.execute() requires 'sink' and 'sink_table' to be configured."
            )
        ds = self.load(name=name or self._sink_table)
        self._sink.write(ds.to_pandas(), self._sink_table)
        return ds

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} connector={self._connector.name!r} "
            f"source={self._source!r} "
            f"sink_table={self._sink_table!r}>"
        )


class LandingLayer(BronzeLayer):
    """
    Landing layer — connect to an upstream table and load it as-is.

    TraceBi-positioned name for the ingest step. Identical behaviour to
    ``BronzeLayer`` but stamps lineage with ``operation="landing"``.
    """

    operation: str = "landing"
    layer_label: str = "landing"

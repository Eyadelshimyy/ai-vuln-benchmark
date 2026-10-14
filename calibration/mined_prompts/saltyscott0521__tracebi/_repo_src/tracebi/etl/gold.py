"""
GoldLayer / FinalLayer — aggregated, analytics-ready DataSets built from
a DataModel's star-schema query surface.

The new canonical name is ``FinalLayer`` — the serving layer that uses
the facts and dimensions declared on a ``DataModel`` to produce a clean
dataset ready for a report or dashboard. ``GoldLayer`` remains a fully
supported alias for back-compat.

Delegates to ``DataModel.query()`` and stamps the result with an additional
lineage node tagged with the layer's ``operation`` ("final" or "gold").
"""

from __future__ import annotations

from typing import Any, Optional

from tracebi.connectors.base import BaseConnector
from tracebi.model.dataset import DataSet, LineageNode
from tracebi.model.data_model import DataModel


class GoldLayer:
    """Analytics-ready aggregation layer backed by a DataModel's star-schema query."""

    operation: str = "gold"
    layer_label: str = "gold"

    def __init__(
        self,
        model: DataModel,
        fact: Optional[str] = None,
        measures: Optional[dict[str, str]] = None,
        dimensions: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        aggregate: bool = True,
        sink: Optional[BaseConnector] = None,
        sink_table: Optional[str] = None,
    ) -> None:
        self._model = model
        self._fact = fact
        self._measures = measures
        self._dimensions = dimensions
        self._filters = filters
        self._aggregate = aggregate
        self._sink = sink
        self._sink_table = sink_table

    def query(
        self,
        fact: str,
        measures: dict[str, str],
        dimensions: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        aggregate: bool = True,
        name: Optional[str] = None,
    ) -> DataSet:
        """Execute a star-schema query and return a final/gold-layer DataSet."""
        ds = self._model.query(
            fact=fact,
            measures=measures,
            dimensions=dimensions,
            filters=filters,
            aggregate=aggregate,
        )
        node = LineageNode(
            operation=self.operation,
            description=(
                f"{self.layer_label.title()} layer: {fact} "
                f"measures={list(measures.keys())} "
                f"dims={dimensions or []} "
                f"filters={filters or {}}"
            ),
            metadata={
                "layer":      self.layer_label,
                "fact":       fact,
                "measures":   measures,
                "dimensions": dimensions or [],
                "filters":    filters or {},
                "aggregate":  aggregate,
            },
        )
        output_name = name or f"{fact}_{self.layer_label}"
        return DataSet(df=ds.to_pandas(), name=output_name, lineage=ds.lineage + [node])

    def execute(self, name: Optional[str] = None) -> DataSet:
        if self._fact is None or self._measures is None:
            raise RuntimeError(
                f"{type(self).__name__}.execute() requires 'fact' and 'measures' to be configured."
            )
        if self._sink is None or self._sink_table is None:
            raise RuntimeError(
                f"{type(self).__name__}.execute() requires 'sink' and 'sink_table'."
            )
        ds = self.query(
            fact=self._fact,
            measures=self._measures,
            dimensions=self._dimensions,
            filters=self._filters,
            aggregate=self._aggregate,
            name=name or self._sink_table,
        )
        self._sink.write(ds.to_pandas(), self._sink_table)
        return ds

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} model={self._model.name!r} "
            f"fact={self._fact!r} "
            f"sink_table={self._sink_table!r}>"
        )


class FinalLayer(GoldLayer):
    """
    Final/serving layer — facts + dimensions resolved into a clean DataSet
    ready for reports or dashboards.

    TraceBi-positioned name. Identical behaviour to ``GoldLayer`` but stamps
    lineage with ``operation="final"``.
    """

    operation: str = "final"
    layer_label: str = "final"

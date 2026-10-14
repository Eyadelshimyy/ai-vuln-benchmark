"""
TraceBi — code-first, traceable BI and analytics framework.

Quick start (in-memory, no config):
    import pandas as pd
    from tracebi import DataModel, MemoryConnector

    df = pd.DataFrame({"region": ["NE", "SE"], "revenue": [100.0, 200.0]})
    model = DataModel("Demo").add_connector(MemoryConnector("mem", {"orders": df}))
    model.add_table("orders", connector="mem", source="orders")

    ds = model.load("orders")   # DataSet: immutable DataFrame + lineage chain
    ds.help()                   # API cheat sheet (DataModel and Report have one too)
"""

from tracebi._version import __version__
from tracebi import model_registry
from tracebi import pipeline_registry
from tracebi.connectors.base import BaseConnector
from tracebi.connectors.csv_connector import CSVConnector
from tracebi.connectors.sql_connector import SQLConnector
from tracebi.connectors.memory_connector import MemoryConnector
from tracebi.connectors.duckdb_connector import DuckDBConnector
from tracebi.connectors.bigquery_connector import BigQueryConnector
from tracebi.connectors.snowflake_connector import SnowflakeConnector
from tracebi.model.dataset import DataSet, LineageNode
from tracebi.model.data_model import DataModel, MeasureDef, QuerySpec
from tracebi.etl.bronze import BronzeLayer, LandingLayer
from tracebi.etl.silver import SilverLayer, ManipulationLayer
from tracebi.etl.gold import GoldLayer, FinalLayer
from tracebi.lineage.diagram import LineageDiagram
from tracebi.pipeline import model_pipeline
from tracebi.pipeline.runner import PipelineRunner
# NB: import the class only. Binding the `registry` singleton here would
# shadow the `tracebi.registry` submodule, breaking the documented
# `from tracebi.registry import registry`.
from tracebi.contracts import ContractViolation, contract
from tracebi.registry import Registry
from tracebi.spec import DataRef, ReportSpec
from tracebi.web.register import register

__all__ = [
    # Connectors
    "BaseConnector",
    "CSVConnector",
    "SQLConnector",
    "MemoryConnector",
    "DuckDBConnector",
    "BigQueryConnector",
    "SnowflakeConnector",
    # Core
    "DataSet",
    "LineageNode",
    "DataModel",
    "MeasureDef",
    "QuerySpec",
    # Layers (canonical)
    "LandingLayer",
    "ManipulationLayer",
    "FinalLayer",
    # Layers (back-compat aliases)
    "BronzeLayer",
    "SilverLayer",
    "GoldLayer",
    # Visualisation & orchestration
    "LineageDiagram",
    "PipelineRunner",
    "model_pipeline",
    # Project registry + registration facade
    # (the singleton is `from tracebi.registry import registry`)
    "Registry",
    "register",
    # Reports as data
    "ReportSpec",
    "DataRef",
    # Transform (sink) contracts
    "contract",
    "ContractViolation",
    # Project-scope registries (import as modules: tracebi.model_registry.get_model)
    "model_registry",
    "pipeline_registry",
    # Metadata
    "__version__",
]

"""Google BigQuery connector."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from tracebi.connectors.base import BaseConnector


class BigQueryConnector(BaseConnector):
    """
    Load tables from Google BigQuery.

    Requires: pip install 'tracebi[bigquery]'

    Usage:
        connector = BigQueryConnector(
            "bq",
            project="my-project",
            dataset="analytics",
        )
        model.add_connector(connector)
        model.add_table("events", connector="bq", source="events")

    Args:
        name:        Logical name used to reference this connector in a DataModel.
        project:     GCP project ID.
        dataset:     BigQuery dataset name.
        credentials: Optional google.oauth2.credentials.Credentials object.
                     Uses Application Default Credentials when omitted.
    """

    def __init__(
        self,
        name: str,
        project: str,
        dataset: str,
        credentials=None,
    ) -> None:
        super().__init__(name)
        self.project = project
        self.dataset = dataset
        self.credentials = credentials
        self._client = None

    def storage(self) -> dict:
        return {"kind": "cloud", "exists": None, "size": None,
                "where": f"BigQuery {self.project}.{self.dataset}"}

    def supports_pushdown(self) -> bool:
        return True

    def connect(self) -> None:
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery and db-dtypes are required for BigQueryConnector.\n"
                "Install with: pip install 'tracebi[bigquery]'"
            )
        self._client = bigquery.Client(
            project=self.project,
            credentials=self.credentials,
        )

    def load(
        self,
        source: str,
        filter: Optional[dict[str, Any]] = None,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        if self._client is None:
            self.connect()

        is_query = source.strip().upper().startswith("SELECT")
        if is_query:
            df = self._client.query(source).to_dataframe()
            return self._apply_pandas_pushdown(df, filter, columns)

        from google.cloud import bigquery

        select_cols = (
            ", ".join(self._quote_ident(c, quote="`") for c in columns)
            if columns else "*"
        )
        table_ref = self._quote_ident(f"{self.project}.{self.dataset}.{source}", quote="`")
        query = f"SELECT {select_cols} FROM {table_ref}"
        params: list[Any] = []
        if filter:
            clauses = []
            for i, (col, val) in enumerate(filter.items()):
                pname = f"p{i}"
                clauses.append(f"{self._quote_ident(col, quote='`')} = @{pname}")
                params.append(bigquery.ScalarQueryParameter(
                    pname, self._bq_param_type(val), val,
                ))
            query += " WHERE " + " AND ".join(clauses)
        job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
        return self._client.query(query, job_config=job_config).to_dataframe()

    @staticmethod
    def _bq_param_type(val: Any) -> str:
        if isinstance(val, bool):
            return "BOOL"
        if isinstance(val, int):
            return "INT64"
        if isinstance(val, float):
            return "FLOAT64"
        if isinstance(val, str):
            return "STRING"
        raise ValueError(
            f"Unsupported filter value type for BigQuery push-down: {type(val).__name__}"
        )

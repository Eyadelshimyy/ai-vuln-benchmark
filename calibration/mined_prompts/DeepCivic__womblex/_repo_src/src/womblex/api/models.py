"""Response and request models for the `/v1` API — the OpenAPI surface."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Ready(BaseModel):
    ready: bool
    store: bool
    queue: bool


class RunStatus(BaseModel):
    """One run's queue rollup. ``state`` is derived from ``counts``."""

    run_id: str
    state: Literal["pending", "running", "done", "failed"]
    counts: dict[str, int]
    total: int
    created_at: str | None = None
    updated_at: str | None = None


class RunList(BaseModel):
    runs: list[RunStatus]


class RunManifest(BaseModel):
    run_id: str
    documents: list[dict[str, Any]]


class RunMetrics(BaseModel):
    run_id: str
    stats: dict[str, int]
    workers: list[dict[str, Any]]
    throughput: dict[str, Any]


class RunRequest(BaseModel):
    """A run submission. ``preset`` or ``config`` picks the downstream stages; neither is extraction only."""

    input_prefix: str
    preset: str | None = None
    config: dict[str, Any] | None = None
    batch_size: int = Field(default=50, ge=1)

    @model_validator(mode="after")
    def _one_config_source(self) -> RunRequest:
        if self.preset is not None and self.config is not None:
            raise ValueError("give preset or config, not both")
        return self


class RunSubmitted(BaseModel):
    run_id: str
    document_count: int
    batch_count: int
    stages: list[str]


class RunFile(BaseModel):
    """One object under the run. Contract keys are null for a non-Parquet or pre-contract file."""

    key: str
    rows: int | None = None
    contract_version: str | None = None
    sensitivity: str | None = None


class RunFiles(BaseModel):
    run_id: str
    files: list[RunFile]


class DocumentText(BaseModel):
    """One document's text in one layer; ``rows`` carry that layer's columns."""

    run_id: str
    source_hash: str
    layer: Literal["masked", "chunks", "elements"]
    sensitivity: Literal["raw", "masked", "none"]
    rows: list[dict[str, Any]]


class UploadAccepted(BaseModel):
    """Stored documents; ``input_prefix`` is what ``POST /v1/runs`` takes to run them."""

    upload_id: str
    input_prefix: str
    files: list[str]
    bytes: int

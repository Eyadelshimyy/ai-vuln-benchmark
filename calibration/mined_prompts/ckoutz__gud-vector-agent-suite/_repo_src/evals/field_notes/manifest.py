"""Candidate manifests.

A manifest names the candidates for a run and how to reach them. It holds only
environment variable *names*, never secret values, and never a chosen provider —
selection is an outcome of the evaluation, not an input to it.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from field_notes.adapters.base import ExtractionAdapter
from field_notes.adapters.fake import DegradedAdapter, Fault, OracleAdapter
from field_notes.adapters.schema_output import (
    EndpointConfig,
    SchemaCompletionTransport,
    SchemaOutputAdapter,
)
from field_notes.cases import EvalCase, Split

MANIFEST_ROOT = Path(__file__).parent / "manifests"


class AdapterKind(StrEnum):
    """Which adapter implementation backs a candidate."""

    FAKE_ORACLE = "fake_oracle"
    FAKE_DEGRADED = "fake_degraded"
    SCHEMA_OUTPUT = "schema_output"


class CandidateSpec(BaseModel):
    """One evaluated candidate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: AdapterKind
    model: str | None = None
    endpoint: EndpointConfig | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    faults: list[Fault] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_kind(self) -> CandidateSpec:
        if self.kind is AdapterKind.SCHEMA_OUTPUT and not self.model:
            raise ValueError(f"{self.name}: schema_output candidates require a model")
        if self.kind is not AdapterKind.SCHEMA_OUTPUT and self.model:
            raise ValueError(f"{self.name}: fake candidates must not name a model")
        if self.faults and self.kind is not AdapterKind.FAKE_DEGRADED:
            raise ValueError(f"{self.name}: faults are only valid for fake_degraded")
        if self.kind is AdapterKind.FAKE_DEGRADED and not self.faults:
            raise ValueError(f"{self.name}: fake_degraded requires at least one fault")
        return self


class SuiteManifest(BaseModel):
    """A named run configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    splits: list[Split] = Field(default_factory=lambda: [Split.DEV])
    candidates: list[CandidateSpec]

    @model_validator(mode="after")
    def _validate_names(self) -> SuiteManifest:
        names = [candidate.name for candidate in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        if not self.splits:
            raise ValueError("at least one split is required")
        return self


def load_manifest(path: Path) -> SuiteManifest:
    """Load and validate a manifest file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: manifest must be a mapping")
    return SuiteManifest.model_validate(raw)


def build_adapter(
    spec: CandidateSpec,
    cases: list[EvalCase],
    transport: SchemaCompletionTransport | None = None,
) -> ExtractionAdapter:
    """Instantiate the adapter for ``spec``.

    ``schema_output`` candidates are constructed without a transport unless one is
    injected by the caller, so a manifest alone can never reach a live endpoint.
    """
    match spec.kind:
        case AdapterKind.FAKE_ORACLE:
            return OracleAdapter(cases, name=spec.name)
        case AdapterKind.FAKE_DEGRADED:
            return DegradedAdapter(cases, faults=spec.faults, name=spec.name)
        case AdapterKind.SCHEMA_OUTPUT:
            if spec.model is None:
                raise ValueError(f"{spec.name}: model is required")
            return SchemaOutputAdapter(
                name=spec.name,
                model=spec.model,
                endpoint=spec.endpoint,
                parameters=spec.parameters,
                transport=transport,
            )

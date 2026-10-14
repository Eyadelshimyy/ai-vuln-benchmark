"""Provenance-safe binary bifactor calibration handoff for pilot items."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Any, Iterable

import numpy as np

from ..config import FitConfig
from .models import SCHEMA_VERSION, _identifier, _schema_version, _sha256_hex
from .pilot_observations import (
    MirtPilotDesign,
    PilotItemProvenance,
    PilotObservationRecord,
    PilotResponseState,
    build_mirt_pilot_design,
)

_BIFACTOR_DESIGN_TOKEN = object()


@dataclass(frozen=True)
class BifactorPilotDesign:
    """Content-addressed pilot handoff for the binary ``BIFAC2PLM`` model.

    The wrapped binary design assigns each item to exactly one specific factor
    through its ``query_testlet_id``. ``general_factor_id`` records the single
    general factor that loads on every item. The identifier is audit metadata;
    the existing Rust-backed bifactor estimator receives the binary response
    matrix, the item-to-specific-factor vector, and a validated
    :class:`~fast_mlsirm.config.FitConfig`.

    This design states a bifactor loading pattern only. It does not establish
    model identification, adequate fit, scoreability, fairness, or validity.
    """

    general_factor_id: str
    binary_design: MirtPilotDesign
    schema_version: str = SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction outside the validated public builder."""
        if _design_token is not _BIFACTOR_DESIGN_TOKEN:
            raise ValueError(
                "BifactorPilotDesign must be created by "
                "build_bifactor_pilot_design"
            )
        object.__setattr__(
            self,
            "general_factor_id",
            _identifier(self.general_factor_id, "general_factor_id"),
        )
        if not isinstance(self.binary_design, MirtPilotDesign):
            raise TypeError("binary_design must be a validated MirtPilotDesign")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.schema_version != self.binary_design.schema_version:
            raise ValueError("schema_version must match the wrapped binary design")

    @property
    def pilot_study_id(self) -> str:
        """Return the pilot-study identifier retained by the binary design."""
        return self.binary_design.pilot_study_id

    @property
    def respondent_ids(self) -> tuple[str, ...]:
        """Return respondent identifiers in response-matrix row order."""
        return self.binary_design.respondent_ids

    @property
    def item_provenance(self) -> tuple[PilotItemProvenance, ...]:
        """Return immutable item provenance in response-matrix column order."""
        return self.binary_design.item_provenance

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return item identifiers in response-matrix column order."""
        return self.binary_design.item_ids

    @property
    def specific_factor_testlet_ids(self) -> tuple[str, ...]:
        """Return descriptive specific-factor identifiers in numeric-id order."""
        return self.binary_design.factor_testlet_ids

    @property
    def item_specific_factor_ids(self) -> tuple[int, ...]:
        """Return each item's zero-based specific-factor assignment."""
        return self.binary_design.item_factor_ids

    @property
    def general_factor_item_ids(self) -> tuple[str, ...]:
        """Return the items constrained to load on the declared general factor."""
        return self.item_ids

    @property
    def responses(self) -> tuple[tuple[int | None, ...], ...]:
        """Return the immutable binary response matrix with explicit gaps."""
        return self.binary_design.responses

    @property
    def response_states(self) -> tuple[tuple[PilotResponseState, ...], ...]:
        """Return exact response states retained independently of numeric NaN."""
        return self.binary_design.response_states

    @property
    def rater_assignments(self) -> tuple[tuple[str | None, ...], ...]:
        """Return the provenance-only rater assignment for every matrix cell."""
        return self.binary_design.rater_assignments

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical design content without outer derived identities."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "general_factor_id": self.general_factor_id,
            "general_factor_item_ids": list(self.general_factor_item_ids),
            "specific_factor_testlet_ids": list(self.specific_factor_testlet_ids),
            "item_specific_factor_ids": list(self.item_specific_factor_ids),
            "binary_design": self.binary_design.to_dict(),
        }

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the complete bifactor handoff design."""
        return _sha256_hex(self._content_dict())

    @property
    def design_id(self) -> str:
        """Return a descriptive 128-bit public bifactor-design handle."""
        return f"bifactor_pilot_design_{self.design_fingerprint[:32]}"

    def responses_array(self) -> np.ndarray:
        """Return a fresh float matrix with non-observed states represented by NaN."""
        return self.binary_design.responses_array()

    def specific_factor_id_array(self) -> np.ndarray:
        """Return a fresh per-item specific-factor assignment vector."""
        return self.binary_design.factor_id_array()

    def default_fit_config(self) -> FitConfig:
        """Return the conservative binary bifactor marginal-fit configuration."""
        return FitConfig(model="BIFAC2PLM", estimator="mmle", latent_dim=1)

    def to_fit_kwargs(self, config: FitConfig | None = None) -> dict[str, Any]:
        """Return copied arguments accepted directly by ``fast_mlsirm.fit``.

        A caller-supplied configuration may tune numerical settings, but it
        must retain the binary ``BIFAC2PLM`` model, marginal estimation, and a
        one-dimensional general-factor interaction. This prevents an audit
        artifact labelled as a bifactor handoff from being silently fitted as
        MIRT, a latent-distance model, or a multi-general-factor extension.
        """
        resolved = self.default_fit_config() if config is None else config
        if not isinstance(resolved, FitConfig):
            raise TypeError("config must be a FitConfig or None")
        resolved.validate()
        if resolved.normalized_model() != "BIFAC2PLM":
            raise ValueError("config.model must be BIFAC2PLM for this handoff")
        if resolved.estimator != "mmle":
            raise ValueError("config.estimator must be 'mmle' for BIFAC2PLM")
        if resolved.latent_dim != 1:
            raise ValueError(
                "config.latent_dim must be 1 for the declared single general factor"
            )
        return {
            "responses": self.responses_array(),
            "factor_id": self.specific_factor_id_array(),
            "config": resolved,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical content and deterministic bifactor identities."""
        return {
            **self._content_dict(),
            "design_id": self.design_id,
            "design_fingerprint": self.design_fingerprint,
        }


def build_bifactor_pilot_design(
    records: Iterable[PilotObservationRecord],
    *,
    general_factor_id: str = "general_factor",
) -> BifactorPilotDesign:
    """Build a deterministic binary bifactor handoff from pilot observations.

    The existing binary MIRT assembler remains the fail-closed source of truth
    for response, provenance, missingness, duplicate-cell, category, and dense
    allocation validation. This builder adds an explicit single general-factor
    identity over every item without changing or aggregating any response.
    """
    binary_design = build_mirt_pilot_design(records)
    return BifactorPilotDesign(
        general_factor_id=general_factor_id,
        binary_design=binary_design,
        schema_version=binary_design.schema_version,
        _design_token=_BIFACTOR_DESIGN_TOKEN,
    )

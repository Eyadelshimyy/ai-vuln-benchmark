"""Immutable fail-closed policy contracts for the scoring lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._contract_safety import (
    bounded_positive_integer,
    descriptive_identifier,
    sorted_identifiers,
)
from ._validation import (
    MAX_RATERS_PER_RESPONSE,
    CanonicalContract,
    assessment_error,
    strict_boolean,
)


@dataclass(frozen=True)
class EnginePolicy(CanonicalContract):
    """Allowed human and automated rater boundary for an assessment."""

    policy_id: str
    engine_ids: tuple[str, ...] = ()
    allow_human_raters: bool = True
    allow_automated_raters: bool = False
    minimum_raters_per_response: int = 1

    def __post_init__(self) -> None:
        """Normalize engine identities and reject contradictory rater policies."""
        object.__setattr__(
            self,
            "policy_id",
            descriptive_identifier(self.policy_id, "policy_id"),
        )
        engines = sorted_identifiers(self.engine_ids, "engine_ids", minimum=0)
        object.__setattr__(self, "engine_ids", engines)
        human = strict_boolean(self.allow_human_raters, "allow_human_raters")
        automated = strict_boolean(
            self.allow_automated_raters,
            "allow_automated_raters",
        )
        object.__setattr__(self, "allow_human_raters", human)
        object.__setattr__(self, "allow_automated_raters", automated)
        object.__setattr__(
            self,
            "minimum_raters_per_response",
            bounded_positive_integer(
                self.minimum_raters_per_response,
                "minimum_raters_per_response",
                MAX_RATERS_PER_RESPONSE,
            ),
        )
        if not human and not automated:
            raise assessment_error(
                "no_available_rater_kind",
                "$",
                "engine policy must allow at least one rater kind",
            )
        if automated and not engines:
            raise assessment_error(
                "missing_automated_engine",
                "$.engine_ids",
                "automated scoring requires at least one engine",
            )
        if not automated and engines:
            raise assessment_error(
                "disabled_automated_engine",
                "$.engine_ids",
                "engine_ids must be empty when automated raters are disabled",
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible engine policy."""
        return {
            "policy_id": self.policy_id,
            "engine_ids": list(self.engine_ids),
            "allow_human_raters": self.allow_human_raters,
            "allow_automated_raters": self.allow_automated_raters,
            "minimum_raters_per_response": self.minimum_raters_per_response,
        }

    _content_dict = to_dict


@dataclass(frozen=True)
class CalibrationPolicy(CanonicalContract):
    """Declared calibration model and constructs included in that model."""

    policy_id: str
    model_id: str
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize calibration-model and construct references."""
        object.__setattr__(
            self,
            "policy_id",
            descriptive_identifier(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "model_id",
            descriptive_identifier(self.model_id, "model_id"),
        )
        object.__setattr__(
            self,
            "construct_ids",
            sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible calibration policy."""
        return {
            "policy_id": self.policy_id,
            "model_id": self.model_id,
            "construct_ids": list(self.construct_ids),
        }

    _content_dict = to_dict


@dataclass(frozen=True)
class ValidationPolicy(CanonicalContract):
    """Validation metrics and construct scopes required before score use."""

    policy_id: str
    metric_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize metric and construct references."""
        object.__setattr__(
            self,
            "policy_id",
            descriptive_identifier(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "metric_ids",
            sorted_identifiers(self.metric_ids, "metric_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible validation policy."""
        return {
            "policy_id": self.policy_id,
            "metric_ids": list(self.metric_ids),
            "construct_ids": list(self.construct_ids),
        }

    _content_dict = to_dict


@dataclass(frozen=True)
class AdjudicationPolicy(CanonicalContract):
    """Transparent human-review triggers and their construct scopes."""

    policy_id: str
    trigger_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize adjudication-trigger and construct references."""
        object.__setattr__(
            self,
            "policy_id",
            descriptive_identifier(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "trigger_ids",
            sorted_identifiers(self.trigger_ids, "trigger_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible adjudication policy."""
        return {
            "policy_id": self.policy_id,
            "trigger_ids": list(self.trigger_ids),
            "construct_ids": list(self.construct_ids),
        }

    _content_dict = to_dict


@dataclass(frozen=True)
class MonitoringPolicy(CanonicalContract):
    """Versioned drift metrics and construct scopes for operational monitoring."""

    policy_id: str
    metric_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize monitoring-metric and construct references."""
        object.__setattr__(
            self,
            "policy_id",
            descriptive_identifier(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "metric_ids",
            sorted_identifiers(self.metric_ids, "metric_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible monitoring policy."""
        return {
            "policy_id": self.policy_id,
            "metric_ids": list(self.metric_ids),
            "construct_ids": list(self.construct_ids),
        }

    _content_dict = to_dict


@dataclass(frozen=True)
class ReportingPolicy(CanonicalContract):
    """Allowed report artifacts and construct scopes for one assessment."""

    policy_id: str
    format_ids: tuple[str, ...]
    construct_ids: tuple[str, ...]
    include_exact_values: bool = True

    def __post_init__(self) -> None:
        """Normalize report formats, construct references, and disclosure policy."""
        object.__setattr__(
            self,
            "policy_id",
            descriptive_identifier(self.policy_id, "policy_id"),
        )
        object.__setattr__(
            self,
            "format_ids",
            sorted_identifiers(self.format_ids, "format_ids", minimum=1),
        )
        object.__setattr__(
            self,
            "construct_ids",
            sorted_identifiers(self.construct_ids, "construct_ids", minimum=0),
        )
        object.__setattr__(
            self,
            "include_exact_values",
            strict_boolean(self.include_exact_values, "include_exact_values"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible reporting policy."""
        return {
            "policy_id": self.policy_id,
            "format_ids": list(self.format_ids),
            "construct_ids": list(self.construct_ids),
            "include_exact_values": self.include_exact_values,
        }

    _content_dict = to_dict

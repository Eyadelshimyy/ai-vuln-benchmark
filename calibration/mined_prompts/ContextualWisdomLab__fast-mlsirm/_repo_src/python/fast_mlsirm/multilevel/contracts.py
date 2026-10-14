"""Immutable contextual-membership and longitudinal measurement contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from typing import Any

from ._validation import (
    MULTILEVEL_SCHEMA_VERSION,
    MultilevelContractError,
    artifact_digest,
    autoregressive_coefficient,
    bounded_values,
    contract_error,
    descriptive_identifier,
    exact_integer,
    fingerprint,
    membership_weight,
    schema_version,
    strict_boolean,
)

MAX_CONTEXT_MEMBERSHIPS = 100_000
MAX_TEMPORAL_OCCASIONS = 100_000
MEMBERSHIP_WEIGHT_TOLERANCE = 1e-12

_MEMBERSHIP_TOKEN = object()
_MEMBERSHIP_DESIGN_TOKEN = object()
_OCCASION_TOKEN = object()
_STATE_SPEC_TOKEN = object()
_LONGITUDINAL_DESIGN_TOKEN = object()


class LongitudinalStateKind(str, Enum):
    """Supported repeated-measurement states and their compatibility labels.

    ``RANDOM_INTERCEPT_SLOPE`` is retained as a wire-compatible input label,
    while the current implementation reports its estimand as an independent
    respondent OLS trend rather than a population random-effects fit.
    """

    RANDOM_INTERCEPT_SLOPE = "random_intercept_slope"
    STATIONARY_AUTOREGRESSIVE = "stationary_autoregressive"


def _state_kind(value: Any) -> LongitudinalStateKind:
    """Return one supported state kind without reflecting callback failures."""
    if isinstance(value, LongitudinalStateKind):
        return value
    try:
        return LongitudinalStateKind(value)
    except Exception:
        raise contract_error(
            "invalid_state_kind",
            "$.state_kind",
            "state_kind must be one of the supported longitudinal structures",
        ) from None


def _safe_values(
    values: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> tuple[Any, ...]:
    """Materialize a bounded collection while redacting ordinary callbacks."""
    try:
        return bounded_values(
            values,
            name,
            minimum=minimum,
            maximum=maximum,
        )
    except MultilevelContractError:
        raise
    except Exception:
        raise contract_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} could not be materialized safely",
        ) from None


def _assert_leaf_integrity(
    value: Any,
    *,
    code: str,
    message: str,
) -> None:
    """Verify one package-built leaf against its immutable construction seal."""
    try:
        current = artifact_digest(value._content_dict())
    except Exception:
        current = ""
    if (
        getattr(value, "schema_version", None) != MULTILEVEL_SCHEMA_VERSION
        or current != getattr(value, "_sealed_fingerprint", None)
    ):
        raise contract_error(code, "$", message)


@dataclass(frozen=True)
class ContextMembership:
    """One weighted observation-to-context edge with revision provenance."""

    observation_id: str
    context_dimension_id: str
    context_id: str
    membership_weight: float
    membership_revision_fingerprint: str
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _membership_token: InitVar[object | None] = None
    _sealed_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _membership_token: object | None) -> None:
        """Reject direct construction, normalize content, and seal its identity."""
        if _membership_token is not _MEMBERSHIP_TOKEN:
            raise contract_error(
                "unverified_context_membership",
                "$",
                "use build_context_membership",
            )
        object.__setattr__(
            self,
            "observation_id",
            descriptive_identifier(self.observation_id, "observation_id"),
        )
        object.__setattr__(
            self,
            "context_dimension_id",
            descriptive_identifier(
                self.context_dimension_id,
                "context_dimension_id",
            ),
        )
        object.__setattr__(
            self,
            "context_id",
            descriptive_identifier(self.context_id, "context_id"),
        )
        object.__setattr__(
            self,
            "membership_weight",
            membership_weight(self.membership_weight),
        )
        object.__setattr__(
            self,
            "membership_revision_fingerprint",
            fingerprint(
                self.membership_revision_fingerprint,
                "membership_revision_fingerprint",
            ),
        )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))
        object.__setattr__(
            self,
            "_sealed_fingerprint",
            artifact_digest(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical edge content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "context_dimension_id": self.context_dimension_id,
            "context_id": self.context_id,
            "membership_weight": self.membership_weight,
            "membership_revision_fingerprint": self.membership_revision_fingerprint,
        }

    def _assert_integrity(self) -> None:
        """Reject post-factory mutation before any public identity is exposed."""
        _assert_leaf_integrity(
            self,
            code="context_membership_integrity_mismatch",
            message="membership content no longer matches its package-owned seal",
        )

    @property
    def membership_fingerprint(self) -> str:
        """Return the sealed SHA-256 identity after integrity verification."""
        self._assert_integrity()
        return self._sealed_fingerprint

    @property
    def membership_handle(self) -> str:
        """Return a descriptive 128-bit public membership handle."""
        return f"context_membership_{self.membership_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical edge content only after integrity verification."""
        self._assert_integrity()
        return {
            **self._content_dict(),
            "membership_handle": self.membership_handle,
            "membership_fingerprint": self.membership_fingerprint,
        }


@dataclass(frozen=True)
class ContextMembershipDesign:
    """Factory-sealed sparse cross-classified membership design."""

    memberships: tuple[ContextMembership, ...]
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _design_token: InitVar[object | None] = None
    _sealed_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction and seal the complete aggregate content."""
        if _design_token is not _MEMBERSHIP_DESIGN_TOKEN:
            raise contract_error(
                "unverified_context_membership_design",
                "$",
                "use build_context_membership_design",
            )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))
        object.__setattr__(
            self,
            "_sealed_fingerprint",
            artifact_digest(self._content_dict()),
        )

    def _assert_integrity(self) -> None:
        """Verify the aggregate seal and every exact package-owned child."""
        try:
            if any(type(value) is not ContextMembership for value in self.memberships):
                raise TypeError
            for value in self.memberships:
                value._assert_integrity()
            current = artifact_digest(self._content_dict())
        except Exception:
            current = ""
        if (
            self.schema_version != MULTILEVEL_SCHEMA_VERSION
            or current != self._sealed_fingerprint
        ):
            raise contract_error(
                "context_membership_design_integrity_mismatch",
                "$",
                "membership design no longer matches its package-owned seal",
            )

    @property
    def observation_ids(self) -> tuple[str, ...]:
        """Return observation identifiers in deterministic order."""
        return tuple(sorted({value.observation_id for value in self.memberships}))

    @property
    def context_dimension_ids(self) -> tuple[str, ...]:
        """Return contextual classification identifiers in deterministic order."""
        return tuple(
            sorted({value.context_dimension_id for value in self.memberships})
        )

    @property
    def context_keys(self) -> tuple[tuple[str, str], ...]:
        """Return dimension-qualified context identities."""
        return tuple(
            sorted(
                {
                    (value.context_dimension_id, value.context_id)
                    for value in self.memberships
                }
            )
        )

    def _grouped_memberships(
        self,
    ) -> dict[str, dict[str, tuple[ContextMembership, ...]]]:
        """Return observation and dimension groups in deterministic order."""
        grouped: dict[str, dict[str, list[ContextMembership]]] = {}
        for value in self.memberships:
            grouped.setdefault(value.observation_id, {}).setdefault(
                value.context_dimension_id,
                [],
            ).append(value)
        return {
            observation_id: {
                dimension_id: tuple(grouped[observation_id][dimension_id])
                for dimension_id in self.context_dimension_ids
            }
            for observation_id in self.observation_ids
        }

    @property
    def membership_counts(self) -> tuple[int, ...]:
        """Return total edge counts aligned with observations."""
        grouped = self._grouped_memberships()
        return tuple(
            sum(
                len(grouped[observation_id][dimension_id])
                for dimension_id in self.context_dimension_ids
            )
            for observation_id in self.observation_ids
        )

    @property
    def membership_weights(self) -> tuple[tuple[float, ...], ...]:
        """Return flattened exact weights aligned with canonical edge order."""
        grouped = self._grouped_memberships()
        return tuple(
            tuple(
                edge.membership_weight
                for dimension_id in self.context_dimension_ids
                for edge in grouped[observation_id][dimension_id]
            )
            for observation_id in self.observation_ids
        )

    @property
    def membership_counts_by_dimension(self) -> tuple[tuple[int, ...], ...]:
        """Return edge counts aligned with observations and dimensions."""
        grouped = self._grouped_memberships()
        return tuple(
            tuple(
                len(grouped[observation_id][dimension_id])
                for dimension_id in self.context_dimension_ids
            )
            for observation_id in self.observation_ids
        )

    @property
    def membership_weights_by_dimension(
        self,
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        """Return exact weights aligned with observations and dimensions."""
        grouped = self._grouped_memberships()
        return tuple(
            tuple(
                tuple(
                    edge.membership_weight
                    for edge in grouped[observation_id][dimension_id]
                )
                for dimension_id in self.context_dimension_ids
            )
            for observation_id in self.observation_ids
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical sparse design content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "memberships": [value._content_dict() for value in self.memberships],
        }

    @property
    def design_fingerprint(self) -> str:
        """Return the sealed SHA-256 design identity after replay verification."""
        self._assert_integrity()
        return self._sealed_fingerprint

    @property
    def design_handle(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"context_membership_design_{self.design_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return dimension-qualified aggregate content after integrity checks."""
        self._assert_integrity()
        return {
            "schema_version": self.schema_version,
            "memberships": [value.to_dict() for value in self.memberships],
            "observation_ids": list(self.observation_ids),
            "context_dimension_ids": list(self.context_dimension_ids),
            "context_keys": [list(value) for value in self.context_keys],
            "membership_counts": list(self.membership_counts),
            "membership_weights": [list(value) for value in self.membership_weights],
            "membership_counts_by_dimension": [
                list(value) for value in self.membership_counts_by_dimension
            ],
            "membership_weights_by_dimension": [
                [list(weights) for weights in observation]
                for observation in self.membership_weights_by_dimension
            ],
            "design_handle": self.design_handle,
            "design_fingerprint": self.design_fingerprint,
        }


@dataclass(frozen=True)
class TemporalOccasion:
    """One exact repeated-measurement occasion for one respondent or system."""

    respondent_id: str
    occasion_id: str
    sequence_index: int
    time_offset_milliseconds: int
    occasion_revision_fingerprint: str
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _occasion_token: InitVar[object | None] = None
    _sealed_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _occasion_token: object | None) -> None:
        """Reject direct construction, normalize content, and seal its identity."""
        if _occasion_token is not _OCCASION_TOKEN:
            raise contract_error(
                "unverified_temporal_occasion",
                "$",
                "use build_temporal_occasion",
            )
        object.__setattr__(
            self,
            "respondent_id",
            descriptive_identifier(self.respondent_id, "respondent_id"),
        )
        object.__setattr__(
            self,
            "occasion_id",
            descriptive_identifier(self.occasion_id, "occasion_id"),
        )
        object.__setattr__(
            self,
            "sequence_index",
            exact_integer(self.sequence_index, "sequence_index", minimum=0),
        )
        object.__setattr__(
            self,
            "time_offset_milliseconds",
            exact_integer(
                self.time_offset_milliseconds,
                "time_offset_milliseconds",
            ),
        )
        object.__setattr__(
            self,
            "occasion_revision_fingerprint",
            fingerprint(
                self.occasion_revision_fingerprint,
                "occasion_revision_fingerprint",
            ),
        )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))
        object.__setattr__(
            self,
            "_sealed_fingerprint",
            artifact_digest(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical occasion content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "respondent_id": self.respondent_id,
            "occasion_id": self.occasion_id,
            "sequence_index": self.sequence_index,
            "time_offset_milliseconds": self.time_offset_milliseconds,
            "occasion_revision_fingerprint": self.occasion_revision_fingerprint,
        }

    def _assert_integrity(self) -> None:
        """Reject post-factory mutation before any public identity is exposed."""
        _assert_leaf_integrity(
            self,
            code="temporal_occasion_integrity_mismatch",
            message="occasion content no longer matches its package-owned seal",
        )

    @property
    def occasion_fingerprint(self) -> str:
        """Return the sealed SHA-256 identity after integrity verification."""
        self._assert_integrity()
        return self._sealed_fingerprint

    @property
    def occasion_handle(self) -> str:
        """Return a descriptive 128-bit public occasion handle."""
        return f"temporal_occasion_{self.occasion_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical occasion content only after integrity verification."""
        self._assert_integrity()
        return {
            **self._content_dict(),
            "occasion_handle": self.occasion_handle,
            "occasion_fingerprint": self.occasion_fingerprint,
        }


@dataclass(frozen=True)
class LongitudinalStateSpec:
    """One immutable latent-state and residual-dependence specification."""

    state_kind: LongitudinalStateKind
    autoregressive_coefficient: float | None
    include_lagged_response_dependence: bool
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _state_spec_token: InitVar[object | None] = None
    _sealed_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _state_spec_token: object | None) -> None:
        """Reject direct construction, normalize content, and seal its identity."""
        if _state_spec_token is not _STATE_SPEC_TOKEN:
            raise contract_error(
                "unverified_longitudinal_state_spec",
                "$",
                "use build_longitudinal_state_spec",
            )
        normalized_kind = _state_kind(self.state_kind)
        object.__setattr__(self, "state_kind", normalized_kind)
        if normalized_kind is LongitudinalStateKind.STATIONARY_AUTOREGRESSIVE:
            normalized_coefficient = autoregressive_coefficient(
                self.autoregressive_coefficient
            )
            object.__setattr__(
                self,
                "autoregressive_coefficient",
                0.0 if normalized_coefficient == 0.0 else normalized_coefficient,
            )
        elif self.autoregressive_coefficient is not None:
            raise contract_error(
                "unexpected_autoregressive_coefficient",
                "$.autoregressive_coefficient",
                "random-intercept/slope states do not accept an AR coefficient",
            )
        object.__setattr__(
            self,
            "include_lagged_response_dependence",
            strict_boolean(
                self.include_lagged_response_dependence,
                "include_lagged_response_dependence",
            ),
        )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))
        object.__setattr__(
            self,
            "_sealed_fingerprint",
            artifact_digest(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical state content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "state_kind": self.state_kind.value,
            "autoregressive_coefficient": self.autoregressive_coefficient,
            "include_lagged_response_dependence": self.include_lagged_response_dependence,
        }

    def _assert_integrity(self) -> None:
        """Reject post-factory mutation before any public identity is exposed."""
        _assert_leaf_integrity(
            self,
            code="longitudinal_state_spec_integrity_mismatch",
            message="state specification no longer matches its package-owned seal",
        )

    @property
    def state_spec_fingerprint(self) -> str:
        """Return the sealed SHA-256 identity after integrity verification."""
        self._assert_integrity()
        return self._sealed_fingerprint

    @property
    def state_spec_handle(self) -> str:
        """Return a descriptive 128-bit public state-specification handle."""
        return f"longitudinal_state_spec_{self.state_spec_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical state content only after integrity verification."""
        self._assert_integrity()
        return {
            **self._content_dict(),
            "state_spec_handle": self.state_spec_handle,
            "state_spec_fingerprint": self.state_spec_fingerprint,
        }


@dataclass(frozen=True)
class LongitudinalDesign:
    """Factory-sealed repeated-measurement design with ordered provenance."""

    occasions: tuple[TemporalOccasion, ...]
    state_spec: LongitudinalStateSpec
    schema_version: str = MULTILEVEL_SCHEMA_VERSION
    _design_token: InitVar[object | None] = None
    _sealed_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction and seal the complete longitudinal content."""
        if _design_token is not _LONGITUDINAL_DESIGN_TOKEN:
            raise contract_error(
                "unverified_longitudinal_design",
                "$",
                "use build_longitudinal_design",
            )
        object.__setattr__(self, "schema_version", schema_version(self.schema_version))
        object.__setattr__(
            self,
            "_sealed_fingerprint",
            artifact_digest(self._content_dict()),
        )

    def _assert_integrity(self) -> None:
        """Verify the aggregate seal and every exact package-owned child."""
        try:
            if type(self.state_spec) is not LongitudinalStateSpec:
                raise TypeError
            if any(type(value) is not TemporalOccasion for value in self.occasions):
                raise TypeError
            self.state_spec._assert_integrity()
            for value in self.occasions:
                value._assert_integrity()
            current = artifact_digest(self._content_dict())
        except Exception:
            current = ""
        if (
            self.schema_version != MULTILEVEL_SCHEMA_VERSION
            or current != self._sealed_fingerprint
        ):
            raise contract_error(
                "longitudinal_design_integrity_mismatch",
                "$",
                "longitudinal design no longer matches its package-owned seal",
            )

    @property
    def respondent_ids(self) -> tuple[str, ...]:
        """Return respondent identifiers in deterministic design order."""
        return tuple(sorted({value.respondent_id for value in self.occasions}))

    def _grouped_occasions(self) -> dict[str, tuple[TemporalOccasion, ...]]:
        """Return deterministic respondent-level occasion sequences."""
        grouped: dict[str, list[TemporalOccasion]] = {}
        for value in self.occasions:
            grouped.setdefault(value.respondent_id, []).append(value)
        return {
            respondent_id: tuple(grouped[respondent_id])
            for respondent_id in self.respondent_ids
        }

    @property
    def occasion_counts(self) -> tuple[int, ...]:
        """Return occasion counts aligned with respondent identifiers."""
        grouped = self._grouped_occasions()
        return tuple(len(grouped[value]) for value in self.respondent_ids)

    @property
    def time_offsets_milliseconds(self) -> tuple[tuple[int, ...], ...]:
        """Return exact irregular time offsets aligned with respondent order."""
        grouped = self._grouped_occasions()
        return tuple(
            tuple(
                occasion.time_offset_milliseconds
                for occasion in grouped[respondent_id]
            )
            for respondent_id in self.respondent_ids
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical longitudinal content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "occasions": [value._content_dict() for value in self.occasions],
            "state_spec": self.state_spec._content_dict(),
        }

    @property
    def design_fingerprint(self) -> str:
        """Return the sealed SHA-256 design identity after replay verification."""
        self._assert_integrity()
        return self._sealed_fingerprint

    @property
    def design_handle(self) -> str:
        """Return a descriptive 128-bit public design handle."""
        return f"longitudinal_design_{self.design_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return longitudinal design content after aggregate integrity checks."""
        self._assert_integrity()
        return {
            "schema_version": self.schema_version,
            "occasions": [value.to_dict() for value in self.occasions],
            "state_spec": self.state_spec.to_dict(),
            "respondent_ids": list(self.respondent_ids),
            "occasion_counts": list(self.occasion_counts),
            "design_handle": self.design_handle,
            "design_fingerprint": self.design_fingerprint,
        }


def build_context_membership(
    *,
    observation_id: str,
    context_dimension_id: str,
    context_id: str,
    membership_weight: float,
    membership_revision_fingerprint: str,
) -> ContextMembership:
    """Build one validated observation-to-context membership edge."""
    return ContextMembership(
        observation_id=observation_id,
        context_dimension_id=context_dimension_id,
        context_id=context_id,
        membership_weight=membership_weight,
        membership_revision_fingerprint=membership_revision_fingerprint,
        _membership_token=_MEMBERSHIP_TOKEN,
    )


def _replay_membership(value: ContextMembership, index: int) -> ContextMembership:
    """Rebuild and verify one exact package-owned membership before aggregation."""
    path = f"$.memberships[{index}]"
    try:
        value._assert_integrity()
        replayed = build_context_membership(
            observation_id=value.observation_id,
            context_dimension_id=value.context_dimension_id,
            context_id=value.context_id,
            membership_weight=value.membership_weight,
            membership_revision_fingerprint=value.membership_revision_fingerprint,
        )
        replayed._assert_integrity()
        if replayed.membership_fingerprint != value._sealed_fingerprint:
            raise ValueError
    except Exception:
        raise contract_error(
            "context_membership_integrity_mismatch",
            path,
            "membership content no longer matches its package-owned seal",
        ) from None
    return replayed


def build_context_membership_design(
    memberships: Iterable[ContextMembership],
) -> ContextMembershipDesign:
    """Build one bounded canonical cross-classified membership design."""
    raw = _safe_values(
        memberships,
        "memberships",
        minimum=1,
        maximum=MAX_CONTEXT_MEMBERSHIPS,
    )
    replayed_values: list[ContextMembership] = []
    for index, value in enumerate(raw):
        if type(value) is not ContextMembership:
            raise contract_error(
                "invalid_context_membership",
                f"$.memberships[{index}]",
                "memberships must contain exact ContextMembership values",
            )
        replayed_values.append(_replay_membership(value, index))
    ordered = tuple(
        sorted(
            replayed_values,
            key=lambda value: (
                value.observation_id,
                value.context_dimension_id,
                value.context_id,
                value.membership_revision_fingerprint,
            ),
        )
    )
    cells: set[tuple[str, str, str]] = set()
    revisions: dict[str, tuple[str, str, str, float]] = {}
    grouped_weights: dict[tuple[str, str], list[float]] = {}
    observation_dimensions: dict[str, set[str]] = {}
    all_dimensions = {value.context_dimension_id for value in ordered}
    for index, value in enumerate(ordered):
        cell = (
            value.observation_id,
            value.context_dimension_id,
            value.context_id,
        )
        if cell in cells:
            raise contract_error(
                "duplicate_context_membership",
                f"$.memberships[{index}]",
                "each observation-dimension-context cell may occur only once",
            )
        cells.add(cell)
        revision_contract = (*cell, value.membership_weight)
        previous = revisions.get(value.membership_revision_fingerprint)
        if previous is not None and previous != revision_contract:
            raise contract_error(
                "membership_revision_conflict",
                f"$.memberships[{index}].membership_revision_fingerprint",
                "one membership revision cannot identify different assignments",
            )
        revisions[value.membership_revision_fingerprint] = revision_contract
        group = (value.observation_id, value.context_dimension_id)
        grouped_weights.setdefault(group, []).append(value.membership_weight)
        observation_dimensions.setdefault(value.observation_id, set()).add(
            value.context_dimension_id
        )
    if any(dimensions != all_dimensions for dimensions in observation_dimensions.values()):
        raise contract_error(
            "missing_context_dimension_membership",
            "$.memberships",
            "every observation must contain every declared context dimension",
        )
    for weights in grouped_weights.values():
        if not math.isclose(
            math.fsum(weights),
            1.0,
            rel_tol=0.0,
            abs_tol=MEMBERSHIP_WEIGHT_TOLERANCE,
        ):
            raise contract_error(
                "membership_weight_total_mismatch",
                "$.memberships",
                "membership weights must sum to one within every context dimension",
            )
    return ContextMembershipDesign(
        memberships=ordered,
        _design_token=_MEMBERSHIP_DESIGN_TOKEN,
    )


def build_temporal_occasion(
    *,
    respondent_id: str,
    occasion_id: str,
    sequence_index: int,
    time_offset_milliseconds: int,
    occasion_revision_fingerprint: str,
) -> TemporalOccasion:
    """Build one validated repeated-measurement occasion."""
    return TemporalOccasion(
        respondent_id=respondent_id,
        occasion_id=occasion_id,
        sequence_index=sequence_index,
        time_offset_milliseconds=time_offset_milliseconds,
        occasion_revision_fingerprint=occasion_revision_fingerprint,
        _occasion_token=_OCCASION_TOKEN,
    )


def build_longitudinal_state_spec(
    *,
    state_kind: LongitudinalStateKind | str,
    autoregressive_coefficient: float | None = None,
    include_lagged_response_dependence: bool = False,
) -> LongitudinalStateSpec:
    """Build one validated latent-state and lag-dependence specification."""
    return LongitudinalStateSpec(
        state_kind=state_kind,  # type: ignore[arg-type]
        autoregressive_coefficient=autoregressive_coefficient,
        include_lagged_response_dependence=include_lagged_response_dependence,
        _state_spec_token=_STATE_SPEC_TOKEN,
    )


def _replay_occasion(value: TemporalOccasion, index: int) -> TemporalOccasion:
    """Rebuild and verify one exact package-owned occasion before aggregation."""
    path = f"$.occasions[{index}]"
    try:
        value._assert_integrity()
        replayed = build_temporal_occasion(
            respondent_id=value.respondent_id,
            occasion_id=value.occasion_id,
            sequence_index=value.sequence_index,
            time_offset_milliseconds=value.time_offset_milliseconds,
            occasion_revision_fingerprint=value.occasion_revision_fingerprint,
        )
        replayed._assert_integrity()
        if replayed.occasion_fingerprint != value._sealed_fingerprint:
            raise ValueError
    except Exception:
        raise contract_error(
            "temporal_occasion_integrity_mismatch",
            path,
            "occasion content no longer matches its package-owned seal",
        ) from None
    return replayed


def _replay_state_spec(value: LongitudinalStateSpec) -> LongitudinalStateSpec:
    """Rebuild and verify one exact package-owned state specification."""
    try:
        value._assert_integrity()
        replayed = build_longitudinal_state_spec(
            state_kind=value.state_kind,
            autoregressive_coefficient=value.autoregressive_coefficient,
            include_lagged_response_dependence=(
                value.include_lagged_response_dependence
            ),
        )
        replayed._assert_integrity()
        if replayed.state_spec_fingerprint != value._sealed_fingerprint:
            raise ValueError
    except Exception:
        raise contract_error(
            "longitudinal_state_spec_integrity_mismatch",
            "$.state_spec",
            "state specification no longer matches its package-owned seal",
        ) from None
    return replayed


def build_longitudinal_design(
    *,
    occasions: Iterable[TemporalOccasion],
    state_spec: LongitudinalStateSpec,
) -> LongitudinalDesign:
    """Build one bounded canonical longitudinal measurement design."""
    raw = _safe_values(
        occasions,
        "occasions",
        minimum=1,
        maximum=MAX_TEMPORAL_OCCASIONS,
    )
    if type(state_spec) is not LongitudinalStateSpec:
        raise contract_error(
            "invalid_longitudinal_state_spec",
            "$.state_spec",
            "state_spec must be an exact LongitudinalStateSpec",
        )
    replayed_state = _replay_state_spec(state_spec)
    replayed_occasions: list[TemporalOccasion] = []
    for index, value in enumerate(raw):
        if type(value) is not TemporalOccasion:
            raise contract_error(
                "invalid_temporal_occasion",
                f"$.occasions[{index}]",
                "occasions must contain exact TemporalOccasion values",
            )
        replayed_occasions.append(_replay_occasion(value, index))
    ordered = tuple(
        sorted(
            replayed_occasions,
            key=lambda value: (
                value.respondent_id,
                value.sequence_index,
                value.occasion_id,
                value.occasion_revision_fingerprint,
            ),
        )
    )
    revisions: dict[str, tuple[str, str, int, int]] = {}
    grouped: dict[str, list[tuple[int, TemporalOccasion]]] = {}
    for index, value in enumerate(ordered):
        revision_contract = (
            value.respondent_id,
            value.occasion_id,
            value.sequence_index,
            value.time_offset_milliseconds,
        )
        previous = revisions.get(value.occasion_revision_fingerprint)
        if previous is not None and previous != revision_contract:
            raise contract_error(
                "occasion_revision_conflict",
                f"$.occasions[{index}].occasion_revision_fingerprint",
                "one occasion revision cannot identify different occasions",
            )
        revisions[value.occasion_revision_fingerprint] = revision_contract
        grouped.setdefault(value.respondent_id, []).append((index, value))
    for values in grouped.values():
        occasion_ids: set[str] = set()
        sequence_indices: set[int] = set()
        time_offsets: set[int] = set()
        previous_time: int | None = None
        for index, value in values:
            if value.occasion_id in occasion_ids:
                raise contract_error(
                    "duplicate_temporal_occasion",
                    f"$.occasions[{index}].occasion_id",
                    "occasion identifiers must be unique within a respondent",
                )
            occasion_ids.add(value.occasion_id)
            if value.sequence_index in sequence_indices:
                raise contract_error(
                    "duplicate_temporal_sequence",
                    f"$.occasions[{index}].sequence_index",
                    "sequence indices must be unique within a respondent",
                )
            sequence_indices.add(value.sequence_index)
            if value.time_offset_milliseconds in time_offsets:
                raise contract_error(
                    "duplicate_temporal_offset",
                    f"$.occasions[{index}].time_offset_milliseconds",
                    "time offsets must be unique within a respondent",
                )
            time_offsets.add(value.time_offset_milliseconds)
            if (
                previous_time is not None
                and value.time_offset_milliseconds <= previous_time
            ):
                raise contract_error(
                    "nonincreasing_temporal_order",
                    "$.occasions",
                    "time offsets must increase with sequence order",
                )
            previous_time = value.time_offset_milliseconds
    return LongitudinalDesign(
        occasions=ordered,
        state_spec=replayed_state,
        _design_token=_LONGITUDINAL_DESIGN_TOKEN,
    )


__all__ = [
    "ContextMembership",
    "ContextMembershipDesign",
    "LongitudinalDesign",
    "LongitudinalStateKind",
    "LongitudinalStateSpec",
    "MAX_CONTEXT_MEMBERSHIPS",
    "MAX_TEMPORAL_OCCASIONS",
    "MEMBERSHIP_WEIGHT_TOLERANCE",
    "MultilevelContractError",
    "TemporalOccasion",
    "build_context_membership",
    "build_context_membership_design",
    "build_longitudinal_design",
    "build_longitudinal_state_spec",
    "build_temporal_occasion",
]

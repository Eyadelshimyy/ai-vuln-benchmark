"""Provenance-safe observed-score DIF handoff for generated-item pilots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass, replace
from typing import Any

import numpy as np

from .models import SCHEMA_VERSION, _identifier, _schema_version, _sha256_hex
from .pilot_observations import (
    MirtPilotDesign,
    PilotItemProvenance,
    PilotObservationRecord,
    PilotResponseState,
    _error,
    build_mirt_pilot_design,
)

_DIF_DESIGN_TOKEN = object()


@dataclass(frozen=True)
class DifPilotDesign:
    """Content-addressed pilot handoff for binary observed-score DIF screens.

    The design wraps the replay-verified binary MIRT pilot matrix and binds each
    respondent to one explicitly named reference or focal group. Exact response
    states and per-cell rater provenance remain available through
    :attr:`binary_design`; no missing response is deleted, imputed, or coerced
    to an incorrect response.

    The current observed-score DIF APIs require a complete binary matrix.
    :meth:`to_observed_score_dif_kwargs` therefore fails closed whenever any
    response state is not ``observed``. A successful handoff establishes only
    API-compatible provenance and array construction. It does not establish
    measurement invariance, fairness, causal bias, scoreability, or validity.
    """

    binary_design: MirtPilotDesign
    reference_group_id: str
    focal_group_id: str
    respondent_group_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    _design_token: InitVar[object | None] = None

    def __post_init__(self, _design_token: object | None) -> None:
        """Reject direct construction and validate the complete group contract."""
        if _design_token is not _DIF_DESIGN_TOKEN:
            raise ValueError("DifPilotDesign must be created by build_dif_pilot_design")
        if type(self.binary_design) is not MirtPilotDesign:
            raise TypeError("binary_design must be a validated MirtPilotDesign")
        object.__setattr__(
            self,
            "reference_group_id",
            _identifier(self.reference_group_id, "reference_group_id"),
        )
        object.__setattr__(
            self,
            "focal_group_id",
            _identifier(self.focal_group_id, "focal_group_id"),
        )
        if self.reference_group_id == self.focal_group_id:
            raise ValueError("reference_group_id and focal_group_id must differ")
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.schema_version != self.binary_design.schema_version:
            raise ValueError("schema_version must match the wrapped binary design")
        if len(self.respondent_group_ids) != len(self.binary_design.respondent_ids):
            raise ValueError(
                "respondent_group_ids must align one-to-one with respondent_ids"
            )
        allowed = {self.reference_group_id, self.focal_group_id}
        normalized_groups = tuple(
            _identifier(group_id, f"respondent_group_ids[{index}]")
            for index, group_id in enumerate(self.respondent_group_ids)
        )
        if any(group_id not in allowed for group_id in normalized_groups):
            raise ValueError(
                "respondent_group_ids may contain only the declared reference and focal groups"
            )
        if self.reference_group_id not in normalized_groups:
            raise ValueError("reference group must contain at least one respondent")
        if self.focal_group_id not in normalized_groups:
            raise ValueError("focal group must contain at least one respondent")
        object.__setattr__(self, "respondent_group_ids", normalized_groups)

    def _revalidated_copy(self) -> DifPilotDesign:
        """Replay package-owned group invariants on the exact sealed record."""
        if type(self) is not DifPilotDesign:
            raise TypeError("DIF pilot handoff requires an exact DifPilotDesign")
        if type(self.binary_design) is not MirtPilotDesign:
            raise TypeError("binary_design must be a validated MirtPilotDesign")
        if type(self.respondent_group_ids) is not tuple:
            raise TypeError("respondent_group_ids must be an exact tuple")
        return replace(self, _design_token=_DIF_DESIGN_TOKEN)

    @property
    def pilot_study_id(self) -> str:
        """Return the pilot-study identity retained by the binary design."""
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
    def responses(self) -> tuple[tuple[int | None, ...], ...]:
        """Return the immutable binary response matrix with explicit gaps."""
        return self.binary_design.responses

    @property
    def response_states(self) -> tuple[tuple[PilotResponseState, ...], ...]:
        """Return exact response states retained independently of numeric NaN."""
        return self.binary_design.response_states

    @property
    def rater_assignments(self) -> tuple[tuple[str | None, ...], ...]:
        """Return provenance-only rater assignments for every response cell."""
        return self.binary_design.rater_assignments

    @property
    def is_complete_observed_matrix(self) -> bool:
        """Return whether every respondent-item cell is explicitly observed."""
        return all(
            state is PilotResponseState.OBSERVED
            for row in self.response_states
            for state in row
        )

    def _content_dict_unchecked(self) -> dict[str, Any]:
        """Return canonical content for an already replay-validated design."""
        return {
            "schema_version": self.schema_version,
            "pilot_study_id": self.pilot_study_id,
            "reference_group_id": self.reference_group_id,
            "focal_group_id": self.focal_group_id,
            "respondent_group_ids": list(self.respondent_group_ids),
            "binary_design": self.binary_design.to_dict(),
        }

    def _content_dict(self) -> dict[str, Any]:
        """Return replay-validated canonical design content."""
        return self._revalidated_copy()._content_dict_unchecked()

    @property
    def design_fingerprint(self) -> str:
        """Return SHA-256 over the binary design and ordered group assignments."""
        return _sha256_hex(self._content_dict())

    @property
    def design_id(self) -> str:
        """Return a descriptive 128-bit public DIF-design handle."""
        return f"dif_pilot_design_{self.design_fingerprint[:32]}"

    def responses_array(self) -> np.ndarray:
        """Return a fresh float matrix preserving non-observed cells as NaN."""
        return self.binary_design.responses_array()

    def _group_array_unchecked(self) -> np.ndarray:
        """Build the binary group vector from an already validated design."""
        return np.asarray(
            [
                0 if group_id == self.reference_group_id else 1
                for group_id in self.respondent_group_ids
            ],
            dtype=np.int64,
        )

    def group_array(self) -> np.ndarray:
        """Return a fresh integer vector using 0=reference and 1=focal."""
        return self._revalidated_copy()._group_array_unchecked()

    def to_observed_score_dif_kwargs(self) -> dict[str, np.ndarray]:
        """Return copied arrays accepted by the repository's binary DIF APIs.

        The returned ``responses`` and ``group`` arguments are accepted by
        ``mantel_haenszel_dif``, ``logistic_dif``, their purified variants, and
        ``sibtest``. The method rejects the first non-observed cell instead of
        silently using complete-case deletion or imputation.
        """
        validated = self._revalidated_copy()
        for respondent_index, row in enumerate(validated.response_states):
            for item_index, state in enumerate(row):
                if state is not PilotResponseState.OBSERVED:
                    raise _error(
                        "dif_incomplete_response_matrix",
                        (
                            "$.binary_design.response_states"
                            f"[{respondent_index}][{item_index}]"
                        ),
                        "observed-score DIF requires a complete binary matrix; "
                        "no deletion or imputation is performed",
                    )
        return {
            "responses": np.asarray(validated.responses, dtype=np.int64),
            "group": validated._group_array_unchecked(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return canonical design content and deterministic identities."""
        validated = self._revalidated_copy()
        content = validated._content_dict_unchecked()
        fingerprint = _sha256_hex(content)
        return {
            **content,
            "design_id": f"dif_pilot_design_{fingerprint[:32]}",
            "design_fingerprint": fingerprint,
            "is_complete_observed_matrix": validated.is_complete_observed_matrix,
        }


def build_dif_pilot_design(
    records: Iterable[PilotObservationRecord],
    *,
    respondent_groups: Mapping[str, str],
    reference_group_id: str,
    focal_group_id: str,
) -> DifPilotDesign:
    """Build a deterministic binary pilot design for observed-score DIF APIs.

    ``respondent_groups`` must assign every respondent in the replay-verified
    binary design exactly once to either ``reference_group_id`` or
    ``focal_group_id``. Unknown respondents, omitted respondents, undeclared
    group identities, and one-group designs are rejected with stable structured
    errors. Missing response states remain in the nested design and cause the
    estimator-argument handoff to reject rather than select or impute cases.
    """
    binary_design = build_mirt_pilot_design(records)
    if not isinstance(respondent_groups, Mapping):
        raise TypeError("respondent_groups must be a mapping")
    if len(respondent_groups) > len(binary_design.respondent_ids):
        raise _error(
            "dif_group_assignment_count_exceeded",
            "$.respondent_groups",
            "respondent group assignments cannot exceed indexed respondents",
        )

    reference = _identifier(reference_group_id, "reference_group_id")
    focal = _identifier(focal_group_id, "focal_group_id")
    if reference == focal:
        raise ValueError("reference_group_id and focal_group_id must differ")

    normalized_mapping: dict[str, str] = {}
    for raw_respondent_id, raw_group_id in respondent_groups.items():
        respondent_id = _identifier(raw_respondent_id, "respondent_groups key")
        group_id = _identifier(
            raw_group_id,
            f"respondent_groups[{respondent_id}]",
        )
        if respondent_id in normalized_mapping:
            raise _error(
                "dif_duplicate_group_assignment",
                "$.respondent_groups",
                "respondent group keys must remain unique after normalization",
            )
        normalized_mapping[respondent_id] = group_id

    expected = set(binary_design.respondent_ids)
    supplied = set(normalized_mapping)
    if supplied - expected:
        raise _error(
            "dif_unknown_respondent_assignment",
            "$.respondent_groups",
            "group assignments may reference only respondents in the pilot design",
        )
    if expected - supplied:
        raise _error(
            "dif_missing_group_assignment",
            "$.respondent_groups",
            "every respondent must have one explicit reference or focal assignment",
        )

    allowed_groups = {reference, focal}
    if any(group_id not in allowed_groups for group_id in normalized_mapping.values()):
        raise _error(
            "dif_unknown_group_assignment",
            "$.respondent_groups",
            "assignments may contain only the declared reference and focal groups",
        )

    ordered_groups = tuple(
        normalized_mapping[respondent_id]
        for respondent_id in binary_design.respondent_ids
    )
    if reference not in ordered_groups:
        raise _error(
            "dif_empty_reference_group",
            "$.respondent_groups",
            "the reference group must contain at least one respondent",
        )
    if focal not in ordered_groups:
        raise _error(
            "dif_empty_focal_group",
            "$.respondent_groups",
            "the focal group must contain at least one respondent",
        )

    return DifPilotDesign(
        binary_design=binary_design,
        reference_group_id=reference,
        focal_group_id=focal,
        respondent_group_ids=ordered_groups,
        _design_token=_DIF_DESIGN_TOKEN,
    )
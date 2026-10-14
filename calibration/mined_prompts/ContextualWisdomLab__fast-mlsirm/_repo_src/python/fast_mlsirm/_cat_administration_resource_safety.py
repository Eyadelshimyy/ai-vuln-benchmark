"""Resource preflight for dichotomous CAT administration evidence.

The CAT numerical owner remains Rust.  This module rejects structurally
impossible EAP/MLE administrations and oversized standard-error mask evidence
from inert container metadata before existing CAT validators perform value-wise
scans or dense dtype marshalling.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_SHAPE_ERROR = "administered and responses must be 1D arrays of equal length"
_LENGTH_ERROR = "administration length cannot exceed item bank size"
_STANDARD_ERROR_RESOURCE_ERROR = (
    "ability_standard_error administered evidence exceeds resource limit"
)
_MAX_STANDARD_ERROR_ADMINISTERED_CELLS = 20_000_000
_VALIDATE_MARKER = "__fast_mlsirm_cat_administration_resource_safe__"
_STANDARD_ERROR_MARKER = "__fast_mlsirm_cat_standard_error_resource_safe__"


def _vector_length(value: object) -> int | None:
    """Return inert one-dimensional length, or ``None`` for unsupported carriers."""
    if type(value) is np.ndarray:
        if value.ndim != 1:
            raise ValueError(_SHAPE_ERROR)
        return int(value.shape[0])
    if type(value) in (list, tuple):
        return len(value)
    return None


def _standard_error_logical_cells(value: object) -> int | None:
    """Return inert mask size without imposing EAP/MLE rank or uniqueness rules."""
    if type(value) is np.ndarray:
        return int(value.size)
    if type(value) in (list, tuple):
        return len(value)
    return None


def _reject_over_bank_administration(bank: Any, administered: object) -> None:
    """Reject exact-carrier administrations that cannot be unique in the bank."""
    administered_length = _vector_length(administered)
    if administered_length is None:
        return
    n_items = int(np.asarray(bank.b).shape[0])
    if administered_length > n_items:
        raise ValueError(_LENGTH_ERROR)


def install(cat_module: ModuleType) -> None:
    """Install fail-fast CAT resource preflights idempotently."""
    original_validate = cat_module._validate_administration
    if not bool(getattr(original_validate, _VALIDATE_MARKER, False)):

        def safe_validate_administration(
            bank: Any,
            factor_id: np.ndarray,
            administered: object,
            responses: object,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Unsupported administered providers retain the original callback/type
            # rejection contract.  Once administered exposes inert exact-container
            # metadata, reject structurally impossible unique administrations before
            # inspecting the response carrier so an invalid/unsupported response
            # cannot force an oversized administered vector through dense validation.
            administered_length = _vector_length(administered)
            if administered_length is None:
                return original_validate(bank, factor_id, administered, responses)

            _reject_over_bank_administration(bank, administered)

            responses_length = _vector_length(responses)
            if responses_length is None:
                return original_validate(bank, factor_id, administered, responses)

            if administered_length != responses_length:
                raise ValueError(_SHAPE_ERROR)

            return original_validate(bank, factor_id, administered, responses)

        setattr(safe_validate_administration, _VALIDATE_MARKER, True)
        cat_module._validate_administration = safe_validate_administration

    original_standard_error = cat_module.ability_standard_error
    if not bool(getattr(original_standard_error, _STANDARD_ERROR_MARKER, False)):

        def safe_ability_standard_error(
            bank: Any,
            factor_id: np.ndarray,
            theta: np.ndarray,
            *,
            administered: object | None = None,
            model: str = "MLS2PLM",
        ) -> np.ndarray:
            # Standard-error administration is a set-valued mask rather than the
            # unique 1-D EAP/MLE history.  Preserve duplicate and multidimensional
            # exact NumPy evidence, but bound its logical size before signed-int64
            # value scans, dense conversion, and np.unique deduplication.
            if administered is not None:
                logical_cells = _standard_error_logical_cells(administered)
                if (
                    logical_cells is not None
                    and logical_cells > _MAX_STANDARD_ERROR_ADMINISTERED_CELLS
                ):
                    raise ValueError(_STANDARD_ERROR_RESOURCE_ERROR)
            return original_standard_error(
                bank,
                factor_id,
                theta,
                administered=administered,
                model=model,
            )

        setattr(safe_ability_standard_error, _STANDARD_ERROR_MARKER, True)
        cat_module.ability_standard_error = safe_ability_standard_error

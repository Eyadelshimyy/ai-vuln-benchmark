"""Resource-bound regressions for CAT standard-error administration masks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm
import fast_mlsirm._cat_administration_resource_safety as resource_safety
import fast_mlsirm.cat as cat_module
from fast_mlsirm.types import MLSIRMParams


class _CoreSentinel:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"compiled CAT core must not be reached for oversized mask evidence: {name}"
        )


def _bank() -> MLSIRMParams:
    return MLSIRMParams(
        theta=np.zeros((1, 1), dtype=np.float64),
        alpha=np.zeros(2, dtype=np.float64),
        b=np.zeros(2, dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((2, 1), dtype=np.float64),
        tau=0.0,
    )


@pytest.mark.parametrize(
    "standard_error",
    [cat_module.ability_standard_error, fast_mlsirm.ability_standard_error],
    ids=["module", "package"],
)
def test_standard_error_rejects_oversized_mask_before_value_scan_or_core(
    monkeypatch: pytest.MonkeyPatch,
    standard_error: Callable[..., np.ndarray],
) -> None:
    """Logical mask size is bounded without applying EAP/MLE uniqueness rules."""
    administered = np.broadcast_to(np.array([[0]], dtype=np.int64), (2, 2))
    monkeypatch.setattr(
        resource_safety,
        "_MAX_STANDARD_ERROR_ADMINISTERED_CELLS",
        3,
        raising=False,
    )

    def forbidden_index_normalization(values: object) -> np.ndarray:
        raise AssertionError(
            "oversized standard-error mask must fail before signed-64 value scanning"
        )

    monkeypatch.setattr(
        cat_module,
        "_lossless_signed_int64_indices",
        forbidden_index_normalization,
    )
    monkeypatch.setattr(fast_mlsirm, "_core", _CoreSentinel(), raising=False)

    with pytest.raises(
        ValueError,
        match="ability_standard_error administered evidence exceeds resource limit",
    ):
        standard_error(
            _bank(),
            np.zeros(2, dtype=np.int64),
            np.zeros(1, dtype=np.float64),
            administered=administered,
        )

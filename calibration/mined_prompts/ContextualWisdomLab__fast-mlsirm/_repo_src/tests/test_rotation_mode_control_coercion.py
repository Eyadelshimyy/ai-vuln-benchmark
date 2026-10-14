"""Fail-first safety contract for rotation mode validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rotation import rotate_factor_loadings


class _HostileMode:
    """A mode probe whose representation callbacks must never be executed."""

    def __str__(self) -> str:
        raise RuntimeError("ROTATION_MODE_STR_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("ROTATION_MODE_REPR_SENTINEL")


def test_rotation_mode_rejects_non_string_without_representation_callback() -> None:
    """Public mode validation must fail before caller representation or Rust work."""
    loadings = np.array([[0.8, 0.1], [0.2, 0.7], [0.6, 0.3]], dtype=np.float64)

    with pytest.raises(
        ValueError, match=r"^mode must be 'orthogonal' or 'oblique'$"
    ):
        rotate_factor_loadings(loadings, criterion="geomin", mode=_HostileMode())

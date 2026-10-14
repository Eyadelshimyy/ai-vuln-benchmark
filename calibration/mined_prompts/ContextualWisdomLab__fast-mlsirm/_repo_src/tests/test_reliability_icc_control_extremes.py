"""Boundary regressions for extreme trusted ICC semantic scalars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fast_mlsirm.reliability import icc


class _RatingsSentinel:
    """Fail if local control rejection reaches ratings materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Reject every attempted NumPy materialization."""
        raise AssertionError("ratings materialization must not run")


@pytest.mark.parametrize("field", ["r0", "conf_level"])
def test_huge_builtin_integer_controls_fail_as_local_value_errors(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Reject finite-identity integers that cannot normalize to finite f64."""
    from fast_mlsirm import fitstats

    def explode_core() -> None:
        raise AssertionError("_core_module must not run")

    monkeypatch.setattr(fitstats, "_core_module", explode_core)
    with pytest.raises(ValueError, match=field):
        icc(_RatingsSentinel(), **{field: 10**10000})

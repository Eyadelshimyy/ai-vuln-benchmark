"""Fail-first resource contracts for G-theory D-study result rows."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


MAX_EXPECTED_D_STUDY_ROWS = 10_000


def _unexpected_data_materialization(*args, **kwargs):
    """Fail if an over-limit result request reaches score materialization."""
    raise AssertionError("over-limit D-study request reached score materialization")


def _unexpected_core_discovery(*args, **kwargs):
    """Fail if an over-limit result request reaches compiled-core discovery."""
    raise AssertionError("over-limit D-study request reached Rust discovery")


@pytest.mark.parametrize("entrypoint", ["gtheory_pi", "phi_lambda"])
def test_one_facet_dstudy_row_count_is_bounded_before_data_or_rust(
    monkeypatch, entrypoint: str
) -> None:
    """One-facet/Phi D-study vectors cannot request unbounded result rows."""
    monkeypatch.setattr(gtheory, "_validated_real_data", _unexpected_data_materialization)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)
    primes = [1] * (MAX_EXPECTED_D_STUDY_ROWS + 1)

    with pytest.raises(
        ValueError,
        match=rf"D-study requests exceed the {MAX_EXPECTED_D_STUDY_ROWS}-row G-theory limit",
    ):
        if entrypoint == "gtheory_pi":
            gtheory.gtheory_pi(np.empty((0, 0)), n_i_prime=primes)
        else:
            gtheory.phi_lambda(np.empty((0, 0)), 0.5, n_i_prime=primes)


def test_two_facet_dstudy_row_count_is_bounded_before_data_or_rust(monkeypatch) -> None:
    """Two-facet D-study pair vectors cannot request unbounded result rows."""
    monkeypatch.setattr(gtheory, "_validated_real_data", _unexpected_data_materialization)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)
    pairs = [(1, 1)] * (MAX_EXPECTED_D_STUDY_ROWS + 1)

    with pytest.raises(
        ValueError,
        match=rf"D-study requests exceed the {MAX_EXPECTED_D_STUDY_ROWS}-row G-theory limit",
    ):
        gtheory.gtheory_pio(np.empty((0, 0, 0)), n_prime=pairs)

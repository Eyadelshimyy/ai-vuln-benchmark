"""Response-admission regressions for sibling cognitive-diagnosis APIs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.cdm as cdm
import fast_mlsirm.fitstats as fitstats


def _unexpected_core_discovery() -> object:
    """Fail if invalid observed evidence reaches native-core discovery."""

    raise AssertionError("compiled core discovered before CDM response admission")


def _complex_binary_responses() -> np.ndarray:
    return np.array(
        [[0.0 + 1.0j, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )


def _binary_q() -> np.ndarray:
    return np.array([[1], [1]], dtype=np.int64)


def _higher_order_q() -> np.ndarray:
    # Every attribute is measured and every item has at least one required attribute.
    return np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int64)


@pytest.mark.parametrize(
    ("entrypoint", "args"),
    [
        (cdm.validate_q_matrix, (_binary_q(),)),
        (cdm.gdina_wald_selection, (_binary_q(),)),
        (cdm.fit_ho_cdm, (_higher_order_q(),)),
        (cdm.fit_ho_gdina, (_higher_order_q(),)),
        (cdm.fit_seq_gdina, (_binary_q(),)),
        (
            cdm.fit_seq_gdina_qr,
            (np.array([[1], [1]], dtype=np.int64), np.array([1, 1], dtype=np.int64)),
        ),
    ],
)
def test_sibling_cdm_entrypoints_reject_complex_responses_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Callable[..., object],
    args: tuple[object, ...],
) -> None:
    """Every calibration/selection sibling must reject imaginary evidence first."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="responses must be real-valued"):
        entrypoint(_complex_binary_responses(), *args)

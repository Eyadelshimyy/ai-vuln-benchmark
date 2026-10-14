"""Finiteness ordering for public GRM/GPCM prediction admission."""

from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm._polytomous_prediction_admission as prediction_admission
from fast_mlsirm.polytomous import PolytomousFit


@pytest.mark.parametrize(
    ("theta", "slope", "cat_params", "message"),
    [
        ([np.inf], [1.0], [[0.0]], "theta must be a non-empty finite 1-D array"),
        ([0.0], [np.inf], [[0.0]], "fit item parameters must be finite"),
        ([0.0], [1.0], [[np.nan]], "fit item parameters must be finite"),
    ],
)
def test_prediction_admission_rejects_nonfinite_evidence_before_delegate(
    theta, slope, cat_params, message
):
    """Non-finite trusted arrays must fail before the raw prediction delegate."""

    def unexpected_delegate(*args, **kwargs):
        raise AssertionError("non-finite evidence reached raw prediction delegate")

    module = SimpleNamespace(
        PolytomousFit=PolytomousFit,
        _polytomous_predictions=unexpected_delegate,
    )
    prediction_admission.install(module)
    fit = PolytomousFit("gpcm", slope, cat_params, 0.0, 0)

    with pytest.raises(ValueError, match=message):
        module._polytomous_predictions(fit, theta)

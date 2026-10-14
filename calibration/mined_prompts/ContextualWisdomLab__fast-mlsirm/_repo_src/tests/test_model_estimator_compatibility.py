"""Public configuration contracts for model-estimator compatibility."""

from __future__ import annotations

import pytest

from fast_mlsirm.config import FitConfig, VALID_ESTIMATORS, VALID_MODELS


@pytest.mark.parametrize("estimator", sorted(VALID_ESTIMATORS))
@pytest.mark.parametrize("model", sorted(VALID_MODELS))
def test_fit_config_model_estimator_compatibility_matrix(
    model: str, estimator: str
) -> None:
    """Every advertised model-estimator pair must match executable fit support."""
    if model == "BIFAC2PLM" and estimator == "jmle":
        with pytest.raises(ValueError, match=r"BIFAC2PLM.*mmle"):
            FitConfig(model=model, estimator=estimator)
        return

    config = FitConfig(model=model, estimator=estimator)
    config.validate()


def test_bifactor_jmle_fails_during_configuration_validation() -> None:
    """Bifactor JMLE must fail before response preparation or fitting work."""
    with pytest.raises(ValueError, match=r"BIFAC2PLM.*mmle"):
        FitConfig(model="BIFAC2PLM", estimator="jmle")

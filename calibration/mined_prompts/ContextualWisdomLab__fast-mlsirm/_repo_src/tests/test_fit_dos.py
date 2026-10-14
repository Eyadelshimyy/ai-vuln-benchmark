import numpy as np
import pytest
from fast_mlsirm.fit import fit
from fast_mlsirm.config import FitConfig

def test_fit_memory_dos_prevention():
    responses = np.ones((10, 2))
    factors = np.array([2000000000, 0])
    with pytest.raises(ValueError, match="factor_id implies more dimensions than items"):
        fit(responses=responses, factor_id=factors, config=FitConfig(max_iter=1))

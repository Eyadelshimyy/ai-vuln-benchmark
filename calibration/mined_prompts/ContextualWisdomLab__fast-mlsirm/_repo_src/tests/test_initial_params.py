import numpy as np

from fast_mlsirm.fit import _initial_params
from fast_mlsirm.config import FitConfig

def test_initial_params_matches_loop():
    # Setup consistent dimensions
    n_persons = 50
    n_items = 20
    n_dims = 3
    latent_dim = 2

    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.5, size=(n_persons, n_items)).astype(np.float64)
    observed = rng.binomial(1, 0.8, size=(n_persons, n_items)).astype(np.float64)
    factor_id = rng.integers(0, n_dims, size=n_items)

    config = FitConfig(latent_dim=latent_dim)

    # Run vectorized version
    rng_vect = np.random.default_rng(42)
    params_vect = _initial_params(y, observed, factor_id, n_dims, latent_dim, config, rng_vect)

    # Run loop version (simulated inside test)
    theta_loop = np.zeros((n_persons, n_dims), dtype=np.float64)
    for d in range(n_dims):
        items = factor_id == d
        denom = np.maximum(observed[:, items].sum(axis=1), 1)
        x = (y[:, items] * observed[:, items]).sum(axis=1) / denom

        # replicate standardize logic manually here since we use internal
        # fast_mlsirm.math.standardize in the actual code
        mean = np.nanmean(x)
        sd = np.nanstd(x)
        if np.isfinite(sd) and sd >= 1e-12:
            theta_loop[:, d] = (x - mean) / sd

    np.testing.assert_allclose(params_vect.theta, theta_loop, atol=1e-10)

def test_initial_params_missing_dimension():
    # Test what happens when a dimension has no items
    n_persons = 10
    n_items = 5
    n_dims = 4  # one extra dim
    latent_dim = 1

    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.5, size=(n_persons, n_items)).astype(np.float64)
    observed = np.ones((n_persons, n_items), dtype=np.float64)
    # Dimension 3 will have no items mapped to it
    factor_id = np.array([0, 1, 0, 2, 1])

    config = FitConfig(latent_dim=latent_dim)

    params_vect = _initial_params(y, observed, factor_id, n_dims, latent_dim, config, rng)

    # Dimension 3 should be all zeros since there are no items
    np.testing.assert_allclose(params_vect.theta[:, 3], np.zeros(n_persons))

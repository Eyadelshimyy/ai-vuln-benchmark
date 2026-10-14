import numpy as np

from fast_mlsirm import MLS2PLMConfig, simulate
from fast_mlsirm.math import sigmoid


def test_simulate_seed_reproducible():
    config = MLS2PLMConfig(n_persons=20, n_dims=2, items_per_dim=3, latent_dim=2, seed=42)
    a = simulate(config)
    b = simulate(config)

    assert np.array_equal(a.Y, b.Y)
    assert np.array_equal(a.factor_id, b.factor_id)
    assert np.allclose(a.truth.theta, b.truth.theta)
    assert np.allclose(a.truth.a, b.truth.a)
    assert np.allclose(a.truth.b, b.truth.b)
    assert np.allclose(a.truth.xi, b.truth.xi)
    assert np.allclose(a.truth.zeta, b.truth.zeta)


def test_gamma_zero_removes_distance_term():
    data = simulate(MLS2PLMConfig(n_persons=10, n_dims=2, items_per_dim=2, gamma=0.0, seed=7))
    manual_eta = data.truth.a[None, :] * data.truth.theta[:, data.factor_id] + data.truth.b[None, :]
    assert np.allclose(data.probabilities, sigmoid(manual_eta))


def test_gamma_distance_matches_broadcast_distance():
    gamma = 0.75
    data = simulate(MLS2PLMConfig(n_persons=8, n_dims=2, items_per_dim=3, gamma=gamma, seed=11))

    dist = np.linalg.norm(data.truth.xi[:, None, :] - data.truth.zeta[None, :, :], axis=2)
    manual_eta = data.truth.a[None, :] * data.truth.theta[:, data.factor_id] + data.truth.b[None, :] - gamma * dist

    assert np.allclose(data.probabilities, sigmoid(manual_eta))

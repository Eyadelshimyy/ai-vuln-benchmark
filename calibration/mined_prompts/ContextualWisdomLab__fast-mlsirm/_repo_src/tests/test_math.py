import numpy as np

from fast_mlsirm.math import logit, normalize_latent_positions, sigmoid, softplus, standardize
from fast_mlsirm.types import MLSIRMParams

def test_sigmoid():
    # Test basic values
    x = np.array([-10.0, 0.0, 10.0])
    res = sigmoid(x)
    assert np.allclose(res, [1 / (1 + np.exp(10)), 0.5, 1 / (1 + np.exp(-10))])

    # Test overflow stability for large negative and positive values
    x_large = np.array([-1000.0, 1000.0])
    res_large = sigmoid(x_large)
    assert np.allclose(res_large, [0.0, 1.0])


def test_softplus():
    x = np.array([-10.0, 0.0, 10.0])
    res = softplus(x)
    expected = np.log1p(np.exp(x))
    assert np.allclose(res, expected)

    # Test numerical stability
    x_large = np.array([-1000.0, 1000.0])
    res_large = softplus(x_large)
    assert np.allclose(res_large, [0.0, 1000.0])


def test_logit():
    # Test normal values
    p = np.array([0.1, 0.5, 0.9])
    res = logit(p)
    assert np.allclose(res, np.log(p / (1 - p)))

    # Test clipping
    p_edge = np.array([0.0, 1.0])
    res_edge = logit(p_edge, eps=1e-6)
    expected_edge = np.log(np.array([1e-6, 1.0 - 1e-6]) / (1.0 - np.array([1e-6, 1.0 - 1e-6])))
    assert np.allclose(res_edge, expected_edge)


def test_standardize():
    # Test normal array
    x = np.array([1.0, 2.0, 3.0])
    res = standardize(x)
    assert np.isclose(np.mean(res), 0.0)
    assert np.isclose(np.std(res), 1.0)

    # Test constant array (should return zeros)
    x_const = np.array([2.0, 2.0, 2.0])
    res_const = standardize(x_const)
    assert np.allclose(res_const, np.zeros_like(x_const))

    # Test with NaN values
    x_nan = np.array([1.0, 2.0, 3.0, np.nan])
    res_nan = standardize(x_nan)
    # The nan output should be nan, but the rest standardized based on non-nan elements
    assert np.isnan(res_nan[-1])
    assert np.isclose(np.nanmean(res_nan), 0.0)
    assert np.isclose(np.nanstd(res_nan), 1.0)


def test_normalize_latent_positions():
    # Create valid MLSIRMParams
    params = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.array([0.0]),
        b=np.array([0.0]),
        xi=np.array([[1.0, 2.0], [3.0, 4.0]]),
        zeta=np.array([[5.0, 6.0], [7.0, 8.0]]),
        tau=0.0,
    )
    res = normalize_latent_positions(params)

    combined_res = np.vstack([res.xi, res.zeta])
    # Check mean is zero
    assert np.allclose(combined_res.mean(axis=0), 0.0)
    # Check std is 1.0
    assert np.isclose(np.std(combined_res), 1.0)

    # Check tau is updated
    combined_orig = np.vstack([params.xi, params.zeta])
    center = combined_orig.mean(axis=0)
    centered = combined_orig - center
    orig_std = np.std(centered)
    assert np.isclose(res.tau, np.log(orig_std))


def test_normalize_latent_positions_empty():
    params = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.array([0.0]),
        b=np.array([0.0]),
        xi=np.array([]),
        zeta=np.array([]),
        tau=0.0,
    )
    res = normalize_latent_positions(params)
    assert res.xi.size == 0
    assert res.zeta.size == 0


def test_normalize_latent_positions_constant():
    # Create valid MLSIRMParams with constant xi and zeta
    params = MLSIRMParams(
        theta=np.array([[0.0]]),
        alpha=np.array([0.0]),
        b=np.array([0.0]),
        xi=np.array([[2.0, 2.0], [2.0, 2.0]]),
        zeta=np.array([[2.0, 2.0], [2.0, 2.0]]),
        tau=0.0,
    )
    res = normalize_latent_positions(params)

    # If std is 0, it shouldn't scale, but still centers
    combined_res = np.vstack([res.xi, res.zeta])
    assert np.allclose(combined_res.mean(axis=0), 0.0)
    assert np.allclose(combined_res, 0.0)
    assert res.tau == 0.0

import numpy as np
import pytest

from fast_mlsirm.scaling import lsr_rankings


def lsr_oracle(rankings, n, alpha=0.0):
    """Independent oracle computing LSR via continuous-time Markov chain eigenvalue."""
    A = np.full((n, n), alpha, dtype=float)
    np.fill_diagonal(A, 0.0)
    for rank in rankings:
        for i, winner in enumerate(rank[:-1]):
            rate = 1.0 / (len(rank) - i)
            for loser in rank[i + 1 :]:
                A[loser, winner] += rate
    Q = A
    np.fill_diagonal(Q, -Q.sum(axis=1))
    evals, evecs = np.linalg.eig(Q.T)
    pi = np.real(evecs[:, np.argmin(np.abs(evals))])
    weights = pi / pi.sum() * n
    log_pi = np.log(weights)
    return log_pi - np.mean(log_pi), weights


def test_lsr_rankings_numerical_oracle():
    """Asserts that lsr_rankings matches an independent Markov-chain oracle exact calculation."""
    rankings = [[0, 1, 2], [2, 0]]
    n = 3
    alpha = 0.1

    oracle_params, oracle_weights = lsr_oracle(rankings, n, alpha=alpha)
    res = lsr_rankings(rankings, n, alpha=alpha)

    np.testing.assert_allclose(res.params, oracle_params, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res.weights, oracle_weights, rtol=1e-10, atol=1e-10)
    assert res.iterations == 1


def test_lsr_rankings_public_invariants():
    """Verifies public invariants: positive finite weights, weights summing to n, and parameter centering."""
    rankings = [[0, 1, 2], [2, 0, 1], [1, 0]]
    n = 3
    res = lsr_rankings(rankings, n)

    assert np.all(np.isfinite(res.weights))
    assert np.all(res.weights > 0)

    # weights.sum() == n
    np.testing.assert_allclose(res.weights.sum(), n, rtol=1e-12, atol=1e-12)

    # centered params
    np.testing.assert_allclose(res.params.sum(), 0.0, rtol=1e-12, atol=1e-12)

    # params == log(weights) - mean(log(weights))
    expected_params = np.log(res.weights) - np.mean(np.log(res.weights))
    np.testing.assert_allclose(res.params, expected_params, rtol=1e-12, atol=1e-12)


def test_lsr_rankings_permutation_invariance_and_repeated():
    """Tests that rearranging the input order of independent rankings, or adding duplicates, behaves consistently."""
    rankings = [[0, 1, 2], [2, 1, 0], [0, 2]]
    n = 3

    # Base calculation
    res_base = lsr_rankings(rankings, n)

    # Permute order of rankings
    res_permuted = lsr_rankings([rankings[2], rankings[0], rankings[1]], n)
    np.testing.assert_allclose(
        res_base.params, res_permuted.params, rtol=1e-12, atol=1e-12
    )

    # Repeated rankings (weighting)
    res_repeated = lsr_rankings(rankings * 3, n, alpha=0.0)
    np.testing.assert_allclose(
        res_base.params, res_repeated.params, rtol=1e-12, atol=1e-12
    )

    # Non-uniform repeated rankings compared to the independent oracle
    non_uniform_rankings = rankings + [[0, 2]] * 5
    oracle_nu_params, oracle_nu_weights = lsr_oracle(
        non_uniform_rankings, n, alpha=0.0
    )
    res_nu = lsr_rankings(non_uniform_rankings, n, alpha=0.0)
    np.testing.assert_allclose(res_nu.params, oracle_nu_params, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        res_nu.weights, oracle_nu_weights, rtol=1e-10, atol=1e-10
    )


def test_lsr_rankings_invalid_cases():
    """Verifies that invalid bounds, duplicates, structural types, and broken graphs raise ValueError."""
    n = 3

    # Invalid empty rankings
    with pytest.raises(ValueError, match="at least one ranking is required"):
        lsr_rankings([], n)

    # Invalid short ranking
    with pytest.raises(ValueError, match="fewer than 2 items"):
        lsr_rankings([[0]], n)

    # Invalid duplicate items within ranking
    with pytest.raises(ValueError, match="duplicate item"):
        lsr_rankings([[0, 0]], n)

    # Negative / out-of-range bounds
    with pytest.raises(ValueError, match=">= n"):
        lsr_rankings([[0, 3]], n)

    # Invalid 'n' (e.g. 0 or 1, which cannot support pairwise graph)
    with pytest.raises(ValueError):
        lsr_rankings([[0, 1]], 1)

    # Disconnected graph at alpha=0
    with pytest.raises(
        ValueError, match="stationary distribution could not be computed"
    ):
        lsr_rankings([[0, 1]], 3, alpha=0.0)

    # Valid with alpha > 0
    assert lsr_rankings([[0, 1]], 3, alpha=0.1).weights.shape == (3,)

    # Invalid alpha (negative or non-finite)
    with pytest.raises(ValueError, match="alpha"):
        lsr_rankings([[0, 1], [1, 2], [2, 0]], n, alpha=-1.0)

    with pytest.raises(ValueError, match="alpha"):
        lsr_rankings([[0, 1], [1, 2], [2, 0]], n, alpha=np.nan)

    # Negative item indices
    with pytest.raises(ValueError, match="negative"):
        lsr_rankings([[-1, 1]], n)

    # Non-integral items (float, string, bool)
    with pytest.raises(ValueError, match="non-integer"):
        lsr_rankings([[0.5, 1.5]], n)

    with pytest.raises(ValueError, match="non-integer"):
        lsr_rankings([["a", "b"]], n)

    with pytest.raises(ValueError, match="non-integer"):
        lsr_rankings([[True, False]], n)

    with pytest.raises(ValueError, match="non-integer"):
        lsr_rankings([[np.bool_(True), np.bool_(False)]], n)

    # Invalid n type
    with pytest.raises(ValueError, match="n must be an integer"):
        lsr_rankings([[0, 1]], True)

    with pytest.raises(ValueError, match="n must be an integer"):
        lsr_rankings([[0, 1]], 2.5)

    # Non-finite alpha (inf)
    with pytest.raises(ValueError, match="finite"):
        lsr_rankings([[0, 1], [1, 2], [2, 0]], n, alpha=np.inf)

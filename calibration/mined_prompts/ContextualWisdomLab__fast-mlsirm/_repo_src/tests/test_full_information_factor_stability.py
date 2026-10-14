"""Full-information item factor model stability via the Bayes loading constraint.

Full-information item factor analysis estimates a multidimensional IRT/factor
model by marginal maximum likelihood over the latent distribution (Bock,
Gibbons, & Muraki, 1988, *Applied Psychological Measurement, 12*, 261-280). That
paper reports a specific stability requirement: **"Bayes constraints on the
factor loadings are found to be necessary to suppress Heywood cases."** A Heywood
case is a loading/discrimination driven toward divergence (an out-of-range
communality / negative unique variance) by a near-deterministic response
configuration, which an unconstrained full-information fit will chase.

fast-mlsirm's MAP penalty on the discrimination parameter (``lambda_alpha``) *is*
that Bayes constraint. This test pins the paper's property on a two-factor,
simple-structure model whose first-factor items form a near-perfect Guttman
scalogram (the Heywood trigger): as the Bayes constraint strengthens, the largest
loading magnitude is suppressed monotonically and every estimate stays finite, so
the fit never runs off to a Heywood divergence. No existing test covers this
loading-suppression stability property (``test_irt_stability.py`` and the
recovery suite exercise identified, well-conditioned fits).

The property is backend-independent (the Rust/NumPy parity gate keeps both cores
numerically identical); the NumPy reference backend is pinned here for exact,
deterministic reproducibility of the reported loading magnitudes.
"""

import numpy as np

from fast_mlsirm import FitConfig, PenaltyConfig


def _heywood_prone_responses() -> tuple[np.ndarray, np.ndarray]:
    """Return a two-factor response set whose factor-0 items nearly scale perfectly.

    Items 0-3 load on factor 0 and form a near-Guttman scalogram (each higher
    row keeps a nested prefix correct), the configuration that inflates a
    full-information factor loading toward a Heywood case; items 4-7 load on
    factor 1 and carry ordinary noisy responses so the model stays identified.
    """
    responses = np.array(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    factors = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    return responses, factors


def _fit_max_loading(lambda_alpha: float) -> tuple[float, bool]:
    """Fit the 2-factor model at one Bayes-constraint strength.

    Returns the maximum absolute loading and whether every estimate (loadings,
    difficulties, abilities, objective) is finite.
    """
    responses, factors = _heywood_prone_responses()
    from fast_mlsirm.reference import fit_reference

    result = fit_reference(
        responses,
        factors,
        config=FitConfig(
            model="MIRT",
            optimizer="adam",
            max_iter=400,
            n_restarts=1,
            latent_dim=2,
            seed=5,
            backend="numpy",
            penalty=PenaltyConfig(
                lambda_theta=0.01, lambda_b=0.01, lambda_alpha=lambda_alpha
            ),
        ),
    )
    alpha = np.asarray(result.params.alpha)
    finite = (
        bool(np.all(np.isfinite(alpha)))
        and bool(np.all(np.isfinite(result.params.b)))
        and bool(np.all(np.isfinite(result.params.theta)))
        and bool(np.isfinite(result.objective))
    )
    return float(np.max(np.abs(alpha))), finite


def test_bayes_loading_constraint_suppresses_heywood_case():
    """A stronger Bayes loading constraint monotonically bounds the loadings.

    Per Bock, Gibbons, & Muraki (1988), the Bayes constraint suppresses Heywood
    cases: as ``lambda_alpha`` increases, the largest loading shrinks monotonically
    and the fit stays finite, instead of chasing the near-Guttman item to a
    divergent discrimination.
    """
    lambdas = [1e-4, 0.1, 1.0, 5.0]
    max_loadings = []
    for lambda_alpha in lambdas:
        max_abs, finite = _fit_max_loading(lambda_alpha)
        assert finite, lambda_alpha  # full-information fit stays finite (stable)
        max_loadings.append(max_abs)

    # The Bayes constraint suppresses loading inflation monotonically (a small
    # tolerance absorbs benign optimizer noise; the endpoints carry the claim).
    for stronger, weaker in zip(max_loadings[1:], max_loadings[:-1]):
        assert stronger <= weaker + 0.1

    # Near-unpenalized, the Guttman item inflates the loading (Heywood-prone);
    # a strong Bayes constraint reins it well below any divergence.
    assert max_loadings[0] > 2.0
    assert max_loadings[-1] < 1.0
    assert max_loadings[0] > 3.0 * max_loadings[-1]

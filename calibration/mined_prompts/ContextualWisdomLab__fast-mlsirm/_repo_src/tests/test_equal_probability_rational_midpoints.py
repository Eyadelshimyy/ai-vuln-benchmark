"""One-ULP discriminator for mathematical rational midpoint probabilities.

Reference literals agree between two independent Decimal CDF/inverse methods
at 80 and 110 digits (bracket endpoints round identically). They are not a
certified interval bound or a continuous-score scientific acceptance gate.
The target is Phi^-1((2*k-1)/(2*n)), NOT reflected binary64 upper probabilities.
"""
import numpy as np

from fast_mlsirm.estimators.mmle import equal_probability_normal_nodes


def test_eleven_rational_midpoint_quantiles_within_one_ulp():
    reference = np.array([
        -1.6906216295848981, -1.0968035620935133,
        -0.7478585947633022, -0.47278912099226733,
        -0.2298841175792321, 0.0, 0.2298841175792321,
        0.47278912099226733, 0.7478585947633022,
        1.0968035620935133, 1.6906216295848981,
    ])
    actual, weights = equal_probability_normal_nodes(11)
    np.testing.assert_array_equal(actual, -actual[::-1])
    np.testing.assert_allclose(weights, 1.0 / 11, rtol=0.0, atol=0.0)
    np.testing.assert_array_max_ulp(actual, reference, maxulp=1)

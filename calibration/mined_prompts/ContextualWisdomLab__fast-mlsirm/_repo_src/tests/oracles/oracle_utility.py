"""Independent scipy oracle for selection utility analysis.

Regenerates the numeric fixtures pinned in ``tests/unit/utility_tests.rs``
and ``tests/test_paper_features.py::TestSelectionUtility``. Values are
printed at full ``repr`` precision; the test files pin these values either
verbatim or rounded to fewer digits, always within the tolerance asserted
by the test (the rho=0 fixture ``q = 0.21`` is the analytic ``sr*br``).
Run with any Python that has numpy + scipy (values were pinned with
scipy 1.x)::

    python tests/oracles/oracle_utility.py

Quantities:

- ``ux(sr) = phi(xc)/sr``, ``xc = Phi^-1(1-sr)`` (truncated-normal
  selection intensity),
- ``pux = rxy * ux(sr)`` (Naylor-Shine),
- BCG utility ``n*period*sdy*pux - cost_total``,
- Taylor-Russell success ratio ``Q(xc, yc, rho)/sr``, ``yc = Phi^-1(1-br)``.

``Q`` is computed two independent ways:

1. the multivariate-normal CDF identity
   ``Q(h,k,rho) = 1 - Phi(h) - Phi(k) + F(h,k)``, and
2. adaptive quadrature (``scipy.integrate.quad``, epsrel 1e-12) on the same
   conditional-normal integral the Rust core uses,
   ``int_h^inf phi(x) Phi((rho*x - k)/sqrt(1-rho^2)) dx`` — this is the
   oracle for the near-degenerate-|rho| regression fixtures.
"""

import math

from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm


def q_mvn(h, k, rho):
    f = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k])
    return 1.0 - norm.cdf(h) - norm.cdf(k) + f


def q_quad(h, k, rho):
    s = math.sqrt(1.0 - rho * rho)
    val, _ = quad(
        lambda x: norm.pdf(x) * norm.cdf((rho * x - k) / s),
        h,
        math.inf,
        epsabs=1e-18,
        epsrel=1e-12,
        limit=1000,
    )
    return val


def ux(sr):
    xc = norm.ppf(1 - sr)
    return norm.pdf(xc) / sr


def taylor_russell(rxy, sr, br, q=q_mvn):
    xc = norm.ppf(1 - sr)
    yc = norm.ppf(1 - br)
    qv = q(xc, yc, rxy)
    return qv / sr, qv


def bcg(n, sdy, rxy, sr, cost_total, period):
    return n * period * sdy * rxy * ux(sr) - cost_total


if __name__ == "__main__":
    def r(x):
        return repr(float(x))

    print("=== ux / pux fixtures ===")
    for sr in [0.05, 0.3, 0.5, 0.9]:
        print(f"sr={sr}: xc={r(norm.ppf(1 - sr))} ux={r(ux(sr))}")
    print(f"pux(rxy=.5, sr=.3)  = {r(0.5 * ux(0.3))}")
    print(f"pux(rxy=-.5, sr=.3) = {r(-0.5 * ux(0.3))}")

    print("\n=== Taylor-Russell fixtures (mvn-CDF oracle) ===")
    for (rxy, sr, br) in [
        (0.5, 0.5, 0.6),
        (0.5, 0.3, 0.2),
        (0.3, 0.05, 0.8),
        (-0.6, 0.3, 0.5),
        (0.8, 0.3, 0.2),
        (0.0, 0.3, 0.7),
    ]:
        s, q = taylor_russell(rxy, sr, br)
        print(f"rxy={rxy} sr={sr} br={br}: success={r(s)} q={r(q)}")
    s, _ = taylor_russell(0.7, 0.9999, 0.37, q=q_quad)
    print(f"sr->1 limit fixture (rxy=.7, sr=.9999, br=.37): success={r(s)}")

    print("\n=== BCG fixtures ===")
    print(f"bcg(1, 10000, .5, .3, 0, 1)     = {r(bcg(1, 10000, .5, .3, 0, 1))}")
    print(f"bcg(50, 8000, .4, .2, 25000, 3) = {r(bcg(50, 8000, .4, .2, 25000, 3))}")

    print("\n=== near-degenerate-rho regression fixtures (quad oracle) ===")
    for (rxy, sr, br) in [(-0.999999, 0.9, 0.9), (0.999999, 1e-12, 1e-12)]:
        s, q = taylor_russell(rxy, sr, br, q=q_quad)
        print(f"rxy={rxy} sr={sr} br={br}: success={r(s)} q={r(q)}")

    print("\n=== mutation predictions ===")
    print(f"M1 (drop rxy) pux at sr=.3: {r(ux(0.3))} vs good {r(0.5 * ux(0.3))}")
    xc03, yc08 = norm.ppf(1 - 0.3), norm.ppf(1 - 0.8)
    print(
        "M4 (sr<->br role swap) at rho=.5, sr=.3, br=.8: "
        f"good success={r(q_mvn(xc03, yc08, 0.5) / 0.3)} "
        f"mutant={r(q_mvn(yc08, xc03, 0.5) / 0.8)}"
    )

"""Paired-comparison scaling: Thurstone (1927) Case V, Bradley-Terry, and
Luce Spectral Ranking (LSR / I-LSR).

Thin wrappers over the Rust core (``mlsirm_core::scaling``). The Thurstone
algorithm follows the ``thurstone()`` function of the psych R package
(Revelle, 2025; source READ), which implements Thurstone's (1927) law of
comparative judgment under Case V (equal discriminal dispersions, zero
correlations). The Bradley-Terry maximum-likelihood fit uses the
minorization-maximization (MM) algorithm as implemented by choix 0.4.1
(Maystre, 2015-2020; ``opt.mm`` pairwise path, source READ), which choix
attributes to Hunter (2004). The LSR and I-LSR estimators follow choix's
``lsr.py`` dense pairwise path, which choix attributes to Maystre and
Grossglauser (2015). Thurstone (1927), Bradley and Terry (1952), Hunter
(2004), and Maystre and Grossglauser (2015) themselves were NOT read; they
are cited as the origins of the models/algorithms as described by the read
sources.

References
----------
Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block
    designs: I. The method of paired comparisons. Biometrika, 39(3/4),
    324-345. [NOT READ; cited as described in choix source]
Hunter, D. R. (2004). MM algorithms for generalized Bradley-Terry models.
    The Annals of Statistics, 32(1), 384-406. [NOT READ; cited as described
    in choix source]
Maystre, L. (2020). choix: Inference algorithms for models based on Luce's
    choice axiom (Python package, version 0.4.1).
    https://github.com/lucasmaystre/choix [READ]
Maystre, L., & Grossglauser, M. (2015). Fast and accurate inference of
    Plackett-Luce models. Advances in Neural Information Processing
    Systems, 28, 172-180. [NOT READ; cited as described in choix source]
Revelle, W. (2025). psych: Procedures for psychological, psychometric, and
    personality research (R package). https://CRAN.R-project.org/package=psych
Thurstone, L. L. (1927). A law of comparative judgment. Psychological
    Review, 34(4), 273-286. [NOT READ; cited as described in psych source]

The circular-triads consistency test and Kendall's coefficient of agreement
follow eba 1.10-0 (Wickelmaier & Schmid; ``circular.R`` and ``kendall.u.R``,
source READ). Kendall and Babington Smith (1940) and Alway (1962, exact null
tables) were NOT read; they are cited as the origins as described by the eba
source and manual pages.

Kendall, M. G., & Babington Smith, B. (1940). On the method of paired
    comparisons. Biometrika, 31(3/4), 324-345. [NOT READ; cited as described
    in eba source]
Wickelmaier, F., & Schmid, C. (2004). A Matlab function to estimate choice
    model parameters from paired-comparison data. Behavior Research Methods,
    Instruments, and Computers, 36(1), 29-40. [eba package source READ]

The Elo rating system follows the CRAN PlayerRatings 1.1-0 package's
``elo()`` (Stephenson & Sonas, 2020; ``R/ratings.R`` and the ``elo_c`` C
kernel, source READ): batch-per-period updates where all expected scores
within a period use the period-start ratings. Elo (1978) was NOT read; it
is cited as the origin of the method as described by PlayerRatings.

The Glicko rating system follows the same package's ``glicko()`` (``glicko_c``
kernel, source READ) and Glickman's technical note *The Glicko system*
(READ; http://www.glicko.net/glicko/glicko.pdf), whose worked example the
implementation reproduces. Glickman (1999), the derivation paper, was NOT
read; it is cited as the method's origin as described by both READ sources.

Elo, A. E. (1978). The rating of chessplayers, past and present. Arco.
    [NOT READ; cited as described in PlayerRatings source]
Glickman, M. E. (n.d.). The Glicko system [Technical note]. Harvard
    University. http://www.glicko.net/glicko/glicko.pdf [READ]
Glickman, M. E. (1999). Parameter estimation in large dynamic paired
    comparison experiments. Applied Statistics, 48(3), 377-394. [NOT READ;
    cited as described in the READ sources]
Stephenson, A., & Sonas, J. (2020). PlayerRatings: Dynamic updating methods
    for player ratings estimation (R package, version 1.1-0).
    https://CRAN.R-project.org/package=PlayerRatings [READ]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThurstoneResult:
    """Case V scaling result: ``scale[j]`` is the scale value of object j
    (minimum shifted to exactly 0), ``gof`` the psych goodness of fit
    ``1 - sse/ssc`` over the FULL model matrix (including the diagonal --
    this pins the psych *code* behavior; the .Rd prose saying "lower off
    diagonal" is stale), ``model`` the fitted choice probabilities
    ``Phi(scale[j] - scale[i])`` and ``residual = model - choice``, both
    shaped (n, n)."""

    scale: np.ndarray
    gof: float
    model: np.ndarray
    residual: np.ndarray


def thurstone_case_v(choice) -> ThurstoneResult:
    """Scale n objects from an n x n choice-probability matrix.

    ``choice[i, j]`` is the proportion of judges preferring object *j* over
    object *i* (psych convention: column beats row). All entries must be
    strictly in (0, 1) -- a deliberate safety divergence from psych, whose
    direct path lets ``qnorm(0)/qnorm(1)`` produce infinities.
    """
    from .fitstats import _core_module

    arr = np.asarray(choice)
    if np.iscomplexobj(arr):
        raise ValueError("thurstone_case_v: choice must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "thurstone_case_v: choice must be numeric"
            ) from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("thurstone_case_v: choice must be a square 2-D matrix")
    n = arr.shape[0]
    core = _core_module()
    res = core.thurstone_case_v(arr.ravel(), n)
    return ThurstoneResult(
        scale=np.asarray(res["scale"], dtype=np.float64),
        gof=float(res["gof"]),
        model=np.asarray(res["model"], dtype=np.float64).reshape(n, n),
        residual=np.asarray(res["residual"], dtype=np.float64).reshape(n, n),
    )


@dataclass
class BradleyTerryResult:
    """Bradley-Terry MM fit: ``params[i]`` is the centered log-worth of item
    i (mean exactly 0), ``weights[i] = n * softmax(params)[i]`` the exp-scale
    worths (sum exactly n, choix convention), ``iterations`` the number of MM
    updates performed when the L1 convergence criterion
    ``sum |new - prev| <= tol * n`` first fired (the check compares two
    consecutive *updates*, so at least two are always performed)."""

    params: np.ndarray
    weights: np.ndarray
    iterations: int


def bradley_terry_mm(wins, alpha=0.0, max_iter=10000, tol=1e-8):
    """Fit Bradley-Terry worths from an n x n win-count matrix.

    ``wins[i, j]`` is the number of times item *i* beat item *j* (row beats
    column). Counts must be finite and non-negative with a zero diagonal;
    non-integer counts are accepted (weighted comparisons -- a DERIVED
    extension of choix's pair-list interface). ``alpha`` is choix's
    regularization added to both numerator and denominator of each MM update;
    with ``alpha = 0`` an item with zero wins has no finite log-worth and a
    ValueError is raised. An all-zero matrix is rejected for every alpha
    (deliberate divergence from choix, which returns the uniform solution
    when alpha > 0). Non-convergence within ``max_iter`` raises (e.g. when an
    item never loses, violating Ford's (1957) strong-connectivity condition
    as cited by choix; Ford 1957 NOT READ).
    """
    from .fitstats import _core_module

    arr = np.asarray(wins)
    if np.iscomplexobj(arr):
        raise ValueError("bradley_terry_mm: wins must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("bradley_terry_mm: wins must be numeric") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("bradley_terry_mm: wins must be a square 2-D matrix")
    n = arr.shape[0]
    core = _core_module()
    res = core.bradley_terry_mm(
        arr.ravel(), n, float(alpha), int(max_iter), float(tol)
    )
    return BradleyTerryResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


@dataclass
class BrattResult:
    """Bradley-Terry-with-ties (VGAM ``bratt``) MM fit: ``alpha[i]`` is the
    worth of contestant i (``alpha[ref_index] == ref_value``), ``alpha0`` the
    additive tie parameter on the same joint scale, ``iterations`` the number
    of MM updates performed when the max absolute parameter change (across
    ``alpha`` AND ``alpha0``, after the reference rescale) first fell to
    ``tol``, and ``log_likelihood`` the model log-likelihood at the returned
    parameters."""

    alpha: np.ndarray
    alpha0: float
    iterations: int
    log_likelihood: float


def bratt_mm(wins, ties, ref_index=0, ref_value=1.0, max_iter=10000, tol=1e-10):
    """Fit the Bradley-Terry model with ties (additive ``alpha0``) by MM.

    Model (VGAM 1.1-14 ``bratt()`` family, R source READ; Bradley & Terry
    1952 NOT READ, cited as the model origin): ``P(i beats j) =
    alpha_i / (alpha_i + alpha_j + alpha0)`` and ``P(i ties j) =
    alpha0 / (alpha_i + alpha_j + alpha0)``. This additive-``alpha0`` ties
    model is NOT the Rao-Kupper or Davidson ties model (neither read; named
    only to disambiguate). The MM ascent is hand-derived using the same
    supporting-hyperplane pattern as :func:`bradley_terry_mm`.

    ``wins[i, j]`` is the (possibly fractional, nonnegative) count of wins
    of *i* over *j*; ``ties[i, j]`` the tie count of the unordered pair,
    stored symmetrically. Diagonals must be zero. Data with no ties at all
    are rejected (``alpha0`` has no positive MLE -- use
    :func:`bradley_terry_mm`; this is a contract of this API, not VGAM
    behavior), as is any contestant with zero wins (its ML worth is 0,
    outside the positive parameter space).
    """
    from .fitstats import _core_module

    mats = []
    for name, x in (("wins", wins), ("ties", ties)):
        if isinstance(x, np.ma.MaskedArray):
            raise ValueError(f"bratt_mm: {name} must not be a masked array")
        arr = np.asarray(x)
        if np.iscomplexobj(arr):
            raise ValueError(f"bratt_mm: {name} must be real-valued")
        if arr.dtype == object:
            if any(isinstance(v, (bool, np.bool_)) for v in arr.flat):
                raise ValueError(f"bratt_mm: {name} must be numeric, not boolean")
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bratt_mm: {name} must be numeric") from exc
        if arr.dtype.kind not in "fiu":
            raise ValueError(f"bratt_mm: {name} must be numeric (got {arr.dtype})")
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"bratt_mm: {name} must be a square 2-D matrix")
        mats.append(arr)
    w, t = mats
    if w.shape != t.shape:
        raise ValueError("bratt_mm: wins and ties must have the same shape")
    if isinstance(ref_index, bool) or not isinstance(ref_index, (int, np.integer)):
        raise ValueError("bratt_mm: ref_index must be an integer")
    if ref_index < 0:
        raise ValueError("bratt_mm: ref_index must be nonnegative")
    n = w.shape[0]
    core = _core_module()
    res = core.bratt_mm(
        w.ravel(),
        t.ravel(),
        n,
        int(ref_index),
        float(ref_value),
        int(max_iter),
        float(tol),
    )
    return BrattResult(
        alpha=np.asarray(res["alpha"], dtype=np.float64),
        alpha0=float(res["alpha0"]),
        iterations=int(res["iterations"]),
        log_likelihood=float(res["log_likelihood"]),
    )


@dataclass
class LsrResult:
    """Luce Spectral Ranking fit: ``params[i]`` is the centered log-worth of
    item i (mean exactly 0), ``weights[i]`` the stationary distribution of
    the LSR Markov chain scaled to sum exactly n (choix ``statdist``
    convention), ``iterations`` the number of LSR passes (1 for the one-shot
    spectral estimator; the pass count when the L1 criterion
    ``sum |new - prev| <= tol * n`` first fired for I-LSR, which always
    performs at least two passes)."""

    params: np.ndarray
    weights: np.ndarray
    iterations: int


def _lsr_validate(name, wins):
    arr = np.asarray(wins)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name}: wins must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: wins must be numeric") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}: wins must be a square 2-D matrix")
    return arr


def lsr_pairwise(wins, alpha=0.0):
    """One-shot Luce Spectral Ranking from an n x n win-count matrix.

    ``wins[i, j]`` is the number of times item *i* beat item *j* (row beats
    column; zero diagonal; non-integer counts accepted -- same DERIVED
    extension as :func:`bradley_terry_mm`). Builds the LSR Markov chain
    (rate ``c / (w_i + w_j)`` on each loser-to-winner edge under uniform
    initial worths, plus ``alpha`` everywhere as a regularizer) and returns
    the centered log of its stationary distribution (choix
    ``lsr_pairwise_dense``). Raises ValueError when the stationary
    distribution does not exist or is not unique -- e.g. a disconnected
    comparison graph at ``alpha = 0`` (choix raises there too) -- and on
    overflow from huge counts or alpha. NOTE: ``alpha`` regularizes the
    chain rates, which is NOT the Dirichlet-MAP ``alpha`` of
    :func:`bradley_terry_mm`; the two disagree for ``alpha > 0``.
    """
    from .fitstats import _core_module

    arr = _lsr_validate("lsr_pairwise", wins)
    res = _core_module().lsr_pairwise(arr.ravel(), arr.shape[0], float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


def ilsr_pairwise(wins, alpha=0.0, max_iter=100, tol=1e-8):
    """Iterative Luce Spectral Ranking (Bradley-Terry MLE) from an n x n
    win-count matrix.

    Repeats the LSR pass of :func:`lsr_pairwise`, feeding each pass the
    worths from the previous one, until the L1 change is ``<= tol * n``
    (choix ``ilsr_pairwise_dense``; default ``max_iter=100`` as in choix).
    At ``alpha = 0`` the fixed point is the Bradley-Terry maximum-likelihood
    estimate and agrees with :func:`bradley_terry_mm`; for ``alpha > 0`` the
    two regularization paths deliberately differ (each follows its source).
    Raises ValueError on invalid input, a disconnected comparison graph at
    ``alpha = 0``, overflow, or non-convergence within ``max_iter`` passes.
    """
    from .fitstats import _core_module

    arr = _lsr_validate("ilsr_pairwise", wins)
    res = _core_module().ilsr_pairwise(
        arr.ravel(), arr.shape[0], float(alpha), int(max_iter), float(tol)
    )
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )

def rank_centrality(wins, alpha=0.0):
    """Rank Centrality: spectral ranking from the *ratios* of pairwise wins.

    ``wins[i, j]`` is the number of times item *i* beat item *j* (row beats
    column; zero diagonal). The Markov chain accrues, on each
    loser-to-winner edge, the win *ratio*
    ``(alpha + c_win) / (2 * alpha + c_win + c_lose)`` -- choix
    ``rank_centrality``, which ports Negahban, Oh, & Shah's (2017)
    algorithm as a continuous-time chain (the paper's discrete-time
    max-degree walk is NOT what choix, or therefore this port, computes;
    only the choix variant is implemented and verified). Returns the
    centered log of the stationary distribution as :class:`LsrResult`
    (``iterations`` always 1).

    At ``alpha = 0`` the result is exactly invariant under a global
    rescaling of all counts; for fixed ``alpha > 0`` it is not. Raises
    ValueError on invalid input, a disconnected comparison graph at
    ``alpha = 0``, or overflowing intermediates (e.g. ``alpha = 1e308``,
    where the ratio denominator overflows -- choix instead silently
    produces near-zero ratios). An all-zero wins matrix is rejected even
    when ``alpha > 0`` (documented divergence from choix, which would
    regularize it to a uniform chain).
    """
    from .fitstats import _core_module

    arr = _lsr_validate("rank_centrality", wins)
    res = _core_module().rank_centrality(arr.ravel(), arr.shape[0], float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )

# Live CSR payload budget for ranking materialization: flattened item
# indices plus CSR start offsets, each stored as fixed-width uint64
# (8 bytes). This is an input-resource ceiling for the Python→Rust handoff
# arrays only; it does not claim a process-wide memory capacity.
MAX_RANKING_CSR_BYTES = 8 * 1024 * 1024


def _ranking_csr_budget_allows(flat_count: int, start_count: int) -> bool:
    """Return True when (flat + starts) * 8 fits under the CSR byte ceiling."""
    total = flat_count + start_count
    # Division-before-multiplication avoids oversized intermediate products.
    return total <= (MAX_RANKING_CSR_BYTES // 8)


def _rankings_to_csr(name, rankings, n):
    """Validate rankings (best first) and CSR-flatten to bounded u64 storage.

    Rejects, BEFORE any unsigned cast: non-integer entries, negative
    indices (a documented divergence -- Python's negative indices would
    silently wrap in choix), booleans, complex/object dtypes, rankings
    shorter than 2 items (choix silently no-ops those), overlong rankings
    (> n items), out-of-range items, and streams that would exceed
    :data:`MAX_RANKING_CSR_BYTES` of live flat/start uint64 payload.
    Ordinary caller iteration failures become stable ``ValueError`` text
    without reflecting the rejected payload. Process-control exceptions
    (``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit``) propagate.
    Duplicate detection within a ranking is enforced by the Rust core
    (documented divergence: choix accepts duplicates whenever the chain
    stays connected).

    Implementation uses a validate-then-allocate handoff: pure-Python integer
    lists accumulate admitted rankings, then exact-size ``uint64`` flat/start
    arrays are allocated once. That avoids reallocation peaks where old and
    replacement buffers would both be live above the declared byte ceiling,
    and avoids list→``uint64`` temporaries beside the live CSR payload.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or int(n) < 2:
        raise ValueError(f"{name}: n must be an integer >= 2")
    n = int(n)
    if n > 10_000:
        # Mirrors the Rust dense-chain cap BEFORE any uint64/usize cast,
        # so a huge n raises ValueError, never a raw OverflowError.
        raise ValueError(f"{name}: n = {n} exceeds the 10000-item cap")

    validated: list[list[int]] = []
    flat_count = 0

    try:
        ranking_iter = iter(rankings)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        raise ValueError(f"{name}: ranking iteration failed") from None

    while True:
        try:
            ranking = next(ranking_iter)
        except StopIteration:
            break
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise ValueError(f"{name}: ranking iteration failed") from None

        r = len(validated)
        # Budget one additional start offset for this ranking before reading items.
        if not _ranking_csr_budget_allows(flat_count, len(validated) + 2):
            raise ValueError(
                f"{name}: ranking CSR byte limit exceeded "
                f"(MAX_RANKING_CSR_BYTES={MAX_RANKING_CSR_BYTES})"
            )

        try:
            item_iter = iter(ranking)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise ValueError(f"{name}: ranking iteration failed") from None

        ranking_items: list[int] = []
        for _ in range(n + 1):
            try:
                x = next(item_iter)
            except StopIteration:
                break
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                raise ValueError(f"{name}: ranking iteration failed") from None

            if isinstance(x, (bool, np.bool_)) or np.iscomplexobj(np.asarray(x)):
                raise ValueError(f"{name}: ranking {r} has a non-integer item")
            try:
                xi = int(x)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{name}: ranking {r} has a non-integer item")
            if xi != x:
                raise ValueError(f"{name}: ranking {r} has a non-integer item")
            if xi < 0:
                raise ValueError(
                    f"{name}: ranking {r} has a negative item index "
                    "(negative indices do not wrap here)"
                )
            if xi >= n:
                raise ValueError(f"{name}: ranking {r} has item {xi} >= n = {n}")

            if not _ranking_csr_budget_allows(
                flat_count + len(ranking_items) + 1, len(validated) + 2
            ):
                raise ValueError(
                    f"{name}: ranking CSR byte limit exceeded "
                    f"(MAX_RANKING_CSR_BYTES={MAX_RANKING_CSR_BYTES})"
                )
            ranking_items.append(xi)
        else:
            # Consumed n+1 without StopIteration => overlong ranking.
            raise ValueError(
                f"{name}: ranking {r} has more than n = {n} items"
            )

        if len(ranking_items) < 2:
            raise ValueError(
                f"{name}: ranking {r} has fewer than 2 items "
                "(choix silently ignores such rankings; this port rejects them)"
            )

        validated.append(ranking_items)
        flat_count += len(ranking_items)

    if not validated:
        raise ValueError(f"{name}: at least one ranking is required")

    start_count = len(validated) + 1
    if not _ranking_csr_budget_allows(flat_count, start_count):
        raise ValueError(
            f"{name}: ranking CSR byte limit exceeded "
            f"(MAX_RANKING_CSR_BYTES={MAX_RANKING_CSR_BYTES})"
        )

    # Exact-size CSR handoff arrays only. Allocate once so reallocation peaks
    # cannot exceed the declared live payload ceiling.
    flat = np.empty(flat_count, dtype=np.uint64)
    starts = np.empty(start_count, dtype=np.uint64)
    starts[0] = 0
    pos = 0
    for i, ranking_items in enumerate(validated):
        for offset, xi in enumerate(ranking_items):
            flat[pos + offset] = xi
        pos += len(ranking_items)
        starts[i + 1] = pos

    # Already contiguous exact-size arrays; keep the historical return shape.
    flat_out = np.ascontiguousarray(flat, dtype=np.uint64)
    starts_out = np.ascontiguousarray(starts, dtype=np.uint64)
    return flat_out, starts_out, n



def lsr_rankings(rankings, n, alpha=0.0):
    """Luce Spectral Ranking for full or partial rankings (one shot).

    ``rankings`` is an iterable of rankings, each an iterable of item
    indices in ``0..n-1`` ordered best first (partial rankings -- any
    length >= 2 -- are allowed and are what distinguishes the
    Plackett-Luce subset denominator from a naive all-items one). Ports
    choix 0.4.1 ``lsr_rankings``: each ranking is a sequence of Luce
    choices; position *i* accrues rate ``1 / (sum of remaining ranked
    worths)`` on every loser-to-winner edge, plus ``alpha`` everywhere.
    Returns the centered log stationary distribution as
    :class:`LsrResult` (``iterations`` always 1). Raises ValueError on
    invalid input, within-ranking duplicates, a disconnected item graph
    at ``alpha = 0``, or overflow.
    """
    from .fitstats import _core_module

    rk, st, n = _rankings_to_csr("lsr_rankings", rankings, n)
    res = _core_module().lsr_rankings(rk, st, n, float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


def ilsr_rankings(rankings, n, alpha=0.0, max_iter=100, tol=1e-8):
    """Iterative LSR for rankings (Plackett-Luce MLE at ``alpha = 0``).

    Repeats the :func:`lsr_rankings` pass, feeding each pass the worths
    from the previous one, until the L1 parameter change is
    ``<= tol * n`` (choix 0.4.1 ``ilsr_rankings``; defaults match choix).
    Raises ValueError on invalid input, a disconnected item graph at
    ``alpha = 0``, overflow, or non-convergence within ``max_iter``.
    """
    from .fitstats import _core_module

    rk, st, n = _rankings_to_csr("ilsr_rankings", rankings, n)
    res = _core_module().ilsr_rankings(
        rk, st, n, float(alpha), int(max_iter), float(tol)
    )
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )

def _top1_to_csr(name, data, n):
    """Validate top-1 choice data and CSR-flatten to bounded u64 arrays.

    ``data`` is an iterable of ``(winner, losers)`` pairs. Rejects,
    BEFORE any unsigned cast: non-integer entries (bool/np.bool_,
    complex, non-integral floats, int() overflow), negative indices (a
    documented divergence -- Python's negative indices would silently
    wrap in choix), empty loser sets (choix silently no-ops those),
    overlong loser streams (more than ``n - 1`` items), out-of-range
    indices, and streams that would exceed
    :data:`MAX_RANKING_CSR_BYTES` of live winner/loser/start uint64
    payload. Ordinary outer/inner iteration failures become stable
    package-owned ``ValueError`` text without reflecting the rejected
    payload. Process-control exceptions
    (``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit``) propagate.
    Winner-in-losers and duplicate-loser detection is enforced by the
    Rust core (documented divergences: choix accepts both, silently
    corrupting the denominator).

    Implementation validates into pure-Python integer structures first,
    then allocates exact-size ``uint64`` winner/loser/start arrays once
    so reallocation peaks and list→``uint64`` temporaries do not exceed
    the declared CSR ceiling.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or int(n) < 2:
        raise ValueError(f"{name}: n must be an integer >= 2")
    n = int(n)
    if n > 10_000:
        # Mirrors the Rust dense-chain cap BEFORE any uint64/usize cast,
        # so a huge n raises ValueError, never a raw OverflowError.
        raise ValueError(f"{name}: n = {n} exceeds the 10000-item cap")

    def _index(x, what, r):
        if isinstance(x, (bool, np.bool_)) or np.iscomplexobj(np.asarray(x)):
            raise ValueError(f"{name}: observation {r} has a non-integer {what}")
        try:
            xi = int(x)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name}: observation {r} has a non-integer {what}")
        if xi != x:
            raise ValueError(f"{name}: observation {r} has a non-integer {what}")
        if xi < 0:
            raise ValueError(
                f"{name}: observation {r} has a negative {what} index "
                "(negative indices do not wrap here)"
            )
        if xi >= n:
            raise ValueError(f"{name}: observation {r} has {what} {xi} >= n = {n}")
        return xi

    def _top1_budget_allows(winner_count: int, loser_count: int, start_count: int) -> bool:
        # Winner ids + loser flat + start offsets, each fixed-width uint64.
        total = winner_count + loser_count + start_count
        return total <= (MAX_RANKING_CSR_BYTES // 8)

    winners: list[int] = []
    losers_flat: list[int] = []
    starts: list[int] = [0]

    try:
        observation_iter = iter(data)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        raise ValueError(f"{name}: top-1 observation iteration failed") from None

    while True:
        try:
            obs = next(observation_iter)
        except StopIteration:
            break
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise ValueError(f"{name}: top-1 observation iteration failed") from None

        r = len(winners)
        # Budget the next winner id + start offset before reading losers.
        if not _top1_budget_allows(len(winners) + 1, len(losers_flat), len(starts) + 1):
            raise ValueError(
                f"{name}: top-1 CSR byte limit exceeded "
                f"(MAX_RANKING_CSR_BYTES={MAX_RANKING_CSR_BYTES})"
            )

        try:
            winner, losers = obs
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise ValueError(f"{name}: top-1 observation iteration failed") from None

        winner_idx = _index(winner, "winner", r)

        try:
            loser_iter = iter(losers)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise ValueError(f"{name}: loser iteration failed") from None

        ranking_losers: list[int] = []
        # At most n-1 losers in a complete choice set that excludes the winner.
        for _ in range(n):
            try:
                x = next(loser_iter)
            except StopIteration:
                break
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                raise ValueError(f"{name}: loser iteration failed") from None

            if not _top1_budget_allows(
                len(winners) + 1,
                len(losers_flat) + len(ranking_losers) + 1,
                len(starts) + 1,
            ):
                raise ValueError(
                    f"{name}: top-1 CSR byte limit exceeded "
                    f"(MAX_RANKING_CSR_BYTES={MAX_RANKING_CSR_BYTES})"
                )
            ranking_losers.append(_index(x, "loser", r))
        else:
            # Consumed n loser items without StopIteration => overlong loser set.
            raise ValueError(f"{name}: loser set has more than n - 1 items")

        if not ranking_losers:
            raise ValueError(
                f"{name}: observation {r} has an empty loser set "
                "(choix silently ignores such observations; this port rejects them)"
            )

        winners.append(winner_idx)
        losers_flat.extend(ranking_losers)
        starts.append(len(losers_flat))

    if len(starts) < 2:
        raise ValueError(f"{name}: at least one observation is required")

    if not _top1_budget_allows(len(winners), len(losers_flat), len(starts)):
        raise ValueError(
            f"{name}: top-1 CSR byte limit exceeded "
            f"(MAX_RANKING_CSR_BYTES={MAX_RANKING_CSR_BYTES})"
        )

    # Exact-size handoff arrays allocated once under the CSR ceiling.
    winners_out = np.empty(len(winners), dtype=np.uint64)
    losers_out = np.empty(len(losers_flat), dtype=np.uint64)
    starts_out = np.empty(len(starts), dtype=np.uint64)
    for i, value in enumerate(winners):
        winners_out[i] = value
    for i, value in enumerate(losers_flat):
        losers_out[i] = value
    for i, value in enumerate(starts):
        starts_out[i] = value
    return (
        np.ascontiguousarray(winners_out, dtype=np.uint64),
        np.ascontiguousarray(losers_out, dtype=np.uint64),
        np.ascontiguousarray(starts_out, dtype=np.uint64),
        n,
    )



def lsr_top1(data, n, alpha=0.0):
    """Luce Spectral Ranking for top-1 choice data (one shot).

    ``data`` is an iterable of ``(winner, losers)`` pairs: the winner
    was chosen out of the choice set ``{winner} | set(losers)``. Ports
    choix 0.4.1 ``lsr_top1``: each observation accrues rate
    ``1 / (sum of choice-set worths)`` on every loser-to-winner edge,
    plus ``alpha`` everywhere off-diagonal. Returns the centered log
    stationary distribution as :class:`LsrResult` (``iterations``
    always 1). Raises ValueError on invalid input, empty loser sets,
    winner-in-losers, duplicate losers, a disconnected item graph at
    ``alpha = 0``, or overflow.
    """
    from .fitstats import _core_module

    wn, ls, st, n = _top1_to_csr("lsr_top1", data, n)
    res = _core_module().lsr_top1(wn, ls, st, n, float(alpha))
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


def ilsr_top1(data, n, alpha=0.0, max_iter=100, tol=1e-8):
    """Iterative LSR for top-1 choice data (Luce-choice MLE at alpha=0).

    Repeats the :func:`lsr_top1` pass, feeding each pass the worths from
    the previous one, until the L1 parameter change is ``<= tol * n``
    (choix 0.4.1 ``ilsr_top1``; defaults match choix). Raises ValueError
    on invalid input, a disconnected item graph at ``alpha = 0``,
    overflow, or non-convergence within ``max_iter``.
    """
    from .fitstats import _core_module

    wn, ls, st, n = _top1_to_csr("ilsr_top1", data, n)
    res = _core_module().ilsr_top1(
        wn, ls, st, n, float(alpha), int(max_iter), float(tol)
    )
    return LsrResult(
        params=np.asarray(res["params"], dtype=np.float64),
        weights=np.asarray(res["weights"], dtype=np.float64),
        iterations=int(res["iterations"]),
    )


@dataclass
class CircularTriadsResult:
    """Circular-triads consistency test (Kendall & Babington Smith, 1940,
    as implemented by eba's ``circular()``): ``t`` the number of circular
    triads T, ``t_max`` its maximum, ``t_exp`` its null expectation
    C(n,3)/4, ``zeta = 1 - t/t_max`` the consistency coefficient, and the
    test of the null "preferences are random". For ``n <= 10`` the p-value
    is EXACT (``exact=True``, ``chi2``/``df`` are NaN); for ``n >= 11`` a
    chi-square approximation is used and ``chi2`` / ``df`` are filled."""

    t: float
    t_max: float
    t_exp: float
    zeta: float
    chi2: float
    df: float
    p_value: float
    exact: bool


def _kendall_matrix(name, mat):
    arr = np.asarray(mat)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name}: mat must be real-valued")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: mat must be numeric") from exc
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name}: mat must be a square 2-D matrix")
    return arr, arr.shape[0]


def circular_triads(mat, alternative="two.sided", correct=True):
    """Test one judge's paired comparisons for intransitivity.

    ``mat[i, j] = 1`` means object *i* was preferred over object *j*; the
    tournament must be complete (``mat[i, j] + mat[j, i] == 1`` for every
    pair, zero diagonal). ``alternative`` is one of ``"two.sided"``,
    ``"less"`` (fewer circular triads than chance, i.e. consistency), or
    ``"greater"``. ``correct`` applies the +-0.5 continuity correction on
    the chi-square path (``n >= 11``) only. Deliberate divergences from
    eba's ``circular()``: ``n = 2`` is rejected (T_max = 0 makes zeta
    undefined) and malformed tournaments raise instead of returning
    garbage. Raises ValueError on invalid input.
    """
    from .fitstats import _core_module

    arr, n = _kendall_matrix("circular_triads", mat)
    res = _core_module().circular_triads(
        arr.ravel(), n, str(alternative), bool(correct)
    )
    return CircularTriadsResult(
        t=float(res["t"]),
        t_max=float(res["t_max"]),
        t_exp=float(res["t_exp"]),
        zeta=float(res["zeta"]),
        chi2=float(res["chi2"]),
        df=float(res["df"]),
        p_value=float(res["p_value"]),
        exact=bool(res["exact"]),
    )


@dataclass
class KendallUResult:
    """Kendall's coefficient of agreement u between m judges (Kendall &
    Babington Smith, 1940, as implemented by eba's ``kendall.u()``):
    ``sigma`` the number of agreeing judge pairs, ``u`` the agreement
    coefficient (1 = perfect agreement; minimum ``min_u`` is -1/m for odd
    m, -1/(m-1) for even m), and the chi-square test of the null "agreement
    is by chance". ``chi2`` is returned RAW and can be negative under
    strong disagreement with the continuity correction; only the p-value
    clamps (a negative statistic gives p = 1)."""

    sigma: float
    u: float
    min_u: float
    chi2: float
    df: float
    p_value: float


def kendall_u(mat, correct=True):
    """Agreement between m judges from an n x n frequency matrix.

    ``mat[i, j]`` counts the judges who preferred object *i* over object
    *j*. Every pair must have the same number of observations
    ``m = mat[i, j] + mat[j, i] >= 3`` (a documented STRICTER check than
    eba, which reads m from the first pair only); entries must be
    nonnegative integers and the diagonal zero. ``correct`` applies the
    continuity correction (subtract 1 from Sigma). Raises ValueError on
    invalid input.
    """
    from .fitstats import _core_module

    arr, n = _kendall_matrix("kendall_u", mat)
    res = _core_module().kendall_u(arr.ravel(), n, bool(correct))
    return KendallUResult(
        sigma=float(res["sigma"]),
        u=float(res["u"]),
        min_u=float(res["min_u"]),
        chi2=float(res["chi2"]),
        df=float(res["df"]),
        p_value=float(res["p_value"]),
    )

@dataclass
class EloResult:
    """Elo ratings and bookkeeping (Elo, 1978, as implemented by the CRAN
    PlayerRatings package's ``elo()``; batch-per-period update). Wins,
    draws, and losses count only scores exactly 1, 0.5, and 0; other
    fractional scores count a game without a W/D/L. ``lag`` is the number
    of rating periods since the player's last appearance (0 for players
    appearing in the final period or never playing)."""

    ratings: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def elo_rating(games, n_players, init=2200.0, kfac=27.0, gamma=None):
    """Elo ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]``: an integer
    rating-period label, integer player indices in ``0..n_players``, and
    white's score in [0, 1]. Rows are grouped by ascending period value
    (matching R ``split()`` ordering); within a period all expected scores
    use the period-start ratings (batch update). All players start at
    ``init``; ``kfac`` is the scalar K factor (any finite value, including
    0); ``gamma`` is white's per-game advantage — a scalar (broadcast) or a
    length-g array. Defaults ``init=2200, kfac=27`` are the PlayerRatings
    defaults. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("elo_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elo_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"elo_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("elo_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("elo_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"elo_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("elo_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above 2**53 (distinct labels would silently merge into one
    # rating period). Take periods losslessly from an integer input array,
    # and otherwise reject labels the float path cannot represent exactly.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("elo_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        # Reject labels at/above the source dtype's exact-integer bound
        # (2**mantissa_bits + 1 is the first non-representable integer, so a
        # value equal to the bound is already ambiguous): float64/others
        # 2**53, float32 2**24, float16 2**11. np.finfo(...).nmant excludes
        # the implicit leading bit, so the exact-integer bound is nmant + 1.
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"elo_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("elo_rating: player indices must be nonnegative")
    n = int(n_players)
    if n != n_players:
        raise ValueError("elo_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        gamma_arr = np.asarray(gamma, dtype=float)
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("elo_rating: gamma must be real, not complex")
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"elo_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    init = float(init)
    kfac = float(kfac)
    res = _core_module().elo_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        n,
        init,
        kfac,
    )
    return EloResult(
        ratings=np.asarray(res["ratings"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class GlickoResult:
    """Glicko ratings, deviations, and bookkeeping (Glickman, n.d., as
    implemented by the CRAN PlayerRatings package's ``glicko()``;
    batch-per-period update with participant-only deviation inflation
    ``RD = min(sqrt(RD^2 + (lag+1) c^2), rdmax)``). Wins, draws, and losses
    count only scores exactly 1, 0.5, and 0; other fractional scores count
    a game without a W/D/L. ``lag`` is the number of rating periods since
    the player's last appearance."""

    ratings: "np.ndarray"
    deviations: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def glicko_rating(
    games, n_players, init=(2200.0, 300.0), gamma=None, cval=15.0, rdmax=350.0
):
    """Glicko ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`. ``init`` is either a ``(rating, deviation)`` scalar
    pair broadcast to all players or a pair of length-``n_players`` arrays
    (per-player starting ratings/deviations); ``cval`` is the per-period
    uncertainty growth constant and ``rdmax`` the deviation ceiling
    (Glickman's Step 1b: ``RD = min(sqrt(RD^2 + (lag+1) c^2), rdmax)``).
    Defaults ``init=(2200, 300), cval=15, rdmax=350`` are the PlayerRatings
    defaults. Unlike R's fresh-start path, results cover ALL players in
    ``0..n_players``: never-playing players keep their init rating and
    deviation with zero tallies. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("glicko_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"glicko_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("glicko_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("glicko_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"glicko_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("glicko_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above the dtype's exact-integer bound (distinct labels would
    # silently merge into one rating period). Same contract as elo_rating.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("glicko_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"glicko_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("glicko_rating: player indices must be nonnegative")
    n = int(n_players)
    if n != n_players:
        raise ValueError("glicko_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("glicko_rating: gamma must be real, not complex")
        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"glicko_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    try:
        init_r_in, init_d_in = init
    except (TypeError, ValueError):
        raise ValueError(
            "glicko_rating: init must be a (rating, deviation) pair"
        ) from None
    if np.iscomplexobj(np.asarray(init_r_in)) or np.iscomplexobj(np.asarray(init_d_in)):
        raise ValueError("glicko_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init_r_in, dtype=float)
        init_d = np.asarray(init_d_in, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0 and init_d.ndim == 0:
        init_r = np.full(n, float(init_r))
        init_d = np.full(n, float(init_d))
    elif init_r.shape != (n,) or init_d.shape != (n,):
        raise ValueError(
            "glicko_rating: init must be a scalar pair or a pair of "
            f"length-{n} arrays, got shapes {init_r.shape} and {init_d.shape}"
        )
    res = _core_module().glicko_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_d),
        float(cval),
        float(rdmax),
    )
    return GlickoResult(
        ratings=np.asarray(res["ratings"]),
        deviations=np.asarray(res["deviations"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class Glicko2Result:
    """Glicko-2 ratings, deviations, volatilities, and bookkeeping
    (Glickman, 2022, as implemented by the CRAN PlayerRatings package's
    ``glicko2()``; batch-per-period update with participant-only variance
    inflation ``phi^2 <- min(phi^2 + lag * sigma^2, (q * rdmax)^2)`` and the
    Illinois volatility iteration of Glickman's Step 5). Wins, draws, and
    losses count only scores exactly 1, 0.5, and 0; other fractional scores
    count a game without a W/D/L. ``lag`` is the number of rating periods
    since the player's last appearance."""

    ratings: "np.ndarray"
    deviations: "np.ndarray"
    volatilities: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def glicko2_rating(
    games, n_players, init=(2200.0, 300.0, 0.15), gamma=None, tau=1.2, rdmax=350.0
):
    """Glicko-2 ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`. ``init`` is either a ``(rating, deviation,
    volatility)`` scalar triple broadcast to all players or a triple of
    length-``n_players`` arrays; ``tau`` constrains per-period volatility
    change (``tau=0`` freezes volatility) and ``rdmax`` is the deviation
    ceiling (volatility is capped at ``ln(10)/400 * rdmax``). Defaults
    ``init=(2200, 300, 0.15), tau=1.2, rdmax=350`` are the PlayerRatings
    defaults. Unlike R's fresh-start path, results cover ALL players in
    ``0..n_players``: never-playing players keep their init state with zero
    tallies. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("glicko2_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko2_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"glicko2_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("glicko2_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("glicko2_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"glicko2_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("glicko2_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above the dtype's exact-integer bound (distinct labels would
    # silently merge into one rating period). Same contract as elo_rating.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("glicko2_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"glicko2_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("glicko2_rating: player indices must be nonnegative")
    try:
        n = int(n_players)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"glicko2_rating: n_players is not an integer: {exc}") from None
    if n != n_players:
        raise ValueError("glicko2_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("glicko2_rating: gamma must be real, not complex")
        try:
            gamma_arr = np.asarray(gamma, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"glicko2_rating: gamma is not numeric: {exc}") from None
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"glicko2_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    try:
        init_r_in, init_d_in, init_v_in = init
    except (TypeError, ValueError):
        raise ValueError(
            "glicko2_rating: init must be a (rating, deviation, volatility) triple"
        ) from None
    if (
        np.iscomplexobj(np.asarray(init_r_in))
        or np.iscomplexobj(np.asarray(init_d_in))
        or np.iscomplexobj(np.asarray(init_v_in))
    ):
        raise ValueError("glicko2_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init_r_in, dtype=float)
        init_d = np.asarray(init_d_in, dtype=float)
        init_v = np.asarray(init_v_in, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"glicko2_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0 and init_d.ndim == 0 and init_v.ndim == 0:
        init_r = np.full(n, float(init_r))
        init_d = np.full(n, float(init_d))
        init_v = np.full(n, float(init_v))
    elif init_r.shape != (n,) or init_d.shape != (n,) or init_v.shape != (n,):
        raise ValueError(
            "glicko2_rating: init must be a scalar triple or a triple of "
            f"length-{n} arrays, got shapes {init_r.shape}, {init_d.shape}, "
            f"and {init_v.shape}"
        )
    try:
        tau_f = float(tau)
        rdmax_f = float(rdmax)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"glicko2_rating: tau/rdmax is not numeric: {exc}") from None
    res = _core_module().glicko2_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_d),
        np.ascontiguousarray(init_v),
        tau_f,
        rdmax_f,
    )
    return Glicko2Result(
        ratings=np.asarray(res["ratings"]),
        deviations=np.asarray(res["deviations"]),
        volatilities=np.asarray(res["volatilities"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class StephensonResult:
    """Stephenson ratings, deviations, and bookkeeping (the CRAN
    PlayerRatings package's ``steph()``; no journal paper exists for this
    system, so the R driver and C kernel of PlayerRatings 1.1.0 are the
    normative sources -- READ, lines cited in the Rust core). Extends
    Glicko with a per-game neighborhood term ``hval``, a bonus ``bval``
    added to each played game's score, and a lambda drift toward
    opponents' ratings. Wins, draws, and losses count only scores exactly
    1, 0.5, and 0 in the CURRENT run; ``lag`` is the number of rating
    periods since the player's last appearance."""

    ratings: "np.ndarray"
    deviations: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"


def stephenson_rating(
    games,
    n_players,
    init=(2200.0, 300.0),
    gamma=None,
    init_games=None,
    init_lag=None,
    cval=10.0,
    hval=10.0,
    bval=0.0,
    lambda_=2.0,
    rdmax=350.0,
):
    """Stephenson ratings from a (g, 4) game schedule.

    Each row of ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`. ``init`` is either a ``(rating, deviation)`` scalar
    pair broadcast to all players or a pair of length-``n_players`` arrays;
    ``init_games``/``init_lag`` optionally continue a prior run (length-
    ``n_players`` nonnegative integer arrays, default all-zero). ``cval`` is
    the per-period deviation inflation, ``hval`` the per-game neighborhood
    inflation, ``bval`` the per-game bonus (added to each played game's
    score as ``bval/100``), ``lambda_`` the drift toward opponents' ratings
    (``lambda_=0`` disables drift), and ``rdmax`` the deviation ceiling.
    Defaults ``init=(2200, 300), cval=10, hval=10, bval=0, lambda_=2,
    rdmax=350`` are the PlayerRatings defaults. Results cover ALL players
    in ``0..n_players``: never-playing players keep their init state with
    zero current-run tallies. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("stephenson_rating: games must be real, not complex")
    raw = np.asarray(games)
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stephenson_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"stephenson_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("stephenson_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("stephenson_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"stephenson_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("stephenson_rating: period labels must be nonnegative")
    # Period labels are u64 in the core; a float column loses integer
    # fidelity above the dtype's exact-integer bound (distinct labels would
    # silently merge into one rating period). Same contract as elo_rating.
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("stephenson_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
        white_u64 = raw[:, 1].astype(np.uint64)
        black_u64 = raw[:, 2].astype(np.uint64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        for name, col in (("period", periods), ("white", white), ("black", black)):
            if np.any(col >= fidelity):
                raise ValueError(
                    f"stephenson_rating: {name} labels at or above {int(fidelity)} are not "
                    f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                    "pass games as an integer array"
                )
        periods_u64 = periods.astype(np.uint64)
        white_u64 = white.astype(np.uint64)
        black_u64 = black.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("stephenson_rating: player indices must be nonnegative")
    try:
        n = int(n_players)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"stephenson_rating: n_players is not an integer: {exc}"
        ) from None
    if n != n_players:
        raise ValueError("stephenson_rating: n_players must be an integer")
    # Enforce the core's cap BEFORE any length-n allocation below.
    if n < 2:
        raise ValueError("stephenson_rating: at least two players are required")
    if n > 10_000:
        raise ValueError(
            f"stephenson_rating: n = {n} exceeds the supported cap of 10000"
        )
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("stephenson_rating: gamma must be real, not complex")
        try:
            gamma_arr = np.asarray(gamma, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stephenson_rating: gamma is not numeric: {exc}"
            ) from None
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"stephenson_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    try:
        init_r_in, init_d_in = init
    except (TypeError, ValueError):
        raise ValueError(
            "stephenson_rating: init must be a (rating, deviation) pair"
        ) from None
    if np.iscomplexobj(np.asarray(init_r_in)) or np.iscomplexobj(np.asarray(init_d_in)):
        raise ValueError("stephenson_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init_r_in, dtype=float)
        init_d = np.asarray(init_d_in, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stephenson_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0 and init_d.ndim == 0:
        init_r = np.full(n, float(init_r))
        init_d = np.full(n, float(init_d))
    elif init_r.shape != (n,) or init_d.shape != (n,):
        raise ValueError(
            "stephenson_rating: init must be a scalar pair or a pair of "
            f"length-{n} arrays, got shapes {init_r.shape} and {init_d.shape}"
        )

    def _count_vec(name, val):
        if val is None:
            return np.zeros(n, dtype=np.uint64)
        if np.iscomplexobj(np.asarray(val)):
            raise ValueError(f"stephenson_rating: {name} must be real, not complex")
        try:
            v = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stephenson_rating: {name} is not numeric: {exc}"
            ) from None
        if v.shape != (n,):
            raise ValueError(
                f"stephenson_rating: {name} must be a length-{n} array, got {v.shape}"
            )
        if not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v != np.floor(v)):
            raise ValueError(
                f"stephenson_rating: {name} must be nonnegative integers"
            )
        if np.any(v >= 2.0**53):
            raise ValueError(
                f"stephenson_rating: {name} values at or above 2**53 are not "
                "reliably representable; pass smaller counts"
            )
        return v.astype(np.uint64)

    init_g = _count_vec("init_games", init_games)
    init_l = _count_vec("init_lag", init_lag)
    try:
        cval_f = float(cval)
        hval_f = float(hval)
        bval_f = float(bval)
        lambda_f = float(lambda_)
        rdmax_f = float(rdmax)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"stephenson_rating: cval/hval/bval/lambda_/rdmax is not numeric: {exc}"
        ) from None
    res = _core_module().stephenson_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white_u64),
        np.ascontiguousarray(black_u64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_d),
        np.ascontiguousarray(init_g),
        np.ascontiguousarray(init_l),
        cval_f,
        hval_f,
        bval_f,
        lambda_f,
        rdmax_f,
    )
    return StephensonResult(
        ratings=np.asarray(res["ratings"]),
        deviations=np.asarray(res["deviations"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
    )


@dataclass
class ElomResult:
    """Multiplayer Elo ratings and bookkeeping (the CRAN PlayerRatings
    package's ``elom()``; no journal paper exists for this system, so the
    R driver and C kernel of PlayerRatings 1.1.0 are the normative
    sources -- READ, lines cited in the Rust core). Each event seats up
    to ``nn`` players; per period every player gets a single update
    ``K * (actual - expected)`` where actual is the summed seat base
    score and expected sums ``(r_p - mean rating of the event) / 40``
    over the player's events. ``places[p, j]`` counts finishes at rank
    ``j + 1`` in the CURRENT run; ``lag`` is periods since last play."""

    ratings: "np.ndarray"
    games: "np.ndarray"
    places: "np.ndarray"
    lag: "np.ndarray"


def elom_rating(
    periods,
    players,
    scores,
    n_players,
    base=(30.0, 10.0, -10.0, -30.0),
    init=1500.0,
    init_games=None,
    init_lag=None,
    init_places=None,
    kfac=("kriichi", 400.0, 0.2),
    placing=False,
):
    """Multiplayer (nn-seat) Elo ratings from an event schedule.

    ``periods`` is a length-``g`` non-decreasing array of period labels;
    ``players`` a ``(g, nn)`` integer array of player ids in ``0..
    n_players`` with ``-1`` marking an empty seat; ``scores`` a ``(g,
    nn)`` float array (NaN exactly where the seat is empty). ``base`` is
    the length-``nn`` rank base-score vector; when an event has empty
    seats a ONCE-shrunk base is used (PlayerRatings quirk: the shrink is
    applied to the original base exactly once regardless of how many
    seats are empty). ``kfac`` is either a positive float (constant K)
    or ``("kriichi", gv, kv)`` for the experience-decay K factor ``max(
    kv, 1 - (1 - kv) * games / gv)``. ``placing=True`` treats scores as
    placings (rank 1 best = LOWEST score). Defaults ``base=(30, 10, -10,
    -30), init=1500, kfac=("kriichi", 400, 0.2), placing=False`` are the
    PlayerRatings defaults. Results cover ALL players in
    ``0..n_players``. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    try:
        n = int(n_players)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"elom_rating: n_players is not an integer: {exc}") from None
    if n != n_players:
        raise ValueError("elom_rating: n_players must be an integer")
    # Enforce the core's cap BEFORE any length-n allocation below.
    if n < 2:
        raise ValueError("elom_rating: at least two players are required")
    if n > 10_000:
        raise ValueError(f"elom_rating: n = {n} exceeds the supported cap of 10000")

    raw_players = np.asarray(players)
    if np.iscomplexobj(raw_players):
        raise ValueError("elom_rating: players must be real, not complex")
    try:
        players_f = np.asarray(players, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elom_rating: players is not numeric: {exc}") from None
    if players_f.ndim != 2:
        raise ValueError(
            f"elom_rating: players must be a (g, nn) array, got shape {players_f.shape}"
        )
    g, nn = players_f.shape
    if g == 0:
        raise ValueError("elom_rating: at least one event is required")
    if np.any(~np.isfinite(players_f)) or np.any(players_f != np.floor(players_f)):
        raise ValueError("elom_rating: players must be integral (use -1 for empty seats)")
    if np.any(players_f < -1):
        raise ValueError("elom_rating: player ids must be >= -1")
    if raw_players.dtype.kind in "iu":
        if raw_players.dtype.kind == "u" and raw_players.size and int(
            raw_players.max()
        ) > np.iinfo(np.int64).max:
            # An unsigned id above i64::MAX would wrap to a negative value
            # (uint64::MAX -> -1) and silently become the empty-seat
            # sentinel instead of being rejected.
            raise ValueError(
                "elom_rating: player ids above int64 max are not valid player indices"
            )
        players_i64 = raw_players.astype(np.int64)
    else:
        # np.finfo(...).nmant excludes the implicit leading bit, so the
        # exact-integer bound is nmant + 1 (float64 2**53, float32 2**24).
        if raw_players.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw_players.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(players_f >= fidelity):
            raise ValueError(
                f"elom_rating: player ids at or above {int(fidelity)} are not "
                "reliably representable; pass players as an integer array"
            )
        players_i64 = players_f.astype(np.int64)

    if np.iscomplexobj(np.asarray(scores)):
        raise ValueError("elom_rating: scores must be real, not complex")
    try:
        scores_f = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elom_rating: scores is not numeric: {exc}") from None
    if scores_f.shape != (g, nn):
        raise ValueError(
            f"elom_rating: scores must match players shape {(g, nn)}, got {scores_f.shape}"
        )

    raw_periods = np.asarray(periods)
    if np.iscomplexobj(raw_periods):
        raise ValueError("elom_rating: periods must be real, not complex")
    try:
        periods_f = np.asarray(periods, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elom_rating: periods is not numeric: {exc}") from None
    if periods_f.shape != (g,):
        raise ValueError(
            f"elom_rating: periods must be a length-{g} array, got shape {periods_f.shape}"
        )
    if np.any(~np.isfinite(periods_f)) or np.any(periods_f != np.floor(periods_f)):
        raise ValueError("elom_rating: period labels must be integral")
    if np.any(periods_f < 0):
        raise ValueError("elom_rating: period labels must be nonnegative")
    if raw_periods.dtype.kind in "iu":
        periods_u64 = raw_periods.astype(np.uint64)
    else:
        if raw_periods.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw_periods.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods_f >= fidelity):
            raise ValueError(
                f"elom_rating: period labels at or above {int(fidelity)} are not "
                "reliably representable; pass periods as an integer array"
            )
        periods_u64 = periods_f.astype(np.uint64)

    if np.iscomplexobj(np.asarray(base)):
        raise ValueError("elom_rating: base must be real, not complex")
    try:
        base_f = np.asarray(base, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elom_rating: base is not numeric: {exc}") from None
    if base_f.shape != (nn,):
        raise ValueError(
            f"elom_rating: base must be a length-{nn} array, got shape {base_f.shape}"
        )

    if np.iscomplexobj(np.asarray(init)):
        raise ValueError("elom_rating: init must be real, not complex")
    try:
        init_r = np.asarray(init, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"elom_rating: init is not numeric: {exc}") from None
    if init_r.ndim == 0:
        init_r = np.full(n, float(init_r))
    elif init_r.shape != (n,):
        raise ValueError(
            f"elom_rating: init must be a scalar or a length-{n} array, got shape {init_r.shape}"
        )

    def _count_vec(name, val, shape):
        if val is None:
            return np.zeros(shape, dtype=np.uint64)
        if np.iscomplexobj(np.asarray(val)):
            raise ValueError(f"elom_rating: {name} must be real, not complex")
        try:
            v = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"elom_rating: {name} is not numeric: {exc}") from None
        if v.shape != shape:
            raise ValueError(
                f"elom_rating: {name} must have shape {shape}, got {v.shape}"
            )
        if not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v != np.floor(v)):
            raise ValueError(f"elom_rating: {name} must be nonnegative integers")
        if np.any(v >= 2.0**53):
            raise ValueError(
                f"elom_rating: {name} values at or above 2**53 are not "
                "reliably representable; pass smaller counts"
            )
        return v.astype(np.uint64)

    init_g = _count_vec("init_games", init_games, (n,))
    init_l = _count_vec("init_lag", init_lag, (n,))
    init_p = _count_vec("init_places", init_places, (n, nn))

    if isinstance(kfac, (tuple, list)):
        if len(kfac) != 3 or kfac[0] != "kriichi":
            raise ValueError(
                'elom_rating: kfac must be a positive float or ("kriichi", gv, kv)'
            )
        try:
            kfac_gv = float(kfac[1])
            kfac_kv = float(kfac[2])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"elom_rating: kriichi gv/kv is not numeric: {exc}") from None
        mode, kfac_k = "kriichi", 0.0
    else:
        try:
            kfac_k = float(kfac)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"elom_rating: kfac is not numeric: {exc}") from None
        mode, kfac_gv, kfac_kv = "scalar", 0.0, 0.0

    res = _core_module().elom_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(players_i64.reshape(-1)),
        np.ascontiguousarray(scores_f.reshape(-1)),
        np.ascontiguousarray(base_f),
        np.ascontiguousarray(init_r),
        np.ascontiguousarray(init_g),
        np.ascontiguousarray(init_l),
        np.ascontiguousarray(init_p.reshape(-1)),
        mode,
        kfac_k,
        kfac_gv,
        kfac_kv,
        bool(placing),
    )
    return ElomResult(
        ratings=np.asarray(res["ratings"]),
        games=np.asarray(res["games"]),
        places=np.asarray(res["places"]).reshape(n, nn),
        lag=np.asarray(res["lag"]),
    )


def metrics_rating(act, pred, cap=(0.01, 0.99), scale=True):
    """Prediction-quality metrics for binary-outcome forecasts.

    Thin wrapper over the Rust core reimplementation of CRAN PlayerRatings
    1.1-0 ``metrics()`` (R/ratings.R lines 936-957; no journal paper exists
    -- the CRAN source is the normative reference, READ). Per predictor
    column: binomial deviance (on the ``cap``-clamped predictions), RMSE,
    and MAE (both on the RAW uncapped predictions -- the R source quirk),
    each times 100 and, when ``scale`` is true, divided by the
    0.5-constant-predictor baseline. NaN marks missing values and is
    dropped elementwise; Inf is rejected.

    ``act`` is length ``nr``; ``pred`` is ``(nr,)`` or ``(nr, np)``.
    Returns an ``(np, 3)`` float array of per-column ``[bdev, mse, mae]``.
    """
    from .fitstats import _core_module

    def _as_float(name, x, ndim):
        if isinstance(x, np.ma.MaskedArray):
            # np.asarray silently drops the mask, turning masked missing
            # values into observed data; require explicit np.nan instead.
            raise ValueError(
                f"metrics_rating: {name} must not be a masked array; "
                "encode missing values as np.nan"
            )
        arr = np.asarray(x)
        if np.iscomplexobj(arr):
            raise ValueError(f"metrics_rating: {name} must be real-valued")
        if arr.dtype == object:
            if any(
                v is None or isinstance(v, (str, bytes, bool, np.bool_))
                for v in arr.flat
            ):
                # None is rejected rather than silently cast to NaN;
                # callers must pass an explicit np.nan for missing values.
                raise ValueError(f"metrics_rating: {name} must be numeric")
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"metrics_rating: {name} must be numeric"
                ) from exc
        if arr.dtype.kind not in "fiu":
            raise ValueError(
                f"metrics_rating: {name} must be numeric, got dtype {arr.dtype}"
            )
        # ndim is checked BEFORE ascontiguousarray, which would promote
        # 0-D scalars to 1-D and bypass the documented shape contract.
        if arr.ndim != ndim:
            raise ValueError(
                f"metrics_rating: {name} must be {ndim}-D, got {arr.ndim}-D"
            )
        return np.ascontiguousarray(arr, dtype=np.float64)

    act_arr = _as_float("act", act, 1)
    pred_in = np.asarray(pred)
    pred_arr = _as_float("pred", pred, 2 if pred_in.ndim == 2 else 1)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(-1, 1)
    nr, n_pred = pred_arr.shape
    if act_arr.shape[0] != nr:
        raise ValueError(
            f"metrics_rating: act has length {act_arr.shape[0]} "
            f"but pred has {nr} rows"
        )
    cap_arr = _as_float("cap", cap, 1)
    if cap_arr.shape[0] != 2:
        raise ValueError("metrics_rating: cap must be a (lo, hi) pair")
    core = _core_module()
    out = core.metrics_rating(
        act_arr,
        np.ascontiguousarray(pred_arr).ravel(),
        nr,
        n_pred,
        float(cap_arr[0]),
        float(cap_arr[1]),
        bool(scale),
    )
    return np.asarray(out, dtype=np.float64).reshape(n_pred, 3)


@dataclass
class FideResult:
    """FIDE-style Elo ratings and bookkeeping (CRAN PlayerRatings 1.1-0
    ``fide()``; batch-per-period update with the kfide K-factor schedule:
    K = kv[0] for elite players, kv[1] for players with >= 30 period-start
    games, else kv[2]). ``elite`` is the sticky 0/1 flag set once a
    post-update rating reaches 2400 (never cleared); ``opponent`` is the
    running mean of post-update opponent ratings. Wins, draws, and losses
    count only scores exactly 1, 0.5, and 0."""

    ratings: "np.ndarray"
    games: "np.ndarray"
    wins: "np.ndarray"
    draws: "np.ndarray"
    losses: "np.ndarray"
    lag: "np.ndarray"
    elite: "np.ndarray"
    opponent: "np.ndarray"


def fide_rating(games, n_players, init=2200.0, kv=(10.0, 15.0, 30.0), gamma=None):
    """FIDE-style Elo ratings from a (g, 4) game schedule.

    Thin wrapper over the Rust core reimplementation of CRAN PlayerRatings
    1.1-0 ``fide()`` (R/ratings.R lines 125-272 with ``kfide()`` lines
    959-972; the CRAN source is the normative reference, READ). Each row of
    ``games`` is ``[period, white, black, score]`` exactly as in
    :func:`elo_rating`; the difference is the per-player K factor: ``kv``
    is the (elite, experienced, novice) triple applied from PERIOD-START
    games/elite state (R evaluates ``kfac`` before the bookkeeping
    updates). Defaults ``init=2200, kv=(10, 15, 30)`` are the PlayerRatings
    defaults. The 30-game and 2400-rating thresholds are hard-coded as in
    the R source. Raises ValueError on invalid input.
    """
    import numpy as np

    from .fitstats import _core_module

    if isinstance(games, np.ma.MaskedArray):
        raise ValueError(
            "fide_rating: games must not be a masked array; "
            "masked entries would be silently unmasked"
        )
    if np.iscomplexobj(np.asarray(games)):
        raise ValueError("fide_rating: games must be real, not complex")
    raw = np.asarray(games)
    if raw.dtype != object and raw.dtype.kind not in "fiu":
        # Rejects bool, datetime64, timedelta64, and other non-numeric
        # ndarray dtypes that np.asarray(..., dtype=float) would coerce.
        raise ValueError(
            f"fide_rating: games must be numeric, got dtype {raw.dtype}"
        )
    # Nested Python lists coerce bools/datetimes into legal-looking numbers
    # before the dtype check can see them; scan the original elements.
    probe = raw if raw.dtype == object else None
    if probe is None and not isinstance(games, np.ndarray):
        probe = np.asarray(games, dtype=object)
    if probe is not None and any(
        v is None
        or isinstance(
            v, (str, bytes, bool, np.bool_, np.datetime64, np.timedelta64)
        )
        for v in probe.flat
    ):
        raise ValueError("fide_rating: games must be numeric")
    try:
        arr = np.asarray(games, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fide_rating: games is not numeric: {exc}") from None
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"fide_rating: games must be (g, 4) [period, white, black, score], got {arr.shape}"
        )
    if arr.shape[0] == 0:
        raise ValueError("fide_rating: at least one game is required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("fide_rating: games contains non-finite values")
    periods, white, black, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    for name, col in (("period", periods), ("white", white), ("black", black)):
        if np.any(col != np.floor(col)):
            raise ValueError(f"fide_rating: {name} column must be integral")
    if np.any(periods < 0):
        raise ValueError("fide_rating: period labels must be nonnegative")
    # Same u64 fidelity contract as elo_rating: take period labels
    # losslessly from integer inputs, otherwise reject labels at/above the
    # source float dtype's exact-integer bound (nmant + 1 bits).
    if raw.ndim == 2 and raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw[:, 0] < 0):
            raise ValueError("fide_rating: period labels must be nonnegative")
        periods_u64 = raw[:, 0].astype(np.uint64)
    else:
        if raw.dtype.kind == "f":
            fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
        else:
            fidelity = 2.0**53
        if np.any(periods >= fidelity):
            raise ValueError(
                f"fide_rating: period labels at or above {int(fidelity)} are not "
                f"reliably representable in {raw.dtype if raw.dtype.kind == 'f' else 'float64'}; "
                "pass games as an integer array"
            )
        periods_u64 = periods.astype(np.uint64)
    if np.any(white < 0) or np.any(black < 0):
        raise ValueError("fide_rating: player indices must be nonnegative")
    n = int(n_players)
    if n != n_players:
        raise ValueError("fide_rating: n_players must be an integer")
    g = arr.shape[0]
    if gamma is None:
        gamma_arr = np.zeros(g)
    else:
        if np.iscomplexobj(np.asarray(gamma)):
            raise ValueError("fide_rating: gamma must be real, not complex")
        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.ndim == 0:
            gamma_arr = np.full(g, float(gamma_arr))
        elif gamma_arr.shape != (g,):
            raise ValueError(
                f"fide_rating: gamma must be a scalar or length-{g} array, got {gamma_arr.shape}"
            )
    kv_raw = kv if isinstance(kv, np.ndarray) else np.asarray(kv, dtype=object)
    if np.iscomplexobj(kv_raw) or kv_raw.dtype.kind == "b":
        raise ValueError("fide_rating: kv must be real numeric, not complex/bool")
    if kv_raw.dtype == object:
        for v in np.ravel(kv_raw):
            if isinstance(v, (bool, np.bool_)):
                raise ValueError("fide_rating: kv must be real numeric, not bool")
            if isinstance(v, (complex, np.complexfloating)):
                raise ValueError("fide_rating: kv must be real, not complex")
    try:
        kv_arr = np.asarray(kv, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fide_rating: kv is not numeric: {exc}") from None
    if kv_arr.shape != (3,):
        raise ValueError(
            f"fide_rating: kv must be a (elite, experienced, novice) triple, got shape {kv_arr.shape}"
        )
    init_raw = np.asarray(init)
    if (
        isinstance(init, (bool, np.bool_))
        or init_raw.dtype.kind == "b"
        or (
            init_raw.dtype == object
            and init_raw.ndim == 0
            and isinstance(init_raw.item(), (bool, np.bool_))
        )
    ):
        raise ValueError("fide_rating: init must be real numeric, not bool")
    if np.iscomplexobj(init_raw):
        raise ValueError("fide_rating: init must be real, not complex")
    try:
        init = float(init)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fide_rating: init is not numeric: {exc}") from None
    res = _core_module().fide_rating(
        np.ascontiguousarray(periods_u64),
        np.ascontiguousarray(white, dtype=np.uint64),
        np.ascontiguousarray(black, dtype=np.uint64),
        np.ascontiguousarray(score),
        np.ascontiguousarray(gamma_arr),
        n,
        init,
        float(kv_arr[0]),
        float(kv_arr[1]),
        float(kv_arr[2]),
    )
    return FideResult(
        ratings=np.asarray(res["ratings"]),
        games=np.asarray(res["games"]),
        wins=np.asarray(res["wins"]),
        draws=np.asarray(res["draws"]),
        losses=np.asarray(res["losses"]),
        lag=np.asarray(res["lag"]),
        elite=np.asarray(res["elite"]),
        opponent=np.asarray(res["opponent"]),
    )


def _predict_int_index_array(x, name, fname):
    """Validate a player-index array: integers >= -1 (-1 = unmatched)."""
    if isinstance(x, np.ma.MaskedArray):
        raise ValueError(f"{fname}: masked arrays are not supported for {name}")
    raw = x if isinstance(x, np.ndarray) else np.asarray(x, dtype=object)
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError(f"{fname}: {name} must be integer indices, not complex/bool")
    if raw.dtype == object:
        for v in np.ravel(raw):
            if v is None or isinstance(
                v, (bool, np.bool_, str, bytes, np.datetime64, np.timedelta64)
            ):
                raise ValueError(f"{fname}: {name} contains a non-numeric value")
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fname}: {name} is not numeric: {exc}") from None
    if arr.ndim != 1:
        raise ValueError(f"{fname}: {name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{fname}: {name} must be finite")
    if not np.all(arr == np.floor(arr)):
        raise ValueError(f"{fname}: {name} must contain integers")
    if np.any(arr < -1):
        raise ValueError(f"{fname}: {name} indices must be >= -1 (-1 = unmatched)")
    return arr.astype(np.int64)


def _predict_float_array(x, name, fname, allow_nan):
    """Validate a float array; NaN optionally allowed (R NA), Inf rejected."""
    if isinstance(x, np.ma.MaskedArray):
        raise ValueError(f"{fname}: masked arrays are not supported for {name}")
    raw = x if isinstance(x, np.ndarray) else np.asarray(x, dtype=object)
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError(f"{fname}: {name} must be real numeric, not complex/bool")
    if raw.dtype == object:
        for v in np.ravel(raw):
            if isinstance(
                v, (bool, np.bool_, str, bytes, np.datetime64, np.timedelta64)
            ) or v is None:
                raise ValueError(f"{fname}: {name} contains a non-numeric value")
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fname}: {name} is not numeric: {exc}") from None
    if arr.ndim != 1:
        raise ValueError(f"{fname}: {name} must be one-dimensional")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{fname}: {name} must not contain infinities")
    if not allow_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{fname}: {name} must not contain NaN")
    return arr


def _predict_scalar(x, name, fname):
    """Validate a finite real scalar parameter."""
    import math
    if isinstance(x, (bool, np.bool_)):
        raise ValueError(f"{fname}: {name} must be real numeric, not bool")
    raw = np.asarray(x)
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError(f"{fname}: {name} must be real numeric, not complex/bool")
    if raw.dtype == object and raw.ndim == 0 and isinstance(
        raw.item(), (bool, np.bool_)
    ):
        raise ValueError(f"{fname}: {name} must be real numeric, not bool")
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fname}: {name} is not numeric: {exc}") from None
    if not math.isfinite(v):
        raise ValueError(f"{fname}: {name} must be finite")
    return v


def _predict_games_u64(x, fname):
    """Validate games counts losslessly into u64.

    Integer-dtype ndarrays (and Python-int sequences, which numpy keeps in
    an integer dtype) pass through without a float round-trip, preserving
    counts at or above 2**53. Float inputs are bounded by the source
    dtype's exact-integer limit (np.finfo(...).nmant + 1 bits: float64
    2**53, float32 2**24) so a rounded count can never silently shift the
    strict games < tng cutoff."""
    name = "games"
    if isinstance(x, np.ma.MaskedArray):
        raise ValueError(f"{fname}: masked arrays are not supported for {name}")
    if isinstance(x, np.ndarray):
        raw = x
    else:
        probe = np.asarray(x, dtype=object)
        if np.iscomplexobj(probe):
            raise ValueError(f"{fname}: {name} must be real numeric, not complex/bool")
        for v in np.ravel(probe):
            if v is None or isinstance(
                v, (bool, np.bool_, str, bytes, np.datetime64, np.timedelta64)
            ):
                raise ValueError(f"{fname}: {name} contains a non-numeric value")
        try:
            raw = np.asarray(x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{fname}: {name} is not numeric: {exc}") from None
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError(f"{fname}: {name} must be real numeric, not complex/bool")
    if raw.ndim != 1:
        raise ValueError(f"{fname}: {name} must be one-dimensional")
    if raw.dtype.kind in "iu":
        if raw.dtype.kind == "i" and np.any(raw < 0):
            raise ValueError(f"{fname}: {name} must be nonnegative integers")
        return raw.astype(np.uint64)
    arr = _predict_float_array(x, name, fname, allow_nan=False)
    if np.any(arr < 0) or not np.all(arr == np.floor(arr)):
        raise ValueError(f"{fname}: {name} must be nonnegative integers")
    if raw.dtype.kind == "f":
        fidelity = 2.0 ** (np.finfo(raw.dtype).nmant + 1)
    else:
        fidelity = 2.0**53
    if np.any(arr >= fidelity):
        raise ValueError(
            f"{fname}: {name} values at or above {int(fidelity)} are not "
            "reliably representable in this float dtype; pass games as an "
            "integer array"
        )
    return arr.astype(np.uint64)


def _predict_tng_u64(tng, fname):
    """Validate tng losslessly into u64 (no float round-trip for ints)."""
    import math

    if isinstance(tng, (bool, np.bool_)):
        raise ValueError(f"{fname}: tng must be real numeric, not bool")
    if isinstance(tng, (int, np.integer)):
        t = int(tng)
        if t < 0:
            raise ValueError(f"{fname}: tng must be a nonnegative integer")
        if t > 2**64 - 1:
            raise ValueError(f"{fname}: tng must fit in an unsigned 64-bit integer")
        return t
    if (
        isinstance(tng, np.ndarray)
        and tng.ndim == 0
        and tng.dtype.kind in "iu"
    ):
        return _predict_tng_u64(int(tng), fname)
    v = _predict_scalar(tng, "tng", fname)
    if v < 0 or v != math.floor(v):
        raise ValueError(f"{fname}: tng must be a nonnegative integer")
    if v >= 2.0**53:
        raise ValueError(
            f"{fname}: tng values at or above 2**53 are not reliably "
            "representable as float; pass tng as an int"
        )
    return int(v)


def predict_rating(
    ratings,
    games,
    white,
    black,
    deviations=None,
    gamma=30.0,
    tng=15,
    trat=None,
    thresh=None,
):
    """Predicted win probabilities for two-player games from fitted ratings.

    Implements the two-player branches of ``predict.rating`` from CRAN
    PlayerRatings 1.1-0 (R/ratings.R lines 1056-1133; source read). Without
    ``deviations`` this is the Elo branch (used for ``elo()``/``fide()``
    fits); with ``deviations`` it is the deviation-shrunk branch shared by
    Glicko, Glicko-2, and Stephenson fits. Players are addressed by index;
    ``-1`` marks an unmatched player (R's ``match()`` NA). Players with
    ``games < tng`` (strict) are treated as unrated; ``trat`` (a scalar, or
    a ``(rating, deviation)`` pair when ``deviations`` is supplied)
    replaces ALL missing extracted values, matching R. ``thresh`` maps
    predictions to 1 when ``pred >= thresh`` else 0 (NaN kept). ``gamma``
    is the per-game (or scalar, broadcast) first-player advantage.
    """
    from .fitstats import _core_module

    fname = "predict_rating"
    rat = _predict_float_array(ratings, "ratings", fname, allow_nan=True)
    n = rat.shape[0]
    if n < 2 or n > 10000:
        raise ValueError(f"{fname}: number of players must be in 2..=10000, got {n}")
    games_u64 = _predict_games_u64(games, fname)
    if games_u64.shape[0] != n:
        raise ValueError(
            f"{fname}: games must have one entry per player ({n}), "
            f"got {games_u64.shape[0]}"
        )
    w = _predict_int_index_array(white, "white", fname)
    b = _predict_int_index_array(black, "black", fname)
    dev = None
    if deviations is not None:
        dev = _predict_float_array(deviations, "deviations", fname, allow_nan=True)
    ng = w.shape[0]
    gam_raw = np.asarray(gamma)
    if np.iscomplexobj(gam_raw) or gam_raw.dtype.kind == "b":
        raise ValueError(f"{fname}: gamma must be real numeric, not complex/bool")
    if gam_raw.ndim == 0:
        gam = np.full(ng, _predict_scalar(gamma, "gamma", fname))
    else:
        gam = _predict_float_array(gamma, "gamma", fname, allow_nan=False)
        if gam.shape != (ng,):
            raise ValueError(
                f"{fname}: gamma must be a scalar or length-{ng} array, got {gam.shape}"
            )
    tng_v = _predict_tng_u64(tng, fname)
    trat_rating = None
    trat_deviation = None
    if trat is not None:
        if dev is not None:
            if not (isinstance(trat, (tuple, list)) and len(trat) == 2):
                raise ValueError(
                    f"{fname}: trat must be a (rating, deviation) pair when "
                    "deviations are supplied"
                )
            trat_rating = _predict_scalar(trat[0], "trat rating", fname)
            trat_deviation = _predict_scalar(trat[1], "trat deviation", fname)
        else:
            if isinstance(trat, (tuple, list)):
                if len(trat) != 1:
                    raise ValueError(
                        f"{fname}: trat must be a scalar (length 1) without deviations"
                    )
                trat = trat[0]
            trat_rating = _predict_scalar(trat, "trat", fname)
    thresh_v = None if thresh is None else _predict_scalar(thresh, "thresh", fname)
    out = _core_module().predict_rating_two(
        np.ascontiguousarray(rat),
        None if dev is None else np.ascontiguousarray(dev),
        np.ascontiguousarray(games_u64),
        np.ascontiguousarray(w),
        np.ascontiguousarray(b),
        np.ascontiguousarray(gam),
        int(tng_v),
        trat_rating,
        trat_deviation,
        thresh_v,
    )
    return np.asarray(out)


def predict_rating_multi(
    ratings,
    games,
    players,
    tng=15,
    trat=None,
    placing=False,
):
    """Predicted expected scores (or placings) for multi-player EloM events.

    Implements the EloM branch of ``predict.rating`` from CRAN
    PlayerRatings 1.1-0 (R/ratings.R lines 1103-1105, 1123-1125,
    1129-1130; source read): per event row,
    ``pred = (rating - rowmean) / 40`` with the row mean over non-missing
    seats (``rowMeans(rats, na.rm=TRUE)``). ``players`` is an ``(nr, np)``
    index matrix with ``-1`` = empty/unmatched seat. Players with
    ``games < tng`` are treated as unrated; scalar ``trat`` replaces all
    missing extracted ratings. With ``placing=True`` each row is replaced
    by min-tie ranks of the predictions (rank 1 = highest; NaN kept),
    matching R's ``rank(-preds, na.last="keep", ties.method="min")``.
    """
    from .fitstats import _core_module

    fname = "predict_rating_multi"
    rat = _predict_float_array(ratings, "ratings", fname, allow_nan=True)
    n = rat.shape[0]
    if n < 2 or n > 10000:
        raise ValueError(f"{fname}: number of players must be in 2..=10000, got {n}")
    games_u64 = _predict_games_u64(games, fname)
    if games_u64.shape[0] != n:
        raise ValueError(
            f"{fname}: games must have one entry per player ({n}), "
            f"got {games_u64.shape[0]}"
        )
    if isinstance(players, np.ma.MaskedArray):
        raise ValueError(f"{fname}: masked arrays are not supported for players")
    p_raw = players if isinstance(players, np.ndarray) else np.asarray(players, dtype=object)
    if p_raw.ndim != 2:
        raise ValueError(f"{fname}: players must be a 2-D (events, seats) matrix")
    nr, np_seats = p_raw.shape
    flat = _predict_int_index_array(np.ravel(p_raw), "players", fname)
    if not (2 <= np_seats <= 1000):
        raise ValueError(
            f"{fname}: seats per event must be in 2..=1000, got {np_seats}"
        )
    tng_v = _predict_tng_u64(tng, fname)
    trat_v = None if trat is None else _predict_scalar(trat, "trat", fname)
    if not isinstance(placing, (bool, np.bool_)):
        raise ValueError(f"{fname}: placing must be a bool")
    out = _core_module().predict_rating_multi(
        np.ascontiguousarray(rat),
        np.ascontiguousarray(games_u64),
        np.ascontiguousarray(flat),
        int(nr),
        int(np_seats),
        int(tng_v),
        trat_v,
        bool(placing),
    )
    return np.asarray(out).reshape(nr, np_seats)

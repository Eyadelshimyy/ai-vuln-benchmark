"""Metropolis-Hastings Robbins-Monro (MH-RM) confirmatory multidimensional IFA (Cai, 2010b).

A stochastic-approximation EM that scales confirmatory item factor analysis to a latent
dimensionality where the deterministic Gauss-Hermite / quasi-Monte-Carlo E-steps of
:func:`fast_mlsirm.fit_2pl` become infeasible. Each cycle imputes the traits with a short persistent
random-walk Metropolis chain, then takes one Robbins-Monro stochastic-Newton step on the
block-diagonal (per-item) complete-data score and information. Supports the binary 2PL and the ordered
polytomous GPCM (Muraki, 1992) response families, orthogonal or correlated factors. The numerical
estimation runs in Rust."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES
from .irt_contract import validate_irt_response_matrix
from .models import ConfirmatoryModel, ExploratoryModel, IrtModel, _resolve_model

_MAX_DIMS = 64
_MAX_RESPONSE_CELLS = 200_000_000
_NUMPY_INTEGER_TYPES = tuple(
    np.dtype(name).type
    for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
)
_NUMPY_FLOAT_TYPES = tuple(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)
_NUMPY_COMPLEX_TYPES = tuple(
    np.dtype(name).type for name in ("complex64", "complex128", "clongdouble")
)
_TRUSTED_RESPONSE_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_TYPES,
    *_NUMPY_FLOAT_TYPES,
    *_NUMPY_COMPLEX_TYPES,
)
_TRUSTED_RESPONSE_ARRAY_KINDS = ("b", "i", "u", "f", "c")
_TRUSTED_INTEGRAL_CONTROL_TYPES = (int, float, *_NUMPY_INTEGER_TYPES, *_NUMPY_FLOAT_TYPES)
_TRUSTED_SEED_TYPES = (int, *_NUMPY_INTEGER_TYPES)
_TRUSTED_REAL_CONTROL_TYPES = (
    int,
    float,
    *_NUMPY_INTEGER_TYPES,
    *_NUMPY_FLOAT_TYPES,
)


def _is_exact_type(value_type: type, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value_type`` is one package-trusted scalar identity."""

    return any(value_type is trusted_type for trusted_type in trusted_types)


def _trusted_response_array(responses: object) -> np.ndarray:
    """Materialize MH-RM responses only after callback-free identity/resource preflight."""

    resource_error = (
        f"responses exceed the {_MAX_RESPONSE_CELLS:,}-cell MH-RM resource limit"
    )
    if type(responses) is np.ndarray:
        if int(responses.size) > _MAX_RESPONSE_CELLS:
            raise ValueError(resource_error)
        return responses

    error = "responses must be a trusted NumPy array or built-in response matrix"
    if type(responses) is not list and type(responses) is not tuple:
        raise ValueError(error)

    # A valid non-empty 2-D built-in response matrix with N scalar cells visits
    # at most N row nodes + N scalar nodes.  This preserves every valid matrix
    # inside the native 200M-cell envelope while bounding malformed empty or
    # over-nested container fan-out before NumPy materialization.
    structural_budget = 2 * _MAX_RESPONSE_CELLS
    structural_nodes = 0

    # Each frame is [object, next_child_index, entered, logical_cell_count].
    # Memoized container counts preserve shared acyclic subtrees while charging
    # every logical occurrence against the same Rust-side persons×items ceiling.
    frames: list[list[object]] = [[responses, 0, False, 0]]
    active_container_ids: set[int] = set()
    validated_container_cells: dict[int, int] = {}

    while frames:
        frame = frames[-1]
        item = frame[0]
        item_type = type(item)

        if _is_exact_type(item_type, _TRUSTED_RESPONSE_SCALAR_TYPES):
            frames.pop()
            cells = 1
            if frames:
                parent_cells = int(frames[-1][3]) + cells
                if parent_cells > _MAX_RESPONSE_CELLS:
                    raise ValueError(resource_error)
                frames[-1][3] = parent_cells
            continue
        if item_type is np.ndarray:
            if item.dtype.kind not in _TRUSTED_RESPONSE_ARRAY_KINDS:
                raise ValueError(error)
            cells = int(item.size)
            if cells > _MAX_RESPONSE_CELLS:
                raise ValueError(resource_error)
            frames.pop()
            if frames:
                parent_cells = int(frames[-1][3]) + cells
                if parent_cells > _MAX_RESPONSE_CELLS:
                    raise ValueError(resource_error)
                frames[-1][3] = parent_cells
            continue

        if item_type is not list and item_type is not tuple:
            raise ValueError(error)

        item_id = id(item)
        if not bool(frame[2]):
            if item_id in validated_container_cells:
                cells = validated_container_cells[item_id]
                frames.pop()
                if frames:
                    parent_cells = int(frames[-1][3]) + cells
                    if parent_cells > _MAX_RESPONSE_CELLS:
                        raise ValueError(resource_error)
                    frames[-1][3] = parent_cells
                continue
            if item_id in active_container_ids:
                raise ValueError(error)
            active_container_ids.add(item_id)
            frame[2] = True

        child_index = int(frame[1])
        if child_index < len(item):
            frame[1] = child_index + 1
            structural_nodes += 1
            if structural_nodes > structural_budget:
                raise ValueError(
                    "responses exceeded structural traversal budget of "
                    f"{structural_budget:,} nodes"
                )
            frames.append([item[child_index], 0, False, 0])
            continue

        cells = int(frame[3])
        active_container_ids.remove(item_id)
        validated_container_cells[item_id] = cells
        frames.pop()
        if frames:
            parent_cells = int(frames[-1][3]) + cells
            if parent_cells > _MAX_RESPONSE_CELLS:
                raise ValueError(resource_error)
            frames[-1][3] = parent_cells

    return np.asarray(responses)


def _finite_integer_control(value: object, name: str) -> int:
    """Normalize one finite integral control without invoking caller conversion protocols."""

    value_type = type(value)
    if not _is_exact_type(value_type, _TRUSTED_INTEGRAL_CONTROL_TYPES):
        raise ValueError(f"{name} must be a finite integer")
    if _is_exact_type(value_type, _NUMPY_FLOAT_TYPES) or value_type is float:
        if not bool(np.isfinite(value)) or value != np.floor(value):
            raise ValueError(f"{name} must be a finite integer")
    normalized = int(value)
    if normalized >= 2**64:
        raise ValueError(f"{name} must be in [0, 2**64)")
    return normalized


def _finite_real_control(value: object, name: str) -> float:
    """Normalize one finite real control without invoking caller conversion protocols."""

    if not _is_exact_type(type(value), _TRUSTED_REAL_CONTROL_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _seed_control(value: object) -> int:
    """Normalize the public unsigned-64 seed without caller integer dispatch."""

    if not _is_exact_type(type(value), _TRUSTED_SEED_TYPES):
        raise TypeError("seed must be a non-negative integer")
    seed = int(value)
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return seed


def _boolean_control(value: object, name: str) -> bool:
    """Normalize one Boolean flag without invoking arbitrary truth protocols."""

    if type(value) is bool:
        return value
    if type(value) is np.bool_:
        return bool(value)
    raise TypeError(f"{name} must be a boolean")


def _family_control(value: object) -> str:
    """Normalize the response-family selector without invoking caller text conversion."""

    if type(value) is not str:
        raise ValueError("family must be '2pl' or 'gpcm'")
    family = value.lower()
    if family not in ("2pl", "gpcm"):
        raise ValueError("family must be '2pl' or 'gpcm'")
    return family


@dataclass
class MhrmFit:
    """Fitted MH-RM confirmatory multidimensional 2PL (Cai, 2010b).

    ``loading`` is the ``n_items x n_dims`` matrix of free loadings ``a_id`` (exactly ``0`` where the
    confirmatory pattern is ``0``), per-dimension reflection-canonicalized so each dimension's largest
    pure anchor loads positive; ``theta`` the ``n_persons x n_dims`` trait EAP (Monte-Carlo mean of the
    imputed draws over the convergence stage); ``corr`` the ``n_dims x n_dims`` latent correlation
    matrix ``Phi`` (identity when ``estimate_corr=False``, unit diagonal with estimated off-diagonals
    otherwise).

    The category parameters depend on :attr:`family`. For the binary 2PL, ``intercept`` holds the
    per-item ``b_i`` (and ``step`` is empty); the model is
    ``P(X_ij=1 | theta_j) = sigmoid(sum_d a_id theta_jd + b_i)``. For the GPCM (Muraki, 1992), ``step``
    is the ``n_items x (n_cat - 1)`` matrix of UNORDERED step intercepts (and ``intercept`` is empty);
    with ``base_ij = sum_d a_id theta_jd`` (no intercept),
    ``P(X_ij=k | theta_j) = softmax_k(k*base_ij + step_ik)`` (``step_i0 = 0`` pinned), ``theta_j ~
    MVN(0, Phi)``. ``se_loading`` / ``se_intercept`` / ``se_step`` are the matching Louis (1982)
    observed-information standard errors (empty when ``estimate_se=False``; a block falls back to the
    complete-data Fisher information where the finite-sample Louis block is not positive-definite).

    ``acceptance_rate`` is the final tuned Metropolis acceptance; ``termination_reason`` is
    ``"converged"`` or ``"max_cycles_reached"``; ``final_param_change`` the windowed mean parameter
    change at termination."""

    model: IrtModel
    family: str
    n_cat: int
    loading: np.ndarray
    intercept: np.ndarray
    step: np.ndarray
    theta: np.ndarray
    corr: np.ndarray
    se_loading: np.ndarray
    se_intercept: np.ndarray
    se_step: np.ndarray
    acceptance_rate: float
    n_cycles: int
    converged: bool
    termination_reason: str
    final_param_change: float
    n_parameters: int

    @property
    def n_dims(self) -> int:
        """Latent dimension count derived from :attr:`model`."""

        return self.model.n_dims


def fit_mhrm(
    responses: np.ndarray,
    model: int | ExploratoryModel | ConfirmatoryModel,
    family: str = "2pl",
    n_cat: int = 2,
    *,
    max_cycles: int,
    burn_in: int,
    mh_steps: int,
    proposal_sd: float = 1.0,
    target_accept: float,
    tol: float,
    seed: int,
    estimate_se: bool = True,
    estimate_corr: bool = False,
) -> MhrmFit:
    """Fit the confirmatory multidimensional 2PL by Metropolis-Hastings Robbins-Monro (compute in
    Rust; Cai, 2010b).

    A stochastic-approximation EM for the general compensatory 2PL,
    ``P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`` with ``theta_j ~ MVN(0,
    I_D)``. Unlike :func:`fast_mlsirm.fit_2pl`, the marginal-likelihood integral is not quadratured:
    each cycle (1) imputes each person's ``theta`` by a short PERSISTENT (warm-started) random-walk
    Metropolis chain from its current posterior, and (2) takes one Robbins-Monro stochastic-Newton
    step ``xi <- xi + gain_k Gamma_k^{-1} s_k`` on the block-diagonal per-item complete-data score
    ``s_k`` and Robbins-Monro-smoothed information ``Gamma_k``. The gain follows a constant-gain
    burn-in then a decreasing ``1/(k - burn_in)`` schedule (``sum gain = inf``, ``sum gain^2 < inf``),
    converging almost surely to a marginal-score root. Because the per-item work is closed-form and
    ``D``-independent, MH-RM scales to a latent dimensionality (``n_dims`` up to 64) where the
    ``q**n_dims`` Gauss-Hermite grid and the QMC E-step are infeasible.

    Identification: unit trait variances fix the loading scale; a PURE single-dimension anchor item
    per dimension pins the rotation; the per-dimension sign is CANONICALIZED (largest pure anchor
    positive), enforced in-loop each cycle so the stochastic running average stays in one mirror mode.
    Loadings are UNCONSTRAINED so reverse-keyed / negative cross-loadings are representable. Standard
    errors are the Louis (1982) observed information accumulated over the convergence stage.

    The response family is selected by ``family``: ``"2pl"`` (binary, default) or ``"gpcm"`` (the
    ordered polytomous generalized partial credit model, Muraki, 1992, with ``n_cat`` integer
    categories ``0..n_cat-1``). For the GPCM each item keeps a SINGLE discrimination vector ``a_i`` and
    gains ``n_cat - 1`` free UNORDERED step intercepts, ``P(X_ij=k) = softmax_k(k*sum_d a_id theta_jd +
    step_ik)``, estimated by the same MH-RM machinery with the closed-form multinomial complete-data
    Hessian as the Robbins-Monro preconditioner and Louis information. ``family="gpcm"`` requires every
    declared category to be observed for every item (else the corresponding step is unidentified).

    ``responses`` is a persons x items array (``NaN`` = missing, dropped under MAR): ``0/1`` for the
    2PL, integer categories ``0..n_cat-1`` for the GPCM. For ``model=1`` all item loadings on the single
    factor are free; a multidimensional confirmatory structure is supplied with
    ``model=models.confirmatory(loading_pattern)``; every dimension needs a pure single-loading anchor
    item. ``burn_in`` must be less than ``max_cycles``; ``proposal_sd`` is the initial random-walk SD,
    auto-tuned toward ``target_accept`` during burn-in. With ``estimate_corr=True`` a free latent
    CORRELATION matrix ``Phi`` (``theta ~ MVN(0, Phi)``, unit diagonal) is estimated by a per-cycle
    Robbins-Monro gradient step (Cai, 2010b); with ``False`` (default) the factors are orthogonal
    (``Phi = I``) and the fit is bit-identical to the flag off.

    References (APA 7th ed.):
        Cai, L. (2010a). High-dimensional exploratory item factor analysis by a Metropolis-Hastings
            Robbins-Monro algorithm. *Psychometrika, 75*(1), 33–57.
            https://doi.org/10.1007/s11336-009-9136-x
        Cai, L. (2010b). Metropolis-Hastings Robbins-Monro algorithm for confirmatory item factor
            analysis. *Journal of Educational and Behavioral Statistics, 35*(3), 307–335.
            https://doi.org/10.3102/1076998609353115
        Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm.
            *Journal of the Royal Statistical Society: Series B (Methodological), 44*(2), 226–233.
            https://doi.org/10.1111/j.2517-6161.1982.tb01203.x
        Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied
            Psychological Measurement, 16*(2), 159–176. https://doi.org/10.1177/014662169201600206
    """
    max_cycles_int = _finite_integer_control(max_cycles, "max_cycles")
    burn_in_int = _finite_integer_control(burn_in, "burn_in")
    mh_steps_int = _finite_integer_control(mh_steps, "mh_steps")
    proposal_sd_float = _finite_real_control(proposal_sd, "proposal_sd")
    target_accept_float = _finite_real_control(target_accept, "target_accept")
    tol_float = _finite_real_control(tol, "tol")
    seed_int = _seed_control(seed)
    estimate_se_bool = _boolean_control(estimate_se, "estimate_se")
    estimate_corr_bool = _boolean_control(estimate_corr, "estimate_corr")
    fam = _family_control(family)
    n_cat_int = _finite_integer_control(n_cat, "n_cat")

    if max_cycles_int <= 0 or burn_in_int < 0 or burn_in_int >= max_cycles_int:
        raise ValueError("require 0 <= burn_in < max_cycles")
    if mh_steps_int <= 0:
        raise ValueError("mh_steps must be positive")
    if proposal_sd_float <= 0.0:
        raise ValueError("proposal_sd must be finite and positive")
    if not 0.0 <= target_accept_float <= 1.0:
        raise ValueError("target_accept must be in [0, 1]")
    if tol_float <= 0.0:
        raise ValueError("tol must be finite and positive")
    if fam == "2pl":
        n_cat_int = 2
    elif not 2 <= n_cat_int <= MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(
            "n_cat must be between 2 and "
            f"{MAX_POLYTOMOUS_CATEGORIES} for family='gpcm'"
        )

    response_input = _trusted_response_array(responses)
    if np.iscomplexobj(response_input) or response_input.dtype == object:
        raise ValueError("responses must be real-valued")
    if response_input.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a real numeric array")
    y = response_input.astype(np.float64, copy=False)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    resolved_model, pat = _resolve_model(model, n_items)
    n_dims = pat.shape[1]
    if not 1 <= n_dims <= _MAX_DIMS:
        raise ValueError(f"loading_pattern dimensions must be between 1 and {_MAX_DIMS}")

    observed = ~np.isnan(y)
    if np.any(observed):
        obs_y = y[observed]
        if fam == "2pl":
            if np.any((obs_y != 0) & (obs_y != 1)):
                raise ValueError("responses must be 0, 1, or NaN (missing)")
        else:
            if np.any(obs_y != np.floor(obs_y)) or np.any(obs_y < 0) or np.any(obs_y >= n_cat_int):
                raise ValueError(
                    "responses must be integer categories in "
                    f"0..{n_cat_int - 1}, or NaN (missing)"
                )
    validation_y = np.where(observed, y, np.nan)
    validate_irt_response_matrix(
        validation_y,
        "dichotomous" if fam == "2pl" else "polytomous",
        n_categories=None if fam == "2pl" else n_cat_int,
    )
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        raise RuntimeError("fit_mhrm requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    res = core.fit_mhrm(
        yy,
        observed.reshape(-1),
        pat.astype(np.int64).reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_dims),
        max_cycles_int,
        burn_in_int,
        mh_steps_int,
        proposal_sd_float,
        target_accept_float,
        tol_float,
        seed_int,
        estimate_se_bool,
        estimate_corr_bool,
        fam,
        n_cat_int,
    )
    se_loading = np.asarray(res["se_loading"], dtype=np.float64)
    se_intercept = np.asarray(res["se_intercept"], dtype=np.float64)
    se_step = np.asarray(res["se_step"], dtype=np.float64)
    n_free_cat = int(res["n_cat"]) - 1
    step = np.asarray(res["step"], dtype=np.float64)
    return MhrmFit(
        model=resolved_model,
        family=fam,
        n_cat=int(res["n_cat"]),
        loading=np.asarray(res["loading"], dtype=np.float64).reshape(n_items, n_dims),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        step=step.reshape(n_items, n_free_cat) if step.size else step,
        theta=np.asarray(res["theta"], dtype=np.float64).reshape(n_persons, n_dims),
        corr=np.asarray(res["corr"], dtype=np.float64).reshape(n_dims, n_dims),
        se_loading=se_loading.reshape(n_items, n_dims) if se_loading.size else se_loading,
        se_intercept=se_intercept,
        se_step=se_step.reshape(n_items, n_free_cat) if se_step.size else se_step,
        acceptance_rate=float(res["acceptance_rate"]),
        n_cycles=int(res["n_cycles"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_param_change=float(res["final_param_change"]),
        n_parameters=int(res["n_parameters"]),
    )

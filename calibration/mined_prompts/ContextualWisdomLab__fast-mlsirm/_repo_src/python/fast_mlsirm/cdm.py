"""Cognitive diagnosis models: DINA (conjunctive / AND gate) and DINO
(disjunctive / OR gate), estimated by marginal-ML EM over the ``2^K`` binary
attribute-mastery profiles in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER


_MAX_ATTRIBUTES = 15
# Keep the Python trust boundary aligned with the Rust sequential-CDM resource cap.
_SEQ_MAX_CAT = 50
_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_CDM_MODELS = frozenset(("dina", "dino"))


def _reject_untrusted_response_container(value: object) -> None:
    """Delegate response-container admission to the canonical safety module.

    The package initializer installs the same canonical guard on ordinary import.
    Keeping this reload fallback as a delegation, rather than a second implementation,
    preserves identical callback-safe semantics when ``fast_mlsirm.cdm`` is reloaded
    directly without re-running package initialization.
    """

    from ._cdm_response_safety import _reject_untrusted_response_container as reject

    reject(value)


def _response_array(value: np.ndarray) -> np.ndarray:
    """Materialize accepted real response storage without lossy coercion."""
    _reject_untrusted_response_container(value)
    response_array = np.asarray(value)
    if np.iscomplexobj(response_array):
        raise ValueError("responses must be real-valued")
    if response_array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = response_array.astype(np.float64, copy=False)
        round_tripped = converted.astype(response_array.dtype, copy=False)
    if not np.array_equal(response_array, round_tripped, equal_nan=True):
        raise ValueError("responses must be exactly representable as float64")
    return converted


def _prepare_binary_responses(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the flattened values/mask for 0/1 data with NaN-only missingness."""
    if np.isinf(y).any():
        raise ValueError("responses must contain only 0, 1, or NaN (missing)")
    observed = ~np.isnan(y)
    values = y[observed]
    if values.size and not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("responses must contain only 0, 1, or NaN (missing)")
    return np.where(observed, y, 0.0).reshape(-1), observed.reshape(-1)


def _validate_stopping_controls(max_iter: int, tol: float) -> tuple[int, float]:
    """Validate exact scalar EM controls without caller-controlled coercion hooks."""
    max_iter_type = type(max_iter)
    if max_iter_type is int:
        iteration_cap = max_iter
    elif any(max_iter_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        iteration_cap = int(max_iter)
    else:
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")
    if not 1 <= iteration_cap <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be an integer between 1 and {MAX_MAX_ITER}")

    tol_type = type(tol)
    if not (
        tol_type is int
        or tol_type is float
        or any(
            tol_type is scalar_type
            for scalar_type in (*_NUMPY_INTEGER_SCALAR_TYPES, *_NUMPY_FLOAT_SCALAR_TYPES)
        )
    ):
        raise ValueError("tol must be a finite number > 0")
    try:
        tolerance = float(tol)
    except OverflowError as exc:
        raise ValueError("tol must be a finite number > 0") from exc
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tol must be a finite number > 0")
    return iteration_cap, tolerance


def _validate_model_selector(model: str) -> str:
    """Accept only the exact built-in DINA/DINO selector vocabulary."""
    if type(model) is not str or model not in _CDM_MODELS:
        raise ValueError("model must be 'dina' or 'dino'")
    return model


def _validate_q_matrix_input(
    value: np.ndarray, name: str, n_items: int
) -> tuple[np.ndarray, int]:
    """Validate and safely coerce a public Q-matrix before native dispatch."""
    q = np.asarray(value)
    if q.ndim != 2:
        raise ValueError(f"{name} must be a 2-D items x attributes array")
    if q.shape[0] != n_items:
        if name == "step_q":
            raise ValueError("step_q must have sum(n_steps) rows")
        raise ValueError(f"{name} must have one row per item")
    n_attributes = q.shape[1]
    if not 1 <= n_attributes <= _MAX_ATTRIBUTES:
        raise ValueError(f"{name} must have between 1 and {_MAX_ATTRIBUTES} attributes")
    if not (
        np.issubdtype(q.dtype, np.number) or np.issubdtype(q.dtype, np.bool_)
    ) or np.issubdtype(q.dtype, np.complexfloating):
        raise ValueError(f"{name} entries must be numeric 0 or 1")
    if not np.all(np.isfinite(q)) or not np.all((q == 0) | (q == 1)):
        raise ValueError(f"{name} entries must be finite and exactly 0 or 1")
    return q.astype(np.int64, copy=False), n_attributes


def _validate_seq_n_steps(value: object, n_items: int) -> tuple[np.ndarray, int]:
    """Admit sequential step counts without executing caller conversion protocols."""

    value_type = type(value)
    if value_type is np.ndarray:
        raw_steps = value
        if raw_steps.ndim != 1 or raw_steps.shape[0] != n_items:
            raise ValueError("n_steps must be a 1-D array of length n_items")
        if raw_steps.dtype.kind not in ("i", "u"):
            raise ValueError("n_steps entries must be positive integers")
        if np.any(raw_steps < 1):
            raise ValueError("n_steps entries must be positive integers")
        if np.any(raw_steps > _SEQ_MAX_CAT):
            raise ValueError(f"n_steps entries must be <= {_SEQ_MAX_CAT}")
        steps = raw_steps.astype(np.int64, copy=False)
        return steps, sum(int(step) for step in steps)

    if value_type is not list and value_type is not tuple:
        raise ValueError(
            "n_steps must be a trusted 1-D integer NumPy array or built-in sequence"
        )
    if len(value) != n_items:
        raise ValueError("n_steps must be a 1-D array of length n_items")

    normalized: list[int] = []
    for step in value:
        step_type = type(step)
        if step_type is int or any(
            step_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES
        ):
            step_value = int(step)
        elif step_type in (bool, float, str, bytes, np.bool_) or any(
            step_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES
        ):
            raise ValueError("n_steps entries must be positive integers")
        else:
            raise ValueError(
                "n_steps must be a trusted 1-D integer NumPy array or built-in sequence"
            )
        if step_value < 1:
            raise ValueError("n_steps entries must be positive integers")
        if step_value > _SEQ_MAX_CAT:
            raise ValueError(f"n_steps entries must be <= {_SEQ_MAX_CAT}")
        normalized.append(step_value)

    steps = np.asarray(normalized, dtype=np.int64)
    return steps, sum(normalized)


@dataclass
class CdmFit:
    """Fitted DINA/DINO cognitive diagnosis model.

    ``slip``/``guess`` are the per-item ``s_i = P(X=0 | mastered)`` and
    ``g_i = P(X=1 | not mastered)``; ``profile_prob`` the population probability of
    each of the ``2^K`` attribute profiles (bit-encoded: attribute ``k`` is mastered
    in profile ``c`` iff ``(c >> k) & 1``); ``map_profile`` the per-person posterior
    mode (bit-encoded); ``attr_prob`` the persons x attributes marginal mastery
    probabilities ``P(alpha_jk = 1 | X_j)`` (attribute EAP)."""

    model: str
    slip: np.ndarray
    guess: np.ndarray
    profile_prob: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int

    def attribute_mastery(self) -> np.ndarray:
        """Hard 0/1 attribute-mastery classification (``attr_prob >= 0.5``)."""
        return (self.attr_prob >= 0.5).astype(np.int64)

    def profile_bits(self) -> np.ndarray:
        """Decode ``map_profile`` into a persons x attributes 0/1 matrix."""
        k = self.attr_prob.shape[1]
        codes = self.map_profile.astype(np.int64)
        return ((codes[:, None] >> np.arange(k)) & 1).astype(np.int64)


def fit_cdm(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    model: str = "dina",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CdmFit:
    """Fit a DINA or DINO cognitive diagnosis model (compute in Rust).

    Each respondent has a binary attribute-mastery profile ``alpha in {0,1}^K``; the
    Q-matrix specifies which of the ``K`` attributes each item requires. The ideal
    (latent) response is ``eta = prod_k alpha_k^{q_k}`` for DINA (mastery of ALL
    required attributes) or ``eta = 1 - prod_k (1 - alpha_k)^{q_k}`` for DINO (ANY
    required attribute); the observed response adds a per-item slip and guess,
    ``P(X=1 | alpha) = (1 - s)^{eta} g^{1 - eta}``. Parameters are estimated by
    marginal-ML EM over the ``2^K`` profiles with a free profile distribution;
    persons are classified by their posterior-mode profile (``map_profile``) and
    marginal attribute probabilities (``attr_prob``).

    ``responses`` is a persons x items array of 0/1 (``NaN`` marks a missing cell,
    dropped under a missing-at-random assumption). ``q_matrix`` is an items x
    attributes 0/1 array; all-zero rows (an item measuring nothing) and all-zero
    columns (an attribute measured by no item) are rejected. These checks do not
    establish global model identifiability: DINA identifiability requires stronger
    Q-matrix conditions (Gu & Xu, 2019), which callers must assess for their design.

    References (APA 7th ed.):
        de la Torre, J. (2009). DINA model and parameter estimation: A didactic.
            *Journal of Educational and Behavioral Statistics, 34*(1), 115–130.
            https://doi.org/10.3102/1076998607309474
        Gu, Y., & Xu, G. (2019). The sufficient and necessary condition for the
            identifiability and estimability of the DINA model. *Psychometrika,
            84*(2), 468–483. https://doi.org/10.1007/s11336-018-9619-8
        Junker, B. W., & Sijtsma, K. (2001). Cognitive assessment models with few
            assumptions, and connections with nonparametric item response theory.
            *Applied Psychological Measurement, 25*(3), 258–272.
            https://doi.org/10.1177/01466210122032064
        Templin, J. L., & Henson, R. A. (2006). Measurement of psychological
            disorders using cognitive diagnosis models. *Psychological Methods,
            11*(3), 287–305. https://doi.org/10.1037/1082-989X.11.3.287
    """
    model = _validate_model_selector(model)
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(q_matrix, "q_matrix", n_items)
    yy, observed = _prepare_binary_responses(y)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_cdm"):
        raise RuntimeError("fit_cdm requires the compiled Rust core")
    res = core.fit_cdm(
        yy,
        observed,
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        model,
        max_iter,
        tol,
    )
    return CdmFit(
        model=str(res["model"]),
        slip=np.asarray(res["slip"], dtype=np.float64),
        guess=np.asarray(res["guess"], dtype=np.float64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class GdinaFit:
    """Fitted saturated G-DINA model (de la Torre, 2011).

    Item parameters are ragged: item ``i`` has ``2 ** k_required[i]`` reduced
    attribute-mastery classes, stored as the CSR slice
    ``[item_off[i]:item_off[i+1]]`` of ``item_prob`` and ``item_delta``.
    ``item_prob`` holds the free success probabilities ``P(X_i = 1 | reduced
    class l)``; ``item_delta`` the identity-link parameters (intercept, main
    effects, interactions). ``map_profile``/``attr_prob`` are the per-person MAP
    profile and marginal attribute-mastery probabilities."""

    item_off: np.ndarray
    item_prob: np.ndarray
    item_delta: np.ndarray
    k_required: np.ndarray
    profile_prob: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int

    def item_prob_row(self, i: int) -> np.ndarray:
        """Success probabilities of item ``i``'s ``2 ** K_i`` reduced classes."""
        return self.item_prob[self.item_off[i] : self.item_off[i + 1]]

    def item_delta_row(self, i: int) -> np.ndarray:
        """Identity-link parameters of item ``i`` (intercept, mains, interactions)."""
        return self.item_delta[self.item_off[i] : self.item_off[i + 1]]


def fit_gdina(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> GdinaFit:
    """Fit the saturated G-DINA model (compute in Rust; de la Torre, 2011).

    G-DINA is the general cognitive-diagnosis framework: for item ``i`` requiring
    ``K_i`` attributes, each of the ``2 ** K_i`` reduced attribute-mastery classes
    gets a FREE success probability, estimated by marginal-ML EM over the ``2 ** K``
    profiles with the closed-form saturated M-step ``p_il = R_il / I_il`` (expected
    correct / expected total in reduced class ``l``). DINA, DINO, A-CDM, LLM and
    R-RUM are constrained special cases readable off the fitted identity-link
    ``item_delta`` (e.g. DINA leaves only the intercept and the highest-order
    interaction nonzero). ``responses`` is a persons x items 0/1 array (``NaN`` =
    missing, dropped under MAR); ``q_matrix`` is an items x attributes 0/1 array.

    The saturated fit constrains each success probability only to ``[0, 1]``. It does
    not impose subset-lattice order restrictions, so Q-matrix identifiability alone
    does not guarantee that mastering more required attributes increases success.
    Order-restricted G-DINA estimation is a distinct model choice (Hong et al., 2016)
    and is not implemented by this function.

    References (APA 7th ed.):
        de la Torre, J. (2011). The generalized DINA model framework.
            *Psychometrika, 76*(2), 179-199.
            https://doi.org/10.1007/s11336-011-9207-7
        Hong, C.-Y., Chang, Y.-W., & Tsai, R.-C. (2016). Estimation of generalized
            DINA model with order restrictions. *Journal of Classification, 33*(3),
            460-484. https://doi.org/10.1007/s00357-016-9215-5
        Ma, W., & de la Torre, J. (2020). GDINA: An R package for cognitive
            diagnosis modeling. *Journal of Statistical Software, 93*(14), 1-26.
            https://doi.org/10.18637/jss.v093.i14
    """
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(q_matrix, "q_matrix", n_items)
    yy, observed = _prepare_binary_responses(y)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gdina"):
        raise RuntimeError("fit_gdina requires the compiled Rust core")
    res = core.fit_gdina(
        yy,
        observed,
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        max_iter,
        tol,
    )
    return GdinaFit(
        item_off=np.asarray(res["item_off"], dtype=np.int64),
        item_prob=np.asarray(res["item_prob"], dtype=np.float64),
        item_delta=np.asarray(res["item_delta"], dtype=np.float64),
        k_required=np.asarray(res["k_required"], dtype=np.int64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class QMatrixValidation:
    """Result of empirical Q-matrix validation (de la Torre & Chiu, 2016).

    ``suggested_q`` is the validated items x attributes 0/1 Q-matrix — per item the
    fewest-attribute vector whose PVAF (proportion of variance accounted for)
    reaches ``epsilon``. ``suggested_pvaf``/``provisional_pvaf`` are the per-item
    PVAF of the suggested and the caller's provisional q-vector; ``flagged`` marks
    the items whose suggested vector differs from the provisional one. The
    ``calibration_*`` fields report the exact stopping evidence for the provisional
    G-DINA calibration used to construct the suggestions."""

    suggested_q: np.ndarray
    suggested_pvaf: np.ndarray
    provisional_pvaf: np.ndarray
    flagged: np.ndarray
    epsilon: float
    calibration_n_iter: int
    calibration_max_iter: int
    calibration_termination_reason: str
    calibration_final_loglik_change: float
    calibration_tol: float


def validate_q_matrix(
    responses: np.ndarray,
    provisional_q: np.ndarray,
    epsilon: float = 0.95,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> QMatrixValidation:
    """Validate a Q-matrix by the PVAF method (compute in Rust; de la Torre & Chiu, 2016).

    The G-DINA item response function varies across the ``2^K`` latent attribute
    classes. A candidate q-vector groups those classes into masters vs. non-masters
    of its required attributes; the proportion of the item's across-class variance
    that grouping captures is its ``PVAF``. For each item the method returns the
    q-vector with the FEWEST required attributes whose ``PVAF >= epsilon``: an
    under-specified provisional vector falls short of the cutoff and is enlarged, an
    over-specified one is trimmed because a smaller vector already reaches it.

    The class distribution and identified attribute labels come from a G-DINA fit
    with the provisional Q; each item's saturated success probability over all
    ``2^K`` classes is then recovered from the fitted posteriors, so a mis-specified
    item's true attribute dependence is exposed by the attributes identified from
    the other items. The method therefore assumes the provisional Q is mostly
    correct (enough items to identify the attributes).

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under
    MAR); ``provisional_q`` is an items x attributes 0/1 array, each item loading at
    least one attribute (``K`` up to 10). ``epsilon`` is the PVAF cutoff. A
    nonconverged provisional G-DINA calibration raises ``ValueError`` instead of
    producing PVAF suggestions from an unfinished fit.

    References (APA 7th ed.):
        de la Torre, J., & Chiu, C.-Y. (2016). A general method of empirical Q-matrix
            validation. *Psychometrika, 81*(2), 253–273.
            https://doi.org/10.1007/s11336-015-9467-8
        de la Torre, J. (2008). An empirically based method of Q-matrix validation
            for the DINA model: Development and applications. *Journal of Educational
            Measurement, 45*(4), 343-362.
            https://doi.org/10.1111/j.1745-3984.2008.00069.x
    """
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(provisional_q, "provisional_q", n_items)
    yy, observed = _prepare_binary_responses(y)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "validate_q_matrix"):
        raise RuntimeError("validate_q_matrix requires the compiled Rust core")
    res = core.validate_q_matrix(
        yy,
        observed,
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        float(epsilon),
        max_iter,
        tol,
    )
    return QMatrixValidation(
        suggested_q=np.asarray(res["suggested_q"], dtype=np.int64).reshape(n_items, n_attributes),
        suggested_pvaf=np.asarray(res["suggested_pvaf"], dtype=np.float64),
        provisional_pvaf=np.asarray(res["provisional_pvaf"], dtype=np.float64),
        flagged=np.asarray(res["flagged"], dtype=bool),
        epsilon=float(res["epsilon"]),
        calibration_n_iter=int(res["calibration_n_iter"]),
        calibration_max_iter=int(res["calibration_max_iter"]),
        calibration_termination_reason=str(res["calibration_termination_reason"]),
        calibration_final_loglik_change=float(res["calibration_final_loglik_change"]),
        calibration_tol=float(res["calibration_tol"]),
    )


@dataclass
class WaldModelSelection:
    """Item-level CDM model selection by Wald test (de la Torre & Lee, 2013).

    ``models`` names the candidate reduced models (parsimony order). ``wald_stat``,
    ``wald_df`` and ``p_value`` are items x models arrays of the Wald statistic,
    degrees of freedom, and upper-tail p-value (``NaN``/0 where a test is undefined,
    i.e. an item requiring fewer than two attributes). ``selected`` is per item the
    index into ``models`` of the chosen reduced model, or ``-1`` for the saturated
    G-DINA."""

    models: list
    wald_stat: np.ndarray
    wald_df: np.ndarray
    p_value: np.ndarray
    selected: np.ndarray
    alpha: float


def gdina_wald_selection(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    alpha: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> WaldModelSelection:
    """Select item-level CDMs by Wald test (Rust; de la Torre & Lee, 2013).

    For each item the saturated G-DINA is compared with reduced models that are exact
    linear restrictions of the reduced attribute-mastery success probabilities ``P``.
    DINA/DINO/A-CDM restrict the identity-link parameters ``delta = M^{-1} P``; LLM and
    R-RUM restrict the transformed parameters ``delta^h = M^{-1} h(P)`` on the link on
    which each is additive:

    * **DINA** (conjunctive): only the intercept and the top-order interaction free.
    * **DINO** (disjunctive): the non-intercept coordinates tied onto one line
      ``delta_S = (-1)^{|S|+1} Delta`` (a general, non-coordinate linear restriction).
    * **A-CDM** (additive on the identity link): all interaction terms zero
      (intercept + main effects).
    * **LLM** (linear logistic model; additive on the logit link): interaction terms of
      ``delta^{logit} = M^{-1} logit(P)`` zero.
    * **R-RUM** (reduced reparameterized unified model; additive on the log link):
      interaction terms of ``delta^{log} = M^{-1} log(P)`` zero.

    The Wald statistic ``W = (R delta)' (R Sigma_delta R')^{-1} (R delta) ~ chi^2(df)``
    tests whether the restriction ``R delta = 0`` holds. For the identity link
    ``Sigma_delta = M^{-1} Var(P) M^{-T}`` with ``Var(P_l) = P_l(1-P_l)/I_l``
    (complete-data / expected information); for a transformed link the delta method uses
    ``Var(h(P_l)) = h'(P_l)^2 Var(P_l)`` (LLM: ``1/(I_l P_l(1-P_l))``; R-RUM:
    ``(1-P_l)/(I_l P_l)``). Per item the fewest-parameter model with ``p > alpha`` is
    selected (DINA and DINO cost two parameters; A-CDM, LLM and R-RUM each cost
    ``1 + K``, so ties are broken by the larger p-value); if all reduced models are
    rejected, the saturated G-DINA is kept.

    Note: the complete-data covariance uses expected rather than observed information,
    so the test is mildly liberal (Type I slightly above ``alpha``); the gap shrinks
    with sample size and item discrimination and with strong attribute identification.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under
    MAR); ``q_matrix`` is an items x attributes 0/1 array. A nonconverged saturated
    G-DINA calibration raises ``ValueError`` instead of returning Wald statistics and
    a model choice from unfinished parameters.

    References (APA 7th ed.):
        de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
            76*(2), 179–199. https://doi.org/10.1007/s11336-011-9207-7
        de la Torre, J., & Lee, Y.-S. (2013). Evaluating the Wald test for item-level
            comparison of saturated and reduced models in cognitive diagnosis.
            *Journal of Educational Measurement, 50*(4), 355–373.
            https://doi.org/10.1111/jedm.12022
        Ma, W., Iaconangelo, C., & de la Torre, J. (2016). Model similarity, model
            selection, and attribute classification. *Applied Psychological
            Measurement, 40*(3), 200–217. https://doi.org/10.1177/0146621615621717
    """
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(q_matrix, "q_matrix", n_items)
    yy, observed = _prepare_binary_responses(y)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "gdina_wald_selection"):
        raise RuntimeError("gdina_wald_selection requires the compiled Rust core")
    res = core.gdina_wald_selection(
        yy,
        observed,
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        float(alpha),
        max_iter,
        tol,
    )
    models = list(res["models"])
    n_models = len(models)
    return WaldModelSelection(
        models=models,
        wald_stat=np.asarray(res["wald_stat"], dtype=np.float64).reshape(n_items, n_models),
        wald_df=np.asarray(res["wald_df"], dtype=np.int64).reshape(n_items, n_models),
        p_value=np.asarray(res["p_value"], dtype=np.float64).reshape(n_items, n_models),
        selected=np.asarray(res["selected"], dtype=np.int64),
        alpha=float(res["alpha"]),
    )


@dataclass
class HoCdmFit:
    """Fitted higher-order DINA/DINO model (de la Torre & Douglas, 2004).

    A continuous higher-order trait ``theta ~ N(0,1)`` structures attribute mastery,
    ``P(alpha_k=1 | theta) = sigmoid(attr_slope_k * theta + attr_intercept_k)``, with
    attributes conditionally independent given ``theta``. ``slip``/``guess`` are the
    per-item DINA parameters; ``profile_prob`` the implied ``2^K`` class distribution;
    ``theta`` the per-person EAP trait; ``map_profile``/``attr_prob`` the per-person
    MAP profile and marginal attribute mastery. The higher-order parameters are a
    genuine (identified) restriction only for ``K >= 3``. ``termination_reason``,
    ``n_iter``/``max_iter``, ``final_loglik_change``, and ``stopping_tolerance``
    report the outer EM stopping evidence."""

    model: str
    slip: np.ndarray
    guess: np.ndarray
    attr_slope: np.ndarray
    attr_intercept: np.ndarray
    profile_prob: np.ndarray
    theta: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    max_iter: int
    termination_reason: str
    final_loglik_change: float
    stopping_tolerance: float
    n_parameters: int

    def attribute_mastery(self) -> np.ndarray:
        """Hard 0/1 attribute-mastery classification (``attr_prob >= 0.5``)."""
        return (self.attr_prob >= 0.5).astype(np.int64)


def fit_ho_cdm(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    model: str = "dina",
    max_iter: int = 500,
    tol: float = 1e-6,
) -> HoCdmFit:
    """Fit the higher-order DINA/DINO model (compute in Rust; de la Torre & Douglas, 2004).

    Unlike :func:`fit_cdm` (which estimates a free ``2^K`` class distribution), the
    attribute-mastery distribution here is *structured* by a continuous higher-order
    trait ``theta ~ N(0,1)``: ``P(alpha_k=1 | theta) = sigmoid(a_k theta + d_k)``, with
    attributes conditionally independent given ``theta``. This replaces ``2^K - 1`` free
    class probabilities with ``2K`` interpretable attribute parameters. The item part
    (slip/guess, DINA or DINO gate) is unchanged. Estimated by marginal-ML EM over the
    joint ``(alpha, theta)`` grid; the structural step is ``K`` independent 2PL
    calibrations of attribute mastery on the trait. De la Torre and Douglas (2004)
    introduced the higher-order model and estimated it by Bayesian MCMC; the
    quadrature-EM estimator is this package's implementation choice.

    The observed-data likelihood depends on ``(a_k, d_k)`` only through the implied
    class distribution, so the higher-order parameters are identified only for
    ``K >= 3``; at ``K <= 2`` only ``profile_prob`` and the attribute classification
    are identified. ``attr_slope`` is anchored non-negative.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR);
    ``q_matrix`` is an items x attributes 0/1 array; ``model`` is ``"dina"`` or ``"dino"``.

    References (APA 7th ed.):
        de la Torre, J., & Douglas, J. A. (2004). Higher-order latent trait models for
            cognitive diagnosis. *Psychometrika, 69*(3), 333-353.
            https://doi.org/10.1007/BF02295640
    """
    model = _validate_model_selector(model)
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(q_matrix, "q_matrix", n_items)
    yy, observed = _prepare_binary_responses(y)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_ho_cdm"):
        raise RuntimeError("fit_ho_cdm requires the compiled Rust core")
    res = core.fit_ho_cdm(
        yy,
        observed,
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        model,
        max_iter,
        tol,
    )
    return HoCdmFit(
        model=str(res["model"]),
        slip=np.asarray(res["slip"], dtype=np.float64),
        guess=np.asarray(res["guess"], dtype=np.float64),
        attr_slope=np.asarray(res["attr_slope"], dtype=np.float64),
        attr_intercept=np.asarray(res["attr_intercept"], dtype=np.float64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        max_iter=int(res["max_iter"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class HoGdinaFit:
    """Fitted higher-order G-DINA model (de la Torre & Douglas, 2004; de la Torre, 2011).

    The saturated G-DINA item model (ragged CSR: item ``i`` has ``2 ** k_required[i]``
    reduced-class success probabilities at ``item_prob[item_off[i]:item_off[i+1]]``,
    with the identity-link ``item_delta``) under a higher-order structural attribute
    prior ``P(alpha_k=1 | theta) = sigmoid(attr_slope_k*theta + attr_intercept_k)``,
    ``theta ~ N(0,1)``. ``profile_prob`` is the implied ``2^K`` class distribution;
    ``theta``/``map_profile``/``attr_prob`` the per-person trait EAP, MAP profile, and
    marginal attribute mastery."""

    item_off: np.ndarray
    item_prob: np.ndarray
    item_delta: np.ndarray
    k_required: np.ndarray
    attr_slope: np.ndarray
    attr_intercept: np.ndarray
    profile_prob: np.ndarray
    theta: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    final_relative_loglik_change: float
    stopping_tolerance: float
    n_parameters: int

    def item_prob_row(self, i: int) -> np.ndarray:
        """Success probabilities of item ``i``'s ``2 ** K_i`` reduced classes."""
        return self.item_prob[self.item_off[i] : self.item_off[i + 1]]


def fit_ho_gdina(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> HoGdinaFit:
    """Fit the higher-order G-DINA model (compute in Rust; de la Torre & Douglas, 2004;
    de la Torre, 2011).

    Combines the saturated G-DINA item model (each item's reduced attribute-mastery
    classes get a free success probability, as in :func:`fit_gdina`) with a
    higher-order structural attribute prior in which a continuous trait
    ``theta ~ N(0,1)`` drives mastery, ``P(alpha_k=1 | theta) = sigmoid(a_k theta +
    d_k)``, with attributes conditionally independent given ``theta`` (as in
    :func:`fit_ho_cdm`). It generalizes :func:`fit_ho_cdm` (which restricts the item
    model to DINA slip/guess) and constrains :func:`fit_gdina`'s free class
    distribution to the ``2K``-parameter structured family. Estimated by marginal-ML
    EM over the joint ``(alpha, theta)`` grid: the saturated item M-step marginalizes
    the trait out, and the structural step is ``K`` independent 2PL calibrations of
    attribute mastery on the trait. The higher-order parameters are identified for
    ``K >= 3``; fits with fewer attributes are rejected rather than returning
    unidentified structural parameters. ``attr_slope`` is anchored non-negative.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under MAR;
    positive/negative infinity is invalid); ``q_matrix`` is an items x attributes 0/1
    array. Convergence uses the scale-free observed-data likelihood change
    ``abs(delta log L) / (1 + abs(log L_previous)) < tol``; the raw and relative
    terminal changes and the stable termination reason are returned explicitly.

    References (APA 7th ed.):
        de la Torre, J., & Douglas, J. A. (2004). Higher-order latent trait models for
            cognitive diagnosis. *Psychometrika, 69*(3), 333-353.
            https://doi.org/10.1007/BF02295640
        de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
            76*(2), 179-199. https://doi.org/10.1007/s11336-011-9207-7
    """
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(q_matrix, "q_matrix", n_items)
    yy, observed = _prepare_binary_responses(y)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_ho_gdina"):
        raise RuntimeError("fit_ho_gdina requires the compiled Rust core")
    res = core.fit_ho_gdina(
        yy,
        observed,
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        max_iter,
        tol,
    )
    return HoGdinaFit(
        item_off=np.asarray(res["item_off"], dtype=np.int64),
        item_prob=np.asarray(res["item_prob"], dtype=np.float64),
        item_delta=np.asarray(res["item_delta"], dtype=np.float64),
        k_required=np.asarray(res["k_required"], dtype=np.int64),
        attr_slope=np.asarray(res["attr_slope"], dtype=np.float64),
        attr_intercept=np.asarray(res["attr_intercept"], dtype=np.float64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        final_relative_loglik_change=float(res["final_relative_loglik_change"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class SeqGdinaFit:
    """Fitted shared-Q sequential (continuation-ratio) G-DINA (Ma & de la Torre, 2016).

    Ordered polytomous cognitive diagnosis. Item ``i`` has ``M_i = max_cat[i]`` ordered
    steps over ``2 ** k_required[i]`` reduced attribute classes (ragged, CLASS-MAJOR CSR):
    ``step_prob[s_off[i] + l*M_i + (k-1)] = s_ik(l) = P(X_i >= k | X_i >= k-1, class l)``
    for step ``k in 1..=M_i`` and reduced class ``l``; the implied category probabilities
    are ``cat_prob[cat_off[i] + l*(M_i+1) + x] = P(X_i = x | class l)`` for ``x in 0..=M_i``.
    ``profile_prob`` is the free ``2^K`` class distribution; ``map_profile``/``attr_prob``
    the per-person MAP profile and marginal attribute mastery.

    Restriction: every step of an item uses the SAME item Q-vector (shared-Q) — a
    restriction of Ma & de la Torre's general per-step ``q_ik`` model. Use
    :func:`fit_seq_gdina_qr` for step-distinct attributes; for this result, supply each
    item's Q-vector as the union of its steps' required attributes."""

    s_off: np.ndarray
    step_prob: np.ndarray
    cat_off: np.ndarray
    cat_prob: np.ndarray
    max_cat: np.ndarray
    k_required: np.ndarray
    profile_prob: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    final_relative_loglik_change: float
    stopping_tolerance: float
    n_parameters: int

    def item_step_prob(self, i: int) -> np.ndarray:
        """Step (continuation) probabilities of item ``i`` as a ``2**K_i x M_i`` array
        (row = reduced class ``l``, column = step ``k-1``)."""
        m = int(self.max_cat[i])
        return self.step_prob[self.s_off[i] : self.s_off[i + 1]].reshape(-1, m)

    def item_cat_prob(self, i: int) -> np.ndarray:
        """Category probabilities of item ``i`` as a ``2**K_i x (M_i+1)`` array
        (row = reduced class ``l``, column = category ``x``)."""
        m1 = int(self.max_cat[i]) + 1
        return self.cat_prob[self.cat_off[i] : self.cat_off[i + 1]].reshape(-1, m1)


def fit_seq_gdina(
    responses: np.ndarray,
    q_matrix: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> SeqGdinaFit:
    """Fit the shared-Q sequential G-DINA for ordered polytomous responses (compute in
    Rust; Ma & de la Torre, 2016).

    Each ordered category of an item is reached through a sequence of *steps*: the
    continuation probability ``s_ik(l) = P(X_i >= k | X_i >= k-1, reduced class l)`` is a
    saturated G-DINA over the item's reduced attribute-mastery classes, and the category
    probabilities are the sequential decomposition ``P(X_i = k | l) = (prod_{v<=k}
    s_iv(l))(1 - s_{i,k+1}(l))`` (stop sentinel ``s_{i,M_i+1} = 0``). The population is a
    free profile distribution (as in :func:`fit_gdina`); estimation is marginal-ML EM with
    the closed-form saturated step ``s_ik(l) = (expected count reaching >= k) / (expected
    count reaching >= k-1)`` in reduced class ``l``. With one step per item (binary data)
    it reduces exactly to :func:`fit_gdina`.

    **Restriction (shared item Q-vector).** Every step of item ``i`` is a saturated G-DINA
    over the SAME required attributes (row ``i`` of ``q_matrix``). This is a restriction of
    Ma & de la Torre's (2016) general per-step ``q_ik`` model, whose headline feature is
    *step-distinct* attribute requirements. Use :func:`fit_seq_gdina_qr` for that model; for
    this shared-Q entry point, supply each item's Q-vector as the UNION of its steps'
    required attributes.

    ``responses`` is a persons x items array of ordered integer categories ``0..M_i``
    (``NaN`` = missing, dropped under MAR); ``M_i`` (the number of steps) is derived as the
    maximum observed category of item ``i``, and an item whose observed maximum is 0 (never
    leaves the base category) is rejected. ``q_matrix`` is an items x attributes 0/1 array.
    Nonzero Q rows/columns are necessary sanity checks, not a certificate of global model
    identifiability; supply an identified design and inspect ``converged``.
    Convergence uses the absolute observed-data log-likelihood increment and is checked
    before another M-step. The stable termination reason, completed M-step count, signed and
    relative terminal increments, and requested tolerance are returned explicitly.

    References (APA 7th ed.):
        Ma, W., & de la Torre, J. (2016). A sequential cognitive diagnosis model for
            polytomous responses. *British Journal of Mathematical and Statistical
            Psychology, 69*(3), 253-275. https://doi.org/10.1111/bmsp.12070
        Tutz, G. (1990). Sequential item response models with an ordered response.
            *British Journal of Mathematical and Statistical Psychology, 43*(1), 39-55.
            https://doi.org/10.1111/j.2044-8317.1990.tb00925.x
        de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
            76*(2), 179-199. https://doi.org/10.1007/s11336-011-9207-7
    """
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    q, n_attributes = _validate_q_matrix_input(q_matrix, "q_matrix", n_items)
    if np.isinf(y).any():
        raise ValueError("responses must be finite ordered categories or NaN (missing)")
    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_seq_gdina"):
        raise RuntimeError("fit_seq_gdina requires the compiled Rust core")
    res = core.fit_seq_gdina(
        yy,
        observed.reshape(-1),
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_attributes),
        max_iter,
        tol,
    )
    return SeqGdinaFit(
        s_off=np.asarray(res["s_off"], dtype=np.int64),
        step_prob=np.asarray(res["step_prob"], dtype=np.float64),
        cat_off=np.asarray(res["cat_off"], dtype=np.int64),
        cat_prob=np.asarray(res["cat_prob"], dtype=np.float64),
        max_cat=np.asarray(res["max_cat"], dtype=np.int64),
        k_required=np.asarray(res["k_required"], dtype=np.int64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        final_relative_loglik_change=float(res["final_relative_loglik_change"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
        n_parameters=int(res["n_parameters"]),
    )


@dataclass
class SeqGdinaQrFit:
    """Fitted per-step-Q sequential G-DINA (Ma & de la Torre, 2016, restricted-Q).

    Each ordered STEP has its own attribute requirement ``q_ik``. Step probabilities are
    STEP-ROW-major: item ``i``'s step ``k`` is step row ``g = step_off[i] + (k-1)`` and owns
    ``2 ** step_kq[g]`` reduced classes at ``step_prob[spo[g]:spo[g+1]]`` (``step_kq[g] =
    |q_ik|``). Category probabilities are UNION-class-major: item ``i``'s union
    ``u_i = OR_k q_ik`` has ``2 ** union_k[i]`` classes and
    ``cat_prob[cat_off[i] + uc*(M_i+1) + x] = P(X_i = x | union class uc)``. ``max_cat`` is
    ``M_i`` (the number of steps); ``map_profile``/``attr_prob`` the per-person MAP profile and
    marginal attribute mastery."""

    step_off: np.ndarray
    spo: np.ndarray
    step_prob: np.ndarray
    step_kq: np.ndarray
    cat_off: np.ndarray
    cat_prob: np.ndarray
    max_cat: np.ndarray
    union_k: np.ndarray
    profile_prob: np.ndarray
    map_profile: np.ndarray
    attr_prob: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    termination_reason: str
    final_loglik_change: float
    final_relative_loglik_change: float
    stopping_tolerance: float
    n_parameters: int

    def item_step_prob(self, i: int, k: int) -> np.ndarray:
        """Step ``k`` (1-based) of item ``i``: its ``2 ** |q_ik|`` reduced-class continuation
        probabilities."""
        g = int(self.step_off[i]) + (k - 1)
        return self.step_prob[self.spo[g] : self.spo[g + 1]]


def fit_seq_gdina_qr(
    responses: np.ndarray,
    step_q: np.ndarray,
    n_steps,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> SeqGdinaQrFit:
    """Fit the per-step-Q sequential G-DINA (compute in Rust; Ma & de la Torre, 2016).

    The full restricted-Q sequential CDM: each ordered STEP ``k`` of item ``i`` is a saturated
    G-DINA over its OWN required attributes ``q_ik`` (step 1 may need attribute A, step 2 need A
    and B, etc.). Generalizes :func:`fit_seq_gdina` (which is this with every step of an item
    sharing the item's Q-vector). Estimated by marginal-ML EM with the closed-form saturated
    step ratio; each step's reduced class is computed directly from the attribute profile, and
    the item's union class indexes the category probabilities.

    ``responses`` is a persons x items array of ordered integer categories ``0..M_i`` (``NaN`` =
    missing, dropped MAR). ``step_q`` is a ``(sum_i n_steps[i]) x n_attributes`` 0/1 array (row
    ``step_off[i] + (k-1)`` is step ``k`` of item ``i``, ``step_off = cumsum(n_steps)``);
    ``n_steps[i] = M_i`` is item ``i``'s number of steps, which must equal its maximum observed
    category. Every declared step must measure at least one attribute, and every attribute must
    be required by at least one step. Those are necessary sanity checks, not a certificate of
    global model identifiability; supply an identified design and inspect ``converged``.

    References (APA 7th ed.):
        Ma, W., & de la Torre, J. (2016). A sequential cognitive diagnosis model for polytomous
            responses. *British Journal of Mathematical and Statistical Psychology, 69*(3),
            253-275. https://doi.org/10.1111/bmsp.12070
        de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika, 76*(2),
            179-199. https://doi.org/10.1007/s11336-011-9207-7
    """
    max_iter, tol = _validate_stopping_controls(max_iter, tol)
    y = _response_array(responses)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    steps, n_step_rows = _validate_seq_n_steps(n_steps, n_items)
    sq, n_attributes = _validate_q_matrix_input(step_q, "step_q", n_step_rows)
    if np.isinf(y).any():
        raise ValueError("responses must be finite ordered categories or NaN (missing)")
    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_seq_gdina_qr"):
        raise RuntimeError("fit_seq_gdina_qr requires the compiled Rust core")
    res = core.fit_seq_gdina_qr(
        yy,
        observed.reshape(-1),
        sq.reshape(-1),
        [int(m) for m in steps],
        int(n_persons),
        int(n_items),
        int(n_attributes),
        max_iter,
        tol,
    )
    return SeqGdinaQrFit(
        step_off=np.asarray(res["step_off"], dtype=np.int64),
        spo=np.asarray(res["spo"], dtype=np.int64),
        step_prob=np.asarray(res["step_prob"], dtype=np.float64),
        step_kq=np.asarray(res["step_kq"], dtype=np.int64),
        cat_off=np.asarray(res["cat_off"], dtype=np.int64),
        cat_prob=np.asarray(res["cat_prob"], dtype=np.float64),
        max_cat=np.asarray(res["max_cat"], dtype=np.int64),
        union_k=np.asarray(res["union_k"], dtype=np.int64),
        profile_prob=np.asarray(res["profile_prob"], dtype=np.float64),
        map_profile=np.asarray(res["map_profile"], dtype=np.int64),
        attr_prob=np.asarray(res["attr_prob"], dtype=np.float64).reshape(n_persons, n_attributes),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        final_loglik_change=float(res["final_loglik_change"]),
        final_relative_loglik_change=float(res["final_relative_loglik_change"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
        n_parameters=int(res["n_parameters"]),
    )

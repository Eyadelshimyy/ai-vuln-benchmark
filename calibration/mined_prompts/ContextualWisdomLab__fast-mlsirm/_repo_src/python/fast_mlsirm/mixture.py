"""Mixed Rasch / mixture IRT (Rost, 1990): the population is a mixture of latent
classes, each with its own item parameters, fit by marginal-ML EM in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_AGGREGATE_ITERS, MAX_MAX_ITER, MAX_RESTARTS


MAX_MIXTURE_CLASSES = 64
MAX_MIXTURE_BUFFER_CELLS = 60_000_000
_MAX_U64 = (1 << 64) - 1
_SUPPORTED_MIXTURE_MODELS = ("rasch", "Rasch", "RASCH", "2pl", "2PL", "twopl", "TwoPl")
_NUMPY_INTEGER_TYPES = tuple(
    np.dtype(name).type
    for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
)
_NUMPY_FLOAT_TYPES = tuple(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)


def _is_exact_type(value_type: type, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value_type`` is one trusted type without callbacks."""

    return any(value_type is trusted_type for trusted_type in trusted_types)


def _bounded_integer(value: object, name: str, lower: int, upper: int) -> int:
    """Return a trusted integer in ``lower..upper`` without subclass coercion."""

    value_type = type(value)
    if value_type is int:
        result = value
    elif _is_exact_type(value_type, _NUMPY_INTEGER_TYPES):
        result = int(value)
    else:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return result


def _finite_nonnegative_real(value: object, name: str) -> float:
    """Return a trusted finite non-negative real without subclass coercion."""

    value_type = type(value)
    if not (
        value_type is int
        or value_type is float
        or _is_exact_type(value_type, _NUMPY_INTEGER_TYPES)
        or _is_exact_type(value_type, _NUMPY_FLOAT_TYPES)
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _mixture_model_name(value: object) -> str:
    """Return one exact model alias accepted by the Rust mixture binding."""

    if type(value) is not str or value not in _SUPPORTED_MIXTURE_MODELS:
        raise ValueError("model must be one of rasch, Rasch, RASCH, 2pl, 2PL, twopl, TwoPl")
    return value


@dataclass
class MixtureFit:
    """Fitted mixture-IRT model (Rost, 1990).

    ``a``/``b`` are the per-class item discriminations and difficulties, shape
    ``(n_classes, n_items)`` (``a`` is all ones for the Rasch model); ``pi`` the
    mixing proportions; ``class_posterior`` the ``(n_persons, n_classes)`` class
    responsibilities ``P(class | x_j)``; ``map_class`` the per-person modal class;
    ``theta`` the mixture-EAP ability. Classes are in canonical order (mixing weight
    descending, ties broken by mean difficulty ascending)."""

    model: str
    n_classes: int
    a: np.ndarray
    b: np.ndarray
    pi: np.ndarray
    class_posterior: np.ndarray
    map_class: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int


def fit_mixture(
    responses: np.ndarray,
    n_classes: int = 2,
    *,
    model: str,
    n_starts: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> MixtureFit:
    """Fit a mixed Rasch / mixture-IRT model (compute in Rust; Rost, 1990).

    The population is modeled as a mixture of ``n_classes`` latent classes, each with
    its own item parameters and a mixing proportion, detecting unobserved
    heterogeneity (qualitatively different response strategies). Within a class,
    responses follow a Rasch (``model="rasch"``, discrimination fixed at 1) or 2PL
    (``model="2pl"``) model with ability ``theta ~ N(0, 1)``, estimated by marginal-ML
    EM. Because the mixture likelihood is multimodal, pass ``n_starts > 1`` to run
    several restarts and keep the highest-likelihood fit (start 0 is a deterministic
    warm start). ``responses`` is a persons x items 0/1 array (``NaN`` = missing,
    dropped under MAR). Classes are returned in a canonical order.

    Note: Rost (1990) and ``psychomix`` (Frick et al., 2012) fit Rasch mixtures by
    conditional ML within class. This function instead combines Rost's latent-class
    structure with a fixed-standard-normal, Bock-Aitkin marginal-ML EM estimator. That
    estimator is a repository-specific operationalization; finite-sample equivalence
    to the conditional-ML item estimates is not asserted.

    References (APA 7th ed.):
        Rost, J. (1990). Rasch models in latent classes: An integration of two
            approaches to item analysis. *Applied Psychological Measurement, 14*(3),
            271–282. https://doi.org/10.1177/014662169001400305
        Rost, J., & von Davier, M. (1995). Mixture distribution Rasch models. In G. H.
            Fischer & I. W. Molenaar (Eds.), *Rasch models: Foundations, recent
            developments, and applications* (pp. 257–268). Springer.
            https://doi.org/10.1007/978-1-4612-4230-7_14
        Frick, H., Strobl, C., Leisch, F., & Zeileis, A. (2012). Flexible Rasch
            mixture models with package psychomix. *Journal of Statistical Software,
            48*(7), 1–25. https://doi.org/10.18637/jss.v048.i07
    """
    from .fitstats import _core_module

    n_classes_int = _bounded_integer(n_classes, "n_classes", 1, MAX_MIXTURE_CLASSES)
    n_starts_int = _bounded_integer(n_starts, "n_starts", 1, MAX_RESTARTS)
    max_iter_int = _bounded_integer(max_iter, "max_iter", 1, MAX_MAX_ITER)
    model_value = _mixture_model_name(model)
    tol_value = _finite_nonnegative_real(tol, "tol")
    seed_int = _bounded_integer(seed, "seed", 0, _MAX_U64)

    raw_y = np.asarray(responses)
    if np.iscomplexobj(raw_y) or raw_y.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("responses must be real-valued")
    y = np.asarray(raw_y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    if np.isinf(y).any():
        raise ValueError("responses must be finite where not missing")
    n_persons, n_items = y.shape

    aggregate_iterations = n_classes_int * n_starts_int * max_iter_int
    if aggregate_iterations > MAX_AGGREGATE_ITERS:
        raise ValueError(
            f"n_classes x n_starts x max_iter ({aggregate_iterations}) exceeds the "
            f"{MAX_AGGREGATE_ITERS}-iteration mixture budget"
        )
    for label, cells in (
        ("person-class", n_persons * n_classes_int),
        ("class-item", n_classes_int * n_items),
    ):
        if cells > MAX_MIXTURE_BUFFER_CELLS:
            raise ValueError(
                f"{label} buffer ({cells} cells) exceeds the "
                f"{MAX_MIXTURE_BUFFER_CELLS}-cell mixture limit"
            )
    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)

    core = _core_module()
    if core is None or not hasattr(core, "fit_mixture"):
        raise RuntimeError("fit_mixture requires the compiled Rust core")

    res = core.fit_mixture(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        n_classes_int,
        model_value,
        n_starts_int,
        max_iter_int,
        tol_value,
        seed_int,
    )
    c = int(res["n_classes"])
    return MixtureFit(
        model=str(res["model"]),
        n_classes=c,
        a=np.asarray(res["a"], dtype=np.float64).reshape(c, n_items),
        b=np.asarray(res["b"], dtype=np.float64).reshape(c, n_items),
        pi=np.asarray(res["pi"], dtype=np.float64),
        class_posterior=np.asarray(res["class_posterior"], dtype=np.float64).reshape(
            n_persons, c
        ),
        map_class=np.asarray(res["map_class"], dtype=np.int64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )

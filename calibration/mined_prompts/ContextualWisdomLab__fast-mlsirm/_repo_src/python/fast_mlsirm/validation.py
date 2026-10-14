"""Machine-scoring validation gates for LLM-as-a-Judge calibration.

Implements the operational criteria of Williamson, Xi & Breyer (2012),
"A Framework for Evaluation and Use of Automated Scoring" (EM:IP 31(1)):
quadratic-weighted kappa >= .70, Pearson r >= .70, degradation from the
human-human baseline <= .10, |SMD| <= .15 overall and <= .10 within every
subgroup; exact/adjacent agreement are reported but are explicitly NOT gates.
All computation runs in the Rust core (`mlsirm_core::agreement`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


MAX_JUDGE_CATEGORIES = 1_000
_TRUSTED_NUMPY_INTEGER_SCALAR_TYPES = tuple(
    np.dtype(code).type
    for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "p", "P")
)
_TRUSTED_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _is_exact_numpy_integer_scalar_type(value_type: type) -> bool:
    """Return whether ``value_type`` is a package-trusted NumPy integer type.

    Identity comparisons deliberately avoid hashing or equality on a
    caller-controlled metaclass. This keeps type admission inert even for a
    NumPy scalar subclass that overrides metaclass ``__hash__`` or ``__eq__``.
    """
    return any(value_type is trusted_type for trusted_type in _TRUSTED_NUMPY_INTEGER_SCALAR_TYPES)


def _trusted_policy_real(value: object, name: str) -> float:
    """Normalize one trusted policy threshold without caller callback dispatch."""
    value_type = type(value)
    if not (
        value_type is int
        or value_type is float
        or _is_exact_numpy_integer_scalar_type(value_type)
        or any(value_type is scalar_type for scalar_type in _TRUSTED_NUMPY_FLOAT_SCALAR_TYPES)
    ):
        raise ValueError(f"{name} must be a real number in 0..1")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number in 0..1") from exc
    if not (0.0 <= normalized <= 1.0):
        raise ValueError(f"{name} must be in 0..1")
    return normalized


def _trusted_judge_category_count(value: object) -> int:
    """Return a trusted built-in category count without caller coercion callbacks.

    Only an exact built-in :class:`int` or an exact concrete NumPy integer scalar
    identity is normalized. Python/NumPy subclasses, booleans, and arbitrary
    integer-protocol providers are rejected before ``int`` can execute caller
    code. The returned value is always an exact built-in integer suitable for
    downstream NumPy validation and PyO3 marshalling.
    """
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif _is_exact_numpy_integer_scalar_type(value_type):
        normalized = int(value)
    else:
        raise ValueError("k (number of categories) must be an integer")

    if normalized < 2:
        raise ValueError("k (number of categories) must be >= 2")
    if normalized > MAX_JUDGE_CATEGORIES:
        # k drives a dense k-by-k confusion matrix in the Rust core.
        raise ValueError(f"k (number of categories) must be <= {MAX_JUDGE_CATEGORIES}")
    return normalized


def _validate_labels(a, name: str, *, k: int | None = None, n: int | None = None) -> np.ndarray:
    """Validate caller-supplied category labels before the uint32 conversion the
    Rust gate expects: reject non-1-D, wrong-length, non-finite, non-integer,
    negative, or (when ``k`` given) out-of-range values instead of silently
    truncating/wrapping them (which would let malformed labels pass the gate)."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} length must match the paired labels")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr.astype(np.float64))):
        raise ValueError(f"{name} must be finite")
    fl = arr.astype(np.float64)
    if np.any(fl < 0) or np.any(fl != np.floor(fl)):
        raise ValueError(f"{name} must be non-negative integers")
    if np.any(fl > np.iinfo(np.uint32).max):
        raise ValueError(f"{name} values must fit in uint32")
    if k is not None and np.any(fl >= k):
        raise ValueError(f"{name} values must be in 0..k-1")
    return arr.astype(np.uint32)


DEFAULT_VALIDATION_POLICY_ID = "williamson_high_stakes"
DEFAULT_VALIDATION_POLICY_VERSION = "1.0"


@dataclass(frozen=True)
class ValidationPolicy:
    """Governed threshold policy for automated-scoring acceptance gates.

    Thresholds are validated in Python and marshaled to the Rust decision owner.
    Defaults match Williamson, Xi & Breyer (2012) high-stakes guidance.
    """

    policy_id: str = DEFAULT_VALIDATION_POLICY_ID
    policy_version: str = DEFAULT_VALIDATION_POLICY_VERSION
    qwk_min: float = 0.70
    pearson_r_min: float = 0.70
    degradation_max: float = 0.10
    overall_smd_max: float = 0.15
    subgroup_smd_max: float = 0.10
    min_subgroup_n: int = 2

    def __post_init__(self) -> None:
        """Reject empty identities and thresholds outside closed unit intervals."""
        if type(self.policy_id) is not str or not self.policy_id.strip():
            raise ValueError("policy_id must be a non-empty string")
        if type(self.policy_version) is not str or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        for name in (
            "qwk_min",
            "pearson_r_min",
            "degradation_max",
            "overall_smd_max",
            "subgroup_smd_max",
        ):
            object.__setattr__(self, name, _trusted_policy_real(getattr(self, name), name))
        n = self.min_subgroup_n
        if type(n) is not int or n < 2:
            raise ValueError("min_subgroup_n must be an integer >= 2")

    def rust_kwargs(self) -> dict[str, Any]:
        """Keyword arguments consumed by the Rust ``validate_scoring`` owner."""
        return {
            "qwk_min": float(self.qwk_min),
            "pearson_r_min": float(self.pearson_r_min),
            "degradation_max": float(self.degradation_max),
            "overall_smd_max": float(self.overall_smd_max),
            "subgroup_smd_max": float(self.subgroup_smd_max),
            "min_subgroup_n": int(self.min_subgroup_n),
        }


@dataclass
class ValidationVerdict:
    """Outcome of validating a judge against reference labels.

    Holds the per-gate results, the exact and adjacent agreement rates, the
    overall pass/fail verdict, the names of any gates that failed, and the
    governing policy identity/version.
    """

    gates: list[dict[str, Any]]
    exact_agreement: float
    adjacent_agreement: float
    passed: bool
    failed_gates: list[str] = field(default_factory=list)
    policy_id: str = DEFAULT_VALIDATION_POLICY_ID
    policy_version: str = DEFAULT_VALIDATION_POLICY_VERSION


def validate_judge(
    judge: np.ndarray,
    human: np.ndarray,
    k: int = 2,
    human_human: tuple[np.ndarray, np.ndarray] | None = None,
    subgroup: np.ndarray | None = None,
    policy: ValidationPolicy | None = None,
) -> ValidationVerdict:
    """Run the Williamson et al. (2012) conjunctive acceptance gates.

    ``judge``/``human`` are paired labels in ``0..k-1``; ``human_human`` is an
    optional double-scored human baseline (pair of label vectors) for the
    degradation criterion; ``subgroup`` labels each observation for the
    fairness SMD.
    """
    active_policy = policy if policy is not None else ValidationPolicy()
    if not isinstance(active_policy, ValidationPolicy):
        raise TypeError("policy must be a ValidationPolicy")

    category_count = _trusted_judge_category_count(k)
    judge_v = _validate_labels(judge, "judge", k=category_count)
    human_v = _validate_labels(human, "human", k=category_count, n=judge_v.shape[0])
    kwargs: dict[str, Any] = {}
    if human_human is not None:
        kwargs["human_a"] = _validate_labels(
            human_human[0], "human_a", k=category_count, n=judge_v.shape[0]
        )
        kwargs["human_b"] = _validate_labels(
            human_human[1], "human_b", k=category_count, n=kwargs["human_a"].shape[0]
        )
    if subgroup is not None:
        sg = _validate_labels(subgroup, "subgroup", n=judge_v.shape[0])
        # Compact to contiguous ids: the Rust core loops 0..max(subgroup)+1,
        # so a sparse label (e.g. uint32 max) is an O(n_groups) CPU-DoS.
        _uniq, sg_compact = np.unique(sg, return_inverse=True)
        kwargs["subgroup"] = sg_compact.astype(np.uint32)
    kwargs.update(active_policy.rust_kwargs())

    from . import _core  # computation lives in the Rust core

    res = _core.validate_scoring(
        judge_v,
        human_v,
        category_count,
        **kwargs,
    )
    gates = [dict(g) for g in res["gates"]]
    return ValidationVerdict(
        gates=gates,
        exact_agreement=float(res["exact_agreement"]),
        adjacent_agreement=float(res["adjacent_agreement"]),
        passed=bool(res["pass"]),
        failed_gates=[g["name"] for g in gates if not g["pass"]],
        policy_id=active_policy.policy_id,
        policy_version=active_policy.policy_version,
    )


@dataclass
class FleissKappaResult:
    """Result of :func:`fleiss_kappa`. In exact (Conger) mode ``z`` and
    ``p_value`` are NaN and the category arrays are empty, mirroring irr's
    ``kappam.fleiss(exact=TRUE)`` which returns neither."""

    kappa: float
    subjects_used: int
    z: float
    p_value: float
    category_kappa: np.ndarray
    category_z: np.ndarray
    category_p: np.ndarray


def fleiss_kappa(
    ratings: np.ndarray,
    k: int | None = None,
    exact: bool = False,
) -> FleissKappaResult:
    """Fleiss' kappa for nominal agreement among multiple raters, with the
    exact (Conger) chance-agreement variant.

    Reimplements ``kappam.fleiss()`` from CRAN irr 0.85 (R source READ in
    full; algorithm source of truth). Model origins — cited as origins only,
    NOT READ: Fleiss, J. L. (1971). Measuring nominal scale agreement among
    many raters. *Psychological Bulletin, 76*(5), 378-382; Conger, A. J.
    (1980). Integration and generalization of kappas for multiple raters.
    *Psychological Bulletin, 88*(2), 322-328. Computation runs in the Rust
    core (``mlsirm_core::agreement::fleiss_kappa``).

    ``ratings`` is a 2-D ``(n_subjects, n_raters)`` array of integer category
    codes ``0..k-1``. NaN or any negative value marks a missing rating and
    drops the whole subject row (listwise, as in irr). ``k=None`` infers
    ``max(code)+1``; pass ``k`` explicitly to include trailing empty
    categories in the category-wise detail (their kappas are NaN, matching
    R's 0/0).
    """
    from . import _core  # computation lives in the Rust core

    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D (subjects x raters) array")
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype == object:
        for v in arr.flat:
            if v is None or isinstance(v, (bool, np.bool_, str, bytes)):
                raise ValueError("ratings must be numeric, not boolean/str/None")
        arr = arr.astype(np.float64)
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be integer codes, not booleans")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"ratings dtype {arr.dtype} is not numeric")
    ns, nr = arr.shape
    if arr.dtype.kind == "f":
        finite = np.isfinite(arr)
        if np.any(np.isinf(arr)):
            raise ValueError("ratings must not contain infinities")
        if np.any(arr[finite] != np.floor(arr[finite])):
            raise ValueError("ratings must be integer category codes")
        if np.any(np.abs(arr[finite]) > 2.0**53):
            raise ValueError("ratings exceed exact float64 integer range")
        codes = np.where(finite, arr, -1.0).astype(np.int64)
    else:
        if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
            raise ValueError("ratings values must fit in int64")
        codes = arr.astype(np.int64)
    if k is not None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise ValueError("k must be an integer")
        k = int(k)
    if k is None:
        if codes.size == 0 or int(codes.max()) < 0:
            raise ValueError("cannot infer k: no observed category codes")
        k = int(codes.max()) + 1
    res = _core.fleiss_kappa(
        np.ascontiguousarray(codes.reshape(-1)), int(ns), int(nr), int(k), bool(exact)
    )
    return FleissKappaResult(
        kappa=float(res["kappa"]),
        subjects_used=int(res["subjects_used"]),
        z=float(res["z"]),
        p_value=float(res["p_value"]),
        category_kappa=np.asarray(res["category_kappa"]),
        category_z=np.asarray(res["category_z"]),
        category_p=np.asarray(res["category_p"]),
    )


@dataclass
class LightKappaResult:
    """Result of :func:`light_kappa`: mean pairwise unweighted Cohen's kappa
    (``value``), the per-pair kappas in ``(i, j)``, ``i < j`` order, and
    Light's chance-product z test."""

    value: float
    subjects_used: int
    raters: int
    kappas: np.ndarray
    z: float
    p_value: float


def light_kappa(ratings: np.ndarray) -> LightKappaResult:
    """Light's kappa: mean pairwise unweighted Cohen's kappa over all rater
    pairs, with Light's chance-product z statistic.

    Reimplements ``kappam.light()`` and the unweighted branch of ``kappa2()``
    from CRAN irr 0.85 (both R sources READ in full; algorithm source of
    truth). Method origin — cited as origin only, NOT READ: Light, R. J.
    (1971). Measures of response agreement for qualitative data: Some
    generalizations and alternatives. *Psychological Bulletin, 76*(5),
    365-377. Computation runs in the Rust core
    (``mlsirm_core::agreement::light_kappa``).

    ``ratings`` is a 2-D ``(n_subjects, n_raters)`` array of integer category
    codes (any integer labels; levels are compacted internally). NaN or any
    negative integer marks a missing rating and drops the whole subject row
    (listwise, as in irr's ``na.omit``). Negative *floats* are rejected
    rather than treated as missing: a float payload is only accepted when
    every finite value is a non-negative integer, so a stray ``-1.0`` or
    ``2.5`` fails loudly instead of silently changing the level set.
    """
    from . import _core  # computation lives in the Rust core

    if isinstance(ratings, np.ma.MaskedArray):
        raise ValueError("masked arrays are not supported; use NaN for missing")
    arr = np.asarray(ratings)
    if arr.dtype == object:
        raise ValueError("object-dtype ratings are not supported")
    if arr.ndim != 2:
        raise ValueError("ratings must be a 2-D (subjects x raters) array")
    if np.iscomplexobj(arr):
        raise ValueError("ratings must be real-valued")
    if arr.dtype.kind == "b":
        raise ValueError("ratings must be integer codes, not booleans")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"ratings dtype {arr.dtype} is not numeric")
    ns, nr = arr.shape
    if arr.dtype.kind == "f":
        if np.any(np.isinf(arr)):
            raise ValueError("ratings must not contain infinities")
        finite = np.isfinite(arr)
        vals = arr[finite]
        if np.any(vals != np.floor(vals)):
            raise ValueError("ratings must be integer category codes")
        if np.any(vals < 0):
            raise ValueError(
                "negative float ratings are rejected; use NaN for missing"
            )
        if vals.size and np.any(vals > 2.0**32):
            raise ValueError("ratings must be <= 2^32")
        codes = np.where(finite, arr, -1.0).astype(np.int64)
    else:
        if arr.size and int(arr.max()) > 2**32:
            raise ValueError("ratings must be <= 2^32")
        codes = arr.astype(np.int64)
    res = _core.light_kappa(np.ascontiguousarray(codes.reshape(-1)), int(ns), int(nr))
    return LightKappaResult(
        value=float(res["value"]),
        subjects_used=int(res["subjects_used"]),
        raters=int(res["raters"]),
        kappas=np.asarray(res["kappas"]),
        z=float(res["z"]),
        p_value=float(res["p_value"]),
    )

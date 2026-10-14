"""Automated test assembly (ATA) toward a target information function.

This module is a *downstream application* on already-calibrated item
parameters. It does not touch the calibration objective, likelihood, gradients,
or the MLS2PLM formula; it reuses the repository's 2PL Fisher item information
(:func:`fast_mlsirm.test_design.item_information`) to assemble a fixed-length
form whose test information function (TIF) approximates a caller-specified
target ``T(theta_k)`` at a set of trait points, subject to length, content, and
basic exposure constraints.

Relationship to :func:`fast_mlsirm.test_design.assemble_test_form`
------------------------------------------------------------------
``assemble_test_form`` maximizes a single precomputed item-information vector
under content constraints. This module instead targets a *test information
function evaluated at several trait points* and adds exposure control, matching
the optimal-test-design framing of van der Linden (2005): the assembled form's
TIF is driven toward the target across all specified ``theta`` points rather
than maximized at one point.

Method
------
The exact formulation of van der Linden (2005) is a 0-1 integer program that
selects ``x_i in {0,1}`` to make ``I(theta_k) = sum_i x_i I_i(theta_k)`` meet a
target ``T(theta_k)`` subject to ``sum_i x_i = n`` and content/exposure/enemy
constraints. To stay dependency-free (no SciPy/solver), this uses the standard
greedy surrogate: at each step add the eligible item that most reduces the total
*capped shortfall* ``sum_k max(0, T_k - I(theta_k))`` -- i.e. maximizes
``sum_k [min(T_k, I_k + I_ik) - min(T_k, I_k)]``. Capping at the target rewards
covering under-informed points and does not reward overshoot, so information is
spread to match the target curve. This is deterministic given ``seed`` (used
only to break exact ties reproducibly).

References (APA 7th ed.)
------------------------
van der Linden, W. J. (2005). *Linear models for optimal test design*.
    Springer. https://doi.org/10.1007/0-387-29054-0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._ata_core_loader import ata_core
from .test_design import item_information
from .types import MLSIRMParams

__all__ = [
    "AssembledForm",
    "assemble_to_target",
    "item_information_matrix",
]

_TIE_EPS = 1e-12
# Keep caller target evidence and the dense target-point × item information
# surface inside the repository's established scientific-evidence envelope.
MAX_ATA_TARGET_CELLS = 20_000_000
MAX_ATA_INFORMATION_CELLS = 20_000_000
MAX_ATA_TARGET_NESTING = 64
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
_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_NUMPY_REAL_SCALAR_TYPES = (
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
)


@dataclass
class AssembledForm:
    """Result of a target-information assembly.

    Attributes
    ----------
    items:
        Selected item indices (assembly order).
    achieved_info:
        Achieved test information ``sum_i I_i(theta_k)`` at each target point.
    target_info:
        The requested target information at each target point.
    shortfall:
        Total capped shortfall ``sum_k max(0, T_k - achieved_k)`` (0 when the
        target is met at every point).
    content_counts:
        Count of selected items per content label (empty when no content was
        supplied).
    """

    items: np.ndarray
    achieved_info: np.ndarray
    target_info: np.ndarray
    shortfall: float
    content_counts: dict[str, int]


def _is_exact_public_real_scalar(value: object) -> bool:
    """Return whether ``value`` is one package-trusted real scalar identity."""
    value_type = type(value)
    return value_type in {bool, int, float} or any(
        value_type is scalar_type for scalar_type in _NUMPY_REAL_SCALAR_TYPES
    )


def _require_lossless_float64_scalar(value: object, name: str) -> None:
    """Require one already-trusted scalar to preserve identity in binary64."""
    value_type = type(value)
    if value_type is bool or value_type is np.bool_ or value_type is float:
        return
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be real numeric evidence") from None

    if value_type is int or any(
        value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES
    ):
        if not np.isfinite(converted):
            raise ValueError(f"{name} must be real numeric evidence")
        if int(converted) != int(value):
            raise ValueError(f"{name} could not be converted losslessly to float64")
        return

    # The only remaining admitted identities are concrete NumPy real scalars.
    if not np.isfinite(converted):
        if np.isfinite(value):
            raise ValueError(f"{name} could not be converted losslessly to float64")
        return
    if value_type(converted) != value:
        raise ValueError(f"{name} could not be converted losslessly to float64")


def _raise_target_resource_limit(name: str) -> None:
    """Raise the stable ATA target-evidence resource diagnostic."""
    raise ValueError(
        f"{name} exceeds the {MAX_ATA_TARGET_CELLS}-cell ATA evidence limit"
    )


def _raise_target_nesting_limit(name: str) -> None:
    """Raise the stable ATA target-evidence nesting diagnostic."""
    raise ValueError(
        f"{name} exceeds the {MAX_ATA_TARGET_NESTING}-level ATA nesting limit"
    )


def _raise_information_resource_limit() -> None:
    """Raise the stable ATA dense-information resource diagnostic."""
    raise ValueError(
        "target information matrix exceeds the "
        f"{MAX_ATA_INFORMATION_CELLS}-cell ATA limit"
    )


def _preflight_builtin_real_tree(value: list | tuple, name: str) -> tuple[tuple[int, ...], int]:
    """Return rectangular shape/cell count without eager sibling expansion.

    The explicit stack holds one frame per active nesting level. Logical scalar
    cells, nesting depth, and structural nodes are independently bounded so
    malformed trees made mostly of empty containers cannot consume unbounded
    Python traversal work. Exact NumPy real-numeric array leaves retain their
    inert shape while their logical cells are charged before materialization.
    """
    # frame: [container, next child index, first child shape, logical cells]
    stack: list[list[object]] = [[value, 0, None, 0]]
    active: set[int] = set()
    structural_nodes = 1
    max_structural_nodes = 2 * MAX_ATA_TARGET_CELLS + 1

    while stack:
        current = stack[-1][0]
        if type(current) is not list and type(current) is not tuple:
            raise ValueError(f"{name} must be real numeric evidence")
        child_index = int(stack[-1][1])
        if child_index == 0:
            current_id = id(current)
            if current_id in active:
                raise ValueError(f"{name} must be real numeric evidence")
            active.add(current_id)

        if child_index >= len(current):
            active.remove(id(current))
            first_child_shape = stack[-1][2]
            logical_cells = int(stack[-1][3])
            if first_child_shape is None:
                child_shape: tuple[int, ...] = ()
            elif type(first_child_shape) is tuple:
                child_shape = first_child_shape
            else:
                raise ValueError(f"{name} must be real numeric evidence")
            shape = (len(current),) + child_shape
            stack.pop()
            if not stack:
                return shape, logical_cells
            parent_shape = stack[-1][2]
            if parent_shape is None:
                stack[-1][2] = shape
            elif parent_shape != shape:
                raise ValueError(f"{name} must be real numeric evidence")
            stack[-1][3] = int(stack[-1][3]) + logical_cells
            if int(stack[-1][3]) > MAX_ATA_TARGET_CELLS:
                _raise_target_resource_limit(name)
            continue

        child = current[child_index]
        stack[-1][1] = child_index + 1
        structural_nodes += 1
        if structural_nodes > max_structural_nodes:
            _raise_target_resource_limit(name)

        child_type = type(child)
        if child_type is list or child_type is tuple:
            if id(child) in active:
                raise ValueError(f"{name} must be real numeric evidence")
            if len(stack) >= MAX_ATA_TARGET_NESTING:
                _raise_target_nesting_limit(name)
            stack.append([child, 0, None, 0])
            continue
        if child_type is np.ndarray:
            if child.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(f"{name} must be real numeric evidence")
            child_cells = int(child.size)
            next_cells = int(stack[-1][3]) + child_cells
            if next_cells > MAX_ATA_TARGET_CELLS:
                _raise_target_resource_limit(name)
            child_shape = child.shape
            first_child_shape = stack[-1][2]
            if first_child_shape is None:
                stack[-1][2] = child_shape
            elif first_child_shape != child_shape:
                raise ValueError(f"{name} must be real numeric evidence")
            stack[-1][3] = next_cells
            continue
        if not _is_exact_public_real_scalar(child):
            raise ValueError(f"{name} must be real numeric evidence")
        first_child_shape = stack[-1][2]
        if first_child_shape is None:
            stack[-1][2] = ()
        elif first_child_shape != ():
            raise ValueError(f"{name} must be real numeric evidence")
        stack[-1][3] = int(stack[-1][3]) + 1
        if int(stack[-1][3]) > MAX_ATA_TARGET_CELLS:
            _raise_target_resource_limit(name)

    raise RuntimeError("ATA target preflight terminated without a result")


def _preflight_real_evidence(value: object, name: str) -> tuple[tuple[int, ...], int]:
    """Return trusted real-evidence shape/cell count before per-cell conversion."""
    if type(value) is np.ndarray:
        if value.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(f"{name} must be real numeric evidence")
        if value.size > MAX_ATA_TARGET_CELLS:
            _raise_target_resource_limit(name)
        return value.shape, int(value.size)
    if _is_exact_public_real_scalar(value):
        if MAX_ATA_TARGET_CELLS < 1:
            _raise_target_resource_limit(name)
        return (), 1
    if type(value) is list or type(value) is tuple:
        return _preflight_builtin_real_tree(value, name)
    raise ValueError(f"{name} must be real numeric evidence")


def _trusted_real_array(value: object, name: str) -> np.ndarray:
    """Materialize bounded inert real evidence without caller callbacks.

    Exact NumPy real-numeric arrays, exact trusted real scalars, and exact
    built-in list/tuple trees whose leaves are package-trusted real scalars or
    exact NumPy real-numeric arrays are admitted. Logical cells, nesting depth,
    and malformed structural traversal are bounded before NumPy conversion.
    Every admitted scalar must preserve its value when normalized to binary64.
    Already-normalized exact float64 arrays need no Python-level scalar replay.
    Arbitrary array providers, ndarray/container/numeric subclasses,
    object/text/complex storage, and cyclic built-in trees fail before NumPy
    conversion.
    """
    _preflight_real_evidence(value, name)
    if type(value) is np.ndarray:
        if value.dtype != np.dtype(np.float64):
            for scalar in value.flat:
                _require_lossless_float64_scalar(scalar, name)
    elif _is_exact_public_real_scalar(value):
        _require_lossless_float64_scalar(value, name)
    else:
        if type(value) is not list and type(value) is not tuple:
            raise ValueError(f"{name} must be real numeric evidence")
        active: set[int] = set()
        stack: list[tuple[list | tuple, int]] = [(value, 0)]
        logical_cells = 0
        structural_nodes = 1
        max_structural_nodes = 2 * MAX_ATA_TARGET_CELLS + 1
        while stack:
            current, child_index = stack[-1]
            if child_index == 0:
                current_id = id(current)
                if current_id in active:
                    raise ValueError(f"{name} must be real numeric evidence")
                active.add(current_id)
            if child_index >= len(current):
                active.remove(id(current))
                stack.pop()
                continue
            child = current[child_index]
            stack[-1] = (current, child_index + 1)
            structural_nodes += 1
            if structural_nodes > max_structural_nodes:
                _raise_target_resource_limit(name)
            if type(child) is list or type(child) is tuple:
                if id(child) in active:
                    raise ValueError(f"{name} must be real numeric evidence")
                stack.append((child, 0))
                continue
            if type(child) is np.ndarray:
                if child.dtype.kind not in {"b", "i", "u", "f"}:
                    raise ValueError(f"{name} must be real numeric evidence")
                child_cells = int(child.size)
                logical_cells += child_cells
                if logical_cells > MAX_ATA_TARGET_CELLS:
                    _raise_target_resource_limit(name)
                for scalar in child.flat:
                    _require_lossless_float64_scalar(scalar, name)
                continue
            if not _is_exact_public_real_scalar(child):
                raise ValueError(f"{name} must be real numeric evidence")
            logical_cells += 1
            if logical_cells > MAX_ATA_TARGET_CELLS:
                _raise_target_resource_limit(name)
            _require_lossless_float64_scalar(child, name)

    try:
        return np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be real numeric evidence") from None


def _target_theta_rows(
    target_thetas: object,
    n_dims: int,
    *,
    n_items: int | None = None,
) -> np.ndarray:
    """Return a trusted target-trait matrix with shape ``(n_points, n_dims)``."""
    shape, _ = _preflight_real_evidence(target_thetas, "target_thetas")
    if len(shape) == 1:
        if n_dims == 1:
            n_points = shape[0]
        elif shape == (n_dims,):
            n_points = 1
        else:
            raise ValueError("target_thetas must have shape (n_points, n_dims)")
    elif len(shape) == 2 and shape[1] == n_dims:
        n_points = shape[0]
    else:
        raise ValueError("target_thetas must have shape (n_points, n_dims)")
    if n_points < 1:
        raise ValueError("at least one target theta point is required")
    if n_items is not None and n_points * n_items > MAX_ATA_INFORMATION_CELLS:
        _raise_information_resource_limit()

    thetas = _trusted_real_array(target_thetas, "target_thetas")
    if thetas.ndim == 1:
        if n_dims == 1:
            thetas = thetas[:, None]
        else:
            thetas = thetas[None, :]
    if thetas.ndim != 2 or thetas.shape[1] != n_dims:
        raise ValueError("target_thetas must have shape (n_points, n_dims)")
    if not np.all(np.isfinite(thetas)):
        raise ValueError("target_thetas must be finite")
    return thetas


def _target_info_vector(target_info: object, n_points: int) -> np.ndarray:
    """Return one trusted finite non-negative target-information vector."""
    _, logical_cells = _preflight_real_evidence(target_info, "target_info")
    if logical_cells != n_points:
        raise ValueError("target_info must have one entry per target theta point")
    target = _trusted_real_array(target_info, "target_info").ravel()
    if target.shape != (n_points,):
        raise ValueError("target_info must have one entry per target theta point")
    if np.any(target < 0) or not np.all(np.isfinite(target)):
        raise ValueError("target_info must be finite and non-negative")
    return target


def item_information_matrix(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    target_thetas: np.ndarray,
    *,
    model: str = "MLS2PLM",
) -> np.ndarray:
    """Return the ``(n_points, n_items)`` item-information matrix.

    Entry ``[k, i]`` is the reused 2PL Fisher information ``I_i(theta_k)`` of
    item ``i`` at target trait point ``k`` (holding the latent-space position at
    the bank's population mean, as in
    :func:`fast_mlsirm.test_design.item_information`).
    """
    n_dims = int(np.asarray(bank.theta).shape[1])
    n_items = int(np.asarray(bank.b).shape[0])
    thetas = _target_theta_rows(target_thetas, n_dims, n_items=n_items)
    matrix = np.empty((thetas.shape[0], n_items), dtype=np.float64)
    for k in range(thetas.shape[0]):
        matrix[k] = item_information(bank, factor_id, theta=thetas[k], model=model)
    return matrix


def _content_feasible(
    labels: np.ndarray | None,
    selected: list[int],
    counts: dict[str, int],
    eligible_now: np.ndarray,
    length: int,
    min_counts: dict[str, int],
) -> bool:
    """Check that remaining min-content requirements can still be satisfied."""
    slots_left = length - len(selected)
    required_left = sum(max(0, m - counts.get(lbl, 0)) for lbl, m in min_counts.items())
    if required_left > slots_left:
        return False
    if labels is None or not min_counts:
        return True
    for lbl, minimum in min_counts.items():
        needed = max(0, minimum - counts.get(lbl, 0))
        if needed == 0:
            continue
        available = int(np.sum(labels[eligible_now] == lbl))
        if available < needed:
            return False
    return True


def _is_exact_public_string(value: object) -> bool:
    """Return whether ``value`` has one package-trusted string scalar identity."""
    value_type = type(value)
    return value_type is str or value_type is np.str_


def _exact_public_string(value: object, error_message: str) -> str:
    """Return an inert string while rejecting caller-defined string subclasses."""
    if not _is_exact_public_string(value):
        raise ValueError(error_message)
    return value if type(value) is str else str(value)


def _validated_content_labels(content: np.ndarray | None, n_items: int) -> np.ndarray | None:
    """Return bounded exact string labels without arbitrary object coercion.

    Caller-controlled objects and string subclasses are rejected by exact type
    before NumPy is allowed to stringify them. This keeps ``__str__`` and
    ``__repr__`` callbacks outside the ATA trust boundary and lets invalid labels
    fail before item-information work.
    """
    if content is None:
        return None
    labels = np.asarray(content, dtype=object)
    if labels.shape != (n_items,):
        raise ValueError("content length must match the number of items")
    normalized = [
        _exact_public_string(label, "content labels must be strings")
        for label in labels.flat
    ]
    return np.asarray(normalized, dtype=str).reshape(labels.shape)


def _is_exact_public_integer(value: object) -> bool:
    """Return whether ``value`` has one package-trusted integer scalar identity."""
    value_type = type(value)
    return value_type is int or any(
        value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES
    )


def _exact_public_integer(value: object, name: str) -> int:
    """Return one exact integer while rejecting bools and conversion hooks.

    Public ATA scalar/map counts are a type boundary: only exact Python ``int``
    and explicitly supported NumPy integer scalar identities are admitted.
    Booleans, caller-defined subclasses, and arbitrary ``__int__``/``__index__``
    providers are rejected before any caller conversion callback can execute.
    """
    if not _is_exact_public_integer(value):
        raise ValueError(f"{name} must be an integer")
    return value if type(value) is int else int(value)


def _validated_content_constraints(
    min_per_content: dict[str, int] | None,
    max_per_content: dict[str, int] | None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Validate content constraint maps and finite-domain count semantics."""

    def _one(raw: dict[str, int] | None) -> dict[str, int]:
        """Normalize one optional content-count map without coercion callbacks."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("content constraints must be mappings")
        out: dict[str, int] = {}
        for key, count in raw.items():
            trusted_key = _exact_public_string(
                key,
                "content constraint keys must be strings",
            )
            if not _is_exact_public_integer(count):
                raise ValueError("content constraint counts must be integers")
            value = count if type(count) is int else int(count)
            if value < 0:
                raise ValueError("content constraint counts must be non-negative")
            out[trusted_key] = value
        return out

    minimums = _one(min_per_content)
    maximums = _one(max_per_content)
    for label in minimums.keys() & maximums.keys():
        if minimums[label] > maximums[label]:
            raise ValueError("minimum content constraint cannot exceed maximum")
    return minimums, maximums


def _validated_exposure_counts(
    exposure_counts: dict[int, int] | None,
    n_items: int,
) -> dict[int, int]:
    """Validate exposure-map types, ranges, and bank membership before scoring."""
    if exposure_counts is None:
        return {}
    if not isinstance(exposure_counts, dict):
        raise ValueError("exposure_counts must be a mapping")
    out: dict[int, int] = {}
    for key, count in exposure_counts.items():
        if not _is_exact_public_integer(key) or not _is_exact_public_integer(count):
            raise ValueError("exposure_counts keys and values must be integers")
        item_index = key if type(key) is int else int(key)
        usage_count = count if type(count) is int else int(count)
        if not 0 <= item_index < n_items:
            raise ValueError("exposure_counts keys must identify existing items")
        if usage_count < 0:
            raise ValueError("exposure_counts values must be non-negative")
        out[item_index] = usage_count
    return out


def _validated_exclude(exclude: object, n_items: int) -> set[int]:
    """Return exact item exclusions without invoking caller conversion hooks.

    Only one-dimensional NumPy integer arrays and ordinary list/tuple containers
    of exact Python/NumPy integers are admitted. Boolean, fractional, object and
    arbitrary iterable inputs fail closed before psychometric scoring.
    """
    if exclude is None:
        return set()

    values: list[object]
    if isinstance(exclude, np.ndarray):
        if exclude.ndim != 1 or exclude.dtype.kind not in {"i", "u"}:
            raise ValueError("exclude must contain integer item indices")
        values = exclude.tolist()
    elif isinstance(exclude, (list, tuple)):
        values = list(exclude)
    else:
        raise ValueError("exclude must contain integer item indices")

    validated: set[int] = set()
    for value in values:
        if not _is_exact_public_integer(value):
            raise ValueError("exclude must contain integer item indices")
        item_index = value if type(value) is int else int(value)
        if not 0 <= item_index < n_items:
            raise ValueError("exclude item indices must identify existing items")
        validated.add(item_index)
    return validated


def assemble_to_target(
    bank: MLSIRMParams,
    factor_id: np.ndarray,
    target_thetas: np.ndarray,
    target_info: np.ndarray,
    length: int,
    *,
    model: str = "MLS2PLM",
    content: np.ndarray | None = None,
    min_per_content: dict[str, int] | None = None,
    max_per_content: dict[str, int] | None = None,
    exclude: np.ndarray | None = None,
    exposure_counts: dict[int, int] | None = None,
    exposure_max: int | None = None,
    seed: int = 0,
) -> AssembledForm:
    """Greedily assemble a length-``length`` form matching a target TIF.

    Selects items to drive ``sum_i I_i(theta_k)`` toward ``target_info[k]`` at
    every ``target_thetas`` point (van der Linden, 2005), honoring:

    * length (``length`` items),
    * content (``min_per_content`` / ``max_per_content`` keyed by ``content``
      labels),
    * exclusions (``exclude`` item indices), and
    * exposure: an item is ineligible when its prior usage count in
      ``exposure_counts`` reaches ``exposure_max``.

    Content labels and constraint-map keys/counts must already be the admitted
    types (strings / exact integers); arbitrary objects and conversion hooks are
    rejected before psychometric scoring rather than coerced through caller
    callbacks. Semantic count/range constraints, target-curve evidence, and
    exclusion indices are validated before item-information evaluation.
    Deterministic given ``seed``. Raises ``ValueError`` if no form satisfying the
    constraints can be assembled.
    """
    n_items = int(np.asarray(bank.b).shape[0])
    labels = _validated_content_labels(content, n_items)

    # Validate semantic controls before any item-information evaluation so hostile
    # conversion hooks and invalid finite-domain controls never reach scoring.
    length = _exact_public_integer(length, "length")
    if not (1 <= length <= n_items):
        raise ValueError("length must be between 1 and the number of items")

    min_counts, max_counts = _validated_content_constraints(min_per_content, max_per_content)
    if (min_counts or max_counts) and labels is None:
        raise ValueError("content labels are required for content constraints")

    excluded = _validated_exclude(exclude, n_items)
    exposure_counts = _validated_exposure_counts(exposure_counts, n_items)
    if exposure_max is not None:
        exposure_max = _exact_public_integer(exposure_max, "exposure_max")
        if exposure_max < 0:
            raise ValueError("exposure_max must be non-negative")
    seed = _exact_public_integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    n_dims = int(np.asarray(bank.theta).shape[1])
    thetas = _target_theta_rows(target_thetas, n_dims, n_items=n_items)
    target = _target_info_vector(target_info, thetas.shape[0])

    matrix = item_information_matrix(bank, factor_id, thetas, model=model)
    n_points, matrix_n_items = matrix.shape
    if matrix_n_items != n_items:
        raise ValueError("item-information matrix must match the number of items")
    if target.shape != (n_points,):
        raise ValueError("target_info must have one entry per target theta point")

    raw_info = matrix.sum(axis=0)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    counts: dict[str, int] = {}
    accum = np.zeros(n_points, dtype=np.float64)
    core = ata_core()

    for _ in range(length):
        # Eligibility mask independent of the current partial selection.
        eligible = np.ones(n_items, dtype=bool)
        for i in selected:
            eligible[i] = False
        for i in excluded:
            eligible[i] = False
        if exposure_max is not None:
            for i in range(n_items):
                if exposure_counts.get(i, 0) >= exposure_max:
                    eligible[i] = False
        if labels is not None:
            for i in np.nonzero(eligible)[0]:
                lbl = str(labels[i])
                if counts.get(lbl, 0) >= max_counts.get(lbl, length):
                    eligible[i] = False

        # Only consider items whose selection keeps the min-content requirements
        # satisfiable given the still-eligible pool after picking that item.
        candidates: list[int] = []
        for i in np.nonzero(eligible)[0]:
            remaining = eligible.copy()
            remaining[i] = False
            next_counts = dict(counts)
            if labels is not None:
                next_counts[str(labels[i])] = next_counts.get(str(labels[i]), 0) + 1
            if _content_feasible(labels, selected + [int(i)], next_counts, remaining, length, min_counts):
                candidates.append(int(i))
        if not candidates:
            raise ValueError("could not assemble a form that satisfies the constraints")

        cand = np.asarray(candidates, dtype=np.int64)
        gain = np.asarray(
            core.target_information_gains(matrix, cand, target, accum),
            dtype=np.float64,
        )
        best_gain = float(np.max(gain))
        tied = cand[gain >= best_gain - _TIE_EPS]
        best_raw = float(np.max(raw_info[tied]))
        tied = tied[raw_info[tied] >= best_raw - _TIE_EPS]
        chosen = int(tied[0]) if tied.size == 1 else int(rng.choice(tied))

        selected.append(chosen)
        accum = accum + matrix[:, chosen]
        if labels is not None:
            counts[str(labels[chosen])] = counts.get(str(labels[chosen]), 0) + 1

    for lbl, minimum in min_counts.items():
        if counts.get(lbl, 0) < minimum:
            raise ValueError(f"minimum content constraint not met: {lbl}")

    shortfall = float(np.sum(np.maximum(0.0, target - accum)))
    return AssembledForm(
        items=np.asarray(selected, dtype=np.int64),
        achieved_info=accum,
        target_info=target,
        shortfall=shortfall,
        content_counts=dict(counts),
    )

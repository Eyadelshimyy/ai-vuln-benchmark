"""Coverage for model-specification validation and resolution (models.py)."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import models
from fast_mlsirm.models import (
    ConfirmatoryModel,
    ExploratoryModel,
    _resolve_model,
    confirmatory,
    exploratory,
)


def test_exploratory_happy_path_and_n_dims():
    spec = exploratory(2)
    assert isinstance(spec, ExploratoryModel)
    assert spec.dimensions == 2
    assert spec.n_dims == 2


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_exploratory_rejects_non_positive_integer(bad):
    with pytest.raises(ValueError):
        exploratory(bad)


def test_exploratory_accepts_genuine_numpy_integer_scalar():
    spec = exploratory(np.int64(2))
    assert spec.dimensions == 2


def test_exploratory_rejects_integer_subclasses_before_callbacks():
    calls: list[str] = []

    class HostileInt(int):
        def __int__(self) -> int:
            calls.append("python-__int__")
            raise AssertionError("integer coercion callback executed")

        def __repr__(self) -> str:
            calls.append("python-__repr__")
            raise AssertionError("representation callback executed")

    class HostileNumpyInt(np.int64):
        def __int__(self) -> int:
            calls.append("numpy-__int__")
            raise AssertionError("integer coercion callback executed")

        def __repr__(self) -> str:
            calls.append("numpy-__repr__")
            raise AssertionError("representation callback executed")

    for value in (HostileInt(2), HostileNumpyInt(2)):
        with pytest.raises(ValueError, match="positive integer"):
            exploratory(value)

    assert calls == []


def test_exploratory_rejects_hostile_metaclass_hash_before_callback():
    calls: list[str] = []

    class HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("type-__hash__")
            raise AssertionError("type hash callback executed")

    class HostileInt(int, metaclass=HostileMeta):
        pass

    class HostileNumpyInt(np.int64, metaclass=HostileMeta):
        pass

    for value in (HostileInt(2), HostileNumpyInt(2)):
        with pytest.raises(ValueError, match="positive integer"):
            exploratory(value)

    assert calls == []


def test_confirmatory_happy_path_and_n_dims():
    pattern = np.array([[1, 0], [0, 1], [1, 0]])
    spec = confirmatory(pattern)
    assert isinstance(spec, ConfirmatoryModel)
    assert spec.n_dims == 2
    # loading pattern is stored read-only as int64
    assert spec.loading_pattern.dtype == np.int64
    with pytest.raises(ValueError):
        spec.loading_pattern[0, 0] = 1  # write-protected buffer


def test_confirmatory_rejects_non_2d_or_empty():
    with pytest.raises(ValueError):
        confirmatory(np.array([1, 0, 1]))  # 1-D
    with pytest.raises(ValueError):
        confirmatory(np.zeros((0, 2)))  # empty rows


def test_confirmatory_rejects_non_numeric_dtype():
    with pytest.raises(ValueError):
        confirmatory(np.array([["a", "b"], ["c", "d"]]))


def test_confirmatory_rejects_complex():
    with pytest.raises(ValueError):
        confirmatory(np.array([[1 + 0j, 0 + 0j]]))


def test_confirmatory_rejects_non_binary_values():
    with pytest.raises(ValueError):
        confirmatory(np.array([[1, 2], [0, 1]]))
    with pytest.raises(ValueError):
        confirmatory(np.array([[1.0, np.nan]]))


def test_resolve_model_rejects_bool():
    with pytest.raises(TypeError):
        _resolve_model(True, 4)


def test_resolve_model_int_one_is_single_factor():
    spec, pat = _resolve_model(1, 5)
    assert isinstance(spec, ExploratoryModel)
    assert pat.shape == (5, 1)
    assert np.all(pat == 1)


def test_resolve_model_accepts_genuine_numpy_integer_scalar():
    spec, pat = _resolve_model(np.uint64(1), 5)
    assert isinstance(spec, ExploratoryModel)
    assert spec.dimensions == 1
    assert pat.shape == (5, 1)


def test_resolve_model_rejects_integer_subclasses_before_callbacks():
    calls: list[str] = []

    class HostileInt(int):
        def __int__(self) -> int:
            calls.append("python-__int__")
            raise AssertionError("integer coercion callback executed")

        def __repr__(self) -> str:
            calls.append("python-__repr__")
            raise AssertionError("representation callback executed")

    class HostileNumpyInt(np.int64):
        def __int__(self) -> int:
            calls.append("numpy-__int__")
            raise AssertionError("integer coercion callback executed")

        def __repr__(self) -> str:
            calls.append("numpy-__repr__")
            raise AssertionError("representation callback executed")

    for value in (HostileInt(1), HostileNumpyInt(1)):
        with pytest.raises(TypeError, match="factor count"):
            _resolve_model(value, 5)

    assert calls == []


def test_resolve_model_rejects_hostile_metaclass_hash_before_callback():
    calls: list[str] = []

    class HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("type-__hash__")
            raise AssertionError("type hash callback executed")

    class HostileInt(int, metaclass=HostileMeta):
        pass

    class HostileNumpyInt(np.int64, metaclass=HostileMeta):
        pass

    for value in (HostileInt(1), HostileNumpyInt(1)):
        with pytest.raises(TypeError, match="factor count"):
            _resolve_model(value, 5)

    assert calls == []


def test_resolve_model_rejects_exploratory_subclass_before_attribute_callback():
    """Reject exploratory subclasses before reading caller-controlled fields."""
    calls: list[str] = []

    class HostileExploratory(ExploratoryModel):
        """Expose a hostile dimensions attribute on a caller subclass."""

        def __getattribute__(self, name: str):
            """Fail if model resolution reads caller-controlled dimensions."""
            if name == "dimensions":
                calls.append(name)
                raise AssertionError("model attribute callback executed")
            return super().__getattribute__(name)

    spec = object.__new__(HostileExploratory)
    object.__setattr__(spec, "dimensions", 1)

    with pytest.raises(TypeError, match="factor count"):
        _resolve_model(spec, 5)

    assert calls == []


def test_resolve_model_rejects_confirmatory_subclass_before_attribute_callback():
    """Reject confirmatory subclasses before reading loading-pattern metadata."""
    calls: list[str] = []

    class HostileConfirmatory(ConfirmatoryModel):
        """Expose a hostile loading-pattern attribute on a caller subclass."""

        def __getattribute__(self, name: str):
            """Fail if model resolution reads caller-controlled metadata."""
            if name == "loading_pattern":
                calls.append(name)
                raise AssertionError("model attribute callback executed")
            return super().__getattribute__(name)

    spec = object.__new__(HostileConfirmatory)
    object.__setattr__(spec, "loading_pattern", np.ones((5, 1), dtype=np.int64))

    with pytest.raises(TypeError, match="factor count"):
        _resolve_model(spec, 5)

    assert calls == []


def test_resolve_model_multidim_exploratory_not_implemented():
    with pytest.raises(NotImplementedError):
        _resolve_model(2, 6)
    with pytest.raises(NotImplementedError):
        _resolve_model(exploratory(3), 6)


def test_resolve_model_confirmatory_row_mismatch():
    spec = confirmatory(np.array([[1, 0], [0, 1]]))
    with pytest.raises(ValueError):
        _resolve_model(spec, 5)


def test_resolve_model_confirmatory_ok():
    spec = confirmatory(np.array([[1, 0], [0, 1], [1, 1]]))
    resolved, pat = _resolve_model(spec, 3)
    assert resolved is spec
    assert pat.shape == (3, 2)


def test_resolve_model_rejects_unknown_type():
    with pytest.raises(TypeError):
        _resolve_model("not-a-model", 3)


def test_models_module_exposes_helpers():
    assert models.exploratory is exploratory
    assert models.confirmatory is confirmatory

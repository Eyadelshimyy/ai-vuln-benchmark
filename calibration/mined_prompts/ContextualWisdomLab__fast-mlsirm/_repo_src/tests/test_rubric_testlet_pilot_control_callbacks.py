"""Callback-safety regressions for testlet pilot handoff controls."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_mlsirm.rubric import build_testlet_pilot_design

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_testlet_pilot_design.py"))
)
_testlet_records = _FIXTURES["_testlet_records"]


def _design():
    """Return the established deterministic generated-item testlet pilot."""
    return build_testlet_pilot_design(_testlet_records())


class _IndexProbe:
    """Arbitrary integer-like object whose protocol must never be invoked."""

    callbacks: list[str] = []

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 21

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "index_probe"


class _IntSubclassProbe(int):
    """Python integer subclass that must not inherit trusted-control status."""

    callbacks: list[str] = []

    def __new__(cls):
        return super().__new__(cls, 21)

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 21

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        return 21

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "int_subclass_probe"


class _NumpyIntSubclassProbe(np.int64):
    """NumPy integer subclass that must not inherit trusted-control status."""

    callbacks: list[str] = []

    def __new__(cls):
        return np.int64.__new__(cls, 21)

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 21

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        return 21

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "numpy_int_subclass_probe"


class _FloatSubclassProbe(float):
    """Python floating subclass whose conversion hook must remain unreachable."""

    callbacks: list[str] = []

    def __new__(cls):
        return super().__new__(cls, 0.5)

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        return 0.5

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "float_subclass_probe"


class _NumpyFloatSubclassProbe(np.float64):
    """NumPy floating subclass whose conversion hook must remain unreachable."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        return 0.5

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "numpy_float_subclass_probe"


class _StringSubclassProbe(str):
    """String subclass whose normalization hook must remain unreachable."""

    callbacks: list[str] = []

    def __new__(cls):
        return super().__new__(cls, "rasch")

    def casefold(self) -> str:
        type(self).callbacks.append("casefold")
        return "rasch"

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "string_subclass_probe"


class _HostileScalarMeta(type):
    """Metaclass whose hashing/equality must never run during type admission."""

    callbacks: list[str] = []

    def __hash__(cls) -> int:
        type(cls).callbacks.append("__hash__")
        return hash(np.int64)

    def __eq__(cls, other: object) -> bool:
        type(cls).callbacks.append("__eq__")
        return False


class _MetaclassIntProbe(int, metaclass=_HostileScalarMeta):
    """Integer subclass with caller-controlled type hashing/equality."""

    def __new__(cls):
        return int.__new__(cls, 21)

    def __repr__(self) -> str:
        _HostileScalarMeta.callbacks.append("__repr__")
        return "metaclass_int_probe"


@pytest.mark.parametrize(
    "probe_type",
    (_IndexProbe, _IntSubclassProbe, _NumpyIntSubclassProbe),
)
@pytest.mark.parametrize("field", ("max_iter", "q_gamma"))
def test_integer_controls_reject_callback_capable_values(probe_type, field):
    """Rejected integer controls execute no caller conversion or repr callback."""
    design = _design()
    probe_type.callbacks = []
    value = probe_type()

    with pytest.raises(ValueError, match=field):
        design.to_fit_testlet_kwargs(**{field: value})

    assert probe_type.callbacks == []


@pytest.mark.parametrize("probe_type", (_FloatSubclassProbe, _NumpyFloatSubclassProbe))
@pytest.mark.parametrize("field", ("tol", "init_sigma2"))
def test_float_controls_reject_callback_capable_values(probe_type, field):
    """Rejected floating controls execute no caller conversion or repr callback."""
    design = _design()
    probe_type.callbacks = []
    value = probe_type()

    with pytest.raises(ValueError, match=field):
        design.to_fit_testlet_kwargs(**{field: value})

    assert probe_type.callbacks == []


def test_model_rejects_string_subclass_before_normalization_callback():
    """A caller-defined string subclass cannot execute model normalization."""
    design = _design()
    _StringSubclassProbe.callbacks = []

    with pytest.raises(ValueError, match="model"):
        design.to_fit_testlet_kwargs(model=_StringSubclassProbe())

    assert _StringSubclassProbe.callbacks == []


@pytest.mark.parametrize("field", ("max_iter", "q_gamma", "tol", "init_sigma2"))
def test_type_admission_rejects_metaclass_callbacks(field):
    """Type admission never hashes or compares a caller-controlled metaclass."""
    design = _design()
    _HostileScalarMeta.callbacks = []

    with pytest.raises(ValueError, match=field):
        design.to_fit_testlet_kwargs(**{field: _MetaclassIntProbe()})

    assert _HostileScalarMeta.callbacks == []


def test_genuine_numpy_scalar_controls_remain_supported():
    """Exact NumPy scalar classes preserve the established handoff API."""
    kwargs = _design().to_fit_testlet_kwargs(
        max_iter=np.int64(250),
        q_gamma=np.uint32(31),
        tol=np.float32(0.0),
        init_sigma2=np.float64(0.5),
        estimate_sigma=np.bool_(False),
        require_convergence=np.bool_(True),
    )

    assert kwargs["max_iter"] == 250 and type(kwargs["max_iter"]) is int
    assert kwargs["q_gamma"] == 31 and type(kwargs["q_gamma"]) is int
    assert kwargs["tol"] == 0.0 and type(kwargs["tol"]) is float
    assert kwargs["init_sigma2"] == 0.5 and type(kwargs["init_sigma2"]) is float
    assert kwargs["estimate_sigma"] is False
    assert kwargs["require_convergence"] is True

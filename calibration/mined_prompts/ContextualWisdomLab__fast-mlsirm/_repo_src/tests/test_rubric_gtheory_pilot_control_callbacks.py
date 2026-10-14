"""Callback-safety regressions for G-theory pilot handoff controls."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pytest

from fast_mlsirm.rubric import build_gtheory_pi_pilot_design

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_pilot_observations.py"))
)
_pilot = _FIXTURES["_pilot"]
_observation = _FIXTURES["_observation"]


def _design():
    """Return a deterministic valid 2-person by 2-item pilot design."""
    item_alpha = _pilot("generated_item_alpha")
    item_beta = _pilot(
        "generated_item_beta",
        query_testlet_id="query_testlet_beta",
    )
    records = (
        _observation(item_alpha, respondent_id="respondent_alpha", category=0),
        _observation(item_beta, respondent_id="respondent_alpha", category=1),
        _observation(item_alpha, respondent_id="respondent_beta", category=2),
        _observation(item_beta, respondent_id="respondent_beta", category=1),
    )
    return build_gtheory_pi_pilot_design(records)


class _IndexProbe:
    """Arbitrary integer-like object whose protocol must never be invoked."""

    callbacks: list[str] = []

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 2

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "index_probe"


class _IntSubclassProbe(int):
    """Python integer subclass that must not inherit trusted-control status."""

    callbacks: list[str] = []

    def __new__(cls):
        return super().__new__(cls, 2)

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 2

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        return 2

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "int_subclass_probe"


class _NumpyIntSubclassProbe(np.int64):
    """NumPy integer subclass that must not inherit trusted-control status."""

    callbacks: list[str] = []

    def __new__(cls):
        return np.int64.__new__(cls, 2)

    def __int__(self) -> int:
        type(self).callbacks.append("__int__")
        return 2

    def __index__(self) -> int:
        type(self).callbacks.append("__index__")
        return 2

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "numpy_int_subclass_probe"


class _FloatSubclassProbe(float):
    """Python float subclass whose conversion hook must remain unreachable."""

    callbacks: list[str] = []

    def __new__(cls):
        return super().__new__(cls, 1.5)

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        return 1.5

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "float_subclass_probe"


class _NumpyFloatSubclassProbe(np.float64):
    """NumPy float subclass whose conversion hook must remain unreachable."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        return 1.5

    def __repr__(self) -> str:
        type(self).callbacks.append("__repr__")
        return "numpy_float_subclass_probe"


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
        return int.__new__(cls, 2)

    def __repr__(self) -> str:
        _HostileScalarMeta.callbacks.append("__repr__")
        return "metaclass_int_probe"


@pytest.mark.parametrize(
    "probe_type",
    (_IndexProbe, _IntSubclassProbe, _NumpyIntSubclassProbe),
)
def test_d_study_sizes_reject_callback_capable_integer_controls(probe_type):
    """Rejected D-study controls execute no caller conversion or repr callback."""
    design = _design()
    probe_type.callbacks = []
    value = probe_type()

    with pytest.raises(ValueError, match="must be an integer"):
        design.to_gtheory_pi_kwargs((value,))

    assert probe_type.callbacks == []


@pytest.mark.parametrize(
    "probe_type",
    (_IntSubclassProbe, _FloatSubclassProbe, _NumpyIntSubclassProbe, _NumpyFloatSubclassProbe),
)
def test_mastery_cut_rejects_callback_capable_numeric_subclasses(probe_type):
    """Rejected mastery cuts execute no caller conversion or repr callback."""
    design = _design()
    probe_type.callbacks = []
    value = probe_type()

    with pytest.raises(ValueError, match="cut must be a finite number"):
        design.to_phi_lambda_kwargs(value)

    assert probe_type.callbacks == []


@pytest.mark.parametrize("handoff", ("d_study", "mastery_cut"))
def test_type_admission_rejects_metaclass_callbacks(handoff):
    """Type admission never hashes or compares a caller-controlled metaclass."""
    design = _design()
    _HostileScalarMeta.callbacks = []
    value = _MetaclassIntProbe()

    with pytest.raises(ValueError):
        if handoff == "d_study":
            design.to_gtheory_pi_kwargs((value,))
        else:
            design.to_phi_lambda_kwargs(value)

    assert _HostileScalarMeta.callbacks == []


def test_genuine_numpy_scalar_controls_remain_supported():
    """Exact NumPy scalar classes preserve the established public handoff API."""
    design = _design()

    pi_kwargs = design.to_gtheory_pi_kwargs((np.int64(2), np.uint32(4)))
    assert pi_kwargs["n_i_prime"] == (2, 4)
    assert all(type(value) is int for value in pi_kwargs["n_i_prime"])

    assert design.to_phi_lambda_kwargs(np.float32(1.5))["cut"] == pytest.approx(1.5)
    assert design.to_phi_lambda_kwargs(np.int64(2))["cut"] == 2.0

"""Callback-safety regressions for automated-scoring validation policy controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.validation import ValidationPolicy


class _HostileText(str):
    """String subclass that records any caller-dispatchable normalization."""

    def __new__(cls, value: str, calls: list[str]) -> "_HostileText":
        """Attach the callback log to one hostile text scalar."""
        instance = super().__new__(cls, value)
        instance.calls = calls
        return instance

    def strip(self, *args: object, **kwargs: object) -> str:
        """Fail if validation dispatches caller-controlled text normalization."""
        self.calls.append("strip")
        raise AssertionError("caller string callback executed")


class _HostileFloat(float):
    """Float subclass that records caller-dispatchable numeric coercion."""

    def __new__(cls, value: float, calls: list[str]) -> "_HostileFloat":
        """Attach the callback log to one hostile floating scalar."""
        instance = super().__new__(cls, value)
        instance.calls = calls
        return instance

    def __float__(self) -> float:
        """Fail if validation dispatches caller-controlled float conversion."""
        self.calls.append("float")
        raise AssertionError("caller float callback executed")


class _HostileInt(int):
    """Integer subclass that records numeric coercion and comparisons."""

    def __new__(cls, value: int, calls: list[str]) -> "_HostileInt":
        """Attach the callback log to one hostile integer scalar."""
        instance = super().__new__(cls, value)
        instance.calls = calls
        return instance

    def __float__(self) -> float:
        """Fail if validation dispatches caller-controlled float conversion."""
        self.calls.append("float")
        raise AssertionError("caller integer float callback executed")

    def __lt__(self, other: object) -> bool:
        """Fail if validation dispatches a caller-controlled range comparison."""
        self.calls.append("lt")
        raise AssertionError("caller integer comparison executed")


@pytest.mark.parametrize("field", ["policy_id", "policy_version"])
def test_policy_identity_rejects_string_subclass_without_callbacks(field: str) -> None:
    """Policy identities reject caller string subclasses before ``strip`` dispatch."""
    calls: list[str] = []

    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        ValidationPolicy(**{field: _HostileText("trusted-looking", calls)})

    assert calls == []


@pytest.mark.parametrize(
    "field",
    [
        "qwk_min",
        "pearson_r_min",
        "degradation_max",
        "overall_smd_max",
        "subgroup_smd_max",
    ],
)
def test_policy_threshold_rejects_float_subclass_without_callbacks(field: str) -> None:
    """Every real threshold rejects caller float subclasses before coercion."""
    calls: list[str] = []

    with pytest.raises(ValueError, match=rf"{field} must be a real number in 0\.\.1"):
        ValidationPolicy(**{field: _HostileFloat(0.5, calls)})

    assert calls == []


def test_policy_threshold_rejects_integer_subclass_without_callbacks() -> None:
    """Numeric admission must not treat an ``int`` subclass as a trusted scalar."""
    calls: list[str] = []

    with pytest.raises(ValueError, match=r"qwk_min must be a real number in 0\.\.1"):
        ValidationPolicy(qwk_min=_HostileInt(1, calls))

    assert calls == []


def test_min_subgroup_n_rejects_integer_subclass_without_callbacks() -> None:
    """The subgroup-size control rejects integer subclasses before comparison."""
    calls: list[str] = []

    with pytest.raises(ValueError, match=r"min_subgroup_n must be an integer >= 2"):
        ValidationPolicy(min_subgroup_n=_HostileInt(2, calls))

    assert calls == []


def test_policy_builtin_controls_still_normalize_for_rust_marshalling() -> None:
    """Trusted built-in policy controls preserve the established Rust payload."""
    policy = ValidationPolicy(
        policy_id="research_diagnostic",
        policy_version="2.0",
        qwk_min=1,
        pearson_r_min=0.8,
        degradation_max=0,
        overall_smd_max=0.2,
        subgroup_smd_max=0.1,
        min_subgroup_n=3,
    )

    assert policy.policy_id == "research_diagnostic"
    assert policy.policy_version == "2.0"
    assert type(policy.qwk_min) is float
    assert type(policy.degradation_max) is float
    assert policy.rust_kwargs() == {
        "qwk_min": 1.0,
        "pearson_r_min": 0.8,
        "degradation_max": 0.0,
        "overall_smd_max": 0.2,
        "subgroup_smd_max": 0.1,
        "min_subgroup_n": 3,
    }


@pytest.mark.parametrize(
    "threshold",
    [
        np.float16(0.5),
        np.float32(0.5),
        np.float64(0.5),
        np.longdouble(0.5),
        np.int64(1),
    ],
)
def test_policy_trusted_numpy_thresholds_marshal_as_builtin_floats(threshold: object) -> None:
    """Supported concrete NumPy scalar identities normalize to Rust-ready floats."""
    policy = ValidationPolicy(qwk_min=threshold)
    rust_kwargs = policy.rust_kwargs()

    assert type(policy.qwk_min) is float
    assert type(rust_kwargs["qwk_min"]) is float
    assert rust_kwargs["qwk_min"] == float(threshold)

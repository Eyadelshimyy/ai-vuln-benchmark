"""Trust-boundary regressions for fixed-form test assembly."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.test_design import assemble_test_form


class _HostileInt(int):
    """Fail if an untrusted length invokes caller integer behavior."""

    callbacks = 0

    @classmethod
    def reset(cls) -> None:
        cls.callbacks = 0

    def _boom(self, *args, **kwargs):  # noqa: ANN002, ANN003
        type(self).callbacks += 1
        raise AssertionError("caller integer callback executed")

    __lt__ = _boom
    __le__ = _boom
    __gt__ = _boom
    __ge__ = _boom
    __int__ = _boom
    __index__ = _boom


class _ArrayTrap:
    """Fail if invalid semantic controls reach information materialization."""

    def __array__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("information materialization executed")


class _FloatTrap:
    """Fail if package admission converts an object information cell."""

    callbacks = 0

    def __float__(self) -> float:
        type(self).callbacks += 1
        raise AssertionError("caller float conversion executed")


class _StringTrap:
    """Fail if package admission stringifies one caller-owned label."""

    callbacks = 0

    def __str__(self) -> str:
        type(self).callbacks += 1
        raise AssertionError("caller string conversion executed")


def test_length_subclass_rejected_before_information_materialization() -> None:
    _HostileInt.reset()
    with pytest.raises(ValueError, match="length must be an integer"):
        assemble_test_form(_ArrayTrap(), _HostileInt(2))
    assert _HostileInt.callbacks == 0


def test_complex_information_rejected_before_real_narrowing() -> None:
    information = np.array([1.0 + 1.0j, 2.0, 3.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="information must be a real numeric array"):
        assemble_test_form(information, 2)


def test_object_information_rejected_without_element_conversion() -> None:
    _FloatTrap.callbacks = 0
    information = np.array([1.0, _FloatTrap(), 3.0], dtype=object)
    with pytest.raises(ValueError, match="information must be a real numeric array"):
        assemble_test_form(information, 2)
    assert _FloatTrap.callbacks == 0


def test_non_text_content_rejected_without_stringification() -> None:
    _StringTrap.callbacks = 0
    labels = np.array(["a", _StringTrap(), "b"], dtype=object)
    with pytest.raises(ValueError, match="content must contain string labels"):
        assemble_test_form(np.array([3.0, 2.0, 1.0]), 2, content=labels)
    assert _StringTrap.callbacks == 0


def test_mixed_builtin_content_sequence_rejected_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_mlsirm

    class _CoreTrap:
        @staticmethod
        def assemble_test_form_greedy(*args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("native assembly executed for invalid content labels")

    monkeypatch.setattr(fast_mlsirm, "_core", _CoreTrap())
    with pytest.raises(ValueError, match="content must contain string labels"):
        assemble_test_form(np.array([3.0, 2.0, 1.0]), 2, content=["a", 1, "b"])


def test_numpy_string_scalars_have_consistent_sequence_and_object_array_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fast_mlsirm

    observed_labels: list[list[str]] = []

    class _Core:
        @staticmethod
        def assemble_test_form_greedy(
            scores, length, labels, min_counts, max_counts, exclude_list  # noqa: ANN001
        ):
            observed_labels.append(labels)
            return [0, 1]

    monkeypatch.setattr(fast_mlsirm, "_core", _Core())
    expected = ["a", "b", "c"]
    for content in (
        [np.str_("a"), np.str_("b"), np.str_("c")],
        np.array([np.str_("a"), np.str_("b"), np.str_("c")], dtype=object),
    ):
        selected = assemble_test_form(np.array([3.0, 2.0, 1.0]), 2, content=content)
        np.testing.assert_array_equal(selected, [0, 1])
        assert observed_labels[-1] == expected
        assert all(type(label) is str for label in observed_labels[-1])


@pytest.mark.parametrize(
    "overflow_index",
    (np.uint64(2**63), np.uint64(np.iinfo(np.uint64).max)),
)
def test_unsigned_exclusion_overflow_rejected_before_signed_narrowing(
    overflow_index: np.uint64,
) -> None:
    """Signed-64 overflow stays rejected across NumPy promotion semantics."""
    exclude = np.array([overflow_index], dtype=np.uint64)
    with pytest.raises(ValueError, match="exclude must contain valid item indices"):
        assemble_test_form(np.array([3.0, 2.0, 1.0]), 2, exclude=exclude)


def test_constraint_maps_require_exact_text_keys_and_integer_counts() -> None:
    with pytest.raises(ValueError, match="content constraints must use string keys"):
        assemble_test_form(
            np.array([3.0, 2.0, 1.0]),
            2,
            content=np.array(["a", "a", "b"]),
            min_per_content={1: 1},
        )
    with pytest.raises(ValueError, match="content constraint counts must be non-negative integers"):
        assemble_test_form(
            np.array([3.0, 2.0, 1.0]),
            2,
            content=np.array(["a", "a", "b"]),
            max_per_content={"a": 1.5},
        )

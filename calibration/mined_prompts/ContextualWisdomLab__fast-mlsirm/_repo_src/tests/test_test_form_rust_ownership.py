"""Ownership contracts for fixed-form maximum-information assembly."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
from fast_mlsirm.test_design import assemble_test_form


def test_public_test_form_assembly_delegates_selection_to_rust(monkeypatch) -> None:
    """Ordering, exclusion, and content-feasibility decisions come from Rust."""
    information = np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float64)
    content = np.array(["A", "A", "B", "B"], dtype=object)
    exclude = np.array([3], dtype=np.int64)
    information_before = information.copy()
    content_before = content.copy()
    exclude_before = exclude.copy()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_assemble(*args: object, **kwargs: object) -> list[int]:
        calls.append((args, kwargs))
        # A deliberately different valid form from the current Python greedy
        # result proves public result ownership rather than mere helper reuse.
        return [2, 0]

    monkeypatch.setattr(core, "assemble_test_form_greedy", fake_assemble, raising=False)

    selected = assemble_test_form(
        information,
        length=2,
        content=content,
        min_per_content={"B": 1},
        max_per_content={"A": 1},
        exclude=exclude,
    )

    assert len(calls) == 1
    assert np.array_equal(selected, np.array([2, 0], dtype=np.int64))
    assert np.array_equal(information, information_before)
    assert np.array_equal(content, content_before)
    assert np.array_equal(exclude, exclude_before)


def test_invalid_content_constraint_error_does_not_reflect_caller_label() -> None:
    """Validation failures must not echo caller-controlled content labels."""
    sensitive_label = "customer_secret_content_category"
    information = np.array([1.0], dtype=np.float64)
    content = np.array([sensitive_label], dtype=object)

    with pytest.raises(ValueError) as exc_info:
        assemble_test_form(
            information,
            length=1,
            content=content,
            min_per_content={sensitive_label: -1},
        )

    assert sensitive_label not in str(exc_info.value)

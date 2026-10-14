"""Coverage-B: guard branches of validation.py and preprocessing.py."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import preprocessing, validation
from fast_mlsirm.validation import _validate_labels


# -- validation --------------------------------------------------------------


def test_validate_labels_rejects_empty_array():
    with pytest.raises(ValueError, match="must be non-empty"):
        _validate_labels(np.array([]), "judge")


def test_validate_judge_rejects_too_few_categories():
    with pytest.raises(ValueError, match="must be >= 2"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=1)


def test_validate_judge_accepts_human_human_baseline():
    judge = np.array([0, 1, 1, 0, 1, 0])
    human = np.array([0, 1, 0, 0, 1, 1])
    verdict = validation.validate_judge(
        judge, human, k=2, human_human=(np.array([0, 1, 0, 0, 1, 0]), human)
    )
    assert isinstance(verdict, validation.ValidationVerdict)
    assert isinstance(verdict.passed, bool)
    assert any(g["name"] == "degradation" for g in verdict.gates)


# -- preprocessing -----------------------------------------------------------


def test_irtree_expand_rejects_non_2d_mapping():
    responses = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="mapping must be nodes x categories"):
        preprocessing.irtree_expand(responses, np.array([0.0, 1.0]))


def test_irtree_expand_handles_all_missing_responses():
    responses = np.array([[np.nan], [np.nan]])
    mapping = np.array([[0.0, 1.0]])
    expanded, factor_id = preprocessing.irtree_expand(responses, mapping)
    assert expanded.shape == (2, 1)
    assert np.all(np.isnan(expanded))
    assert np.array_equal(factor_id, np.array([0]))


def test_irtree_expand_rejects_complex_responses_before_lossy_coercion():
    responses = np.array([[0.0 + 1.0j], [1.0 + 0.0j]])
    mapping = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="responses must be real-valued"):
        preprocessing.irtree_expand(responses, mapping)


def test_irtree_expand_rejects_complex_mapping_before_lossy_coercion():
    responses = np.array([[0.0], [1.0]])
    mapping = np.array([[0.0 + 1.0j, 1.0 + 0.0j]])
    with pytest.raises(ValueError, match="mapping must be real-valued"):
        preprocessing.irtree_expand(responses, mapping)


def test_irtree_expand_preserves_real_input_and_nan_semantics():
    responses = np.array([[0.0], [np.nan], [1.0]], dtype=np.float32)
    mapping = np.array([[0.0, 1.0]], dtype=np.float32)
    expanded, factor_id = preprocessing.irtree_expand(responses, mapping)
    assert np.array_equal(expanded[[0, 2], 0], np.array([0.0, 1.0]))
    assert np.isnan(expanded[1, 0])
    assert np.array_equal(factor_id, np.array([0]))


def test_irtree_expand_rejects_wrong_shape_node_dims():
    responses = np.array([[0.0], [1.0]])
    mapping = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="node_dims must have one entry"):
        preprocessing.irtree_expand(responses, mapping, node_dims=np.array([0, 1]))


def test_irtree_expand_rejects_complex_node_dims_before_lossy_coercion():
    responses = np.array([[0.0], [1.0]])
    mapping = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="node_dims must be real-valued"):
        preprocessing.irtree_expand(
            responses,
            mapping,
            node_dims=np.array([0.0 + 1.0j]),
        )


def test_irtree_expand_rejects_non_numeric_node_dims():
    responses = np.array([[0.0], [1.0]])
    mapping = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="node_dims must be finite non-negative integers"):
        preprocessing.irtree_expand(responses, mapping, node_dims=np.array(["x"]))

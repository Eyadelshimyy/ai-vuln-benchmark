"""Coverage for CAT item selection and test-form assembly (test_design.py)."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.test_design import (
    assemble_test_form,
    item_information,
    select_cat_item,
)
from fast_mlsirm.types import MLSIRMParams


def _params(n_persons=3, n_items=6, latent=2):
    rng = np.random.default_rng(0)
    return MLSIRMParams(
        theta=rng.standard_normal((n_persons, 1)),
        alpha=np.log(0.7 + 0.4 * rng.random(n_items)),
        b=np.linspace(-1.0, 1.0, n_items),
        xi=rng.standard_normal((n_persons, latent)),
        zeta=rng.standard_normal((n_items, latent)),
        tau=0.0,
    )


def test_item_information_happy_path():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    info = item_information(params, fid)
    assert info.shape == params.alpha.shape
    assert np.all(info >= 0)


def test_item_information_rejects_factor_id_shape():
    params = _params()
    with pytest.raises(ValueError):
        item_information(params, np.zeros(3, dtype=np.int64))


def test_item_information_explicit_theta_and_person_index():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    # explicit theta path (xi averaged), and person-index path
    a = item_information(params, fid, theta=np.array([0.0]))
    b = item_information(params, fid, person_index=1)
    c = item_information(params, fid, theta=np.array([0.0]), person_index=2)
    assert a.shape == fid.shape and b.shape == fid.shape and c.shape == fid.shape


def test_item_information_theta_dimension_mismatch():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    with pytest.raises(ValueError):
        item_information(params, fid, theta=np.array([0.0, 1.0]))


def test_select_cat_item_picks_max_information():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    info = item_information(params, fid)
    chosen = select_cat_item(params, fid)
    assert chosen == int(np.argmax(info))


def test_select_cat_item_skips_administered():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    info = item_information(params, fid)
    best = int(np.argmax(info))
    chosen = select_cat_item(params, fid, administered=np.array([best]))
    assert chosen != best


def test_select_cat_item_administered_out_of_range():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    with pytest.raises(ValueError):
        select_cat_item(params, fid, administered=np.array([999]))
    with pytest.raises(ValueError):
        select_cat_item(params, fid, administered=np.array([-1]))


def test_select_cat_item_no_candidates_remaining():
    params = _params()
    fid = np.zeros(params.alpha.shape[0], dtype=np.int64)
    all_items = np.arange(params.alpha.shape[0])
    with pytest.raises(ValueError):
        select_cat_item(params, fid, administered=all_items)


def test_assemble_form_greedy_no_constraints():
    info = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    form = assemble_test_form(info, length=3)
    assert form.tolist() == [0, 1, 2]


def test_assemble_form_rejects_bad_shapes():
    with pytest.raises(ValueError):
        assemble_test_form(np.zeros((2, 2)), length=1)
    with pytest.raises(ValueError):
        assemble_test_form(np.array([1.0, 2.0]), length=0)
    with pytest.raises(ValueError):
        assemble_test_form(np.array([1.0, 2.0]), length=3)


def test_assemble_form_content_required_and_length_match():
    info = np.array([5.0, 4.0, 3.0])
    with pytest.raises(ValueError):
        assemble_test_form(info, length=2, min_per_content={"a": 1})
    with pytest.raises(ValueError):
        assemble_test_form(
            info, length=2, content=np.array(["a", "b"]), min_per_content={"a": 1}
        )


def test_assemble_form_excludes_items():
    info = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    form = assemble_test_form(info, length=2, exclude=np.array([0]))
    assert 0 not in form.tolist()
    assert form.tolist() == [1, 2]


def test_assemble_form_with_min_and_max_content():
    info = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    content = np.array(["a", "a", "b", "b", "c", "c"])
    form = assemble_test_form(
        info,
        length=3,
        content=content,
        min_per_content={"c": 1},
        max_per_content={"a": 1},
    )
    labels = content[form]
    assert (labels == "a").sum() <= 1
    assert (labels == "c").sum() >= 1


def test_assemble_form_infeasible_raises():
    info = np.array([5.0, 4.0])
    content = np.array(["a", "a"])
    with pytest.raises(ValueError):
        assemble_test_form(
            info, length=2, content=content, min_per_content={"b": 1}
        )


def test_assemble_form_min_equals_max_caps_availability_scan():
    # min == max on the same label forces the feasibility look-ahead to stop
    # counting available items once the cap is reached (the break branch).
    info = np.array([6.0, 5.0, 4.0, 3.0])
    content = np.array(["a", "a", "a", "a"])
    form = assemble_test_form(
        info,
        length=2,
        content=content,
        min_per_content={"a": 2},
        max_per_content={"a": 2},
    )
    assert (content[form] == "a").sum() == 2

"""Callback-safety regressions for untrusted in-memory serving bundles."""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest

from fast_mlsirm import serving


class _HostileDict(dict):
    calls = 0

    def _trip(self) -> NoReturn:
        type(self).calls += 1
        raise AssertionError("caller dict callback executed")

    def get(self, *args, **kwargs) -> NoReturn:
        return self._trip()

    def __getitem__(self, key) -> NoReturn:
        return self._trip()

    def __contains__(self, key) -> bool:
        return self._trip()

    def __iter__(self) -> NoReturn:
        return self._trip()

    def __len__(self) -> int:
        return self._trip()


class _HostileList(list):
    calls = 0

    def _trip(self) -> NoReturn:
        type(self).calls += 1
        raise AssertionError("caller list callback executed")

    def __len__(self) -> int:
        return self._trip()

    def __iter__(self) -> NoReturn:
        return self._trip()

    def __getitem__(self, key) -> NoReturn:
        return self._trip()


class _HostileText(str):
    calls = 0

    def _trip(self) -> NoReturn:
        type(self).calls += 1
        raise AssertionError("caller text callback executed")

    def __hash__(self) -> int:
        return self._trip()

    def __eq__(self, other) -> bool:
        return self._trip()

    def __ne__(self, other) -> bool:
        return self._trip()


class _ArmedHostileKey(str):
    calls = 0
    armed = False

    def __hash__(self) -> int:
        if type(self).armed:
            type(self).calls += 1
            raise AssertionError("caller key hash callback executed")
        return super().__hash__()

    def __eq__(self, other) -> bool:
        if type(self).armed:
            type(self).calls += 1
            raise AssertionError("caller key equality callback executed")
        return super().__eq__(other)


class _HostileInt(int):
    calls = 0

    def _trip(self) -> NoReturn:
        type(self).calls += 1
        raise AssertionError("caller integer callback executed")

    def __eq__(self, other) -> bool:
        return self._trip()

    def __ne__(self, other) -> bool:
        return self._trip()

    def __lt__(self, other) -> bool:
        return self._trip()

    def __le__(self, other) -> bool:
        return self._trip()

    def __gt__(self, other) -> bool:
        return self._trip()

    def __ge__(self, other) -> bool:
        return self._trip()

    def __hash__(self) -> int:
        return self._trip()


def _bundle() -> dict:
    """Return one valid minimal MIRT serving bundle using inert built-ins only."""
    return {
        "schema_version": serving.SCHEMA_VERSION,
        "model": "MIRT",
        "n_items": 1,
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": -30.0,
        "population": None,
        "items": [
            {
                "code": "q0",
                "factor_id": 0,
                "alpha": 0.0,
                "b": 0.0,
                "zeta": [0.0],
            }
        ],
    }


def _fail_if_core_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make native discovery an explicit failure for invalid bundle controls."""
    monkeypatch.setattr(
        serving,
        "_core_module",
        lambda: pytest.fail("invalid serving bundle reached compiled-core discovery"),
    )


def test_top_level_dict_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileDict.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _HostileDict(_bundle())

    with pytest.raises(ValueError, match="JSON object"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileDict.calls == 0


def test_serving_prior_rejects_top_level_dict_subclass_without_callbacks():
    _HostileDict.calls = 0
    bundle = _HostileDict(_bundle())

    with pytest.raises(ValueError, match="JSON object"):
        serving.serving_prior(bundle)

    assert _HostileDict.calls == 0


def test_top_level_key_subclass_is_rejected_without_callbacks(monkeypatch):
    _ArmedHostileKey.calls = 0
    _ArmedHostileKey.armed = False
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    value = bundle.pop("schema_version")
    hostile_key = _ArmedHostileKey("schema_version")
    bundle[hostile_key] = value

    _ArmedHostileKey.calls = 0
    _ArmedHostileKey.armed = True
    try:
        with pytest.raises(ValueError, match="keys must be strings"):
            serving.score_respondents(bundle, np.array([[1.0]]))
    finally:
        _ArmedHostileKey.armed = False

    assert _ArmedHostileKey.calls == 0


def test_schema_integer_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileInt.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["schema_version"] = _HostileInt(serving.SCHEMA_VERSION)

    with pytest.raises(ValueError, match="schema_version"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileInt.calls == 0


def test_model_string_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileText.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["model"] = _HostileText("MIRT")

    with pytest.raises(ValueError, match="model must be one of"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileText.calls == 0


def test_population_kind_string_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileText.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["population"] = {
        "kind": _HostileText("multilevel"),
        "sigma_u": 1.0,
    }

    with pytest.raises(ValueError, match="population kind must be a string"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileText.calls == 0


def test_dim_names_list_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileList.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["dim_names"] = _HostileList(["ability"])

    with pytest.raises(ValueError, match="dim_names"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileList.calls == 0


def test_dim_name_string_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileText.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["dim_names"] = [_HostileText("ability")]

    with pytest.raises(ValueError, match="dim_names"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileText.calls == 0


def test_dim_name_count_must_match_n_dims_before_native(monkeypatch):
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["dim_names"] = ["ability", "extra"]

    with pytest.raises(ValueError, match="dim_names"):
        serving.score_respondents(bundle, np.array([[1.0]]))


def test_items_list_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileList.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["items"] = _HostileList(bundle["items"])

    with pytest.raises(ValueError, match="items must be a list"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileList.calls == 0


def test_item_dict_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileDict.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["items"][0] = _HostileDict(bundle["items"][0])

    with pytest.raises(ValueError, match="item 0 must be an object"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileDict.calls == 0


def test_item_code_string_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileText.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["items"][0]["code"] = _HostileText("q0")

    with pytest.raises(ValueError, match="unique string code"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileText.calls == 0


def test_quadrature_integer_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileInt.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["quadrature"]["q_theta"] = _HostileInt(7)

    with pytest.raises(ValueError, match="quadrature q_theta"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileInt.calls == 0


def test_eapsum_table_list_subclass_is_rejected_without_callbacks(monkeypatch):
    _HostileList.calls = 0
    _fail_if_core_discovered(monkeypatch)
    bundle = _bundle()
    bundle["eapsum_tables"] = _HostileList([])

    with pytest.raises(ValueError, match="eapsum_tables"):
        serving.score_respondents(bundle, np.array([[1.0]]))

    assert _HostileList.calls == 0

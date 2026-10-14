"""Security regressions for plausible-values public control marshalling.

These tests keep psychometric arithmetic in Rust while proving the Python serving
boundary rejects hostile coercion objects before native-core discovery and
normalizes trusted scalar controls to the exact types expected by PyO3.
"""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.serving as serving


def _bundle() -> dict[str, object]:
    """Return the smallest structurally valid one-factor serving bundle."""
    return {
        "schema_version": 1,
        "model": "MIRT",
        "n_items": 2,
        "n_dims": 1,
        "latent_dim": 1,
        "quadrature": {"q_theta": 7, "q_xi": 7},
        "eps_distance": 1e-8,
        "tau": 0.0,
        "population": None,
        "eapsum_tables": [
            {
                "dim": 0,
                "n_items_dim": 2,
                "score_prob": [0.3, 0.4, 0.3],
                "eap": [-1.0, 0.0, 1.0],
                "sd": [0.5, 0.4, 0.5],
            }
        ],
        "items": [
            {"code": "i0", "factor_id": 0, "alpha": 0.2, "b": -0.3, "zeta": [0.0]},
            {"code": "i1", "factor_id": 0, "alpha": 0.1, "b": 0.4, "zeta": [0.0]},
        ],
    }


def _assert_rejected_before_core(
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected: str,
    **controls: object,
) -> None:
    """Assert one invalid control fails without discovering the native core."""
    core_calls: list[bool] = []

    def _unexpected_core() -> None:
        core_calls.append(True)
        return None

    monkeypatch.setattr(serving, "_core_module", _unexpected_core)
    with pytest.raises(ValueError, match=expected):
        serving.plausible_values(_bundle(), {"i0": 1, "i1": 0}, **controls)
    assert core_calls == []


def test_plausible_values_rejects_hostile_n_draws_before_core(monkeypatch):
    """An ``int`` subclass must not run caller code through ``int(value)``."""
    coercions: list[bool] = []

    class HostileInt(int):
        def __int__(self) -> int:
            coercions.append(True)
            return 5

    _assert_rejected_before_core(
        monkeypatch,
        expected="n_draws must be an integer",
        n_draws=HostileInt(5),
    )
    assert coercions == []


def test_plausible_values_rejects_hostile_seed_before_core(monkeypatch):
    """Seed coercion must reject ``int`` subclasses without invoking them."""
    coercions: list[bool] = []

    class HostileSeed(int):
        def __int__(self) -> int:
            coercions.append(True)
            return 7

    _assert_rejected_before_core(
        monkeypatch,
        expected="seed must be an integer",
        seed=HostileSeed(7),
    )
    assert coercions == []


def test_plausible_values_rejects_scalar_type_metaclass_callbacks(monkeypatch):
    """Trusted-type admission must not hash or compare a caller metaclass."""
    callbacks: list[str] = []

    class HostileMeta(type):
        def __hash__(cls) -> int:
            callbacks.append("hash")
            return type.__hash__(cls)

        def __eq__(cls, other: object) -> bool:
            callbacks.append("eq")
            return type.__eq__(cls, other)

    class HostileSeed(int, metaclass=HostileMeta):
        pass

    hostile_seed = HostileSeed(7)
    callbacks.clear()
    _assert_rejected_before_core(
        monkeypatch,
        expected="seed must be an integer",
        seed=hostile_seed,
    )
    assert callbacks == []


def test_plausible_values_rejects_hostile_device_before_core(monkeypatch):
    """Device marshalling must not execute a caller-controlled ``__str__``."""
    coercions: list[bool] = []

    class HostileDevice(str):
        def __str__(self) -> str:
            coercions.append(True)
            return "cpu"

    _assert_rejected_before_core(
        monkeypatch,
        expected="device must be a string",
        device=HostileDevice("cpu"),
    )
    assert coercions == []


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_plausible_values_bounds_seed_to_rust_u64(monkeypatch, seed):
    """Python must reject seeds that cannot cross the PyO3 ``u64`` boundary."""
    _assert_rejected_before_core(
        monkeypatch,
        expected=r"seed must be between 0 and 18446744073709551615",
        seed=seed,
    )


def test_plausible_values_rejects_unknown_device_before_core(monkeypatch):
    """Only the Rust-supported device vocabulary may cross the boundary."""
    _assert_rejected_before_core(
        monkeypatch,
        expected=r"device must be one of \['cpu', 'gpu', 'auto'\]",
        device="cuda",
    )


@pytest.mark.parametrize(
    ("controls", "expected"),
    [
        ({"n_draws": True}, "n_draws must be an integer"),
        ({"n_draws": False}, "n_draws must be an integer"),
        ({"n_draws": np.bool_(True)}, "n_draws must be an integer"),
        ({"seed": True}, "seed must be an integer"),
        ({"seed": np.bool_(False)}, "seed must be an integer"),
        ({"n_draws": 1.5}, "n_draws must be an integer"),
        ({"n_draws": np.float64(2.0)}, "n_draws must be an integer"),
        ({"seed": 1.5}, "seed must be an integer"),
        ({"n_draws": 0}, r"n_draws must be between 1 and 100000"),
        ({"n_draws": serving.MAX_DRAWS + 1}, r"n_draws must be between 1 and 100000"),
        ({"device": True}, "device must be a string"),
    ],
)
def test_plausible_values_rejects_bool_float_and_draw_bounds_before_core(
    monkeypatch, controls, expected
):
    """Booleans, floats, and out-of-range draws fail before native discovery."""
    _assert_rejected_before_core(monkeypatch, expected=expected, **controls)


def test_plausible_values_rejects_index_provider_before_core(monkeypatch):
    """An ``__index__`` provider must not be admitted as an integer control."""
    coercions: list[bool] = []

    class HostileIndex:
        def __index__(self) -> int:
            coercions.append(True)
            return 5

    _assert_rejected_before_core(
        monkeypatch,
        expected="n_draws must be an integer",
        n_draws=HostileIndex(),
    )
    assert coercions == []


def test_plausible_values_valid_input_discovers_core_only_at_dispatch(monkeypatch):
    """A valid request discovers the compiled core exactly once, at dispatch."""
    calls = 0

    def missing_core():
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(serving, "_core_module", missing_core)
    with pytest.raises(
        RuntimeError, match="plausible_values requires the compiled Rust core"
    ):
        serving.plausible_values(_bundle(), {"i0": 1, "i1": 0})
    assert calls == 1


def test_plausible_values_normalizes_trusted_numpy_integer_controls(monkeypatch):
    """Exact NumPy integer scalars marshal to exact built-in integers."""
    captured: dict[str, object] = {}

    class StubCore:
        @staticmethod
        def plausible_values(*args, **kwargs):
            captured.update(kwargs)
            assert type(kwargs["n_draws"]) is int
            assert type(kwargs["seed"]) is int
            assert type(kwargs["device"]) is str
            n_people = int(args[2])
            return [0.0] * (n_people * kwargs["n_draws"])

    monkeypatch.setattr(serving, "_core_module", lambda: StubCore())
    result = serving.plausible_values(
        _bundle(),
        {"i0": 1, "i1": 0},
        n_draws=np.int64(2),
        seed=np.uint64(3),
        device="cpu",
    )

    assert result.shape == (1, 2, 1)
    assert captured["n_draws"] == 2
    assert captured["seed"] == 3
    assert captured["device"] == "cpu"

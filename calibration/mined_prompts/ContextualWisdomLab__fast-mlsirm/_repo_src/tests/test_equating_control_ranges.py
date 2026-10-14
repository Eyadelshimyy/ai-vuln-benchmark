"""Range and alias regressions for inert observed-score equating controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.equating as E
import fast_mlsirm.fitstats as fitstats

_TOTAL = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_ANCHOR = np.array([0.0, 1.0, 2.0], dtype=np.float64)
_COUNTS = np.array([1.0, 2.0, 1.0], dtype=np.float64)


def _invoke(case: str, value: object) -> object:
    """Invoke one public adapter with a selected exact primitive control."""
    if case.startswith("equate_neat."):
        kwargs: dict[str, object] = {
            "method": "chained",
            "k_x": 2,
            "k_y": 2,
            "k_v": 2,
            "w1": 0.5,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equate_neat(_TOTAL, _ANCHOR, _TOTAL, _ANCHOR, **kwargs)
    if case.startswith("equate_neat_linear."):
        kwargs = {
            "method": "tucker",
            "anchor_kind": "internal",
            "k_x": 2,
            "k_y": 2,
            "w1": 0.5,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equate_neat_linear(_TOTAL, _ANCHOR, _TOTAL, _ANCHOR, **kwargs)
    if case == "loglinear.degree":
        return E.loglinear_smooth(_COUNTS, degree=value)
    if case.startswith("kernel."):
        kwargs = {
            "continuization": "gaussian",
            "k_x": 2,
            "k_y": 2,
            "smooth_x": 1,
            "smooth_y": 1,
            "bandwidth_x": 0.5,
            "bandwidth_y": 0.5,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equate_observed_scores_kernel(_TOTAL, _TOTAL, **kwargs)
    if case.startswith("see."):
        kwargs = {
            "method": "mean",
            "route": "bootstrap",
            "k_x": 2,
            "k_y": 2,
            "n_boot": 2,
            "ci_level": 0.95,
            "seed": 0,
        }
        kwargs[case.rsplit(".", 1)[1]] = value
        return E.equating_standard_errors(_TOTAL, _TOTAL, **kwargs)
    raise AssertionError(f"unhandled test case: {case}")


_INVALID_CASES = (
    ("equate_neat.method", "bogus"),
    ("equate_neat_linear.method", "bogus"),
    ("equate_neat_linear.anchor_kind", "bogus"),
    ("kernel.continuization", "bogus"),
    ("see.method", "bogus"),
    ("see.route", "bogus"),
    ("equate_neat.k_x", 0),
    ("loglinear.degree", 0),
    ("kernel.smooth_x", 0),
    ("see.n_boot", 0),
    ("see.seed", -1),
    ("equate_neat.w1", -0.1),
    ("equate_neat.w1", 1.1),
    ("equate_neat.w1", float("nan")),
    ("equate_neat.w1", float("inf")),
    ("equate_neat.w1", 10**10000),
    ("kernel.bandwidth_x", 0.0),
    ("see.ci_level", 0.0),
    ("see.ci_level", 1.0),
)


@pytest.mark.parametrize(
    ("case", "value"),
    _INVALID_CASES,
    ids=[f"{case}-{index}" for index, (case, _) in enumerate(_INVALID_CASES)],
)
def test_exact_invalid_controls_fail_before_rust_discovery(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    value: object,
) -> None:
    """Exact but unsupported controls fail locally with package-owned errors."""
    core_calls: list[str] = []

    def forbidden_core():
        core_calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)

    with pytest.raises(ValueError, match=case.rsplit(".", 1)[1]):
        _invoke(case, value)

    assert core_calls == []


class _ReachedCore(RuntimeError):
    """Mark that validated controls reached the intended Rust-shaped method."""


def test_rust_aliases_and_optional_controls_remain_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented aliases survive while optional controls stay exact ``None``."""
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class Core:
        """Capture each validated boundary and stop before numerical work."""

        def equate_neat(self, *args, **kwargs):
            calls.append(("neat", args, kwargs))
            raise _ReachedCore

        def equate_neat_linear(self, *args, **kwargs):
            calls.append(("linear", args, kwargs))
            raise _ReachedCore

        def equate_observed_scores_ext(self, *args, **kwargs):
            calls.append(("kernel", args, kwargs))
            raise _ReachedCore

        def bootstrap_see(self, *args, **kwargs):
            calls.append(("see", args, kwargs))
            raise _ReachedCore

    monkeypatch.setattr(fitstats, "_core_module", lambda: Core())
    routes: tuple[Callable[[], object], ...] = (
        lambda: E.equate_neat(
            _TOTAL,
            _ANCHOR,
            _TOTAL,
            _ANCHOR,
            method="fe",
            w1=1,
        ),
        lambda: E.equate_neat_linear(
            _TOTAL,
            _ANCHOR,
            _TOTAL,
            _ANCHOR,
            method="l",
            anchor_kind="ext",
            w1=0,
        ),
        lambda: E.equate_observed_scores_kernel(
            _TOTAL,
            _TOTAL,
            continuization="normal",
        ),
        lambda: E.equating_standard_errors(
            _TOTAL,
            _TOTAL,
            method="m",
            route="bootstrap",
            n_boot=1,
            ci_level=0.5,
            seed=0,
        ),
    )

    for route in routes:
        with pytest.raises(_ReachedCore):
            route()

    assert [name for name, _, _ in calls] == ["neat", "linear", "kernel", "see"]
    assert calls[0][2] == {"method": "fe", "w1": 1.0}
    assert calls[1][2] == {"method": "l", "anchor_kind": "ext", "w1": 0.0}
    assert calls[2][2] == {
        "continuization": "normal",
        "smooth_degree_x": None,
        "smooth_degree_y": None,
        "bandwidth_x": None,
        "bandwidth_y": None,
    }
    assert calls[3][2] == {
        "method": "m",
        "n_boot": 1,
        "ci_level": 0.5,
        "seed": 0,
    }

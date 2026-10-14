#!/usr/bin/env python3
"""Coverage-guided fuzz harness for the fast-mlsirm config validators.

``MLS2PLMConfig`` and ``FitConfig`` are the request/DTO validators of the
package: every CLI invocation and every public API entry point funnels
user-supplied numeric parameters through construction-time ``.validate()``.
The contract is that construction and ``validate()`` are *total* on arbitrary
field values -- they must either return normally or raise ``ValueError`` with a
message. Any other exception (``TypeError`` from an unexpected type,
``OverflowError``, ``ZeroDivisionError`` in the equicorrelation bound, ...) is
a bug.

This harness draws arbitrary field values from the fuzzer, builds both configs,
and asserts the validator only ever rejects with ``ValueError``. When a config
validates cleanly, derived invariants are checked (e.g. ``n_items`` stays a
non-negative product).

Run::

    python fuzz/atheris/fuzz_config.py -atheris_runs=50000 -max_total_time=90
"""

from __future__ import annotations

import atheris

with atheris.instrument_imports():
    import sys

    from fast_mlsirm.config import FitConfig, MLS2PLMConfig, PenaltyConfig


def _maybe_nan_float(fdp: "atheris.FuzzedDataProvider") -> float:
    # Deliberately include the pathological floats parsers tend to mishandle.
    choice = fdp.ConsumeIntInRange(0, 4)
    if choice == 0:
        return float("nan")
    if choice == 1:
        return float("inf")
    if choice == 2:
        return float("-inf")
    return fdp.ConsumeRegularFloat()


def _test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    try:
        sim = MLS2PLMConfig(
            n_persons=fdp.ConsumeIntInRange(-8, 4096),
            n_dims=fdp.ConsumeIntInRange(-8, 512),
            items_per_dim=fdp.ConsumeIntInRange(-8, 512),
            latent_dim=fdp.ConsumeIntInRange(-8, 512),
            phi=_maybe_nan_float(fdp),
            gamma=_maybe_nan_float(fdp),
            seed=fdp.ConsumeInt(8),
            dtype=fdp.PickValueInList(["float64", "float32", "int8", "", "FLOAT64"]),
        )
        sim.validate()
    except ValueError:
        pass
    else:
        # n_items is a pure product of two validated positive ints.
        assert sim.n_items >= 1, f"validated config produced n_items={sim.n_items}"

    try:
        fit = FitConfig(
            model=fdp.PickValueInList(["MLS2PLM", "mls2plm", "MIRT", "bogus", ""]),
            latent_dim=fdp.ConsumeIntInRange(-8, 512),
            optimizer=fdp.PickValueInList(["adam", "lbfgs", "adam_lbfgs", "sgd", ""]),
            max_iter=fdp.ConsumeIntInRange(-8, 100000),
            n_restarts=fdp.ConsumeIntInRange(-8, 512),
            learning_rate=_maybe_nan_float(fdp),
            seed=fdp.ConsumeInt(8),
            eps_distance=_maybe_nan_float(fdp),
            init_gamma=_maybe_nan_float(fdp),
            backend=fdp.PickValueInList(["numpy", "rust", "auto", "", "NUMPY"]),
            penalty=PenaltyConfig(),
        )
        fit.validate()
    except ValueError:
        pass
    else:
        assert fit.normalized_model() in {"MIRT", "MLS2PLM", "MLSRM", "ULS2PLM", "ULSRM"}


def main() -> None:
    atheris.Setup(sys.argv, _test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

"""Replay-integrity regressions for cross-engine run provenance."""

from __future__ import annotations

import pytest

from fast_mlsirm.cross_engine_conformance import ConformanceRunProvenance

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SOURCE_COMMIT = "0" * 40


class _HostileSeedTuple(tuple[int, ...]):
    """Tuple subclass that fails if manifest replay iterates caller state."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Record and reject unsafe iteration of rebound provenance state."""
        type(self).callbacks += 1
        raise AssertionError("hostile seed iteration executed")


def _provenance() -> ConformanceRunProvenance:
    """Return one valid source-free conformance run record."""
    return ConformanceRunProvenance(
        harness_commit=_SOURCE_COMMIT,
        environment_sha256=_SHA_C,
        environment_kind="environment_lock",
        operating_system="linux",
        architecture="x86_64",
        rng_algorithm="pcg64_dxsm",
        rng_seeds=(17, 23),
        mapping_schema_version="1.0.0",
        mapping_sha256=_SHA_B,
        model_configuration_sha256=_SHA_A,
        convergence_controls_sha256=_SHA_C,
        tolerance_sha256=_SHA_A,
        tolerance_rationale="fixed-parameter double-precision comparison",
        raw_output_sha256=None,
        normalized_output_sha256=None,
        license_classification="synthetic_or_open",
        redistribution_status="metadata_only",
    )


def test_run_provenance_manifest_revalidates_rebound_seed_container() -> None:
    """Direct manifest replay must reject post-init seed rebinding without callbacks."""
    _HostileSeedTuple.callbacks = 0
    provenance = _provenance()
    object.__setattr__(provenance, "rng_seeds", _HostileSeedTuple((17, 23)))

    with pytest.raises(ValueError, match="rng_seeds must be a list or tuple"):
        provenance.to_manifest()

    assert _HostileSeedTuple.callbacks == 0

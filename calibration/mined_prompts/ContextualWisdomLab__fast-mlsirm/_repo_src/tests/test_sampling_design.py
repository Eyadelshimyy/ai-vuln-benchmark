"""Tests for the Rust-owned finite-population sampling-design contract."""

from __future__ import annotations

from dataclasses import replace

import fast_mlsirm.sampling_design as sampling_design_module
import pytest
from fast_mlsirm import (
    ACHIEVED_PROPORTION_SCHEMA_VERSION,
    SAMPLING_DESIGN_SCHEMA_VERSION,
    SamplingStratum,
    finite_population_achieved_proportion,
    finite_population_proportion_design,
)


def test_finite_population_design_matches_rust_reference() -> None:
    """The public API returns the versioned Rust result without Python arithmetic."""
    design = finite_population_proportion_design(
        1_000,
        0.95,
        0.05,
        [SamplingStratum(1_000, 0.5)],
        allocation_method="proportional",
    )

    assert design.schema_version == SAMPLING_DESIGN_SCHEMA_VERSION
    assert design.source_identity == "fast-mlsirm.mlsirm-core.sampling-design"
    assert design.algorithm_version == "1.1.0"
    assert design.strata == (SamplingStratum(1_000, 0.5),)
    assert design.sample_size == 278
    assert design.stratum_sample_sizes == (278,)
    assert design.stratum_inclusion_probability_ratios == ((278, 1_000),)
    assert design.finite_population_correction == pytest.approx(
        (722.0 / 999.0) ** 0.5
    )
    assert all(
        len(identity) == 64
        for identity in (
            design.source_sha256,
            design.input_sha256,
            design.output_sha256,
            design.artifact_sha256,
        )
    )


def test_sampling_artifact_replay_has_stable_content_identity() -> None:
    """Exact canonical inputs reproduce every Rust-owned artifact identity."""
    first = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(60, 0.1), SamplingStratum(40, 0.5)],
        allocation_method="neyman",
    )
    replay = finite_population_proportion_design(
        first.population_size,
        first.confidence_level,
        first.margin_of_error,
        list(first.strata),
        allocation_method=first.allocation_method,  # type: ignore[arg-type]
    )

    assert replay == first
    assert first.input_sha256 == (
        "b42d91dffeb48a50e3a133d7492f4ba90b42c806a2238b7c292f5ab0c52c9290"
    )
    assert first.output_sha256 == (
        "4ee75c38735aeb19907f4d0cc2bb7a45add698238c132529a8c85226f528dbb4"
    )


def test_neyman_allocation_uses_caller_supplied_stratum_variability() -> None:
    """Neyman allocation differs only through declared stratum evidence."""
    design = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(60, 0.1), SamplingStratum(40, 0.5)],
        allocation_method="neyman",
    )

    assert design.sample_size == 43
    assert design.stratum_sample_sizes == (20, 23)
    assert design.stratum_inclusion_probability_ratios == ((20, 60), (23, 40))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": True},
        {"confidence_level": 1.0},
        {"margin_of_error": float("nan")},
        {"strata": [SamplingStratum(100, 0.0)]},
        {"allocation_method": "equal"},
    ],
)
def test_sampling_design_rejects_implicit_or_invalid_evidence(
    kwargs: dict[str, object],
) -> None:
    """The Python seam rejects coercion and unsupported allocation rules."""
    arguments: dict[str, object] = {
        "population_size": 100,
        "confidence_level": 0.95,
        "margin_of_error": 0.1,
        "strata": [SamplingStratum(100, 0.5)],
        "allocation_method": "proportional",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        finite_population_proportion_design(**arguments)  # type: ignore[arg-type]


def test_sampling_resource_bounds_fail_before_rust_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rust resource/domain bounds are replayed before caller evidence or core work."""
    core_calls = 0

    def forbidden_core(*args: object, **kwargs: object) -> object:
        nonlocal core_calls
        core_calls += 1
        raise AssertionError("Rust dispatch must not run for invalid resource controls")

    monkeypatch.setattr(
        sampling_design_module._core,
        "finite_population_proportion_design",
        forbidden_core,
    )

    with pytest.raises(ValueError, match=r"population_size must be between 1 and 2\^53"):
        finite_population_proportion_design(
            (1 << 53) + 1,
            0.95,
            0.1,
            [SamplingStratum((1 << 53) + 1, 0.5)],
            allocation_method="proportional",
        )

    with pytest.raises(
        ValueError,
        match=r"strata\[0\]\.population_size must be between 1 and 2\^53",
    ):
        finite_population_proportion_design(
            100,
            0.95,
            0.1,
            [SamplingStratum((1 << 53) + 1, 0.5)],
            allocation_method="proportional",
        )

    oversized_strata = [SamplingStratum(1, 0.5)] * 100_001
    with pytest.raises(ValueError, match="strata must contain between 1 and 100000 entries"):
        finite_population_proportion_design(
            100_001,
            0.95,
            0.1,
            oversized_strata,
            allocation_method="proportional",
        )

    assert core_calls == 0


def test_sampling_schema_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A foreign Rust result schema is rejected before result marshalling."""
    assert (
        sampling_design_module.SAMPLING_DESIGN_SCHEMA_VERSION
        == sampling_design_module._core.SAMPLING_DESIGN_SCHEMA_VERSION
    )

    monkeypatch.setattr(
        sampling_design_module._core,
        "finite_population_proportion_design",
        lambda *args, **kwargs: {"schema_version": "fast-mlsirm.sampling-design.v2"},
    )

    with pytest.raises(ValueError, match="unsupported sampling-design schema version"):
        finite_population_proportion_design(
            100,
            0.95,
            0.1,
            [SamplingStratum(100, 0.5)],
            allocation_method="proportional",
        )


def test_sampling_algorithm_mismatch_fails_before_payload_marshalling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-schema result from a foreign Rust algorithm fails closed immediately."""
    monkeypatch.setattr(
        sampling_design_module._core,
        "finite_population_proportion_design",
        lambda *args, **kwargs: {
            "schema_version": sampling_design_module.SAMPLING_DESIGN_SCHEMA_VERSION,
            "algorithm_version": "1.0.0",
        },
    )

    with pytest.raises(ValueError, match="unsupported sampling-design algorithm version"):
        finite_population_proportion_design(
            100,
            0.95,
            0.1,
            [SamplingStratum(100, 0.5)],
            allocation_method="proportional",
        )


def test_sampling_inclusion_ratio_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed or inconsistent Rust ratio evidence cannot reach public marshalling."""
    real_core = sampling_design_module._core.finite_population_proportion_design
    raw_result = dict(
        real_core(
            100,
            0.95,
            0.1,
            [60, 40],
            [0.1, 0.5],
            "neyman",
        )
    )

    invalid_results: list[dict[str, object]] = []

    missing = dict(raw_result)
    missing.pop("stratum_inclusion_probability_ratios")
    invalid_results.append(missing)

    malformed = dict(raw_result)
    malformed["stratum_inclusion_probability_ratios"] = [object(), (23, 40)]
    invalid_results.append(malformed)

    wrong_count = dict(raw_result)
    wrong_count["stratum_inclusion_probability_ratios"] = [(20, 60)]
    invalid_results.append(wrong_count)

    inconsistent = dict(raw_result)
    inconsistent["stratum_inclusion_probability_ratios"] = [(21, 60), (22, 40)]
    invalid_results.append(inconsistent)

    for invalid_result in invalid_results:
        monkeypatch.setattr(
            sampling_design_module._core,
            "finite_population_proportion_design",
            lambda *args, _result=invalid_result, **kwargs: _result,
        )
        with pytest.raises(ValueError, match="invalid sampling-design inclusion ratios"):
            finite_population_proportion_design(
                100,
                0.95,
                0.1,
                [SamplingStratum(60, 0.1), SamplingStratum(40, 0.5)],
                allocation_method="neyman",
            )


def test_giant_integer_probability_fails_with_package_value_error() -> None:
    """Strict probability controls do not overflow through unnecessary float coercion."""
    with pytest.raises(ValueError, match="confidence_level must be finite"):
        finite_population_proportion_design(
            100,
            10**400,
            0.1,
            [SamplingStratum(100, 0.5)],
            allocation_method="proportional",
        )


def test_achieved_proportion_preserves_uncertainty_after_all_successes() -> None:
    """A perfect sample does not become an unsupported perfect-population claim."""
    design = finite_population_proportion_design(
        43_814,
        0.95,
        0.1,
        [SamplingStratum(43_814, 0.5)],
        allocation_method="proportional",
    )
    result = finite_population_achieved_proportion(design, design.sample_size)

    assert result.schema_version == ACHIEVED_PROPORTION_SCHEMA_VERSION
    assert result.algorithm_version == "1.0.0"
    assert result.design_artifact_sha256 == design.artifact_sha256
    assert result.estimated_proportion == 1.0
    assert result.design_variance == 0.0
    assert result.interval_method == "wang_konijn_equal_tailed"
    assert result.lower_proportion < 1.0
    assert result.upper_proportion == 1.0
    assert all(
        len(identity) == 64
        for identity in (
            result.source_sha256,
            result.input_sha256,
            result.output_sha256,
            result.artifact_sha256,
        )
    )


def test_achieved_proportion_rejects_partial_tampered_or_stratified_designs() -> None:
    """Only one complete replayable SRSWOR design reaches terminal arithmetic."""
    design = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(100, 0.5)],
        allocation_method="proportional",
    )
    stratified = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(60, 0.5), SamplingStratum(40, 0.5)],
        allocation_method="proportional",
    )
    with pytest.raises(ValueError, match="ProportionSamplingDesign"):
        finite_population_achieved_proportion(object(), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one SRSWOR stratum"):
        finite_population_achieved_proportion(stratified, 1)
    with pytest.raises(ValueError, match="retained inputs"):
        finite_population_achieved_proportion(
            replace(design, artifact_sha256="f" * 64), 1
        )
    with pytest.raises(ValueError, match="retained inputs"):
        finite_population_achieved_proportion(
            replace(design, sample_size=design.sample_size - 1), 1
        )
    for invalid in (True, -1, design.sample_size + 1):
        with pytest.raises(ValueError, match="success_count"):
            finite_population_achieved_proportion(design, invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "future", "schema version"),
        ("algorithm_version", "future", "algorithm version"),
    ],
)
def test_achieved_proportion_rejects_unknown_rust_contract(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    """Unknown terminal Rust contracts fail before public marshalling."""
    design = finite_population_proportion_design(
        100,
        0.95,
        0.1,
        [SamplingStratum(100, 0.5)],
        allocation_method="proportional",
    )
    real_core = sampling_design_module._core.finite_population_achieved_proportion
    raw = dict(
        real_core(
            design.artifact_sha256,
            design.population_size,
            design.sample_size,
            1,
            design.confidence_level,
        )
    )
    raw[field] = value
    monkeypatch.setattr(
        sampling_design_module._core,
        "finite_population_achieved_proportion",
        lambda *args, **kwargs: raw,
    )
    with pytest.raises(ValueError, match=message):
        finite_population_achieved_proportion(design, 1)

"""Versioned Rust-backed finite-population proportion sampling designs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from . import _core

SAMPLING_DESIGN_SCHEMA_VERSION: str = _core.SAMPLING_DESIGN_SCHEMA_VERSION
ACHIEVED_PROPORTION_SCHEMA_VERSION: str = _core.ACHIEVED_PROPORTION_SCHEMA_VERSION
_SAMPLING_DESIGN_ALGORITHM_VERSION = "1.1.0"
_ACHIEVED_PROPORTION_ALGORITHM_VERSION = "1.0.0"
_MAX_EXACT_F64_INTEGER = 1 << 53
_MAX_SAMPLING_STRATA = 100_000


@dataclass(frozen=True)
class SamplingStratum:
    """One disjoint population stratum with prior- or pilot-derived prevalence."""

    population_size: int
    expected_proportion: float


@dataclass(frozen=True)
class ProportionSamplingDesign:
    """Auditable finite-population sample size, FPC, and stratum allocation."""

    schema_version: str
    source_identity: str
    source_sha256: str
    algorithm_version: str
    population_size: int
    expected_proportion: float
    confidence_level: float
    critical_value: float
    margin_of_error: float
    uncorrected_sample_size: float
    sample_size: int
    finite_population_correction: float
    allocation_method: str
    strata: tuple[SamplingStratum, ...]
    stratum_sample_sizes: tuple[int, ...]
    stratum_inclusion_probability_ratios: tuple[tuple[int, int], ...]
    input_sha256: str
    output_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class AchievedProportion:
    """Terminal SRSWOR estimate, design variance, and exact interval."""

    schema_version: str
    source_identity: str
    source_sha256: str
    algorithm_version: str
    design_artifact_sha256: str
    population_size: int
    sample_size: int
    success_count: int
    estimated_proportion: float
    design_variance: float
    confidence_level: float
    interval_method: str
    lower_success_count: int
    upper_success_count: int
    lower_proportion: float
    upper_proportion: float
    input_sha256: str
    output_sha256: str
    artifact_sha256: str


def _exact_positive_integer(name: str, value: object) -> int:
    """Admit one exact positive built-in integer inside the Rust f64 identity domain."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    if value > _MAX_EXACT_F64_INTEGER:
        raise ValueError(f"{name} must be between 1 and 2^53")
    return value


def _exact_nonnegative_integer(name: str, value: object) -> int:
    """Admit one exact nonnegative built-in integer inside the Rust identity domain."""
    if type(value) is not int or value < 0 or value > _MAX_EXACT_F64_INTEGER:
        raise ValueError(f"{name} must be a built-in integer between zero and 2^53")
    return value


def _exact_probability(name: str, value: object) -> float:
    """Admit one exact built-in finite probability strictly inside the unit interval."""
    if type(value) is int:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    if type(value) is not float:
        raise ValueError(f"{name} must be a built-in real number")
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return value


def _required_result_field(result: object, name: str) -> object:
    """Read one package-owned Rust result field with a stable fail-closed error."""
    if type(result) is not dict or name not in result:
        raise ValueError(f"invalid sampling-design result: missing {name}")
    return result[name]


def _validated_inclusion_ratios(
    result: object,
) -> tuple[tuple[int, int], tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    """Replay the Rust stratum/inclusion contract before public construction."""
    raw_population_sizes = _required_result_field(result, "stratum_population_sizes")
    raw_expected_proportions = _required_result_field(
        result, "stratum_expected_proportions"
    )
    raw_sample_sizes = _required_result_field(result, "stratum_sample_sizes")
    if type(result) is not dict or "stratum_inclusion_probability_ratios" not in result:
        raise ValueError("invalid sampling-design inclusion ratios")
    raw_ratios = result["stratum_inclusion_probability_ratios"]
    if (
        type(raw_population_sizes) is not list
        or type(raw_expected_proportions) is not list
        or type(raw_sample_sizes) is not list
        or type(raw_ratios) is not list
    ):
        raise ValueError("invalid sampling-design inclusion ratios")
    if not (
        len(raw_population_sizes)
        == len(raw_expected_proportions)
        == len(raw_sample_sizes)
        == len(raw_ratios)
    ):
        raise ValueError("invalid sampling-design inclusion ratios")

    population_sizes: list[int] = []
    expected_proportions: list[float] = []
    sample_sizes: list[int] = []
    ratios: list[tuple[int, int]] = []
    for index, (population, proportion, sample, ratio) in enumerate(
        zip(
            raw_population_sizes,
            raw_expected_proportions,
            raw_sample_sizes,
            raw_ratios,
            strict=True,
        )
    ):
        if (
            type(population) is not int
            or type(proportion) is not float
            or type(sample) is not int
            or type(ratio) not in (list, tuple)
            or len(ratio) != 2
            or type(ratio[0]) is not int
            or type(ratio[1]) is not int
            or ratio[0] != sample
            or ratio[1] != population
        ):
            raise ValueError(
                f"invalid sampling-design inclusion ratios at stratum {index}"
            )
        population_sizes.append(population)
        expected_proportions.append(proportion)
        sample_sizes.append(sample)
        ratios.append((ratio[0], ratio[1]))

    return (
        tuple(ratios),
        tuple(sample_sizes),
        tuple(population_sizes),
        tuple(expected_proportions),
    )


def finite_population_proportion_design(
    population_size: int,
    confidence_level: float,
    margin_of_error: float,
    strata: list[SamplingStratum] | tuple[SamplingStratum, ...],
    *,
    allocation_method: Literal["proportional", "neyman"],
) -> ProportionSamplingDesign:
    """Return the NIST/ABS sampling design computed entirely by the Rust core.

    Each expected proportion must come from declared prior or pilot evidence.
    The API deliberately has no implicit ``0.5`` or allocation-weight default.
    """
    normalized_population_size = _exact_positive_integer(
        "population_size", population_size
    )
    normalized_confidence_level = _exact_probability(
        "confidence_level", confidence_level
    )
    normalized_margin_of_error = _exact_probability(
        "margin_of_error", margin_of_error
    )
    if type(strata) not in (list, tuple) or not strata:
        raise ValueError("strata must be a non-empty built-in list or tuple")
    if len(strata) > _MAX_SAMPLING_STRATA:
        raise ValueError(
            f"strata must contain between 1 and {_MAX_SAMPLING_STRATA} entries"
        )
    if type(allocation_method) is not str or allocation_method not in (
        "proportional",
        "neyman",
    ):
        raise ValueError("allocation_method must be proportional or neyman")
    population_sizes: list[int] = []
    expected_proportions: list[float] = []
    for index, stratum in enumerate(strata):
        if type(stratum) is not SamplingStratum:
            raise ValueError(f"strata[{index}] must be a SamplingStratum")
        population_sizes.append(
            _exact_positive_integer(
                f"strata[{index}].population_size", stratum.population_size
            )
        )
        expected_proportions.append(
            _exact_probability(
                f"strata[{index}].expected_proportion",
                stratum.expected_proportion,
            )
        )

    result = _core.finite_population_proportion_design(
        normalized_population_size,
        normalized_confidence_level,
        normalized_margin_of_error,
        population_sizes,
        expected_proportions,
        allocation_method,
    )
    schema_version = _required_result_field(result, "schema_version")
    if schema_version != SAMPLING_DESIGN_SCHEMA_VERSION:
        raise ValueError(
            "unsupported sampling-design schema version: "
            f"{schema_version!r}"
        )
    algorithm_version = _required_result_field(result, "algorithm_version")
    if algorithm_version != _SAMPLING_DESIGN_ALGORITHM_VERSION:
        raise ValueError(
            "unsupported sampling-design algorithm version: "
            f"{algorithm_version!r}"
        )
    (
        inclusion_ratios,
        result_sample_sizes,
        result_population_sizes,
        result_expected_proportions,
    ) = _validated_inclusion_ratios(result)

    return ProportionSamplingDesign(
        schema_version=schema_version,
        source_identity=_required_result_field(result, "source_identity"),
        source_sha256=_required_result_field(result, "source_sha256"),
        algorithm_version=algorithm_version,
        population_size=_required_result_field(result, "population_size"),
        expected_proportion=_required_result_field(result, "expected_proportion"),
        confidence_level=_required_result_field(result, "confidence_level"),
        critical_value=_required_result_field(result, "critical_value"),
        margin_of_error=_required_result_field(result, "margin_of_error"),
        uncorrected_sample_size=_required_result_field(
            result, "uncorrected_sample_size"
        ),
        sample_size=_required_result_field(result, "sample_size"),
        finite_population_correction=_required_result_field(
            result, "finite_population_correction"
        ),
        allocation_method=_required_result_field(result, "allocation_method"),
        strata=tuple(
            SamplingStratum(population, proportion)
            for population, proportion in zip(
                result_population_sizes,
                result_expected_proportions,
                strict=True,
            )
        ),
        stratum_sample_sizes=result_sample_sizes,
        stratum_inclusion_probability_ratios=inclusion_ratios,
        input_sha256=_required_result_field(result, "input_sha256"),
        output_sha256=_required_result_field(result, "output_sha256"),
        artifact_sha256=_required_result_field(result, "artifact_sha256"),
    )


def finite_population_achieved_proportion(
    design: ProportionSamplingDesign,
    success_count: int,
) -> AchievedProportion:
    """Terminate one complete one-stratum SRSWOR design through Rust.

    The sample denominator is the bound design's required sample size. A caller
    cannot submit a partial result or silently replace failed sampled units.
    """
    if type(design) is not ProportionSamplingDesign:
        raise ValueError("design must be a ProportionSamplingDesign")
    normalized_success_count = _exact_nonnegative_integer(
        "success_count", success_count
    )
    if len(design.strata) != 1 or len(design.stratum_sample_sizes) != 1:
        raise ValueError("achieved proportion currently requires one SRSWOR stratum")
    replayed = finite_population_proportion_design(
        design.population_size,
        design.confidence_level,
        design.margin_of_error,
        list(design.strata),
        allocation_method=design.allocation_method,
    )
    if replayed != design:
        raise ValueError("sampling design artifact does not match its retained inputs")
    if normalized_success_count > replayed.sample_size:
        raise ValueError("success_count must be between zero and design.sample_size")
    result = _core.finite_population_achieved_proportion(
        replayed.artifact_sha256,
        replayed.population_size,
        replayed.sample_size,
        normalized_success_count,
        replayed.confidence_level,
    )
    schema_version = _required_result_field(result, "schema_version")
    if schema_version != ACHIEVED_PROPORTION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported achieved-proportion schema version: {schema_version!r}"
        )
    algorithm_version = _required_result_field(result, "algorithm_version")
    if algorithm_version != _ACHIEVED_PROPORTION_ALGORITHM_VERSION:
        raise ValueError(
            "unsupported achieved-proportion algorithm version: "
            f"{algorithm_version!r}"
        )
    return AchievedProportion(
        schema_version=schema_version,
        source_identity=_required_result_field(result, "source_identity"),
        source_sha256=_required_result_field(result, "source_sha256"),
        algorithm_version=algorithm_version,
        design_artifact_sha256=_required_result_field(
            result, "design_artifact_sha256"
        ),
        population_size=_required_result_field(result, "population_size"),
        sample_size=_required_result_field(result, "sample_size"),
        success_count=_required_result_field(result, "success_count"),
        estimated_proportion=_required_result_field(result, "estimated_proportion"),
        design_variance=_required_result_field(result, "design_variance"),
        confidence_level=_required_result_field(result, "confidence_level"),
        interval_method=_required_result_field(result, "interval_method"),
        lower_success_count=_required_result_field(result, "lower_success_count"),
        upper_success_count=_required_result_field(result, "upper_success_count"),
        lower_proportion=_required_result_field(result, "lower_proportion"),
        upper_proportion=_required_result_field(result, "upper_proportion"),
        input_sha256=_required_result_field(result, "input_sha256"),
        output_sha256=_required_result_field(result, "output_sha256"),
        artifact_sha256=_required_result_field(result, "artifact_sha256"),
    )

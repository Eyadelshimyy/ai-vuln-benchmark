"""Public package surface for fast-mlsirm."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from . import _legacy_init as _legacy_init
from . import cat as _cat
from . import cdm as _cdm
from . import dif as _dif
from . import exposure as _exposure
from . import fitstats as _fitstats
from . import inference as _inference
from . import polytomous as _polytomous
from . import reliability as _reliability
from . import scaling as _scaling
from . import serving as _serving
from . import validation as _validation
from ._cat_administration_resource_safety import (
    install as _install_cat_administration_resource_safety,
)
from ._cdm_response_safety import install as _install_cdm_response_safety
from ._dif_control_safety import install as _install_dif_control_safety
from ._exposure_array_safety import install as _install_exposure_array_safety
from ._exposure_flexilevel_safety import install as _install_exposure_flexilevel_safety
from ._fit_public import fit as _public_fit
from ._fitstats_control_safety import install as _install_fitstats_control_safety
from ._fleiss_control_safety import install as _install_fleiss_control_safety
from ._icc_control_safety import install as _install_icc_control_safety
from ._inference_admission_safety import install as _install_inference_admission_safety
from ._polytomous_prediction_admission import (
    install as _install_polytomous_prediction_admission,
)
from ._scaling_control_safety import install as _install_scaling_control_safety
from ._serving_export_safety import install as _install_serving_export_safety
from .interaction_map import ResidualInteractionMap as ResidualInteractionMap
from .interaction_map import residual_interaction_map as residual_interaction_map
from .sampling_design import (
    ACHIEVED_PROPORTION_SCHEMA_VERSION as ACHIEVED_PROPORTION_SCHEMA_VERSION,
)
from .sampling_design import (
    AchievedProportion as AchievedProportion,
)
from .sampling_design import (
    ProportionSamplingDesign as ProportionSamplingDesign,
)
from .sampling_design import SAMPLING_DESIGN_SCHEMA_VERSION as SAMPLING_DESIGN_SCHEMA_VERSION
from .sampling_design import SamplingStratum as SamplingStratum
from .sampling_design import (
    finite_population_achieved_proportion as finite_population_achieved_proportion,
)
from .sampling_design import (
    finite_population_proportion_design as finite_population_proportion_design,
)
from .bifactor_recursion import (
    bifactor_lord_wingersky as bifactor_lord_wingersky,
    direct_enumeration_bifactor as direct_enumeration_bifactor,
    enumerate_bifactor_direct as enumerate_bifactor_direct,
    enumerate_bifactor_lord_wingersky as enumerate_bifactor_lord_wingersky,
)
from .bifactor_bootstrap import (
    BifactorBootstrapResult as BifactorBootstrapResult,
    run_bifactor_bootstrap as run_bifactor_bootstrap,
)
from .personfit_multidim import compute_person_fit_multidim as compute_person_fit_multidim

# Harden historical public adapters before copying legacy exports. These
# wrappers validate and normalize semantic controls/evidence only; result
# arithmetic remains in the existing Rust-backed implementations.
_install_cat_administration_resource_safety(_cat)
_install_exposure_array_safety(_exposure)
_install_exposure_flexilevel_safety(_exposure)
_install_fitstats_control_safety(_fitstats)
_install_cdm_response_safety(_cdm)
_install_icc_control_safety(_reliability)

_install_dif_control_safety(_dif)
for _dif_name in (
    "detect_dif_logistic",
    "detect_dif_mantel_haenszel_purified",
    "detect_dif_logistic_purified",
):
    if hasattr(_legacy_init, _dif_name):
        setattr(_legacy_init, _dif_name, getattr(_dif, _dif_name))

_install_inference_admission_safety(_inference)
_install_scaling_control_safety(_scaling)
_install_fleiss_control_safety(_validation)
_install_polytomous_prediction_admission(_polytomous)
_install_serving_export_safety(_serving)
_legacy_init.ability_standard_error = _cat.ability_standard_error
_legacy_init.ccat_select = _exposure.ccat_select
_legacy_init.flexilevel_administer = _exposure.flexilevel_administer
_legacy_init.flexilevel_score_distribution = _exposure.flexilevel_score_distribution
_legacy_init.icc = _reliability.icc
_legacy_init.observed_information = _inference.observed_information
_legacy_init.second_order_test = _inference.second_order_test
_legacy_init.vcov_from_hessian = _inference.vcov_from_hessian
_legacy_init.standard_errors_from_vcov = _inference.standard_errors_from_vcov
_legacy_init.bradley_terry_mm = _scaling.bradley_terry_mm
_legacy_init.bratt_mm = _scaling.bratt_mm
_legacy_init.fleiss_kappa = _validation.fleiss_kappa
_legacy_init.export_serving_bundle = _serving.export_serving_bundle

del (
    _cat,
    _exposure,
    _inference,
    _install_cat_administration_resource_safety,
    _install_dif_control_safety,
    _install_exposure_array_safety,
    _install_exposure_flexilevel_safety,
    _fitstats,
    _install_fitstats_control_safety,
    _cdm,
    _install_cdm_response_safety,
    _install_fleiss_control_safety,
    _install_icc_control_safety,
    _install_inference_admission_safety,
    _install_polytomous_prediction_admission,
    _install_scaling_control_safety,
    _install_serving_export_safety,
    _polytomous,
    _dif,
    _dif_name,
    _reliability,
    _scaling,
    _serving,
)

# Copy only declared legacy exports that are currently defined. This preserves
# the established package surface without leaking helper imports from the
# compatibility module or making import depend on unrelated stale names.
for _public_name in _legacy_init.__all__:
    if hasattr(_legacy_init, _public_name):
        globals()[_public_name] = getattr(_legacy_init, _public_name)

# The legacy compatibility module imports the implementation-level ``fit``
# callable, which carries private reference-backend authority for
# ``fit_reference``. Rebind the package export after the legacy copy so callers
# can never acquire that authority from the public ``fast_mlsirm.fit`` API.
fit = _public_fit

from .bifactor_scoreability import (
    BifactorScoreabilityResult as BifactorScoreabilityResult,
)
from .bifactor_scoreability import (
    bifactor_scoreability as bifactor_scoreability,
)
from .bifactor_scoreability import (
    bifactor_scoreability_from_logit_slopes as bifactor_scoreability_from_logit_slopes,
)
from .bifactor_scoreability import (
    assess_bifactor_scoreability as assess_bifactor_scoreability,
)
from .bifactor_scoreability import (
    assess_bifactor_scoreability_from_logit_slopes as assess_bifactor_scoreability_from_logit_slopes,
)
from .irt_contract import (
    MIN_FACTOR_ANCHOR_ITEMS as MIN_FACTOR_ANCHOR_ITEMS,
)
from .irt_contract import (
    MIN_IRT_ITEMS as MIN_IRT_ITEMS,
)
from .irt_contract import (
    MIN_IRT_PERSONS as MIN_IRT_PERSONS,
)
from .irt_contract import (
    MIN_ITEM_DISTINCT_VALUES as MIN_ITEM_DISTINCT_VALUES,
)
from .irt_contract import (
    MIN_OBSERVED_PER_ITEM as MIN_OBSERVED_PER_ITEM,
)
from .irt_contract import (
    IRTItemType as IRTItemType,
)
from .irt_contract import (
    fit_irt_experiment as fit_irt_experiment,
)
from .irt_contract import (
    validate_irt_experiment_readiness as validate_irt_experiment_readiness,
)
from .irt_contract import (
    validate_irt_response_matrix as validate_irt_response_matrix,
)
from .judge_calibration import (
    CALIBRATION_VARIANTS as CALIBRATION_VARIANTS,
)
from .judge_calibration import (
    CONTAMINATION_STATUSES as CONTAMINATION_STATUSES,
)
from .judge_calibration import (
    JudgeCalibrationCase as JudgeCalibrationCase,
)
from .judge_calibration import (
    JudgeCalibrationOutcome as JudgeCalibrationOutcome,
)
from .judge_calibration import (
    JudgeCalibrationReport as JudgeCalibrationReport,
)
from .judge_calibration import (
    build_multiple_choice_calibration_cases as build_multiple_choice_calibration_cases,
)
from .judge_calibration import (
    evaluate_paired_calibration as evaluate_paired_calibration,
)
from .judge_construct import (
    ABSOLUTE_JUDGE_CONSTRUCT_FLOOR as ABSOLUTE_JUDGE_CONSTRUCT_FLOOR,
    DEFAULT_JUDGE_CONSTRUCT_POLICY as DEFAULT_JUDGE_CONSTRUCT_POLICY,
    JUDGE_MEASUREMENT_GUIDANCE as JUDGE_MEASUREMENT_GUIDANCE,
    JudgeConstructPolicy as JudgeConstructPolicy,
    JudgeConstructSpec as JudgeConstructSpec,
    MAX_JUDGE_CONSTRUCT_ITEMS as MAX_JUDGE_CONSTRUCT_ITEMS,
    MIN_JUDGE_CONSTRUCT_ITEMS as MIN_JUDGE_CONSTRUCT_ITEMS,
    RECOMMENDED_JUDGE_CONSTRUCT_ITEMS as RECOMMENDED_JUDGE_CONSTRUCT_ITEMS,
    ZERO_BASED_CATEGORY_CODING as ZERO_BASED_CATEGORY_CODING,
    describe_measurement_contract as describe_measurement_contract,
    project_judge_results_to_matrix as project_judge_results_to_matrix,
    validate_judge_construct as validate_judge_construct,
)
from .llm_judge import (
    CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1 as CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1,
)
from .llm_judge import (
    MAX_BINARY_THRESHOLD_CALLS as MAX_BINARY_THRESHOLD_CALLS,
)
from .llm_judge import (
    MAX_JUDGE_CATEGORIES as MAX_JUDGE_CATEGORIES,
)
from .llm_judge import (
    ContextualOrchestratorJudge as ContextualOrchestratorJudge,
)
from .llm_judge import (
    JudgeCriterion as JudgeCriterion,
)
from .llm_judge import (
    JudgeFormatError as JudgeFormatError,
)
from .llm_judge import (
    LLMJudgeResult as LLMJudgeResult,
)
from .rating_range import (
    RatingRangeEvidence as RatingRangeEvidence,
)
from .rating_range import (
    paired_rating_range_evidence as paired_rating_range_evidence,
)
from .regression import (
    chi2_sf_df1 as chi2_sf_df1,
)
from .regression import (
    conditional_slope as conditional_slope,
)
from .regression import (
    contrast as contrast,
)
from .regression import (
    design_row_dot as design_row_dot,
)
from .regression import (
    f_sf as f_sf,
)
from .regression import (
    fit_ols_hc as fit_ols_hc,
)
from .regression import (
    slope_difference as slope_difference,
)
from .regression import (
    t_sf as t_sf,
)
from .regression import (
    xwz_e_design_row as xwz_e_design_row,
)
from .rotation import (
    RotationCriterionInfo as RotationCriterionInfo,
)
from .rotation import (
    RotationSolution as RotationSolution,
)
from .rotation import (
    available_rotation_criteria as available_rotation_criteria,
)
from .rotation import (
    rotate_factor_loadings as rotate_factor_loadings,
)
from .rotation import (
    rotation_criterion_value_gradient as rotation_criterion_value_gradient,
)

# Bind the rating-range API on the historical validation namespace without
# duplicating implementation or arithmetic ownership.
_validation.RatingRangeEvidence = RatingRangeEvidence
_validation.paired_rating_range_evidence = paired_rating_range_evidence

# Resolve distribution metadata at the package boundary on every reload. The
# compatibility module may remain cached, so its copied value is not sufficient
# for reliable source-checkout fallback behavior.
try:
    __version__ = _distribution_version("fast-mlsirm")
except _PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = list(_legacy_init.__all__) + [
    "CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1",
    "BifactorScoreabilityResult",
    "bifactor_scoreability",
    "bifactor_scoreability_from_logit_slopes",
    "assess_bifactor_scoreability",
    "assess_bifactor_scoreability_from_logit_slopes",
    "RotationCriterionInfo",
    "RotationSolution",
    "available_rotation_criteria",
    "rotate_factor_loadings",
    "rotation_criterion_value_gradient",
    "ContextualOrchestratorJudge",
    "JudgeCriterion",
    "JudgeFormatError",
    "LLMJudgeResult",
    "MAX_BINARY_THRESHOLD_CALLS",
    "MAX_JUDGE_CATEGORIES",
    "CALIBRATION_VARIANTS",
    "CONTAMINATION_STATUSES",
    "JudgeCalibrationCase",
    "JudgeCalibrationOutcome",
    "JudgeCalibrationReport",
    "build_multiple_choice_calibration_cases",
    "evaluate_paired_calibration",
    "ABSOLUTE_JUDGE_CONSTRUCT_FLOOR",
    "DEFAULT_JUDGE_CONSTRUCT_POLICY",
    "JUDGE_MEASUREMENT_GUIDANCE",
    "JudgeConstructPolicy",
    "JudgeConstructSpec",
    "MAX_JUDGE_CONSTRUCT_ITEMS",
    "MIN_JUDGE_CONSTRUCT_ITEMS",
    "RECOMMENDED_JUDGE_CONSTRUCT_ITEMS",
    "ZERO_BASED_CATEGORY_CODING",
    "describe_measurement_contract",
    "project_judge_results_to_matrix",
    "validate_judge_construct",
    "IRTItemType",
    "fit_irt_experiment",
    "MIN_IRT_ITEMS",
    "validate_irt_response_matrix",
    "MIN_IRT_PERSONS",
    "MIN_OBSERVED_PER_ITEM",
    "MIN_ITEM_DISTINCT_VALUES",
    "MIN_FACTOR_ANCHOR_ITEMS",
    "validate_irt_experiment_readiness",
    "RatingRangeEvidence",
    "paired_rating_range_evidence",
    "chi2_sf_df1",
    "conditional_slope",
    "contrast",
    "design_row_dot",
    "f_sf",
    "fit_ols_hc",
    "slope_difference",
    "t_sf",
    "xwz_e_design_row",
    "SAMPLING_DESIGN_SCHEMA_VERSION",
    "ACHIEVED_PROPORTION_SCHEMA_VERSION",
    "SamplingStratum",
    "ProportionSamplingDesign",
    "AchievedProportion",
    "finite_population_proportion_design",
    "finite_population_achieved_proportion",
    "bifactor_lord_wingersky",
    "direct_enumeration_bifactor",
    "enumerate_bifactor_lord_wingersky",
    "enumerate_bifactor_direct",
    "BifactorBootstrapResult",
    "run_bifactor_bootstrap",
]

del _PackageNotFoundError, _distribution_version, _public_fit, _public_name

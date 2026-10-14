"""Construct-contract tests for the LLM-as-a-Judge measurement boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fast_mlsirm import grm
from fast_mlsirm.judge_construct import (
    ABSOLUTE_JUDGE_CONSTRUCT_FLOOR,
    DEFAULT_JUDGE_CONSTRUCT_POLICY,
    JUDGE_MEASUREMENT_GUIDANCE,
    JudgeConstructPolicy,
    JudgeFormatError,
    LLMJudgeResult,
    MAX_JUDGE_CONSTRUCT_ITEMS,
    MIN_JUDGE_CONSTRUCT_ITEMS,
    RECOMMENDED_JUDGE_CONSTRUCT_ITEMS,
    ZERO_BASED_CATEGORY_CODING,
    describe_measurement_contract,
    project_judge_results_to_matrix,
    validate_judge_construct,
)


def _result(
    categories: dict[str, int],
    *,
    category_count: int = 4,
) -> LLMJudgeResult:
    """Build one validated judge decision carrying explicit categories."""
    scores = {criterion_id: category / (category_count - 1)
              for criterion_id, category in categories.items()}
    return LLMJudgeResult(
        score=sum(scores.values()) / len(scores),
        accepted=True,
        rationale="contract fixture",
        criterion_scores=scores,
        raw_output="{}",
        orchestration_mode="direct",
        trace_step_count=1,
        usage={"total_tokens": 1},
        criterion_categories=categories,
        category_count=category_count,
    )


FIVE_IDS = tuple(f"criterion_{index}" for index in range(5))


class TestJudgeConstructPolicy:
    def test_default_policy_matches_documented_bounds(self) -> None:
        assert DEFAULT_JUDGE_CONSTRUCT_POLICY.min_items == MIN_JUDGE_CONSTRUCT_ITEMS == 5
        assert (
            DEFAULT_JUDGE_CONSTRUCT_POLICY.recommended_items
            == RECOMMENDED_JUDGE_CONSTRUCT_ITEMS
            == 7
        )
        assert DEFAULT_JUDGE_CONSTRUCT_POLICY.max_items == MAX_JUDGE_CONSTRUCT_ITEMS == 11

    def test_policy_to_dict_round_trips_bounds(self) -> None:
        policy = JudgeConstructPolicy(min_items=4, recommended_items=6, max_items=9)
        assert policy.to_dict() == {
            "min_items": 4,
            "recommended_items": 6,
            "max_items": 9,
        }

    def test_policy_rejects_floor_violating_minimum(self) -> None:
        with pytest.raises(ValueError, match="min_items must be between"):
            JudgeConstructPolicy(min_items=2)

    def test_policy_rejects_recommended_outside_range(self) -> None:
        with pytest.raises(ValueError, match="recommended_items must be between"):
            JudgeConstructPolicy(min_items=5, recommended_items=3)

    def test_policy_rejects_non_integer_bounds(self) -> None:
        with pytest.raises(TypeError, match="exact built-in integer"):
            JudgeConstructPolicy(min_items=4.0)  # type: ignore[arg-type]


class TestValidateJudgeConstruct:
    def test_policy_floor_construct_warns_below_recommended_count(self) -> None:
        spec = validate_judge_construct(FIVE_IDS, n_categories=4)
        assert spec.n_items == 5
        assert spec.meets_policy is True
        assert spec.category_coding == ZERO_BASED_CATEGORY_CODING
        assert any("below the recommended" in warning for warning in spec.warnings)

    def test_recommended_count_carries_no_warnings(self) -> None:
        recommended_ids = tuple(
            f"criterion_{index}" for index in range(RECOMMENDED_JUDGE_CONSTRUCT_ITEMS)
        )
        spec = validate_judge_construct(recommended_ids, n_categories=4)
        assert spec.meets_policy is True
        assert spec.warnings == ()

    def test_below_recommendation_warns_but_meets_policy(self) -> None:
        spec = validate_judge_construct(FIVE_IDS[:5], n_categories=2)
        assert spec.meets_policy is True

    def test_short_form_requires_explicit_escape(self) -> None:
        with pytest.raises(JudgeFormatError, match="allow_short_form=True"):
            validate_judge_construct(FIVE_IDS[:4], n_categories=3)

    def test_short_form_escape_carries_precision_warning(self) -> None:
        spec = validate_judge_construct(
            FIVE_IDS[:4], n_categories=3, allow_short_form=True
        )
        assert spec.meets_policy is False
        assert any("short form admitted" in warning for warning in spec.warnings)

    def test_identification_floor_blocks_two_item_facets_even_with_escape(self) -> None:
        with pytest.raises(JudgeFormatError, match="factor identification"):
            validate_judge_construct(
                FIVE_IDS[:2], n_categories=3, allow_short_form=True
            )

    def test_absolute_floor_constant_matches_thurstone_identification_bound(
        self,
    ) -> None:
        assert ABSOLUTE_JUDGE_CONSTRUCT_FLOOR == 3

    def test_oversized_facet_is_rejected_with_split_guidance(self) -> None:
        oversized = tuple(f"criterion_{i}" for i in range(MAX_JUDGE_CONSTRUCT_ITEMS + 1))
        with pytest.raises(JudgeFormatError, match="split the rubric"):
            validate_judge_construct(oversized, n_categories=4)

    def test_duplicate_criteria_are_rejected(self) -> None:
        with pytest.raises(JudgeFormatError, match="unique"):
            validate_judge_construct(("a", "a", "b", "c", "d"), n_categories=2)

    def test_non_string_criteria_are_rejected(self) -> None:
        with pytest.raises(JudgeFormatError, match="non-empty strings"):
            validate_judge_construct(("a", 1, "b", "c", "d"), n_categories=2)  # type: ignore[list-item]

    def test_dichotomous_spec_fixes_category_count_and_forbids_n_categories(
        self,
    ) -> None:
        spec = validate_judge_construct(FIVE_IDS, item_type="dichotomous")
        assert spec.n_categories == 2
        with pytest.raises(JudgeFormatError, match="do not accept n_categories"):
            validate_judge_construct(FIVE_IDS, item_type="dichotomous", n_categories=4)

    def test_polytomous_requires_bounded_n_categories(self) -> None:
        with pytest.raises(JudgeFormatError, match="n_categories in"):
            validate_judge_construct(FIVE_IDS, n_categories=None)
        with pytest.raises(JudgeFormatError, match="n_categories in"):
            validate_judge_construct(FIVE_IDS, n_categories=1)

    def test_invalid_item_type_is_rejected(self) -> None:
        with pytest.raises(JudgeFormatError, match="dichotomous or polytomous"):
            validate_judge_construct(FIVE_IDS, item_type="graded")  # type: ignore[arg-type]

    def test_custom_policy_is_enforced(self) -> None:
        strict = JudgeConstructPolicy(min_items=6, recommended_items=8, max_items=9)
        with pytest.raises(JudgeFormatError, match="policy requires at least 6"):
            validate_judge_construct(FIVE_IDS, n_categories=2, policy=strict)
        admitted = validate_judge_construct(
            FIVE_IDS, n_categories=2, policy=strict, allow_short_form=True
        )
        assert admitted.meets_policy is False

    def test_non_policy_object_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="JudgeConstructPolicy"):
            validate_judge_construct(FIVE_IDS, n_categories=2, policy=object())  # type: ignore[arg-type]

    def test_spec_rejects_malformed_direct_construction(self) -> None:
        from fast_mlsirm.judge_construct import JudgeConstructSpec

        with pytest.raises(TypeError, match="non-empty tuple of strings"):
            JudgeConstructSpec((), "polytomous", 4, ZERO_BASED_CATEGORY_CODING, True)
        valid_ids = ("a", "b")
        with pytest.raises(ValueError, match="unique"):
            JudgeConstructSpec(
                ("a", "a"), "polytomous", 4, ZERO_BASED_CATEGORY_CODING, True
            )
        with pytest.raises(ValueError, match="item_type must be"):
            JudgeConstructSpec(
                valid_ids, "ordinal", 4, ZERO_BASED_CATEGORY_CODING, True
            )
        with pytest.raises(ValueError, match="category_coding must be"):
            JudgeConstructSpec(valid_ids, "polytomous", 4, "one_based", True)
        with pytest.raises(ValueError, match="short-form specs require"):
            JudgeConstructSpec(valid_ids, "polytomous", 4, ZERO_BASED_CATEGORY_CODING, False)

    def test_spec_dict_reports_contract_fields(self) -> None:
        spec = validate_judge_construct(FIVE_IDS, n_categories=4)
        payload = spec.to_dict()
        assert payload["n_items"] == 5
        assert payload["category_coding"] == ZERO_BASED_CATEGORY_CODING
        assert payload["meets_policy"] is True


class TestProjectJudgeResultsToMatrix:
    def _spec(self):
        return validate_judge_construct(FIVE_IDS, n_categories=4)

    def test_projection_shape_dtype_and_zero_based_range(self) -> None:
        results = [
            _result({criterion_id: index % 4 for index, criterion_id in enumerate(FIVE_IDS)})
            for _ in range(7)
        ]
        matrix = project_judge_results_to_matrix(results, self._spec())
        assert matrix.shape == (7, 5)
        assert matrix.dtype == np.int64
        assert matrix.min() >= 0
        assert matrix.max() <= 3

    def test_projection_preserves_person_order(self) -> None:
        first = _result(dict.fromkeys(FIVE_IDS, 0))
        second = _result(dict.fromkeys(FIVE_IDS, 3))
        matrix = project_judge_results_to_matrix([first, second], self._spec())
        assert matrix.tolist() == [[0, 0, 0, 0, 0], [3, 3, 3, 3, 3]]

    def test_non_result_entries_are_rejected(self) -> None:
        with pytest.raises(TypeError, match="LLMJudgeResult values"):
            project_judge_results_to_matrix([object()], self._spec())  # type: ignore[list-item]

    def test_empty_results_are_rejected(self) -> None:
        with pytest.raises(JudgeFormatError, match="at least one judged response"):
            project_judge_results_to_matrix([], self._spec())

    def test_criterion_set_mismatch_is_rejected(self) -> None:
        result = _result({"other_a": 0, "other_b": 1, "other_c": 2, "other_d": 3})
        with pytest.raises(JudgeFormatError, match="does not match the validated"):
            project_judge_results_to_matrix([result], self._spec())

    def test_dichotomous_projection_stays_binary(self) -> None:
        spec = validate_judge_construct(FIVE_IDS, item_type="dichotomous")
        low = _result(dict.fromkeys(FIVE_IDS, 0), category_count=2)
        high = _result(dict.fromkeys(FIVE_IDS, 1), category_count=2)
        matrix = project_judge_results_to_matrix([low, high], spec)
        assert set(np.unique(matrix)) <= {0, 1}


def _simulate_grm_panel(
    rng: np.random.Generator,
    theta: np.ndarray,
    slopes: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Draw ordered categories under a graded response model."""
    categories = np.zeros((theta.size, slopes.size))
    for item_index, slope in enumerate(slopes):
        cumulative = 1.0 / (
            1.0 + np.exp(-slope * (theta[:, None] - thresholds[item_index][None, :]))
        )
        draws = rng.random(theta.size)
        categories[:, item_index] = (draws[:, None] < cumulative).sum(axis=1)
    return categories


class TestJudgePanelGrmRecovery:
    def test_projected_judge_panel_recovers_true_parameters(self) -> None:
        """A synthetic judge panel scored through the contract must recover
        known GRM parameters: RMSE on slopes below 0.25 and person-theta
        rank correlation above 0.90 at N=800, k=4, and the recommended
        seven-criterion facet size."""
        rng = np.random.default_rng(20260825)
        true_slopes = np.array([1.80, 1.60, 1.90, 1.70, 1.85, 1.65, 1.95])
        true_thresholds = np.array(
            [
                [-1.60, -0.50, 0.50],
                [-1.40, -0.30, 0.70],
                [-1.80, -0.60, 0.40],
                [-1.30, -0.20, 0.80],
                [-1.50, -0.40, 0.55],
                [-1.70, -0.55, 0.45],
                [-1.20, -0.10, 0.85],
            ]
        )
        true_theta = rng.normal(size=800)
        panel = _simulate_grm_panel(rng, true_theta, true_slopes, true_thresholds)

        criterion_ids = tuple(
            f"criterion_{index}" for index in range(RECOMMENDED_JUDGE_CONSTRUCT_ITEMS)
        )
        results = [
            _result(
                {
                    criterion_id: int(panel[person_index, item_index])
                    for item_index, criterion_id in enumerate(criterion_ids)
                }
            )
            for person_index in range(panel.shape[0])
        ]
        spec = validate_judge_construct(criterion_ids, n_categories=4)
        matrix = project_judge_results_to_matrix(results, spec)
        fit = grm.fit_grm(matrix.astype(float), n_cat=4, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)

        slope_rmse = float(np.sqrt(np.mean((fit.slope[:, 0] - true_slopes) ** 2)))
        estimated_theta = np.asarray(fit.theta).reshape(-1)
        correlation = float(np.corrcoef(true_theta, estimated_theta)[0, 1])
        assert slope_rmse < 0.25, f"slope RMSE {slope_rmse:.3f}"
        assert correlation > 0.90, f"theta r {correlation:.3f}"


class TestAgentFacingContract:
    def test_describe_measurement_contract_reports_all_axes(self) -> None:
        description = describe_measurement_contract()
        assert description["category_coding"] == ZERO_BASED_CATEGORY_CODING
        assert description["policy"]["min_items"] == MIN_JUDGE_CONSTRUCT_ITEMS
        assert description["absolute_floor"] == ABSOLUTE_JUDGE_CONSTRUCT_FLOOR
        assert len(description["references"]) >= 4

    def test_guidance_names_zero_based_coding_and_item_bounds(self) -> None:
        assert "zero-based" in JUDGE_MEASUREMENT_GUIDANCE
        assert str(MIN_JUDGE_CONSTRUCT_ITEMS) in JUDGE_MEASUREMENT_GUIDANCE

    def test_score_fallback_bins_into_zero_based_categories(self) -> None:
        """Score-only results must still produce zero-based categories."""
        spec = validate_judge_construct(FIVE_IDS, n_categories=4)
        base = _result(dict.fromkeys(FIVE_IDS, 1))
        score_only = replace(base, criterion_categories=None, category_count=None)
        matrix = project_judge_results_to_matrix([score_only], spec)
        assert matrix.min() >= 0 and matrix.max() < 4

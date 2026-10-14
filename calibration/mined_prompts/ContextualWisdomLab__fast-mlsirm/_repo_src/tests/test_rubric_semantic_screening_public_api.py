"""Public-surface contract for governed semantic screening."""

from fast_mlsirm import rubric
from fast_mlsirm.rubric import semantic_screening


def test_semantic_screening_contract_is_available_on_rubric_namespace() -> None:
    """Governed screening types must follow the established rubric API surface."""
    assert rubric.CandidateScreeningResult is semantic_screening.CandidateScreeningResult
    assert rubric.SemanticScreeningCheck is semantic_screening.SemanticScreeningCheck
    assert rubric.ScreeningDimension is semantic_screening.ScreeningDimension
    assert rubric.ScreeningEvaluatorKind is semantic_screening.ScreeningEvaluatorKind
    assert rubric.ScreeningStatus is semantic_screening.ScreeningStatus
    assert rubric.build_candidate_screening_result is semantic_screening.build_candidate_screening_result
    assert rubric.build_semantic_screening_check is semantic_screening.build_semantic_screening_check

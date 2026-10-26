"""Knowledge compilation contract — phenomenology → System 1 feature."""

from jobbot.ops.compile import COMPRESS_CONTRACT, PHENOMENOLOGY_CONTRACT, PROMOTION_STEPS


def test_phenomenology_rejects_requirements_satisfaction() -> None:
    assert "do not satisfy needs via requirements" in PHENOMENOLOGY_CONTRACT
    assert "conditions of possibility" in PHENOMENOLOGY_CONTRACT
    assert "need for a need" in PHENOMENOLOGY_CONTRACT


def test_compress_contract_states_conditions_not_tickets() -> None:
    assert "System 2" in COMPRESS_CONTRACT
    assert "System 1" in COMPRESS_CONTRACT
    assert "conditions of possibility" in COMPRESS_CONTRACT
    assert "requirement ticket" in COMPRESS_CONTRACT


def test_promotion_steps_analyze_before_implement() -> None:
    assert PROMOTION_STEPS[0] == "capture_symptom_redacted_locally"
    assert PROMOTION_STEPS[1] == "analyze_conditions_of_possibility"
    assert "name_falsifiable_structural_rule" in PROMOTION_STEPS
    assert PROMOTION_STEPS[-1] == "triage_symptom_resolved"

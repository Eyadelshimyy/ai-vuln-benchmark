"""Deterministic bounded compilation of rubric item-authoring blueprints."""

from __future__ import annotations

from dataclasses import replace

from .models import BlueprintPlan, ItemBlueprint, RubricSpecification, _sha256_hex

MAX_BLUEPRINTS = 10_000


def compile_item_blueprints(
    rubric: RubricSpecification,
    plan: BlueprintPlan | None = None,
) -> tuple[ItemBlueprint, ...]:
    """Compile an ordered, content-addressed task/evidence blueprint matrix.

    Parameters
    ----------
    rubric:
        Exact versioned rubric whose construct, evidence, and score model are
        copied into every generated blueprint.
    plan:
        Optional matrix dimensions. Omitting it uses every difficulty band,
        single-source evidence, one item per cell, and seed zero.

    Returns
    -------
    tuple[ItemBlueprint, ...]
        Stable task-family, difficulty, evidence-mode, replicate order.
    """
    if not isinstance(rubric, RubricSpecification):
        raise TypeError("rubric must be a RubricSpecification")
    resolved_plan = BlueprintPlan() if plan is None else plan
    if not isinstance(resolved_plan, BlueprintPlan):
        raise TypeError("plan must be a BlueprintPlan")

    total = (
        len(rubric.task_families)
        * len(resolved_plan.difficulty_bands)
        * len(resolved_plan.evidence_modes)
        * resolved_plan.items_per_cell
    )
    if total > MAX_BLUEPRINTS:
        raise ValueError(
            f"requested blueprint matrix ({total}) exceeds "
            f"MAX_BLUEPRINTS={MAX_BLUEPRINTS}"
        )

    rubric_fingerprint = rubric.fingerprint
    scoring_levels = tuple(level.score for level in rubric.levels)
    compiled: list[ItemBlueprint] = []
    for task_family in rubric.task_families:
        for difficulty_band in resolved_plan.difficulty_bands:
            for evidence_mode in resolved_plan.evidence_modes:
                for replicate_index in range(resolved_plan.items_per_cell):
                    seed_identity = {
                        "schema_version": rubric.schema_version,
                        "rubric_version": rubric.rubric_version,
                        "rubric_fingerprint": rubric_fingerprint,
                        "plan_seed": resolved_plan.seed,
                        "task_family": task_family,
                        "difficulty_band": difficulty_band.value,
                        "evidence_mode": evidence_mode.value,
                        "replicate_index": replicate_index,
                    }
                    seed_digest = _sha256_hex(seed_identity)
                    provisional = ItemBlueprint(
                        blueprint_id="item_blueprint_pending",
                        rubric_id=rubric.rubric_id,
                        rubric_fingerprint=rubric_fingerprint,
                        task_family=task_family,
                        difficulty_band=difficulty_band,
                        evidence_mode=evidence_mode,
                        replicate_index=replicate_index,
                        generation_seed=int(seed_digest[:16], 16),
                        response_format=rubric.response_format,
                        scoring_levels=scoring_levels,
                        evidence_requirements=rubric.evidence_requirements,
                        prohibited_patterns=rubric.prohibited_patterns,
                        rubric_version=rubric.rubric_version,
                        schema_version=rubric.schema_version,
                    )
                    compiled.append(
                        replace(
                            provisional,
                            blueprint_id=(
                                "item_blueprint_"
                                f"{provisional.blueprint_fingerprint[:16]}"
                            ),
                        )
                    )
    return tuple(compiled)

"""Knowledge compilation — phenomenology of software → System 1.

Not classical requirements (“satisfy the need”). We analyze the
**conditions of possibility** of the *need for a need*:

- **System 2 / vibecode:** the need *appears* (high client-token cost).
- **Symptom:** the same appearance *returns* (Lacanian: what repeats).
- **Conditions of possibility:** structural answer to why that need-for-a-need
  can arise — falsifiable, not a wish-list item.
- **System 1 / feature:** those conditions encoded as a cheap callable.

Capture appearances locally and redacted via ``jobbot ops symptom note``.
``--rule`` names the structural condition, not the product desire.

See ``docs/software-design.md`` §3.8 and ``AGENTS.md`` Design economics.
"""

from __future__ import annotations

COMPRESS_CONTRACT = (
    "appearance (System 2) → symptom marks the return → "
    "analyze conditions of possibility of the need-for-a-need → "
    "encode as feature (System 1); never unpaid System 2 replay; "
    "never confuse compression with satisfying a requirement ticket"
)

PHENOMENOLOGY_CONTRACT = (
    "we do not satisfy needs via requirements; "
    "we analyze the conditions of possibility of the need for a need"
)

PROMOTION_STEPS: tuple[str, ...] = (
    "capture_symptom_redacted_locally",
    "analyze_conditions_of_possibility",
    "name_falsifiable_structural_rule",
    "implement_deterministic",
    "optional_llm_inside_via_gateway",
    "test_first_then_cli_if_human",
    "call_feature_do_not_rederive",
    "triage_symptom_resolved",
)

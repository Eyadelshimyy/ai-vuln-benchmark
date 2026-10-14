"""Evaluator implementations for natural-language FSLM decisions."""

from __future__ import annotations

import json

from ellements.core import LLMClient
from pydantic import BaseModel, ConfigDict, Field

from .models import (
    DecisionResult,
    FSLMEvent,
    GuardSpec,
    InvariantSpec,
    MachineSnapshot,
    MachineSpec,
)


class _LLMDecision(BaseModel):
    """Provider-safe structured decision schema for live LLM calls."""

    model_config = ConfigDict(extra="forbid")

    id: str
    allowed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)

    def to_decision_result(self) -> DecisionResult:
        """Convert the provider-safe schema into the public result model."""
        return DecisionResult(
            id=self.id,
            allowed=self.allowed,
            confidence=self.confidence,
            evidence=self.evidence,
            uncertainties=self.uncertainties,
            alternatives=self.alternatives,
        )


class LLMDecisionEvaluator:
    """Natural-language guard and invariant evaluator backed by `LLMClient`."""

    def __init__(self, client: LLMClient, *, model: str | None = None) -> None:
        self.client = client
        self.model = model

    async def evaluate_guard(
        self,
        guard: GuardSpec,
        *,
        spec: MachineSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> DecisionResult:
        """Evaluate an NL guard as a structured decision."""
        return await self._evaluate(
            decision_id=guard.id,
            rule_text=guard.text,
            rule_kind="guard",
            spec=spec,
            snapshot=snapshot,
            event=event,
        )

    async def check_invariant(
        self,
        invariant: InvariantSpec,
        *,
        spec: MachineSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> DecisionResult:
        """Evaluate an NL invariant as a structured decision."""
        return await self._evaluate(
            decision_id=invariant.id,
            rule_text=invariant.text,
            rule_kind="invariant",
            spec=spec,
            snapshot=snapshot,
            event=event,
        )

    async def _evaluate(
        self,
        *,
        decision_id: str,
        rule_text: str,
        rule_kind: str,
        spec: MachineSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> DecisionResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You evaluate finite-state linguistic machine rules. Return only the "
                    "structured schema. Be conservative: if evidence is missing "
                    "or ambiguous, set allowed=false or lower confidence and "
                    "explain uncertainty."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"Decision id: {decision_id}",
                        f"Rule kind: {rule_kind}",
                        f"Rule text:\n{rule_text}",
                        "Machine context:",
                        json.dumps(
                            {
                                "machine": spec.name,
                                "state": snapshot.current_state,
                                "variables": snapshot.variables,
                                "event": event.model_dump(mode="json"),
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                    ]
                ),
            },
        ]
        result = await self.client.complete_structured(
            messages,
            response_model=_LLMDecision,
            model=self.model,
            temperature=0.0,
        )
        parsed = result.to_decision_result()
        if parsed.id != decision_id:
            return parsed.model_copy(update={"id": decision_id})
        return parsed


__all__ = ["LLMDecisionEvaluator"]

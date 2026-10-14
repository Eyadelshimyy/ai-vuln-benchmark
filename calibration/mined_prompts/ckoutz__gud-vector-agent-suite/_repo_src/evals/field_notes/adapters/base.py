"""Provider-neutral runner contract.

An adapter receives the prior field state, the conversation so far, and the new
transcript, and returns a :class:`TurnResult` (or a schema failure). The same
contract is intended to cover managed schema-output APIs and an OpenAI-compatible
vLLM endpoint; today only deterministic local adapters are implemented.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from field_notes.schema import NoteFields, TurnResult


class HistoryTurn(BaseModel):
    """One prior utterance in the conversation."""

    model_config = ConfigDict(extra="forbid")

    role: str
    text: str


class TurnRequest(BaseModel):
    """Everything an adapter is given for one turn."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    turn_index: int
    prior_fields: NoteFields
    history: list[HistoryTurn] = Field(default_factory=list)
    transcript: str


class TokenUsage(BaseModel):
    """Provider-neutral token accounting, as reported by whatever served the turn.

    Every field is optional because not all endpoints report all of them, and the suite
    stores what it is told rather than deriving or estimating anything.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TurnOutcome(BaseModel):
    """Adapter response plus the operational slots real runs populate."""

    model_config = ConfigDict(extra="forbid")

    result: TurnResult | None = None
    schema_valid: bool
    error: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    usage: TokenUsage | None = None
    raw_response: str | None = None


class AdapterUnavailableError(RuntimeError):
    """Raised when an adapter cannot run without configuration this suite refuses to embed."""


@runtime_checkable
class ExtractionAdapter(Protocol):
    """Contract every evaluated candidate is driven through."""

    @property
    def name(self) -> str:
        """Stable candidate name used in reports."""
        ...

    def run_turn(self, request: TurnRequest) -> TurnOutcome:
        """Produce a turn outcome for ``request`` without mutating suite state."""
        ...

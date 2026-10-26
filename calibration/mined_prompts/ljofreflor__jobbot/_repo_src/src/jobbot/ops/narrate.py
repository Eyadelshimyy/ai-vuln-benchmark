"""Phase narration for long JobBot loops (local stdout only; no telemetry)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class Phase(StrEnum):
    """The loop as the candidate experiences it: push CV out, learn, get asked back."""

    PROPAGATING_CV = "propagando cv"
    UPDATING_WORLD = "actualizando el mundo"
    RECEIVING_WORLD = "recibiendo información del mundo"
    EXPLORING = "explorando el mundo"
    SURFACING = "emergiendo con lo encontrado"


_MAY_ASK_SUFFIX = "puede que tenga que hacerte algunas preguntas"


@dataclass(frozen=True)
class ExplorationOutcome:
    """What a sounding run actually established, kept in three separate buckets.

    Collapsing `refused` into `found=0` is how a probe turns into a false claim: a
    site that blocks us said nothing about whether it hires. The three counts stay
    apart so the report can be optimistic about the search without lying about it.
    """

    found: int = 0
    refused: int = 0
    unknown: int = 0


def outcome_line(outcome: ExplorationOutcome) -> str:
    """One line that never turns a closed door into an absence."""
    parts = [f"{outcome.found} con hallazgo"]
    if outcome.refused:
        parts.append(f"{outcome.refused} nos rechazaron (seguimos sin saber)")
    if outcome.unknown:
        parts.append(f"{outcome.unknown} sin señal todavía")
    return "  " + ", ".join(parts)


def phase_line(phase: Phase, detail: str | None = None, *, may_ask: bool = False) -> str:
    """Render one narration line. Kept pure so tests can assert the wording."""
    text = phase.value
    if may_ask:
        text = f"{text}, {_MAY_ASK_SUFFIX}"
    if detail:
        text = f"{text} ({detail})"
    return f"{text} …"


@dataclass
class Narrator:
    """Prints phase lines through an injected sink (Rich console, list, /dev/null)."""

    sink: Callable[[str], None] | None = None
    quiet: bool = False
    lines: list[str] = field(default_factory=list)

    def phase(self, phase: Phase, detail: str | None = None, *, may_ask: bool = False) -> str:
        line = phase_line(phase, detail, may_ask=may_ask)
        self.lines.append(line)
        if not self.quiet and self.sink is not None:
            self.sink(line)
        return line

    def outcome(self, outcome: ExplorationOutcome) -> str:
        line = outcome_line(outcome)
        self.lines.append(line)
        if not self.quiet and self.sink is not None:
            self.sink(line)
        return line

    def note(self, text: str) -> str:
        line = f"  {text}"
        self.lines.append(line)
        if not self.quiet and self.sink is not None:
            self.sink(line)
        return line

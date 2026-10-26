"""Phase narration wording for the long loops."""

from __future__ import annotations

from jobbot.ops.narrate import Narrator, Phase, phase_line


def test_phase_lines_read_like_the_loop() -> None:
    assert phase_line(Phase.PROPAGATING_CV) == "propagando cv …"
    assert phase_line(Phase.UPDATING_WORLD) == "actualizando el mundo …"
    assert phase_line(Phase.RECEIVING_WORLD, may_ask=True) == (
        "recibiendo información del mundo, puede que tenga que hacerte algunas preguntas …"
    )


def test_phase_line_carries_detail() -> None:
    assert phase_line(Phase.UPDATING_WORLD, "12 recruiter posts") == (
        "actualizando el mundo (12 recruiter posts) …"
    )


def test_narrator_writes_to_its_sink_unless_quiet() -> None:
    written: list[str] = []
    narrator = Narrator(sink=written.append)
    narrator.phase(Phase.PROPAGATING_CV, "Empresa · Data Scientist")
    narrator.note("cv adapted for J0001")
    assert written == [
        "propagando cv (Empresa · Data Scientist) …",
        "  cv adapted for J0001",
    ]

    quiet = Narrator(sink=written.append, quiet=True)
    quiet.phase(Phase.UPDATING_WORLD)
    assert len(written) == 2
    assert quiet.lines == ["actualizando el mundo …"]


def test_exploration_has_its_own_phases() -> None:
    from jobbot.ops.narrate import Phase as P

    assert phase_line(P.EXPLORING, "42 empresas") == "explorando el mundo (42 empresas) …"
    assert phase_line(P.SURFACING) == "emergiendo con lo encontrado …"


def test_a_refusal_is_never_reported_as_an_absence() -> None:
    """A 403 means we could not look, which is not the same as 'there is nothing'."""
    from jobbot.ops.narrate import ExplorationOutcome, outcome_line

    line = outcome_line(ExplorationOutcome(found=7, refused=2, unknown=3))

    assert "7" in line
    assert "2" in line and "rechaz" in line
    assert "3" in line
    assert "no tienen" not in line, "a refusal must not become a conclusion"


def test_finding_nothing_still_says_what_happened() -> None:
    from jobbot.ops.narrate import ExplorationOutcome, outcome_line

    only_blocked = outcome_line(ExplorationOutcome(found=0, refused=4, unknown=0))
    assert "rechaz" in only_blocked
    assert "sin hallazgos" not in only_blocked.casefold()

    empty = outcome_line(ExplorationOutcome(found=0, refused=0, unknown=0))
    assert empty


def test_narrator_records_the_outcome_line() -> None:
    from jobbot.ops.narrate import ExplorationOutcome

    written: list[str] = []
    narrator = Narrator(sink=written.append)
    narrator.outcome(ExplorationOutcome(found=1, refused=0, unknown=2))

    assert written and written[0] == narrator.lines[-1]
    assert "1" in written[0]

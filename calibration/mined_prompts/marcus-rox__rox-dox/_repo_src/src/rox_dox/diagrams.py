from __future__ import annotations

import sys
from urllib.parse import quote

from rox_dox.links import source_url
from rox_dox.model import Page, Sequence, Source, StateMachine


MAX_DIAGRAM_ELEMENTS = 12
MAX_BLOCK_NODES = 24
LINK_TOOLTIP = "Open cited source"
SKINPARAMS = """skinparam backgroundColor transparent
skinparam defaultFontName Helvetica
skinparam defaultFontSize 13
skinparam roundCorner 14
skinparam shadowing false
skinparam ArrowColor #333333
skinparam rectangleBackgroundColor #E8F0FE
skinparam rectangleBorderColor #3B6FD8
skinparam databaseBackgroundColor #FFF4E5
skinparam databaseBorderColor #E08A1E
skinparam cloudBackgroundColor #F3E8FD
skinparam cloudBorderColor #8B4FD1
skinparam queueBackgroundColor #E7F6EC
skinparam queueBorderColor #2F9E5B
skinparam participantBackgroundColor #E8F0FE
skinparam participantBorderColor #3B6FD8
skinparam sequenceLifeLineBorderColor #9AA4B2
skinparam stateBackgroundColor #E8F0FE
skinparam stateBorderColor #3B6FD8
skinparam classBackgroundColor #E8F0FE
skinparam classHeaderBackgroundColor #E8F0FE
skinparam classBorderColor #3B6FD8"""
PLANTUML_URL_SAFE_CHARACTERS = "/:#?&=+%"


def _escape_label(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _quoted_label(value: str) -> str:
    return f'"{_escape_label(value)}"'


def _plantuml_url(url: str) -> str:
    return quote(url, safe=PLANTUML_URL_SAFE_CHARACTERS)


def _linked_label(url: str, label: str) -> str:
    escaped_label = (
        _escape_label(label)
        .replace("[", "~[")
        .replace("]", "~]")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )
    return f"[[{_plantuml_url(url)}{{{LINK_TOOLTIP}}} {escaped_label}]]"


def _diagram_header(*, smetana: bool) -> list[str]:
    lines = ["@startuml"]
    if smetana:
        lines.append("!pragma layout smetana")
    lines.extend(SKINPARAMS.splitlines())
    return lines


def _element_url(page: Page, repo_url: str, source: Source) -> str:
    return source_url(source, repo_url=repo_url, commit=page.commit)


def _aliases(ids: list[str], prefix: str) -> dict[str, str]:
    return {element_id: f"{prefix}_{index}" for index, element_id in enumerate(ids)}


def sequence_plantuml(
    page: Page,
    sequence: Sequence,
    *,
    repo_url: str,
) -> str:
    participant_aliases = _aliases(
        [participant.id for participant in sequence.participants],
        "participant",
    )
    lines = _diagram_header(smetana=False)
    lines.extend(
        [
            f"title {_quoted_label(sequence.title)}",
            "autonumber",
        ]
    )
    for participant in sequence.participants:
        alias = participant_aliases[participant.id]
        lines.append(f"participant {_quoted_label(participant.label)} as {alias}")
        url = _element_url(page, repo_url, participant.source)
        lines.append(f"url of {alias} is [[{_plantuml_url(url)}]]")
    for step in sequence.steps:
        url = _element_url(page, repo_url, step.source)
        lines.append(
            f"{participant_aliases[step.src]} -> {participant_aliases[step.dst]} : "
            f"{_linked_label(url, step.message)}"
        )
    lines.append("@enduml")
    return "\n".join(lines)


def state_plantuml(
    page: Page,
    state_machine: StateMachine,
    *,
    repo_url: str,
) -> str:
    state_aliases = _aliases([state.id for state in state_machine.states], "state")
    lines = _diagram_header(smetana=True)
    lines.append(f"title {_quoted_label(state_machine.title)}")
    for state in state_machine.states:
        alias = state_aliases[state.id]
        lines.append(f"state {_quoted_label(state.label)} as {alias}")
        url = _element_url(page, repo_url, state.source)
        lines.append(f"url of {alias} is [[{_plantuml_url(url)}]]")

    if state_machine.states:
        first_state = state_machine.states[0]
        url = _element_url(page, repo_url, first_state.source)
        lines.append(
            f"[*] --> {state_aliases[first_state.id]} : {_linked_label(url, 'start')}"
        )
    for transition in state_machine.transitions:
        url = _element_url(page, repo_url, transition.source)
        lines.append(
            f"{state_aliases[transition.src]} --> {state_aliases[transition.dst]} : "
            f"{_linked_label(url, transition.event)}"
        )
    lines.append("@enduml")
    return "\n".join(lines)


def diagram_size_warnings(page: Page) -> list[str]:
    warnings = []
    if len(page.block.nodes) > MAX_BLOCK_NODES:
        warnings.append(
            f"warning: page '{page.id}' block diagram has {len(page.block.nodes)} nodes "
            f"(guide: {MAX_BLOCK_NODES})"
        )
    warnings.extend(
        f"warning: page '{page.id}' sequence '{sequence.title}' has "
        f"{len(sequence.participants)} participants (guide: {MAX_DIAGRAM_ELEMENTS})"
        for sequence in page.sequences
        if len(sequence.participants) > MAX_DIAGRAM_ELEMENTS
    )
    warnings.extend(
        f"warning: page '{page.id}' state machine '{state_machine.title}' has "
        f"{len(state_machine.states)} states (guide: {MAX_DIAGRAM_ELEMENTS})"
        for state_machine in page.states
        if len(state_machine.states) > MAX_DIAGRAM_ELEMENTS
    )
    return warnings


def emit_diagram_warnings(page: Page) -> None:
    for warning in diagram_size_warnings(page):
        print(warning, file=sys.stderr)

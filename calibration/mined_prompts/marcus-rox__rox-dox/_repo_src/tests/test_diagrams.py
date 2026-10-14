from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from rox_dox.block_svg import block_layout_problems
from rox_dox.diagrams import (
    emit_diagram_warnings,
    sequence_plantuml,
    state_plantuml,
)
from rox_dox.links import source_url
from rox_dox.model import BlockDiagram, Page
from rox_dox.plantuml import DiagramError, render_svg
from rox_dox.schema import extract_tables
from rox_dox.schema_svg import schema_svg

REPO_URL = "https://github.com/Rox-AI/rox-core"


def _page(data: dict[str, object]) -> Page:
    return Page.model_validate(copy.deepcopy(data))


def _links(svg: str) -> set[str]:
    return {
        href
        for href in re.findall(r'(?:xlink:)?href="([^"]+)"', svg)
        if not href.startswith("data:")
    }


def test_node_kind_defaults_to_component(page_data: dict[str, object]) -> None:
    page = _page(page_data)

    assert page.block.nodes[0].kind == "component"
    assert page.block.nodes[0].many is False


def test_authored_integration_layer_has_adjacent_column_edges() -> None:
    pages_path = Path(__file__).resolve().parents[1] / "pages/rox-core.json"
    payload = json.loads(pages_path.read_text(encoding="utf-8"))
    figure_data = next(
        figure
        for figure in payload["block_figures"]
        if figure["id"] == "integration-layer"
    )
    diagram = BlockDiagram.model_validate(figure_data["block"])

    assert block_layout_problems(diagram) == []


def test_authored_many_flags_match_plural_runtime_nodes() -> None:
    pages_path = Path(__file__).resolve().parents[1] / "pages/rox-core.json"
    payload = json.loads(pages_path.read_text(encoding="utf-8"))
    figures = {
        figure["id"]: figure["block"]["nodes"] for figure in payload["block_figures"]
    }

    assert {node["id"] for node in payload["block"]["nodes"] if node["many"]} == {
        "interaction",
        "webhook",
        "agent_workers",
        "data_workers",
        "outreach_sched",
        "temporal_workers",
        "sqs",
    }
    assert {node["id"] for node in figures["task-pipeline"] if node["many"]} == {
        "sqs",
        "executor",
    }
    assert {node["id"] for node in figures["integration-layer"] if node["many"]} == {
        "listener",
        "sqs_aws",
    }
    assert {
        node["id"] for node in figures["connect-google-workspace"] if node["many"]
    } == {"fanout"}
    assert not any(node["many"] for node in figures["chat-turn"])


def test_schema_diagram_links_tables_stores_and_in_scope_foreign_keys(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    page = _page(page_data)
    tables = extract_tables(repo, commit)
    svg = schema_svg(page, repo_url=REPO_URL, tables=tables)

    assert re.search(
        r'<svg[^>]+width="\d+" height="\d+" viewBox="0 0 \d+ \d+"',
        svg,
    )
    assert all(marker in svg for marker in ("users", "sessions", "redis"))
    assert all(
        source_url(tables[name].source, repo_url=REPO_URL, commit=page.commit)
        in _links(svg)
        for name in ("users", "sessions")
    )
    assert source_url(
        page.data.nosql[0].source,
        repo_url=REPO_URL,
        commit=page.commit,
    ) in _links(svg)

    page.data.sql_tables = ["sessions"]
    out_of_scope_svg = schema_svg(page, repo_url=REPO_URL, tables=tables)
    assert 'marker-start="url(#arrow-enforced)"' not in out_of_scope_svg


def test_schema_relations_render_as_linked_dashed_edges(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    relation_source = {"path": "pkg/a.py", "lines": [1, 2]}
    payload["data"]["relations"] = [
        {
            "src": "sessions.user_id",
            "dst": "users.id",
            "label": "implicit foreign key",
            "source": relation_source,
        },
        {
            "src": "sessions.id",
            "dst": "sessions.id",
            "label": "self relation",
            "source": relation_source,
        },
        {
            "src": "cache::not-a-declared-field",
            "dst": "users.id",
            "label": "NoSQL field is free text",
            "source": relation_source,
        },
    ]
    page = _page(payload)
    tables = extract_tables(repo, commit)

    svg = schema_svg(page, repo_url=REPO_URL, tables=tables)
    relation_url = source_url(
        page.data.relations[0].source,
        repo_url=REPO_URL,
        commit=page.commit,
    )

    assert 'stroke-dasharray="6 4"' in svg
    assert "implicit foreign key" in svg
    assert "self relation" in svg
    assert "NoSQL field is free text" in svg
    assert relation_url in _links(svg)


@pytest.mark.parametrize(
    ("src", "dst", "expected"),
    [
        ("outside.id", "users.id", "SQL table 'outside'"),
        ("users.missing", "sessions.id", "SQL column 'users.missing'"),
        ("unknown::field", "users.id", "endpoint 'unknown::field'"),
        ("malformed", "users.id", "endpoint 'malformed'"),
    ],
)
def test_schema_relation_endpoints_must_be_declared(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    src: str,
    dst: str,
    expected: str,
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["relations"] = [
        {
            "src": src,
            "dst": dst,
            "label": "invalid relation",
            "source": {"path": "pkg/a.py", "lines": [1, 2]},
        }
    ]
    page = _page(payload)

    with pytest.raises(DiagramError) as error:
        schema_svg(
            page,
            repo_url=REPO_URL,
            tables=extract_tables(repo, commit),
        )

    message = str(error.value)
    assert "page 'rox-core'" in message
    assert f"relation '{src} -> {dst}'" in message
    assert expected in message


def test_sequence_diagram_autonumbers_and_links_participants_and_steps(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    _, commit = git_repo
    payload = copy.deepcopy(page_data)
    sequence = payload["sequences"][0]
    sequence["participants"][0]["source"]["lines"] = [1, 2]
    sequence["participants"][1]["source"]["lines"] = [2, 3]
    sequence["steps"][0]["source"]["lines"] = [4, 5]
    page = _page(payload)

    source = sequence_plantuml(page, page.sequences[0], repo_url=REPO_URL)
    svg = render_svg(source, plantuml_jar)

    assert "autonumber" in source
    assert "!pragma layout smetana" not in source
    assert all(color in svg for color in ("#E8F0FE", "#3B6FD8", "#9AA4B2"))
    assert {
        source_url(
            participant.source,
            repo_url=REPO_URL,
            commit=commit,
        )
        for participant in page.sequences[0].participants
    }.issubset(_links(svg))
    assert source_url(
        page.sequences[0].steps[0].source,
        repo_url=REPO_URL,
        commit=commit,
    ) in _links(svg)


def test_sequence_step_message_with_braces_renders_literal_label(
    page_data: dict[str, object],
    plantuml_jar: Path,
) -> None:
    payload = copy.deepcopy(page_data)
    payload["sequences"][0]["steps"][0]["message"] = "POST /message/{x}"
    page = _page(payload)
    source = sequence_plantuml(page, page.sequences[0], repo_url=REPO_URL)
    svg = render_svg(source, plantuml_jar)

    assert "POST /message/&#123;x&#125;" in source
    assert "POST /message/{x}" in svg


def test_state_diagram_links_states_initial_arrow_and_transitions(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
    plantuml_jar: Path,
) -> None:
    _, commit = git_repo
    payload = copy.deepcopy(page_data)
    state_machine = payload["states"][0]
    state_machine["states"][0]["source"]["lines"] = [1, 2]
    state_machine["states"][1]["source"]["lines"] = [2, 3]
    state_machine["transitions"][0]["source"]["lines"] = [4, 5]
    page = _page(payload)

    source = state_plantuml(page, page.states[0], repo_url=REPO_URL)
    svg = render_svg(source, plantuml_jar)

    first_state_url = source_url(
        page.states[0].states[0].source,
        repo_url=REPO_URL,
        commit=commit,
    )
    assert f"[*] --> state_0 : [[{first_state_url}" in source
    assert all(color in svg for color in ("#E8F0FE", "#3B6FD8"))
    assert {
        source_url(state.source, repo_url=REPO_URL, commit=commit)
        for state in page.states[0].states
    }.issubset(_links(svg))
    assert source_url(
        page.states[0].transitions[0].source,
        repo_url=REPO_URL,
        commit=commit,
    ) in _links(svg)


def test_large_block_sequence_and_state_diagrams_warn(
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 3]}
    payload["block"]["nodes"].extend(
        {
            "id": f"extra-{index}",
            "label": f"Extra {index}",
            "source": source,
        }
        for index in range(23)
    )
    payload["sequences"][0]["participants"].extend(
        {
            "id": f"participant-{index}",
            "label": f"Participant {index}",
            "source": source,
        }
        for index in range(11)
    )
    payload["states"][0]["states"].extend(
        {
            "id": f"state-{index}",
            "label": f"State {index}",
            "source": source,
        }
        for index in range(11)
    )

    emit_diagram_warnings(_page(payload))

    warning_output = capsys.readouterr().err
    assert "block diagram has 25 nodes (guide: 24)" in warning_output
    assert "sequence 'send message' has 13 participants" in warning_output
    assert "state machine 'message lifecycle' has 13 states" in warning_output


def test_diagram_size_guides_do_not_warn_at_limits(
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 3]}
    payload["block"]["nodes"].extend(
        {
            "id": f"extra-{index}",
            "label": f"Extra {index}",
            "source": source,
        }
        for index in range(22)
    )
    payload["sequences"][0]["participants"].extend(
        {
            "id": f"participant-{index}",
            "label": f"Participant {index}",
            "source": source,
        }
        for index in range(10)
    )
    payload["states"][0]["states"].extend(
        {
            "id": f"state-{index}",
            "label": f"State {index}",
            "source": source,
        }
        for index in range(10)
    )

    emit_diagram_warnings(_page(payload))

    assert capsys.readouterr().err == ""


def test_schema_size_does_not_emit_a_size_warning(
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = [f"table_{index}" for index in range(13)]

    emit_diagram_warnings(_page(payload))

    assert capsys.readouterr().err == ""


def test_invalid_plantuml_syntax_raises_diagram_error(plantuml_jar: Path) -> None:
    with pytest.raises(DiagramError, match="Syntax Error"):
        render_svg("@startuml\nnot valid\n@enduml", plantuml_jar)


def test_missing_plantuml_jar_points_to_fetch_script(tmp_path: Path) -> None:
    with pytest.raises(DiagramError, match="scripts/fetch_plantuml.sh"):
        render_svg("@startuml\n@enduml", tmp_path / "missing.jar")

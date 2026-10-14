from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from rox_dox.links import source_url
from rox_dox.model import Page
from rox_dox.plantuml import DiagramError
from rox_dox.schema import Column, extract_tables
from rox_dox.schema_svg import schema_svg

REPO_URL = "https://github.com/Rox-AI/rox-core"
SOURCE = {"path": "pkg/a.py", "lines": [1, 2]}


def _page(payload: dict[str, object]) -> Page:
    return Page.model_validate(copy.deepcopy(payload))


def _domain_payload(page_data: dict[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["columns"] = [["identity", "cache"]]
    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users", "sessions"],
            "key_tables": ["users"],
            "notes": [
                {
                    "text": "Shared tenant keys are summarized here.",
                    "sources": [SOURCE],
                }
            ],
            "page": "rox-core/identity",
        }
    ]
    payload["data"]["relations"] = []
    return payload


def test_domain_schema_cards_show_key_tables_and_all_tables_in_guide(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    page = _page(_domain_payload(page_data))
    tables = extract_tables(repo, commit)

    svg = schema_svg(page, repo_url=REPO_URL, tables=tables)
    source_href = source_url(
        tables["users"].source,
        repo_url=REPO_URL,
        commit=page.commit,
    )
    note_href = source_url(
        page.data.domains[0].notes[0].sources[0],
        repo_url=REPO_URL,
        commit=page.commit,
    )

    assert svg.count('class="schema-card schema-domain"') == 1
    assert 'width="348" height="266" viewBox="0 0 348 266"' in svg
    assert "Identity" in svg
    assert "2 tables" in svg
    assert svg.count(">users</text>") == 1
    assert "sessions" not in svg
    assert f'href="{source_href}"' in svg
    assert 'href="#schema-guide-identity"' in svg
    assert "Shared tenant" in svg
    assert "summarized" in svg
    assert f'href="{note_href}"' in svg


def test_nosql_store_width_expands_for_long_keys(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = _domain_payload(page_data)
    store_name = "chat:conversation:{conversation_id}:stream:{stream_id}"
    payload["data"]["nosql"][0]["name"] = store_name
    payload["data"]["columns"] = [["identity", store_name]]
    page = _page(payload)

    svg = schema_svg(page, repo_url=REPO_URL, tables=extract_tables(repo, commit))
    store_markup = svg.split('class="schema-card schema-store"', 1)[1].split(
        "</g>",
        1,
    )[0]
    header_lines = re.findall(
        r'<text[^>]*font-size="12.5"[^>]*>(.*?)</text>',
        store_markup,
    )

    assert "".join(header_lines) == store_name
    assert len(header_lines) == 1
    store_width = re.search(r'<rect x="0" y="0" width="([0-9]+)"', store_markup)
    assert store_width is not None
    assert 400 <= int(store_width.group(1)) <= 520


def test_wrapped_header_text_stays_inside_header_box(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = _domain_payload(page_data)
    payload["data"]["domains"][0]["title"] = (
        "Tenancy organizations users authentication billing and permissions"
    )
    page = _page(payload)

    svg = schema_svg(page, repo_url=REPO_URL, tables=extract_tables(repo, commit))
    root = ET.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    domain_card = root.find(
        './/svg:g[@class="schema-card schema-domain"]',
        namespace,
    )
    assert domain_card is not None
    header = next(
        rect
        for rect in domain_card.findall("svg:rect", namespace)
        if rect.get("fill") == "#1f2937"
    )
    header_height = float(header.get("height", "0"))
    header_text = [
        text
        for text in domain_card.iter("{http://www.w3.org/2000/svg}text")
        if text.get("font-size") in {"10.5", "12.5"}
    ]
    header_lines = [
        text
        for text in domain_card.iter("{http://www.w3.org/2000/svg}text")
        if text.get("font-size") == "12.5"
    ]
    first_row_top = min(
        float(line.get("y1", "0"))
        for line in domain_card.findall("svg:line", namespace)
    )

    assert len(header_lines) >= 2
    assert header_text
    assert all(0 < float(text.get("y", "0")) <= header_height for text in header_text)
    assert first_row_top >= header_height


def test_domain_schema_relations_use_kind_styles_and_key_endpoints(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["columns"] = [["sessions"], ["identity", "cache"]]
    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users"],
            "key_tables": ["users"],
        },
        {
            "id": "sessions",
            "title": "Sessions",
            "tables": ["sessions"],
            "key_tables": ["sessions"],
        },
    ]
    payload["data"]["relations"] = [
        {
            "src": "sessions.user_id",
            "dst": "users.id",
            "label": "declared relation",
            "kind": "enforced",
            "source": SOURCE,
        },
        {
            "src": "users.id",
            "dst": "cache::free-form",
            "label": "symbolic relation",
            "kind": "symbolic",
            "source": SOURCE,
        },
        {
            "src": "cache::key",
            "dst": "sessions.id",
            "label": "blob relation",
            "kind": "blob",
            "source": SOURCE,
        },
    ]
    page = _page(payload)

    svg = schema_svg(page, repo_url=REPO_URL, tables=extract_tables(repo, commit))

    assert 'stroke="#2563eb" stroke-width="1.6"' in svg
    assert 'stroke="#9ca3af" stroke-width="1.6" stroke-dasharray="6 4"' in svg
    assert 'stroke="#d97706" stroke-width="1.6" stroke-dasharray="1 4"' in svg
    assert re.search(
        r'<text[^>]+fill="#2563eb"[^>]*>user_id</text>',
        svg,
    )
    assert re.search(r'<text[^>]+fill="#b45309"[^>]*>id</text>', svg)
    for kind, color in (
        ("enforced", "#2563eb"),
        ("symbolic", "#9ca3af"),
        ("blob", "#d97706"),
    ):
        assert f'id="arrow-{kind}"' in svg
        assert f'id="dot-{kind}"' in svg
        assert f'<circle cx="4" cy="4" r="3" fill="{color}"/>' in svg
    assert "users.id → cache::free-form: symbolic relation (symbolic)" in svg
    assert "schema-edge:hover path { stroke-width: 2.8; }" in svg


def test_table_level_relations_connect_sql_table_headers(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["relations"] = [
        {
            "src": "sessions",
            "dst": "users",
            "label": "uses",
            "kind": "enforced",
            "source": SOURCE,
        }
    ]
    page = _page(payload)

    svg = schema_svg(page, repo_url=REPO_URL, tables=extract_tables(repo, commit))

    assert "sessions → users: uses (enforced)" in svg
    assert 'stroke="#2563eb" stroke-width="1.6"' in svg


def test_domain_key_row_truncates_long_relation_column_lists(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["columns"] = [["identity", "sessions"]]
    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users"],
            "key_tables": ["users"],
        },
        {
            "id": "sessions",
            "title": "Sessions",
            "tables": ["sessions"],
            "key_tables": ["sessions"],
        },
    ]
    relation_columns = [f"relation_column_{index}" for index in range(20)]
    payload["data"]["relations"] = [
        {
            "src": f"users.{column_name}",
            "dst": "sessions.id",
            "label": column_name,
            "source": SOURCE,
        }
        for column_name in relation_columns
    ]
    tables = extract_tables(repo, commit)
    users = tables["users"]
    tables["users"] = users.model_copy(
        update={
            "columns": [
                *users.columns,
                *(
                    Column(name=column_name, type="String")
                    for column_name in relation_columns
                ),
            ]
        }
    )

    svg = schema_svg(
        _page(payload),
        repo_url=REPO_URL,
        tables=tables,
    )
    users_card = svg.split('class="schema-card schema-domain"', 1)[1].split(
        "</g>",
        1,
    )[0]

    assert "…" in users_card
    assert "relation_column_19" not in users_card


def test_domain_schema_rejects_enforced_mismatch_and_intra_domain_relations(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users"],
            "key_tables": ["users"],
        },
        {
            "id": "sessions",
            "title": "Sessions",
            "tables": ["sessions"],
            "key_tables": ["sessions"],
        },
    ]
    payload["data"]["relations"] = [
        {
            "src": "sessions.user_id",
            "dst": "users.email",
            "label": "mismatched constraint",
            "kind": "enforced",
            "source": SOURCE,
        }
    ]
    page = _page(payload)

    with pytest.raises(
        DiagramError,
        match="page 'rox-core' relation 'sessions.user_id -> users.email'.*not 'users.email'",
    ):
        schema_svg(page, repo_url=REPO_URL, tables=extract_tables(repo, commit))

    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users", "sessions"],
            "key_tables": ["users"],
        }
    ]
    payload["data"]["relations"][0].update(
        {
            "src": "sessions.id",
            "dst": "users.id",
            "kind": "symbolic",
            "label": "non-key endpoint",
        }
    )
    with pytest.raises(
        DiagramError,
        match="page 'rox-core' relation 'sessions.id -> users.id'.*not listed",
    ):
        schema_svg(
            _page(payload),
            repo_url=REPO_URL,
            tables=extract_tables(repo, commit),
        )

    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users", "sessions"],
            "key_tables": ["users", "sessions"],
        }
    ]
    payload["data"]["relations"][0].update(
        {
            "src": "sessions.id",
            "dst": "users.id",
            "kind": "symbolic",
            "label": "intra-domain",
        }
    )
    with pytest.raises(
        DiagramError,
        match="page 'rox-core'.*intra-domain relation belongs on the domain page",
    ):
        schema_svg(
            _page(payload),
            repo_url=REPO_URL,
            tables=extract_tables(repo, commit),
        )


def test_table_schema_styles_columns_and_draws_declared_foreign_keys(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    page_payload = copy.deepcopy(page_data)
    page_payload["data"]["sql_tables"] = ["sessions", "users"]
    tables = extract_tables(repo, commit)
    tables["sessions"] = tables["sessions"].model_copy(
        update={
            "columns": [
                *tables["sessions"].columns,
                Column(name="public_id", type="String"),
                Column(name="created_at", type="DateTime"),
                Column(name="rox_org_id", type="String"),
            ]
        }
    )
    page_payload["data"]["relations"] = [
        {
            "src": "sessions.user_id",
            "dst": "users.id",
            "label": "same declared FK",
            "kind": "enforced",
            "source": SOURCE,
        }
    ]
    page = _page(page_payload)

    svg = schema_svg(page, repo_url=REPO_URL, tables=tables)

    assert 'fill="#b45309" font-family="Menlo, Consolas, monospace"' in svg
    assert 'fill="#2563eb" font-family="Menlo, Consolas, monospace"' in svg
    assert 'fill="#9ca3af" font-family="Menlo, Consolas, monospace"' in svg
    assert re.search(r'<text[^>]+fill="#9ca3af"[^>]*>id</text>', svg)
    assert svg.count('marker-start="url(#arrow-enforced)"') == 1
    assert "sessions.user_id → users.id: same declared FK (enforced)" in svg


def test_same_column_relations_use_distinct_right_side_routes(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    repo, commit = git_repo
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = ["sessions"]
    payload["data"]["nosql"] = []
    payload["data"]["relations"] = [
        {
            "src": "sessions.id",
            "dst": "sessions.id",
            "label": f"self relation {index}",
            "source": SOURCE,
        }
        for index in range(2)
    ]
    svg = schema_svg(
        _page(payload),
        repo_url=REPO_URL,
        tables=extract_tables(repo, commit),
    )

    assert 'width="422"' in svg
    assert "C 384 " in svg
    assert "C 398 " in svg

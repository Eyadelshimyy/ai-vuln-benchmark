import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rox_dox.cli import main
from rox_dox.model import Page, page_sources
from rox_dox.schema import extract_tables


def test_valid_page_passes_check_command(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    (pages_dir / "root.json").write_text(json.dumps(page_data), encoding="utf-8")

    exit_code = main(["check", str(pages_dir), "--repo", str(git_repo[0])])

    assert exit_code == 0
    assert "1 pages OK" in capsys.readouterr().out


def test_edge_endpoint_must_name_declared_node_or_group(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["dst"] = "cache"

    with pytest.raises(
        ValidationError, match="block edge api->cache.*unknown endpoint"
    ):
        Page.model_validate(page_data)


def test_domain_page_requires_at_least_one_path(
    page_data: dict[str, object],
) -> None:
    page_data.update(
        {
            "id": "domain",
            "kind": "domain",
            "parent": "rox-core",
            "paths": [],
        }
    )

    with pytest.raises(ValidationError):
        Page.model_validate(page_data)


def test_group_can_be_an_edge_endpoint(page_data: dict[str, object]) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    source = {"path": "pkg/a.py", "lines": [1, 3]}
    block["groups"] = [
        {"id": "backend", "label": "Backend", "source": source},
    ]
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["group"] = "backend"
    edges = block["edges"]
    assert isinstance(edges, list)
    edge = edges[0]
    assert isinstance(edge, dict)
    edge["src"] = "backend"

    page = Page.model_validate(page_data)

    assert page.block.edges[0].src == "backend"


def test_queue_with_one_flow_side_is_valid(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    nodes[1]["kind"] = "queue"

    page = Page.model_validate(page_data)

    assert page.block.nodes[1].kind == "queue"


def test_queue_without_any_flow_side_is_invalid(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    nodes[1]["kind"] = "queue"
    block["edges"] = []

    with pytest.raises(
        ValidationError,
        match="block diagram: queue 'store' needs a producer or a consumer edge",
    ):
        Page.model_validate(page_data)


def test_node_ids_must_be_unique(page_data: dict[str, object]) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    duplicate_node = dict(nodes[0])
    nodes.append(duplicate_node)

    with pytest.raises(ValidationError, match="duplicate node id 'api'"):
        Page.model_validate(page_data)


def test_node_cannot_reference_an_unknown_group(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["group"] = "missing"

    with pytest.raises(ValidationError, match="node 'api'.*unknown group 'missing'"):
        Page.model_validate(page_data)


def test_group_parent_must_reference_a_declared_group(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
            "parent": "missing",
        }
    ]

    with pytest.raises(
        ValidationError,
        match="group 'backend'.*unknown parent 'missing'",
    ):
        Page.model_validate(page_data)


def test_group_parent_cycles_are_rejected(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
            "parent": "workers",
        },
        {
            "id": "workers",
            "label": "Workers",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
            "parent": "backend",
        },
    ]
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    nodes[0]["group"] = "backend"
    nodes[1]["group"] = "workers"

    with pytest.raises(ValidationError, match="group parent cycle"):
        Page.model_validate(page_data)


def test_group_without_nodes_or_children_is_rejected(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "empty",
            "label": "Empty",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        }
    ]

    with pytest.raises(
        ValidationError,
        match="group 'empty' has no nodes or child groups",
    ):
        Page.model_validate(page_data)


def test_group_ids_must_be_unique(page_data: dict[str, object]) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        },
        {
            "id": "backend",
            "label": "Other backend",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        },
    ]

    with pytest.raises(ValidationError, match="duplicate group id 'backend'"):
        Page.model_validate(page_data)


def test_group_and_node_ids_must_be_unique(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    block["groups"] = [
        {
            "id": "api",
            "label": "API group",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        }
    ]

    with pytest.raises(ValidationError, match="duplicate group/node id 'api'"):
        Page.model_validate(page_data)


def test_block_figure_ids_must_be_unique(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    source = payload["block"]["nodes"][0]["source"]
    figure = {
        "id": "task-pipeline",
        "title": "Background task pipeline",
        "block": {
            "nodes": [{"id": "caller", "label": "Caller", "source": source}],
            "edges": [],
        },
    }
    payload["block_figures"] = [figure, copy.deepcopy(figure)]

    with pytest.raises(ValidationError, match="duplicate block figure id"):
        Page.model_validate(payload)


def test_block_figure_edges_must_reference_declared_node_or_group(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = payload["block"]["nodes"][0]["source"]
    payload["block_figures"] = [
        {
            "id": "task-pipeline",
            "title": "Background task pipeline",
            "block": {
                "nodes": [{"id": "caller", "label": "Caller", "source": source}],
                "edges": [
                    {
                        "src": "caller",
                        "dst": "missing",
                        "label": "unknown",
                        "source": source,
                    }
                ],
            },
        }
    ]

    with pytest.raises(
        ValidationError,
        match="block edge caller->missing: unknown endpoint 'missing'",
    ):
        Page.model_validate(payload)


def test_node_details_are_limited_to_seven(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["details"] = [
        {
            "text": f"Detail {detail_number}",
            "sources": [{"path": "pkg/a.py", "lines": [1, 3]}],
        }
        for detail_number in range(1, 8)
    ]
    Page.model_validate(page_data)
    node["details"].append(
        {
            "text": "Detail 8",
            "sources": [{"path": "pkg/a.py", "lines": [1, 3]}],
        }
    )

    with pytest.raises(ValidationError, match="block node 'api' has 8 details"):
        Page.model_validate(page_data)


def test_node_detail_text_is_limited_to_ninety_characters(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["details"] = [
        {
            "text": "x" * 91,
            "sources": [{"path": "pkg/a.py", "lines": [1, 3]}],
        }
    ]

    with pytest.raises(
        ValidationError,
        match="block node 'api' detail 1: text length 91",
    ):
        Page.model_validate(page_data)


def test_node_detail_text_accepts_ninety_characters(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["details"] = [
        {
            "text": "x" * 90,
            "sources": [{"path": "pkg/a.py", "lines": [1, 3]}],
        }
    ]

    page = Page.model_validate(page_data)

    assert len(page.block.nodes[0].details[0].text) == 90


def test_node_detail_text_cannot_be_empty(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["details"] = [
        {
            "text": "",
            "sources": [{"path": "pkg/a.py", "lines": [1, 3]}],
        }
    ]

    with pytest.raises(
        ValidationError,
        match="block node 'api' detail 1: text length 0",
    ):
        Page.model_validate(page_data)


def test_page_sources_include_group_and_node_detail_citations(
    page_data: dict[str, object],
) -> None:
    block = page_data["block"]
    assert isinstance(block, dict)
    source = {"path": "pkg/a.py", "lines": [1, 3]}
    block["groups"] = [
        {
            "id": "backend",
            "label": "Backend",
            "source": source,
        }
    ]
    nodes = block["nodes"]
    assert isinstance(nodes, list)
    node = nodes[0]
    assert isinstance(node, dict)
    node["group"] = "backend"
    node["details"] = [{"text": "Handles requests", "sources": [source]}]
    page = Page.model_validate(page_data)

    sources = page_sources(page)

    assert ("block group backend", page.block.groups[0].source) in sources
    assert (
        "block node api detail 1",
        page.block.nodes[0].details[0].sources[0],
    ) in sources


def test_page_sources_include_membership_row_citations(
    page_data: dict[str, object],
) -> None:
    source = {"path": "pkg/a.py", "lines": [2, 3]}
    page_data["membership"] = [
        {
            "layer": "Web screens",
            "rows": [
                {
                    "path": "web/src/page.tsx",
                    "primary": True,
                    "evidence": "route tag campaigns",
                    "sources": [source],
                }
            ],
        }
    ]
    page = Page.model_validate(page_data)

    assert (
        "membership Web screens file web/src/page.tsx",
        page.membership[0].rows[0].sources[0],
    ) in page_sources(page)


def test_page_sources_include_focused_block_figure_elements(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [1, 3]}
    payload["block_figures"] = [
        {
            "id": "task-pipeline",
            "title": "Background task pipeline",
            "notes": [{"text": "Based on the task pattern.", "sources": [source]}],
            "block": {
                "groups": [
                    {"id": "worker", "label": "Worker", "source": source},
                ],
                "nodes": [
                    {
                        "id": "listener",
                        "label": "Listener",
                        "source": source,
                        "group": "worker",
                        "details": [{"text": "Polls SQS.", "sources": [source]}],
                    },
                    {"id": "handler", "label": "Handler", "source": source},
                ],
                "edges": [
                    {
                        "src": "listener",
                        "dst": "worker",
                        "label": "execute",
                        "source": source,
                    }
                ],
            },
        }
    ]
    page = Page.model_validate(payload)

    sources = page_sources(page)
    labels = {label for label, _ in sources}

    assert {
        "block figure task-pipeline note 1",
        "block figure task-pipeline group worker",
        "block figure task-pipeline node listener",
        "block figure task-pipeline node listener detail 1",
        "block figure task-pipeline edge listener->worker",
    } <= labels


def test_schema_domains_require_key_tables_to_be_members(
    page_data: dict[str, object],
) -> None:
    page_data["data"]["sql_tables"] = []
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["sessions"],
        }
    ]

    with pytest.raises(ValidationError, match="schema domain 'people'.*key tables"):
        Page.model_validate(page_data)


def test_schema_domain_key_tables_must_not_repeat(
    page_data: dict[str, object],
) -> None:
    page_data["data"]["sql_tables"] = []
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["users", "users"],
        }
    ]

    with pytest.raises(
        ValidationError,
        match="schema domain 'people': duplicate key table 'users'",
    ):
        Page.model_validate(page_data)


def test_schema_domain_ids_must_be_unique(page_data: dict[str, object]) -> None:
    page_data["data"]["sql_tables"] = []
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["users"],
        },
        {
            "id": "people",
            "title": "Other people",
            "tables": ["sessions"],
            "key_tables": ["sessions"],
        },
    ]

    with pytest.raises(ValidationError, match="schema: duplicate domain id 'people'"):
        Page.model_validate(page_data)


def test_schema_table_can_belong_to_only_one_domain(
    page_data: dict[str, object],
) -> None:
    page_data["data"]["sql_tables"] = []
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["users"],
        },
        {
            "id": "sessions",
            "title": "Sessions",
            "tables": ["users"],
            "key_tables": ["users"],
        },
    ]

    with pytest.raises(ValidationError, match="table 'users' belongs to both domain"):
        Page.model_validate(page_data)


def test_schema_domains_and_table_view_are_mutually_exclusive(
    page_data: dict[str, object],
) -> None:
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["users"],
        }
    ]

    with pytest.raises(
        ValidationError, match="domains and sql_tables are mutually exclusive"
    ):
        Page.model_validate(page_data)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ([["unknown"]], "unknown columns item 'unknown'"),
        ([["users"], ["users"]], "duplicate columns item 'users'"),
    ],
)
def test_schema_columns_must_be_declared_and_unique(
    page_data: dict[str, object],
    columns: list[list[str]],
    message: str,
) -> None:
    page_data["data"]["columns"] = columns

    with pytest.raises(ValidationError, match=message):
        Page.model_validate(page_data)


def test_page_sources_include_domain_note_citations(
    page_data: dict[str, object],
) -> None:
    page_data["data"]["sql_tables"] = []
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["users"],
            "notes": [
                {
                    "text": "The domain has shared tenancy fields.",
                    "sources": [{"path": "pkg/a.py", "lines": [1, 2]}],
                }
            ],
        }
    ]
    page = Page.model_validate(page_data)

    assert (
        "schema domain people note 1",
        page.data.domains[0].notes[0].sources[0],
    ) in page_sources(page)


def test_page_sources_include_key_table_model_citations(
    page_data: dict[str, object],
    git_repo: tuple[Path, str],
) -> None:
    page_data["data"]["sql_tables"] = []
    page_data["data"]["domains"] = [
        {
            "id": "people",
            "title": "People",
            "tables": ["users"],
            "key_tables": ["users"],
        }
    ]
    page = Page.model_validate(page_data)
    tables = extract_tables(*git_repo)

    assert (
        "schema domain people key table users",
        tables["users"].source,
    ) in page_sources(page, tables=tables)


def test_summary_table_rows_must_match_column_count(
    page_data: dict[str, object],
) -> None:
    tldr = page_data["tldr"]
    assert isinstance(tldr, dict)
    table = tldr["table"]
    assert isinstance(table, dict)
    table["rows"] = [["API", "extra"]]

    with pytest.raises(ValidationError, match="row 1 has 2 cells"):
        Page.model_validate(page_data)


def test_related_item_requires_exactly_one_target(
    page_data: dict[str, object],
) -> None:
    related = page_data["related"]
    assert isinstance(related, list)
    related.append(
        {
            "label": "Storage",
            "page": "storage",
            "url": "https://example.com/storage",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        }
    )

    with pytest.raises(ValidationError, match="exactly one of page or url"):
        Page.model_validate(page_data)


def test_page_kind_is_required(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    del payload["kind"]

    with pytest.raises(ValidationError, match="kind"):
        Page.model_validate(payload)


@pytest.mark.parametrize(
    "paths",
    [[], [""], ["/absolute"], ["pkg/../private"], ["pkg/"]],
)
def test_page_paths_must_be_nonempty_and_repo_relative(
    page_data: dict[str, object],
    paths: list[str],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["paths"] = paths

    with pytest.raises(ValidationError):
        Page.model_validate(payload)


def test_page_rejects_legacy_notion_field(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    payload["notion"] = []

    with pytest.raises(ValidationError, match="notion"):
        Page.model_validate(payload)


def test_relation_source_is_enumerated_for_citation_checks(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["data"]["relations"] = [
        {
            "src": "sessions.user_id",
            "dst": "users.id",
            "label": "implicit foreign key",
            "source": {"path": "pkg/a.py", "lines": [4, 5]},
        }
    ]
    page = Page.model_validate(payload)

    assert (
        "relation sessions.user_id -> users.id",
        page.data.relations[0].source,
    ) in page_sources(page)


def test_page_block_note_source_is_enumerated_for_citation_checks(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    source = {"path": "pkg/a.py", "lines": [4, 5]}
    payload["block"]["notes"] = [
        {"text": "Runtime flow note.", "sources": [source]}
    ]
    page = Page.model_validate(payload)

    assert ("block note 1", page.block.notes[0].sources[0]) in page_sources(page)

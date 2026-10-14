from __future__ import annotations

import copy
import json
from pathlib import Path

from rox_dox.cli import main
from rox_dox.model import Page
from rox_dox.tree import build_tree, tree_problems


def test_duplicate_page_ids_are_reported(page_data: dict[str, object]) -> None:
    duplicate = copy.deepcopy(page_data)

    problems = tree_problems(
        [Page.model_validate(page_data), Page.model_validate(duplicate)]
    )

    assert "duplicate page id 'rox-core'" in problems


def test_zero_root_pages_are_reported(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    payload["id"] = "child"
    payload["kind"] = "feature"
    payload["parent"] = "missing"
    payload["paths"] = ["pkg"]

    problems = tree_problems([Page.model_validate(payload)])

    assert "expected exactly one root page, found 0" in problems


def test_multiple_root_pages_are_reported(page_data: dict[str, object]) -> None:
    second_root = copy.deepcopy(page_data)
    second_root["id"] = "second-root"
    second_root["title"] = "Second root"

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(second_root),
        ]
    )

    assert "expected exactly one root page, found 2" in problems


def test_missing_parent_names_child_and_parent(page_data: dict[str, object]) -> None:
    child = copy.deepcopy(page_data)
    child["id"] = "child"
    child["kind"] = "feature"
    child["parent"] = "missing"
    child["paths"] = ["pkg"]

    problems = tree_problems([Page.model_validate(child)])

    assert "page 'child' has missing parent 'missing'" in problems


def test_parent_cycles_are_reported(page_data: dict[str, object]) -> None:
    first = copy.deepcopy(page_data)
    first["id"] = "first"
    first["kind"] = "feature"
    first["parent"] = "second"
    first["paths"] = ["pkg"]
    second = copy.deepcopy(page_data)
    second["id"] = "second"
    second["kind"] = "feature"
    second["parent"] = "first"
    second["paths"] = ["pkg/sub"]

    problems = tree_problems([Page.model_validate(first), Page.model_validate(second)])

    assert any(problem.startswith("parent cycle:") for problem in problems)


def test_missing_block_node_page_link_is_reported(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["block"]["nodes"][0]["link"] = "missing"

    problems = tree_problems([Page.model_validate(payload)])

    assert "block node api: link to missing page 'missing'" in problems


def test_missing_related_page_link_is_reported(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["related"] = [
        {
            "label": "Missing module",
            "page": "missing",
            "source": {"path": "pkg/a.py", "lines": [1, 3]},
        }
    ]

    problems = tree_problems([Page.model_validate(payload)])

    assert "related 'Missing module': link to missing page 'missing'" in problems


def test_missing_schema_domain_page_link_is_reported(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["data"]["sql_tables"] = []
    payload["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users"],
            "key_tables": ["users"],
            "page": "missing",
        }
    ]

    problems = tree_problems([Page.model_validate(payload)])

    assert "schema domain 'identity': link to missing page 'missing'" in problems


def test_missing_additional_tldr_table_page_link_is_reported(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["tldr"]["additional_tables"] = [
        {
            "title": "Links to other domains",
            "columns": ["This table", "Other table", "Other domain", "Signal"],
            "rows": [["sessions", "users", "People", "ID column name"]],
            "links": [{"row": 0, "column": 2, "page": "domain-people"}],
            "sources": [{"path": "pkg/a.py", "lines": [1, 1]}],
        }
    ]

    problems = tree_problems([Page.model_validate(payload)])

    assert "TLDR table: link to missing page 'domain-people'" in problems


def test_schema_domain_page_link_must_match_page_kind(
    page_data: dict[str, object],
) -> None:
    root = copy.deepcopy(page_data)
    root["kind"] = "root"
    root["data"]["sql_tables"] = []
    root["data"]["domains"] = [
        {
            "id": "identity",
            "title": "Identity",
            "tables": ["users"],
            "key_tables": ["users"],
            "page": "feature",
        }
    ]
    feature = copy.deepcopy(page_data)
    feature["id"] = "feature"
    feature["kind"] = "feature"
    feature["parent"] = root["id"]
    feature["paths"] = ["pkg"]

    problems = tree_problems([Page.model_validate(root), Page.model_validate(feature)])

    assert (
        "schema domain 'identity': page 'feature' has kind 'feature'; expected 'domain'"
    ) in problems


def test_missing_summary_table_page_link_is_reported(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["tldr"]["table"]["links"] = [{"row": 0, "column": 0, "page": "missing"}]

    problems = tree_problems([Page.model_validate(payload)])

    assert "TLDR table: link to missing page 'missing'" in problems


def test_root_domain_feature_kinds_form_a_valid_tree(
    page_data: dict[str, object],
) -> None:
    root = copy.deepcopy(page_data)
    root["kind"] = "root"
    domain = copy.deepcopy(page_data)
    domain["id"] = "domain-seq"
    domain["kind"] = "domain"
    domain["parent"] = "rox-core"
    domain["paths"] = ["pkg"]
    feature = copy.deepcopy(page_data)
    feature["id"] = "feature-seq-sequence"
    feature["kind"] = "feature"
    feature["parent"] = "domain-seq"
    feature["paths"] = ["pkg/a.py"]

    problems = tree_problems(
        [
            Page.model_validate(root),
            Page.model_validate(domain),
            Page.model_validate(feature),
        ]
    )

    assert problems == []


def test_feature_map_pages_can_share_paths(
    page_data: dict[str, object],
) -> None:
    root = copy.deepcopy(page_data)
    root["kind"] = "root"
    domain = copy.deepcopy(page_data)
    domain["id"] = "domain-a"
    domain["kind"] = "domain"
    domain["parent"] = "rox-core"
    domain["paths"] = ["pkg/shared.py"]
    features = []
    for feature_id in ("feature-a", "feature-b"):
        feature = copy.deepcopy(page_data)
        feature["id"] = feature_id
        feature["kind"] = "feature"
        feature["parent"] = "domain-a"
        feature["paths"] = ["pkg/shared.py"]
        features.append(Page.model_validate(feature))

    problems = tree_problems(
        [Page.model_validate(root), Page.model_validate(domain), *features]
    )

    assert problems == []


def test_root_paths_must_be_exactly_the_repository_root(
    page_data: dict[str, object],
) -> None:
    payload = copy.deepcopy(page_data)
    payload["paths"] = ["pkg"]

    problems = tree_problems([Page.model_validate(payload)])

    assert "root page 'rox-core' paths must be exactly ['.']" in problems


def test_domain_must_have_root_parent(
    page_data: dict[str, object],
) -> None:
    feature = copy.deepcopy(page_data)
    feature["id"] = "feature"
    feature["kind"] = "feature"
    feature["parent"] = "rox-core"
    feature["paths"] = ["pkg"]
    domain = copy.deepcopy(page_data)
    domain["id"] = "domain"
    domain["kind"] = "domain"
    domain["parent"] = "feature"
    domain["paths"] = ["pkg"]

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(feature),
            Page.model_validate(domain),
        ]
    )

    assert any(
        "page 'domain' has kind 'domain' but parent 'feature' has kind 'feature'"
        in problem
        for problem in problems
    )


def test_feature_cannot_be_a_direct_child_of_root(
    page_data: dict[str, object],
) -> None:
    feature = copy.deepcopy(page_data)
    feature["id"] = "feature"
    feature["kind"] = "feature"
    feature["parent"] = "rox-core"
    feature["paths"] = ["pkg"]

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(feature),
        ]
    )

    assert (
        "page 'feature' has kind 'feature' but parent 'rox-core' has kind 'root'; "
        "features require a domain or feature parent"
    ) in problems


def test_non_root_page_cannot_have_root_kind(page_data: dict[str, object]) -> None:
    nested_root = copy.deepcopy(page_data)
    nested_root["id"] = "nested-root"
    nested_root["title"] = "Nested root"
    nested_root["parent"] = "rox-core"
    nested_root["paths"] = ["pkg"]

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(nested_root),
        ]
    )

    assert (
        "page 'nested-root' has kind 'root' but parent 'rox-core' has kind 'root'"
    ) in problems


def test_root_page_must_have_root_kind(page_data: dict[str, object]) -> None:
    payload = copy.deepcopy(page_data)
    payload["kind"] = "domain"

    problems = tree_problems([Page.model_validate(payload)])

    assert (
        "page 'rox-core' has kind 'domain' with parent kind 'none'; "
        "root pages must have kind 'root'"
    ) in problems


def test_feature_paths_can_be_outside_parent_paths(
    page_data: dict[str, object],
) -> None:
    domain = copy.deepcopy(page_data)
    domain["id"] = "domain"
    domain["kind"] = "domain"
    domain["parent"] = "rox-core"
    domain["paths"] = ["pkg"]
    feature = copy.deepcopy(page_data)
    feature["id"] = "feature"
    feature["kind"] = "feature"
    feature["parent"] = "domain"
    feature["paths"] = ["other/thing"]

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(domain),
            Page.model_validate(feature),
        ]
    )

    assert problems == []


def test_features_can_share_a_path(page_data: dict[str, object]) -> None:
    domain = copy.deepcopy(page_data)
    domain["id"] = "domain"
    domain["kind"] = "domain"
    domain["parent"] = "rox-core"
    domain["paths"] = ["pkg"]
    first = copy.deepcopy(page_data)
    first["id"] = "first"
    first["title"] = "First"
    first["kind"] = "feature"
    first["parent"] = "domain"
    first["paths"] = ["other/thing"]
    second = copy.deepcopy(first)
    second["id"] = "second"
    second["title"] = "Second"

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(domain),
            Page.model_validate(first),
            Page.model_validate(second),
        ]
    )

    assert problems == []


def test_feature_can_have_a_feature_parent(page_data: dict[str, object]) -> None:
    domain = copy.deepcopy(page_data)
    domain["id"] = "domain"
    domain["kind"] = "domain"
    domain["parent"] = "rox-core"
    domain["paths"] = ["pkg"]
    parent_feature = copy.deepcopy(page_data)
    parent_feature["id"] = "parent-feature"
    parent_feature["kind"] = "feature"
    parent_feature["parent"] = "domain"
    parent_feature["paths"] = ["other/thing"]
    child_feature = copy.deepcopy(page_data)
    child_feature["id"] = "child-feature"
    child_feature["kind"] = "feature"
    child_feature["parent"] = "parent-feature"
    child_feature["paths"] = ["pkg/sub"]

    problems = tree_problems(
        [
            Page.model_validate(page_data),
            Page.model_validate(domain),
            Page.model_validate(parent_feature),
            Page.model_validate(child_feature),
        ]
    )

    assert problems == []


def test_tree_orders_children_and_returns_root_first_ancestors(
    page_data: dict[str, object],
) -> None:
    middle = copy.deepcopy(page_data)
    middle["id"] = "middle"
    middle["kind"] = "domain"
    middle["parent"] = "rox-core"
    middle["paths"] = ["pkg"]
    first = copy.deepcopy(page_data)
    first["id"] = "z-page"
    first["title"] = "Same title"
    first["kind"] = "feature"
    first["parent"] = "middle"
    first["paths"] = ["pkg/z"]
    second = copy.deepcopy(first)
    second["id"] = "a-page"
    second["paths"] = ["pkg/a"]
    root = Page.model_validate(page_data)
    pages = [
        root,
        Page.model_validate(middle),
        Page.model_validate(first),
        Page.model_validate(second),
    ]

    tree = build_tree(pages)

    assert tree.children_of("middle") == ["a-page", "z-page"]
    assert tree.ancestors("a-page") == ["rox-core", "middle"]


def test_check_reports_tree_problems(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    capsys,
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    payload = copy.deepcopy(page_data)
    payload["paths"] = ["pkg"]
    (pages_dir / "root.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["check", str(pages_dir), "--repo", str(git_repo[0])])

    assert exit_code == 1
    assert "root page 'rox-core' paths must be exactly ['.']" in (
        capsys.readouterr().out
    )


def test_build_tree_problem_writes_nothing(
    tmp_path: Path,
    git_repo: tuple[Path, str],
    page_data: dict[str, object],
    plantuml_jar: Path,
    capsys,
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    output_dir = tmp_path / "site"
    payload = copy.deepcopy(page_data)
    payload["paths"] = ["pkg"]
    payload["data"]["sql_tables"] = []
    (pages_dir / "root.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "build",
            str(pages_dir),
            "--repo",
            str(git_repo[0]),
            "--repo-url",
            "https://github.com/Rox-AI/rox-core",
            "--out",
            str(output_dir),
            "--plantuml-jar",
            str(plantuml_jar),
        ]
    )

    assert exit_code == 1
    assert "root page 'rox-core' paths must be exactly ['.']" in (
        capsys.readouterr().out
    )
    assert not output_dir.exists()

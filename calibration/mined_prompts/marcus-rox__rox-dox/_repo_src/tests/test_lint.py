from __future__ import annotations

import json
from pathlib import Path

import pytest

import rox_dox.cli as cli
import rox_dox.lint as lint
from rox_dox.model import Page


def _source() -> dict[str, object]:
    return {"path": "pkg/a.py", "lines": [1, 2]}


def _page_data(
    *,
    page_id: str = "domain-activity",
    node_count: int = 3,
    flow_note_count: int = 3,
) -> dict[str, object]:
    source = _source()
    return {
        "id": page_id,
        "title": "Activity",
        "kind": "feature"
        if page_id.startswith("feature-")
        else ("root" if page_id == "rox-core" else "domain"),
        "commit": "a" * 40,
        "parent": None,
        "paths": ["pkg"],
        "tldr": {
            "summary": [{"text": "Summary.", "sources": [source]}],
            "key_points": [],
            "table": {
                "columns": ["Component"],
                "rows": [["API"]],
                "sources": [source],
            },
            "notes": [],
        },
        "block": {
            "notes": [
                {"text": f"Flow note {index}.", "sources": [source]}
                for index in range(flow_note_count)
            ],
            "groups": [],
            "nodes": [
                {
                    "id": f"component-{index}",
                    "label": f"Component {index}",
                    "source": source,
                }
                for index in range(node_count)
            ],
            "edges": [],
        },
        "data": {
            "sql_tables": ["users", "sessions"],
            "nosql": [
                {
                    "name": "cache",
                    "kind": "redis",
                    "fields": ["key", "value"],
                    "source": source,
                }
            ],
        },
        "sequences": [],
        "states": [],
        "related": [],
    }


def _page(**kwargs: object) -> Page:
    return Page.model_validate(_page_data(**kwargs))


def _component_ids(page: Page) -> frozenset[str]:
    return frozenset(node.id for node in page.block.nodes)


def _edge(src: str, dst: str, label: str) -> dict[str, object]:
    return {"src": src, "dst": dst, "label": label, "source": _source()}


def _relation(src: str, dst: str, label: str) -> dict[str, object]:
    return {"src": src, "dst": dst, "label": label, "source": _source()}


def _rules(page: Page, component_ids: frozenset[str]) -> list[str]:
    return [finding.rule for finding in lint.lint_page(page, component_ids)]


def test_b1_errors_when_block_has_more_than_fifteen_boxes() -> None:
    page = _page(node_count=16)

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B1"]
    assert findings[0].severity == "error"


def test_b1_warns_when_block_has_fewer_than_three_boxes() -> None:
    page = _page(node_count=2)

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B1"]
    assert findings[0].severity == "warning"


def test_b2_rejects_generated_node_ids_outside_the_component_catalog() -> None:
    payload = _page_data()
    payload["block"]["nodes"][0]["id"] = "unregistered"
    page = Page.model_validate(payload)

    findings = lint.lint_page(page, frozenset({"component-1", "component-2"}))

    assert [finding.rule for finding in findings] == ["B2"]
    assert "unregistered" in findings[0].message


def test_b3_errors_when_edge_count_exceeds_density_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _page_data()
    payload["block"]["edges"] = [
        _edge(f"component-{index % 3}", f"component-{(index + 1) % 3}", f"flow-{index}")
        for index in range(6)
    ]
    page = Page.model_validate(payload)
    monkeypatch.setattr(lint, "block_layout_problems", lambda _diagram: [])

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B3"]
    assert "6 edges for 3 boxes" in findings[0].message
    assert "limit is 5" in findings[0].message


def test_b4_flags_backward_edges_using_renderer_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _page_data()
    source = _source()
    payload["block"]["groups"] = [
        {"id": "left", "label": "Left", "source": source},
        {"id": "right", "label": "Right", "source": source},
    ]
    payload["block"]["nodes"][0]["group"] = "left"
    payload["block"]["nodes"][1]["group"] = "right"
    payload["block"]["edges"] = [_edge("component-1", "component-0", "back")]
    page = Page.model_validate(payload)
    monkeypatch.setattr(lint, "block_layout_problems", lambda _diagram: [])

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B4"]
    assert "columns 1 -> 0" in findings[0].message


@pytest.mark.parametrize(
    ("problems", "expected_severities"),
    [
        (["one layout issue"], ["warning"]),
        ([f"layout issue {index}" for index in range(6)], ["warning"] * 6 + ["error"]),
    ],
)
def test_b5_reports_layout_problems_and_limit(
    monkeypatch: pytest.MonkeyPatch,
    problems: list[str],
    expected_severities: list[str],
) -> None:
    page = _page()
    calls = []

    def layout_problems(_diagram: object) -> list[str]:
        calls.append(None)
        return problems

    monkeypatch.setattr(lint, "block_layout_problems", layout_problems)

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B5"] * len(expected_severities)
    assert [finding.severity for finding in findings] == expected_severities
    assert len(calls) == 1


def test_b6_rejects_empty_edge_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _page_data()
    payload["block"]["edges"] = [_edge("component-0", "component-1", "  ")]
    page = Page.model_validate(payload)
    monkeypatch.setattr(lint, "block_layout_problems", lambda _diagram: [])

    assert _rules(page, _component_ids(page)) == ["B6"]


def test_b6_rejects_duplicate_edges_once_per_duplicate_triple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _page_data()
    payload["block"]["edges"] = [
        _edge("component-0", "component-1", "calls"),
        _edge("component-0", "component-1", "calls"),
    ]
    page = Page.model_validate(payload)
    monkeypatch.setattr(lint, "block_layout_problems", lambda _diagram: [])

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B6"]
    assert "duplicate edge" in findings[0].message


def test_b7_warns_when_generated_block_has_too_few_flow_notes() -> None:
    page = _page(flow_note_count=2)

    assert _rules(page, _component_ids(page)) == ["B7"]


def test_s1_rejects_same_file_relation_labels() -> None:
    payload = _page_data()
    payload["data"]["relations"] = [
        _relation("users.id", "sessions.user_id", "same_file foreign key")
    ]
    page = Page.model_validate(payload)

    assert _rules(page, _component_ids(page)) == ["S1"]


def test_s2_rejects_duplicate_relations_without_direction() -> None:
    payload = _page_data()
    payload["data"]["relations"] = [
        _relation("users.id", "sessions.user_id", "references"),
        _relation("sessions.user_id", "users.id", "references"),
    ]
    page = Page.model_validate(payload)

    assert _rules(page, _component_ids(page)) == ["S2"]


def test_s3_rejects_unknown_relation_endpoints() -> None:
    payload = _page_data()
    payload["data"]["relations"] = [_relation("missing.id", "users.id", "references")]
    page = Page.model_validate(payload)

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["S3"]
    assert "missing.id" in findings[0].message


def test_clean_generated_page_has_no_findings() -> None:
    page = _page()

    assert lint.lint_page(page, _component_ids(page)) == []


def test_root_page_downgrades_block_findings_to_warnings() -> None:
    page = _page(page_id="rox-core", node_count=2)

    findings = lint.lint_page(page, _component_ids(page))

    assert [finding.rule for finding in findings] == ["B1"]
    assert findings[0].severity == "warning"


def test_lint_diagrams_cli_emits_one_json_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    page = _page(flow_note_count=2)
    (pages_dir / "components.json").write_text(
        json.dumps([{"id": node.id} for node in page.block.nodes]),
        encoding="utf-8",
    )
    (pages_dir / "domain.json").write_text(
        page.model_dump_json(),
        encoding="utf-8",
    )

    exit_code = cli.main(["lint-diagrams", str(pages_dir), "--json"])
    output = capsys.readouterr().out
    report = json.loads(output)

    assert exit_code == 0
    assert output.count("\n") == 1
    assert report["pages"] == 1
    assert report["errors"] == 0
    assert report["warnings"] == 1
    assert report["findings"][0]["rule"] == "B7"


def test_lint_diagrams_cli_rejects_unknown_page_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    page = _page()
    (pages_dir / "components.json").write_text("[]", encoding="utf-8")
    (pages_dir / "domain.json").write_text(page.model_dump_json(), encoding="utf-8")

    exit_code = cli.main(["lint-diagrams", str(pages_dir), "--page", "domain-missing"])

    assert exit_code == 2
    assert "domain-missing" in capsys.readouterr().err


def test_lint_diagrams_cli_rejects_missing_pages_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_dir = tmp_path / "missing"

    exit_code = cli.main(["lint-diagrams", str(missing_dir)])

    assert exit_code == 2
    assert str(missing_dir) in capsys.readouterr().err

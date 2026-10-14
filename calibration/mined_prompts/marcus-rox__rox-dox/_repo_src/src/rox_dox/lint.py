from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from rox_dox.block_svg import MAX_LAYOUT_PROBLEMS, block_layout_problems
from rox_dox.block_svg import _columns  # Reuse the renderer's column assignment.
from rox_dox.model import BlockDiagram, Page

GENERATED_PREFIXES = ("domain-", "feature-")
MERGED_NODE_IDS = frozenset({"provider-apis", "background-workers"})
MAX_BOXES = 15
MAX_COMFORTABLE_BOXES = 12
MIN_BOXES = 3
MAX_EDGES_PER_BOX = 1.5
MIN_FLOW_NOTES = 3
MAX_FLOW_NOTES = 6
_SAME_FILE_LABEL = "same_file"


@dataclass(frozen=True)
class Finding:
    page: str
    figure: str
    rule: str
    severity: Literal["error", "warning"]
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class LintInputError(Exception):
    pass


def load_component_ids(pages_dir: Path) -> frozenset[str]:
    path = pages_dir / "components.json"
    try:
        components = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LintInputError(f"{path}: {error}") from error
    if not isinstance(components, list):
        raise LintInputError(f"{path}: expected a JSON list of component objects")

    component_ids = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise LintInputError(
                f"{path}: item {index} must be an object with a string 'id'"
            )
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id:
            raise LintInputError(
                f"{path}: item {index} has invalid 'id': {component_id!r}"
            )
        component_ids.append(component_id)
    return frozenset(component_ids)


def load_pages(pages_dir: Path) -> list[Page]:
    pages = []
    for path in sorted(pages_dir.rglob("*.json")):
        if path == pages_dir / "components.json":
            continue
        try:
            page_data = json.loads(path.read_text(encoding="utf-8"))
            pages.append(Page.model_validate(page_data))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise LintInputError(f"{path}: {error}") from error
    return pages


def _finding(
    page: Page,
    figure: str,
    rule: str,
    severity: Literal["error", "warning"],
    message: str,
) -> Finding:
    return Finding(page.id, figure, rule, severity, message)


def _endpoint_columns(diagram: BlockDiagram) -> dict[str, int]:
    columns = _columns(diagram)
    result = {
        card.node.id: column_index
        for column_index, column in enumerate(columns)
        for card in column.cards
    }
    result.update(
        {
            section.group.id: column_index
            for column_index, column in enumerate(columns)
            for section in column.sections
        }
    )
    top_level_index = 0
    for group in diagram.groups:
        if group.parent is None:
            result[group.id] = top_level_index
            top_level_index += 1
    return result


def _lint_block_figure(
    page: Page,
    figure: str,
    diagram: BlockDiagram,
    component_ids: frozenset[str],
    *,
    generated: bool,
    include_flow_notes: bool,
) -> list[Finding]:
    findings = []
    box_count = len(diagram.nodes)
    if box_count > MAX_BOXES:
        findings.append(
            _finding(
                page,
                figure,
                "B1",
                "error",
                f"{box_count} boxes (max {MAX_BOXES})",
            )
        )
    elif box_count > MAX_COMFORTABLE_BOXES or box_count < MIN_BOXES:
        findings.append(
            _finding(
                page,
                figure,
                "B1",
                "warning",
                f"{box_count} boxes; comfortable range is "
                f"{MIN_BOXES}–{MAX_COMFORTABLE_BOXES}",
            )
        )

    if generated:
        allowed_ids = component_ids | MERGED_NODE_IDS
        for node in diagram.nodes:
            if node.id not in allowed_ids:
                findings.append(
                    _finding(
                        page,
                        figure,
                        "B2",
                        "error",
                        f"node id '{node.id}' is not in the component catalog",
                    )
                )

    edge_count = len(diagram.edges)
    edge_limit = math.ceil(MAX_EDGES_PER_BOX * box_count)
    if edge_count > edge_limit:
        findings.append(
            _finding(
                page,
                figure,
                "B3",
                "error",
                f"{edge_count} edges for {box_count} boxes; limit is {edge_limit}",
            )
        )

    endpoint_columns = _endpoint_columns(diagram)
    for edge in diagram.edges:
        source_column = endpoint_columns[edge.src]
        target_column = endpoint_columns[edge.dst]
        if target_column < source_column:
            findings.append(
                _finding(
                    page,
                    figure,
                    "B4",
                    "error",
                    f"backward edge {edge.src} -> {edge.dst} "
                    f"(columns {source_column} -> {target_column})",
                )
            )

    layout_problems = block_layout_problems(diagram)
    for problem in layout_problems:
        findings.append(_finding(page, figure, "B5", "warning", problem))
    if len(layout_problems) > MAX_LAYOUT_PROBLEMS:
        findings.append(
            _finding(
                page,
                figure,
                "B5",
                "error",
                f"{len(layout_problems)} layout problems (max {MAX_LAYOUT_PROBLEMS})",
            )
        )

    seen_edges = set()
    for edge in diagram.edges:
        if not edge.label.strip():
            findings.append(
                _finding(
                    page,
                    figure,
                    "B6",
                    "error",
                    f"edge {edge.src} -> {edge.dst} has an empty label",
                )
            )
        edge_key = (edge.src, edge.dst, edge.label)
        if edge_key in seen_edges:
            findings.append(
                _finding(
                    page,
                    figure,
                    "B6",
                    "error",
                    f"duplicate edge {edge.src} -> {edge.dst} "
                    f"with label '{edge.label}'",
                )
            )
        seen_edges.add(edge_key)

    if include_flow_notes and (
        len(page.block.notes) < MIN_FLOW_NOTES or len(page.block.notes) > MAX_FLOW_NOTES
    ):
        findings.append(
            _finding(
                page,
                figure,
                "B7",
                "warning",
                f"{len(page.block.notes)} flow notes; expected "
                f"{MIN_FLOW_NOTES}–{MAX_FLOW_NOTES}",
            )
        )

    if not generated:
        findings = [
            Finding(
                finding.page,
                finding.figure,
                finding.rule,
                "warning",
                finding.message,
            )
            for finding in findings
        ]
    return findings


def _relation_table_name(endpoint: str, nosql_names: frozenset[str]) -> str:
    if "::" in endpoint:
        return endpoint.split("::", maxsplit=1)[0]
    if endpoint in nosql_names:
        return endpoint
    return endpoint.split(".", maxsplit=1)[0]


def _lint_schema(page: Page) -> list[Finding]:
    findings = []
    relations = page.data.relations
    seen_relations = set()
    known_names = set(page.data.sql_tables) | {
        table for domain in page.data.domains for table in domain.tables
    }
    nosql_names = frozenset(store.name for store in page.data.nosql)
    known_names.update(nosql_names)

    for relation in relations:
        if _SAME_FILE_LABEL in relation.label:
            findings.append(
                _finding(
                    page,
                    "schema",
                    "S1",
                    "error",
                    f"relation {relation.src} -> {relation.dst} "
                    f"uses label '{relation.label}'",
                )
            )

        relation_key = (frozenset({relation.src, relation.dst}), relation.label)
        if relation_key in seen_relations:
            findings.append(
                _finding(
                    page,
                    "schema",
                    "S2",
                    "error",
                    f"duplicate relation {relation.src} -> {relation.dst} "
                    f"with label '{relation.label}'",
                )
            )
        seen_relations.add(relation_key)

        for endpoint in (relation.src, relation.dst):
            name = _relation_table_name(endpoint, nosql_names)
            if name not in known_names:
                findings.append(
                    _finding(
                        page,
                        "schema",
                        "S3",
                        "error",
                        f"unknown relation endpoint '{endpoint}'",
                    )
                )
    return findings


def lint_page(page: Page, component_ids: frozenset[str]) -> list[Finding]:
    generated = page.id.startswith(GENERATED_PREFIXES)
    findings = _lint_block_figure(
        page,
        "block",
        page.block,
        component_ids,
        generated=generated,
        include_flow_notes=True,
    )
    for figure in page.block_figures:
        findings.extend(
            _lint_block_figure(
                page,
                figure.id,
                figure.block,
                component_ids,
                generated=generated,
                include_flow_notes=False,
            )
        )
    findings.extend(_lint_schema(page))
    return findings


def lint_pages(
    pages: Iterable[Page],
    component_ids: frozenset[str],
) -> list[Finding]:
    return [finding for page in pages for finding in lint_page(page, component_ids)]


def summary_line(findings: Iterable[Finding], page_count: int) -> str:
    errors = 0
    warnings = 0
    for finding in findings:
        if finding.severity == "error":
            errors += 1
        else:
            warnings += 1
    return f"{errors} errors, {warnings} warnings in {page_count} pages"


def format_finding(finding: Finding) -> str:
    return (
        f"{finding.severity}: {finding.page} {finding.figure} "
        f"{finding.rule}: {finding.message}"
    )

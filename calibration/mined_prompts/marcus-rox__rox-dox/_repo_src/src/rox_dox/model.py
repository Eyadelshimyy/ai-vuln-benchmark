from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from rox_dox.schema import Table


RelationKind = Literal["enforced", "symbolic", "blob"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _first_duplicate(values: list[str]) -> str | None:
    counts = Counter(values)
    return next((value for value in values if counts[value] > 1), None)


class CodeSource(Model):
    path: str
    lines: tuple[int, int]


class NotionSource(Model):
    notion: str


Source = CodeSource | NotionSource


class MembershipRow(Model):
    path: str
    primary: bool
    evidence: str
    sources: list[Source] = Field(min_length=1)


class MembershipGroup(Model):
    layer: str
    rows: list[MembershipRow]


_MAX_NODE_DETAILS = 7
_MAX_NODE_DETAIL_TEXT_LENGTH = 90


class Claim(Model):
    text: str
    sources: list[Source] = Field(min_length=1)


class Group(Model):
    id: str
    label: str
    source: Source
    parent: str | None = None


class Node(Model):
    id: str
    label: str
    source: Source
    link: str | None = None
    kind: Literal["component", "store", "external", "queue"] = "component"
    group: str | None = None
    many: bool = False
    details: list[Claim] = Field(default_factory=list, max_length=_MAX_NODE_DETAILS)

    @model_validator(mode="before")
    @classmethod
    def validate_details(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        node_id = values.get("id", "<unknown>")
        details = values.get("details", [])
        if not isinstance(details, list):
            return values
        if len(details) > _MAX_NODE_DETAILS:
            raise ValueError(
                f"block node '{node_id}' has {len(details)} details; "
                f"maximum is {_MAX_NODE_DETAILS}"
            )
        for detail_number, detail in enumerate(details, start=1):
            if isinstance(detail, dict):
                text = detail.get("text")
            elif isinstance(detail, Claim):
                text = detail.text
            else:
                text = None
            if isinstance(text, str) and not (
                1 <= len(text) <= _MAX_NODE_DETAIL_TEXT_LENGTH
            ):
                raise ValueError(
                    f"block node '{node_id}' detail {detail_number}: "
                    f"text length {len(text)} must be between 1 and "
                    f"{_MAX_NODE_DETAIL_TEXT_LENGTH}"
                )
        return values


class Edge(Model):
    src: str
    dst: str
    label: str
    source: Source

    @model_validator(mode="before")
    @classmethod
    def require_source(cls, values: Any) -> Any:
        if isinstance(values, dict) and "source" not in values:
            src = values.get("src", "<unknown>")
            dst = values.get("dst", "<unknown>")
            raise ValueError(f"block edge {src}->{dst}: missing source")
        return values


class BlockDiagram(Model):
    notes: list[Claim] = Field(default_factory=list)
    groups: list[Group] = Field(default_factory=list)
    nodes: list[Node]
    edges: list[Edge]
    unreached_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_structure(self) -> BlockDiagram:
        group_ids = [group.id for group in self.groups]
        duplicate_group_id = _first_duplicate(group_ids)
        if duplicate_group_id is not None:
            raise ValueError(
                f"block diagram: duplicate group id '{duplicate_group_id}'"
            )

        node_ids = [node.id for node in self.nodes]
        declared_nodes = set(node_ids)
        duplicate_id = _first_duplicate(node_ids)
        if duplicate_id is not None:
            raise ValueError(f"block diagram: duplicate node id '{duplicate_id}'")

        declared_groups = set(group_ids)
        shared_id = next(
            (group_id for group_id in group_ids if group_id in declared_nodes),
            None,
        )
        if shared_id is not None:
            raise ValueError(f"block diagram: duplicate group/node id '{shared_id}'")

        groups_by_id = {group.id: group for group in self.groups}
        for group in self.groups:
            if group.parent is not None and group.parent not in declared_groups:
                raise ValueError(
                    f"block diagram: group '{group.id}': "
                    f"unknown parent '{group.parent}'"
                )
        for node in self.nodes:
            if node.group is not None and node.group not in declared_groups:
                raise ValueError(
                    f"block diagram: node '{node.id}': unknown group '{node.group}'"
                )

        for group in self.groups:
            visited: set[str] = set()
            current_id: str | None = group.id
            while current_id is not None:
                if current_id in visited:
                    raise ValueError(
                        f"block diagram: group parent cycle includes '{current_id}'"
                    )
                visited.add(current_id)
                current_id = groups_by_id[current_id].parent

        groups_with_children = {
            group.parent for group in self.groups if group.parent is not None
        }
        groups_with_nodes = {
            node.group for node in self.nodes if node.group is not None
        }
        for group in self.groups:
            if (
                group.id not in groups_with_children
                and group.id not in groups_with_nodes
            ):
                raise ValueError(
                    f"block diagram: group '{group.id}' has no nodes or child groups"
                )

        declared_endpoints = declared_nodes | declared_groups
        for edge in self.edges:
            if edge.src not in declared_endpoints:
                raise ValueError(
                    f"block edge {edge.src}->{edge.dst}: unknown endpoint '{edge.src}'"
                )
            if edge.dst not in declared_endpoints:
                raise ValueError(
                    f"block edge {edge.src}->{edge.dst}: unknown endpoint '{edge.dst}'"
                )

        incoming_nodes = {edge.dst for edge in self.edges}
        outgoing_nodes = {edge.src for edge in self.edges}
        for node in self.nodes:
            if (
                node.kind == "queue"
                and node.id not in incoming_nodes
                and node.id not in outgoing_nodes
            ):
                raise ValueError(
                    f"block diagram: queue '{node.id}' needs a producer or a "
                    "consumer edge"
                )
        return self


class BlockFigure(Model):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    title: str
    notes: list[Claim] = Field(default_factory=list)
    block: BlockDiagram


class Participant(Model):
    id: str
    label: str
    source: Source


class Step(Model):
    src: str
    dst: str
    message: str
    source: Source


class Sequence(Model):
    title: str
    participants: list[Participant]
    steps: list[Step]

    @model_validator(mode="after")
    def validate_structure(self) -> Sequence:
        participant_ids = [participant.id for participant in self.participants]
        declared_participants = set(participant_ids)
        duplicate_id = _first_duplicate(participant_ids)
        if duplicate_id is not None:
            raise ValueError(
                f"sequence '{self.title}': duplicate participant id '{duplicate_id}'"
            )

        for step_number, step in enumerate(self.steps, start=1):
            if step.src not in declared_participants:
                raise ValueError(
                    f"sequence '{self.title}' step {step_number} "
                    f"{step.src}->{step.dst}: "
                    f"unknown participant '{step.src}'"
                )
            if step.dst not in declared_participants:
                raise ValueError(
                    f"sequence '{self.title}' step {step_number} "
                    f"{step.src}->{step.dst}: "
                    f"unknown participant '{step.dst}'"
                )
        return self


class State(Model):
    id: str
    label: str
    source: Source


class Transition(Model):
    src: str
    dst: str
    event: str
    source: Source


class StateMachine(Model):
    title: str
    entity: str
    states: list[State]
    transitions: list[Transition]

    @model_validator(mode="after")
    def validate_structure(self) -> StateMachine:
        state_ids = [state.id for state in self.states]
        declared_states = set(state_ids)
        duplicate_id = _first_duplicate(state_ids)
        if duplicate_id is not None:
            raise ValueError(
                f"state machine '{self.title}': duplicate state id '{duplicate_id}'"
            )

        for transition in self.transitions:
            if transition.src not in declared_states:
                raise ValueError(
                    f"state machine '{self.title}' transition "
                    f"{transition.src}->{transition.dst}: "
                    f"unknown state '{transition.src}'"
                )
            if transition.dst not in declared_states:
                raise ValueError(
                    f"state machine '{self.title}' transition "
                    f"{transition.src}->{transition.dst}: "
                    f"unknown state '{transition.dst}'"
                )
        return self


class NoSqlStore(Model):
    name: str
    kind: str
    fields: list[str]
    source: Source


class Relation(Model):
    src: str
    dst: str
    label: str
    source: Source
    kind: RelationKind = "symbolic"


class SchemaDomain(Model):
    id: str
    title: str
    tables: list[str] = Field(min_length=1)
    key_tables: list[str] = Field(min_length=1)
    notes: list[Claim] = Field(default_factory=list)
    page: str | None = None

    @model_validator(mode="after")
    def validate_tables(self) -> SchemaDomain:
        duplicate = _first_duplicate(self.key_tables)
        if duplicate is not None:
            raise ValueError(
                f"schema domain '{self.id}': duplicate key table '{duplicate}'"
            )
        missing = sorted(set(self.key_tables) - set(self.tables))
        if missing:
            raise ValueError(
                f"schema domain '{self.id}': key tables not in domain: {missing}"
            )
        return self


class DataModel(Model):
    sql_tables: list[str] = Field(default_factory=list)
    nosql: list[NoSqlStore] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    domains: list[SchemaDomain] = Field(default_factory=list)
    columns: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_schema_layout(self) -> DataModel:
        domain_ids = [domain.id for domain in self.domains]
        duplicate_domain = _first_duplicate(domain_ids)
        if duplicate_domain is not None:
            raise ValueError(f"schema: duplicate domain id '{duplicate_domain}'")

        table_domains: dict[str, str] = {}
        for domain in self.domains:
            for table in domain.tables:
                previous_domain = table_domains.get(table)
                if previous_domain is not None:
                    raise ValueError(
                        f"schema: table '{table}' belongs to both domain "
                        f"'{previous_domain}' and domain '{domain.id}'"
                    )
                table_domains[table] = domain.id

        if self.domains and self.sql_tables:
            raise ValueError("schema: domains and sql_tables are mutually exclusive")

        layout_items = [item for column in self.columns for item in column]
        duplicate_item = _first_duplicate(layout_items)
        if duplicate_item is not None:
            raise ValueError(f"schema: duplicate columns item '{duplicate_item}'")

        declared_items = (
            set(domain_ids)
            | set(self.sql_tables)
            | {store.name for store in self.nosql}
        )
        unknown_item = next(
            (item for item in layout_items if item not in declared_items),
            None,
        )
        if unknown_item is not None:
            raise ValueError(f"schema: unknown columns item '{unknown_item}'")
        return self


class SummaryTableLink(Model):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    page: str


class SummaryTable(Model):
    columns: list[str]
    rows: list[list[str]]
    sources: list[Source] = Field(min_length=1)
    links: list[SummaryTableLink] = Field(default_factory=list)
    title: str | None = None

    @model_validator(mode="after")
    def validate_rows(self) -> SummaryTable:
        for row_number, row in enumerate(self.rows, start=1):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"TLDR table row {row_number} has {len(row)} cells; "
                    f"expected {len(self.columns)} columns"
                )
        positions = [(link.row, link.column) for link in self.links]
        duplicate = _first_duplicate([f"{row}:{column}" for row, column in positions])
        if duplicate is not None:
            raise ValueError(f"TLDR table has duplicate page link at {duplicate}")
        for row, column in positions:
            if row >= len(self.rows) or column >= len(self.columns):
                raise ValueError(
                    f"TLDR table page link at row {row}, column {column} "
                    "is outside the table"
                )
        return self


class Tldr(Model):
    summary: list[Claim]
    key_points: list[Claim]
    table: SummaryTable
    notes: list[Claim]
    additional_tables: list[SummaryTable] | None = None


class Related(Model):
    label: str
    page: str | None = None
    url: str | None = None
    source: Source

    @model_validator(mode="after")
    def validate_target(self) -> Related:
        if (self.page is None) == (self.url is None):
            raise ValueError("related item must specify exactly one of page or url")
        return self


class Page(Model):
    id: str
    title: str
    kind: Literal["root", "domain", "feature"]
    commit: str
    parent: str | None
    paths: list[str] = Field(min_length=1)
    membership: list[MembershipGroup] = Field(default_factory=list)
    tldr: Tldr
    block: BlockDiagram
    block_figures: list[BlockFigure] = Field(default_factory=list)
    data: DataModel
    sequences: list[Sequence]
    states: list[StateMachine]
    related: list[Related]

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        for path in paths:
            parsed = PurePosixPath(path)
            if (
                not path
                or path.endswith("/")
                or parsed.is_absolute()
                or ".." in parsed.parts
            ):
                raise ValueError(f"path '{path}' is not repo-relative")
        return paths

    @model_validator(mode="after")
    def validate_block_figures(self) -> Page:
        figure_ids = [figure.id for figure in self.block_figures]
        duplicate_id = _first_duplicate(figure_ids)
        if duplicate_id is not None:
            raise ValueError(f"duplicate block figure id '{duplicate_id}'")
        return self


def _claim_sources(claims: list[Claim], section: str) -> list[tuple[str, Source]]:
    return [
        (f"TLDR {section}: {claim.text[:60]}", source)
        for claim in claims
        for source in claim.sources
    ]


def page_sources(
    page: Page,
    tables: Mapping[str, Table] | None = None,
) -> list[tuple[str, Source]]:
    sources: list[tuple[str, Source]] = []
    sources.extend(_claim_sources(page.tldr.summary, "summary"))
    sources.extend(_claim_sources(page.tldr.key_points, "key point"))
    sources.extend((("TLDR table", source) for source in page.tldr.table.sources))
    sources.extend(
        (f"TLDR {table.title or 'additional'} table", source)
        for table in page.tldr.additional_tables or []
        for source in table.sources
    )
    sources.extend(_claim_sources(page.tldr.notes, "note"))
    sources.extend(
        (
            f"block note {note_number}",
            source,
        )
        for note_number, note in enumerate(page.block.notes, start=1)
        for source in note.sources
    )
    sources.extend(
        (f"block group {group.id}", group.source) for group in page.block.groups
    )
    sources.extend((f"block node {node.id}", node.source) for node in page.block.nodes)
    sources.extend(
        (
            f"block node {node.id} detail {detail_number}",
            source,
        )
        for node in page.block.nodes
        for detail_number, detail in enumerate(node.details, start=1)
        for source in detail.sources
    )
    sources.extend(
        (f"block edge {edge.src}->{edge.dst}", edge.source) for edge in page.block.edges
    )
    for figure in page.block_figures:
        sources.extend(
            (
                f"block figure {figure.id} note {note_number}",
                source,
            )
            for note_number, note in enumerate(figure.notes, start=1)
            for source in note.sources
        )
        sources.extend(
            (
                f"block figure {figure.id} group {group.id}",
                group.source,
            )
            for group in figure.block.groups
        )
        sources.extend(
            (
                f"block figure {figure.id} node {node.id}",
                node.source,
            )
            for node in figure.block.nodes
        )
        sources.extend(
            (
                f"block figure {figure.id} node {node.id} detail {detail_number}",
                source,
            )
            for node in figure.block.nodes
            for detail_number, detail in enumerate(node.details, start=1)
            for source in detail.sources
        )
        sources.extend(
            (
                f"block figure {figure.id} edge {edge.src}->{edge.dst}",
                edge.source,
            )
            for edge in figure.block.edges
        )

    for sequence in page.sequences:
        sources.extend(
            (
                f"sequence '{sequence.title}' participant {participant.id}",
                participant.source,
            )
            for participant in sequence.participants
        )
        sources.extend(
            (
                f"sequence '{sequence.title}' step {step_number} "
                f"{step.src}->{step.dst}",
                step.source,
            )
            for step_number, step in enumerate(sequence.steps, start=1)
        )

    for state_machine in page.states:
        sources.extend(
            (
                f"state machine '{state_machine.title}' state {state.id}",
                state.source,
            )
            for state in state_machine.states
        )
        sources.extend(
            (
                f"state machine '{state_machine.title}' transition "
                f"{transition.src}->{transition.dst}",
                transition.source,
            )
            for transition in state_machine.transitions
        )

    sources.extend(
        (f"NoSQL store {store.name}", store.source) for store in page.data.nosql
    )
    sources.extend(
        (f"schema domain {domain.id} note {note_number}", source)
        for domain in page.data.domains
        for note_number, note in enumerate(domain.notes, start=1)
        for source in note.sources
    )
    if tables is not None:
        sources.extend(
            (f"SQL table {table_name}", tables[table_name].source)
            for table_name in page.data.sql_tables
            if table_name in tables
        )
        sources.extend(
            (
                f"schema domain {domain.id} key table {table_name}",
                tables[table_name].source,
            )
            for domain in page.data.domains
            for table_name in domain.key_tables
            if table_name in tables
        )
    sources.extend(
        (f"relation {relation.src} -> {relation.dst}", relation.source)
        for relation in page.data.relations
    )
    sources.extend(
        (f"related '{related.label}'", related.source) for related in page.related
    )
    sources.extend(
        (f"membership {group.layer} file {row.path}", source)
        for group in page.membership
        for row in group.rows
        for source in row.sources
    )
    return sources

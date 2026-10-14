import json
from pathlib import Path

import pytest

from rox_dox.block_svg import block_layout_problems
from rox_dox.components import Endpoint, FileFacts
from rox_dox.feature_pages import (
    _domain_data,
    _feature_callers,
    _table_relations,
    feature_pages,
    layer_of,
)
from rox_dox.features import (
    Feature,
    FeatureCounts,
    FeatureEvidence,
    FeatureFile,
    FeatureMap,
    TableLink,
)
from rox_dox.model import CodeSource, Page, page_sources
from rox_dox.schema import Column, Table


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("web/apps/web/src/app/routes.py", "web"),
        (".agents/skills/models.py", "skills"),
        (".github/workflows/jobs/routes.py", "deploy"),
        ("backend/src/routes/jobs/worker.py", "routes"),
        ("backend/src/temporal/tasks/worker.py", "workers"),
        ("backend/src/models/user.py", "models"),
        ("backend/src/services/send.py", "logic"),
    ],
)
def test_layer_classifier_uses_first_matching_rule(
    path: str,
    expected: str,
) -> None:
    assert layer_of(path) == expected


def _caller_facts(path: str, endpoints: list[Endpoint]) -> FileFacts:
    return FileFacts(
        path=path,
        api_group=None,
        endpoints=endpoints,
        workers=[],
        externals=[],
    )


def test_two_hop_feature_caller_cites_its_first_import() -> None:
    endpoint_path = "api/notes.py"
    helper_path = "notes/business.py"
    member_path = "notes/service.py"
    feature = Feature(
        id="notes",
        tables=[],
        files=[
            FeatureFile(
                path=member_path,
                primary=True,
                reason="calls",
                evidence=[],
            )
        ],
        table_links=[],
    )
    callers = _feature_callers(
        feature,
        {
            endpoint_path: _caller_facts(
                endpoint_path,
                [Endpoint(method="GET", path="/notes", handler="list_notes", line=5)],
            ),
            helper_path: _caller_facts(helper_path, []),
        },
        {
            endpoint_path: {helper_path: 12},
            helper_path: {member_path: 19},
        },
        {},
    )

    assert len(callers) == 1
    assert callers[0].path == endpoint_path
    assert callers[0].targets == (member_path,)
    assert callers[0].hops == 2
    assert callers[0].import_sources == (
        CodeSource(path=endpoint_path, lines=(12, 12)),
    )


def test_feature_caller_traversal_stops_at_endpoint_files() -> None:
    caller_path = "api/notes.py"
    endpoint_path = "notes/endpoints.py"
    member_path = "notes/service.py"
    feature = Feature(
        id="notes",
        tables=[],
        files=[
            FeatureFile(
                path=member_path,
                primary=True,
                reason="calls",
                evidence=[],
            )
        ],
        table_links=[],
    )
    callers = _feature_callers(
        feature,
        {
            caller_path: _caller_facts(
                caller_path,
                [Endpoint(method="GET", path="/notes", handler="list_notes", line=5)],
            ),
            endpoint_path: _caller_facts(
                endpoint_path,
                [Endpoint(method="POST", path="/notes", handler="create_note", line=9)],
            ),
        },
        {
            caller_path: {endpoint_path: 12},
            endpoint_path: {member_path: 19},
        },
        {},
    )

    assert [(caller.path, caller.targets, caller.hops) for caller in callers] == [
        (endpoint_path, (member_path,), 1)
    ]


def test_four_hop_feature_caller_is_not_found() -> None:
    caller_path = "api/notes.py"
    helper_paths = [
        "notes/helper_one.py",
        "notes/helper_two.py",
        "notes/helper_three.py",
    ]
    member_path = "notes/service.py"
    feature = Feature(
        id="notes",
        tables=[],
        files=[
            FeatureFile(
                path=member_path,
                primary=True,
                reason="calls",
                evidence=[],
            )
        ],
        table_links=[],
    )
    component_facts = {
        caller_path: _caller_facts(
            caller_path,
            [Endpoint(method="GET", path="/notes", handler="list_notes", line=5)],
        ),
        **{path: _caller_facts(path, []) for path in helper_paths},
    }
    component_imports = {
        caller_path: {helper_paths[0]: 12},
        helper_paths[0]: {helper_paths[1]: 19},
        helper_paths[1]: {helper_paths[2]: 23},
        helper_paths[2]: {member_path: 31},
    }

    assert _feature_callers(feature, component_facts, component_imports, {}) == ()


def test_feature_callers_and_targets_are_ordered_deterministically() -> None:
    caller_paths = ["api/z_notes.py", "api/a_notes.py"]
    member_paths = ["notes/z_service.py", "notes/a_service.py"]
    feature = Feature(
        id="notes",
        tables=[],
        files=[
            FeatureFile(
                path=path,
                primary=True,
                reason="calls",
                evidence=[],
            )
            for path in reversed(member_paths)
        ],
        table_links=[],
    )
    component_facts = {
        path: _caller_facts(
            path,
            [Endpoint(method="GET", path="/notes", handler="list_notes", line=5)],
        )
        for path in caller_paths
    }
    callers = _feature_callers(
        feature,
        component_facts,
        {
            caller_paths[0]: {
                member_paths[0]: 41,
                member_paths[1]: 12,
            },
            caller_paths[1]: {member_paths[0]: 53},
        },
        {},
    )

    assert [caller.path for caller in callers] == [
        "api/a_notes.py",
        "api/z_notes.py",
    ]
    assert callers[1].targets == ("notes/a_service.py", "notes/z_service.py")
    assert callers[1].hops == 1
    assert callers[1].import_sources == (
        CodeSource(path="api/z_notes.py", lines=(12, 12)),
        CodeSource(path="api/z_notes.py", lines=(41, 41)),
    )


def test_feature_caller_import_sources_are_capped_to_first_lines() -> None:
    caller_path = "api/notes.py"
    helper_paths = [f"notes/helpers/helper_{index}.py" for index in range(5)]
    member_paths = [f"notes/services/service_{index}.py" for index in range(5)]
    import_lines = [50, 10, 40, 20, 30]
    feature = Feature(
        id="notes",
        tables=[],
        files=[
            FeatureFile(
                path=path,
                primary=True,
                reason="calls",
                evidence=[],
            )
            for path in member_paths
        ],
        table_links=[],
    )
    component_facts = {
        caller_path: _caller_facts(
            caller_path,
            [Endpoint(method="GET", path="/notes", handler="list_notes", line=5)],
        ),
        **{path: _caller_facts(path, []) for path in helper_paths},
    }
    component_imports = {
        caller_path: dict(zip(helper_paths, import_lines, strict=True)),
        **{
            helper_path: {member_path: 100 + index}
            for index, (helper_path, member_path) in enumerate(
                zip(helper_paths, member_paths, strict=True)
            )
        },
    }

    callers = _feature_callers(feature, component_facts, component_imports, {})

    assert len(callers) == 1
    assert callers[0].hops == 2
    assert callers[0].targets == tuple(sorted(member_paths))
    assert callers[0].import_sources == (
        CodeSource(path=caller_path, lines=(10, 10)),
        CodeSource(path=caller_path, lines=(20, 20)),
        CodeSource(path=caller_path, lines=(30, 30)),
    )


def test_generated_feature_pages_pass_block_layout() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    page_files = sorted((repository_root / "pages" / "features").glob("*.json"))
    assert page_files

    for page_file in page_files:
        page = Page.model_validate(json.loads(page_file.read_text(encoding="utf-8")))
        problems = block_layout_problems(page.block)
        assert not problems, f"{page_file}: {problems}"


def test_empty_feature_and_domain_use_key_table_definition_path() -> None:
    source = CodeSource(path="backend/src/models/user.py", lines=(12, 12))
    table = Table(
        name="users",
        class_name="User",
        columns=[Column(name="id", type="Integer", primary_key=True)],
        source=source,
    )
    feature_map = FeatureMap(
        commit="a" * 40,
        domain="test",
        threshold=0.25,
        tables=["users"],
        features=[Feature(id="users", tables=["users"], files=[], table_links=[])],
        uncovered=[],
        unmapped_tags=[],
        counts=FeatureCounts(
            domain_files=0,
            primary_placed=0,
            shared=0,
            uncovered=0,
            web=0,
            deployment=0,
            skills=0,
            unparseable=0,
        ),
    )

    pages = feature_pages(
        feature_map,
        root_id="rox-core",
        domain_title="Test",
        names={},
        tables={"users": table},
    )

    assert pages[0].paths == [source.path]
    assert pages[1].paths == [source.path]
    assert "No HTTP endpoints or background workers" in pages[1].tldr.summary[-1].text


def test_table_relations_keep_the_strongest_signal_per_unordered_pair() -> None:
    links = [
        TableLink(
            a="sessions",
            b="users",
            signal="commented_fk",
            path="backend/src/models/session.py",
            line=10,
        ),
        TableLink(
            a="users",
            b="sessions",
            signal="comparison",
            path="backend/src/services/users.py",
            line=20,
        ),
        TableLink(
            a="sessions",
            b="users",
            signal="primaryjoin",
            path="backend/src/models/session.py",
            line=30,
        ),
        TableLink(
            a="sessions",
            b="users",
            signal="fk",
            path="backend/src/models/session.py",
            line=40,
        ),
        TableLink(
            a="sessions",
            b="users",
            signal="id_column",
            path="backend/src/models/session.py",
            line=35,
        ),
        TableLink(
            a="sessions",
            b="users",
            signal="same_file",
            path="backend/src/models/session.py",
            line=50,
        ),
        TableLink(
            a="sessions",
            b="users",
            signal="comparison:ambiguous",
            path="backend/src/services/users.py",
            line=60,
        ),
    ]

    relations = _table_relations(links)

    assert len(relations) == 1
    assert (
        relations[0].src,
        relations[0].dst,
        relations[0].label,
        relations[0].kind,
        relations[0].source.path,
        relations[0].source.lines,
    ) == (
        "sessions",
        "users",
        "model ForeignKey",
        "symbolic",
        "backend/src/models/session.py",
        (40, 40),
    )


def test_id_column_schema_relation_is_symbolic() -> None:
    relation = _table_relations(
        [
            TableLink(
                a="email_message",
                b="person",
                signal="id_column",
                path="backend/src/models/email_message.py",
                line=18,
            )
        ]
    )[0]

    assert (relation.label, relation.kind, relation.source.lines) == (
        "ID column name",
        "symbolic",
        (18, 18),
    )


def test_domain_schema_key_tables_include_cross_feature_relation_endpoints() -> None:
    table_names = ["a", "b", "c", "d", "z"]
    tables = {
        name: Table(
            name=name,
            class_name=name.capitalize(),
            columns=[Column(name="id", type="Integer", primary_key=True)],
            source=CodeSource(
                path=f"backend/src/models/{name}.py",
                lines=(1, 1),
            ),
        )
        for name in table_names
    }
    feature_map = FeatureMap(
        commit="a" * 40,
        domain="test",
        threshold=0.25,
        tables=table_names,
        features=[
            Feature(id="a", tables=["a", "b", "c", "d"], files=[], table_links=[]),
            Feature(id="z", tables=["z"], files=[], table_links=[]),
        ],
        cross_links=[
            TableLink(
                a="d",
                b="z",
                signal="primaryjoin",
                path="backend/src/models/d.py",
                line=8,
            )
        ],
        uncovered=[],
        unmapped_tags=[],
        counts=FeatureCounts(
            domain_files=0,
            primary_placed=0,
            shared=0,
            uncovered=0,
            web=0,
            deployment=0,
            skills=0,
            unparseable=0,
        ),
    )

    data = _domain_data(feature_map, {}, tables)

    assert [domain.key_tables for domain in data.domains] == [
        ["a", "b", "c", "d"],
        ["z"],
    ]
    assert [
        (relation.src, relation.dst, relation.label) for relation in data.relations
    ] == [("d", "z", "ORM join")]


def test_domain_tldr_lists_cited_links_to_other_domains() -> None:
    source = CodeSource(path="backend/src/models/alpha.py", lines=(8, 8))
    tables = {
        name: Table(
            name=name,
            class_name=name.capitalize(),
            columns=[Column(name="id", type="Integer", primary_key=True)],
            source=CodeSource(
                path=f"backend/src/models/{name}.py",
                lines=(1, 4),
            ),
        )
        for name in ("alpha", "person")
    }
    feature_map = FeatureMap(
        commit="a" * 40,
        domain="activity",
        threshold=0.25,
        tables=["alpha"],
        features=[Feature(id="alpha", tables=["alpha"], files=[], table_links=[])],
        cross_links=[],
        external_links=[
            TableLink(
                a="alpha",
                b="person",
                signal="id_column",
                path=source.path,
                line=source.lines[0],
            )
        ],
        uncovered=[],
        unmapped_tags=[],
        counts=FeatureCounts(
            domain_files=0,
            primary_placed=0,
            shared=0,
            uncovered=0,
            web=0,
            deployment=0,
            skills=0,
            unparseable=0,
        ),
    )

    domain_page = feature_pages(
        feature_map,
        root_id="rox-core",
        domain_title="Activity",
        names={},
        tables=tables,
        domain_titles={"people": "People"},
        table_domains={"alpha": "activity", "person": "people"},
    )[0]

    link_table = domain_page.tldr.additional_tables[0]
    assert link_table.title == "Links to other domains"
    assert link_table.columns == [
        "This table",
        "Other table",
        "Other domain",
        "Signal",
    ]
    assert link_table.rows == [["alpha", "person", "People", "ID column name"]]
    assert link_table.links[0].page == "domain-people"
    assert ("TLDR Links to other domains table", source) in page_sources(domain_page)


def test_domain_tldr_preserves_cross_feature_import_summary() -> None:
    alpha_path = "backend/src/services/alpha.py"
    beta_path = "backend/src/services/beta.py"
    tables = {
        name: Table(
            name=name,
            class_name=name.capitalize(),
            columns=[Column(name="id", type="Integer", primary_key=True)],
            source=CodeSource(path=f"backend/src/models/{name}.py", lines=(1, 1)),
        )
        for name in ("alpha", "beta")
    }
    alpha_file = FeatureFile(
        path=alpha_path,
        primary=True,
        reason="tables",
        evidence=[
            FeatureEvidence(
                kind="call",
                path=alpha_path,
                line=20,
                to=beta_path,
            )
        ],
    )
    beta_file = FeatureFile(
        path=beta_path,
        primary=True,
        reason="tables",
        evidence=[],
    )
    feature_map = FeatureMap(
        commit="a" * 40,
        domain="test",
        threshold=0.25,
        tables=["alpha", "beta"],
        features=[
            Feature(
                id="alpha",
                tables=["alpha"],
                files=[alpha_file],
                table_links=[],
            ),
            Feature(
                id="beta",
                tables=["beta"],
                files=[beta_file],
                table_links=[],
            ),
        ],
        uncovered=[],
        unmapped_tags=[],
        counts=FeatureCounts(
            domain_files=2,
            primary_placed=2,
            shared=0,
            uncovered=0,
            web=0,
            deployment=0,
            skills=0,
            unparseable=0,
        ),
    )

    domain_page = feature_pages(
        feature_map,
        root_id="rox-core",
        domain_title="Test",
        names={},
        tables=tables,
    )[0]

    assert any(
        "alpha imports code from beta (1 imports)" in note.text.lower()
        for note in domain_page.tldr.notes
    )
    assert any(
        note.sources[0].path == alpha_path and note.sources[0].lines == (20, 20)
        for note in domain_page.tldr.notes
    )

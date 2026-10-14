from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from rox_dox import cache as cache_module
from rox_dox import features as feature_module
from rox_dox.features import (
    FeatureMap,
    _RawLink,
    _domain_scope,
    _lowest_score_feature,
    _parse_graph,
    _snapshot,
    build_feature_map,
    build_feature_maps,
)
from rox_dox.schema import extract_tables


def _commit_sources(tmp_path: Path, sources: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "feature-source"
    repo.mkdir()
    for relative_path, source in sources.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(
        ["git", "init", str(repo)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", *sources],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add feature fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def _base_sources(*, foreign_key: bool = False) -> dict[str, str]:
    sequence_column = (
        '    sequence_id = Column(Integer, ForeignKey("sequence.id"))\n'
        if foreign_key
        else ""
    )
    return {
        "backend/src/rox_core/sequence.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Sequence:\n"
            '    __tablename__ = "sequence"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        "backend/src/rox_core/campaign.py": (
            "from sqlalchemy import Column, Integer, ForeignKey\n"
            "\n"
            "class Campaign:\n"
            '    __tablename__ = "campaign"\n'
            "    id = Column(Integer, primary_key=True)\n"
            f"{sequence_column}"
        ),
        "backend/src/rox_core/sequence_service.py": (
            "from .sequence import Sequence\n"
            "from .shared import shared\n"
            "\n"
            "def run() -> Sequence:\n"
            "    return shared()\n"
        ),
        "backend/src/rox_core/campaign_service.py": (
            "from .campaign import Campaign\n"
            "\n"
            "def run() -> Campaign:\n"
            "    return Campaign()\n"
        ),
        "backend/src/rox_core/shared.py": 'def shared() -> str:\n    return "shared"\n',
        "backend/src/rox_core/split.py": (
            "from .sequence_service import run as sequence_run\n"
            "from .campaign_service import run as campaign_run\n"
        ),
        "backend/src/rox_core/seq_routes.py": (
            "from flask_restx import Namespace as FlaskRestxNamespace\n"
            "from .sequence_service import run\n"
            '\nns = FlaskRestxNamespace(name="sequences")\n'
        ),
        "backend/src/rox_core/broken.py": "def broken(:\n",
        "backend/src/tests/ignored.py": "from .sequence import Sequence\n",
        "backend/src/rox_core/test_helper.py": "from .sequence import Sequence\n",
        "web/apps/web/src/page.tsx": (
            'import useSequences from "#/app/hooks/react-query/generated/'
            'sequences/sequences";\n'
        ),
        "web/apps/web/src/unmapped.tsx": (
            'import useMissing from "#/app/hooks/react-query/generated/'
            'missing/missing";\n'
        ),
        ".github/workflows/map.yml": "module: rox_core.sequence_service\n",
        ".agents/skills/sequence/SKILL.md": (
            "See backend/src/rox_core/sequence_service.py.\n"
        ),
    }


def _build(
    tmp_path: Path,
    *,
    foreign_key: bool = False,
    tables: list[str] | None = None,
) -> FeatureMap:
    repo, commit = _commit_sources(
        tmp_path,
        _base_sources(foreign_key=foreign_key),
    )
    return build_feature_map(
        repo,
        commit,
        {"seq": tables or ["campaign", "sequence"]},
        "seq",
    )


def _feature_by_table(feature_map: FeatureMap, table: str):
    return next(feature for feature in feature_map.features if table in feature.tables)


def _dump(feature_map: FeatureMap) -> str:
    return (
        json.dumps(
            feature_map.model_dump(exclude_none=True),
            indent=1,
        )
        + "\n"
    )


def test_domain_scope_clustering_and_call_propagation(tmp_path: Path) -> None:
    feature_map = _build(tmp_path)

    assert feature_map.counts.domain_files == 7
    assert feature_map.counts.unparseable == 1
    assert {frozenset(feature.tables) for feature in feature_map.features} == {
        frozenset({"campaign"}),
        frozenset({"sequence"}),
    }
    sequence = _feature_by_table(feature_map, "sequence")
    campaign = _feature_by_table(feature_map, "campaign")
    assert "backend/src/rox_core/sequence_service.py" in {
        file.path for file in sequence.files
    }
    assert "backend/src/rox_core/campaign_service.py" in {
        file.path for file in campaign.files
    }
    assert "backend/src/rox_core/shared.py" in {file.path for file in sequence.files}


def test_domain_scope_split_vote_stays_unassigned(tmp_path: Path) -> None:
    repo, commit = _commit_sources(tmp_path, _base_sources())
    snapshot = _snapshot(repo, commit)
    tables = extract_tables(repo, commit)
    graph, _ = _parse_graph(snapshot, tables, {"campaign", "sequence"})

    domains, _ = _domain_scope(
        graph,
        {"seq": ["sequence"], "other": ["campaign"]},
    )

    assert "backend/src/rox_core/split.py" not in domains


def test_relative_imports_resolve_for_packages_and_modules(
    tmp_path: Path,
) -> None:
    sources = {
        "backend/src/pkg/thing.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Thing:\n"
            '    __tablename__ = "thing"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        "backend/src/pkg/service.py": (
            "from .thing import Thing\n"
            "\n"
            "def get_thing() -> Thing:\n"
            "    return Thing()\n"
        ),
        "backend/src/pkg/__init__.py": "from .service import Thing\n",
        "backend/src/pkg/a.py": "from .b import x\n",
        "backend/src/pkg/b.py": "x = 1\n",
    }
    repo, commit = _commit_sources(tmp_path, sources)
    snapshot = _snapshot(repo, commit)
    tables = extract_tables(repo, commit)
    graph, _ = _parse_graph(snapshot, tables, {"thing"})
    package_init = "backend/src/pkg/__init__.py"
    service = "backend/src/pkg/service.py"
    assert graph[package_init].imports == [service]
    assert graph["backend/src/pkg/a.py"].imports == ["backend/src/pkg/b.py"]

    feature_map = build_feature_map(repo, commit, {"seq": ["thing"]}, "seq")
    package_entry = next(
        file
        for feature in feature_map.features
        for file in feature.files
        if file.path == package_init
    )
    import_evidence = next(
        evidence
        for evidence in package_entry.evidence
        if evidence.kind == "call" and evidence.to == service
    )
    assert import_evidence.line == 1


def test_runtime_caller_imports_use_the_feature_graph_and_import_lines(
    tmp_path: Path,
) -> None:
    caller_path = "backend/src/rox_core/api/caller.py"
    member_path = "backend/src/rox_core/integrations/member.py"
    sources = {
        "backend/src/rox_core/__init__.py": "",
        "backend/src/rox_core/api/__init__.py": "",
        "backend/src/rox_core/integrations/__init__.py": "",
        caller_path: (
            "from rox_core.integrations.member import handle\n"
            "\n"
            "def callback():\n"
            "    return handle()\n"
        ),
        member_path: "def handle():\n    return None\n",
        "backend/src/rox_core/tests/ignored.py": (
            "from rox_core.integrations.member import handle\n"
        ),
        "backend/src/rox_core/migrations/ignored.py": (
            "from rox_core.integrations.member import handle\n"
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)

    graph, unparseable = _parse_graph(_snapshot(repo, commit), {}, set())

    assert unparseable == 0
    assert graph[caller_path].imports == [member_path]
    assert graph[caller_path].import_lines[member_path] == 1
    assert "backend/src/rox_core/tests/ignored.py" not in graph
    assert "backend/src/rox_core/migrations/ignored.py" not in graph


def test_called_by_pass_does_not_cascade(tmp_path: Path) -> None:
    sources = {
        "backend/src/pkg/a.py": (
            "from sqlalchemy import Column, Integer\n"
            "from .b import value\n"
            "\n"
            "class Sequence:\n"
            '    __tablename__ = "sequence"\n'
            "    id = Column(Integer, primary_key=True)\n"
        ),
        "backend/src/pkg/b.py": "from .c import value\n",
        "backend/src/pkg/c.py": "value = 1\n",
    }
    repo, commit = _commit_sources(tmp_path, sources)
    snapshot = _snapshot(repo, commit)
    tables = extract_tables(repo, commit)
    graph, _ = _parse_graph(snapshot, tables, {"sequence"})

    domains, reasons = _domain_scope(graph, {"seq": ["sequence"]})

    assert domains["backend/src/pkg/a.py"] == ["seq"]
    assert domains["backend/src/pkg/b.py"] == ["seq"]
    assert reasons["backend/src/pkg/b.py"] == "called_by"
    assert "backend/src/pkg/c.py" not in domains


def test_tableless_call_tie_uses_lowest_feature_index(tmp_path: Path) -> None:
    sources = {
        "backend/src/pkg/alpha.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Alpha:\n"
            '    __tablename__ = "alpha"\n'
            "    id = Column(Integer, primary_key=True)\n"
            "\n"
            "def get_alpha():\n"
            "    return Alpha()\n"
        ),
        "backend/src/pkg/zeta.py": (
            "from sqlalchemy import Column, Integer\n"
            "\n"
            "class Zeta:\n"
            '    __tablename__ = "zeta"\n'
            "    id = Column(Integer, primary_key=True)\n"
            "\n"
            "def get_zeta():\n"
            "    return Zeta()\n"
        ),
        "backend/src/pkg/choice.py": (
            "from .alpha import get_alpha\nfrom .zeta import get_zeta\n"
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)
    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["alpha", "zeta"]},
        "seq",
    )

    alpha = _feature_by_table(feature_map, "alpha")
    assert any(
        file.path == "backend/src/pkg/choice.py" and file.primary
        for file in alpha.files
    )


def test_table_score_tie_uses_epsilon_and_lowest_feature_index() -> None:
    assert _lowest_score_feature([0.7 - 0.5e-9, 0.7]) == 0
    assert _lowest_score_feature([0.7 - 2e-9, 0.7]) == 1


def test_foreign_key_link_merges_clusters_at_default_threshold(
    tmp_path: Path,
) -> None:
    feature_map = _build(tmp_path, foreign_key=True)

    assert len(feature_map.features) == 1
    assert set(feature_map.features[0].tables) == {"campaign", "sequence"}
    assert any(link.signal == "fk" for link in feature_map.features[0].table_links)


def test_cross_links_keep_only_relations_between_distinct_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, commit = _commit_sources(tmp_path, _base_sources())
    path = "backend/src/rox_core/campaign.py"
    links = [
        _RawLink("campaign", "sequence", "fk", path, 1),
        _RawLink("campaign", "sequence", "comparison", path, 1),
        _RawLink("campaign", "sequence", "same_file", path, 1),
        _RawLink("campaign", "sequence", "primaryjoin:ambiguous", path, 1),
    ]
    monkeypatch.setattr(
        feature_module,
        "_table_links",
        lambda *_args: {("campaign", "sequence"): links},
    )

    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["campaign", "sequence"]},
        "seq",
        threshold=1.0,
    )

    owner = {
        table: feature.id
        for feature in feature_map.features
        for table in feature.tables
    }
    assert len(set(owner.values())) == 2
    assert [(link.a, link.b, link.signal) for link in feature_map.cross_links] == [
        ("campaign", "sequence", "comparison"),
        ("campaign", "sequence", "fk"),
    ]
    assert all(owner[link.a] != owner[link.b] for link in feature_map.cross_links)


def test_id_column_links_do_not_change_clustering_and_include_external_domains(
    tmp_path: Path,
) -> None:
    sources = {
        "backend/src/activity/alpha.py": (
            "from sqlalchemy import Column, String\n"
            "\n"
            "class Alpha:\n"
            '    __tablename__ = "alpha"\n'
            "    id = Column(String)\n"
            "    person_id = Column(String)\n"
        ),
        "backend/src/activity/beta.py": (
            "from sqlalchemy import Column, String\n"
            "\n"
            "class Beta:\n"
            '    __tablename__ = "beta"\n'
            "    id = Column(String)\n"
            "    alpha_id = Column(String)\n"
        ),
        "backend/src/activity/gamma_delta.py": (
            "from sqlalchemy import Column, String\n"
            "\n"
            "class Gamma:\n"
            '    __tablename__ = "gamma"\n'
            "    id = Column(String)\n"
            "\n"
            "class Delta:\n"
            '    __tablename__ = "delta"\n'
            "    gamma_id = Column(String)\n"
        ),
        "backend/src/people/person.py": (
            "from sqlalchemy import Column, String\n"
            "\n"
            "class Person:\n"
            '    __tablename__ = "person"\n'
            "    id = Column(String)\n"
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)

    feature_map = build_feature_map(
        repo,
        commit,
        {
            "activity": ["alpha", "beta", "gamma", "delta"],
            "people": ["person"],
        },
        "activity",
    )

    assert {frozenset(feature.tables) for feature in feature_map.features} == {
        frozenset({"alpha"}),
        frozenset({"beta"}),
        frozenset({"gamma", "delta"}),
    }
    assert [(link.a, link.b, link.signal) for link in feature_map.cross_links] == [
        ("alpha", "beta", "id_column")
    ]
    grouped_tables = next(
        feature
        for feature in feature_map.features
        if set(feature.tables) == {"gamma", "delta"}
    )
    assert any(link.signal == "id_column" for link in grouped_tables.table_links)
    assert [(link.a, link.b, link.signal) for link in feature_map.external_links] == [
        ("alpha", "person", "id_column")
    ]


def test_evidence_check_reports_partnerless_file(tmp_path: Path) -> None:
    sources = _base_sources()
    sources["backend/src/rox_core/orphan.py"] = (
        "from sqlalchemy import Column, Integer\n"
        "\n"
        "class Orphan:\n"
        '    __tablename__ = "orphan"\n'
        "    id = Column(Integer, primary_key=True)\n"
    )
    repo, commit = _commit_sources(tmp_path, sources)
    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["campaign", "orphan", "sequence"]},
        "seq",
    )

    assert {item.path: item.reason for item in feature_map.uncovered}[
        "backend/src/rox_core/orphan.py"
    ] == ("no shared table or call in its feature")


def test_excluded_paths_never_appear(tmp_path: Path) -> None:
    feature_map = _build(tmp_path)

    paths = {
        file.path for feature in feature_map.features for file in feature.files
    } | {file.path for file in feature_map.uncovered}
    assert "backend/src/tests/ignored.py" not in paths
    assert "backend/src/rox_core/test_helper.py" not in paths


def test_web_routes_and_unmapped_tags(tmp_path: Path) -> None:
    feature_map = _build(tmp_path)
    sequence = _feature_by_table(feature_map, "sequence")
    sequence_paths = {file.path for file in sequence.files}

    assert "web/apps/web/src/page.tsx" in sequence_paths
    assert "missing" in feature_map.unmapped_tags
    assert feature_map.counts.web == 1


def test_web_calls_use_all_imports_and_two_snapshot_passes(tmp_path: Path) -> None:
    sources = {
        **_base_sources(),
        "web/apps/web/src/route-one.tsx": (
            'import "#/app/hooks/react-query/generated/sequences/sequences";\n'
        ),
        "web/apps/web/src/route-two.tsx": (
            'import "#/app/hooks/react-query/generated/sequences/sequences";\n'
        ),
        "web/apps/web/src/unplaced-one.ts": "export const one = 1;\n",
        "web/apps/web/src/unplaced-two.ts": "export const two = 2;\n",
        "web/apps/web/src/unplaced-three.ts": "export const three = 3;\n",
        "web/apps/web/src/below-threshold.tsx": (
            'import routeOne from "./route-one";\n'
            'import one from "./unplaced-one";\n'
            'import two from "./unplaced-two";\n'
        ),
        "web/apps/web/src/above-threshold.tsx": (
            'import routeOne from "./route-one";\n'
            'import routeTwo from "./route-two";\n'
            'import three from "./unplaced-three";\n'
        ),
        "web/apps/web/src/chain-a.tsx": 'import chainB from "./chain-b";\n',
        "web/apps/web/src/chain-b.tsx": 'import chainC from "./chain-c";\n',
        "web/apps/web/src/chain-c.tsx": (
            'import "#/app/hooks/react-query/generated/sequences/sequences";\n'
        ),
    }
    repo, commit = _commit_sources(tmp_path, sources)
    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["campaign", "sequence"]},
        "seq",
    )
    sequence = _feature_by_table(feature_map, "sequence")
    sequence_files = {file.path: file for file in sequence.files}
    placed_paths = {
        file.path for feature in feature_map.features for file in feature.files
    }

    assert "web/apps/web/src/below-threshold.tsx" not in placed_paths
    assert sequence_files["web/apps/web/src/above-threshold.tsx"].reason == "calls"
    assert sequence_files["web/apps/web/src/chain-b.tsx"].reason == "calls"
    assert sequence_files["web/apps/web/src/chain-a.tsx"].reason == "calls"
    assert {
        evidence.to
        for evidence in sequence_files["web/apps/web/src/above-threshold.tsx"].evidence
        if evidence.kind == "call"
    } == {
        "web/apps/web/src/route-one.tsx",
        "web/apps/web/src/route-two.tsx",
    }


def test_deployment_and_skill_references_join_their_feature(
    tmp_path: Path,
) -> None:
    feature_map = _build(tmp_path)
    sequence = _feature_by_table(feature_map, "sequence")
    sequence_paths = {file.path for file in sequence.files}

    assert ".github/workflows/map.yml" in sequence_paths
    assert ".agents/skills/sequence/SKILL.md" in sequence_paths
    assert feature_map.counts.deployment == 1
    assert feature_map.counts.skills == 1


def test_every_evidence_line_exists_at_the_commit(tmp_path: Path) -> None:
    repo, commit = _commit_sources(tmp_path, _base_sources())
    feature_map = build_feature_map(
        repo,
        commit,
        {"seq": ["campaign", "sequence"]},
        "seq",
    )

    for feature in feature_map.features:
        for file in feature.files:
            for evidence in file.evidence:
                source = subprocess.run(
                    ["git", "-C", str(repo), "show", f"{commit}:{evidence.path}"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()
                assert 1 <= evidence.line <= len(source)


def test_feature_map_is_deterministic(tmp_path: Path) -> None:
    repo, commit = _commit_sources(tmp_path, _base_sources())
    first = build_feature_map(
        repo,
        commit,
        {"seq": ["campaign", "sequence"]},
        "seq",
    )
    second = build_feature_map(
        repo,
        commit,
        {"seq": ["campaign", "sequence"]},
        "seq",
    )

    assert _dump(first) == _dump(second)


def test_build_feature_maps_reuses_shared_graph_and_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sources = _base_sources()
    sources["backend/src/rox_core/cache_scope_check.py"] = "scope_marker = True\n"
    repo, commit = _commit_sources(tmp_path, sources)
    domain_tables = {"seq": ["campaign", "sequence"]}
    counts = {"_snapshot": 0, "_parse_graph": 0, "_domain_scope": 0}

    for name in counts:
        original = getattr(feature_module, name)

        def tracked(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(feature_module, name, tracked)

    feature_maps = build_feature_maps(repo, commit, domain_tables)

    assert counts == {"_snapshot": 1, "_parse_graph": 1, "_domain_scope": 1}
    assert feature_maps["seq"] == build_feature_map(
        repo,
        commit,
        domain_tables,
        "seq",
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("User()", (1, None)),
        ("insert(User)", (1, None)),
        ("update(User)", (1, None)),
        ("delete(User)", (1, None)),
        ("bulk_insert_mappings(User, rows)", (1, None)),
        ("bulk_update_mappings(User, rows)", (1, None)),
        ("session.query(User).update({})", (1, None)),
        ("session.select(User).delete()", (1, None)),
        ("query(User)", (None, 1)),
        ("select(User)", (None, 1)),
        ("User.query", (None, 1)),
        ("User.find_by_email(email)", (None, 1)),
        ("User.get_by_id(identifier)", (None, 1)),
        ("User.list_active()", (None, 1)),
        ("User.fetch_latest()", (None, 1)),
        ("User.load_related()", (None, 1)),
    ],
)
def test_table_access_classifier_recognizes_read_and_write_patterns(
    expression: str,
    expected: tuple[int | None, int | None],
) -> None:
    accesses = feature_module._table_accesses(
        ast.parse(f"{expression}\n"),
        {"User": ["users"]},
        {},
        {},
    )

    assert accesses == {"users": expected}


def test_table_access_classifier_resolves_dynamic_model_aliases() -> None:
    tree = ast.parse(
        "UserModel = type[User]\n"
        "def get_model() -> type[User]:\n"
        "    return User\n"
        "def read(model: UserModel = User):\n"
        "    return session.execute(select(model))\n"
        "def write():\n"
        "    model = get_model()\n"
        "    session.bulk_update_mappings(model, [])\n"
    )

    accesses = feature_module._table_accesses(tree, {"User": ["users"]}, {}, {})

    assert accesses == {"users": (8, 5)}


def test_parse_graph_cache_is_partitioned_by_extractor_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cache_module, "CACHE_ROOT", tmp_path / "cache")
    version = cache_module.CACHE_VERSION
    parse_versions = []

    def parse_graph(*args: object) -> tuple[dict, int]:
        parse_versions.append(cache_module.CACHE_VERSION)
        return {}, 0

    monkeypatch.setattr(feature_module, "_parse_graph", parse_graph)
    snapshot = feature_module._Snapshot(files={}, all_paths=set())

    feature_module._cached_parse_graph(
        tmp_path, "cache-version-test", snapshot, {}, set()
    )
    feature_module._cached_parse_graph(
        tmp_path, "cache-version-test", snapshot, {}, set()
    )
    monkeypatch.setattr(cache_module, "CACHE_VERSION", version + 1)
    feature_module._cached_parse_graph(
        tmp_path, "cache-version-test", snapshot, {}, set()
    )

    assert parse_versions == [version, version + 1]


def test_unclassified_model_method_remains_uses_evidence() -> None:
    accesses = feature_module._table_accesses(
        ast.parse("User.save()\n"),
        {"User": ["users"]},
        {},
        {},
    )

    assert accesses == {}

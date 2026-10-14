import subprocess
from pathlib import Path

from rox_dox import cache, components, features, schema
from rox_dox.schema import extract_tables


def test_warm_cache_returns_the_cold_value_without_recomputing(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    calls = []

    def compute():
        calls.append("computed")
        return {"files": ["a.py", "b.py"], "count": 2}

    cold = cache.get_or_compute("abc1234", "fixture", compute)
    def unexpected_compute():
        raise AssertionError("warm cache missed")

    warm = cache.get_or_compute("abc1234", "fixture", unexpected_compute)

    assert cold == warm == {"files": ["a.py", "b.py"], "count": 2}
    assert calls == ["computed"]
    assert (
        tmp_path
        / cache._commit_key("abc1234")
        / f"v{cache.CACHE_VERSION}"
        / "fixture.pickle"
    ).is_file()


def test_extract_tables_uses_warm_commit_cache(
    tmp_path,
    monkeypatch,
    git_repo: tuple[Path, str],
):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    repo, commit = git_repo
    cold = extract_tables(repo, commit)

    def unexpected_extract(_repo, _commit):
        raise AssertionError("warm table cache missed")

    monkeypatch.setattr(schema, "_extract_tables", unexpected_extract)

    assert extract_tables(repo, commit) == cold


def test_snapshot_and_parse_graph_use_warm_commit_caches(
    tmp_path,
    monkeypatch,
    git_repo: tuple[Path, str],
):
    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path / "cache")
    repo, _ = git_repo
    source_path = repo / "backend" / "src" / "handlers.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("from pkg import service\n", encoding="utf-8")
    service_path = source_path.parent / "pkg" / "service.py"
    service_path.parent.mkdir()
    service_path.write_text("class Service:\n    pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "backend/src"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Add backend fixture"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cold_snapshot = features._snapshot(repo, commit)

    def unexpected_snapshot(_repo, _commit):
        raise AssertionError("warm snapshot cache missed")

    monkeypatch.setattr(features, "_snapshot_from_git", unexpected_snapshot)
    warm_snapshot = features._snapshot(repo, commit)
    assert warm_snapshot.files == cold_snapshot.files
    assert warm_snapshot.all_paths == cold_snapshot.all_paths

    cold_graph = features._cached_parse_graph(repo, commit, cold_snapshot, {}, set())

    def unexpected_parse(*_arguments):
        raise AssertionError("warm graph cache missed")

    monkeypatch.setattr(features, "_parse_graph", unexpected_parse)

    warm_graph = features._cached_parse_graph(repo, commit, warm_snapshot, {}, set())
    assert list(warm_graph[0]) == list(cold_graph[0])
    assert warm_graph[1] == cold_graph[1]
    assert warm_graph[0]["backend/src/handlers.py"].imports == [
        "backend/src/pkg/service.py",
    ]

    cold_facts = components.extract_file_facts(
        repo,
        commit,
        ["backend/src/handlers.py", "backend/src/pkg/service.py"],
    )

    def unexpected_facts(_repo, _commit, _requested):
        raise AssertionError("warm file-facts cache missed")

    monkeypatch.setattr(components, "_extract_file_facts", unexpected_facts)
    warm_facts = components.extract_file_facts(
        repo,
        commit,
        ["backend/src/handlers.py", "backend/src/pkg/service.py"],
    )
    assert warm_facts == cold_facts

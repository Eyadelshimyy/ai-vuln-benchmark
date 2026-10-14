"""Phase 17 incremental analysis.

A warm scan must match a clean scan. Changing one file recomputes that file
and leaves the rest of the parse cache in place. Secret files and credential
settings are not cache keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.parsing.model import SyntaxGraph
from app.security.cross_file import build_project
from app.security.engine import SecurityAnalysisEngine, SecurityScanResult
from app.security.incremental import (
    ANALYSIS_CACHE_GENERATION,
    AnalysisIndex,
    CachedGraph,
    analyze_incremental,
    config_identity,
    content_hash,
    is_sensitive_path,
)

LEAF_COUNT = 160


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _corpus(root: Path) -> list[Path]:
    _write(
        root,
        "app.py",
        "\n".join(
            [
                "from helpers import run_code",
                "from models import Box",
                "from nested.deep.leaf import keep",
                "import mods.m000 as first_mod",
                "",
                "alias = run_code",
                "obj = {}",
                'obj["payload"] = request.args.get("q")',
                'alias(obj["payload"])',
                "box = Box()",
                'box.run(request.args.get("id"))',
                "keep()",
                "first_mod.keep()",
                "",
            ]
        ),
    )
    _write(root, "helpers.py", "def run_code(value):\n    eval(value)\n")
    _write(
        root,
        "models.py",
        "class Box:\n    def run(self, value):\n        eval(value)\n",
    )
    _write(root, "nested/__init__.py", "")
    _write(root, "nested/deep/__init__.py", "")
    _write(root, "nested/deep/leaf.py", "def keep():\n    return 1\n")
    _write(root, "mods/__init__.py", "")
    for index in range(LEAF_COUNT):
        _write(root, f"mods/m{index:03d}.py", f"def keep():\n    return {index}\n")
    _write(root, "broken.py", "def broken(:\n    pass\n")
    _write(root, ".env", "TOKEN=not-a-real-secret\n")
    _write(root, ".env.py", 'def leak():\n    eval(request.args.get("q"))\n')
    _write(root, "id_rsa", "not-a-key\n")
    _write(root, "credentials.json", "{}\n")
    _write(root, "certs/api.pem", "not-a-certificate\n")
    return sorted(root.rglob("*"))


def _canon(repo: Path, result: SecurityScanResult) -> tuple[object, ...]:
    project = build_project(result.graphs, repo_root=repo)
    observations = tuple(
        sorted(
            (
                obs.file_path,
                obs.line,
                obs.rule_id,
                obs.vulnerability_class.value,
                obs.metadata.get("taint", ""),
                obs.metadata.get("field_path", ""),
                obs.metadata.get("analysis_incomplete", ""),
                obs.title,
            )
            for obs in result.observations
        )
    )
    diagnostics = tuple(
        sorted(
            (item.file_path, item.kind, item.rule_id, item.message) for item in result.diagnostics
        )
    )
    edges = tuple(sorted((edge.kind, edge.source_id, edge.target_id) for edge in project.edges))
    externals: list[tuple[object, ...]] = []
    for path in sorted(project.externals):
        for name in sorted(project.externals[path]):
            callee = project.externals[path][name]
            externals.append(
                (
                    path,
                    name,
                    callee.callee_file,
                    callee.callee_function,
                    callee.relationship,
                    callee.semantic_id,
                    callee.partial,
                    callee.hops,
                    callee.return_effects,
                    callee.param_sinks,
                )
            )
    findings = tuple(
        sorted(
            (
                str(finding.status),
                finding.vulnerability_class,
                finding.title,
                finding.source_location.file_path if finding.source_location else "",
                finding.source_location.line if finding.source_location else 0,
            )
            for finding in result.findings
        )
    )
    return observations, diagnostics, edges, tuple(externals), findings, project.incomplete


def _cacheable(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.suffix == ".py" and not is_sensitive_path(path.name)]


@pytest.fixture
def wide_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "taint_cross_file_budget_ms", 30_000)
    monkeypatch.setattr(settings, "taint_max_files", 400)


def test_config_identity_omits_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    before = config_identity()
    monkeypatch.setattr(settings, "secret_key", "SENTINEL_SECRET_PHASE17")
    monkeypatch.setattr(settings, "ai_api_key", "SENTINEL_TOKEN_PHASE17")
    assert config_identity() == before
    assert "SENTINEL" not in config_identity()
    monkeypatch.setattr(settings, "taint_max_field_depth", settings.taint_max_field_depth + 1)
    assert config_identity() != before


def test_sensitive_names() -> None:
    assert is_sensitive_path(".env")
    assert is_sensitive_path("pkg/.env.local")
    assert is_sensitive_path("id_rsa")
    assert is_sensitive_path("credentials.json")
    assert is_sensitive_path("secrets.json")
    assert is_sensitive_path("certs/api.pem")
    assert is_sensitive_path("server.key")
    assert not is_sensitive_path("app.py")
    assert not is_sensitive_path("helpers.py")


def test_content_hash_is_stable() -> None:
    assert content_hash("eval(value)\n") == content_hash("eval(value)\n")
    assert content_hash("eval(value)\n") != content_hash("return value\n")
    assert ANALYSIS_CACHE_GENERATION == "phase17-v1"


def test_negative_limits_stay_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(taint_max_field_depth=-1)


def test_incremental_matches_clean_scan(tmp_path: Path, wide_budget: None) -> None:
    root = tmp_path / "repo"
    paths = _corpus(root)
    py_files = [path for path in paths if path.suffix == ".py"]
    cacheable = _cacheable(py_files)
    engine = SecurityAnalysisEngine()
    index = AnalysisIndex()
    index.records[".env"] = CachedGraph(
        path=".env",
        content_hash="seed",
        parser_id="seed",
        config_id="seed",
        graph=SyntaxGraph(language="python", file_path=".env", source="", lines=()),
    )
    index.records[".env.py"] = CachedGraph(
        path=".env.py",
        content_hash="seed",
        parser_id="seed",
        config_id="seed",
        graph=SyntaxGraph(language="python", file_path=".env.py", source="", lines=()),
    )

    cold, index = analyze_incremental(engine, root, paths, index)
    clean = engine.analyze_repository(root, paths)
    assert index.misses == len(cacheable)
    assert index.hits == 0
    assert index.skipped_sensitive >= 5
    assert ".env" not in index.records
    assert ".env.py" not in index.records
    assert "id_rsa" not in index.records
    assert "credentials.json" not in index.records
    assert "certs/api.pem" not in index.records
    cold_canon = _canon(root, cold)
    assert cold_canon == _canon(root, clean)
    assert any(
        item[0] == "app.py"
        and item[3] == "dangerous_dynamic_execution"
        and item[5] == 'obj["payload"]'
        for item in cold_canon[0]
    )
    assert any(
        item[0] == "app.py" and item[3] == "dangerous_dynamic_execution" and item[5] == ""
        for item in cold_canon[0]
    )
    assert any(item[1] == "parser_error" and item[0] == "broken.py" for item in cold_canon[1])
    assert all(item[0] in {"potential", "corroborated"} for item in cold_canon[4])
    stored = index.records["helpers.py"]
    assert stored.parser_id == f"python:cpython_ast:full_ast:{ANALYSIS_CACHE_GENERATION}"
    assert stored.content_hash == content_hash((root / "helpers.py").read_text(encoding="utf-8"))
    assert stored.path == "helpers.py"

    warm, index = analyze_incremental(engine, root, paths, index)
    assert index.hits == len(cacheable)
    assert index.misses == 0
    assert _canon(root, warm) == _canon(root, clean)

    helper = root / "helpers.py"
    helper.write_text("def run_code(value):\n    return value\n", encoding="utf-8")
    changed, index = analyze_incremental(engine, root, paths, index)
    clean_changed = engine.analyze_repository(root, paths)
    assert index.misses == 1
    assert index.hits == len(cacheable) - 1
    changed_canon = _canon(root, changed)
    assert changed_canon == _canon(root, clean_changed)
    assert changed_canon != cold_canon
    assert not any(item[5] == 'obj["payload"]' for item in changed_canon[0])
    assert any(
        item[0] == "app.py" and item[3] == "dangerous_dynamic_execution"
        for item in changed_canon[0]
    )

    leaf = root / "mods" / "m010.py"
    leaf.write_text("def keep():\n    return 999\n", encoding="utf-8")
    one_leaf, index = analyze_incremental(engine, root, paths, index)
    assert index.misses == 1
    assert index.hits == len(cacheable) - 1
    assert _canon(root, one_leaf) == _canon(root, engine.analyze_repository(root, paths))
    assert any(edge[0] == "import" for edge in _canon(root, one_leaf)[2])


def test_config_change_invalidates_cache(
    tmp_path: Path, wide_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    paths = _corpus(root)
    cacheable = _cacheable([path for path in paths if path.suffix == ".py"])
    engine = SecurityAnalysisEngine()
    _, index = analyze_incremental(engine, root, paths, AnalysisIndex())
    monkeypatch.setattr(settings, "taint_max_field_depth", 1)
    _, index = analyze_incremental(engine, root, paths, index)
    assert index.misses == len(cacheable)
    assert index.hits == 0


def test_secret_change_does_not_invalidate_cache(
    tmp_path: Path, wide_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    paths = _corpus(root)
    cacheable = _cacheable([path for path in paths if path.suffix == ".py"])
    engine = SecurityAnalysisEngine()
    _, index = analyze_incremental(engine, root, paths, AnalysisIndex())
    monkeypatch.setattr(settings, "secret_key", "another-local-test-secret")
    monkeypatch.setattr(settings, "ai_api_key", "another-local-test-token")
    _, index = analyze_incremental(engine, root, paths, index)
    assert index.hits == len(cacheable)
    assert index.misses == 0

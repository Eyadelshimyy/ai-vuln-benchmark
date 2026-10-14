"""Phase 20 adversarial checks against the existing semantic engine.

These tests do not add a second analysis engine. They check that uncertainty
stays unresolved, limits finish, and static results are not verified.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.domain.findings import FindingStatus
from app.security.engine import SecurityAnalysisEngine
from app.security.incremental import config_identity


def _write(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _scan(root: Path):
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".js", ".json"}
    ]
    return SecurityAnalysisEngine().analyze_repository(root, files)


def _canon(result) -> tuple:
    observations = tuple(
        sorted(
            (
                obs.file_path,
                obs.line,
                obs.rule_id,
                obs.vulnerability_class.value,
                obs.metadata.get("taint", ""),
                obs.metadata.get("field_path", ""),
            )
            for obs in result.observations
        )
    )
    findings = tuple(
        sorted(
            (
                str(item.status),
                item.vulnerability_class,
                item.title,
                item.source_location.file_path if item.source_location else "",
                item.source_location.line if item.source_location else 0,
                item.finding_key,
            )
            for item in result.findings
        )
    )
    return observations, findings


def test_module_function_shadows_builtin_name(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'def eval(value):\n    return value\neval(request.args.get("q"))\n',
    )
    assert not _scan(tmp_path).observations
    _write(
        tmp_path,
        "app.py",
        'def eval(value):\n    exec(value)\neval(request.args.get("q"))\n',
    )
    result = _scan(tmp_path)
    assert any(obs.evidence_text.strip().startswith("exec(") for obs in result.observations)
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)


def test_comments_and_strings_are_not_sinks(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "# eval(request.args.get('q'))",
                "note = \"eval(request.args.get('q'))\"",
                "def show():",
                "    return note",
                "",
            ]
        ),
    )
    result = _scan(tmp_path)
    assert not result.observations
    assert not result.findings


def test_dynamic_key_and_dynamic_call_stay_unresolved(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "obj = {}",
                'key = "payload"',
                'obj[key] = request.args.get("q")',
                "eval(obj[key])",
                "name = 'eval'",
                "getattr(obj, name)(request.args.get('q'))",
                "",
            ]
        ),
    )
    result = _scan(tmp_path)
    assert not any(
        obs.vulnerability_class.value == "dangerous_dynamic_execution"
        for obs in result.observations
    )


def test_field_reassignment_clears_later_use(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "obj = {}",
                'obj["payload"] = request.args.get("q")',
                'obj["payload"] = "safe"',
                'eval(obj["payload"])',
                "",
            ]
        ),
    )
    result = _scan(tmp_path)
    assert not result.observations


def test_import_cycle_finishes(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "from b import go\n\ndef run(value):\n    go(value)\n")
    _write(tmp_path, "b.py", "from a import run\n\ndef go(value):\n    eval(value)\n")
    _write(tmp_path, "app.py", 'from a import run\nrun(request.args.get("q"))\n')
    result = _scan(tmp_path)
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)
    assert _canon(result) == _canon(_scan(tmp_path))


def test_mutations_change_only_the_expected_flow(tmp_path: Path) -> None:
    _write(tmp_path, "helpers.py", "def run_code(value):\n    eval(value)\n")
    app = _write(
        tmp_path, "app.py", 'from helpers import run_code\nrun_code(request.args.get("q"))\n'
    )
    found = _scan(tmp_path)
    assert any(obs.file_path == "app.py" for obs in found.observations)

    app.write_text('run_code(request.args.get("q"))\n', encoding="utf-8")
    assert not any(obs.file_path == "app.py" for obs in _scan(tmp_path).observations)

    app.write_text(
        'from helpers import run_code\nrun_code(request.args.get("q"))\n', encoding="utf-8"
    )
    _write(tmp_path, "helpers.py", "def run_code(value):\n    return value\n")
    assert not any(obs.file_path == "app.py" for obs in _scan(tmp_path).observations)

    _write(tmp_path, "helpers.py", "def other(value):\n    eval(value)\n")
    assert not any(obs.file_path == "app.py" for obs in _scan(tmp_path).observations)


def test_limits_finish_and_reject_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "app.py", 'eval(request.args.get("q"))\n')
    for name, value in (
        ("taint_max_import_depth", 0),
        ("taint_max_cross_file_edges", 0),
        ("taint_max_field_depth", 0),
        ("taint_max_field_bindings", 1),
        ("taint_max_alias_edges", 1),
        ("taint_max_files", 1),
    ):
        monkeypatch.setattr(settings, name, value)
        result = _scan(tmp_path)
        assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)
    with pytest.raises(ValidationError):
        Settings(taint_max_cross_file_edges=-1)
    with pytest.raises(ValidationError):
        Settings(taint_max_field_depth=-1)


def test_repeated_scans_match(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'obj = {}\nobj["payload"] = request.args.get("q")\neval(obj["payload"])\n',
    )
    _write(tmp_path, "helpers.py", "def keep():\n    return 1\n")
    first = _canon(_scan(tmp_path))
    second = _canon(_scan(tmp_path))
    assert first == second
    assert first[1]
    assert all(item[0] in {"potential", "corroborated"} for item in first[1])


def test_malformed_tsconfig_does_not_resolve_aliases(tmp_path: Path) -> None:
    _write(tmp_path, "tsconfig.json", "{ this is not json\n")
    _write(tmp_path, "helpers.ts", "export function run(value: string) { eval(value); }\n")
    _write(
        tmp_path,
        "app.ts",
        'import { run } from "@app/helpers";\nrun((global as any).request);\n',
    )
    result = SecurityAnalysisEngine().analyze_repository(
        tmp_path,
        [tmp_path / "helpers.ts", tmp_path / "app.ts", tmp_path / "tsconfig.json"],
    )
    assert not any(obs.file_path == "app.ts" for obs in result.observations)


def test_static_and_ai_cannot_verify_by_themselves() -> None:
    from app.domain.findings import SecurityFinding

    static = SecurityFinding.potential("Potential issue")
    assert static.status is FindingStatus.POTENTIAL
    with pytest.raises(ValueError):
        static.verify()
    ai = SecurityFinding.from_hypothesis("Maybe", "the model said so")
    assert ai.status is FindingStatus.POTENTIAL
    with pytest.raises(ValueError):
        ai.verify()


def test_secret_is_not_part_of_analysis_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    before = config_identity()
    monkeypatch.setattr(settings, "secret_key", "phase20-sentinel-secret")
    monkeypatch.setattr(settings, "ai_api_key", "phase20-sentinel-token")
    assert config_identity() == before
    assert "phase20-sentinel" not in config_identity()

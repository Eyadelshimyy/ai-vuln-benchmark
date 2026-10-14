from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "enterprise_due_diligence_gate.py"
SOURCE_COMMIT = "a" * 40


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("enterprise_due_diligence_gate", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_module()


def test_build_gate_manifest_is_currency_explicit_and_not_a_valuation_claim() -> None:
    manifest = GATE.build_gate_manifest(source_commit=SOURCE_COMMIT)

    assert manifest == {
        "currency_code": "KRW",
        "gate_name": "enterprise_due_diligence_gate",
        "legacy_gate_aliases": [
            "20b",
            "20b_product",
            "20b_product_readiness",
            "require_20b_product",
        ],
        "scenario_amount": 2_000_000_000,
        "scenario_name": "krw_2000000000_procurement_scenario",
        "schema_version": "1.0.0",
        "source_commit": SOURCE_COMMIT,
        "valuation_claim": False,
    }


def test_normalize_gate_name_accepts_canonical_hyphenated_form() -> None:
    assert GATE.normalize_gate_name(" enterprise-due-diligence-gate ") == GATE.CANONICAL_GATE_NAME


@pytest.mark.parametrize(
    "legacy_alias",
    ["20B", "20b-product", "20b_product_readiness", "require_20b_product"],
)
def test_normalize_gate_name_maps_legacy_aliases_with_warning(legacy_alias: str) -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert GATE.normalize_gate_name(legacy_alias) == GATE.CANONICAL_GATE_NAME


def test_normalize_gate_name_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported due-diligence gate name"):
        GATE.normalize_gate_name("usd_20b_readiness")


@pytest.mark.parametrize("currency_code", ["krw", " usd "])
def test_validate_currency_code_normalizes_ascii_letters(currency_code: str) -> None:
    assert GATE.validate_currency_code(currency_code) == currency_code.strip().upper()


@pytest.mark.parametrize("currency_code", ["KR", "KRWW", "K1W", "₩RW"])
def test_validate_currency_code_rejects_invalid_values(currency_code: str) -> None:
    with pytest.raises(ValueError, match="three ASCII letters"):
        GATE.validate_currency_code(currency_code)


@pytest.mark.parametrize("scenario_amount", [True, False, 0, -1, 2.5])
def test_validate_scenario_amount_rejects_non_positive_integer_values(
    scenario_amount: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GATE.validate_scenario_amount(scenario_amount)


def test_validate_scenario_amount_accepts_positive_integer() -> None:
    assert GATE.validate_scenario_amount(1) == 1


@pytest.mark.parametrize("source_commit", ["b" * 40, "c" * 64])
def test_validate_source_commit_accepts_full_sha1_and_sha256(source_commit: str) -> None:
    assert GATE.validate_source_commit(source_commit) == source_commit


@pytest.mark.parametrize(
    "source_commit",
    [
        "",
        "   ",
        "abc123",
        "a" * 39,
        "a" * 41,
        "a" * 63,
        "a" * 65,
        "A" * 40,
        "g" * 40,
        f" {SOURCE_COMMIT}",
        f"{SOURCE_COMMIT} ",
        "a" * 20 + "\n" + "a" * 19,
    ],
)
def test_validate_source_commit_rejects_noncanonical_identity(source_commit: str) -> None:
    with pytest.raises(ValueError, match="canonical lowercase full Git object identity"):
        GATE.validate_source_commit(source_commit)


def test_validate_source_commit_rejects_string_subclass_without_callbacks() -> None:
    callbacks: list[str] = []

    class HostileCommit(str):
        def strip(self, *args: object, **kwargs: object) -> str:
            callbacks.append("strip")
            return super().strip(*args, **kwargs)

    with pytest.raises(ValueError, match="canonical lowercase full Git object identity"):
        GATE.validate_source_commit(HostileCommit(SOURCE_COMMIT))

    assert callbacks == []


@pytest.mark.parametrize("valuation_claim", [True, "false"])
def test_build_gate_manifest_rejects_valuation_claims(valuation_claim: object) -> None:
    with pytest.raises(ValueError, match=r"valuation_claim|valuation claim"):
        GATE.build_gate_manifest(
            source_commit=SOURCE_COMMIT,
            valuation_claim=valuation_claim,
        )


def test_write_gate_manifest_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = GATE.build_gate_manifest(
        source_commit=SOURCE_COMMIT,
        currency_code="usd",
        scenario_amount=25,
    )
    output_path = Path("nested") / "gate.json"

    GATE.write_gate_manifest(manifest, output_path)

    written_path = tmp_path / output_path
    assert written_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(written_path.read_text(encoding="utf-8")) == manifest
    assert written_path.read_text(encoding="utf-8").splitlines()[1].strip().startswith('"currency_code"')


def test_write_gate_manifest_uses_atomic_portable_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest writes remain available when POSIX descriptor flags are absent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delattr(GATE.os, "O_NOFOLLOW", raising=False)
    manifest = GATE.build_gate_manifest(source_commit=SOURCE_COMMIT)

    GATE.write_gate_manifest(manifest, Path("portable") / "gate.json")

    assert json.loads(
        (tmp_path / "portable" / "gate.json").read_text(encoding="utf-8")
    ) == manifest


def test_write_gate_manifest_cleans_failed_portable_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed portable descriptor write must not leave an orphan manifest."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delattr(GATE.os, "O_NOFOLLOW", raising=False)
    manifest = GATE.build_gate_manifest(source_commit=SOURCE_COMMIT)
    original_fdopen = GATE.os.fdopen

    class FailingStream:
        """Wrap the real descriptor stream while failing its content write."""

        def __init__(self, file_descriptor: int, *args: object, **kwargs: object) -> None:
            self._stream = original_fdopen(file_descriptor, *args, **kwargs)

        def __enter__(self) -> "FailingStream":
            self._stream.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)

        def write(self, content: str) -> int:
            del content
            raise OSError("injected portable write failure")

    monkeypatch.setattr(GATE.os, "fdopen", FailingStream)

    with pytest.raises(ValueError, match="manifest output could not be written"):
        GATE.write_gate_manifest(manifest, Path("portable") / "gate.json")

    assert list((tmp_path / "portable").iterdir()) == []


def test_write_gate_manifest_rejects_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Output validation must prevent writes outside the invocation directory."""
    monkeypatch.chdir(tmp_path)
    manifest = GATE.build_gate_manifest(source_commit=SOURCE_COMMIT)

    with pytest.raises(ValueError, match="remain within the current working directory"):
        GATE.write_gate_manifest(manifest, Path("..") / "outside.json")

    assert not (tmp_path.parent / "outside.json").exists()


def test_write_gate_manifest_rejects_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output validation must not follow a symlinked destination directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    manifest = GATE.build_gate_manifest(source_commit=SOURCE_COMMIT)

    with pytest.raises(ValueError, match="symbolic links"):
        GATE.write_gate_manifest(manifest, Path("linked") / "gate.json")

    assert not (tmp_path / "real" / "gate.json").exists()


def test_main_rejects_directory_output_with_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI returns its stable failure payload for a directory target."""
    monkeypatch.chdir(tmp_path)

    exit_code = GATE.main(["--source-commit", SOURCE_COMMIT, "--out", "."])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert json.loads(captured.err) == {
        "error": "output path must name a regular file",
        "status": "failed",
    }
    assert captured.out == ""


def test_main_rejects_regular_file_parent_with_json_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI converts a non-directory parent failure into JSON evidence."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "parent").write_text("not a directory", encoding="utf-8")

    exit_code = GATE.main(
        ["--source-commit", SOURCE_COMMIT, "--out", str(Path("parent") / "gate.json")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert json.loads(captured.err) == {
        "error": "manifest output could not be written",
        "status": "failed",
    }
    assert captured.out == ""


def test_main_writes_canonical_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("gate.json")

    exit_code = GATE.main(
        [
            "--source-commit",
            SOURCE_COMMIT,
            "--currency-code",
            "usd",
            "--scenario-amount",
            "25",
            "--require-enterprise-due-diligence",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["gate_name"] == GATE.CANONICAL_GATE_NAME
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == {
        "gate_name": GATE.CANONICAL_GATE_NAME,
        "out": str(output_path),
        "status": "ok",
        "valuation_claim": False,
    }


def test_main_supports_deprecated_flag_during_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("legacy.json")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        exit_code = GATE.main(
            [
                "--source-commit",
                SOURCE_COMMIT,
                "--require-20b-product",
                "--out",
                str(output_path),
            ]
        )

    assert exit_code == 0
    assert len(captured) == 2
    assert all(item.category is DeprecationWarning for item in captured)
    assert json.loads(output_path.read_text(encoding="utf-8"))["valuation_claim"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_main_returns_stable_failure_payload_for_invalid_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("invalid.json")

    exit_code = GATE.main(
        [
            "--source-commit",
            SOURCE_COMMIT,
            "--currency-code",
            "invalid",
            "--out",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "currency_code must be exactly three ASCII letters",
        "status": "failed",
    }
    assert not output_path.exists()


def test_main_fails_closed_for_abbreviated_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    output_path = Path("abbreviated.json")

    exit_code = GATE.main(
        [
            "--source-commit",
            "abc123",
            "--out",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "source_commit must be a canonical lowercase full Git object identity",
        "status": "failed",
    }
    assert not output_path.exists()

"""Tests for the command entry points the gate calls."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from clave.cli import _build_parser, _train, main
from clave.training.runner import EpochRecord, TrainingRun

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _capture_info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)


def test_verify_manifest_passes_on_the_real_repository(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The gate's command succeeds on a clean tree."""
    assert main(["--root", str(ROOT), "verify-manifest"]) == 0
    assert "ok       smoke" in caplog.text


def test_verify_manifest_fails_when_bytes_changed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mutated artifact fails the gate, non-zero."""
    (tmp_path / "corpora" / "fixtures").mkdir(parents=True)
    (tmp_path / "corpora" / "fixtures" / "smoke.csv").write_text("tampered\n")
    (tmp_path / "corpora" / "manifest.toml").write_text(
        '[[artifact]]\nname = "smoke"\n'
        'source = "corpora/fixtures/smoke.csv"\nsha256 = "0000"\n'
    )
    assert main(["--root", str(tmp_path), "verify-manifest"]) == 1
    assert "digest_mismatch" in caplog.text


def test_record_digest_reports_a_digest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The command reports rather than writes."""
    (tmp_path / "corpora").mkdir()
    manifest = tmp_path / "corpora" / "manifest.toml"
    manifest.write_text('[[artifact]]\nname = "x"\nsource = "x.csv"\n')
    before = manifest.read_text()
    target = tmp_path / "x.csv"
    target.write_text("hello\n")
    assert main(["--root", str(tmp_path), "record-digest", "x", str(target)]) == 0
    assert any(len(record.message.strip()) == 64 for record in caplog.records)
    assert manifest.read_text() == before


def test_malformed_manifest_exits_non_zero(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The gate fails and names the entry."""
    (tmp_path / "corpora").mkdir()
    (tmp_path / "corpora" / "manifest.toml").write_text(
        '[[artifact]]\nname = "dup"\nsource = "a"\n\n'
        '[[artifact]]\nname = "dup"\nsource = "b"\n'
    )
    assert main(["--root", str(tmp_path), "verify-manifest"]) == 1
    assert "dup" in caplog.text


def test_sim_no_progress_is_off_unless_asked() -> None:
    """AC-STORY-07: the progress bar is the default, and --no-progress turns it off."""
    parser = _build_parser()
    assert parser.parse_args(["sim"]).no_progress is False
    assert parser.parse_args(["sim", "--no-progress"]).no_progress is True
    assert parser.parse_args(["sim"]).frames is None
    assert parser.parse_args(["sim", "--frames"]).frames is True
    assert parser.parse_args(["sim", "--no-frames"]).frames is False


def test_camera_video_is_off_unless_asked() -> None:
    """AC-CAM-01: the detection-camera video is off unless the flag is set."""
    parser = _build_parser()
    assert parser.parse_args(["sim"]).camera_video is False
    assert parser.parse_args(["sim", "--camera-video"]).camera_video is True


def test_no_command_is_an_error() -> None:
    """argparse requires a subcommand."""
    with pytest.raises(SystemExit):
        main([])


def _write_outcomes(path: Path, correct: bool) -> None:
    """Write a records file whose objects are all handled the same way.

    The records are a synthetic fixture. They exercise the command and say
    nothing about any model, because no model has been trained.

    Args:
        path: Where to write.
        correct: Whether each object is classified and routed correctly.
    """
    import json

    records = [
        {
            "object_id": f"obj-{index}",
            "true_class": "M-01",
            "predicted_class": "M-01" if correct else "M-04",
            "picked": True,
            "routed_channel": "CH-PET" if correct else "CH-PLASTIC-OTHER",
            "seen_instance": index % 2 == 0,
            "decision_latency_seconds": 0.1,
            "cycle_time_seconds": 0.5,
        }
        for index in range(40)
    ]
    path.write_text(
        json.dumps(
            {"provenance": "synthetic fixture, not a measurement", "outcomes": records}
        )
    )


def test_validate_run_prints_a_report(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The harness runs from one command."""
    outcomes = tmp_path / "outcomes.json"
    _write_outcomes(outcomes, correct=True)
    main(
        [
            "--root",
            str(ROOT),
            "validate-run",
            "--outcomes",
            str(outcomes),
        ]
    )
    captured = caplog.text
    assert "## What was run" in captured
    assert "## What was concluded" in captured
    assert "synthetic fixture, not a measurement" in captured


def test_validate_run_exits_non_zero_when_a_gate_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unmet gate is a failure, not a reported number."""
    outcomes = tmp_path / "outcomes.json"
    _write_outcomes(outcomes, correct=False)
    assert main(["--root", str(ROOT), "validate-run", "--outcomes", str(outcomes)]) == 1
    assert "FAIL" in caplog.text


def test_validate_run_reports_a_malformed_records_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad record fails the command naming what was wrong."""
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text('{"provenance": "fixture", "outcomes": [{"object_id": "a"}]}')
    assert main(["--root", str(ROOT), "validate-run", "--outcomes", str(outcomes)]) == 1
    assert "true_class" in caplog.text


def test_a_held_out_score_is_logged_beside_the_epoch_loss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An epoch that was scored names that score, and one that was not does not."""
    text = (
        "training:\n"
        "  candidate: resnet50-baseline\n"
        "  dataset: datasets/corpus/train\n"
        "  checkpoints: runs\n"
        "  epochs: 1\n"
        "  batch_size: 1\n"
        "  learning_rate: 0.0001\n"
        "  seed: 0\n"
        "  samples_per_crossing: 3\n"
        "  window_exit_meters: 1.034\n"
        "  act_chunk_size: 10\n"
        "  accumulation_steps: 1\n"
        "  class_balance: none\n"
        "  validation_dataset: datasets/corpus/validation\n"
        "  patience: 3\n"
    )
    config = tmp_path / "train.yml"
    config.write_text(text)
    run = TrainingRun(
        candidate="resnet50-baseline",
        seed=0,
        config_digest="a" * 64,
        dataset_digest="b" * 64,
        machine="test",
        threads=1,
        environment={},
        input_side_pixels=224,
        resident_limit_bytes=1,
        epochs=[
            EpochRecord(0, 0.2, 1.0, None),
            EpochRecord(1, 0.1, 1.5, 0.25),
        ],
        completed=True,
    )

    def _resolved(*_args: object, **_kwargs: object) -> Path:
        return tmp_path

    with (
        patch("clave.data.locate.resolve_dataset", side_effect=_resolved),
        patch("clave.training.runner.train", return_value=run),
    ):
        assert _train(ROOT, config, None) == 0
    assert "epoch  0" in caplog.text
    assert "held-out   0.2500" in caplog.text
    first = next(line for line in caplog.text.splitlines() if "epoch  0" in line)
    assert "held-out" not in first


def test_corpus_and_validate_are_subcommands() -> None:
    """The corpus and its scorer are their own commands."""
    parser = _build_parser()
    corpus = parser.parse_args(["corpus", "--config", "configs/data/corpus.yml"])
    assert corpus.command == "corpus"
    scored = parser.parse_args(
        [
            "validate",
            "--dataset",
            "a" * 64,
            "--candidate",
            "resnet50-baseline",
            "--checkpoint",
            "model.pt",
        ]
    )
    assert scored.command == "validate"

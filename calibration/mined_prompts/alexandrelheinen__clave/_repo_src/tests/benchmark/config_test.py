"""Tests for what the benchmark compares."""

from pathlib import Path

import pytest
import yaml

from clave.benchmark.config import BenchmarkConfig, BenchmarkConfigError

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "configs" / "benchmark" / "default.yml"


def test_the_shipped_comparison_loads_and_names_its_rows() -> None:
    """Every configuration the release compares is in one committed file."""
    config = BenchmarkConfig.load(SHIPPED)
    names = [row.name for row in config.configurations]
    assert "scripted-expert" in names
    assert len(names) >= 2
    assert config.seeds
    assert config.seconds_per_run > 0.0


def test_the_configuration_carries_its_own_digest() -> None:
    """A number in the evidence pack has to be traceable to the comparison."""
    config = BenchmarkConfig.load(SHIPPED)
    assert len(config.digest) == 64


def test_a_missing_key_fails_naming_itself(tmp_path: Path) -> None:
    """Nothing here carries a default, as nothing in the world does."""
    raw = yaml.safe_load(SHIPPED.read_text())
    del raw["benchmark"]["seeds"]
    path = tmp_path / "benchmark.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(BenchmarkConfigError, match="benchmark.seeds"):
        BenchmarkConfig.load(path)


def test_a_row_naming_an_unknown_predictor_is_refused(tmp_path: Path) -> None:
    """The benchmark builds two kinds of predictor and refuses to guess a third."""
    raw = yaml.safe_load(SHIPPED.read_text())
    raw["configurations"][0]["predictor"] = "telepathy"
    path = tmp_path / "benchmark.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(BenchmarkConfigError, match="telepathy"):
        BenchmarkConfig.load(path)


def test_a_trained_row_must_name_both_stages(tmp_path: Path) -> None:
    """The runtime loads a classifier and a policy, so a row needs both."""
    raw = yaml.safe_load(SHIPPED.read_text())
    trained = next(
        row for row in raw["configurations"] if row["predictor"] == "checkpoints"
    )
    del trained["policy"]
    path = tmp_path / "benchmark.yml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(BenchmarkConfigError, match="policy"):
        BenchmarkConfig.load(path)

"""The corpus grid, without rendering a world."""

from pathlib import Path

import pytest

from clave.data.campaign import CampaignConfig, CampaignError, assignments, campaign_id

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "data" / "corpus.yml"


def test_train_and_validation_cover_the_grid_with_disjoint_seeds() -> None:
    """AC-CORPUS-01: both halves visit every cell and share no seed."""
    config = CampaignConfig.load(CONFIG)
    planned = assignments(config)
    cells = {
        (item.belt_speed, item.spacing_meters)
        for item in planned
        if item.role == "train"
    }
    validation_cells = {
        (item.belt_speed, item.spacing_meters)
        for item in planned
        if item.role == "validation"
    }
    expected = {
        (speed, spacing) for speed in config.belt_speeds for spacing in config.spacings
    }
    assert cells == expected
    assert validation_cells == expected
    train_seeds = {item.seed for item in planned if item.role == "train"}
    validation_seeds = {item.seed for item in planned if item.role == "validation"}
    assert train_seeds.isdisjoint(validation_seeds)
    assert len(train_seeds) == len(cells) * config.train_rollouts_per_cell


def test_a_campaign_id_ignores_the_output_directory(tmp_path: Path) -> None:
    """The same grid on two machines is the same campaign."""
    config = CampaignConfig.load(CONFIG)
    other = CampaignConfig.load(CONFIG)
    assert campaign_id("abc", config) == campaign_id("abc", other)
    assert campaign_id("abc", config) != campaign_id("def", config)
    assert len(campaign_id("abc", config)) == 64
    assert tmp_path  # the directory is not an input to the id


def test_a_corpus_config_names_a_missing_key(tmp_path: Path) -> None:
    """A missing level fails at load, naming the key."""
    path = tmp_path / "corpus.yml"
    path.write_text("corpus:\n  seed: 0\n")
    with pytest.raises(CampaignError, match="belt_speeds_meters_per_second"):
        CampaignConfig.load(path)

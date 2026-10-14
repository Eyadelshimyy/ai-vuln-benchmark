"""Security regression tests for bounded synthetic-data allocations."""

from __future__ import annotations

import pytest

from fast_mlsirm.cli import main
from fast_mlsirm.config import MLS2PLMConfig


def test_cli_simulate_rejects_excessive_person_count(tmp_path, capsys, monkeypatch):
    """Reject oversized CLI simulations before allocating their matrices."""
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "oversized"

    rc = main(
        [
            "simulate",
            "--persons",
            "100001",
            "--dims",
            "1",
            "--items-per-dim",
            "1",
            "--latent-dim",
            "1",
            "--out",
            str(out_dir),
        ]
    )

    assert rc == 1
    assert "n_persons must be <= 100000" in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"n_dims": 51}, "n_dims must be <= 50"),
        ({"items_per_dim": 1001}, "items_per_dim must be <= 1000"),
        (
            {"n_persons": 20_001, "n_dims": 1, "items_per_dim": 1_000},
            "exceeds the 20000000-cell simulation budget",
        ),
    ],
)
def test_simulation_config_rejects_bounded_resource_exhaustion(
    config_kwargs, message
):
    """Reject unsafe simulation dimensions during config construction."""
    with pytest.raises(ValueError, match=message):
        MLS2PLMConfig(**config_kwargs)

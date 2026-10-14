"""The proof-of-concept resident-memory budget."""

from pathlib import Path

import pytest

from clave.training.memory import (
    MemoryBudget,
    MemoryBudgetError,
    parse_resident_bytes,
    require_within_budget,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_configured_budget_is_one_and_a_quarter_gibibyte_at_224() -> None:
    """The proof of concept is the file the training command loads."""
    budget = MemoryBudget.load(ROOT / "configs" / "training" / "memory.yml")
    assert budget.resident_limit_bytes == 1342177280
    assert budget.input_side_pixels == 224


def test_vm_rss_is_read_as_bytes() -> None:
    """The status file reports kibibytes."""
    assert parse_resident_bytes("VmRSS:\t2048 kB\n") == 2048 * 1024


def test_ac_mem_03_a_run_stops_when_resident_memory_exceeds_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-MEM-03: over the ceiling, the next batch does not start."""
    monkeypatch.setattr(
        "clave.training.memory.current_resident_bytes", lambda: 2 * 1024**3
    )
    with pytest.raises(MemoryBudgetError, match="over"):
        require_within_budget(1024**3)


def test_ac_mem_04_a_checkpoint_records_the_input_side(tmp_path: Path) -> None:
    """AC-MEM-04: validation can resize to the side this run trained on."""
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from clave.training.config import TrainingConfig
    from clave.training.runner import _checkpoint
    from clave.training.sample import FrameSample
    from clave.training.selection import fresh_selection

    config = replace(
        TrainingConfig.load(ROOT / "configs" / "training" / "default.yml"),
        checkpoints=tmp_path,
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sample = FrameSample(
        samples_per_crossing=1,
        dataset_digest="a" * 64,
        along_travel_meters=0.92,
        capture_interval_seconds=0.2,
        seed=0,
        picks={},
    )
    _checkpoint(
        config,
        model,
        optimizer,
        epoch=0,
        input_side_pixels=224,
        sample=sample,
        weights=None,
        selection=fresh_selection(),
        validation_sample=None,
    )
    state = torch.load(tmp_path / "resnet50-baseline.pt", weights_only=False)
    assert state["input_side_pixels"] == 224

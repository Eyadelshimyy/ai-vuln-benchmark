"""``require_overlays`` — a declared text layer is checked across every batch up front."""

from __future__ import annotations

from pathlib import Path

import pytest

from womblex.process.text_overlay import MissingOverlayError, load_overlay, require_overlays


def _bases(d: Path, n: int) -> list[Path]:
    return [d / f"batch-{i:04d}" for i in range(1, n + 1)]


def test_elements_is_verbatim_and_needs_no_sidecar(tmp_path: Path):
    require_overlays(_bases(tmp_path, 2), "elements")


def test_names_every_batch_missing_the_layer(tmp_path: Path):
    bases = _bases(tmp_path, 3)
    (tmp_path / "batch-0002.normalised_text.parquet").write_bytes(b"")

    with pytest.raises(MissingOverlayError) as exc:
        require_overlays(bases, "normalised")
    assert "batch-0001, batch-0003" in str(exc.value)
    assert "2 of 3" in str(exc.value)


def test_unknown_text_source_is_a_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        require_overlays(_bases(tmp_path, 1), "cleaned")


def test_required_load_raises_the_same_error(tmp_path: Path):
    with pytest.raises(MissingOverlayError):
        load_overlay(tmp_path / "batch-0001", "spellfix", required=True)

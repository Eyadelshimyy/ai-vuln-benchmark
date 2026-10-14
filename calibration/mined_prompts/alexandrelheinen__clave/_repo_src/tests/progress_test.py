"""The frame bar used while training and validating."""

import logging
from datetime import datetime

import numpy as np
import pytest

from clave.data.dataset import DatasetDescription, DatasetFile
from clave.progress import Progress, examples_in, frame_total


def _description() -> DatasetDescription:
    """Two archives in the train part, one of them unlisted."""
    files = (
        DatasetFile("rollout_000.npz", "a" * 64, 10, 150, 0),
        DatasetFile("rollout_001.npz", "b" * 64, 10, 150, 1),
    )
    return DatasetDescription(
        digest="c" * 64,
        seed=0,
        config_digest="d" * 64,
        example_count=300,
        parts={"train": ["rollout_000", "rollout_001"]},
        composition={},
        files=files,
    )


def test_frame_total_sums_one_part() -> None:
    """The bar length is the frames the archives say they hold."""
    assert frame_total(_description(), "train") == 300
    assert frame_total(_description(), "validation") is None


def test_a_missing_archive_record_has_no_known_length() -> None:
    """A part that names a file the description does not list cannot be sized."""
    description = _description()
    broken = DatasetDescription(
        digest=description.digest,
        seed=description.seed,
        config_digest=description.config_digest,
        example_count=description.example_count,
        parts={"train": ["rollout_000", "rollout_009"]},
        composition={},
        files=description.files,
    )
    assert frame_total(broken, "train") is None


def test_examples_in_reads_a_tensor_a_list_and_an_action_chunk() -> None:
    """Each adapter's batch shape counts as frames on the bar."""
    images = np.zeros((4, 3, 8, 8))
    assert examples_in((images, None)) == 4
    assert examples_in((["a", "b"], None)) == 2
    assert examples_in({"observation.image": images}) == 4


def test_the_count_line_names_frames_done_and_time_remaining(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run that is not a terminal still gets a count and a remaining time.

    A second update inside the quiet window does not add another line. The
    next line appears once that window has passed, and it counts the whole
    run, not only the epoch on screen.
    """
    clock = {"now": 1000.0}
    monkeypatch.setattr("clave.progress._open_bar", lambda *_args: None)
    monkeypatch.setattr("clave.progress.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clave.progress._wall_now",
        lambda: datetime(2026, 9, 25, 22, 43, 30),
    )
    expected = (
        "epoch 1/10  10/100 frame  loss 0.5000  rss 900 MiB  "
        "1.00 frame/s  epoch 1m 30s remaining  run 16m 30s remaining  "
        "finishes 2026-09-25 23:00"
    )
    with (
        caplog.at_level(logging.INFO, logger="clave.progress"),
        Progress(100, "epoch 1/10", "frame", repeats=10, repeat_index=0) as bar,
    ):
        clock["now"] = 1010.0
        bar.update(10, loss=0.5, rss_mib=900.0)
        clock["now"] = 1011.0
        bar.update(1, loss=0.4, rss_mib=901.0)
        messages = [record.getMessage() for record in caplog.records]
        assert messages == [expected]
        clock["now"] = 1025.0
        bar.update(5, loss=0.25, rss_mib=800.0)
        messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "16/100" in messages[-1]
    assert "run " in messages[-1]


def test_a_live_bar_keeps_the_estimate_and_does_not_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A drawn bar holds the finish clock, and the count line stays quiet."""
    clock = {"now": 1000.0}
    drawn = _DrawnBar()
    monkeypatch.setattr("clave.progress.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clave.progress._wall_now",
        lambda: datetime(2026, 9, 25, 22, 43, 30),
    )
    monkeypatch.setattr("clave.progress._open_bar", lambda *_args: drawn)
    with (
        caplog.at_level(logging.INFO, logger="clave.progress"),
        Progress(100, "epoch 1/10", "frame", repeats=10, repeat_index=0) as bar,
    ):
        clock["now"] = 1010.0
        bar.update(10, loss=0.5, rss_mib=900.0)
        assert drawn.postfix["finishes"] == "2026-09-25 23:00"
        assert "1m 30s" in drawn.postfix["epoch"]
        assert "16m 30s" in drawn.postfix["run"]
        clock["now"] = 1025.0
        bar.update(5, loss=0.25, rss_mib=800.0)
    assert caplog.records == []
    assert drawn.n == 15


class _DrawnBar:
    """A bar that stays on one line, so a test can read its postfix."""

    def __init__(self) -> None:
        """Start empty."""
        self.disable = False
        self.n = 0
        self.total: int | None = 100
        self.postfix: dict[str, str] = {}

    def set_postfix(self, postfix: dict[str, str], refresh: bool = True) -> None:
        """Remember the latest postfix."""
        self.postfix = dict(postfix)

    def update(self, n: int) -> None:
        """Advance the counter."""
        self.n += n

    def refresh(self) -> None:
        """Stay put."""

    def close(self) -> None:
        """Leave the last postfix readable."""


def test_the_bar_keeps_the_latest_metric_and_closes() -> None:
    """Loss and agreement stay visible as frames complete."""
    with Progress(10, "epoch 1/10", "frame") as bar:
        bar.update(4, loss=0.5)
        bar.update(2, loss=0.25)
        assert bar.completed == 6
        assert bar.metrics == {"loss": 0.25}
    assert bar.completed == 6

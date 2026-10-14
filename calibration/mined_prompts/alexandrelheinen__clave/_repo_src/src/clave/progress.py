"""A frame bar for a long training or validation run.

The bar is the live view. The run record and the validation score stay the
numbers a later command prints. tqdm is optional at import time so a tree
checked without it still trains.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from clave.data.dataset import DatasetDescription

LOGGER = logging.getLogger(__name__)

# A captured log has no carriage-return bar. This is how often a fresh count
# line is written once frames are moving. Short enough to revise an estimate,
# long enough that one epoch does not flood the log.
_COUNT_INTERVAL_SECONDS = 15.0


def frame_total(description: DatasetDescription, part: str) -> int | None:
    """Frames stored for one split part.

    Args:
        description: A dataset description already read from disk.
        part: Split part name, such as `train` or `validation`.

    Returns:
        The sum of archive frame counts, or None when the part or a file
        record is missing. None leaves the bar without a known length.
    """
    members = description.parts.get(part)
    if members is None:
        return None
    by_name = {item.name: item for item in description.files}
    total = 0
    for rollout_id in members:
        recorded = by_name.get(f"{rollout_id}.npz")
        if recorded is None:
            return None
        total += recorded.frame_count
    return total if total > 0 else None


def examples_in(batch: Any) -> int:
    """How many examples one training batch holds.

    Classification and policy batches lead with the image tensor. Detection
    batches lead with a list of images. An action-chunking batch is a
    dictionary keyed the way the policy names an observation.

    Args:
        batch: One batch yielded by the candidate's adapter.

    Returns:
        The example count.
    """
    payload = batch["observation.image"] if isinstance(batch, dict) else batch[0]
    shape = getattr(payload, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return len(payload)


class Progress:
    """Counts frames and keeps the latest metric on the bar.

    One bar covers the epoch on screen: its description is the epoch, and its
    counter is the frames of that epoch. The bar is drawn when standard error
    is a terminal, and it carries the rate, the time left, and the local clock
    time the whole run should finish. A count line carries those same facts
    when the bar is not drawn. The line and the bar are never written together.

    Attributes:
        completed: Frames reported so far.
        metrics: The latest values shown beside the bar.
    """

    def __init__(
        self,
        total: int | None,
        desc: str,
        unit: str,
        *,
        repeats: int = 1,
        repeat_index: int = 0,
    ) -> None:
        """Open a bar.

        Args:
            total: Frames expected, or None when the length is unknown.
            desc: Short label, such as the epoch.
            unit: What one step is, usually a frame.
            repeats: How many times this length is walked, such as the epoch
                count. One means the bar is the whole run.
            repeat_index: Which repeat is on screen, starting at zero.
        """
        self.completed = 0
        self.metrics: dict[str, float] = {}
        self._total = total
        self._desc = desc
        self._unit = unit
        self._repeats = repeats
        self._repeat_index = repeat_index
        self._started = time.monotonic()
        self._reported = self._started
        self._bar = _open_bar(total, desc, unit)

    def update(self, n: int = 1, **metrics: float) -> None:
        """Advance the bar and replace the metrics it shows.

        Args:
            n: Frames just finished. Zero refreshes the metrics only.
            metrics: Named values, printed with four digits.
        """
        self.completed += n
        if metrics:
            self.metrics = dict(metrics)
        elapsed = time.monotonic() - self._started
        if _is_drawn(self._bar):
            postfix = _bar_postfix(
                self.completed,
                self._total,
                self._unit,
                elapsed,
                self.metrics,
                repeats=self._repeats,
                repeat_index=self._repeat_index,
                now=_wall_now(),
            )
            if postfix:
                self._bar.set_postfix(postfix, refresh=n == 0)
            if n:
                self._bar.update(n)
            return
        self._report_count()
        if self._bar is not None and n:
            self._bar.update(n)

    def _report_count(self) -> None:
        """Write one count line when no bar is drawn.

        The quiet window still applies, so a captured log is not a line per frame.
        """
        if self.completed <= 0:
            return
        now = time.monotonic()
        quiet = now - self._reported < _COUNT_INTERVAL_SECONDS
        if self._reported != self._started and quiet:
            return
        elapsed = now - self._started
        if elapsed <= 0:
            return
        LOGGER.info(
            "%s",
            _format_progress(
                self._desc,
                self.completed,
                self._total,
                self._unit,
                elapsed,
                self.metrics,
                repeats=self._repeats,
                repeat_index=self._repeat_index,
                now=_wall_now(),
            ),
        )
        self._reported = now

    def close(self) -> None:
        """Finish the bar.

        A detector skips a frame with nothing visible, and a policy skips a
        frame with no demonstration, so the count can stop short of the
        archive length. The finished bar shows the frames that were actually
        used.
        """
        if self._bar is None:
            return
        total = self._bar.total
        if total is not None and self._bar.n < total:
            self._bar.total = self._bar.n
            self._bar.refresh()
        self._bar.close()
        self._bar = None

    def __enter__(self) -> Progress:
        """Return the bar."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the bar when the block leaves."""
        self.close()


def _wall_now() -> datetime:
    """Local time the finish clock is measured from."""
    return datetime.now().astimezone()


def _run_remaining_seconds(
    completed: int,
    total: int | None,
    elapsed_seconds: float,
    *,
    repeats: int,
    repeat_index: int,
) -> float | None:
    """Seconds until the last repeat ends, at the rate seen so far.

    Args:
        completed: Frames finished in this repeat.
        total: Frames expected in one repeat, or None when unknown.
        elapsed_seconds: Seconds since this repeat started.
        repeats: How many times this length is walked.
        repeat_index: Which repeat is on screen, starting at zero.

    Returns:
        The remaining seconds, or None when the rate is not known yet.
    """
    if total is None or elapsed_seconds <= 0 or completed <= 0:
        return None
    rate = completed / elapsed_seconds
    left = (total - completed) / rate
    later = repeats - repeat_index - 1
    if later > 0:
        return left + (total * later) / rate
    return left


def _finish_text(remaining_seconds: float, now: datetime) -> str:
    """Local clock time a run should finish, to the minute.

    Args:
        remaining_seconds: Seconds still to run.
        now: The clock the estimate is added to.

    Returns:
        The year, month, day, hour, and minute.
    """
    moment = now + timedelta(seconds=remaining_seconds)
    rounded = (moment + timedelta(seconds=30)).replace(second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%d %H:%M")


def _is_drawn(bar: Any) -> bool:
    """Whether the bar is painting a line that a log would scroll away."""
    return bar is not None and not bool(getattr(bar, "disable", False))


def _bar_postfix(
    completed: int,
    total: int | None,
    unit: str,
    elapsed_seconds: float,
    metrics: dict[str, float],
    *,
    repeats: int,
    repeat_index: int,
    now: datetime,
) -> dict[str, str]:
    """The estimate shown on the bar, in place of a count line.

    Args:
        completed: Frames finished in this repeat.
        total: Frames expected in one repeat, or None when unknown.
        unit: What one step is, usually a frame.
        elapsed_seconds: Seconds since this repeat started.
        metrics: Latest named values.
        repeats: How many times this length is walked.
        repeat_index: Which repeat is on screen, starting at zero.
        now: Local time the finish clock is added to.

    Returns:
        Postfix fields. The time fields are absent until a rate exists.
    """
    postfix: dict[str, str] = {}
    for key, value in metrics.items():
        if key == "rss_mib":
            postfix["rss"] = f"{value:.0f} MiB"
        else:
            postfix[key] = f"{value:.4f}"
    if elapsed_seconds <= 0 or completed <= 0:
        return postfix
    rate = completed / elapsed_seconds
    postfix["rate"] = f"{rate:.2f} {unit}/s"
    remaining = _run_remaining_seconds(
        completed,
        total,
        elapsed_seconds,
        repeats=repeats,
        repeat_index=repeat_index,
    )
    if remaining is None or total is None:
        return postfix
    left = (total - completed) / rate
    if repeats > 1:
        postfix["epoch"] = _span(left)
        later = repeats - repeat_index - 1
        if later > 0:
            postfix["run"] = _span(remaining)
    else:
        postfix["left"] = _span(remaining)
    postfix["finishes"] = _finish_text(remaining, now)
    return postfix


def _format_progress(
    desc: str,
    completed: int,
    total: int | None,
    unit: str,
    elapsed_seconds: float,
    metrics: dict[str, float],
    *,
    repeats: int,
    repeat_index: int,
    now: datetime,
) -> str:
    """One count line: frames done, rate, time left, and the finish clock.

    Args:
        desc: Short label, such as the epoch.
        completed: Frames finished in this repeat.
        total: Frames expected in one repeat, or None when unknown.
        unit: What one step is, usually a frame.
        elapsed_seconds: Seconds since this repeat started.
        metrics: Latest named values, in the order they were reported.
        repeats: How many times this length is walked.
        repeat_index: Which repeat is on screen, starting at zero.
        now: Local time the finish clock is added to.

    Returns:
        A single log line.
    """
    rate = completed / elapsed_seconds
    count = f"{completed}/{total}" if total is not None else str(completed)
    parts = [desc, f"{count} {unit}", _metric_text(metrics), f"{rate:.2f} {unit}/s"]
    remaining = _run_remaining_seconds(
        completed,
        total,
        elapsed_seconds,
        repeats=repeats,
        repeat_index=repeat_index,
    )
    if remaining is not None and total is not None:
        left = (total - completed) / rate
        if repeats > 1:
            parts.append(f"epoch {_span(left)} remaining")
            later = repeats - repeat_index - 1
            if later > 0:
                parts.append(f"run {_span(remaining)} remaining")
        else:
            parts.append(f"{_span(remaining)} remaining")
        parts.append(f"finishes {_finish_text(remaining, now)}")
    return "  ".join(parts)


def _metric_text(metrics: dict[str, float]) -> str:
    """Format the latest metrics for a count line."""
    parts: list[str] = []
    for key, value in metrics.items():
        if key == "rss_mib":
            parts.append(f"rss {value:.0f} MiB")
        else:
            parts.append(f"{key} {value:.4f}")
    return "  ".join(parts)


def _span(seconds: float) -> str:
    """A duration as hours and minutes, or minutes and seconds."""
    whole = max(0, int(round(seconds)))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _open_bar(total: int | None, desc: str, unit: str) -> Any:
    """Open a tqdm bar, or nothing when the library is absent."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        disable=None,
        dynamic_ncols=True,
        leave=True,
    )

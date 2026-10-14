"""What one debug run records, and the text and JSON written beside it.

A number in a run is traceable when the revision, the configuration digests
and the command that produced it sit next to the figures.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clave.errors import ClaveError
from clave.world import config

GRASPED_METERS = 0.010
"""How far a visit has to raise an object before the jaw was holding it.

Measured against where the object was lying when the plan was made, not
against the belt: a settled object already rests with its center above the
surface, and by a different amount for every mesh in the set. Ten
millimetres is above the millimetre of settling jitter and far below the
50 mm clearance a retreat lifts to, so nothing sits near the boundary.
"""


class DebugRunError(ClaveError):
    """The debug run cannot be set up from this configuration."""


@dataclass(frozen=True)
class DebugRunReport:
    """What one debug run did.

    Attributes:
        captures: Frames the tracker was shown.
        tracks: Tracks open when it finished.
        reorders: How many captures rebuilt the queue, by the trigger that
            did it. A rebuild because a track appeared is the design
            working; a rebuild because an anchor moved is the estimate having
            genuinely shifted. Counting them together hides whether the
            anchors damp anything.
        head_churn: How many captures swapped the head of the queue while the
            previous head was still there to be served. This is the figure
            that decides whether the arm can work through the queue: a
            rebuild that keeps its head costs nothing, and one that does not
            sends the arm somewhere else mid-traverse.
        served: Tracks the arm finished a visit to.
        missed: Tracks no interception existed for, which the arm gave up
            rather than chased. Distinct from a fault: the pose was fine and
            the timing was not.
        arrivals: How far the flange was from the pose each completed visit
            asked for, in meters. Measured at the end of the dwell under a
            stepped profile and at the instant the jaw reaches the object
            under a planned one. Read its median rather than its mean: the
            distribution is a tight cluster with occasional strays, and a
            mean over seventeen visits moved from 3 mm to 43 mm on one of
            them.
        feed_rate: What the line was asked to carry, in objects per second.
        measured_rate: What it achieved over the controller's window, at the
            end of the run.
        belt_speed: What the feed controller ended up commanding, in meters
            per second. Reported beside the two rates so a reader can tell a
            line that held its setpoint from one that saturated trying.
        lifts: How far each completed visit raised the object that ended up
            nearest the jaw, in meters, against where that object was lying
            when the plan was made. This is the figure that says whether the
            jaw held anything: a failed grasp reads near zero however well
            the arm flew.
        placed: How many objects went down a chute, by channel. A place is
            the object's centre crossing below the belt surface inside a
            mouth's footprint, which is where the system's responsibility
            ends and the plant's begins.
        misrouted: How many of those went down a channel their material
            class does not route to. This is the figure `max_misroute_rate`
            has gated with no way to produce.
        jaw_gaps: How far the nearest object was from the pinch site at the
            instant the jaw shut, in meters, one per visit. It separates the
            two ways a pick fails: a large gap is a visit tick that arrived
            somewhere the object was not, and a small gap with no lift is a
            grasp that could not hold.
        profile: Which task profile the run used, because a distance to a
            tracked pose and a distance to a descended pose are not the same
            measurement.
        closest_approach: The nearest the flange ever came to a pose the
            tracking phase asked for, in meters, or None when it never
            tracked.
        closest_live: The nearest it came to where that object actually was
            at the same instant, in meters, or None. The gap between the two
            is the staleness of a pose decided once per capture while the
            belt keeps moving, and is what an interception has to close.
        faults: Refusals the controller recorded, with the track each was
            about.
        phase: The phase the arm ended in.
        drawn: Markers standing in the world on the final frame.
        geoms: Marker geoms the final frame carried, which is more than
            `drawn` because a jaw is a shaft and two pads, and the park pose
            adds two of its own.
        frames_written: Frames written.
        output: Where they went.
        video_path: The playable file, or None when none was asked for or no
            encoder was found.
        camera_video_path: The detection-camera file, with a box on each
            object the segmentation covered, or None when none was asked for
            or no encoder was found.
        telemetry_path: The CSV telemetry file, or None when telemetry was not
            requested.
        metadata_path: Where the revision and the configuration digests were
            recorded, so a number in this report can be traced to the tree and
            the configuration it came from.
        min_jaw_clearance: The smallest distance any of the jaw's collision
            geometry came to the belt surface, in meters. Negative means the
            geometry was inside the belt.
        belt_contacts: How many ticks had a contact between the jaw and the
            belt.
        worst_tool_tilt_degrees: The largest departure of the tool's axis from
            the belt normal, in degrees.
        abandoned: Visits given up before the descent began, with the reason
            each was given up. Distinct from `missed`: an abandoned visit is a
            plan that was active and that the freshest estimate contradicted,
            and this is the only figure that tells that apart from an arm that
            never saw the object.
        worst_aim_drift: The largest distance a active plan was found
            aiming away from where the freshest estimate put the object, in
            meters, or None when no visit was ever re-aimed. This is the figure
            that catches a plan about to descend onto bare belt, and no arrival
            error can show it: the arm meets a stale pose perfectly.
        grasp_yaw_errors: How far the commanded tool yaw stood from the object
            body's own yaw when the jaw shut, in degrees, folded into the 90
            degrees a jaw is symmetric about, one per grab that had an object
            under it. Empty unless the run was driven from ground truth,
            because a tracker estimate has no body to compare against.
        lurches: How fast the flange's vertical acceleration peaked over each
            planned visit, in meters per second squared, one per visit. The
            figure behind "the arm jumps when it lifts": a grasp that slips
            unloads the arm mid-lift and that is an acceleration, not a pose.
        climbs: The fastest the flange rose during each planned visit, in
            meters per second, one per visit.
        report_path: Where this report was written beside the run's other
            artifacts, so the figures survive the terminal that printed them.
        windowed: Whether a live window was opened.
        reason: Why no window was opened, when none was.
        render_mode: Render mode used ('lite' or 'demo'/'realistic').
    """

    captures: int
    tracks: int
    reorders: dict[str, int]
    head_churn: int
    served: tuple[int, ...]
    missed: tuple[int, ...]
    arrivals: tuple[float, ...]
    lifts: tuple[float, ...]
    jaw_gaps: tuple[float, ...]
    placed: dict[str, int]
    misrouted: int
    feed_rate: float
    measured_rate: float
    belt_speed: float
    profile: str
    closest_approach: float | None
    closest_live: float | None
    faults: tuple[tuple[int | None, str], ...]
    phase: str
    drawn: int
    geoms: int
    frames_written: int
    output: Path
    video_path: Path | None
    telemetry_path: Path | None
    metadata_path: Path
    min_jaw_clearance: float | None
    belt_contacts: int
    worst_tool_tilt_degrees: float | None
    abandoned: tuple[tuple[int, str], ...] = ()
    worst_aim_drift: float | None = None
    grasp_yaw_errors: tuple[float, ...] = ()
    lurches: tuple[float, ...] = ()
    climbs: tuple[float, ...] = ()
    report_path: Path | None = None
    windowed: bool = False
    reason: str | None = None
    ground_truth: bool = False
    camera_video_path: Path | None = None
    render_mode: str = "lite"


def _report_document(report: DebugRunReport) -> dict[str, Any]:
    """Return a report as JSON a reader can diff against another run's.

    AC-MOVE-55. The report used to be printed and nothing else, so a run's
    figures died with the terminal they were printed on: the run this branch
    was opened for lost its own grasp distances that way, and reconstructing
    them from telemetry meant inferring which of thirteen objects each visit
    had been about. Written beside the artifacts instead, in both a machine
    form and the text a reader already knows how to scan.

    Args:
        report: The report.

    Returns:
        Its fields as JSON-compatible values: paths as strings, tuples as
        lists at any depth, and everything else as it is.
    """
    return {name: _plain(getattr(report, name)) for name in report.__dataclass_fields__}


def _plain(value: Any) -> Any:
    """Return one of a report's values as data JSON can carry.

    Args:
        value: The value.

    Returns:
        The same value with paths read as strings and every tuple -- at any
        depth, because a visit given up is a pair inside a tuple -- read as a
        list.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def report_lines(report: DebugRunReport) -> tuple[str, ...]:
    """Return the run's report as the lines it is printed and filed as.

    One formatter for both, because a report that reads one way on a terminal
    and another way in the file beside it is two reports, and a reader
    comparing them has no way to tell which one the numbers came from.

    Args:
        report: The report.

    Returns:
        The lines, in the order a reader scans them: what the run was, what the
        line did, and what the arm did about it.
    """
    lines: list[str] = []
    lines.append(f"  render mode     {report.render_mode}")
    if report.ground_truth:
        lines.append("  targets         ground truth (MuJoCo physics)")
    lines.append(f"  captures        {report.captures}")
    lines.append(f"  tracks open     {report.tracks}")
    lines.append(f"  markers on last {report.drawn}, as {report.geoms} geoms")
    rebuilt = ", ".join(f"{why} {count}" for why, count in report.reorders.items())
    lines.append(f"  queue rebuilt   {rebuilt}, of {report.captures} captures")
    lines.append(
        f"  head swapped    {report.head_churn} times with the old head still there"
    )
    lines.append(f"  profile         {report.profile}")
    lines.append(
        f"  feed rate       {report.measured_rate:.3f} of "
        f"{report.feed_rate:.3f} objects/s, belt at {report.belt_speed:.3f} m/s"
    )
    lines.append(f"  visits served   {len(report.served)} {list(report.served)}")
    if report.missed:
        lines.append(f"  no interception {len(report.missed)} {list(report.missed)}")
    if report.abandoned:
        each = "; ".join(f"{track}: {why}" for track, why in report.abandoned)
        lines.append(f"  visits given up {len(report.abandoned)} ({each})")
    if report.worst_aim_drift is not None:
        lines.append(
            f"  aim checked     worst {report.worst_aim_drift * 1000:.0f} mm between "
            f"the plan and the freshest estimate"
        )
    if report.arrivals:
        # Median rather than mean, and the count of outliers beside it. The
        # mean lied: sixteen visits at 2 to 4 mm and one at 688 mm reads as
        # "43 mm", which describes no visit that happened.
        ranked = sorted(report.arrivals)
        middle = ranked[len(ranked) // 2] * 1000
        stray = sum(1 for gap in ranked if gap > 0.050)
        lines.append(
            f"  arrival error   median {middle:.1f} mm, worst "
            f"{ranked[-1] * 1000:.1f} mm, {stray} over 50 mm"
        )
    if report.jaw_gaps:
        each = ", ".join(f"{gap * 1000:.0f}" for gap in report.jaw_gaps)
        lines.append(f"  jaw to object   {each} mm when the jaw shut")
    if report.grasp_yaw_errors:
        each = ", ".join(f"{error:.1f}" for error in report.grasp_yaw_errors)
        lines.append(f"  tool turned off {each} deg from the object's own axis")
    if report.placed or report.misrouted:
        total = sum(report.placed.values())
        each = ", ".join(f"{c}: {n}" for c, n in sorted(report.placed.items()))
        lines.append(f"  placed          {total} down a chute ({each})")
        lines.append(f"  misrouted       {report.misrouted} of {total}")
    if report.lifts:
        held = sum(1 for lift in report.lifts if lift >= GRASPED_METERS)
        each = ", ".join(f"{lift * 1000:.0f}" for lift in report.lifts)
        lines.append(
            f"  grasps held     {held} of {len(report.lifts)}, lifts {each} mm"
        )
    if report.lurches:
        each = ", ".join(f"{value:.1f}" for value in report.lurches)
        lines.append(f"  vertical lurch  {each} m/s2 at worst per visit")
    lines.append(f"  faults          {len(report.faults)}")
    for track_id, why in report.faults:
        lines.append(f"    track {track_id}: {why}")
    lines.append(f"  ended in        {report.phase}")
    if report.closest_approach is not None:
        lines.append(f"  to commanded    {report.closest_approach * 1000:.0f} mm")
    if report.closest_live is not None:
        lines.append(f"  to the object   {report.closest_live * 1000:.0f} mm")
    if report.min_jaw_clearance is not None:
        lines.append(
            f"  jaw clearance   {report.min_jaw_clearance * 1000:.1f} mm above the "
            f"belt, {report.belt_contacts} ticks in contact"
        )
    if report.worst_tool_tilt_degrees is not None:
        lines.append(
            f"  tool tilt       {report.worst_tool_tilt_degrees:.1f} deg off the "
            f"belt normal at worst"
        )
    lines.append(f"  frames written  {report.frames_written} to {report.output}")
    if report.video_path is not None:
        lines.append(f"  video           {report.video_path}")
    if report.camera_video_path is not None:
        lines.append(f"  camera video    {report.camera_video_path}")
    if report.telemetry_path is not None:
        lines.append(f"  telemetry       {report.telemetry_path}")
    lines.append(f"  metadata        {report.metadata_path}")
    if report.report_path is not None:
        lines.append(f"  report          {report.report_path}")
    return tuple(lines)


def _git(root: Path, *args: str) -> str | None:
    """Return what git says about the tree, or None when it cannot be asked.

    A debug run has to work in a checkout without git and in a copy of the
    source with no repository at all, so an unanswerable question is recorded
    as unanswerable rather than raised.

    Args:
        root: The repository root.
        args: The arguments after `git`.

    Returns:
        The stripped standard output, or None.
    """
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _run_metadata(
    root: Path,
    paths: dict[str, Path],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return what a reader needs to trace this run's numbers to their inputs.

    AC-MOVE-52, extended by AC-MOVE-54. A revision alone is not enough: a dirty
    tree has no revision that describes it, so that is recorded beside it, and
    a digest per configuration is what lets a number be compared against a tree
    that has since moved on. This exists because a run's artifacts could not be
    attributed to a tree, and the phase durations in its telemetry did not
    reproduce from the configuration the tree carried.

    That much still did not make a run reproducible. The seed, the duration and
    the view are not configuration: they are the command somebody typed, and a
    run whose seed is not recorded cannot be run again even by whoever wrote
    it. A reader trying to reproduce a published run hit exactly that: the
    re-run at seed 0 produced a different world, and there was nothing in the
    artifacts to say whether the seed or the world was the difference.

    Args:
        root: The repository root.
        paths: Where each configuration lives, by name.
        parameters: What the run was asked for -- the seed, the duration, the
            view, the flags -- or None for a caller that has none to record.

    Returns:
        The record, in the shape the rest of the repository records a run's
        inputs: a digest per configuration, where it came from, and what the
        run was asked to do.
    """
    from clave.experiment.run import config_digest

    status = _git(root, "status", "--porcelain")
    return {
        "revision": _git(root, "rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "config_digests": {
            name: config_digest(config.load(path)) for name, path in paths.items()
        },
        "config_paths": {name: str(path) for name, path in paths.items()},
        "parameters": dict(parameters or {}),
    }

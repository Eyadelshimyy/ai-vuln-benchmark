"""Running one named scenario, start to finish, from one command."""

from __future__ import annotations

from pathlib import Path

from clave.demo.scenario import Scenario
from clave.demo.video import available as encoder_available
from clave.runtime.inference import CheckpointPredictor, Predictor, ScriptedPredictor
from clave.runtime.loop import RunReport, RuntimeSettings, run, write_record


def predictor_for(
    scenario: Scenario, settings: RuntimeSettings, root: Path
) -> Predictor:
    """Build what proposes a pick for this scenario.

    Args:
        scenario: The scenario being run.
        settings: The runtime settings, carrying the checkpoint directory.
        root: Repository root.

    Returns:
        The predictor.
    """
    if scenario.predictor == "scripted":
        return ScriptedPredictor()
    return CheckpointPredictor(
        checkpoints=root / settings.checkpoints,
        perception=scenario.perception or "",
        policy=scenario.policy or "",
        presence_floor=settings.presence_floor,
        association_radius=settings.association_radius_meters,
    )


def play(root: Path, scenario: Scenario, runtime: Path, output: Path) -> RunReport:
    """Run one scenario and write its record.

    Args:
        root: Repository root.
        scenario: What to run.
        runtime: The runtime configuration to start from.
        output: Where the record and the video go.

    Returns:
        The report.
    """
    settings = RuntimeSettings.load(root / runtime)
    adjusted = RuntimeSettings(
        **{**vars(settings), "seed": scenario.seed, "seconds": scenario.seconds}
    )
    report = run(
        root,
        adjusted,
        predictor_for(scenario, adjusted, root),
        output / "runtime",
        scenario.publish_to_ros,
        scenario.video,
    )
    write_record(output / f"{scenario.name}.json", report)
    return report


def summary(scenario: Scenario, report: RunReport) -> str:
    """Render what the scenario did.

    Args:
        scenario: What was run.
        report: What came out of it.

    Returns:
        The text, without a trailing newline.
    """
    counters = report.counters
    overridden = sum(
        counters.get(key, 0)
        for key in (
            "overridden_reach",
            "overridden_belt_surface",
            "overridden_belt_extent",
        )
    )
    lines = [
        f"  scenario        {scenario.name}",
        f"  predictor       {report.predictor}",
        f"  belt speed      {report.belt_speed:.3f} m/s, "
        f"budget {report.budget_seconds:.3f} s per object",
        f"  presented       {len(report.presented)} objects reached the arm",
        f"  proposals       {report.proposals} over {report.frames} frames",
        f"  published       {counters.get('published', 0)} decisions",
        f"  overridden      {overridden} by the safety layer",
        f"  latency         median {report.percentile(0.50) * 1000:.1f} ms, "
        f"p99 {report.percentile(0.99) * 1000:.1f} ms",
    ]
    if report.published_to_ros or scenario.publish_to_ros:
        lines.append(f"  on ROS 2        {report.published_to_ros} decisions")
    if report.ros_unavailable_reason is not None:
        lines.append(f"  NO ROS          {report.ros_unavailable_reason}")
    if report.video_path is not None:
        speed = scenario.video.speed if scenario.video is not None else 1.0
        lines.append(
            f"  video           {report.video_path}, {report.video_frames} frames "
            f"at {speed:.1f}x simulated time"
        )
    elif scenario.video is not None and not encoder_available():
        lines.append(
            "  NO VIDEO        ffmpeg is not installed, so nothing was recorded"
        )
    return "\n".join(lines)

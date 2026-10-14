"""Running every configuration and turning each run into scored records.

The population a record describes is an object that entered the reachable
window, which is the definition the validation harness already uses: an object
that never came within reach is outside this harness's boundary, since nothing
could have been done about it.

Every record carries `picked = False`. This harness scores the proposal loop
and does not run the jaw, so pick success and cycle time stay unmeasured, and
the evidence pack says so instead of publishing a rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clave.benchmark.config import BenchmarkConfig, Configuration
from clave.runtime.loop import RunReport, RuntimeSettings
from clave.validation.outcomes import ObjectOutcome, OutcomeSet


@dataclass(frozen=True)
class ConfigurationResult:
    """What one row of the comparison produced.

    Attributes:
        configuration: The row.
        available: Whether it ran at all.
        unavailable_reason: Why it did not, when it did not.
        presented: Objects that entered the reachable window, across all seeds.
        decided: Objects a decision was published for.
        proposals: Proposals sent across the boundary.
        overridden: Proposals the safety layer refused.
        outside_window: Decisions about an object that had not entered the
            reachable window, which the safety layer is the last defense
            against.
        simulated_seconds: Simulated time summed over the seeds.
        outcomes: The records, ready for the validation harness.
        latencies_seconds: Frame to published decision, over every proposal.
        belt_speeds: The speed each seed drew, so a throughput figure can be
            read against it.
    """

    configuration: Configuration
    available: bool
    unavailable_reason: str | None
    presented: int
    decided: int
    proposals: int
    overridden: int
    outside_window: int
    simulated_seconds: float
    outcomes: OutcomeSet
    latencies_seconds: tuple[float, ...]
    belt_speeds: tuple[float, ...]

    @property
    def decision_rate(self) -> float:
        """Decisions published per simulated second."""
        if self.simulated_seconds <= 0.0:
            return 0.0
        return self.proposals / self.simulated_seconds

    @property
    def coverage(self) -> float:
        """Share of presented objects a decision was published for."""
        if self.presented == 0:
            return 0.0
        return self.decided / self.presented


def outcomes_for(
    name: str, reports: list[tuple[int, RunReport]]
) -> tuple[OutcomeSet, int]:
    """Turn runs into one record per object that entered the window.

    An object decided about more than once keeps the last decision, because that
    is the one a line would act on. An object never decided about carries no
    predicted class and no decision latency: it was presented and the system
    said nothing, which is a miss rather than an instant decision.

    Args:
        name: Configuration name, carried into the provenance.
        reports: The runs, each with the seed that produced it.

    Returns:
        The records, and how many decisions named an object that had not
        entered the reachable window.
    """
    records: list[ObjectOutcome] = []
    outside = 0
    for seed, report in reports:
        latest = {record.object_id: record for record in report.decisions}
        presented = dict(report.presented)
        outside += sum(1 for key in latest if key not in presented)
        for object_id, true_class in report.presented:
            decision = latest.get(object_id)
            records.append(
                ObjectOutcome(
                    object_id=f"seed{seed}-object{object_id}",
                    true_class=true_class,
                    predicted_class=None
                    if decision is None
                    else decision.predicted_class,
                    picked=False,
                    routed_channel=None,
                    # The benchmark runs the world the models trained on, so
                    # every instance is a seen one. Generalization is not
                    # measured here and the report says so rather than letting
                    # a gate pass on an empty partition.
                    seen_instance=True,
                    decision_latency_seconds=(
                        None if decision is None else decision.latency_seconds
                    ),
                    cycle_time_seconds=None,
                )
            )
    return OutcomeSet(f"{name}, simulated", records), outside


def build_predictor(
    configuration: Configuration, settings: RuntimeSettings, root: Path
) -> Any:
    """Build what proposes a pick for one configuration.

    Args:
        configuration: The row being run.
        settings: The runtime settings, carrying the checkpoint directory.
        root: Repository root.

    Returns:
        The predictor.

    Raises:
        InferenceError: If a checkpoint or a library the row needs is absent.
    """
    from clave.runtime.inference import CheckpointPredictor, ScriptedPredictor

    if configuration.predictor == "scripted":
        return ScriptedPredictor()
    return CheckpointPredictor(
        checkpoints=root / settings.checkpoints,
        perception=configuration.perception or "",
        policy=configuration.policy or "",
        presence_floor=settings.presence_floor,
        association_radius=settings.association_radius_meters,
    )


def run_configuration(
    root: Path,
    config: BenchmarkConfig,
    configuration: Configuration,
    directory: Path,
) -> ConfigurationResult:
    """Run one row over every seed and score it.

    A row whose checkpoint or library is absent is recorded with its reason and
    returns an empty result, so one missing dependency does not cost the whole
    comparison.

    Args:
        root: Repository root.
        config: The comparison this row belongs to.
        configuration: The row.
        directory: Where each run's sockets and configuration go.

    Returns:
        The result, available or not.
    """
    from clave.runtime import loop

    settings = RuntimeSettings.load(root / config.runtime)
    try:
        predictor = build_predictor(configuration, settings, root)
    except Exception as error:
        return _unavailable(configuration, str(error))

    reports: list[tuple[int, RunReport]] = []
    for seed in config.seeds:
        adjusted = RuntimeSettings(
            **{
                **vars(settings),
                "seed": seed,
                "seconds": config.seconds_per_run,
            }
        )
        reports.append(
            (
                seed,
                loop.run(
                    root,
                    adjusted,
                    predictor,
                    directory / configuration.name.replace(" ", "-") / f"seed{seed}",
                ),
            )
        )

    outcomes, outside = outcomes_for(configuration.name, reports)
    decided = sum(
        1 for record in outcomes.outcomes if record.predicted_class is not None
    )
    return ConfigurationResult(
        configuration=configuration,
        available=True,
        unavailable_reason=None,
        presented=len(outcomes),
        decided=decided,
        proposals=sum(report.proposals for _, report in reports),
        overridden=sum(
            report.counters.get("overridden_reach", 0)
            + report.counters.get("overridden_belt_surface", 0)
            + report.counters.get("overridden_belt_extent", 0)
            for _, report in reports
        ),
        outside_window=outside,
        simulated_seconds=config.seconds_per_run * len(config.seeds),
        outcomes=outcomes,
        latencies_seconds=tuple(
            value for _, report in reports for value in report.latencies_seconds
        ),
        belt_speeds=tuple(report.belt_speed for _, report in reports),
    )


def _unavailable(configuration: Configuration, reason: str) -> ConfigurationResult:
    """Record a row that could not run, without stopping the comparison."""
    return ConfigurationResult(
        configuration=configuration,
        available=False,
        unavailable_reason=reason,
        presented=0,
        decided=0,
        proposals=0,
        overridden=0,
        outside_window=0,
        simulated_seconds=0.0,
        outcomes=OutcomeSet(f"{configuration.name}, unavailable", []),
        latencies_seconds=(),
        belt_speeds=(),
    )

"""The boundary every reading crosses before it can reach a track.

`GroundTruth` sits in the same union as real evidence so that simulation
supplies labels through the path real sensors use, which is what keeps the
training pipeline from depending on a shape that exists only in simulation. The
price of that decision is this gate.

A runtime configured for hardware refuses the simulator's label, and refuses it
here rather than in each consumer, because a rule repeated in four places is a
rule that holds in three. The refusal names the source that offered it: on a
real line the useful question is which process is still emitting simulated
labels, not merely that one did.

The deployment is stated and never inferred. A runtime that guessed would guess
the permissive answer, and the one time it mattered it would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from clave.errors import ClaveError
from clave.tracker.evidence import Evidence, Role
from clave.tracker.sensors import SensorSpec


class IntakeError(ClaveError):
    """A reading is refused at the boundary rather than folded into a track."""


class Deployment(Enum):
    """What kind of line this runtime is driving.

    The only distinction that matters to the boundary is whether the simulator
    is a legitimate source of labels.
    """

    SIMULATED = "simulated"
    HARDWARE = "hardware"


@dataclass(frozen=True)
class Intake:
    """The one door into the tracker.

    Attributes:
        deployment: Whether the simulator may supply labels.
        sensors: The installed sensors, used to check that a reading's
            provenance names something the line actually declares.
    """

    deployment: Deployment
    sensors: tuple[SensorSpec, ...]

    def accept(self, evidence: Evidence) -> Evidence:
        """Admit one reading, or refuse it naming why.

        Args:
            evidence: What an adapter produced.

        Returns:
            The same reading, once it has been admitted. It is returned rather
            than validated in place so that a caller holding an admitted
            reading got it from here, which is what makes this the only door
            rather than the recommended one.

        Raises:
            IntakeError: If the deployment is hardware and the reading is the
                simulator's label; if the reading names a sensor the line does
                not declare; or if it names a sensor that does not produce its
                role.
        """
        if isinstance_ground_truth(evidence):
            if self.deployment is Deployment.HARDWARE:
                raise IntakeError(
                    f"{evidence.source_id!r} offered a simulator label to a "
                    f"runtime configured for hardware, where no such label "
                    f"can be true"
                )
            # Ground truth comes from the world rather than from an
            # installation, so it has no entry in `cameras` and could not have
            # one. That exemption is what the branch above exists to close.
            return evidence

        declared = {sensor.source_id: sensor for sensor in self.sensors}
        sensor = declared.get(evidence.source_id)
        if sensor is None:
            raise IntakeError(
                f"{evidence.source_id!r} is not a sensor this line declares, "
                f"so its reading cannot be audited"
            )
        if sensor.role is not evidence.role:
            raise IntakeError(
                f"{evidence.source_id!r} produces the {sensor.role.value!r} "
                f"role and this reading claims {evidence.role.value!r}"
            )
        return evidence


def isinstance_ground_truth(evidence: Evidence) -> bool:
    """Return whether a reading carries the simulator's label.

    Written against the role rather than the payload type, because the role is
    what an operator reads in a report and the envelope already refuses a
    disagreement between the two.

    Args:
        evidence: The reading to classify.

    Returns:
        Whether it is ground truth.
    """
    return evidence.role is Role.GROUND_TRUTH

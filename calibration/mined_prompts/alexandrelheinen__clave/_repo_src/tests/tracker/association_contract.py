"""What any associator has to satisfy, whoever wrote it.

Deliberately not named `*_test.py`. This is a conformance suite rather than a
test module: `perception-record` runs it against `SimulatorIdentity` and
`learned-tracker` runs the identical clauses against its trained model, so the
two are held to one contract rather than to two descriptions of one.

Adding a clause here is a change to the seam and should be read as one.
"""

from __future__ import annotations

import numpy as np

from clave.tracker.association import Association, Associator, Cue
from clave.tracker.belt_frame import Footprint


def a_footprint(x: float = -1.0, y: float = 0.0) -> Footprint:
    """A box resting on the belt."""
    return Footprint(
        center=np.asarray((x, y, 0.93), dtype=np.float64),
        major_extent=0.10,
        minor_extent=0.06,
        yaw=0.0,
    )


def check_an_associator(associator: Associator) -> None:
    """Run every clause of the seam against one implementation.

    Args:
        associator: The implementation under test.

    Raises:
        AssertionError: Naming the clause that failed.
    """
    _names_itself(associator)
    _opens_a_track_when_there_is_nothing_to_join(associator)
    _is_a_pure_function_of_its_arguments(associator)
    _returns_a_track_it_was_offered_or_none(associator)


def _names_itself(associator: Associator) -> None:
    """What decided an association is recorded in the run report."""
    assert associator.name, "an associator states what it is"
    assert isinstance(associator.name, str)


def _opens_a_track_when_there_is_nothing_to_join(associator: Associator) -> None:
    """An observation with no open track opens one.

    Returning some track at any distance is what would let an observation of
    empty belt join a real object's history.
    """
    cue = Cue(observed_at_nanos=1_000, footprint=a_footprint())
    decision = associator.associate(cue, ())
    assert isinstance(decision, Association)
    assert decision.track_id is None


def _is_a_pure_function_of_its_arguments(associator: Associator) -> None:
    """The same cue and the same tracks give the same answer, twice.

    An implementation that read a clock, drew a random number or kept state
    between calls would make a replay irreproducible, and the tracker is what
    holds state.
    """
    cue = Cue(observed_at_nanos=2_000, footprint=a_footprint(x=-0.5))
    first = associator.associate(cue, ())
    second = associator.associate(cue, ())
    assert first == second


def _returns_a_track_it_was_offered_or_none(associator: Associator) -> None:
    """An association names a track that exists, or opens a new one."""
    cue = Cue(observed_at_nanos=3_000, footprint=a_footprint())
    decision = associator.associate(cue, ())
    assert decision.track_id is None, "no tracks were offered, so none can be named"

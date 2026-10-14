"""What proposes a pick, and the identity problem underneath it.

Two predictors share one interface. The scripted one is the expert the policies
were trained to imitate, and it exists so the loop can be exercised on a machine
with no deep learning framework installed. The checkpoint one loads the models
trained at v0.7.0 and is what the reported figures come from.

Neither tracks. CLAVE has no tracker, so the identity a proposal carries comes
from the simulator by associating the predicted point with the nearest labeled
object. A real line needs a tracker and this is where it plugs in; until then,
every identity in a run is ground truth and no reported number should be read as
evidence that identity was recovered from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from clave.data.examples import ObjectLabel
from clave.data.expert import decide
from clave.errors import ClaveError
from clave.taxonomy import MATERIAL_CLASSES


class InferenceError(ClaveError):
    """A checkpoint is missing, or names a candidate other than the one asked."""


@dataclass(frozen=True)
class Prediction:
    """What a predictor proposes for one frame.

    Attributes:
        object_id: Identity of the object, associated from the simulator.
        material_class: Taxonomy identifier of the form `M-NN`.
        confidence: How sure the classifier is, 0.0 to 1.0.
        point: Proposed pick point in belt frame meters.
    """

    object_id: int
    material_class: str
    confidence: float
    point: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", np.asarray(self.point, dtype=np.float64))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prediction):
            return NotImplemented
        return (
            self.object_id == other.object_id
            and self.material_class == other.material_class
            and self.confidence == other.confidence
            and bool(np.allclose(self.point, other.point, rtol=0.0, atol=1e-12))
        )


class Predictor(Protocol):
    """Anything that turns one observation into a proposed pick."""

    @property
    def name(self) -> str:
        """What produced the prediction, recorded in the run report."""

    def predict(
        self,
        frame: NDArray[np.uint8],
        arm_joints: tuple[float, ...],
        labels: tuple[ObjectLabel, ...],
        window_exit: float,
    ) -> Prediction | None:
        """Propose a pick, or decline.

        Args:
            frame: The captured frame, height by width by channel.
            arm_joints: The manipulator's joint angles.
            labels: What the world holds this frame, used for identity alone.
            window_exit: Belt coordinate where the reachable window ends.

        Returns:
            A prediction, or None when nothing should be picked.
        """


def associate(
    point: NDArray[np.float64],
    labels: tuple[ObjectLabel, ...],
    radius: float,
) -> ObjectLabel | None:
    """Find the object a predicted point most plausibly refers to.

    Args:
        point: The predicted pick point.
        labels: Objects the world holds this frame.
        radius: How far an association may reach, in meters.

    Returns:
        The nearest object within the radius, or None when the prediction
        matches nothing. Returning None rather than the nearest object at any
        distance is what keeps a prediction pointing at empty belt from being
        published under some real object's identity.
    """
    if not labels:
        return None
    places = np.asarray([label.position for label in labels], dtype=np.float64)
    gaps = np.linalg.norm(places - np.asarray(point, dtype=np.float64), axis=1)
    at = int(np.argmin(gaps))
    return labels[at] if float(gaps[at]) <= radius else None


class ScriptedPredictor:
    """The expert the policies were trained to imitate.

    It sees the world rather than the frame, so it is not a model and proves
    nothing about perception. It exists so the boundary, the safety layer and
    the publisher can be exercised without a deep learning framework, and as a
    control: the difference between its latency and a checkpoint's is the cost
    of inference alone.
    """

    @property
    def name(self) -> str:
        """The name recorded in the run report."""
        return "scripted-expert"

    def predict(
        self,
        frame: NDArray[np.uint8],
        arm_joints: tuple[float, ...],
        labels: tuple[ObjectLabel, ...],
        window_exit: float,
    ) -> Prediction | None:
        """Propose the reachable object with least time remaining."""
        del frame, arm_joints
        chosen = decide(labels, window_exit)
        if chosen is None:
            return None
        return Prediction(
            object_id=chosen.object_id,
            material_class=chosen.material_class,
            confidence=1.0,
            point=chosen.position,
        )


class CheckpointPredictor:
    """The trained models, loaded from the checkpoints v0.7.0 wrote.

    Perception and policy are separate candidates because they were trained
    separately: the classifier says what is on the belt and the policy says
    where to reach. The classifier abstains when no class clears the presence
    floor, which is how a frame with nothing on it produces no proposal.
    """

    def __init__(
        self,
        checkpoints: Path,
        perception: str,
        policy: str,
        presence_floor: float,
        association_radius: float,
    ):
        """Load both checkpoints.

        Args:
            checkpoints: Directory the training runs wrote to.
            perception: Registry name of the classifier.
            policy: Registry name of the pick policy.
            presence_floor: Probability below which the classifier abstains.
            association_radius: How far an identity association may reach.

        Raises:
            InferenceError: If a checkpoint is absent or holds another
                candidate.
        """
        self._perception_name = perception
        self._policy_name = policy
        self._presence_floor = presence_floor
        self._association_radius = association_radius
        self._perception = _load(checkpoints, perception)
        self._policy = _load(checkpoints, policy)

    @property
    def name(self) -> str:
        """The name recorded in the run report."""
        return f"{self._perception_name} + {self._policy_name}"

    def predict(
        self,
        frame: NDArray[np.uint8],
        arm_joints: tuple[float, ...],
        labels: tuple[ObjectLabel, ...],
        window_exit: float,
    ) -> Prediction | None:
        """Classify the frame and regress a pick point from it."""
        del window_exit
        import torch

        from clave.candidates.fixture import STATE_DIMENSION

        image = torch.from_numpy(
            np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))
        ).unsqueeze(0)
        state = torch.zeros((1, STATE_DIMENSION), dtype=torch.float32)
        width = min(len(arm_joints), STATE_DIMENSION)
        if width:
            state[0, :width] = torch.tensor(arm_joints[:width])

        with torch.no_grad():
            probabilities = torch.sigmoid(self._perception(image))[0]
            index = int(torch.argmax(probabilities))
            confidence = float(probabilities[index])
            if confidence < self._presence_floor:
                return None
            action = self._act(image, state)[:3]

        point = np.asarray(
            (float(action[0]), float(action[1]), float(action[2])), dtype=np.float64
        )
        associated = associate(point, labels, self._association_radius)
        if associated is None:
            return None
        return Prediction(
            object_id=associated.object_id,
            material_class=MATERIAL_CLASSES[index].id,
            confidence=confidence,
            point=point,
        )

    def _act(self, image: Any, state: Any) -> Any:
        """Ask the policy where to pick.

        An action chunking policy answers differently from a feedforward one: it
        produces a chunk and dequeues from it, so the queue is reset first. A
        cached action is a decision about a frame the belt has already carried
        away.
        """
        import torch.nn.functional as functional

        from clave.candidates.fixture import FRAME_HEIGHT, FRAME_WIDTH

        if self._policy_name != "act":
            return self._policy(image, state)[0]
        self._policy.reset()
        return self._policy.select_action(
            {
                "observation.image": functional.interpolate(
                    image, size=(FRAME_HEIGHT, FRAME_WIDTH), mode="bilinear"
                ),
                "observation.state": state,
            }
        )[0]


def _load(checkpoints: Path, candidate: str) -> Any:
    """Build a candidate and restore its trained weights.

    Args:
        checkpoints: Directory the training runs wrote to.
        candidate: Registry name.

    Returns:
        The model, in evaluation mode.

    Raises:
        InferenceError: If the checkpoint is absent or holds another candidate.
    """
    import torch

    from clave.candidates.registry import REGISTRY

    path = checkpoints / f"{candidate}.pt"
    if not path.is_file():
        raise InferenceError(
            f"{path} does not exist. Train it first: "
            f"python -m clave.cli train --candidate {candidate}"
        )
    state = torch.load(path, weights_only=False)
    if state.get("candidate") != candidate:
        raise InferenceError(
            f"{path} holds a checkpoint for {state.get('candidate')!r}, "
            f"not {candidate!r}"
        )
    for entry, _ in REGISTRY:
        if entry.spec.name == candidate:
            loaded = _rebuild(entry, candidate, state)
            if not loaded.loaded:
                raise InferenceError(
                    f"{candidate} cannot be loaded here: {loaded.unavailable_reason}"
                )
            model: Any = loaded.model
            model.load_state_dict(state["model"])
            model.eval()
            return model
    raise InferenceError(f"no candidate named {candidate!r} in the registry")


def _rebuild(entry: Any, candidate: str, state: dict[str, Any]) -> Any:
    """Build the architecture the checkpoint was trained as.

    A name is not always enough. ACT's shape depends on its action chunk, and
    loading weights trained at one chunk into a model built at another either
    fails loudly or, worse, loads a subset. The chunk travels in the checkpoint
    so the reader does not have to guess it.

    Args:
        entry: The registry candidate.
        candidate: Its name.
        state: The loaded checkpoint.

    Returns:
        The loaded candidate, ready for its weights.

    Raises:
        InferenceError: If the checkpoint predates the field it needs.
    """
    if candidate != "act":
        return entry.load()

    from clave.candidates.base import Candidate
    from clave.candidates.policy import ACT, build_act

    chunk = state.get("act_chunk_size")
    if chunk is None:
        raise InferenceError(
            "this act checkpoint does not record the action chunk it was "
            "trained with, so the architecture cannot be rebuilt. Retrain it: "
            "python -m clave.cli train --candidate act"
        )
    return Candidate(ACT, lambda: build_act(int(chunk))).load()

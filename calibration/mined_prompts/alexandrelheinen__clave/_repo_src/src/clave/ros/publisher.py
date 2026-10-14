"""Publishing one decision on a ROS 2 topic.

The mapping onto `vision_msgs/Detection3D` is stated in
`crates/clave-decision/contract/ros-decision.md`, which is what a subscriber
implements against. Two properties of this module are deliberate.

It computes every value it will publish before it touches ROS, in
[fields_for], so the mapping is testable on a machine with no installation. And
it imports ROS inside the function that needs it, so importing this module costs
nothing and a machine without ROS gets a sentence rather than a traceback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clave.ros.decisions import NANOS_PER_SECOND, DecisionError, PublishedDecision
from clave.ros.decisions import decode as decode_decision

TOPIC = "/clave/pick_decisions"
"""Where decisions are published."""

QUEUE_DEPTH = 10
"""How many decisions the publisher keeps for a subscriber that falls behind.

A decision whose window has closed is useless, and the window travels in the
message, so a late subscriber can discard rather than act. Depth is therefore a
buffer against jitter rather than a promise of delivery.
"""

CHANNEL_PREFIX = "channel:"
"""What marks the routing hypothesis, against the taxonomy's `M-` prefix."""


@dataclass(frozen=True)
class DetectionFields:
    """Every value one published message carries, before ROS is involved.

    Attributes:
        frame_id: The coordinate frame the point is expressed in.
        stamp_sec: Whole seconds of the instant the object becomes reachable.
        stamp_nanosec: Remaining nanoseconds of that instant.
        object_id: The tracker's identity, as a string.
        class_id: Taxonomy identifier of the form `M-NN`.
        confidence: Classifier confidence behind that class.
        channel_id: The routing result, as `channel:<n>`.
        position: Pick point in belt frame meters.
        orientation: Pick yaw as a quaternion, ordered x, y, z, w.
        window_seconds: How long the object stays reachable.
    """

    frame_id: str
    stamp_sec: int
    stamp_nanosec: int
    object_id: str
    class_id: str
    confidence: float
    channel_id: str
    position: NDArray[np.float64]
    orientation: tuple[float, float, float, float]
    window_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectionFields):
            return NotImplemented
        return (
            self.frame_id == other.frame_id
            and self.stamp_sec == other.stamp_sec
            and self.stamp_nanosec == other.stamp_nanosec
            and self.object_id == other.object_id
            and self.class_id == other.class_id
            and self.confidence == other.confidence
            and self.channel_id == other.channel_id
            and bool(np.allclose(self.position, other.position, rtol=0.0, atol=1e-12))
            and self.orientation == other.orientation
            and self.window_seconds == other.window_seconds
        )


def fields_for(decision: PublishedDecision, frame_id: str) -> DetectionFields:
    """Compute what one message carries.

    The yaw becomes a quaternion about the belt normal, because a ROS pose
    carries an orientation rather than an angle, and a planar pick is a rotation
    about z alone.

    Args:
        decision: The decision as it arrived on the wire.
        frame_id: The coordinate frame to name in the header.

    Returns:
        The values, with nothing left to decide at publication time.
    """
    half = decision.yaw_radians / 2.0
    return DetectionFields(
        frame_id=frame_id,
        stamp_sec=decision.window_start_nanos // NANOS_PER_SECOND,
        stamp_nanosec=decision.window_start_nanos % NANOS_PER_SECOND,
        object_id=str(decision.object_id),
        class_id=decision.material_class,
        confidence=decision.confidence,
        channel_id=f"{CHANNEL_PREFIX}{decision.channel}",
        position=decision.point,
        orientation=(0.0, 0.0, math.sin(half), math.cos(half)),
        window_seconds=decision.window_seconds,
    )


class RosUnavailable(Exception):
    """ROS 2 or its message packages are not importable here."""


def _imports() -> tuple[Any, Any, Any]:
    """Import ROS, or say what is missing.

    Returns:
        The `rclpy` module, the `Detection3DArray` type, and the
        `Detection3D` type.

    Raises:
        RosUnavailable: Naming the package that could not be imported.
    """
    try:
        import rclpy
        from vision_msgs.msg import Detection3D, Detection3DArray
    except ImportError as error:
        raise RosUnavailable(
            f"{error.name} is not importable. Source a ROS 2 installation, "
            "such as: source /opt/ros/jazzy/setup.bash"
        ) from error
    return rclpy, Detection3DArray, Detection3D


class DecisionPublisher:
    """A ROS 2 node that republishes what the runtime published.

    It takes the bytes the Rust side put on the wire rather than the proposal
    that produced them. Building the message from the proposal would be easier
    and would create a second path that can drift from the contract with no test
    noticing.
    """

    def __init__(self, frame_id: str, topic: str = TOPIC) -> None:
        """Start a node and advertise the topic.

        Args:
            frame_id: The coordinate frame to name in every header.
            topic: Where to publish.

        Raises:
            RosUnavailable: If ROS 2 is not importable here.
        """
        rclpy, array_type, detection_type = _imports()
        self._array_type = array_type
        self._detection_type = detection_type
        self._frame_id = frame_id
        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node = rclpy.create_node("clave_decision_publisher")
        self._publisher = self._node.create_publisher(array_type, topic, QUEUE_DEPTH)
        self.published = 0
        """Decisions put on the topic."""
        self.undecodable = 0
        """Datagrams that were not a decision this build can read."""
        self.unsent = 0
        """Decisions that decoded and that ROS refused to carry."""
        self.failures: list[str] = []
        """What each failure reported, in order."""

    def publish(self, payload: bytes) -> bool:
        """Decode one published decision and put it on the topic.

        Args:
            payload: The bytes the runtime published.

        Returns:
            Whether a message was published. A payload that does not decode,
            and a publication ROS refuses, are both counted and reported rather
            than raised, because neither is a reason to stop a conveyor.
        """
        try:
            decision = decode_decision(payload)
        except DecisionError as error:
            self.undecodable += 1
            self.failures.append(str(error))
            return False
        try:
            self._publisher.publish(self._message(fields_for(decision, self._frame_id)))
        except Exception as error:
            # A shut-down context, a dead middleware, or a serialization
            # failure. Counted and named rather than raised, for the same
            # reason an unreadable datagram is: the conveyor keeps moving
            # whatever the consumer is doing.
            self.unsent += 1
            self.failures.append(f"ROS refused a decision: {error}")
            return False
        self.published += 1
        return True

    def close(self) -> None:
        """Destroy the node, and the context if this publisher started it."""
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def _message(self, fields: DetectionFields) -> Any:
        """Build one `Detection3DArray` carrying a single detection."""
        detection = self._detection_type()
        detection.header.frame_id = fields.frame_id
        detection.header.stamp.sec = fields.stamp_sec
        detection.header.stamp.nanosec = fields.stamp_nanosec
        detection.id = fields.object_id

        material, channel = (
            type(detection.results[0])() if detection.results else _hypothesis(),
            _hypothesis(),
        )
        material.hypothesis.class_id = fields.class_id
        material.hypothesis.score = fields.confidence
        material.pose.pose.position.x = fields.position[0]
        material.pose.pose.position.y = fields.position[1]
        material.pose.pose.position.z = fields.position[2]
        (
            material.pose.pose.orientation.x,
            material.pose.pose.orientation.y,
            material.pose.pose.orientation.z,
            material.pose.pose.orientation.w,
        ) = fields.orientation
        # Routing is deterministic given the class and the operator's
        # threshold, so the hypothesis behind a channel is certain even when
        # the class behind it is not.
        channel.hypothesis.class_id = fields.channel_id
        channel.hypothesis.score = 1.0
        detection.results = [material, channel]

        # BoundingBox3D is the one field of a Detection3D that CLAVE has no
        # other use for, since CLAVE estimates a pick point rather than an
        # object extent. The window duration rides in size.x so a subscriber
        # can tell an actionable decision from one that went stale in transit.
        detection.bbox.center = material.pose.pose
        detection.bbox.size.x = fields.window_seconds

        message = self._array_type()
        message.header = detection.header
        message.detections = [detection]
        return message


def _hypothesis() -> Any:
    """Build one empty `ObjectHypothesisWithPose`."""
    from vision_msgs.msg import ObjectHypothesisWithPose

    return ObjectHypothesisWithPose()

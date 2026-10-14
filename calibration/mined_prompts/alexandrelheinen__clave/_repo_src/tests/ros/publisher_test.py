"""Mapping a decision onto a ROS 2 message, and putting it on a topic.

The mapping is computed before ROS is involved, so everything except the last
test runs on a machine with no installation.
"""

import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest

from clave.ros.decisions import decode, read_vector
from clave.ros.publisher import CHANNEL_PREFIX, TOPIC, DetectionFields, fields_for

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "crates" / "clave-decision" / "contract" / "vectors" / "v1"


def ros_available() -> bool:
    """Whether a ROS 2 installation is importable here."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("rclpy", "vision_msgs")
    )


def nominal_fields(frame_id: str = "belt") -> DetectionFields:
    """Map the nominal golden vector."""
    return fields_for(decode(read_vector(VECTORS / "nominal.hex")), frame_id)


def test_every_field_of_a_decision_reaches_the_message() -> None:
    """A subscriber needs all of it, in one message."""
    fields = nominal_fields()
    assert fields.object_id == "4815162342"
    assert fields.class_id == "M-01"
    assert fields.confidence == pytest.approx(0.94, abs=1e-6)
    assert fields.channel_id == f"{CHANNEL_PREFIX}3"
    assert fields.position == pytest.approx((0.412, -0.085, 0.031))
    assert fields.window_seconds == pytest.approx(0.25)


def test_the_frame_the_point_is_expressed_in_is_named() -> None:
    """Meters mean nothing without the frame they are in."""
    assert nominal_fields("belt").frame_id == "belt"
    assert nominal_fields("world").frame_id == "world"


def test_the_stamp_is_the_instant_the_object_becomes_reachable() -> None:
    """The window travels, so a stale decision is detectable."""
    fields = nominal_fields()
    assert fields.stamp_sec == 9
    assert fields.stamp_nanosec == 0


def test_the_yaw_becomes_a_rotation_about_the_belt_normal() -> None:
    """A ROS pose carries an orientation; a planar pick rotates about z alone."""
    fields = nominal_fields()
    x, y, z, w = fields.orientation
    assert (x, y) == (0.0, 0.0)
    assert z == pytest.approx(math.sin(1.047 / 2))
    assert w == pytest.approx(math.cos(1.047 / 2))
    assert 2.0 * math.atan2(z, w) == pytest.approx(1.047)


def test_the_class_and_the_channel_are_separate_assertions() -> None:
    """A rejected object is a real class and a reject channel."""
    fields = fields_for(decode(read_vector(VECTORS / "rejected.hex")), "belt")
    assert fields.class_id == "M-11"
    assert fields.channel_id == f"{CHANNEL_PREFIX}0"
    # Disjoint prefixes, so a consumer reads the pair without relying on order.
    assert not fields.channel_id.startswith("M-")
    assert not fields.class_id.startswith(CHANNEL_PREFIX)


def test_the_runtime_hands_every_published_decision_to_its_sink(
    tmp_path: Path,
    runtime_binary: Path,
    runtime_config: dict[str, object],
    make_proposal: object,
) -> None:
    """The sink sees the bytes that were published.

    This drives the real runtime rather than reaching into the bridge, because
    what matters is that the bytes arriving at the sink are the ones the
    contract put on the wire.
    """
    from clave.ros.decisions import decode
    from clave.runtime.bridge import Bridge, BridgePaths

    build: Any = make_proposal
    seen: list[bytes] = []
    with Bridge(
        runtime_binary, BridgePaths.under(tmp_path), runtime_config, seen.append
    ) as bridge:
        assert bridge.submit(build(0.0, 0.0, 0.95, 0.9)).accepted
        assert bridge.submit(build(0.0, 2.20, 0.95, 0.9)).overridden
        assert bridge.submit(build(0.0, 0.0, 0.95, 0.1)).accepted
        assert bridge.decisions_received == 2

    assert len(seen) == 2, "the sink saw a different count from the bridge"
    sorted_decision, rejected_decision = (decode(payload) for payload in seen)
    assert sorted_decision.material_class == "M-01"
    assert sorted_decision.channel == 1
    assert rejected_decision.channel == 0


@pytest.mark.skipif(not ros_available(), reason="no ROS 2 installation here")
def test_a_subscriber_receives_the_documented_fields() -> None:
    """What the document describes is what arrives."""
    import rclpy
    from vision_msgs.msg import Detection3DArray

    from clave.ros.publisher import DecisionPublisher

    publisher = DecisionPublisher(frame_id="belt")
    node = rclpy.create_node("clave_decision_publisher_test")
    received: list[Detection3DArray] = []
    node.create_subscription(Detection3DArray, TOPIC, received.append, 10)
    try:
        assert publisher.publish(read_vector(VECTORS / "nominal.hex"))
        for _ in range(200):
            rclpy.spin_once(node, timeout_sec=0.05)
            if received:
                break
    finally:
        node.destroy_node()
        publisher.close()

    assert received, "the subscriber received nothing"
    detection = received[0].detections[0]
    assert detection.header.frame_id == "belt"
    assert detection.id == "4815162342"
    assert detection.results[0].hypothesis.class_id == "M-01"
    assert detection.results[0].hypothesis.score == pytest.approx(0.94, abs=1e-6)
    assert detection.results[1].hypothesis.class_id == f"{CHANNEL_PREFIX}3"
    assert detection.results[0].pose.pose.position.x == pytest.approx(0.412)
    assert detection.bbox.size.x == pytest.approx(0.25)
    assert publisher.published == 1


@pytest.mark.skipif(not ros_available(), reason="no ROS 2 installation here")
def test_an_undecodable_payload_is_counted_and_does_not_stop_the_publisher() -> None:
    """One bad datagram is not a reason to stop a conveyor."""
    from clave.ros.publisher import DecisionPublisher

    publisher = DecisionPublisher(frame_id="belt")
    try:
        assert not publisher.publish(b"not a decision")
        assert publisher.publish(read_vector(VECTORS / "nominal.hex"))
    finally:
        publisher.close()
    assert publisher.undecodable == 1
    assert publisher.published == 1
    assert publisher.failures and "CBOR" in publisher.failures[0]


@pytest.mark.skipif(not ros_available(), reason="no ROS 2 installation here")
def test_a_publication_ros_refuses_is_counted_rather_than_raised() -> None:
    """A middleware that has gone is not a reason to stop deciding."""
    from clave.ros.publisher import DecisionPublisher

    publisher = DecisionPublisher(frame_id="belt")
    payload = read_vector(VECTORS / "nominal.hex")
    assert publisher.publish(payload)
    publisher.close()

    assert not publisher.publish(payload)
    assert publisher.unsent == 1
    assert publisher.published == 1
    assert "ROS refused" in publisher.failures[-1]

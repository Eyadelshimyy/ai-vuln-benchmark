import unittest

from slice.horizon import estimate
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _est():
    return HeuristicPoseEstimator().estimate(synthetic_person())


def _feet(skel):
    l = skel.joints["foot_l"]
    r = skel.joints["foot_r"]
    return l, r


class TestHorizon(unittest.TestCase):
    def test_level_ground_zero_roll(self):
        skel = _est()
        l, r = _feet(skel)
        r.y = l.y  # force level
        e = estimate(skel)
        self.assertEqual(e["state"], "estimated")
        self.assertAlmostEqual(e["roll_deg"], 0.0, places=1)
        self.assertEqual(e["depth_hint"], "level")

    def test_rolled_camera_detected(self):
        skel = _est()
        l, r = _feet(skel)
        dx = abs(r.x - l.x)
        r.y = l.y + dx  # 45deg tilt, right foot lower
        e = estimate(skel)
        self.assertAlmostEqual(e["roll_deg"], 45.0, places=1)
        self.assertEqual(e["depth_hint"], "right_closer")

    def test_left_foot_lower_means_left_closer(self):
        skel = _est()
        l, r = _feet(skel)
        l.y = r.y + 20
        e = estimate(skel)
        self.assertLess(e["roll_deg"], 0)
        self.assertEqual(e["depth_hint"], "left_closer")

    def test_no_feet_is_unknown_not_guessed(self):
        skel = _est()
        for n in ("foot_l", "foot_r", "ankle_l", "ankle_r"):
            del skel.joints[n]
        e = estimate(skel)
        self.assertEqual(e["state"], "unknown")
        self.assertIsNone(e["roll_deg"])
        self.assertIsNone(e["horizon_y"])
        self.assertEqual(e["depth_hint"], "unknown")

    def test_ankle_fallback(self):
        skel = _est()
        del skel.joints["foot_l"]
        del skel.joints["foot_r"]
        e = estimate(skel)
        self.assertEqual(e["state"], "estimated")
        self.assertEqual(e["feet_used"], ["ankle_l", "ankle_r"])

    def test_horizon_above_ground(self):
        e = estimate(_est())
        if e["px_per_cm"] is not None:
            self.assertLess(e["horizon_y"], e["ground_y"])
            self.assertEqual(e["camera_height_cm"], 160.0)
        self.assertIn("upright", e["assumption"])

    def test_predicted_feet_are_unknown(self):
        # a "foot below ankle" guess is not a ground contact: with
        # no observed contact pair the roll must not be measured
        skel = _est()
        for n in ("foot_l", "foot_r", "ankle_l", "ankle_r"):
            del skel.joints[n]
        from slice import predict
        predict.complete(skel)
        e = estimate(skel)
        self.assertEqual(e["state"], "unknown")
        self.assertIsNone(e["roll_deg"])

    def test_single_foot_is_unknown(self):
        skel = _est()
        del skel.joints["foot_r"]
        del skel.joints["ankle_r"]
        e = estimate(skel)
        self.assertEqual(e["state"], "unknown")


if __name__ == "__main__":
    unittest.main()

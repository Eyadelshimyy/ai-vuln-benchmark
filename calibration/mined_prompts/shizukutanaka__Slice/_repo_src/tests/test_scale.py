import unittest

from tests import synthetic_person

from slice import predict, scale
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


class TestScale(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_calibration_from_head(self):
        cal = scale.calibrate(self.skel)
        self.assertIsNotNone(cal["px_per_cm"])
        self.assertEqual(cal["state"], "estimated")

    def test_height_in_plausible_range(self):
        m = scale.measure_cm(self.skel)
        # head..foot ≈ body height; expect human-plausible 100–210cm
        h = m["lengths"].get("height")
        self.assertIsNotNone(h)
        self.assertTrue(80 <= h <= 220)

    def test_predicted_joints_no_scale(self):
        # a predicted neck makes head_px a prior measurement — the
        # scale factor would be fabricated, so it must be withheld
        del self.skel.joints["neck"]
        predict.complete(self.skel)
        cal = scale.calibrate(self.skel)
        self.assertIsNone(cal["px_per_cm"])

    def test_predicted_endpoint_no_length(self):
        del self.skel.joints["foot_l"]
        predict.complete(self.skel)
        m = scale.measure_cm(self.skel)
        self.assertNotIn("height", m["lengths"])

    def test_no_head_no_scale(self):
        self.skel.joints.pop("head")
        cal = scale.calibrate(self.skel)
        self.assertIsNone(cal["px_per_cm"])
        self.assertEqual(scale.measure_cm(self.skel)["lengths"], {})

    def test_legacy_chin_neck_convention(self):
        # skeletons saved before the neck moved to the clavicle keep
        # the old chin convention: basis "head height prior" means the
        # head→neck distance is half a head, so it is doubled — the
        # same px_per_cm the old formula produced, not a halved one.
        leg = Skeleton(100, 300)
        leg.set(Joint("head", 50, 30, .9, "observed",
                      basis="top blob centroid"))
        leg.set(Joint("neck", 50, 50, .7, "observed",
                      basis="head height prior"))
        leg.set(Joint("foot_l", 40, 230, .9, "observed"))
        self.assertAlmostEqual(
            scale.calibrate(leg)["px_per_cm"], 40 / 23, places=5)
        self.assertAlmostEqual(
            scale.measure_cm(leg)["lengths"]["height"], 115, delta=1)


if __name__ == "__main__":
    unittest.main()

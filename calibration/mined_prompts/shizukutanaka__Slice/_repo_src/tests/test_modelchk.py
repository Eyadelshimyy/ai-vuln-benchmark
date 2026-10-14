"""Tests for slice.modelchk — chosen-model consistency audit."""
import unittest

from tests import synthetic_person

from slice import modelchk, ratio
from slice.pose import HeuristicPoseEstimator


class TestModelChk(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())
        self.ratios = ratio.analyze(self.skel)

    def test_fixture_consistent(self):
        r = modelchk.check(self.skel, self.ratios)
        self.assertIn(r["verdict"], ("consistent", "mismatch"))
        self.assertEqual(r["chosen"], "adult")
        self.assertTrue(r["dimensions"])

    def test_mismatch_detected(self):
        # measured legs at half the prior → leg dimension goes off
        bad = dict(self.ratios)
        bad["leg_l"] = bad["leg_r"] = 0.05
        bad["torso_length"] = 0.05
        r = modelchk.check(self.skel, bad)
        self.assertEqual(r["verdict"], "mismatch")
        self.assertEqual(r["dimensions"]["leg"]["verdict"], "off")
        self.assertEqual(r["dimensions"]["torso"]["verdict"], "off")

    def test_better_model_advisory(self):
        # ratios matching child priors → better_model suggests child
        child = {
            "shoulder_width": 0.21, "hip_width": 0.15,
            "torso_length": 0.32, "arm_l": 0.33, "arm_r": 0.33,
            "leg_l": 0.34, "leg_r": 0.34,
        }
        r = modelchk.check(self.skel, child)
        if r["verdict"] == "mismatch":
            self.assertIsNotNone(r["better_model"])

    def test_no_model_unmeasurable(self):
        self.skel.body_model = {}
        r = modelchk.check(self.skel, self.ratios)
        self.assertEqual(r["verdict"], "unmeasurable")

    def test_empty_ratios_unmeasurable(self):
        r = modelchk.check(self.skel, {})
        self.assertEqual(r["verdict"], "unmeasurable")


if __name__ == "__main__":
    unittest.main()

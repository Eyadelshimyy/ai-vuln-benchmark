"""Tests for slice.contrad — cross-layer contradiction rules."""
import unittest

from slice import axis, balance, classify, contrad, evaluate, ground
from slice.pose import HeuristicPoseEstimator


def _layers():
    bmp, _ = evaluate.draw_case(160, 300)
    sk = HeuristicPoseEstimator().estimate(bmp)
    return {
        "classify": classify.analyze(sk),
        "axis": axis.principal(sk) or {},
        "ground": ground.estimate(sk),
        "balance": balance.assess(sk),
    }


class TestContrad(unittest.TestCase):
    def test_fixture_consistent(self):
        r = contrad.check(_layers())
        self.assertEqual(r["verdict"], "consistent")
        self.assertEqual(r["contradictions"], [])

    def test_lying_vertical_axis(self):
        r = contrad.check({
            "classify": {"pose": "lie"},
            "axis": {"angle_deg": 90.0},
        })
        self.assertEqual(r["verdict"], "contradicted")
        self.assertIn("lying_vertical_axis",
                      [c["id"] for c in r["contradictions"]])
        self.assertEqual(
            r["contradictions"][0]["evidence"]["axis.angle_deg"], 90.0)

    def test_standing_airborne(self):
        r = contrad.check({
            "classify": {"pose": "stand"},
            "ground": {"contact": "airborne"},
        })
        self.assertIn("standing_airborne",
                      [c["id"] for c in r["contradictions"]])

    def test_stable_without_support(self):
        r = contrad.check({
            "balance": {"projected": "inside"},
            "ground": {"contact": "airborne", "support_joints": 0},
        })
        self.assertIn("stable_without_support",
                      [c["id"] for c in r["contradictions"]])

    def test_insufficient_layers(self):
        r = contrad.check({})
        self.assertEqual(r["verdict"], "insufficient")

    def test_missing_values_skipped(self):
        r = contrad.check({"classify": {"pose": "lie"}})
        self.assertEqual(r["contradictions"], [])
        self.assertEqual(r["verdict"], "insufficient")


if __name__ == "__main__":
    unittest.main()

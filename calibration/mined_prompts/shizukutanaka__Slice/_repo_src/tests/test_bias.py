"""Tests for slice.bias — per-joint systematic error profile."""
import unittest

from slice import bias, evaluate
from slice.skeleton import Joint
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import OBSERVED


def _pairs(cases=3):
    est = HeuristicPoseEstimator()
    out = []
    for w, h in [(160, 300), (200, 360), (120, 240)][:cases]:
        bmp, truth = evaluate.draw_case(w, h)
        out.append((est.estimate(bmp).joints, truth))
    return out


def _joint(x, y, state=OBSERVED):
    return Joint(name="j", x=x, y=y, confidence=0.9,
                 state=state, basis="t")


class TestBias(unittest.TestCase):
    def test_fixture_profile(self):
        r = bias.profile(_pairs())
        self.assertEqual(r["n_cases"], 3)
        self.assertTrue(r["joints"])
        for name, rec in r["joints"].items():
            self.assertIn(rec["verdict"],
                          ("systematic", "unbiased", "insufficient"))

    def test_constant_shift_detected(self):
        pairs = [({"j": _joint(10 + i, 20)}, {"j": (float(i), 0.0)})
                 for i in range(3)]
        r = bias.profile(pairs)
        rec = r["joints"]["j"]
        self.assertEqual(rec["verdict"], "systematic")
        self.assertAlmostEqual(rec["bias_vector"][0], 10.0, places=2)

    def test_scatter_not_biased(self):
        pairs = [({"j": _joint(10, 0)}, {"j": (0.0, 0.0)}),
                 ({"j": _joint(-10, 0)}, {"j": (0.0, 0.0)}),
                 ({"j": _joint(0, 10)}, {"j": (0.0, 0.0)})]
        r = bias.profile(pairs)
        self.assertEqual(r["joints"]["j"]["verdict"], "unbiased")

    def test_insufficient_samples(self):
        r = bias.profile([({"j": _joint(1, 1)}, {"j": (0.0, 0.0)})])
        self.assertEqual(r["joints"]["j"]["verdict"], "insufficient")

    def test_predicted_ignored(self):
        pairs = [({"j": _joint(50, 50, state="predicted")},
                  {"j": (0.0, 0.0)}) for _ in range(3)]
        r = bias.profile(pairs)
        self.assertNotIn("j", r["joints"])

    def test_correction(self):
        pairs = [({"j": _joint(10 + i, 20)}, {"j": (float(i), 0.0)})
                 for i in range(3)]
        r = bias.profile(pairs)
        dx, dy = bias.correction(r, "j")
        self.assertAlmostEqual(dx, 10.0, places=2)
        self.assertEqual(bias.correction(r, "missing"), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

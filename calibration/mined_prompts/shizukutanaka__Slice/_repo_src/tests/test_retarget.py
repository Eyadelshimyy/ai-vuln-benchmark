import math
import unittest

from tests import synthetic_person

from slice import retarget
from slice.pose import HeuristicPoseEstimator


class TestRetarget(unittest.TestCase):
    def setUp(self):
        self.est = HeuristicPoseEstimator()
        self.src = self.est.estimate(synthetic_person(160, 300))

    def test_same_body_reproduces_pose(self):
        out = retarget.retarget(self.src, self.src)
        for n, j in self.src.joints.items():
            r = out.joints.get(n)
            if r:
                self.assertAlmostEqual(r.x, j.x, delta=1.0)
                self.assertAlmostEqual(r.y, j.y, delta=1.0)

    def test_output_is_predicted_not_observed(self):
        out = retarget.retarget(self.src, self.src)
        for j in out.joints.values():
            self.assertEqual(j.state, "predicted")
            self.assertTrue(j.basis.startswith("retargeted"))

    def test_directions_preserved_on_bigger_body(self):
        big = self.est.estimate(synthetic_person(320, 600))
        out = retarget.retarget(self.src, big)
        # direction of hip_l→knee_l in retarget ≈ source direction
        s = (self.src.point("knee_l")[0] - self.src.point("hip_l")[0],
             self.src.point("knee_l")[1] - self.src.point("hip_l")[1])
        r = (out.point("knee_l")[0] - out.point("hip_l")[0],
             out.point("knee_l")[1] - out.point("hip_l")[1])
        ang_s = math.atan2(s[1], s[0])
        ang_r = math.atan2(r[1], r[0])
        self.assertAlmostEqual(ang_s, ang_r, places=2)
        # but the bone is longer (bigger body)
        Lr = math.hypot(*r)
        Ls = math.hypot(*s)
        self.assertGreater(Lr, Ls)


if __name__ == "__main__":
    unittest.main()

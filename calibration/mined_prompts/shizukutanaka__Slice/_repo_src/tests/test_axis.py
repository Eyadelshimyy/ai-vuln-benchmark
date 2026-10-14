import unittest

from tests import synthetic_person

from slice import axis
from slice.pose import HeuristicPoseEstimator


class TestAxis(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_axis_near_vertical(self):
        r = axis.principal(self.skel)
        self.assertGreater(r["angle_deg"], 60)
        self.assertGreater(r["anisotropy"], 0.3)
        self.assertLess(axis.tilt(self.skel), 30)

    def test_lying_axis_horizontal(self):
        # rotate all joints 90° around centroid: (x,y) -> (y,-x)
        cx = sum(j.x for j in self.skel.joints.values()) / len(self.skel.joints)
        cy = sum(j.y for j in self.skel.joints.values()) / len(self.skel.joints)
        for j in self.skel.joints.values():
            dx, dy = j.x - cx, j.y - cy
            j.x, j.y = cx + dy, cy - dx
        r = axis.principal(self.skel)
        self.assertLess(r["angle_deg"], 30)
        self.assertGreater(axis.tilt(self.skel), 60)

    def test_predicted_joints_do_not_pull_axis(self):
        # fabricated geometry must not steer the measured axis:
        # a predicted horizontal fan can't rotate the observed
        # vertical axis
        from slice.skeleton import Joint, Skeleton
        sk = Skeleton(300, 400)
        for i, n in enumerate(("head", "neck", "pelvis")):
            sk.set(Joint(n, 150, 50 + i * 80, 0.9, state="observed"))
        for i in range(10):
            sk.set(Joint("p%d" % i, 30 + i * 25, 200, 0.2,
                         state="predicted"))
        r = axis.principal(sk)
        self.assertGreater(r["angle_deg"], 60)

    def test_too_few_joints_none(self):
        for n in list(self.skel.joints)[2:]:
            del self.skel.joints[n]
        self.assertIsNone(axis.principal(self.skel))


if __name__ == "__main__":
    unittest.main()

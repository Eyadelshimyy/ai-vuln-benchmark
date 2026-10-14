import unittest

from tests import synthetic_person

from slice import contact
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


class TestContact(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_hands_apart_no_contact(self):
        names = [c["contact"] for c in contact.detect(self.skel)]
        self.assertNotIn("hands_together", names)
        self.assertNotIn("hand_at_head", names)

    def test_hands_together(self):
        wl, wr = self.skel.joints["wrist_l"], self.skel.joints["wrist_r"]
        mx = (wl.x + wr.x) / 2
        my = (wl.y + wr.y) / 2
        wl.x, wl.y, wr.x, wr.y = mx - 5, my, mx + 5, my
        names = [c["contact"] for c in contact.detect(self.skel)]
        self.assertIn("hands_together", names)

    def test_arms_crossed(self):
        wr = self.skel.joints["wrist_r"]
        el = self.skel.joints["elbow_l"]
        wr.x, wr.y = el.x, el.y  # right hand on left elbow
        names = [c["contact"] for c in contact.detect(self.skel)]
        self.assertIn("arms_crossed", names)

    def test_predicted_joint_never_contacts(self):
        wr = self.skel.joints["wrist_r"]
        el = self.skel.joints["elbow_l"]
        wr.x, wr.y = el.x, el.y
        wr.state = "predicted"
        self.assertNotIn("arms_crossed",
                         [c["contact"] for c in contact.detect(self.skel)])

    def test_predicted_joint_does_not_inflate_threshold(self):
        # a predicted foot far below stretches the all-joint span:
        # the hands-together threshold must still be measured on
        # observed geometry, not on prior fill
        sk = Skeleton(100, 600)
        sk.set(Joint("head", 50.0, 10.0, 0.9, state="observed"))
        sk.set(Joint("wrist_l", 40.0, 30.0, 0.9, state="observed"))
        sk.set(Joint("wrist_r", 55.0, 30.0, 0.9, state="observed"))
        sk.set(Joint("foot_l", 50.0, 500.0, 0.5, state="predicted"))
        names = [c["contact"] for c in contact.detect(sk)]
        self.assertNotIn("hands_together", names)

    def test_unmeasurable_scale_means_no_contact(self):
        # only two observed wrists at one height: no head, no
        # neck–pelvis pair → body height is unmeasurable. "near"
        # is a fraction of it, so a raw-pixel default would
        # fabricate the threshold that fires hands_together.
        sk = Skeleton(100, 600)
        sk.set(Joint("wrist_l", 40.0, 30.0, 0.9, state="observed"))
        sk.set(Joint("wrist_r", 55.0, 30.0, 0.9, state="observed"))
        self.assertEqual(contact.detect(sk), [])

    def test_summary_shape(self):
        s = contact.summary(self.skel)
        self.assertIsInstance(s["count"], int)


if __name__ == "__main__":
    unittest.main()

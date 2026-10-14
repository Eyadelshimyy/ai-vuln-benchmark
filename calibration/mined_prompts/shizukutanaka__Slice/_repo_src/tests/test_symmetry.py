import unittest

from tests import synthetic_person

from slice import symmetry
from slice.pose import HeuristicPoseEstimator


class TestSymmetry(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_standing_person_highly_symmetric(self):
        r = symmetry.score(self.skel)
        self.assertGreaterEqual(r["compared"], 5)
        self.assertGreater(r["score"], 0.7)

    def test_one_arm_raised_breaks_arm_symmetry(self):
        wrist = self.skel.joints["wrist_r"]
        wrist.y = wrist.y - 200  # raise right arm
        r = symmetry.score(self.skel)
        arm = [v for k, v in r["pairs"].items() if "wrist" in k]
        leg = [v for k, v in r["pairs"].items() if "ankle" in k]
        self.assertLess(min(arm), min(leg))
        self.assertFalse(symmetry.asymmetric_side(self.skel)["symmetric"])

    def test_missing_side_reported(self):
        del self.skel.joints["wrist_r"]
        r = symmetry.score(self.skel)
        self.assertIn("wrist_pair", r["missing"])

    def test_predicted_side_not_fabricated_symmetry(self):
        # predicted joints are mirror fills: a pair containing them
        # would always read 1.0 — it must report missing, not a
        # perfect measured score
        from slice.skeleton import Joint
        for a, b in (("shoulder_r", "shoulder_l"),
                     ("elbow_r", "elbow_l"), ("wrist_r", "wrist_l")):
            j = self.skel.joints[b]
            cx = self.skel.joints["neck"].x
            self.skel.set(Joint(a, 2 * cx - j.x, j.y, 0.3,
                                state="predicted", basis="mirror"))
        r = symmetry.score(self.skel)
        for k in ("shoulder_pair", "elbow_pair", "wrist_pair"):
            self.assertIn(k, r["missing"])
            self.assertNotIn(k, r["pairs"])

    def test_all_missing_score_none(self):
        for n in list(self.skel.joints):
            del self.skel.joints[n]
        self.assertIsNone(symmetry.score(self.skel)["score"])


if __name__ == "__main__":
    unittest.main()

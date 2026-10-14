"""Tests for slice.basis — provenance vocabulary."""
import unittest

from slice import basis, evaluate, pose
from slice.skeleton import Joint, Skeleton, OBSERVED, PREDICTED


class TestBasis(unittest.TestCase):
    def test_prefix_categories(self):
        self.assertEqual(basis.category("mirrored from wrist_r"),
                         "mirror")
        self.assertEqual(basis.category("prior off pelvis"), "prior")
        self.assertEqual(basis.category(
            "interpolated shoulder_l-wrist_l"), "interpolation")
        self.assertEqual(basis.category("retargeted root"),
                         "transform")

    def test_observed_bases(self):
        self.assertEqual(basis.category("top blob centroid"),
                         "observation")
        self.assertEqual(basis.category("arm blob extremity"),
                         "observation")

    def test_prior_wording(self):
        self.assertEqual(basis.category("head height prior"), "prior")

    def test_first_word_collision_not_observation(self):
        # "top guess" / "arm pose guess" share only a first word with
        # observed bases — counting them as observation would let
        # fabricated provenance launder into evidence
        for b in ("top guess", "arm pose guess", "leg estimate",
                  "merged prior hack", "edge of frame"):
            self.assertEqual(basis.category(b), "unknown", b)

    def test_unknown(self):
        self.assertEqual(basis.category("mumbo jumbo"), "unknown")
        self.assertEqual(basis.category(""), "unknown")
        self.assertEqual(basis.category(None), "unknown")

    def test_audit_on_estimated_skeleton(self):
        bmp, _ = evaluate.draw_case()
        skel = pose.HeuristicPoseEstimator().estimate(bmp)
        rep = basis.audit(skel)
        self.assertEqual(rep["unknown_basis"], [])
        self.assertIn("observation", rep["by_category"])

    def test_predicted_chain_categories(self):
        s = Skeleton(image_width=100, image_height=100)
        s.set(Joint("wrist_r", 90, 50, 0.9, OBSERVED,
                    "arm blob extremity"))
        s.set(Joint("wrist_l", 10, 50, 0.2, PREDICTED,
                    "mirrored from wrist_r"))
        rep = basis.audit(s)
        self.assertEqual(rep["by_category"]["mirror"], 1)
        self.assertEqual(rep["by_category"]["observation"], 1)


if __name__ == "__main__":
    unittest.main()

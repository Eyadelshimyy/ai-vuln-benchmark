import unittest

from tests import synthetic_person

from slice import consistency, predict
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, OBSERVED


class TestAudit(unittest.TestCase):
    def test_clean_figure_has_no_issues(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        self.assertEqual(consistency.audit(skel), [])

    def test_out_of_frame_joint_flagged(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        skel.set(Joint("wrist_l", skel.image_width + 5, 100, 0.5, OBSERVED))
        self.assertIn("wrist_l_out_of_frame", consistency.audit(skel))

    def test_impossible_leg_length_flagged(self):
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        knee = skel.get("knee_l")
        skel.set(Joint("knee_l", knee.x, knee.y - 300, knee.confidence, OBSERVED))
        issues = consistency.audit(skel)
        self.assertIn("hip_l_knee_l_too_long", issues)

    def test_predicted_extent_is_no_body(self):
        # prior-placed head/feet must not substitute for a
        # measured body extent — the audit can't scale honestly
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        for n in ("head", "foot_l", "foot_r"):
            del skel.joints[n]
        predict.complete(skel)
        self.assertEqual(consistency.audit(skel),
                         ["no_body_extent"])

    def test_predicted_endpoint_no_segment_check(self):
        # a bone ending in prior fill yields no measurement —
        # it cannot be flagged too_long/too_short
        skel = HeuristicPoseEstimator().estimate(synthetic_person())
        del skel.joints["knee_l"]
        predict.complete(skel)
        issues = consistency.audit(skel)
        self.assertNotIn("hip_l_knee_l_too_long", issues)
        self.assertNotIn("hip_l_knee_l_too_short", issues)


if __name__ == "__main__":
    unittest.main()

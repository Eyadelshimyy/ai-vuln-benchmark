import unittest

from tests import synthetic_person

from slice import stats
from slice.pose import HeuristicPoseEstimator


class TestStats(unittest.TestCase):
    def test_observed_rate_high_on_clean_figures(self):
        est = HeuristicPoseEstimator()
        skels = [est.estimate(synthetic_person()) for _ in range(3)]
        s = stats.summary(skels)
        self.assertEqual(s["cases"], 3)
        self.assertGreater(s["mean_observed_rate"], 0.5)

    def test_blind_spot_joint_detected(self):
        est = HeuristicPoseEstimator()
        skels = []
        for _ in range(4):
            sk = est.estimate(synthetic_person())
            sk.joints.pop("wrist_l")
            skels.append(sk)
        weak = stats.weakest_joints(stats.joint_stats(skels))
        self.assertIn("wrist_l", weak)

    def test_empty_corpus(self):
        self.assertEqual(stats.summary([])["cases"], 0)

    def test_mean_observed_confidence_disclosed(self):
        from slice.predict import complete
        est = HeuristicPoseEstimator()
        sk1 = est.estimate(synthetic_person())
        sk2 = est.estimate(synthetic_person())
        sk2.joints.pop("wrist_l")
        complete(sk2)  # wrist_l comes back as predicted
        self.assertEqual(sk2.joints["wrist_l"].state, "predicted")
        js = stats.joint_stats([sk1, sk2])
        w = js["wrist_l"]
        self.assertIn("mean_observed_confidence", w)
        # blended mean is dragged down by the predicted occurrence;
        # the observed-only mean reflects only real evidence
        self.assertGreater(w["mean_observed_confidence"],
                           w["mean_confidence"])


if __name__ == "__main__":
    unittest.main()

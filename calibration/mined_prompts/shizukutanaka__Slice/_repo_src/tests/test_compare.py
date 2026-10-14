import unittest

from tests import synthetic_person

from slice import bitmap, compare, pipeline


def _doc(*args):
    return pipeline.strip_runtime(
        pipeline.analyze(bitmap.encode_png(synthetic_person(*args))))


class TestPoseDistance(unittest.TestCase):
    def test_identical_pose_is_zero(self):
        d = _doc(160, 300)
        r = compare.pose_distance(d, d)
        self.assertEqual(r["distance"], 0.0)
        self.assertGreater(r["joints_compared"], 15)

    def test_same_pose_different_size_is_near_zero(self):
        a, b = _doc(160, 300), _doc(320, 600)
        r = compare.pose_distance(a, b)
        self.assertIsNotNone(r)
        self.assertLess(r["distance"], 0.2)

    def test_missing_core_returns_none(self):
        r = compare.pose_distance({"skeleton": {"joints": {}}},
                                  _doc(160, 300))
        self.assertIsNone(r)

    def test_predicted_joints_excluded(self):
        # a guessed joint is prior fill — comparing it measures the
        # prior, not the pose
        a, b = _doc(160, 300), _doc(160, 300)
        b["skeleton"]["joints"]["wrist_l"]["state"] = "predicted"
        r = compare.pose_distance(a, b)
        self.assertNotIn("wrist_l", r["per_joint"])

    def test_low_confidence_joints_excluded(self):
        a, b = _doc(160, 300), _doc(160, 300)
        b["skeleton"]["joints"]["wrist_l"]["confidence"] = 0.1
        r = compare.pose_distance(a, b, min_confidence=0.5)
        self.assertNotIn("wrist_l", r["per_joint"])


if __name__ == "__main__":
    unittest.main()

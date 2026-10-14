import unittest

from tests import synthetic_person

from slice import coco
from slice.pose import HeuristicPoseEstimator


class TestCoco(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())
        self.ann = coco.to_coco(self.skel, image_id=1, annotation_id=2)

    def test_keypoint_layout(self):
        kp = self.ann["keypoints"]
        self.assertEqual(len(kp), 17 * 3)
        self.assertEqual(self.ann["image_id"], 1)
        self.assertEqual(self.ann["id"], 2)

    def test_visibility_flags(self):
        kp = self.ann["keypoints"]
        # observed joints → v=2
        self.assertEqual(kp[5 * 3 + 2], 2)   # left_shoulder
        self.assertEqual(kp[11 * 3 + 2], 2)  # left_hip
        # no eyes in Slice → v=0
        self.assertEqual(kp[1 * 3 + 2], 0)

    def test_predicted_maps_to_v1(self):
        j = self.skel.joints["wrist_l"]
        j.state = "predicted"
        ann = coco.to_coco(self.skel)
        self.assertEqual(ann["keypoints"][9 * 3 + 2], 1)  # left_wrist

    def test_unmapped_listed(self):
        self.assertIn("pelvis", self.ann["unmapped_joints"])
        self.assertIn("neck", self.ann["unmapped_joints"])

    def test_categories_block(self):
        cats = coco.categories()
        self.assertEqual(cats[0]["name"], "person")
        self.assertEqual(len(cats[0]["keypoints"]), 17)


if __name__ == "__main__":
    unittest.main()

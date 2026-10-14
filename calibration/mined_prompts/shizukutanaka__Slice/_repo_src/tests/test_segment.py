import unittest

from tests import synthetic_person

from slice import predict, segment
from slice.pose import HeuristicPoseEstimator


class TestSegment(unittest.TestCase):
    def setUp(self):
        self.est = HeuristicPoseEstimator()
        self.bmp = synthetic_person()
        self.skel = self.est.estimate(self.bmp)
        self.mask = self.est._mask(self.bmp)

    def test_head_pixels_labeled_head(self):
        labels = segment.label_map(self.bmp, self.skel, self.mask)
        hx, hy = self.skel.point("head")
        # scan the head blob for a labeled foreground pixel
        got = set()
        for y in range(int(hy) - 6, int(hy) + 6):
            for x in range(int(hx) - 6, int(hx) + 6):
                v = labels[y * self.bmp.width + x]
                if v:
                    got.add(v)
        self.assertIn("head", got)

    def test_summary_covers_foreground(self):
        s = segment.summary(self.bmp, self.skel, self.mask)
        self.assertGreater(s["total_pixels"], 0)
        self.assertGreater(s["parts"]["torso"]["pixels"], 0)
        self.assertGreater(s["parts"]["thigh_l"]["pixels"], 0)
        # every part with pixels reports a nonzero fraction
        for p, d in s["parts"].items():
            if d["pixels"]:
                self.assertGreater(d["fraction"], 0)

    def test_predicted_bone_labels_nothing(self):
        # a predicted forearm bone is prior geometry — pixels near
        # it must not wear a fabricated "forearm" label
        for name in ("elbow_r", "wrist_r"):
            del self.skel.joints[name]
        predict.complete(self.skel)
        self.assertEqual(self.skel.joints["wrist_r"].state,
                         "predicted")
        labels = segment.label_map(self.bmp, self.skel, self.mask)
        self.assertNotIn("forearm_r", labels)

    def test_missing_wrist_gives_no_forearm(self):
        for name in ("wrist_l", "wrist_r"):
            self.skel.joints.pop(name, None)
        labels = segment.label_map(self.bmp, self.skel, self.mask)
        self.assertNotIn("forearm_l", labels)
        self.assertNotIn("forearm_r", labels)


if __name__ == "__main__":
    unittest.main()

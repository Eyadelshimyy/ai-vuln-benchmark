"""Tests for estimate_multi — one skeleton per foreground component."""
import unittest

from slice import evaluate, pose
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _stamp(dst: Bitmap, src: Bitmap, dx: int, dy: int):
    """Paste non-background pixels of src into dst at offset."""
    for y in range(src.height):
        for x in range(src.width):
            p = src.get(x, y)
            if p != BG:
                dst.set(dx + x, dy + y, p)


def _two_people() -> Bitmap:
    person, _ = evaluate.draw_case(120, 260)
    wide = Bitmap.new(320, 300, BG)
    _stamp(wide, person, 10, 20)
    _stamp(wide, person, 190, 20)
    return wide


class TestEstimateMulti(unittest.TestCase):
    def test_two_people_two_skeletons(self):
        est = pose.HeuristicPoseEstimator()
        skels = est.estimate_multi(_two_people())
        self.assertEqual(len(skels), 2)
        for s in skels:
            self.assertGreaterEqual(len(s.joints), 15)
        # they occupy different horizontal regions
        xs = sorted(s.point("pelvis")[0] for s in skels)
        self.assertGreater(xs[1] - xs[0], 80)

    def test_single_person_single_skeleton(self):
        bmp, _ = evaluate.draw_case()
        skels = pose.HeuristicPoseEstimator().estimate_multi(bmp)
        self.assertEqual(len(skels), 1)

    def test_tiny_blobs_filtered(self):
        bmp, _ = evaluate.draw_case()
        # add a small 3x3 speck far from the figure
        for y in range(3):
            for x in range(3):
                bmp.set(bmp.width - 4 - x, 4 + y, (60, 60, 60, 255))
        skels = pose.HeuristicPoseEstimator().estimate_multi(bmp)
        self.assertEqual(len(skels), 1)

    def test_empty_image(self):
        bmp = Bitmap.new(100, 100, BG)
        self.assertEqual(
            pose.HeuristicPoseEstimator().estimate_multi(bmp), [])

    def test_top_k_cap(self):
        person, _ = evaluate.draw_case(100, 220)
        wide = Bitmap.new(600, 280, BG)
        for i in range(4):
            _stamp(wide, person, 10 + i * 145, 25)
        skels = pose.HeuristicPoseEstimator().estimate_multi(
            wide, top_k=3)
        self.assertEqual(len(skels), 3)

    def test_lying_person_retried(self):
        # a sideways figure is a separate component: it must get the
        # same orientation retry the single-person path applies
        person, _ = evaluate.draw_case(160, 300)
        lying = Bitmap.new(person.height, person.width, BG)
        for y in range(person.height):
            for x in range(person.width):
                lying.set(person.height - 1 - y, x, person.get(x, y))
        skels = pose.HeuristicPoseEstimator().estimate_multi(lying)
        self.assertEqual(len(skels), 1)
        sk = skels[0]
        self.assertGreaterEqual(len(sk.joints), 15)
        head = sk.joints.get("head")
        self.assertIsNotNone(head)
        self.assertIn("rotated mask", head.basis or "")


if __name__ == "__main__":
    unittest.main()

"""Tests for slice.split — distance-field watershed for touching people."""
import unittest

from slice import bitmap, evaluate, pose, split
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _stamp(dst, src, dx, dy):
    for y in range(src.height):
        for x in range(src.width):
            if src.get(x, y) != BG:
                dst.set(dx + x, dy + y, src.get(x, y))


def _mask_of(bmp):
    est = pose.HeuristicPoseEstimator()
    mask = est._mask(est._small(bmp) if hasattr(est, "_small") else bmp)
    return mask, len(mask[0]), len(mask)


def _largest(bmp):
    est = pose.HeuristicPoseEstimator()
    small = bmp if max(bmp.width, bmp.height) <= est.max_dim else bmp
    mask = est._mask(small)
    w, h = len(mask[0]), len(mask)
    comp, _ = est._largest_component(mask, w, h)
    return comp, w, h


class TestSplit(unittest.TestCase):
    def test_overlapping_people_split(self):
        person, _ = evaluate.draw_case(100, 240)
        # 30px horizontal overlap -> ONE connected component
        bmp = Bitmap.new(220, 280, BG)
        _stamp(bmp, person, 10, 20)
        _stamp(bmp, person, 80, 20)
        comp, w, h = _largest(bmp)
        masks, seeds = split.split(comp, top_k=4, min_dist=3.0)
        self.assertGreaterEqual(len(masks), 2)
        self.assertEqual(split.unassigned(comp, masks), 0)
        # each claimed region is a plausible person core, not the whole
        total = sum(sum(row) for row in comp)
        self.assertLess(sum(sum(row) for row in masks[0]), total)

    def test_single_person_tiles_component(self):
        # cores are body parts, not people: a lone figure may split
        # into head/chest/hip cores — every pixel stays claimed
        bmp, _ = evaluate.draw_case()
        comp, w, h = _largest(bmp)
        masks, seeds = split.split(comp)
        self.assertGreaterEqual(len(masks), 1)
        self.assertEqual(split.unassigned(comp, masks), 0)

    def test_empty(self):
        self.assertEqual(split.split([]), ([], []))
        self.assertEqual(split.peaks([]), [])

    def test_explicit_seeds(self):
        person, _ = evaluate.draw_case(100, 240)
        bmp = Bitmap.new(220, 280, BG)
        _stamp(bmp, person, 10, 20)
        _stamp(bmp, person, 80, 20)
        comp, w, h = _largest(bmp)
        masks, used = split.split(comp, seeds=[(40, 60), (110, 60)])
        # both seeds snapped onto the silhouette -> two claimed regions
        self.assertEqual(len(masks), 2)
        self.assertEqual(len(used), 2)
        self.assertEqual(split.unassigned(comp, masks), 0)

    def test_estimate_split_fused_people(self):
        # silhouettes merged into ONE component: estimate_multi gives
        # up, estimate_split still finds two head cores
        person, _ = evaluate.draw_case(100, 240)
        bmp = Bitmap.new(200, 280, BG)
        _stamp(bmp, person, 10, 20)
        _stamp(bmp, person, 50, 20)
        est = pose.HeuristicPoseEstimator()
        self.assertEqual(len(est.estimate_multi(bmp)), 1)
        skels = est.estimate_split(bmp)
        self.assertEqual(len(skels), 2)
        for s in skels:
            self.assertIn("split region",
                          list(s.joints.values())[0].basis)

    def test_estimate_split_single_unchanged(self):
        bmp, _ = evaluate.draw_case()
        est = pose.HeuristicPoseEstimator()
        self.assertEqual(len(est.estimate_split(bmp)), 1)


if __name__ == "__main__":
    unittest.main()

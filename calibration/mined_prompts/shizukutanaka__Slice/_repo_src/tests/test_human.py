"""Tests for slice.human — person-likeness of a foreground blob."""
import unittest

from slice import evaluate, human, mask as mask_mod
from slice.bitmap import Bitmap
from slice.pose import HeuristicPoseEstimator


def _person_comp():
    bmp, _ = evaluate.draw_case(160, 300)
    est = HeuristicPoseEstimator()
    m = mask_mod.foreground(bmp, est)
    comp, _ = est._largest_component(m, bmp.width, bmp.height)
    return comp


def _rect_mask(w=160, h=300, x0=40, y0=40, x1=120, y1=260):
    comp = [bytearray(w) for _ in range(h)]
    for y in range(y0, y1):
        for x in range(x0, x1):
            comp[y][x] = 1
    return comp


class TestHuman(unittest.TestCase):
    def test_person_fixture_liked(self):
        r = human.assess(_person_comp())
        self.assertEqual(r["state"], "measured")
        self.assertTrue(r["person_like"], r["signals"])

    def test_dense_box_rejected(self):
        r = human.assess(_rect_mask(x1=140, y1=260))
        self.assertFalse(r["person_like"], r["signals"])
        self.assertLess(r["signals"]["fill"]["score"], 1.0)

    def test_wide_flat_rejected(self):
        # 20px tall bar — wrong aspect entirely
        r = human.assess(_rect_mask(y0=140, y1=160))
        self.assertFalse(r["person_like"], r["signals"])

    def test_sparse_scatter_rejected(self):
        # a dust cloud inside a person-shaped bbox is not lumpy —
        # it is empty; fill must lose credit below the body band
        comp = [bytearray(160) for _ in range(300)]
        for y in range(40, 260, 4):
            comp[y][78 + (y % 9 - 4)] = 1
        r = human.assess(comp)
        self.assertLess(r["signals"]["fill"]["score"], 1.0)
        self.assertFalse(r["person_like"], r["signals"])

    def test_too_tall_pole_loses_aspect_credit(self):
        # a lamp post is taller than any person — the aspect signal
        # must fade out above the human range, not score a perfect 1.0
        r = human.assess(_rect_mask(x0=75, x1=80, y0=20, y1=280))
        self.assertLess(r["signals"]["aspect"]["score"], 1.0)
        self.assertFalse(r["person_like"], r["signals"])

    def test_empty_unmeasurable(self):
        r = human.assess([bytearray(80) for _ in range(80)])
        self.assertEqual(r["state"], "unmeasurable")
        self.assertIsNone(r["person_like"])

    def test_signals_measured(self):
        r = human.assess(_person_comp())
        for k, v in r["signals"].items():
            self.assertIn("value", v)
            self.assertIn("score", v)
            self.assertGreaterEqual(v["score"], 0.0)
            self.assertLessEqual(v["score"], 1.0)


if __name__ == "__main__":
    unittest.main()

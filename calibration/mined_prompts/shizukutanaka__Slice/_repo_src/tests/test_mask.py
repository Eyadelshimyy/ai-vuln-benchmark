import unittest

from slice.bitmap import Bitmap
from slice.mask import coverage, cutout, foreground, to_bitmap
from tests import synthetic_person


class TestMask(unittest.TestCase):
    def test_foreground_marks_person(self):
        bmp = synthetic_person()
        mask = foreground(bmp)
        self.assertEqual(len(mask), bmp.height)
        self.assertEqual(len(mask[0]), bmp.width)
        self.assertGreater(coverage(mask), 0.01)
        self.assertLess(coverage(mask), 0.9)

    def test_to_bitmap_renders_fg(self):
        bmp = synthetic_person()
        mask = foreground(bmp)
        vis = to_bitmap(mask, bmp.width, bmp.height)
        fg_px = sum(
            1 for y in range(vis.height)
            for x in range(vis.width)
            if vis.get(x, y)[3] == 255)
        self.assertGreater(fg_px, 100)

    def test_cutout_zeroes_background(self):
        bmp = synthetic_person()
        out = cutout(bmp)
        self.assertEqual((out.width, out.height),
                         (bmp.width, bmp.height))
        corner = out.get(0, 0)
        self.assertEqual(corner[3], 0)
        # somewhere on the person stays opaque
        cx, cy = bmp.width // 2, bmp.height // 2
        self.assertEqual(out.get(cx, cy)[3], 255)

    def test_blank_frame_zero_coverage(self):
        blank = Bitmap.new(40, 40, (128, 128, 128, 255))
        self.assertEqual(coverage(foreground(blank)), 0.0)


if __name__ == "__main__":
    unittest.main()

class TestCutoutRerun(unittest.TestCase):
    def test_dark_subject_survives_rerun(self):
        # dark figure on light bg: transparent-black bg used to
        # poison _background's colour estimate on a second pass
        bmp = Bitmap.new(60, 60, (200, 200, 200, 255))
        for y in range(10, 50):
            for x in range(20, 40):
                bmp.set(x, y, (10, 10, 10, 255))
        once = coverage(foreground(cutout(bmp)))
        self.assertGreater(once, 0.05)

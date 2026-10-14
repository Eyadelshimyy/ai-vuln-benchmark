import unittest

from slice.bitmap import Bitmap
from slice.diag import diagnose
from tests import synthetic_person


class TestDiag(unittest.TestCase):
    def test_healthy_person_is_ok(self):
        d = diagnose(synthetic_person())
        self.assertTrue(d["ok"])
        self.assertEqual(d["reasons"], ["ok"])
        self.assertGreater(d["fg_coverage"], 0.005)

    def test_blank_frame_no_foreground(self):
        bmp = Bitmap.new(80, 80, (200, 200, 200, 255))
        d = diagnose(bmp)
        self.assertIn("no_foreground", d["reasons"])
        self.assertFalse(d["ok"])

    def test_low_contrast_flagged(self):
        bmp = Bitmap.new(60, 60, (128, 128, 128, 255))
        for y in range(15, 45):
            for x in range(20, 40):
                bmp.set(x, y, (140, 140, 140, 255))  # 12 apart
        d = diagnose(bmp)
        self.assertFalse(d["ok"])
        # diff 12 < threshold 40 → nothing is foreground at all
        self.assertIn("no_foreground", d["reasons"])

    def test_tiny_foreground_too_small(self):
        bmp = Bitmap.new(200, 200, (255, 255, 255, 255))
        for y in range(5, 8):
            for x in range(5, 8):
                bmp.set(x, y, (0, 0, 0, 255))
        d = diagnose(bmp)
        self.assertIn("too_small", d["reasons"])

    def test_full_frame_subject_at_edge(self):
        # subject reaches all 4 edges but border stays mostly bg
        bmp = Bitmap.new(60, 60, (220, 220, 220, 255))
        for y in range(20, 40):
            for x in range(20, 40):
                bmp.set(x, y, (30, 30, 30, 255))
        for y in range(60):          # thin arm to top+bottom edges
            bmp.set(30, y, (30, 30, 30, 255))
        for x in range(60):          # thin arm to left+right edges
            bmp.set(x, 30, (30, 30, 30, 255))
        d = diagnose(bmp)
        self.assertIn("foreground_at_edge", d["reasons"])
        self.assertEqual(d["borders_touched"], 4)


if __name__ == "__main__":
    unittest.main()

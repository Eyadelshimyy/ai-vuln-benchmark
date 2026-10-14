import unittest

from slice.bitmap import Bitmap
from slice.pad import pad, to_aspect, to_square


def _bmp(w=4, h=2, color=(200, 10, 10, 255)):
    return Bitmap.new(w, h, color)


class TestPad(unittest.TestCase):
    def test_centers_and_reports_offset(self):
        out, ox, oy = pad(_bmp(), 10, 6, bg=(0, 0, 0, 255))
        self.assertEqual((out.width, out.height), (10, 6))
        self.assertEqual((ox, oy), (3, 2))
        self.assertEqual(out.get(3, 2), (200, 10, 10, 255))
        self.assertEqual(out.get(0, 0), (0, 0, 0, 255))

    def test_shrink_refused(self):
        with self.assertRaises(ValueError):
            pad(_bmp(), 3, 2)
        with self.assertRaises(ValueError):
            pad(_bmp(), 4, 1)

    def test_same_size_passthrough(self):
        out, ox, oy = pad(_bmp(), 4, 2)
        self.assertEqual((ox, oy), (0, 0))

    def test_to_aspect_widens(self):
        out, ox, oy = to_aspect(_bmp(4, 2), 4.0)
        self.assertEqual((out.width, out.height), (8, 2))
        self.assertEqual((ox, oy), (2, 0))

    def test_to_aspect_tall(self):
        out, ox, oy = to_aspect(_bmp(4, 2), 0.5)
        self.assertEqual((out.width, out.height), (4, 8))
        self.assertEqual((ox, oy), (0, 3))

    def test_to_square(self):
        out, _, _ = to_square(_bmp(6, 2))
        self.assertEqual((out.width, out.height), (6, 6))

    def test_invalid_aspect(self):
        with self.assertRaises(ValueError):
            to_aspect(_bmp(), 0.0)


if __name__ == "__main__":
    unittest.main()

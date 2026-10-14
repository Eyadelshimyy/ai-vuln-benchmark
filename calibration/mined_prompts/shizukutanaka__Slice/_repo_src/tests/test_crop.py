import unittest

from slice.bitmap import Bitmap
from slice.crop import crop


def _bmp(w=10, h=10):
    bmp = Bitmap.new(w, h, (0, 0, 0, 255))
    for y in range(h):
        for x in range(w):
            bmp.set(x, y, (x, y, 0, 255))
    return bmp


class TestCrop(unittest.TestCase):
    def test_basic_crop_copies_pixels(self):
        c = crop(_bmp(), (2, 3, 4, 5))
        self.assertEqual((c.width, c.height), (4, 5))
        self.assertEqual(c.get(0, 0), (2, 3, 0, 255))
        self.assertEqual(c.get(3, 4), (5, 7, 0, 255))

    def test_rect_clamped_to_frame(self):
        c = crop(_bmp(), (8, 8, 10, 10))
        self.assertEqual((c.width, c.height), (2, 2))
        self.assertEqual(c.get(1, 1), (9, 9, 0, 255))

    def test_negative_origin_clamped(self):
        c = crop(_bmp(), (-5, -5, 8, 8))
        self.assertEqual((c.width, c.height), (3, 3))
        self.assertEqual(c.get(0, 0), (0, 0, 0, 255))

    def test_dict_rect(self):
        c = crop(_bmp(), {"x": 1, "y": 1, "width": 3, "height": 2})
        self.assertEqual((c.width, c.height), (3, 2))

    def test_no_overlap_raises(self):
        with self.assertRaises(ValueError):
            crop(_bmp(), (50, 50, 5, 5))
        with self.assertRaises(ValueError):
            crop(_bmp(), (0, 0, 0, 5))

    def test_full_frame_is_copy(self):
        bmp = _bmp()
        c = crop(bmp, (0, 0, 10, 10))
        self.assertEqual(bytes(c.data), bytes(bmp.data))
        self.assertIsNot(c.data, bmp.data)


if __name__ == "__main__":
    unittest.main()

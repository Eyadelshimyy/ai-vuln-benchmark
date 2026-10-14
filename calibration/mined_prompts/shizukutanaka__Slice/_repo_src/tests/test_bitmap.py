import unittest

from tests import synthetic_person

from slice import bitmap


class TestBitmap(unittest.TestCase):
    def test_png_roundtrip(self):
        bmp = synthetic_person(60, 90)
        raw = bitmap.encode_png(bmp)
        self.assertTrue(raw.startswith(bitmap.PNG_MAGIC))
        back = bitmap.decode(raw)
        self.assertEqual((back.width, back.height), (60, 90))
        self.assertEqual(back.get(30, 45), bmp.get(30, 45))

    def test_png_filters_and_palette_absent(self):
        # grayscale write path through encode uses RGBA; decode handles ct6
        bmp = bitmap.Bitmap.new(4, 4, (10, 20, 30, 255))
        back = bitmap.decode(bitmap.encode_png(bmp))
        self.assertEqual(back.get(0, 0), (10, 20, 30, 255))

    def test_bmp_decode(self):
        import struct
        w, h = 2, 2
        stride = ((w * 24 + 31) // 32) * 4
        px = bytearray(stride * h)
        # bottom-up: image top-left lives in file row h-1
        i = (h - 1) * stride
        px[i:i + 3] = bytes((0, 0, 255))
        info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0,
                           len(px), 2835, 2835, 0, 0)
        head = b"BM" + struct.pack("<IHHI", 14 + 40 + len(px), 0, 0, 54)
        img = bitmap.decode(head + info + px)
        self.assertEqual(img.get(0, 0)[:3], (255, 0, 0))

    def test_unsupported(self):
        with self.assertRaises(bitmap.UnsupportedFormat):
            bitmap.decode(b"\x00\x01\x02\x03")

    def test_downscale(self):
        bmp = synthetic_person(200, 400)
        small = bmp.downscale(100)
        self.assertLessEqual(max(small.width, small.height), 100)

    def test_get_out_of_frame_raises(self):
        # negative coords used to wrap to the image tail — silent
        # fabrication; now every out-of-frame read fails loudly
        bmp = bitmap.Bitmap.new(8, 8, (1, 2, 3, 255))
        for x, y in ((-1, 0), (0, -1), (8, 0), (0, 8)):
            with self.assertRaises(IndexError):
                bmp.get(x, y)


if __name__ == "__main__":
    unittest.main()

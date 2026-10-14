import unittest

from slice.autocrop import person_bbox, suggest
from slice.bitmap import Bitmap
from tests import synthetic_person


class TestAutocrop(unittest.TestCase):
    def test_bbox_covers_the_figure(self):
        bmp = synthetic_person()
        box = person_bbox(bmp)
        self.assertIsNotNone(box)
        x, y, w, h = box
        # figure is centered horizontally, spans ~head to bottom
        self.assertGreater(w, bmp.width * 0.3)
        self.assertGreater(h, bmp.height * 0.8)
        self.assertLess(x + w / 2 - bmp.width / 2, bmp.width * 0.1)
        self.assertGreater(x + w / 2 - bmp.width / 2,
                           -bmp.width * 0.1)

    def test_empty_frame_is_unknown_not_guessed(self):
        bmp = Bitmap.new(50, 50, (235, 235, 235, 255))
        self.assertIsNone(person_bbox(bmp))
        s = suggest(bmp)
        self.assertEqual(s["state"], "unknown")
        self.assertIsNone(s["crop"])

    def test_margin_grows_the_crop(self):
        bmp = synthetic_person()
        tight = suggest(bmp, margin=0.0)
        loose = suggest(bmp, margin=0.2)
        self.assertGreaterEqual(loose["crop"][2], tight["crop"][2])
        self.assertGreaterEqual(loose["crop"][3], tight["crop"][3])
        self.assertEqual(tight["crop"][2:], tight["bbox"][2:])

    def test_aspect_widens_or_tallens(self):
        # small subject in a roomy frame so the aspect can be met
        # without clamping
        bmp = Bitmap.new(200, 200, (235, 235, 235, 255))
        for y in range(60, 140):
            for x in range(80, 100):
                bmp.set(x, y, (60, 60, 60, 255))
        sq = suggest(bmp, aspect=1.0)
        cw, ch = sq["crop"][2], sq["crop"][3]
        self.assertAlmostEqual(cw / ch, 1.0, delta=0.05)
        wide = suggest(bmp, aspect=2.0)
        self.assertAlmostEqual(wide["crop"][2] / wide["crop"][3],
                               2.0, delta=0.05)

    def test_aspect_beyond_frame_clamps_honestly(self):
        # a subject filling the frame can't yield the requested
        # aspect — the crop clamps to the frame rather than lying
        bmp = synthetic_person()
        s = suggest(bmp, aspect=1.0)
        self.assertLessEqual(s["crop"][2], bmp.width)
        self.assertLessEqual(s["crop"][3], bmp.height)

    def test_crop_stays_inside_frame(self):
        bmp = Bitmap.new(60, 40, (235, 235, 235, 255))
        # subject hugging the top-left corner
        for y in range(5):
            for x in range(5):
                bmp.set(x, y, (60, 60, 60, 255))
        s = suggest(bmp, margin=0.5, aspect=1.0)
        x, y, w, h = s["crop"]
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, 60)
        self.assertLessEqual(y + h, 40)

    def test_coverage_reported(self):
        s = suggest(synthetic_person())
        self.assertGreater(s["coverage"], 0.05)
        self.assertLess(s["coverage"], 1.0)
        self.assertIn("margin", s["basis"])


if __name__ == "__main__":
    unittest.main()

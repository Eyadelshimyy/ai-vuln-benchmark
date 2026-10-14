import random
import unittest

from tests import synthetic_person

from slice import bitmap, style


def noise_image(w=128, h=128, seed=1):
    bmp = bitmap.Bitmap.new(w, h, (0, 0, 0, 255))
    rng = random.Random(seed)
    for y in range(h):
        for x in range(w):
            bmp.set(x, y, (rng.randrange(256), rng.randrange(256),
                           rng.randrange(256), 255))
    return bmp


def gradient_image(w=128, h=128):
    bmp = bitmap.Bitmap.new(w, h, (0, 0, 0, 255))
    for y in range(h):
        for x in range(w):
            bmp.set(x, y, (int(255 * x / w), int(200 * y / h), 120, 255))
    return bmp


def skin_image(w=128, h=128, seed=1):
    """Warm skin-tone texture — photographic portrait crop."""
    bmp = bitmap.Bitmap.new(w, h, (0, 0, 0, 255))
    rng = random.Random(seed)
    for y in range(h):
        for x in range(w):
            j = rng.randrange(-70, 70)
            bmp.set(x, y, (min(255, 205 + j), min(255, 165 + j),
                           min(255, 130 + j), 255))
    return bmp


class TestStyle(unittest.TestCase):
    def test_transparent_pixels_are_not_black_evidence(self):
        # a photo-like textured figure on an alpha=0 canvas used to be
        # classified from a fabricated 88% "flat black" — transparent
        # pixels are absent content, not a dominant color
        rng = random.Random(1)
        bmp = bitmap.Bitmap.new(200, 240, (0, 0, 0, 0))
        for y in range(240):
            for x in range(200):
                if 60 <= y < 200 and 80 <= x < 120:
                    bmp.set(x, y, (rng.randrange(256),
                                   rng.randrange(256),
                                   rng.randrange(256), 255))
        rep = style.analyze(bmp)
        self.assertLess(rep["signals"]["top_color_coverage"], 0.25)
        self.assertAlmostEqual(rep["signals"]["opaque_ratio"],
                               40 * 140 / (200 * 240), places=2)
        self.assertEqual(rep["style"], "real")

    def test_fully_transparent_is_unknown(self):
        bmp = bitmap.Bitmap.new(64, 64, (0, 0, 0, 0))
        rep = style.analyze(bmp)
        self.assertEqual(rep["style"], "unknown")
        self.assertEqual(rep["signals"]["opaque_ratio"], 0.0)

    def test_flat_figure_is_anime(self):
        r = style.analyze(synthetic_person(240, 420))
        self.assertEqual(r["style"], "anime")
        self.assertGreater(r["confidence"], 0.5)
        self.assertGreater(r["signals"]["top_color_coverage"], 0.35)

    def test_skin_texture_is_real(self):
        r = style.analyze(skin_image())
        self.assertEqual(r["style"], "real")
        self.assertGreater(r["signals"]["skin_ratio"], 0.05)

    def test_noise_is_real(self):
        r = style.analyze(noise_image())
        self.assertEqual(r["style"], "real")

    def test_gradient_is_illustration(self):
        r = style.analyze(gradient_image())
        self.assertEqual(r["style"], "illustration")

    def test_pipeline_includes_style(self):
        from slice import pipeline
        doc = pipeline.strip_runtime(pipeline.analyze(
            bitmap.encode_png(synthetic_person(240, 420))))
        self.assertEqual(doc["style"]["style"], "anime")


if __name__ == "__main__":
    unittest.main()

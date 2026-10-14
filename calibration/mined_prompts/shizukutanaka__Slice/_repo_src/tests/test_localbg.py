"""Gradient-background robustness for the banded background model.

AUDIT P0-3: a single global bg estimate misclassifies far-gradient
background pixels as foreground. `_background_bands` tracks vertical
drift so the figure stays isolated.
"""
import unittest

from tests import synthetic_person
from slice import pose

EST = pose.HeuristicPoseEstimator()
BG = (235, 235, 235, 255)


def _gradient(depth):
    bmp = synthetic_person(bg=BG)
    h = bmp.height
    for y in range(h):
        v = int(235 - depth * y / max(1, h - 1))
        for x in range(bmp.width):
            if bmp.get(x, y) == BG:
                bmp.set(x, y, (v, v, min(255, v + 4), 255))
    return bmp


class TestBandedBackground(unittest.TestCase):
    def test_mild_gradient_full_body(self):
        skel = EST.estimate(_gradient(30))
        self.assertEqual(len(skel.joints), 19)

    def test_strong_gradient_recovers(self):
        # global bg estimate dropped to 15 joints here; bands restore all
        skel = EST.estimate(_gradient(60))
        obs = sum(1 for j in skel.joints.values()
                  if j.state == "observed")
        self.assertEqual(obs, 19)

    def test_very_strong_gradient_still_works(self):
        skel = EST.estimate(_gradient(140))
        self.assertEqual(len(skel.joints), 19)

    def test_flat_background_unchanged(self):
        skel = EST.estimate(synthetic_person())
        self.assertEqual(len(skel.joints), 19)

    def test_transparent_border_samples_ignored(self):
        # transparent black on top/bottom/left must not become the band
        # bg colour (keeps the alpha-aware _background fix in effect)
        from slice.bitmap import Bitmap
        bmp = Bitmap.new(60, 60, (200, 200, 200, 255))
        for i in range(60):
            bmp.set(i, 0, (0, 0, 0, 0))
            bmp.set(i, 59, (0, 0, 0, 0))
            bmp.set(0, i, (0, 0, 0, 0))
        for band in EST._background_bands(bmp):
            self.assertGreater(band[0], 100)


if __name__ == "__main__":
    unittest.main()

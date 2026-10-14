"""Tests for slice.shadow — cast-shadow pixel rejection."""
import unittest

from slice import evaluate, pose, shadow
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _with_shadow() -> Bitmap:
    bmp, _ = evaluate.draw_case(160, 280)
    # darkened-background ellipse under the feet — hue preserved
    for y in range(250, 275):
        for x in range(30, 150):
            if ((x - 90) / 60) ** 2 + ((y - 262) / 12) ** 2 <= 1:
                r, g, b, a = bmp.get(x, y)
                bmp.set(x, y, (int(r * 0.62), int(g * 0.62),
                               int(b * 0.62), a))
    return bmp


class TestShadow(unittest.TestCase):
    def test_is_shadow_uniform_dim(self):
        self.assertTrue(shadow.is_shadow(160, 160, 160,
                                         (235, 235, 235)))
        self.assertFalse(shadow.is_shadow(160, 160, 235,
                                          (235, 235, 235)))  # hue shift
        self.assertFalse(shadow.is_shadow(230, 230, 230,
                                          (235, 235, 235)))  # too bright
        self.assertFalse(shadow.is_shadow(5, 5, 5,
                                          (235, 235, 235)))  # too dark

    def test_shadow_inflates_mask(self):
        bmp = _with_shadow()
        est = pose.HeuristicPoseEstimator()
        mask = est._mask(bmp)
        w, h = len(mask[0]), len(mask)
        sm = shadow.shadow_pixels(bmp, est._background(bmp), mask)
        removed = sum(sum(r) for r in sm)
        self.assertGreater(removed, 100)  # the ellipse got captured

    def test_reject_shadow_strips_ellipse(self):
        bmp = _with_shadow()
        est = pose.HeuristicPoseEstimator(reject_shadow=True)
        mask = est._mask(bmp)
        # ellipse region back to background
        self.assertEqual(mask[260][140], 0)
        self.assertGreater(est.last_shadow_removed, 100)

    def test_person_survives_shadow_reject(self):
        bmp = _with_shadow()
        sk = pose.HeuristicPoseEstimator(
            reject_shadow=True).estimate(bmp)
        obs = sum(1 for j in sk.joints.values()
                  if j.state == "observed")
        self.assertGreaterEqual(obs, 15)

    def test_off_by_default(self):
        bmp = _with_shadow()
        sk = pose.HeuristicPoseEstimator().estimate(bmp)
        self.assertIsNone(
            pose.HeuristicPoseEstimator().last_shadow_removed)


if __name__ == "__main__":
    unittest.main()

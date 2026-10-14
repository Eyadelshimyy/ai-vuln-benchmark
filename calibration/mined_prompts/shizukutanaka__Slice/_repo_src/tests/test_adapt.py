"""Tests for slice.adapt — Otsu adaptive fg/bg threshold."""
import unittest

from slice import adapt, evaluate, pose
from slice.bitmap import Bitmap


def _flat(w, h, rgb):
    return Bitmap.new(w, h, (*rgb, 255))


def _low_contrast() -> Bitmap:
    """Figure only ~20 units from the background — under the fixed 40."""
    bmp = _flat(80, 120, (200, 200, 200))
    for y in range(20, 100):
        for x in range(30, 50):
            bmp.set(x, y, (182, 182, 182, 255))
    return bmp


class TestOtsu(unittest.TestCase):
    def test_bimodal_split(self):
        bmp, _ = evaluate.draw_case()
        est = pose.HeuristicPoseEstimator()
        bg = est._background(bmp)
        t = adapt.otsu_threshold(adapt.distances(bmp, bg))
        self.assertGreater(t, adapt.MIN_MEANINGFUL)

    def test_flat_no_split(self):
        bmp = _flat(60, 60, (220, 220, 220))
        dist = adapt.distances(bmp, (220, 220, 220))
        self.assertEqual(adapt.otsu_threshold(dist), 0.0)
        thr, method = adapt.threshold(dist)
        self.assertEqual(method, "fixed")
        self.assertEqual(thr, 40.0)

    def test_low_contrast_fixed_misses_otsu_finds(self):
        bmp = _low_contrast()
        est = pose.HeuristicPoseEstimator()
        bg = est._background(bmp)
        # fixed rule: nothing exceeds 40 in any channel
        self.assertFalse(any(any(r) for r in
                             est._mask_fixed(bmp, bg, 40)))
        # adaptive: otsu splits near ~18 and finds the figure
        mask = est.__class__(adaptive=True)._mask(bmp)
        fg = sum(sum(r) for r in mask)
        self.assertGreater(fg, 1000)

    def test_method_recorded(self):
        bmp, _ = evaluate.draw_case()
        est = pose.HeuristicPoseEstimator(adaptive=True)
        est.estimate(bmp)
        self.assertEqual(est.last_threshold[1], "otsu")
        est2 = pose.HeuristicPoseEstimator()
        est2.estimate(bmp)
        self.assertEqual(est2.last_threshold, (40, "fixed"))

    def test_adaptive_estimates_person(self):
        bmp, _ = evaluate.draw_case()
        sk = pose.HeuristicPoseEstimator(adaptive=True).estimate(bmp)
        obs = sum(1 for j in sk.joints.values() if j.state == "observed")
        self.assertGreaterEqual(obs, 15)

    def test_empty_input(self):
        self.assertEqual(adapt.otsu_threshold([]), 0.0)


if __name__ == "__main__":
    unittest.main()

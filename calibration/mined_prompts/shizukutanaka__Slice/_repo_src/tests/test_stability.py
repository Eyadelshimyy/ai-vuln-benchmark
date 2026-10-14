"""Tests for slice.stability — perturbation robustness measurement."""
import unittest

from slice import evaluate, stability
from slice.bitmap import Bitmap


class TestStability(unittest.TestCase):
    def test_fixture_probes(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = stability.probe(bmp)
        self.assertEqual(r["state"], "measured")
        self.assertTrue(r["joints"])
        for name, v in r["joints"].items():
            self.assertIn(v["verdict"],
                          ("stable", "sensitive", "unstable",
                           "single_run"))
            if v["verdict"] != "single_run":
                self.assertIsNotNone(v["max_displacement_px"])

    def test_stable_fraction_range(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = stability.probe(bmp)
        self.assertIsNotNone(r["stable_fraction"])
        self.assertGreaterEqual(r["stable_fraction"], 0.0)
        self.assertLessEqual(r["stable_fraction"], 1.0)
        # fixture figure is strong → most joints stable
        self.assertGreaterEqual(r["stable_fraction"], 0.5)

    def test_unstable_names_sorted(self):
        bmp, _ = evaluate.draw_case(160, 300)
        names = stability.unstable(stability.probe(bmp))
        self.assertEqual(names, sorted(names))

    def test_empty_image_unmeasurable(self):
        r = stability.probe(Bitmap.new(160, 300, (240, 240, 240, 255)))
        self.assertIn(r["state"], ("unmeasurable", "measured"))
        if r["state"] == "unmeasurable":
            self.assertIsNone(r["stable_fraction"])

    def test_delta_recorded(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = stability.probe(bmp, delta=5)
        self.assertEqual(r["probe_delta"], 5)

    def test_adaptive_offset_moves_mask(self):
        # shifting bg_threshold does not move an Otsu split — the
        # perturbation must reach the effective threshold, or the
        # probe reports stability it never measured
        from slice import mask as mask_mod
        from slice.pose import HeuristicPoseEstimator
        bmp, _ = evaluate.draw_case(160, 300)
        fg = lambda e: sum(sum(r) for r in mask_mod.foreground(bmp, e))
        est = HeuristicPoseEstimator(adaptive=True)
        lo = HeuristicPoseEstimator(adaptive=True, threshold_offset=-20)
        hi = HeuristicPoseEstimator(adaptive=True, threshold_offset=20)
        self.assertLess(fg(hi), fg(lo))
        self.assertGreaterEqual(fg(lo), fg(est))
        self.assertLessEqual(fg(hi), fg(est))


if __name__ == "__main__":
    unittest.main()

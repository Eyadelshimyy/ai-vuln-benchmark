"""Tests for slice.bench — benchmark + regression gate."""
import unittest

from slice import bench


class TestBench(unittest.TestCase):
    def test_run_report_shape(self):
        r = bench.run()
        for key in ("cases", "repeats", "ms_per_estimate",
                    "detection_rate", "observed_rate",
                    "mean_error_px", "mean_oks"):
            self.assertIn(key, r, key)
        self.assertEqual(r["detection_rate"], 1.0)

    def test_gate_passes_at_current_accuracy(self):
        ok, fails = bench.gate(bench.run())
        self.assertTrue(ok, fails)

    def test_gate_catches_regression(self):
        bad = {"detection_rate": 0.5, "observed_rate": 0.5,
               "mean_error_px": 40.0, "mean_oks": 0.1,
               "cases": 1, "repeats": 1, "ms_per_estimate": 1.0}
        ok, fails = bench.gate(bad)
        self.assertFalse(ok)
        self.assertEqual(len(fails), 4)

    def test_main_exit_code(self):
        self.assertEqual(bench.main(["--json"]), 0)


if __name__ == "__main__":
    unittest.main()

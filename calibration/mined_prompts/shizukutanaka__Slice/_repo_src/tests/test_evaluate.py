import unittest

from slice import evaluate


class TestEvaluate(unittest.TestCase):
    def test_draw_case_truth_shapes_figure(self):
        bmp, truth = evaluate.draw_case(160, 300)
        self.assertGreater(len(truth), 15)
        for name, (x, y) in truth.items():
            px = bmp.get(int(x), int(y))
            self.assertIsNotNone(px, name)
            self.assertNotEqual(px[:3], (235, 235, 235),
                                f"{name} not on the figure")

    def test_evaluate_reports_metrics(self):
        cases = [evaluate.draw_case(160, 300),
                 evaluate.draw_case(200, 360)]
        r = evaluate.evaluate(cases)
        self.assertEqual(r["cases"], 2)
        self.assertGreater(r["detection_rate"], 0.5)
        self.assertGreater(r["observed_rate"], 0.5)
        self.assertLess(r["mean_error_px"], 40)
        self.assertIn("wrist_l", r["per_joint"])


if __name__ == "__main__":
    unittest.main()

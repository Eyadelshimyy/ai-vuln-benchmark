import unittest

from slice.anatomy import BODY_MODELS, select_model


class TestSelectModel(unittest.TestCase):
    def test_exact_match_picks_model(self):
        for name, m in BODY_MODELS.items():
            best, conf = select_model(m["head_ratio"])
            self.assertEqual(best, name)

    def test_boundary_measurement_drops_confidence(self):
        adult = BODY_MODELS["adult"]["head_ratio"]
        child = BODY_MODELS["child"]["head_ratio"]
        _, conf_match = select_model(adult)
        _, conf_boundary = select_model((adult + child) / 2)
        self.assertLess(conf_boundary, conf_match)
        self.assertLessEqual(conf_boundary, 0.55)

    def test_confidence_bounded(self):
        for probe in (0.0, 0.1, 0.2, 0.5, 1.0):
            _, conf = select_model(probe)
            self.assertTrue(0.2 <= conf <= 0.9)


if __name__ == "__main__":
    unittest.main()

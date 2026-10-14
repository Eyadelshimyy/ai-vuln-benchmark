import unittest

from tests import synthetic_person

from slice import mass, predict
from slice.pose import HeuristicPoseEstimator


class TestMass(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_adult_mass_plausible(self):
        m = mass.estimate(self.skel, area_px=9000)
        self.assertEqual(m["state"], "estimated")
        self.assertGreater(m["kg"], 5)
        self.assertLess(m["kg"], 200)
        self.assertGreater(m["depth_cm"], 0)

    def test_zero_area_none(self):
        self.assertIsNone(mass.estimate(self.skel, 0))

    def test_no_body_none(self):
        for n in list(self.skel.joints):
            del self.skel.joints[n]
        self.assertIsNone(mass.estimate(self.skel, 9000))

    def test_predicted_head_is_unmeasurable(self):
        # a prior-placed head can't anchor the px->cm span —
        # the estimate must abstain, not scale off a guess
        del self.skel.joints["head"]
        predict.complete(self.skel)
        self.assertIsNone(mass.estimate(self.skel, 9000))

    def test_predicted_feet_not_lowest(self):
        # "foot below ankle" priors must not stretch the span
        del self.skel.joints["foot_l"]
        del self.skel.joints["foot_r"]
        predict.complete(self.skel)
        m = mass.estimate(self.skel, 9000)
        ankle_lo = max(self.skel.joints[n].y
                       for n in ("ankle_l", "ankle_r"))
        head = self.skel.joints["head"]
        cm_per_px_expected = m["height_cm"] / (ankle_lo - head.y)
        self.assertAlmostEqual(
            m["volume_l"],
            9000 * cm_per_px_expected ** 2
            * (9000 / (ankle_lo - head.y)) * cm_per_px_expected
            * 0.28 / 1000,
            places=1)

    def test_bmi_shape(self):
        b = mass.bmi(self.skel, 9000)
        self.assertIsNotNone(b)
        self.assertGreater(b, 5)


if __name__ == "__main__":
    unittest.main()

import unittest

from tests import synthetic_person

from slice import mutate
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import OBSERVED


class TestMutate(unittest.TestCase):
    def test_noise_preserves_estimate(self):
        est = HeuristicPoseEstimator()
        bmp = mutate.add_noise(synthetic_person(), 15)
        skel = est.estimate(bmp)
        self.assertIn("pelvis", skel.joints)

    def test_occluded_arm_stays_honest(self):
        """Hide the left arm: its wrist must not be observed."""
        bmp = mutate.occlude(synthetic_person(160, 300),
                             0, 60, 55, 230)
        skel = HeuristicPoseEstimator().estimate(bmp)
        wrist = skel.get("wrist_l")
        if wrist is not None:
            self.assertNotEqual(wrist.state, OBSERVED)
        self.assertEqual(skel.get("wrist_r").state, OBSERVED)

    def test_crop_cuts_figure_without_crash(self):
        bmp = mutate.crop(synthetic_person(160, 300), 10, 0, 150, 200)
        skel = HeuristicPoseEstimator().estimate(bmp)
        # whatever survives must still be well-formed
        for j in skel.joints.values():
            self.assertTrue(0 <= j.x <= bmp.width)
            self.assertTrue(0 <= j.y <= bmp.height)

    def test_crop_out_of_frame_no_wrap(self):
        # Bitmap.get wraps negative indices to the image tail; the
        # region beyond the frame must stay the fill, not smuggled
        # pixels from the opposite edge
        from slice.bitmap import Bitmap
        src = Bitmap.new(10, 10, (10, 20, 30, 255))
        for yy in range(10):
            for xx in range(10):
                src.set(xx, yy, (xx * 25, yy * 25, 7, 255))
        out = mutate.crop(src, -5, -5, 10, 10)
        self.assertEqual(out.get(0, 0), (0, 0, 0, 255))
        self.assertEqual(out.get(5, 5), src.get(0, 0))


if __name__ == "__main__":
    unittest.main()

"""Near-realistic fixture tests — the estimator must survive gradient
walls, sensor noise, occluders and soft shadows (AUDIT P0-4: no real
photos in the repo, so these procedural stand-ins guard the known
fragile paths)."""
import unittest

from tests import realistic
from slice import pose

EST = pose.HeuristicPoseEstimator()


def _observed(skel):
    return {n for n, j in skel.joints.items() if j.state == "observed"}


class TestRealisticBackgrounds(unittest.TestCase):
    def test_gradient_wall_keeps_full_body(self):
        obs = _observed(EST.estimate(realistic.gradient_person()))
        for joint in ("head", "pelvis", "wrist_l", "ankle_r"):
            self.assertIn(joint, obs)

    def test_strong_gradient_still_finds_figure(self):
        # depth=60 gradient: known degradation, must not vanish
        skel = EST.estimate(realistic.gradient_person(depth=60))
        self.assertGreaterEqual(len(_observed(skel)), 10)

    def test_sensor_noise_keeps_full_body(self):
        obs = _observed(EST.estimate(realistic.noisy_person(amp=25)))
        self.assertEqual(len(obs), 19)

    def test_occluder_keeps_remaining_figure(self):
        obs = _observed(EST.estimate(realistic.occluded_person()))
        self.assertIn("head", obs)
        self.assertGreaterEqual(len(obs), 15)

    def test_cast_shadow_does_not_swallow_feet(self):
        obs = _observed(EST.estimate(realistic.shadow_person()))
        self.assertIn("head", obs)
        self.assertGreaterEqual(len(obs), 15)


if __name__ == "__main__":
    unittest.main()

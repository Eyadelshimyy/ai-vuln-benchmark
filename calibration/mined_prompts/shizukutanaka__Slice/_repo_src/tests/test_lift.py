import unittest

from tests import synthetic_person

from slice import lift
from slice.pose import HeuristicPoseEstimator


class TestLift(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_front_view_is_flat(self):
        self.skel.orientation = {"facing": "front"}
        d = lift.lift(self.skel)
        for v in d.values():
            self.assertEqual(v["z"], 0.0)
            self.assertEqual(v["basis"], "no_depth_cue")
        self.assertEqual(lift.depth_spread(d), 0.0)

    def test_side_view_pushes_far_limb_back(self):
        self.skel.orientation = {"facing": "side", "side": "left"}
        d = lift.lift(self.skel)
        # far side gets +z, near side stays 0 (convention aside)
        zs = {d["wrist_l"]["z"], d["wrist_r"]["z"]}
        self.assertEqual(min(zs), 0.0)
        self.assertGreater(max(zs), 0.0)
        self.assertGreater(lift.depth_spread(d), 0.0)

    def test_left_facing_lifts_far_flank(self):
        """facing "left" is the real profile detection: the person's
        left arm sits behind the sagittal axis, so _l joints get +z.
        Before the fix, genuine left/right profiles lifted flat."""
        self.skel.orientation = {"facing": "left"}
        d = lift.lift(self.skel)
        self.assertGreater(d["wrist_l"]["z"], 0.0)
        self.assertEqual(d["wrist_r"]["z"], 0.0)
        self.assertEqual(d["wrist_l"]["basis"], "facing_side")

    def test_right_facing_lifts_far_flank(self):
        self.skel.orientation = {"facing": "right"}
        d = lift.lift(self.skel)
        self.assertGreater(d["wrist_r"]["z"], 0.0)
        self.assertEqual(d["wrist_l"]["z"], 0.0)

    def test_predicted_joints_keep_their_state(self):
        # x,y of a predicted joint is prior fill — the output must
        # say so, not present it indistinguishably from observed
        self.skel.joints["wrist_r"].state = "predicted"
        d = lift.lift(self.skel)
        self.assertEqual(d["wrist_r"]["state"], "predicted")
        self.assertEqual(d["wrist_l"]["state"], "observed")

    def test_unknown_facing_flat(self):
        self.skel.orientation = {}
        d = lift.lift(self.skel)
        self.assertEqual(lift.depth_spread(d), 0.0)

    def test_predicted_width_no_depth_scale(self):
        # depth scale from a predicted shoulder/hip width would write
        # prior geometry into z as if measured — stays flat instead
        self.skel.orientation = {"facing": "side", "side": "left"}
        for n in ("shoulder_l", "shoulder_r", "hip_l", "hip_r"):
            self.skel.joints[n].state = "predicted"
        d = lift.lift(self.skel)
        self.assertEqual(lift.depth_spread(d), 0.0)


if __name__ == "__main__":
    unittest.main()

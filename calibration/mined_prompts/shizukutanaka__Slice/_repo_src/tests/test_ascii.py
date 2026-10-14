import unittest

from slice.ascii import render
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _est():
    return HeuristicPoseEstimator().estimate(synthetic_person())


class TestAscii(unittest.TestCase):
    def test_renders_non_empty_art(self):
        art = render(_est())
        self.assertIn("@", art)
        self.assertIn("#", art)
        self.assertGreater(len(art.strip().splitlines()), 3)

    def test_width_scales_grid(self):
        art = render(_est(), width=60)
        longest = max(len(l) for l in art.splitlines())
        self.assertLessEqual(longest, 60)

    def test_predicted_joint_draws_o(self):
        skel = _est()
        for j in skel.joints.values():
            j.state = "predicted"
        art = render(skel)
        self.assertIn("o", art)
        self.assertNotIn("@", art)

    def test_predicted_bone_draws_colon(self):
        skel = _est()
        skel.joints["wrist_l"].state = "predicted"
        art = render(skel)
        self.assertIn(":", art)

    def test_empty_skeleton_is_blank(self):
        from slice.skeleton import Skeleton
        art = render(Skeleton(10, 10))
        self.assertEqual(art.strip(), "")

    def test_joint_glyph_at_joint_position(self):
        skel = _est()
        j = skel.joints["head"]
        art = render(skel, width=40)
        lines = art.splitlines()
        cy = int(j.y / skel.image_height * len(lines))
        cx = int(j.x / skel.image_width * 40)
        self.assertEqual(lines[cy][cx], "@")


if __name__ == "__main__":
    unittest.main()

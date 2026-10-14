"""Tests for slice.framefit — truncation hints at frame edges."""
import unittest

from slice import framefit
from slice.skeleton import Joint, OBSERVED, Skeleton


def _skel(joints, w=64, h=96):
    return Skeleton(w, h, {k: Joint(k, x, y, 0.9, OBSERVED) for k, (x, y) in joints.items()})


class TestAssess(unittest.TestCase):
    def test_centered_figure_is_clear(self):
        r = framefit.assess(_skel({
            "head": (32, 10), "pelvis": (32, 50), "foot_l": (30, 80),
            "foot_r": (34, 80), "wrist_l": (20, 55), "wrist_r": (44, 55)}))
        self.assertFalse(r["partial"])
        self.assertEqual(r["possibly_truncated"], [])
        self.assertEqual(r["state"], "estimated")

    def test_feet_on_bottom_edge_flagged(self):
        r = framefit.assess(_skel({
            "head": (32, 10), "pelvis": (32, 50), "foot_l": (30, 95),
            "foot_r": (34, 94)}))
        self.assertTrue(r["partial"])
        self.assertIn("bottom", r["possibly_truncated"])
        self.assertEqual(r["edges"]["bottom"]["verdict"], "possibly_truncated")

    def test_head_at_top_edge_flagged(self):
        r = framefit.assess(_skel({
            "head": (32, 1), "pelvis": (32, 50), "foot_l": (30, 80)}))
        self.assertIn("top", r["possibly_truncated"])

    def test_torso_joint_at_edge_not_flagged(self):
        # pelvis flush to the right edge is odd but not an extremity
        r = framefit.assess(_skel({
            "head": (32, 10), "pelvis": (63, 50), "foot_l": (60, 80),
            "foot_r": (62, 80)}))
        self.assertNotIn("right", r["possibly_truncated"])
        self.assertTrue(r["edges"]["right"]["at_edge"])
        self.assertEqual(r["edges"]["right"]["verdict"], "clear")

    def test_empty_skeleton(self):
        r = framefit.assess(Skeleton(64, 96, {}))
        self.assertFalse(r["partial"])
        self.assertEqual(r["n_edges_checked"], 0)


if __name__ == "__main__":
    unittest.main()

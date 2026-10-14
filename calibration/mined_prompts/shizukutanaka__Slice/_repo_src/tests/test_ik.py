import math
import unittest

from slice import ik


class TestIk(unittest.TestCase):
    def test_segment_lengths_satisfied(self):
        mid, end = ik.solve_ik((0, 0), (30, 20), 25, 25)
        d1 = math.hypot(mid[0], mid[1])
        d2 = math.hypot(end[0] - mid[0], end[1] - mid[1])
        self.assertAlmostEqual(d1, 25.0, places=6)
        self.assertAlmostEqual(d2, 25.0, places=6)
        self.assertAlmostEqual(end[0], 30.0, places=6)
        self.assertAlmostEqual(end[1], 20.0, places=6)

    def test_unreachable_clamps_to_extension(self):
        mid, end = ik.solve_ik((0, 0), (100, 0), 30, 20)
        self.assertAlmostEqual(math.hypot(*end), 50.0, delta=0.01)
        # arm fully extended: bend ≈ 180°
        self.assertGreater(ik.bend_angle((0, 0), mid, end), 175)

    def test_bend_side_selectable(self):
        m1, _ = ik.solve_ik((0, 0), (30, 0), 25, 25, bend=1)
        m2, _ = ik.solve_ik((0, 0), (30, 0), 25, 25, bend=-1)
        self.assertTrue(m1[1] * m2[1] < 0 or
                        abs(m1[1] - m2[1]) > 1e-6)

    def test_bent_solution_angle(self):
        mid, end = ik.solve_ik((0, 0), (25, 0), 25, 25)
        ang = ik.bend_angle((0, 0), mid, end)
        self.assertLess(ang, 180)

    def test_degenerate_segments_declined(self):
        # documented contract: no chain to solve -> None, not a
        # fabricated mid/end 1e-6 px from the root
        self.assertIsNone(ik.solve_ik((0, 0), (10, 0), 0, 0))
        self.assertIsNone(ik.solve_ik((0, 0), (10, 0), 0, 20))
        self.assertIsNone(ik.solve_ik((0, 0), (10, 0), 20, 0))
        self.assertIsNone(ik.solve_ik((0, 0), (10, 0), -5, 20))


if __name__ == "__main__":
    unittest.main()

import unittest

from tests import synthetic_person

from slice import topology
from slice.bitmap import Bitmap


class TestTopology(unittest.TestCase):
    def test_person_one_component_no_holes(self):
        d = topology.analyze(synthetic_person())
        self.assertIsNotNone(d)
        self.assertEqual(d["components"], 1)
        self.assertEqual(d["holes"], 0)
        self.assertEqual(d["euler"], 1)

    def test_ring_has_hole(self):
        b = Bitmap.new(80, 80, (0, 0, 0, 255))
        for y in range(20, 60):
            for x in range(20, 60):
                if not (30 <= x < 50 and 30 <= y < 50):
                    b.set(x, y, (200, 200, 200, 255))
        d = topology.analyze(b)
        self.assertEqual(d["components"], 1)
        self.assertEqual(d["holes"], 1)
        self.assertEqual(d["hole_areas"], [400])
        self.assertEqual(d["euler"], 0)

    def test_two_blobs(self):
        b = Bitmap.new(80, 80, (0, 0, 0, 255))
        for y in range(10, 30):
            for x in range(10, 30):
                b.set(x, y, (200, 200, 200, 255))
        for y in range(40, 60):
            for x in range(50, 70):
                b.set(x, y, (200, 200, 200, 255))
        d = topology.analyze(b)
        self.assertEqual(d["components"], 2)
        self.assertEqual(d["euler"], 2)
        self.assertEqual(d["fg_px"], 800)

    def test_hole_in_smaller_blob(self):
        # two disconnected rings: both holes must count
        b = Bitmap.new(160, 80, (0, 0, 0, 255))
        for ox in (10, 90):
            for y in range(10, 50):
                for x in range(ox, ox + 40):
                    if not (ox + 10 <= x < ox + 30
                            and 20 <= y < 40):
                        b.set(x, y, (200, 200, 200, 255))
        d = topology.analyze(b)
        self.assertEqual(d["components"], 2)
        self.assertEqual(d["holes"], 2)
        self.assertEqual(d["euler"], 0)

    def test_many_components_no_overflow(self):
        # checkerboard speckle: >255 isolated fg pixels must not crash
        b = Bitmap.new(40, 40, (0, 0, 0, 255))
        for y in range(40):
            for x in range(40):
                if (x + y) % 2 == 0:
                    b.set(x, y, (200, 200, 200, 255))
        d = topology.analyze(b)
        self.assertIsNotNone(d)

    def test_empty_none(self):
        b = Bitmap.new(50, 50, (255, 255, 255, 255))
        self.assertIsNone(topology.analyze(b))


if __name__ == "__main__":
    unittest.main()

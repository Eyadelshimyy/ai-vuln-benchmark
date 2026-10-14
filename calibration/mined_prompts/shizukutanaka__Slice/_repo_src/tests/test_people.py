import unittest

from slice.bitmap import Bitmap
from slice.people import candidates
from tests import synthetic_person


def _two_blobs():
    bmp = Bitmap.new(120, 80, (240, 240, 240, 255))
    for y in range(10, 70):            # big figure, left
        for x in range(10, 30):
            bmp.set(x, y, (30, 30, 30, 255))
    for y in range(20, 60):            # smaller figure, right
        for x in range(80, 95):
            bmp.set(x, y, (30, 30, 30, 255))
    return bmp


class TestPeople(unittest.TestCase):
    def test_single_person_one_candidate(self):
        r = candidates(synthetic_person())
        self.assertEqual(r["n_candidates"], 1)
        self.assertGreater(r["dominant_share"], 0.005)

    def test_two_blobs_two_candidates(self):
        r = candidates(_two_blobs())
        self.assertEqual(r["n_components"], 2)
        self.assertEqual(r["n_candidates"], 2)
        big, small = r["candidates"]
        self.assertGreater(big["size_px"], small["size_px"])
        self.assertEqual(big["rank"], 0)

    def test_blank_zero_candidates(self):
        r = candidates(Bitmap.new(50, 50, (128, 128, 128, 255)))
        self.assertEqual(r["n_candidates"], 0)
        self.assertEqual(r["candidates"], [])

    def test_edge_touch_detected(self):
        bmp = Bitmap.new(60, 60, (240, 240, 240, 255))
        for y in range(60):
            for x in range(10):       # hugging left edge
                bmp.set(x, y, (30, 30, 30, 255))
        r = candidates(bmp)
        self.assertTrue(r["candidates"][0]["touches_edge"])

    def test_min_frac_filters_noise(self):
        bmp = _two_blobs()
        for y in range(75, 78):       # 9 px speck < 0.5% of 9600
            for x in range(100, 103):
                bmp.set(x, y, (30, 30, 30, 255))
        r = candidates(bmp)
        self.assertEqual(r["n_components"], 3)
        self.assertEqual(r["n_candidates"], 2)


if __name__ == "__main__":
    unittest.main()

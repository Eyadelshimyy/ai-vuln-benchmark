"""Tests for slice.morph — binary morphology cleanup."""
import unittest

from slice import evaluate, morph, pose
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _mask(w, h, cells):
    m = [bytearray(w) for _ in range(h)]
    for x, y in cells:
        m[y][x] = 1
    return m


class TestMorph(unittest.TestCase):
    def test_erode_removes_surface(self):
        m = _mask(9, 9, [(x, y) for x in range(2, 7)
                         for y in range(2, 7)])
        e = morph.erode(m)
        self.assertEqual(e[4][4], 1)
        self.assertEqual(e[2][4], 0)  # edge pixel lost a neighbour

    def test_dilate_grows(self):
        m = _mask(9, 9, [(4, 4)])
        d = morph.dilate(m)
        self.assertEqual(sum(sum(r) for r in d), 5)

    def test_open_removes_speck(self):
        cells = [(x, y) for x in range(2, 8) for y in range(2, 8)]
        cells += [(8, 8)]  # single-pixel speck 1px away
        m = _mask(12, 12, cells)
        o = morph.open(m)
        self.assertEqual(o[8][8], 0)
        self.assertGreater(sum(sum(r) for r in o), 9)

    def test_close_fills_pinhole(self):
        cells = [(x, y) for x in range(2, 8) for y in range(2, 8)]
        cells.remove((5, 5))
        m = _mask(12, 12, cells)
        c = morph.close(m)
        self.assertEqual(c[5][5], 1)

    def test_clean_estimator_speckle(self):
        bmp, _ = evaluate.draw_case()
        # scatter noise far from the person
        for x, y in [(5, 5), (bmp.width - 6, 10), (10, bmp.height - 8)]:
            bmp.set(x, y, (30, 30, 30, 255))
        est = pose.HeuristicPoseEstimator(clean=True)
        mask = est._mask(bmp)
        for x, y in [(5, 5), (bmp.width - 6, 10),
                     (10, bmp.height - 8)]:
            self.assertEqual(mask[y][x], 0)
        sk = est.estimate(bmp)
        obs = sum(1 for j in sk.joints.values()
                  if j.state == "observed")
        self.assertGreaterEqual(obs, 15)

    def test_clean_off_default(self):
        bmp, _ = evaluate.draw_case()
        bmp.set(5, 5, (30, 30, 30, 255))
        mask = pose.HeuristicPoseEstimator()._mask(bmp)
        self.assertEqual(mask[5][5], 1)

    def test_empty(self):
        self.assertEqual(morph.erode([]), [])


if __name__ == "__main__":
    unittest.main()

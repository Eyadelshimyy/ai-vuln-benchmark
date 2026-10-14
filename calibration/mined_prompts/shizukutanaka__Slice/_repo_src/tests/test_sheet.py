import unittest

from slice.bitmap import Bitmap
from slice.pose import HeuristicPoseEstimator
from slice.sheet import sheet, sheet_png
from tests import synthetic_person

_EST = HeuristicPoseEstimator()


def _skel():
    return _EST.estimate(synthetic_person())


class TestSheet(unittest.TestCase):
    def test_single_tile_size(self):
        s = sheet([_skel()], cols=1, cell=64)
        self.assertEqual((s.width, s.height), (64, 64))

    def test_grid_layout(self):
        s = sheet([_skel() for _ in range(5)], cols=3, cell=40)
        self.assertEqual((s.width, s.height), (120, 80))

    def test_tiles_actually_draw(self):
        bg = (10, 10, 10, 255)
        s = sheet([_skel()], cols=1, cell=64, bg=bg)
        drawn = sum(
            1 for y in range(s.height) for x in range(s.width)
            if s.get(x, y) != bg)
        self.assertGreater(drawn, 50)

    def test_aspect_preserved_not_stretched(self):
        skel = _skel()
        s = sheet([skel], cols=1, cell=80)
        # skeleton frame is taller than wide; tile should be
        # letterboxed horizontally (background column at edges)
        self.assertEqual(s.get(0, 40), (18, 18, 22, 255))
        self.assertEqual(s.get(79, 40), (18, 18, 22, 255))

    def test_empty_is_blank_cell(self):
        s = sheet([], cell=32)
        self.assertEqual((s.width, s.height), (32, 32))

    def test_png_encodes(self):
        raw = sheet_png([_skel()], cell=48)
        self.assertTrue(raw.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()

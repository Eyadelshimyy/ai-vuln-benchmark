import unittest

from slice.bitmap import Bitmap
from slice.mask import foreground
from slice.pose import HeuristicPoseEstimator


class TestBackgroundAlpha(unittest.TestCase):
    def test_transparent_border_rgb_ignored(self):
        est = HeuristicPoseEstimator()
        # light-looking bg pixels but fully transparent;
        # one dark opaque figure in the middle
        bmp = Bitmap.new(60, 60, (200, 200, 200, 0))
        for y in range(15, 45):
            for x in range(20, 40):
                bmp.set(x, y, (200, 200, 200, 255))
        # all border pixels transparent → falls back to RGB
        self.assertGreater(est._background(bmp)[0], 100)

    def test_opaque_border_preferred(self):
        est = HeuristicPoseEstimator()
        bmp = Bitmap.new(60, 60, (200, 200, 200, 255))
        # sprinkle transparent black pixels on the border
        for x in range(0, 60, 2):
            bmp.set(x, 0, (0, 0, 0, 0))
        bg = est._background(bmp)
        self.assertGreater(bg[0], 100)  # light wins, not black

    def test_dark_figure_after_cutout_detected(self):
        # reproduces the #101 review scenario end-to-end
        bmp = Bitmap.new(60, 60, (200, 200, 200, 255))
        for y in range(15, 45):
            for x in range(20, 40):
                bmp.set(x, y, (10, 10, 10, 255))
        # simulate cutout: bg alpha zeroed, RGB kept
        for y in range(60):
            for x in range(60):
                if not (15 <= y < 45 and 20 <= x < 40):
                    d = bmp.data
                    d[(y * 60 + x) * 4 + 3] = 0
        mask = foreground(bmp)
        self.assertGreater(sum(sum(r) for r in mask), 100)


if __name__ == "__main__":
    unittest.main()

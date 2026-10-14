import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from slice import bitmap
from slice.__main__ import main
from tests import synthetic_person


def _png(bmp):
    p = tempfile.mktemp(suffix=".png")
    with open(p, "wb") as f:
        f.write(bitmap.encode_png(bmp))
    return p


def _short_legged():
    """A figure with half-length legs -> different bone ratios."""
    bmp = bitmap.Bitmap.new(160, 300, (235, 235, 235, 255))
    cx, skin = 80, (60, 60, 60, 255)
    sh_y, hip_y = 300 * 0.22, 300 * 0.55
    torso_w = 160 * 0.34
    for y in range(14):
        for x in range(int(cx - 14), int(cx + 14)):
            bmp.set(x, y + 8, skin)
    for y in range(int(sh_y), int(hip_y)):
        for x in range(int(cx - torso_w / 2), int(cx + torso_w / 2)):
            bmp.set(x, y, skin)
    # short stubby legs end at 70% height
    for y in range(int(hip_y), int(300 * 0.7)):
        for x in range(int(cx - torso_w / 2), int(cx - 4)):
            bmp.set(x, y, skin)
        for x in range(int(cx + 4), int(cx + torso_w / 2)):
            bmp.set(x, y, skin)
    return bmp


class TestReidCli(unittest.TestCase):
    def setUp(self):
        self.p = _png(synthetic_person())
        self.q = _png(_short_legged())
        self.addCleanup(lambda: os.path.exists(self.p)
                        and os.unlink(self.p))
        self.addCleanup(lambda: os.path.exists(self.q)
                        and os.unlink(self.q))

    def test_same_image_is_same_person(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["reid", self.p, self.p]), 0)
        res = json.loads(buf.getvalue())
        self.assertTrue(res["same_person"])
        self.assertEqual(res["distance"], 0.0)
        self.assertEqual(res["state"], "estimated")

    def test_different_proportions_larger_distance(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["reid", self.p, self.q])
        res = json.loads(buf.getvalue())
        self.assertIsNotNone(res["distance"])
        self.assertGreater(res["distance"], 0.0)
        self.assertIn("features_a", res)


if __name__ == "__main__":
    unittest.main()

import json
import os
import shutil
import tempfile
import unittest

from slice import bitmap
from slice.__main__ import main
from tests import synthetic_person


class TestExportCli(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.d, True))
        self.img = os.path.join(self.d, "p.png")
        with open(self.img, "wb") as f:
            f.write(bitmap.encode_png(synthetic_person()))

    def test_coco_to_file(self):
        out = os.path.join(self.d, "a.json")
        self.assertEqual(
            main(["export", self.img, "--format", "coco",
                  "-o", out]), 0)
        doc = json.load(open(out))
        self.assertGreater(doc["num_keypoints"], 10)

    def test_bvh_contains_hierarchy(self):
        out = os.path.join(self.d, "a.bvh")
        self.assertEqual(
            main(["export", self.img, "--format", "bvh",
                  "-o", out]), 0)
        text = open(out).read()
        self.assertIn("HIERARCHY", text)
        self.assertIn("ROOT", text)

    def test_heatmap_is_png(self):
        out = os.path.join(self.d, "a.png")
        self.assertEqual(
            main(["export", self.img, "--format", "heatmap",
                  "-o", out]), 0)
        bmp = bitmap.decode(open(out, "rb").read())
        self.assertGreater(bmp.width, 0)

    def test_all_formats_accepted(self):
        for fmt in ("bvh", "gltf", "coco", "svg", "ascii",
                    "paf", "heatmap"):
            out = os.path.join(
                self.d, "a." + ("png" if fmt == "heatmap" else "txt"))
            self.assertEqual(
                main(["export", self.img, "--format", fmt,
                      "-o", out]), 0, fmt)
            self.assertGreater(os.path.getsize(out), 10, fmt)

    def test_blank_image_fails(self):
        blank = os.path.join(self.d, "blank.png")
        with open(blank, "wb") as f:
            f.write(bitmap.encode_png(
                bitmap.Bitmap.new(60, 60, (255, 255, 255, 255))))
        self.assertEqual(
            main(["export", blank, "--format", "bvh"]), 1)


if __name__ == "__main__":
    unittest.main()

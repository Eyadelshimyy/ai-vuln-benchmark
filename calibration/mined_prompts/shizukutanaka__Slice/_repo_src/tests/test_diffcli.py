import json
import os
import shutil
import tempfile
import unittest

from slice import bitmap
from slice.__main__ import main
from tests import synthetic_person


class TestDiffCli(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.d, True))
        self.a = os.path.join(self.d, "a.png")
        self.b = os.path.join(self.d, "b.png")
        with open(self.a, "wb") as f:
            f.write(bitmap.encode_png(synthetic_person()))
        with open(self.b, "wb") as f:
            f.write(bitmap.encode_png(synthetic_person(300, 300)))

    def test_identical_images(self):
        out = os.path.join(self.d, "d.json")
        self.assertEqual(
            main(["diff", self.a, self.a, "-o", out]), 0)
        res = json.load(open(out))
        self.assertTrue(res["identical"])

    def test_changed_pose_reports_moves(self):
        out = os.path.join(self.d, "d.json")
        self.assertEqual(
            main(["diff", self.a, self.b, "-o", out]), 0)
        res = json.load(open(out))
        self.assertFalse(res["identical"])
        self.assertTrue(res["joints"]["moved"])
        self.assertIn("pose_distance", res)


if __name__ == "__main__":
    unittest.main()

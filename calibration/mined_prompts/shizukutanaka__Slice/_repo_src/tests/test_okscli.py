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


class TestOksCli(unittest.TestCase):
    def setUp(self):
        self.p = _png(synthetic_person())
        self.q = _png(synthetic_person(320, 600))
        self.addCleanup(lambda: os.path.exists(self.p)
                        and os.unlink(self.p))
        self.addCleanup(lambda: os.path.exists(self.q)
                        and os.unlink(self.q))

    def test_same_pose_perfect_oks(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(
                main(["oks", self.p, self.p]), 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(res["oks"], 1.0)

    def test_scale_invariant_high_oks(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(
                main(["oks", self.p, self.q]), 0)
        res = json.loads(buf.getvalue())
        self.assertIsNotNone(res["oks"])
        self.assertGreater(res["oks"], 0.5)
        self.assertIn("per_joint", res)


if __name__ == "__main__":
    unittest.main()

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


class TestAutocropCli(unittest.TestCase):
    def setUp(self):
        self.p = _png(synthetic_person())
        self.addCleanup(lambda: os.path.exists(self.p)
                        and os.unlink(self.p))

    def test_suggestion_and_crop_write(self):
        out = tempfile.mktemp(suffix=".png")
        self.addCleanup(lambda: os.path.exists(out)
                        and os.unlink(out))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(
                main(["autocrop", self.p, "-o", out]), 0)
        res = json.loads(buf.getvalue())
        self.assertIsNotNone(res["crop"])
        self.assertEqual(res["state"], "observed")
        bmp = bitmap.decode(open(out, "rb").read())
        x, y, w, h = res["crop"]
        self.assertEqual((bmp.width, bmp.height), (w, h))

    def test_blank_reports_no_crop(self):
        blank = _png(bitmap.Bitmap.new(
            80, 80, (128, 128, 128, 255)))
        self.addCleanup(lambda: os.path.exists(blank)
                        and os.unlink(blank))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["autocrop", blank]), 1)
        res = json.loads(buf.getvalue())
        self.assertIsNone(res["crop"])


if __name__ == "__main__":
    unittest.main()

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


class TestImgqualCli(unittest.TestCase):
    def setUp(self):
        self.p = _png(synthetic_person())
        self.addCleanup(lambda: os.path.exists(self.p)
                        and os.unlink(self.p))

    def test_flags_and_verdict(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["imgqual", self.p]), 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(res["verdict"], "adequate")
        self.assertEqual({f["code"] for f in res["flags"]},
                         {"size", "dynamic", "blur", "contrast"})

    def test_inadequate_fails(self):
        blank = _png(bitmap.Bitmap.new(
            40, 40, (128, 128, 128, 255)))
        self.addCleanup(lambda: os.path.exists(blank)
                        and os.unlink(blank))
        self.assertEqual(main(["imgqual", blank]), 1)


if __name__ == "__main__":
    unittest.main()

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from slice import bitmap
from slice.__main__ import _PROBE_LAYERS, main
from tests import synthetic_person


def _png(bmp):
    p = tempfile.mktemp(suffix=".png")
    with open(p, "wb") as f:
        f.write(bitmap.encode_png(bmp))
    return p


class TestProbeCli(unittest.TestCase):
    def setUp(self):
        self.p = _png(synthetic_person())
        self.addCleanup(lambda: os.path.exists(self.p)
                        and os.unlink(self.p))

    def test_every_layer_returns(self):
        for layer in _PROBE_LAYERS:
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(
                    main(["probe", layer, self.p]), 0, layer)
            res = json.loads(buf.getvalue())
            self.assertEqual(res["layer"], layer)
            self.assertIsNotNone(res["result"], layer)

    def test_ground_grounded(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["probe", "ground", self.p])
        res = json.loads(buf.getvalue())
        self.assertEqual(
            res["result"]["estimate"]["contact"], "grounded")

    def test_blank_fails(self):
        blank = _png(bitmap.Bitmap.new(
            80, 80, (128, 128, 128, 255)))
        self.addCleanup(lambda: os.path.exists(blank)
                        and os.unlink(blank))
        self.assertEqual(
            main(["probe", "axis", blank]), 1)


if __name__ == "__main__":
    unittest.main()

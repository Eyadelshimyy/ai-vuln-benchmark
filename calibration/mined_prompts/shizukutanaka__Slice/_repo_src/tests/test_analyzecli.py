"""`slice analyze` — stderr summary surfaces doc.warnings."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from slice import __main__, bitmap, evaluate
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _blob_png() -> bytes:
    """A small unperson-like blob: no joints observed → all warnings."""
    bmp = Bitmap.new(160, 300, BG)
    for y in range(140, 160):
        for x in range(70, 90):
            bmp.set(x, y, (60, 60, 60, 255))
    return bitmap.encode_png(bmp)


def _analyze_stderr(raw: bytes) -> str:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fig.png")
        with open(path, "wb") as f:
            f.write(raw)
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = __main__.main(["analyze", path])
        assert rc == 0
        return err.getvalue()


class AnalyzeCliTest(unittest.TestCase):
    def test_warnings_printed(self):
        err = _analyze_stderr(_blob_png())
        self.assertIn("warnings:", err)
        self.assertIn("few_observed_joints", err)

    def test_clean_doc_no_warnings_line(self):
        fig, _ = evaluate.draw_case(160, 300)
        err = _analyze_stderr(bitmap.encode_png(fig))
        self.assertNotIn("warnings:", err)


if __name__ == "__main__":
    unittest.main()

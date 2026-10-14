import os
import shutil
import tempfile
import unittest

from slice import bitmap
from slice.__main__ import main
from tests import synthetic_person


class TestDescribeCli(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.d, True))
        self.img = os.path.join(self.d, "p.png")
        with open(self.img, "wb") as f:
            f.write(bitmap.encode_png(synthetic_person()))

    def test_describes_synth_person(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["describe", self.img])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("observed", out)
        self.assertIn("Standing", out)

    def test_blank_image(self):
        import io
        from contextlib import redirect_stdout
        blank = os.path.join(self.d, "b.png")
        with open(blank, "wb") as f:
            f.write(bitmap.encode_png(
                bitmap.Bitmap.new(60, 60, (255, 255, 255, 255))))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["describe", blank]), 1)


if __name__ == "__main__":
    unittest.main()

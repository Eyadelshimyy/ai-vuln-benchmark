import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
import io

from slice import bitmap
from slice.__main__ import main
from tests import synthetic_person


def _png(bmp):
    p = tempfile.mktemp(suffix=".png")
    with open(p, "wb") as f:
        f.write(bitmap.encode_png(bmp))
    return p


class TestMutateCli(unittest.TestCase):
    def setUp(self):
        self.p = _png(synthetic_person())
        self.addCleanup(lambda: os.path.exists(self.p)
                        and os.unlink(self.p))

    def test_noise_deterministic(self):
        o1, o2 = (tempfile.mktemp(suffix=".png")
                  for _ in range(2))
        self.addCleanup(lambda: os.path.exists(o1)
                        and os.unlink(o1))
        self.addCleanup(lambda: os.path.exists(o2)
                        and os.unlink(o2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(
                main(["mutate", self.p, "--noise", "20",
                      "-o", o1]), 0)
            self.assertEqual(
                main(["mutate", self.p, "--noise", "20",
                      "-o", o2]), 0)
        self.assertEqual(open(o1, "rb").read(),
                         open(o2, "rb").read())
        res = json.loads(buf.getvalue().splitlines()[0])
        self.assertEqual(res["op"], "noise")

    def test_occlude_and_crop(self):
        out = tempfile.mktemp(suffix=".png")
        self.addCleanup(lambda: os.path.exists(out)
                        and os.unlink(out))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["mutate", self.p,
                      "--occlude", "0,0,50,50",
                      "-o", out]), 0)
        bmp = bitmap.decode(open(out, "rb").read())
        self.assertEqual(bmp.get(10, 10)[:3], (235, 235, 235))


if __name__ == "__main__":
    unittest.main()

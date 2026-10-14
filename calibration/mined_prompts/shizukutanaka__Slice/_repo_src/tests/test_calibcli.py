"""`slice calib` — confidence calibration over ground-truth fixtures."""

import io
import json
import unittest
from contextlib import redirect_stdout

from slice.__main__ import main


class CalibCliTest(unittest.TestCase):
    def test_report(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["calib"])
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["state"], "estimated")
        self.assertIn("hit_radius_px", rep)
        self.assertEqual(len(rep["bins"]), 10)
        filled = [b for b in rep["bins"] if b["n"]]
        self.assertTrue(filled)
        for b in filled:
            self.assertGreaterEqual(b["accuracy"], 0.0)
            self.assertLessEqual(b["accuracy"], 1.0)
            self.assertGreater(b["mean_conf"], 0.0)
        self.assertEqual(
            rep["overconfident_bins"],
            sum(1 for b in filled if b["mean_conf"] - b["accuracy"] > 0.10))
        # exit code is 1 only when overconfident bins exist
        self.assertEqual(rc, 1 if rep["overconfident_bins"] else 0)

    def test_unmeasured_fails(self):
        # an all-empty table measured nothing — must not exit 0
        from unittest.mock import patch
        from slice import __main__ as cli

        empty = {"state": "estimated", "basis": "x", "hit_radius_px": 8.0,
                 "bins": [{"lo": i / 10, "hi": (i + 1) / 10, "n": 0,
                           "mean_conf": None, "accuracy": None,
                           "mean_error": None} for i in range(10)],
                 "overconfident_bins": 0, "underconfident_bins": 0,
                 "note": ""}
        buf = io.StringIO()
        with patch.object(cli.calib, "report", return_value=empty):
            with redirect_stdout(buf):
                rc = main(["calib"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()

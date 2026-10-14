import io
import json
import unittest
from contextlib import redirect_stdout

from slice.__main__ import main


class TestBenchCli(unittest.TestCase):
    def test_gate_passes_and_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["bench", "--json"]), 0)
        rep = json.loads(buf.getvalue())
        for k in ("ms_per_estimate", "detection_rate",
                  "observed_rate", "mean_error_px", "mean_oks"):
            self.assertIn(k, rep)
        self.assertEqual(rep["detection_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

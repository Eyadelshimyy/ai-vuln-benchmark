import io
import json
import unittest
from contextlib import redirect_stdout

from slice.__main__ import main


class TestPriorchkCli(unittest.TestCase):
    def test_tables_sane(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["priorchk"]), 0)
        res = json.loads(buf.getvalue())
        self.assertEqual(res["verdict"], "sane")
        self.assertEqual(res["n_issues"], 0)
        self.assertGreaterEqual(res["models_checked"], 3)
        self.assertIn("head_order", res["checks"])


if __name__ == "__main__":
    unittest.main()

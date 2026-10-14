"""`slice bias` — per-joint systematic-error profile over fixtures."""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from slice.__main__ import main
from slice.skeleton import Skeleton


class BiasCliTest(unittest.TestCase):
    def test_report(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["bias"])
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["n_cases"], 2)
        self.assertTrue(rep["joints"])
        for r in rep["joints"].values():
            self.assertIn(r["verdict"],
                          ("systematic", "unbiased", "insufficient"))
        worst = rep["worst_joint"]
        self.assertIsNotNone(worst)
        # neck bias is fixed: worst joint stays under the bench gate
        self.assertLessEqual(worst["mean_error_px"], 10.0)
        self.assertEqual(rc, 0)

    def test_unmeasured_fails(self):
        # detection failure on every fixture must not pass the gate:
        # no measurable joints → unmeasured → exit 1.
        import slice.__main__ as cli
        with mock.patch.object(cli.pipeline.ESTIMATOR, "estimate",
                               return_value=Skeleton(1, 1)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["bias"])
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["state"], "unmeasured")
        self.assertFalse(rep["joints"])
        self.assertIsNone(rep["worst_joint"])
        self.assertEqual(rc, 1)

    def test_all_insufficient_fails(self):
        # every joint observed in only one fixture → all verdicts
        # "insufficient" — an entry in joints means "seen once", not
        # "measured". The gate must fail, not pass on zero evidence
        # (the old code: joints non-empty → measured → exit 0).
        import slice.__main__ as cli
        from slice.skeleton import Joint

        def one_joint(name):
            sk = Skeleton(1, 1)
            sk.set(Joint(name, 1.0, 1.0, 0.9, state="observed"))
            return sk

        skels = iter([one_joint("j_a"), one_joint("j_b")])
        with mock.patch.object(cli.pipeline.ESTIMATOR, "estimate",
                               side_effect=lambda bmp: next(skels)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["bias"])
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep["state"], "unmeasured")
        self.assertIsNone(rep["worst_joint"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for slice.priorchk — audit of the anatomy priors."""
import unittest

from slice import priorchk


class TestPriorchk(unittest.TestCase):
    def test_builtin_models_sane(self):
        r = priorchk.audit()
        self.assertEqual(r["verdict"], "sane",
                         r["issues"])
        self.assertEqual(r["models_checked"], 3)

    def test_missing_key_flagged(self):
        r = priorchk.audit({"bad": {"head_ratio": 0.2}})
        self.assertIn("keyset", [i["code"] for i in r["issues"]])
        self.assertEqual(r["verdict"], "suspicious")

    def test_limb_order_flagged(self):
        m = {"head_ratio": 0.13, "shoulder_ratio": 0.24,
             "hip_ratio": 0.17, "torso_ratio": 0.30,
             "upper_arm_ratio": 0.19, "forearm_ratio": 0.16,
             "thigh_ratio": 0.18, "shin_ratio": 0.25}
        r = priorchk.audit({"bad": m})
        self.assertIn("limb_order", [i["code"] for i in r["issues"]])

    def test_bounds_flagged(self):
        m = dict(next(iter(__import__("slice.anatomy",
                                     fromlist=["x"]).BODY_MODELS
                        .values())))
        m["head_ratio"] = 0.9
        r = priorchk.audit({"bad": m})
        self.assertIn("bounds", [i["code"] for i in r["issues"]])

    def test_head_order_flagged(self):
        m = {"adult": {"head_ratio": 0.4}, "child": {"head_ratio": 0.1}}
        m["adult"].update({k: 0.2 for k in priorchk.REQUIRED[1:]})
        m["child"].update({k: 0.2 for k in priorchk.REQUIRED[1:]})
        r = priorchk.audit(m)
        self.assertIn("head_order", [i["code"] for i in r["issues"]])

    def test_empty_models(self):
        r = priorchk.audit({})
        self.assertEqual(r["models_checked"], 0)
        self.assertEqual(r["verdict"], "sane")


if __name__ == "__main__":
    unittest.main()

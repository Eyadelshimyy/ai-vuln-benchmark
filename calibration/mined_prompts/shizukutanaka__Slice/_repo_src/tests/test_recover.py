"""Tests for slice.recover — staged fallback estimation."""
import unittest

from slice import evaluate, recover
from slice.bitmap import Bitmap


class TestRecover(unittest.TestCase):
    def test_primary_success(self):
        bmp, _ = evaluate.draw_case(160, 300)
        r = recover.recover(bmp)
        self.assertEqual(r["method"], "primary")
        self.assertEqual(r["state"], "observed")
        self.assertEqual(r["attempts"], 1)
        self.assertTrue(r["skeleton"].joints)
        # no recovery annotation on first-pass joints
        self.assertTrue(all("recovered" not in j.basis
                            for j in r["skeleton"].joints.values()))

    def test_relaxed_recovers_low_contrast(self):
        # figure only ~20 units off the background: under the 40 gate,
        # over the relaxed 20
        bmp, _ = evaluate.draw_case(
            160, 300, skin=(215, 215, 215, 255), bg=(235, 235, 235, 255))
        r = recover.recover(bmp)
        self.assertEqual(r["method"], "relaxed_threshold")
        self.assertEqual(r["state"], "observed")
        self.assertTrue(any("recovered: relaxed" in j.basis
                            for j in r["skeleton"].joints.values()))

    def test_component_retry(self):
        # non-person blob larger than the person → primary picks blob,
        # component retry finds the person
        bmp, truth = evaluate.draw_case(160, 300)
        for y in range(0, 40):
            for x in range(0, 140):
                bmp.set(x, y, (30, 30, 30, 255))
        r = recover.recover(bmp)
        self.assertEqual(r["state"], "observed")
        self.assertIn(r["method"], ("primary", "component_retry"))
        if r["method"] == "component_retry":
            self.assertTrue(any("component_retry" in j.basis
                                for j in r["skeleton"].joints.values()))

    def test_total_failure(self):
        bmp = Bitmap.new(160, 300, (235, 235, 235, 255))
        r = recover.recover(bmp)
        self.assertEqual(r["state"], "failed")
        self.assertEqual(r["method"], "failed")
        self.assertEqual(r["attempts"], len(recover.LADDER))
        self.assertFalse(r["skeleton"].joints)

    def test_relaxed_rung_keeps_profile_flags(self):
        # the relaxed rung must relax only the colour gate — the
        # caller's adaptive/shadow/clean profile must carry over,
        # otherwise the disclosed "gate relaxed to N" understates
        # how much the estimator was loosened
        seen = []
        real = recover.HeuristicPoseEstimator

        class Spy(real):
            def __init__(self, *a, **kw):
                seen.append(kw)
                super().__init__(*a, **kw)

        recover.HeuristicPoseEstimator = Spy
        try:
            est = real(adaptive=True, reject_shadow=True, clean=True)
            recover.recover(Bitmap.new(160, 300, (235, 235, 235, 255)),
                            estimator=est)
        finally:
            recover.HeuristicPoseEstimator = real
        self.assertEqual(len(seen), 1)  # only the relaxed rung builds one
        kw = seen[0]
        self.assertTrue(kw["adaptive"])
        self.assertTrue(kw["reject_shadow"])
        self.assertTrue(kw["clean"])

    def test_no_fabrication_on_failure(self):
        bmp = Bitmap.new(40, 40, (0, 0, 0, 255))
        r = recover.recover(bmp)
        self.assertIn(r["state"], ("observed", "failed"))
        if r["state"] == "failed":
            self.assertEqual(len(r["skeleton"].joints), 0)


if __name__ == "__main__":
    unittest.main()

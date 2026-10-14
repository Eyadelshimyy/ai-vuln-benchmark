"""Tests for slice.trust — per-joint trust synthesis."""
import unittest

from slice import evaluate, trust
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


def _skel():
    bmp, _ = evaluate.draw_case(160, 300)
    return HeuristicPoseEstimator().estimate(bmp)


class TestTrust(unittest.TestCase):
    def test_fixture_grades(self):
        r = trust.grade(_skel())
        self.assertEqual(r["state"], "heuristic")
        self.assertTrue(all(
            v["grade"] in ("high", "medium", "low")
            for v in r["joints"].values()))
        self.assertEqual(sum(r["distribution"].values()),
                         len(r["joints"]))

    def test_off_mask_low(self):
        sk = _skel()
        ev = {"joints": {"wrist_l": {"zone": "off_mask"}}}
        r = trust.grade(sk, evidence=ev)
        self.assertEqual(r["joints"]["wrist_l"]["grade"], "low")
        self.assertIn("off_mask", r["joints"]["wrist_l"]["factors"])
        self.assertNotIn("wrist_l", trust.trusted(r, "medium"))

    def test_predicted_off_mask_stays_medium(self):
        # a predicted joint sitting off the mask is expected
        # (priors legitimately place joints outside the silhouette)
        # — off_mask must not accuse a claim that was never made
        sk = Skeleton(160, 300)
        sk.set(Joint("nose", 80, 40, 0.9,
                     state="predicted", basis="fill"))
        ev = {"joints": {"nose": {"zone": "off_mask"}}}
        r = trust.grade(sk, evidence=ev)
        self.assertEqual(r["joints"]["nose"]["grade"], "medium")
        self.assertNotIn("off_mask", r["joints"]["nose"]["factors"])

    def test_unstable_low_sensitive_medium(self):
        sk = _skel()
        stab = {"joints": {"head": {"verdict": "unstable"},
                           "neck": {"verdict": "sensitive"},
                           "chest": {"verdict": "single_run"}}}
        r = trust.grade(sk, stability=stab)
        self.assertEqual(r["joints"]["head"]["grade"], "low")
        self.assertEqual(r["joints"]["neck"]["grade"], "medium")
        self.assertEqual(r["joints"]["chest"]["grade"], "medium")

    def test_calib_downgrade(self):
        sk = _skel()
        table = [{"lo": i / 10, "hi": (i + 1) / 10, "n": 5,
                  "mean_conf": 0.75, "accuracy": 0.2,
                  "mean_error": 30.0} for i in range(10)]
        r = trust.grade(sk, calib_table=table)
        self.assertTrue(all(
            v["grade"] == "low"
            for n, v in r["joints"].items()
            if sk.joints[n].state == "observed"))

    def test_predicted_capped_medium(self):
        sk = Skeleton(160, 300)
        sk.set(Joint("nose", 80, 40, 0.9,
                     state="predicted", basis="fill"))
        r = trust.grade(sk)
        self.assertEqual(r["joints"]["nose"]["grade"], "medium")
        self.assertIn("predicted_fill",
                      r["joints"]["nose"]["factors"])

    def test_trusted_filter(self):
        r = trust.grade(_skel())
        names = trust.trusted(r, "medium")
        self.assertTrue(names)
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()

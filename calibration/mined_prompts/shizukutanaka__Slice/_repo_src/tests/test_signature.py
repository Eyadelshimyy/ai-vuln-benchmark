import unittest

from tests import synthetic_person

from slice import signature as sig
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


class TestSignature(unittest.TestCase):
    def setUp(self):
        self.est = HeuristicPoseEstimator()
        self.skel = self.est.estimate(synthetic_person())
        self.s = sig.signature(self.skel)

    def test_fixed_length(self):
        self.assertEqual(len(self.s), 17 * 2 + 4)

    def test_same_pose_zero_distance(self):
        self.assertEqual(sig.distance(self.s, self.s), 0.0)

    def test_scale_invariance(self):
        big = self.est.estimate(synthetic_person(320, 600))
        d = sig.distance(self.s, sig.signature(big))
        self.assertIsNotNone(d)
        self.assertLess(d, 0.15)

    def test_predicted_bones_not_evidence(self):
        # a guessed arm must not fingerprint like a measured one:
        # predicted endpoints drop out of the vector like missing ones
        from slice import predict
        other = Skeleton(self.skel.image_width, self.skel.image_height)
        for n, j in self.skel.joints.items():
            other.set(Joint(n, j.x, j.y, j.confidence, state=j.state))
        del other.joints["elbow_r"]
        del other.joints["wrist_r"]
        predict.complete(other)
        d = sig.distance(self.s, sig.signature(other))
        self.assertGreater(d, 0.05)

    def test_missing_feet_stay_torso_normalised(self):
        # without feet the scalars must not become raw pixels
        other = Skeleton(self.skel.image_width, self.skel.image_height)
        for n, j in self.skel.joints.items():
            if n.startswith("foot"):
                continue
            other.set(Joint(n, j.x, j.y, j.confidence, state=j.state))
        v = sig.signature(other)
        self.assertTrue(all(abs(s) < 2.0 for s in v[-4:]),
                        "scalars leaked raw-pixel magnitudes")
        d = sig.distance(self.s, v)
        self.assertLess(d, 0.5)

    def test_predicted_torso_not_in_scalars(self):
        # predicted neck/pelvis must not write prior geometry into
        # the lean scalar or the normaliser — same rule as the bones
        other = Skeleton(self.skel.image_width, self.skel.image_height)
        for n, j in self.skel.joints.items():
            other.set(Joint(n, j.x, j.y, j.confidence,
                            state="predicted" if n in ("neck", "pelvis")
                            else j.state))
        v = sig.signature(other)
        # torso-lean scalar = 0.0 (missing), not prior geometry
        self.assertEqual(v[-4], 0.0)

    def test_no_scale_scalars_are_missing_not_raw_px(self):
        # torso/head/feet all unmeasured: body scale is undefined —
        # `else 1.0` would write raw-px spans into normalised slots
        other = Skeleton(self.skel.image_width, self.skel.image_height)
        keep = {"wrist_l", "wrist_r", "ankle_l", "ankle_r"}
        for n, j in self.skel.joints.items():
            if n in keep:
                other.set(Joint(n, j.x, j.y, j.confidence, state="observed"))
        v = sig.signature(other)
        # last 4 slots: torso-lean + wrist/ankle/shoulder spans
        self.assertEqual(v[-4:], [0.0, 0.0, 0.0, 0.0])

    def test_bent_arm_changes_signature(self):
        other = Skeleton(self.skel.image_width, self.skel.image_height)
        for n, j in self.skel.joints.items():
            other.set(Joint(n, j.x, j.y, j.confidence, state=j.state))
        other.joints["wrist_l"].y -= 120  # raise the hand overhead
        d = sig.distance(self.s, sig.signature(other))
        self.assertGreater(d, 0.02)


if __name__ == "__main__":
    unittest.main()

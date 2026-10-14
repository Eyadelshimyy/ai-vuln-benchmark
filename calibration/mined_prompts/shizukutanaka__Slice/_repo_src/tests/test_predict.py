import unittest

from tests import synthetic_person

from slice import predict
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import OBSERVED, PREDICTED, Skeleton


class TestPredict(unittest.TestCase):
    def test_mirror_completion(self):
        sk = Skeleton(100, 200)
        from slice.skeleton import Joint
        sk.set(Joint("spine", 50, 100, 0.8, OBSERVED))
        sk.set(Joint("hip_l", 40, 120, 0.8, OBSERVED))
        sk.set(Joint("hip_r", 60, 120, 0.8, OBSERVED))
        sk.set(Joint("knee_l", 38, 160, 0.8, OBSERVED))
        sk.set(Joint("ankle_l", 36, 195, 0.8, OBSERVED))
        added = predict.complete(sk)
        names = {j.name for j in added}
        self.assertIn("knee_r", names)
        kr = sk.get("knee_r")
        self.assertEqual(kr.state, PREDICTED)
        self.assertAlmostEqual(kr.x, 62, delta=2)
        self.assertLess(kr.confidence, sk.get("knee_l").confidence)

    def test_prior_completion_when_nothing_visible(self):
        sk = Skeleton(100, 200)
        from slice.skeleton import Joint
        sk.set(Joint("shoulder_l", 30, 50, 0.8, OBSERVED))
        added = predict.complete(sk)
        names = {j.name for j in added}
        self.assertIn("elbow_l", names)
        el = sk.get("elbow_l")
        self.assertEqual(el.state, PREDICTED)
        self.assertGreater(el.y, 50)

    def test_mid_chain_interpolation(self):
        # elbow missing but shoulder and wrist known -> elbow sits on the
        # segment between them, not on a straight prior drop
        sk = Skeleton(100, 200)
        from slice.skeleton import Joint
        sk.set(Joint("shoulder_l", 30, 60, 0.8, OBSERVED))
        sk.set(Joint("wrist_l", 10, 120, 0.8, OBSERVED))
        predict.complete(sk)
        el = sk.get("elbow_l")
        self.assertEqual(el.state, PREDICTED)
        self.assertIn("interpolated", el.basis)
        # adult prior: upper/(upper+fore) ≈ .545 -> el ≈ lerp(sh, wr, .545)
        self.assertAlmostEqual(el.x, 30 + (10 - 30) * 0.545, delta=3)
        self.assertAlmostEqual(el.y, 60 + (120 - 60) * 0.545, delta=3)
        # the straight prior drop would have kept x near shoulder+x
        self.assertLess(el.x, 30)

    def test_predicted_extremity_does_not_inflate_prior_scale(self):
        # A predicted head far above the observed body must not stretch
        # the body_h that scales every prior distance — otherwise prior
        # geometry feeds back into the table it draws from.
        from slice.skeleton import Joint

        def chest_y(skel):
            predict.complete(skel)
            return skel.get("chest").y

        a = Skeleton(100, 200)
        a.set(Joint("neck", 50, 40, 0.8, OBSERVED))
        a.set(Joint("pelvis", 50, 100, 0.8, OBSERVED))

        b = Skeleton(100, 200)
        b.set(Joint("neck", 50, 40, 0.8, OBSERVED))
        b.set(Joint("pelvis", 50, 100, 0.8, OBSERVED))
        b.set(Joint("head", 50, -400, 0.2, PREDICTED, "prior off neck"))

        self.assertEqual(chest_y(a), chest_y(b))

    def test_mirror_axis_uses_centroid_not_image_center(self):
        # a right-of-centre figure with only a shoulder observed:
        # mirroring about the IMAGE centre (x=50) would plant the
        # counterpart in empty space; the measured component centroid
        # is the honest axis
        sk = Skeleton(100, 200, centroid=(80, 100))
        from slice.skeleton import Joint
        sk.set(Joint("shoulder_r", 85, 60, 0.8, OBSERVED))
        predict.complete(sk)
        sl = sk.get("shoulder_l")
        self.assertEqual(sl.state, PREDICTED)
        self.assertAlmostEqual(sl.x, 2 * 80 - 85, delta=1)
        self.assertNotAlmostEqual(sl.x, 2 * 50 - 85, delta=1)

    def test_basis_names_the_prior_model(self):
        # prior-derived joints must say which BODY_MODEL table produced
        # them — "prior off X" alone hid the source anatomy
        sk = Skeleton(100, 200)
        from slice.skeleton import Joint
        sk.set(Joint("shoulder_l", 30, 50, 0.8))
        predict.complete(sk, model="child")
        priors = [j for j in sk.joints.values()
                  if j.basis and j.basis.startswith("prior")]
        self.assertTrue(priors)
        for j in priors:
            self.assertIn("(child)", j.basis)
        # the basis prefix still classifies the evidence category
        from slice import basis
        self.assertEqual(basis.category(sk.get("elbow_l").basis), "prior")

    def test_full_pipeline_needs_no_prior_fallback_for_missing(self):
        sk = HeuristicPoseEstimator().estimate(synthetic_person())
        added = predict.complete(sk)
        for j in added:
            self.assertEqual(j.state, PREDICTED)
            self.assertTrue(0 < j.confidence <= 1)
        # a completed skeleton should have most vocabulary filled
        self.assertGreaterEqual(len(sk.joints), 14)


if __name__ == "__main__":
    unittest.main()

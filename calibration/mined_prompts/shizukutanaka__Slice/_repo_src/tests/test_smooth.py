import unittest

from tests import synthetic_person

from slice import smooth
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, Skeleton


def _copy_with(skel, jitter_by=None):
    out = Skeleton(skel.image_width, skel.image_height)
    for n, j in skel.joints.items():
        dx = jitter_by(n) if jitter_by else 0.0
        out.set(Joint(n, j.x + dx, j.y, j.confidence, state=j.state))
    return out


class TestSmooth(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_jitter_reduced(self):
        # frames alternate pelvis.x ±4
        series = [
            _copy_with(self.skel,
                       jitter_by=lambda n: 4.0 if i % 2 else -4.0)
            for i in range(5)]
        sm = smooth.smooth(series, radius=1)
        self.assertLess(smooth.jitter(sm), smooth.jitter(series))

    def test_missing_frames_keep_missing(self):
        series = [_copy_with(self.skel) for _ in range(3)]
        series[1].joints.pop("wrist_l")
        sm = smooth.smooth(series)
        self.assertNotIn("wrist_l", sm[1].joints)

    def test_observed_not_dragged_by_prediction(self):
        # a predicted neighbour must not pull an observed joint
        # toward its fabricated location
        series = [_copy_with(self.skel) for _ in range(3)]
        w = series[1].joints["wrist_l"]
        w.state = "predicted"
        w.x += 40
        sm = smooth.smooth(series)
        self.assertEqual(sm[0].joints["wrist_l"].x,
                         self.skel.joints["wrist_l"].x)

    def test_state_and_confidence_preserved(self):
        series = [_copy_with(self.skel) for _ in range(3)]
        series[1].joints["wrist_l"].state = "predicted"
        sm = smooth.smooth(series)
        self.assertEqual(sm[1].joints["wrist_l"].state, "predicted")


if __name__ == "__main__":
    unittest.main()

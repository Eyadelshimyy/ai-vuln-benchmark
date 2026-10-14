"""Tests for slice.gate — unified quality verdict."""
import unittest

from slice import evaluate, gate, knowledge
from slice.pose import HeuristicPoseEstimator
from slice.skeleton import Joint, OBSERVED, Skeleton


def _fixture():
    bmp, _ = evaluate.draw_case(160, 300)
    return HeuristicPoseEstimator().estimate(bmp)





class TestGate(unittest.TestCase):
    def test_good_fixture_passes(self):
        sk = _fixture()
        r = gate.check(skel=sk)
        self.assertEqual(r["verdict"], "pass")
        self.assertTrue(gate.keep(r))
        self.assertTrue(r["layers"]["detection"]["found"])

    def test_empty_skeleton_fails(self):
        r = gate.check(skel=Skeleton(160, 300))
        self.assertEqual(r["verdict"], "fail")
        self.assertFalse(gate.keep(r))
        self.assertIn("detection:no_person", r["reasons"])

    def test_doc_layer(self):
        sk = _fixture()
        doc = knowledge.build(sk, {}, source_name="t",
                              engine={"name": "t", "version": "0"})
        r = gate.check(doc=doc)
        self.assertIn("document", r["layers"])
        self.assertIn(r["verdict"], ("pass", "warn"))

    def test_invalid_doc_fails(self):
        r = gate.check(doc={"schema": "slice.knowledge/v1"})
        self.assertEqual(r["verdict"], "fail")

    def test_broken_geometry_warns_or_fails(self):
        sk = Skeleton(160, 300)
        # feet above head → inverted body
        sk.set(Joint("head", 80, 290, 0.9, OBSERVED, basis="x"))
        sk.set(Joint("foot_l", 80, 10, 0.9, OBSERVED, basis="x"))
        sk.set(Joint("foot_r", 82, 10, 0.9, OBSERVED, basis="x"))
        r = gate.check(skel=sk)
        self.assertNotEqual(r["verdict"], "pass")

    def test_no_inputs_passes_vacuously(self):
        r = gate.check()
        self.assertEqual(r["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()

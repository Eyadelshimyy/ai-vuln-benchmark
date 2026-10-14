import json
import unittest

from slice.gltf import export, to_gltf
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _est():
    return HeuristicPoseEstimator().estimate(synthetic_person())


def _node(doc, name):
    return next(n for n in doc["nodes"] if n["name"] == name)


class TestGltf(unittest.TestCase):
    def test_valid_gltf_document(self):
        doc = json.loads(export(_est()))
        self.assertEqual(doc["asset"]["version"], "2.0")
        self.assertEqual(doc["scene"], 0)
        self.assertTrue(doc["nodes"])
        names = {n["name"] for n in doc["nodes"]}
        self.assertIn("pelvis", names)

    def test_every_joint_becomes_a_node(self):
        skel = _est()
        doc = to_gltf(skel)
        self.assertEqual(len(doc["nodes"]), len(skel.joints))
        for i, n in enumerate(doc["nodes"]):
            for c in n["children"]:
                self.assertLess(c, len(doc["nodes"]))
                self.assertIsInstance(c, int)

    def test_parent_relative_translation(self):
        skel = _est()
        doc = to_gltf(skel)
        neck = _node(doc, "neck")
        head = _node(doc, "head")
        pj, cj = skel.joints["neck"], skel.joints["head"]
        self.assertAlmostEqual(head["translation"][0], cj.x - pj.x, 3)
        # image y is down, glTF y is up -> delta negated
        self.assertAlmostEqual(head["translation"][1], -(cj.y - pj.y), 3)
        self.assertEqual(head["translation"][2], 0.0)
        # neck's children must include head's index
        hi = doc["nodes"].index(head)
        self.assertIn(hi, neck["children"])

    def test_root_translation_is_absolute(self):
        skel = _est()
        doc = to_gltf(skel)
        pelvis = _node(doc, "pelvis")
        self.assertAlmostEqual(pelvis["translation"][0],
                               skel.joints["pelvis"].x, 3)
        self.assertAlmostEqual(pelvis["translation"][1],
                               -skel.joints["pelvis"].y, 3)
        pi = doc["nodes"].index(pelvis)
        self.assertIn(pi, doc["scenes"][0]["nodes"])

    def test_extras_carry_honesty_contract(self):
        skel = _est()
        skel.joints["wrist_l"].state = "predicted"
        skel.joints["wrist_l"].basis = "mirrored from wrist_r"
        doc = to_gltf(skel)
        w = _node(doc, "wrist_l")
        self.assertEqual(w["extras"]["state"], "predicted")
        self.assertIn("mirrored", w["extras"]["basis"])
        self.assertIn("confidence", w["extras"])

    def test_missing_parent_promotes_to_root(self):
        skel = _est()
        # Remove the whole torso chain: pelvis/chest/neck gone.
        for n in ("pelvis", "chest", "neck"):
            del skel.joints[n]
        doc = to_gltf(skel)
        names = {doc["nodes"][i]["name"] for i in
                 doc["scenes"][0]["nodes"]}
        # head's parent chain (neck->chest->pelvis) is gone -> head is
        # a scene root; hip_l too (pelvis gone).
        self.assertIn("head", names)
        self.assertIn("hip_l", names)

    def test_flatness_disclosed_not_faked(self):
        doc = to_gltf(_est())
        self.assertIn("Z=0", doc["asset"]["extras"]["note"])
        for n in doc["nodes"]:
            self.assertEqual(n["translation"][2], 0.0)


if __name__ == "__main__":
    unittest.main()

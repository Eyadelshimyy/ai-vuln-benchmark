"""Test for the Phase-3 end-to-end demo."""
import importlib.util
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "demo_3d",
    os.path.join(os.path.dirname(__file__), "..", "examples",
                 "demo_3d.py"))
demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo)


class TestDemo3d(unittest.TestCase):
    def test_full_chain(self):
        out = demo.run(tempfile.mkdtemp())
        self.assertGreaterEqual(out["joints"], 17)
        self.assertGreaterEqual(out["bones"], 15)
        self.assertTrue(out["bvh"].endswith(".bvh"))
        doc = json.load(open(out["gltf"]))
        self.assertEqual(doc["asset"]["version"], "2.0")
        self.assertGreater(len(doc["nodes"]), 10)
        self.assertIn("ROOT", open(out["bvh"]).read(64))


if __name__ == "__main__":
    unittest.main()

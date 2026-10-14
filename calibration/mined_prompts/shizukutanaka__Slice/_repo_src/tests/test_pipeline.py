import unittest

from tests import synthetic_person

from slice import bitmap, knowledge, pipeline


class TestPipeline(unittest.TestCase):
    def test_analyze_produces_document(self):
        doc = pipeline.analyze(
            bitmap.encode_png(synthetic_person()), source_name="x.png")
        out = pipeline.strip_runtime(doc)
        for key in ("schema", "skeleton", "pose", "style", "ratio",
                    "prediction", "warnings", "analysis"):
            self.assertIn(key, out, key)
        self.assertIsInstance(out["warnings"], list)
        self.assertNotIn("no_observed_head", out["warnings"])

    def test_analysis_block_integrates_layers(self):
        doc = pipeline.analyze(
            bitmap.encode_png(synthetic_person()))
        a = doc["analysis"]
        for key in ("angles", "symmetry", "balance", "spine",
                    "gesture", "dynamics", "occlusion", "frame",
                    "consistency"):
            self.assertIn(key, a, key)
        self.assertIsNotNone(a["frame"])
        self.assertIn("elbow_l_flex", a["angles"])
        self.assertIsInstance(a["consistency"]["issues"], list)

    def test_analysis_degrades_on_empty_image(self):
        bmp = bitmap.Bitmap.new(60, 60, (255, 255, 255, 255))
        doc = pipeline.analyze(bitmap.encode_png(bmp))
        self.assertEqual(doc["analysis"]["gesture"]["count"], 0)
        self.assertNotIn("frame", doc["analysis"])
        self.assertEqual(doc["analysis"]["consistency"]["issues"],
                         ["no_body_extent"])
        self.assertEqual(
            knowledge.validate(pipeline.strip_runtime(doc)), [])

    def test_warnings_for_sparse_evidence(self):
        # an image with almost no figure should warn
        bmp = bitmap.Bitmap.new(60, 60, (255, 255, 255, 255))
        doc = pipeline.analyze(bitmap.encode_png(bmp))
        self.assertIn("few_observed_joints", doc["warnings"])
        self.assertIn("no_observed_torso", doc["warnings"])
        self.assertIn("no_observed_head", doc["warnings"])

    def test_warnings_when_torso_predicted(self):
        # limbs observed but the kinematic root fabricated — the
        # document must flag that its anchor is unmeasured
        from slice.skeleton import Joint, PREDICTED, Skeleton
        skel = Skeleton(100, 200)
        for n in ("pelvis", "chest", "neck", "head"):
            skel.set(Joint(n, 50, 60, 0.5, PREDICTED, "prior"))
        for i, n in enumerate(("shoulder_l", "elbow_l", "wrist_l",
                               "shoulder_r", "elbow_r", "wrist_r",
                               "hip_l", "knee_l", "ankle_l")):
            skel.set(Joint(n, 30 + i * 5, 80, 0.9))
        self.assertIn("no_observed_torso",
                      pipeline._warnings(skel))

    def test_unknown_model_warns(self):
        # a bogus model name silently fell back to the adult prior —
        # the document must flag that the request was not honored
        doc = pipeline.analyze(
            bitmap.encode_png(synthetic_person()),
            model="nonexistent-model")
        self.assertIn("unknown_body_model", doc["warnings"])
        # and a real model must not warn
        doc2 = pipeline.analyze(
            bitmap.encode_png(synthetic_person()), model="adult")
        self.assertNotIn("unknown_body_model", doc2["warnings"])

    def test_frame_source_disclosed_on_downscale(self):
        # joint coordinates live in the working space; the source
        # resolution must be recorded or they cannot be mapped back
        bmp = synthetic_person(width=900, height=1200)
        doc = pipeline.analyze(bitmap.encode_png(bmp))
        frame = doc["skeleton"]["frame"]
        self.assertNotEqual((frame["width"], frame["height"]),
                            (900, 1200))
        self.assertEqual(frame["source"], {"width": 900, "height": 1200})
        # at working resolution nothing extra is claimed
        small = synthetic_person(width=100, height=200)
        doc2 = pipeline.analyze(bitmap.encode_png(small))
        self.assertNotIn("source", doc2["skeleton"]["frame"])

    def test_strip_runtime_drops_internals(self):
        doc = pipeline.analyze(bitmap.encode_png(synthetic_person()))
        self.assertIn("_bitmap", doc)
        out = pipeline.strip_runtime(doc)
        self.assertFalse(any(k.startswith("_") for k in out))

    def test_robust_profile_recorded_and_detects(self):
        raw = bitmap.encode_png(synthetic_person())
        doc = pipeline.analyze(raw, robust=True)
        self.assertEqual(doc["engine"]["profile"], "robust")
        self.assertGreater(
            len(doc["prediction"]["observed"]), 8)
        # default profile stays default
        doc2 = pipeline.analyze(raw)
        self.assertEqual(doc2["engine"]["profile"], "default")

    def test_mask_threshold_disclosed(self):
        doc = pipeline.analyze(bitmap.encode_png(synthetic_person()))
        mt = doc["engine"]["mask_threshold"]
        self.assertEqual(mt["method"], "fixed")
        self.assertEqual(mt["value"], 40)
        raw = bitmap.encode_png(synthetic_person())
        mt = pipeline.analyze(raw, robust=True
                              )["engine"]["mask_threshold"]
        self.assertIn(mt["method"], ("otsu", "fixed"))
        self.assertGreater(mt["value"], 0)


if __name__ == "__main__":
    unittest.main()

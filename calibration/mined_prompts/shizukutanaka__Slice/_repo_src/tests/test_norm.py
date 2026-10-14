import unittest

from tests import synthetic_person

from slice import norm
from slice.pose import HeuristicPoseEstimator


class TestNorm(unittest.TestCase):
    def setUp(self):
        self.skel = HeuristicPoseEstimator().estimate(synthetic_person())

    def test_crop_translates(self):
        head = self.skel.point("head")
        out = norm.crop(self.skel, 10, 20, 200, 200)
        self.assertEqual(out.point("head"),
                         (head[0] - 10, head[1] - 20))
        # the declared frame is the crop's, not the source's
        self.assertEqual((out.image_width, out.image_height), (200, 200))

    def test_resize_declares_new_frame(self):
        out = norm.resize(self.skel, 300, 400, 150, 200)
        self.assertEqual((out.image_width, out.image_height), (150, 200))

    def test_unit_declares_unit_frame(self):
        u = norm.to_unit(self.skel, 300, 400)
        self.assertEqual((u.image_width, u.image_height), (1, 1))
        back = norm.from_unit(u, 300, 400)
        self.assertEqual((back.image_width, back.image_height),
                         (300, 400))

    def test_crop_marks_out_of_frame(self):
        head = self.skel.point("head")
        out = norm.crop(self.skel, head[0] + 100, 0, 300, 300)
        # demoted to a legal state (predicted) — "out_of_frame" was
        # outside the validate() vocabulary and made the doc unsavable
        self.assertEqual(out.joints["head"].state, "predicted")
        self.assertIn("lost to transform", out.joints["head"].basis)
        self.assertIn("was observed", out.joints["head"].basis)

    def test_cropped_skeleton_validates(self):
        # a joint lost to the transform must not make the document
        # fail validate() / refuse KnowledgeStore.save()
        from slice import knowledge
        head = self.skel.point("head")
        out = norm.crop(self.skel, head[0] + 100, 0, 300, 300)
        doc = knowledge.build(out, {}, engine={"name": "test"})
        self.assertEqual(knowledge.validate(doc), [])

    def test_centroid_transformed_too(self):
        # the centroid is a frame-space point: a transform that moves
        # joints must move it too, or downstream consumers get a stale
        # coordinate from the old frame
        self.skel.centroid = (200, 150)
        c = norm.crop(self.skel, 100, 20, 200, 200)
        self.assertEqual(c.centroid, (100, 130))
        r = norm.resize(self.skel, 400, 300, 200, 150)
        self.assertAlmostEqual(r.centroid[0], 100)
        self.assertAlmostEqual(r.centroid[1], 75)
        u = norm.to_unit(self.skel, 400, 300)
        self.assertAlmostEqual(u.centroid[0], 0.5)
        b = norm.from_unit(u, 400, 300)
        self.assertAlmostEqual(b.centroid[0], 200)
        self.assertAlmostEqual(b.centroid[1], 150)

    def test_resize_scales(self):
        head = self.skel.point("head")
        out = norm.resize(self.skel, 300, 400, 150, 200)
        self.assertAlmostEqual(out.point("head")[0], head[0] / 2)
        self.assertAlmostEqual(out.point("head")[1], head[1] / 2)

    def test_unit_roundtrip(self):
        u = norm.to_unit(self.skel, 300, 400)
        back = norm.from_unit(u, 300, 400)
        for name in self.skel.joints:
            self.assertAlmostEqual(back.joints[name].x,
                                   self.skel.joints[name].x)
            self.assertAlmostEqual(back.joints[name].y,
                                   self.skel.joints[name].y)


if __name__ == "__main__":
    unittest.main()

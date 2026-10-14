import unittest
import xml.etree.ElementTree as ET

from slice.pose import HeuristicPoseEstimator
from slice.svg import render
from tests import synthetic_person

NS = "{http://www.w3.org/2000/svg}"


def _est():
    return HeuristicPoseEstimator().estimate(synthetic_person())


def _parse(text):
    return ET.fromstring(text)


class TestSvg(unittest.TestCase):
    def test_valid_svg_document(self):
        root = _parse(render(_est()))
        self.assertEqual(root.tag, NS + "svg")
        self.assertIn("viewBox", root.attrib)

    def test_lines_match_present_bones(self):
        skel = _est()
        root = _parse(render(skel))
        lines = root.findall(NS + "line")
        from slice.landmarks import BONES
        want = sum(1 for a, b in BONES
                   if a in skel.joints and b in skel.joints)
        self.assertEqual(len(lines), want)

    def test_observed_blue_predicted_orange_dashed(self):
        skel = _est()
        skel.joints["wrist_l"].state = "predicted"
        root = _parse(render(skel))
        # the elbow_l-wrist_l bone: one predicted endpoint -> orange,
        # dashed
        for line in root.findall(NS + "line"):
            title = line.find(NS + "title").text
            if "predicted" in title:
                self.assertEqual(line.attrib["stroke"], "#ff9600")
                self.assertIn("stroke-dasharray", line.attrib)
                break
        else:
            self.fail("no predicted bone rendered")
        # an observed bone stays solid blue
        solid = [l for l in root.findall(NS + "line")
                 if "stroke-dasharray" not in l.attrib]
        self.assertTrue(solid)
        self.assertEqual(solid[0].attrib["stroke"], "#1e78ff")

    def test_joint_circles_and_tooltips(self):
        skel = _est()
        j = skel.joints["head"]
        j.basis = "topmost silhouette run"
        root = _parse(render(skel))
        circles = root.findall(NS + "circle")
        self.assertEqual(len(circles), 2 * len(skel.joints))  # +white
        tips = [c.find(NS + "title").text for c in circles
                if c.find(NS + "title") is not None]
        self.assertTrue(any("head: observed" in t for t in tips))
        self.assertTrue(any("topmost" in t for t in tips))

    def test_scale_scales_coordinates(self):
        skel = _est()
        jx, jy = skel.joints["head"].x, skel.joints["head"].y
        root = _parse(render(skel, scale=2.0))
        for c in root.findall(NS + "circle"):
            if abs(float(c.attrib["cx"]) - jx * 2) < 0.5:
                self.assertAlmostEqual(float(c.attrib["cy"]), jy * 2,
                                       places=1)
                break
        else:
            self.fail("head joint not found at scaled position")

    def test_empty_skeleton_is_valid_empty_svg(self):
        from slice.skeleton import Skeleton
        root = _parse(render(Skeleton(10, 10)))
        self.assertEqual(root.findall(NS + "line"), [])
        self.assertEqual(root.findall(NS + "circle"), [])

    def test_basis_text_is_escaped(self):
        skel = _est()
        skel.joints["head"].basis = 'a <b> & "c"'
        text = render(skel)
        self.assertIn("&lt;b&gt;", text)
        self.assertIn("&amp;", text)
        _parse(text)  # still well-formed


if __name__ == "__main__":
    unittest.main()

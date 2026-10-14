"""Tests for render.overlay_multi — per-person hue overlay."""
import unittest

from slice import evaluate, pose, render
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _two_people() -> Bitmap:
    person, _ = evaluate.draw_case(100, 240)
    bmp = Bitmap.new(220, 280, BG)
    for dx in (10, 110):
        for y in range(person.height):
            for x in range(person.width):
                if person.get(x, y) != BG:
                    bmp.set(dx + x, 20 + y, person.get(x, y))
    return bmp


class TestOverlayMulti(unittest.TestCase):
    def test_two_hues(self):
        bmp = _two_people()
        skels = pose.HeuristicPoseEstimator().estimate_multi(bmp)
        out = render.overlay_multi(bmp, skels)
        hues = set()
        for p in range(len(skels)):
            t = render.PEOPLE_TINTS[p][:3]
            found = False
            for y in range(out.height):
                for x in range(out.width):
                    if out.get(x, y)[:3] == t:
                        found = True
                        break
                if found:
                    break
            self.assertTrue(found, f"tint {t} missing")
            hues.add(t)

    def test_empty_list(self):
        bmp, _ = evaluate.draw_case()
        out = render.overlay_multi(bmp, [])
        self.assertEqual(out.data, bmp.data)

    def test_png(self):
        bmp = _two_people()
        skels = pose.HeuristicPoseEstimator().estimate_multi(bmp)
        png = render.overlay_multi_png(bmp, skels)
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_single_overlay_unchanged(self):
        bmp, _ = evaluate.draw_case()
        sk = pose.HeuristicPoseEstimator().estimate(bmp)
        out = render.overlay(bmp, sk)
        # single path still uses the classic BLUE observed colour
        self.assertTrue(
            any(out.get(x, y)[:3] == render.BLUE[:3]
                for y in range(out.height)
                for x in range(0, out.width, 4)))


if __name__ == "__main__":
    unittest.main()

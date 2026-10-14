"""Tests for pipeline.analyze_multi — multi-person Knowledge docs."""
import unittest

from slice import bitmap, evaluate, knowledge, pipeline
from slice.bitmap import Bitmap

BG = (235, 235, 235, 255)


def _stamp(dst, src, dx, dy):
    for y in range(src.height):
        for x in range(src.width):
            p = src.get(x, y)
            if p != BG:
                dst.set(dx + x, dy + y, p)


def _two_people_png() -> bytes:
    person, _ = evaluate.draw_case(120, 260)
    wide = Bitmap.new(320, 300, BG)
    _stamp(wide, person, 10, 20)
    _stamp(wide, person, 190, 20)
    return bitmap.encode_png(wide)


class TestAnalyzeMulti(unittest.TestCase):
    def test_two_docs(self):
        docs = pipeline.analyze_multi(_two_people_png())
        self.assertEqual(len(docs), 2)
        for i, d in enumerate(docs):
            self.assertEqual(d["people"]["count"], 2)
            self.assertEqual(d["people"]["index"], i)
            self.assertEqual(knowledge.validate(
                pipeline.strip_runtime(d)), [])
            self.assertGreater(pipeline.observed_count(d), 10)

    def test_single_person(self):
        bmp, _ = evaluate.draw_case()
        docs = pipeline.analyze_multi(bitmap.encode_png(bmp))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["people"]["count"], 1)

    def test_empty(self):
        bmp = Bitmap.new(80, 80, BG)
        docs = pipeline.analyze_multi(bitmap.encode_png(bmp))
        self.assertEqual(docs, [])

    def test_analyze_still_single(self):
        bmp, _ = evaluate.draw_case()
        doc = pipeline.analyze(bitmap.encode_png(bmp))
        self.assertNotIn("people", doc)
        self.assertEqual(pipeline.people_count(doc), 1)
        self.assertEqual(pipeline.people_count(
            pipeline.analyze_multi(bitmap.encode_png(bmp))), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for the `slice batch` CLI — directory -> KnowledgeStore."""
import os
import tempfile
import unittest

from tests import synthetic_person

from slice import __main__ as cli
from slice import bitmap, knowledge


class TestBatch(unittest.TestCase):
    def test_batch_saves_all_images(self):
        with tempfile.TemporaryDirectory() as imgs, \
                tempfile.TemporaryDirectory() as store_dir:
            for name in ("a.png", "b.png"):
                with open(os.path.join(imgs, name), "wb") as f:
                    f.write(bitmap.encode_png(synthetic_person()))
            rc = cli.main(["batch", imgs, "--store", store_dir])
            self.assertEqual(rc, 0)
            self.assertEqual(len(knowledge.KnowledgeStore(store_dir)
                                 .list()), 2)

    def test_batch_skips_undecodable(self):
        with tempfile.TemporaryDirectory() as imgs, \
                tempfile.TemporaryDirectory() as store_dir:
            with open(os.path.join(imgs, "ok.png"), "wb") as f:
                f.write(bitmap.encode_png(synthetic_person()))
            with open(os.path.join(imgs, "bad.png"), "wb") as f:
                f.write(b"not an image at all")
            with open(os.path.join(imgs, "notes.txt"), "w") as f:
                f.write("ignored")
            rc = cli.main(["batch", imgs, "--store", store_dir])
            self.assertEqual(rc, 0)
            self.assertEqual(len(knowledge.KnowledgeStore(store_dir)
                                 .list()), 1)

    def test_batch_robust_profile(self):
        # --robust must reach the estimator: the saved doc records the
        # profile it was produced under
        with tempfile.TemporaryDirectory() as imgs, \
                tempfile.TemporaryDirectory() as store_dir:
            with open(os.path.join(imgs, "a.png"), "wb") as f:
                f.write(bitmap.encode_png(synthetic_person()))
            rc = cli.main(["batch", imgs, "--store", store_dir,
                           "--robust"])
            self.assertEqual(rc, 0)
            doc = knowledge.KnowledgeStore(store_dir).list()[0]
            full = knowledge.KnowledgeStore(store_dir).get(doc["id"])
            self.assertEqual(full["engine"]["profile"], "robust")

    def test_batch_empty_dir_fails(self):
        with tempfile.TemporaryDirectory() as imgs, \
                tempfile.TemporaryDirectory() as store_dir:
            self.assertEqual(
                cli.main(["batch", imgs, "--store", store_dir]), 1)


if __name__ == "__main__":
    unittest.main()

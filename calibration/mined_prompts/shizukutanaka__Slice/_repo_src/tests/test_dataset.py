import csv
import io
import json
import os
import tempfile
import unittest

from slice.dataset import (from_store, joint_rows, summary_rows,
                           to_csv, to_jsonl)
from slice.knowledge import KnowledgeStore, build
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _doc(kid_seed="a"):
    skel = HeuristicPoseEstimator().estimate(synthetic_person())
    doc = build(skel, {"shoulder_hip": 1.2}, {"label": "standing"},
                source_name="test.png", image_sha256="f" * 64,
                engine={"name": "heuristic", "version": "1"})
    return doc


class TestDataset(unittest.TestCase):
    def test_summary_row_fields(self):
        r = summary_rows([_doc()])[0]
        self.assertEqual(r["source_name"], "test.png")
        self.assertGreater(r["n_joints"], 0)
        self.assertEqual(r["n_observed"] + r["n_predicted"],
                         r["n_joints"])
        self.assertGreaterEqual(r["observed_ratio"], 0)
        self.assertLessEqual(r["observed_ratio"], 1)

    def test_csv_summary_parseable(self):
        text = to_csv([_doc(), _doc()])
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(len(rows), 3)  # header + 2
        self.assertEqual(rows[0][0], "id")
        self.assertEqual(len(rows[1]), len(rows[0]))

    def test_csv_joint_long_format(self):
        doc = _doc()
        rows = list(csv.DictReader(io.StringIO(
            to_csv([doc], joints=True))))
        n = len(doc["skeleton"]["joints"])
        self.assertEqual(len(rows), n)
        self.assertIn("state", rows[0])
        self.assertIn("basis", rows[0])

    def test_no_ghost_rows_for_missing_joints(self):
        doc = _doc()
        del doc["skeleton"]["joints"]["foot_l"]
        rows = joint_rows([doc])
        self.assertNotIn("foot_l", {r["joint"] for r in rows})
        self.assertEqual(len(rows),
                         len(doc["skeleton"]["joints"]))

    def test_jsonl_roundtrip(self):
        docs = [_doc(), _doc()]
        lines = to_jsonl(docs).strip().split("\n")
        self.assertEqual(len(lines), 2)
        for line, doc in zip(lines, docs):
            self.assertEqual(json.loads(line)["id"], doc["id"])

    def test_from_store_reads_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            d1, d2 = _doc(), _doc()
            store.save(d1)
            store.save(d2)
            docs = from_store(store)
            self.assertEqual({d["id"] for d in docs},
                             {d1["id"], d2["id"]})
            # a stray invalid file is skipped, not fatal
            with open(os.path.join(tmp, "junk.json"), "w") as f:
                f.write("{not json")
            self.assertEqual(len(from_store(store)), 2)

    def test_empty_dataset_is_header_only(self):
        text = to_csv([])
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(to_jsonl([]), "")

    def test_non_object_store_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            d = _doc()
            store.save(d)
            # valid JSON, wrong shape: a list, not a document
            with open(os.path.join(tmp, "junk.json"), "w") as f:
                f.write("[]")
            docs = from_store(store)
            self.assertEqual([x["id"] for x in docs], [d["id"]])

    def test_no_id_doc_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            d = _doc()
            store.save(d)
            # hand-placed valid dict with no id: list() emits an entry
            # whose id is None — store.get(None) would raise TypeError
            with open(os.path.join(tmp, "k_eeeeeeeeeeee.json"),
                      "w") as f:
                json.dump({"skeleton": {}}, f)
            docs = from_store(store)
            self.assertEqual([x["id"] for x in docs], [d["id"]])

    def test_invalid_schema_doc_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            d = _doc()
            store.save(d)
            # hand-placed file with a valid id but no schema fields
            with open(os.path.join(tmp, "k_ffffffffffff.json"),
                      "w") as f:
                json.dump({"id": "k_ffffffffffff", "skeleton": {}}, f)
            docs = from_store(store)
            self.assertEqual([x["id"] for x in docs], [d["id"]])


if __name__ == "__main__":
    unittest.main()

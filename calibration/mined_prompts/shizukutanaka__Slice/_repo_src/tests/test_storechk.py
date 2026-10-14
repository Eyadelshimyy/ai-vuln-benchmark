import copy
import os
import shutil
import tempfile
import unittest

from slice import bitmap, knowledge, pipeline, storechk
from tests import synthetic_person


def _store(docs):
    d = tempfile.mkdtemp()
    st = knowledge.KnowledgeStore(d)
    for doc in docs:
        st.save(doc)
    return d


class TestStoreCheck(unittest.TestCase):
    def _doc(self):
        raw = bitmap.encode_png(synthetic_person())
        return pipeline.strip_runtime(pipeline.analyze(raw))

    def setUp(self):
        self.dirs = []
        self.addCleanup(
            lambda: [shutil.rmtree(d, True) for d in self.dirs])

    def _store(self, docs):
        d = _store(docs)
        self.dirs.append(d)
        return d

    def test_clean_store_passes(self):
        d = self._store([self._doc()])
        res = storechk.audit_store(d)
        self.assertEqual(res["verdict"], "pass")
        self.assertEqual(res["n_docs"], 1)
        self.assertFalse(res["duplicates"]["pairs"])
        # the synth person yields observed joints — the rate is
        # honest data, not a hidden prediction
        self.assertGreater(res["joint_observed"]["head"], 0)

    def test_duplicate_pair_detected(self):
        doc = self._doc()
        twin = copy.deepcopy(doc)
        twin["id"] = "k_deadbeef0001"
        d = self._store([doc, twin])
        res = storechk.audit_store(d)
        self.assertEqual(len(res["duplicates"]["pairs"]), 1)
        self.assertEqual(res["verdict"], "warn")

    def test_broken_doc_fails_not_crash(self):
        d = self._store([self._doc()])
        with open(os.path.join(d, "k_badbad00.json"), "w") as f:
            f.write("{broken json")
        res = storechk.audit_store(d)
        self.assertEqual(res["verdict"], "fail")
        self.assertEqual(res["unreadable"][0]["id"], "k_badbad00")
        self.assertTrue(any(r.startswith("unreadable:")
                            for r in res["reasons"]))

    def test_empty_store_is_pass(self):
        d = self._store([])
        res = storechk.audit_store(d)
        self.assertEqual(res["verdict"], "pass")
        self.assertEqual(res["n_docs"], 0)


if __name__ == "__main__":
    unittest.main()

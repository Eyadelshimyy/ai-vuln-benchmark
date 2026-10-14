import json
import os
import tempfile
import unittest
import zipfile

from slice.bundle import manifest, pack, unpack
from slice.knowledge import KnowledgeStore, build
from slice.pose import HeuristicPoseEstimator
from tests import synthetic_person


def _doc():
    skel = HeuristicPoseEstimator().estimate(synthetic_person())
    return build(skel, {"shoulder_hip": 1.0}, {"label": "standing"},
                 engine={"name": "heuristic", "version": "1"})


class TestBundle(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            doc = _doc()
            store.save(doc)
            zpath = os.path.join(tmp, "out.zip")
            m = pack(store, zpath)
            self.assertEqual(m["count"], 1)
            docs = unpack(zpath)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]["id"], doc["id"])
            self.assertEqual(docs[0]["skeleton"]["joints"],
                             doc["skeleton"]["joints"])

    def test_manifest_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            store.save(_doc())
            zpath = os.path.join(tmp, "out.zip")
            pack(store, zpath)
            m = manifest(zpath)
            self.assertEqual(m["schema"], "slice.bundle/v1")
            self.assertEqual(m["documents"][0]["body_model"],
                             "adult")
            self.assertEqual(m["skipped"], 0)

    def test_invalid_files_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            store.save(_doc())
            with open(os.path.join(tmp, "broken.json"), "w") as f:
                f.write("{not json")
            with open(os.path.join(tmp, "badschema.json"),
                      "w") as f:
                json.dump({"no": "schema"}, f)
            zpath = os.path.join(tmp, "out.zip")
            m = pack(store, zpath)
            self.assertEqual(m["count"], 1)
            # both invalid files are dropped by store.list() itself —
            # an id-less file is unlistable (its filename is not a
            # retrievable identity), so pack never sees them
            self.assertEqual(m["skipped"], 0)
            self.assertEqual(len(unpack(zpath)), 1)

    def test_unpack_skips_corrupt_members(self):
        # one bad member must not abort the whole archive
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            doc = _doc()
            store.save(doc)
            zpath = os.path.join(tmp, "out.zip")
            pack(store, zpath)
            with zipfile.ZipFile(zpath, "a") as z:
                z.writestr("docs/corrupt.json", "{not json")
                z.writestr("docs/nondict.json", "[1, 2]")
            docs = unpack(zpath)
            self.assertEqual([d["id"] for d in docs], [doc["id"]])

    def test_unpack_survives_unprocessable_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, "b.zip")
            doc = _doc()
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("manifest.json", "{}")
                z.writestr("docs/ok.json", json.dumps(doc))
                z.writestr("docs/bad.json",
                           json.dumps({"id": "k_aaaaaaaaaaaa",
                                       "skeleton": {"joints":
                                                    {"head": 5}}}))
            docs = unpack(zpath)
            self.assertEqual([d["id"] for d in docs], [doc["id"]])

    def test_manifest_rejects_non_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            zpath = os.path.join(tmp, "b.zip")
            with zipfile.ZipFile(zpath, "w") as z:
                z.writestr("manifest.json", "[1, 2]")
            with self.assertRaises(ValueError):
                manifest(zpath)

    def test_pack_survives_non_dict_body_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            store.save(_doc())
            # a schema-valid doc whose body_model is not a dict must
            # be refused at pack — validate does not type-check it,
            # and _list_entry must still tolerate listing it
            doc = _doc()
            doc["skeleton"]["body_model"] = 5
            with open(os.path.join(tmp, doc["id"] + ".json"),
                      "w") as f:
                json.dump(doc, f)
            m = pack(store, os.path.join(tmp, "out.zip"))
            self.assertEqual(m["count"], 1)
            self.assertEqual(m["skipped"], 1)

    def test_pack_skips_non_dict_doc(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            store.save(_doc())
            # a store file replaced with a JSON array is dropped at
            # list() — pack must not index it as a document
            with open(os.path.join(tmp, "k_aaaaaaaaaaaa.json"),
                      "w") as f:
                json.dump([1, 2], f)
            m = pack(store, os.path.join(tmp, "out.zip"))
            self.assertEqual(m["count"], 1)

    def test_unpack_skips_crc_corrupt_member(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            doc = _doc()
            store.save(doc)
            zpath = os.path.join(tmp, "out.zip")
            pack(store, zpath)
            # corrupt the compressed payload of the docs member —
            # z.read raises and must not kill unpack
            import struct
            raw = bytearray(open(zpath, "rb").read())
            with zipfile.ZipFile(zpath) as z:
                info = z.getinfo("docs/%s.json" % doc["id"])
            off = info.header_offset
            nlen = struct.unpack("<H", raw[off + 26:off + 28])[0]
            elen = struct.unpack("<H", raw[off + 28:off + 30])[0]
            raw[off + 30 + nlen + elen + 5] ^= 0xFF
            open(zpath, "wb").write(bytes(raw))
            self.assertEqual(unpack(zpath), [])

    def test_pack_survives_unprocessable_doc(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            store.save(_doc())
            # validate raises inside on malformed internals — pack
            # must skip that doc, not die
            with open(os.path.join(tmp, "k_aaaaaaaaaaaa.json"),
                      "w") as f:
                json.dump({"id": "k_aaaaaaaaaaaa",
                           "skeleton": {"joints": {"head": 5}}}, f)
            m = pack(store, os.path.join(tmp, "out.zip"))
            self.assertEqual(m["count"], 1)
            self.assertEqual(m["skipped"], 1)

    def test_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore(tmp)
            zpath = os.path.join(tmp, "out.zip")
            m = pack(store, zpath)
            self.assertEqual(m["count"], 0)
            self.assertEqual(unpack(zpath), [])


if __name__ == "__main__":
    unittest.main()

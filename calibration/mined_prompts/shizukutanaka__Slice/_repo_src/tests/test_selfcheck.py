import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from tests import synthetic_person

from slice import bitmap, knowledge, rest, selfcheck


def _post(port, path, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, body=body)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data



class TestSelfCheck(unittest.TestCase):
    def test_unmapped_verdict_advisory_not_unmeasured(self):
        """A verdict outside a layer's known vocabulary is
        unrecognized evidence, not absence of evidence — it must
        not be silently neutralized to 'unmeasured'."""
        from slice.selfcheck import _severity
        self.assertEqual(
            _severity("contrad", {"verdict": "insufficient"}),
            "unmeasured")
        self.assertEqual(
            _severity("limbcov", {"verdict": "unmeasurable"}),
            "unmeasured")
        self.assertEqual(
            _severity("gate", {"verdict": "some_new_verdict"}),
            "advisory")
        self.assertEqual(
            _severity("other_layer", {"verdict": "custom"}),
            "advisory")

    def test_synth_person_passes_or_warns(self):
        raw = bitmap.encode_png(synthetic_person())
        r = selfcheck.run(raw)
        self.assertIn(r["verdict"], ("pass", "warn"))
        for layer in ("imgqual", "human", "evid", "limbcov", "fit",
                      "stability", "contrad", "consistency", "gate"):
            self.assertIn(layer, r["layers"], layer)
            self.assertIn(layer, r["severity"], layer)
        self.assertIn("skeleton", r["doc"])

    def test_blank_image_fails_via_detection(self):
        bmp = bitmap.Bitmap.new(60, 60, (255, 255, 255, 255))
        r = selfcheck.run(bitmap.encode_png(bmp))
        self.assertEqual(r["verdict"], "fail")
        self.assertIn("detection:no_person", r["reasons"])
        # unmeasurable layers do not fake a verdict
        self.assertNotIn("evid", r["layers"])

    def test_robust_profile_runs(self):
        raw = bitmap.encode_png(synthetic_person())
        r = selfcheck.run(raw, robust=True)
        self.assertEqual(r["doc"]["engine"]["profile"], "robust")
        self.assertIn(r["verdict"], ("pass", "warn", "fail"))

    def test_unknown_model_disclosed(self):
        raw = bitmap.encode_png(synthetic_person())
        r = selfcheck.run(raw, model="no-such-model")
        self.assertIn("model:unknown_body_model", r["reasons"])
        r = selfcheck.run(raw, model="adult")
        self.assertNotIn("model:unknown_body_model", r["reasons"])
        r = selfcheck.run(raw)
        self.assertNotIn("model:unknown_body_model", r["reasons"])

    def test_large_image_compared_at_estimator_resolution(self):
        # skeleton coords are downscaled; a full-res mask would flag
        # every joint as off-mask — regression for >max_dim inputs
        raw = bitmap.encode_png(synthetic_person(640, 1200))
        r = selfcheck.run(raw)
        self.assertFalse(
            [x for x in r["reasons"] if x.startswith("evid:off_mask")],
            r["reasons"])
        # same-resolution comparison: bones explain real silhouette
        # mass — a full-res mask comparison yields exactly 0.0
        self.assertGreater(
            r["layers"]["fit"]["fraction"], 0.3)

    def test_human_scores_detected_component_not_frame(self):
        # a detached object must not drag the person into
        # not_person_like — the audited shape is the estimated one
        bmp = synthetic_person(300, 300)
        for y in range(50, 230):
            for x in range(5, 55):
                bmp.set(x, y, (60, 60, 60, 255))
        r = selfcheck.run(bitmap.encode_png(bmp))
        self.assertNotIn("human:not_person_like", r["reasons"])

    def test_robust_stability_uses_robust_variants(self):
        # faint figure: only the robust profile finds it; probing
        # with default variants would report unmeasurable
        raw = bitmap.encode_png(synthetic_person(
            skin=(210, 210, 210, 255)))
        r = selfcheck.run(raw, robust=True)
        head = r["layers"]["stability"]["joints"]["head"]
        self.assertNotEqual(head["verdict"], "single_run")
        self.assertGreaterEqual(head["runs"], 3)


    def test_consistency_reasons_not_double_counted(self):
        # selfcheck appends consistency codes itself AND gate.check
        # re-reports the same audit's codes — each finding must be
        # counted once in reasons/n_reasons
        from unittest.mock import patch
        from slice import consistency
        raw = bitmap.encode_png(synthetic_person())
        with patch.object(consistency, "audit",
                          return_value=["shoulder_l_elbow_l_too_long"]):
            r = selfcheck.run(raw)
        self.assertEqual(
            r["reasons"].count("consistency:shoulder_l_elbow_l_too_long"),
            1, r["reasons"])
        self.assertEqual(r["n_reasons"], len(r["reasons"]))


class TestAuditRest(unittest.TestCase):
    def test_post_audit_endpoint(self):
        rest.Handler.store = knowledge.KnowledgeStore(
            tempfile.mkdtemp())
        rest.Handler.token = None
        srv = ThreadingHTTPServer(("127.0.0.1", 0), rest.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            raw = bitmap.encode_png(synthetic_person())
            status, data = _post(port, "/audit", raw)
            self.assertEqual(status, 200)
            body = json.loads(data)
            self.assertIn(body["verdict"], ("pass", "warn"))
            self.assertIn("layers", body)
            self.assertIn("doc", body)
            status, data = _post(port, "/audit?robust=1", raw)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["doc"]["engine"]
                             ["profile"], "robust")
        finally:
            srv.shutdown()


class TestAuditCli(unittest.TestCase):
    def test_audit_store_needs_no_image(self):
        # `audit --store DIR` must not require a positional image
        from slice.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(["audit", "--store", d]), 0)


if __name__ == "__main__":
    unittest.main()

"""Overlay persistence: PNG is stored beside the doc, memory is only
a bounded read-through cache — overlay_url survives a restart and a
long-running server does not grow the dict forever."""
import http.client
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from slice import bitmap, knowledge, rest
from tests import synthetic_person


def _serve():
    tmp = tempfile.mkdtemp()
    rest.Handler.store = knowledge.KnowledgeStore(tmp)
    rest.Handler.token = None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rest.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, tmp


class TestOverlayPersistence(unittest.TestCase):
    def setUp(self):
        self.srv, self.root = _serve()
        self.port = self.srv.server_address[1]
        self.addCleanup(self.srv.shutdown)

    def _analyze_save(self):
        raw = bitmap.encode_png(synthetic_person())
        c = http.client.HTTPConnection("127.0.0.1", self.port,
                                       timeout=15)
        c.request("POST", "/analyze?save=1", body=raw,
                  headers={"Content-Type": "application/octet-stream"})
        r = c.getresponse()
        body = json.loads(r.read())
        c.close()
        self.assertEqual(r.status, 200)
        return body

    def _get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port,
                                       timeout=15)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        ctype = r.getheader("Content-Type")
        c.close()
        return r.status, body, ctype

    def test_overlay_saved_to_disk_and_served(self):
        out = self._analyze_save()
        url = out["overlay_url"]
        kid = url.rsplit("/", 1)[-1].removesuffix(".png")
        self.assertTrue(os.path.isfile(
            os.path.join(self.root, kid + ".overlay.png")))
        status, body, ctype = self._get(url)
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/png")
        self.assertTrue(body.startswith(b"\x89PNG"))

    def test_overlay_survives_cache_loss(self):
        out = self._analyze_save()
        url = out["overlay_url"]
        # a restart loses the in-memory dict; the file must still serve
        rest.Handler.overlays = {}
        status, body, _ = self._get(url)
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"\x89PNG"))

    def test_cache_is_bounded(self):
        rest.Handler.overlays = {}
        inst = rest.Handler.__new__(rest.Handler)
        for i in range(rest.Handler.OVERLAY_CACHE_MAX + 5):
            inst._cache_overlay("k_%012x" % i, b"x")
        self.assertEqual(len(rest.Handler.overlays),
                         rest.Handler.OVERLAY_CACHE_MAX)
        # oldest entries were evicted first (insertion order)
        self.assertNotIn("k_%012x" % 0, rest.Handler.overlays)

    def test_unknown_overlay_404(self):
        status, _, _ = self._get("/overlay/k_000000000000.png")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()

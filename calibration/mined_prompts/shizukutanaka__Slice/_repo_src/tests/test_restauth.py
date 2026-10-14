"""Tests for the REST bearer-token gate (P2-17)."""
import http.client
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from slice import knowledge, rest


def _serve(token):
    tmp = tempfile.mkdtemp()
    rest.Handler.store = knowledge.KnowledgeStore(tmp)
    rest.Handler.token = token
    srv = ThreadingHTTPServer(("127.0.0.1", 0), rest.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _get(port, path, token=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Authorization": "Bearer " + token} if token else {}
    c.request("GET", path, headers=headers)
    r = c.getresponse()
    r.read()
    c.close()
    return r.status


class TestRestAuth(unittest.TestCase):
    def setUp(self):
        self.srv = _serve("s3cret")
        self.port = self.srv.server_address[1]
        self.addCleanup(self.srv.shutdown)

    def test_api_requires_bearer(self):
        self.assertEqual(_get(self.port, "/knowledge"), 401)

    def test_wrong_token_rejected(self):
        self.assertEqual(_get(self.port, "/knowledge", "nope"), 401)

    def test_correct_token_accepted(self):
        self.assertEqual(_get(self.port, "/knowledge", "s3cret"), 200)

    def test_health_stays_open(self):
        self.assertEqual(_get(self.port, "/health"), 200)

    def test_post_analyze_gated(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port,
                                       timeout=5)
        c.request("POST", "/analyze", body=b"x",
                  headers={"Content-Length": "1"})
        self.assertEqual(c.getresponse().status, 401)
        c.close()


if __name__ == "__main__":
    unittest.main()
